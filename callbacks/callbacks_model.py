import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback
import numpy as np
import statsmodels.api as sm

from utils.model_utils import (
    determine_model_type, fit_linear_model, fit_logistic_model,
    create_model_summary_display, create_logistic_roc_plot, create_linear_residual_plot,
    create_model_equation_display # Added new import
)

# Default placeholder card for the equation section
default_equation_card = dbc.Card(
    [
        dbc.CardHeader(
            html.H5("Model Equation & Variable Interpretation", className="mb-0 card-title"),
            style={'backgroundColor': '#005f73', 'color': 'white'} # Using literal colors from model_utils
        ),
        dbc.CardBody(
            html.Div(
                "Build a model to see its equation and variable details. Ensure target and predictor variables are selected and valid.",
                className="text-center text-muted p-3"
            ),
            className="p-3"
        )
    ],
    className="shadow-sm"
)

@callback(
    Output('model-y-selector', 'options', allow_duplicate=True),
    Output('model-y-selector', 'value', allow_duplicate=True),
    Output('model-y-selector', 'disabled', allow_duplicate=True),
    Output('model-x-selector', 'options', allow_duplicate=True),
    Output('model-x-selector', 'value', allow_duplicate=True),
    Output('model-x-selector', 'disabled', allow_duplicate=True),
    Output('build-model-button', 'disabled', allow_duplicate=True),
    Input('cohort-data-store', 'data'),
    State('model-y-selector', 'value'),
    State('model-x-selector', 'value'),
    prevent_initial_call=True
)
def update_model_selectors_on_data_change(cohort_data_dict, current_y, current_x):
    empty_options = []
    default_return = empty_options, None, True, empty_options, [], True, True
    if not cohort_data_dict:
        return default_return

    try:
        df = pd.DataFrame(cohort_data_dict)
        if df.empty:
            return default_return

        numeric_cols_for_x = []
        potential_y_cols = []
        all_current_cols = df.columns

        for col in all_current_cols:
            if col == 'Unnamed: 0': continue
            
            series_no_na = df[col].dropna()
            if series_no_na.empty: continue

            potential_y_cols.append(col)

            is_numeric_candidate_for_x = False
            if pd.api.types.is_numeric_dtype(series_no_na.dtype) and \
               not pd.api.types.is_bool_dtype(series_no_na.dtype):
                is_numeric_candidate_for_x = True
            
            if is_numeric_candidate_for_x:
                numeric_cols_for_x.append(col)

        model_y_options = [{'label': col, 'value': col} for col in sorted(list(set(potential_y_cols)))]
        model_x_options = [{'label': col, 'value': col} for col in sorted(list(set(numeric_cols_for_x)))]
        
        valid_y_values = [opt['value'] for opt in model_y_options]
        new_y_value = current_y if current_y in valid_y_values else None

        valid_x_values = [opt['value'] for opt in model_x_options]
        current_x_list = current_x if isinstance(current_x, list) else ([current_x] if current_x else [])
        # Filter X values: must be valid, and not be the selected Y value
        new_x_value = [x for x in current_x_list if x in valid_x_values and (x != new_y_value if new_y_value else True)]

        y_disabled = not bool(model_y_options)
        x_disabled = not bool(model_x_options)
        
        build_disabled = (
            y_disabled or 
            x_disabled
        )

        return model_y_options, new_y_value, y_disabled, model_x_options, new_x_value, x_disabled, build_disabled

    except Exception as e:
        print(f"Internal Error updating model selectors: {e}\n{traceback.format_exc()}")
        return empty_options, None, True, empty_options, [], True, True


@callback(
    Output('model-summary-output', 'children', allow_duplicate=True),
    Output('model-plot-output', 'children', allow_duplicate=True),
    Output('model-status', 'children', allow_duplicate=True),
    Output('model-equation-container', 'children', allow_duplicate=True),
    Input('build-model-button', 'n_clicks'),
    State('cohort-data-store', 'data'),
    State('schema-properties-store', 'data'),
    State('model-y-selector', 'value'),
    State('model-x-selector', 'value'),
    prevent_initial_call=True
)
def build_and_display_model(n_clicks, cohort_data_dict, schema_props_dict, y_col, x_cols):
    empty_plot_div = html.Div("Plot will appear here after model build.", style={'textAlign': 'center', 'padding': '20px', 'color': 'grey'})
    no_model_summary_div = html.Div("Build a model to see the summary.", style={'textAlign': 'center', 'padding': '20px', 'color': 'grey'})

    if not n_clicks or n_clicks == 0:
        return no_update, no_update, no_update, no_update

    alert_msg_text = ""
    if not cohort_data_dict: alert_msg_text = "Cohort data is not available."
    elif not y_col: alert_msg_text = "Target (Y) column must be selected."
    elif not x_cols: alert_msg_text = "At least one Predictor (X) column must be selected."
    elif y_col in x_cols: alert_msg_text = "Target (Y) column cannot also be a Predictor (X) column."
    
    if alert_msg_text:
        return no_model_summary_div, empty_plot_div, dbc.Alert(alert_msg_text, color="warning", dismissable=True), default_equation_card

    try:
        df = pd.DataFrame(cohort_data_dict)
        if df.empty:
             return no_model_summary_div, empty_plot_div, dbc.Alert("Data is empty.", color="warning", dismissable=True), default_equation_card

        model_type = determine_model_type(df, y_col, schema_props_dict)

        status_msg = f"Building {model_type.capitalize()} Regression model for Y='{y_col}' with X=[{', '.join(x_cols)}]..."
        status_alert = dbc.Alert(status_msg, color="info")

        model_results = None
        plot_output = empty_plot_div
        summary_display = no_model_summary_div
        equation_card = default_equation_card # Initialize with default

        if model_type == 'linear':
            model_results = fit_linear_model(df, y_col, x_cols)
            plot_result = create_linear_residual_plot(model_results, df, y_col, x_cols)
            if isinstance(plot_result, go.Figure): plot_output = dcc.Graph(figure=plot_result)
            else: plot_output = plot_result
            status_msg = "Linear Regression build complete."

        elif model_type == 'logistic':
            model_results = fit_logistic_model(df, y_col, x_cols)
            plot_result = create_logistic_roc_plot(model_results, df, y_col, x_cols)
            if isinstance(plot_result, go.Figure): plot_output = dcc.Graph(figure=plot_result)
            else: plot_output = plot_result
            status_msg = "Logistic Regression build complete."
        
        summary_display = create_model_summary_display(model_results)
        equation_card = create_model_equation_display(model_results, model_type, y_col, x_cols, schema_props_dict)
        status_alert = dbc.Alert(status_msg, color="success", dismissable=True, duration=5000)

        return summary_display, plot_output, status_alert, equation_card

    except (ValueError, RuntimeError, sm.tools.sm_exceptions.PerfectSeparationError, Exception) as e:
        tb_str = traceback.format_exc()
        error_type_msg = "Model Building Error"
        color = "danger"

        if isinstance(e, ValueError) and "determine model type" in str(e).lower():
            error_type_msg = "Model Type Determination Error"
        elif isinstance(e, sm.tools.sm_exceptions.PerfectSeparationError):
            error_type_msg = "Logistic Regression Error (Perfect Separation)"
        elif isinstance(e, RuntimeError) and "convergence" in str(e).lower():
             error_type_msg = "Model Fitting Warning (Convergence)"
             color = "warning"
             if hasattr(e, 'model') and hasattr(e.model, 'results') and e.model.results:
                 partial_results = e.model.results
                 try:
                    # Attempt to show partial results despite convergence issues
                    # Re-determine model_type in case it's needed by display functions and not available from outer scope
                    current_model_type = determine_model_type(df, y_col, schema_props_dict) 
                    partial_summary = create_model_summary_display(partial_results)
                    partial_equation = create_model_equation_display(partial_results, current_model_type, y_col, x_cols, schema_props_dict)
                    partial_plot = html.Div("Plot not generated or may be unreliable due to model convergence issues.", className="text-warning text-center p-3")
                    status_alert_msg = f"{error_type_msg}: {e}. Displaying partial results."
                    return partial_summary, partial_plot, dbc.Alert(status_alert_msg, color="warning", dismissable=True, style={"whiteSpace": "pre-wrap"}), partial_equation
                 except Exception as inner_e:
                     print(f"Error trying to display partial results after convergence error: {inner_e}\n{traceback.format_exc()}")
                     # Fall through to general error handling if displaying partial results fails

        print(f"{error_type_msg} for Y='{y_col}', X=[{', '.join(x_cols)}]: {e}\n{tb_str}")
        status_alert_msg = f"{error_type_msg}: {e}"
        
        return no_model_summary_div, empty_plot_div, dbc.Alert(status_alert_msg, color=color, dismissable=True, style={"whiteSpace": "pre-wrap"}), default_equation_card