# callbacks_model.py
import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback

from model_utils import (
    determine_model_type, fit_linear_model, fit_logistic_model,
    create_model_summary_display, create_logistic_roc_plot, create_linear_residual_plot
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
    State('available-columns-store', 'data'),
    State('model-y-selector', 'value'),
    State('model-x-selector', 'value'),
    prevent_initial_call=True
)
def update_model_selectors_on_data_change(cohort_data_dict, available_cols_data, current_y, current_x):
    empty_options = []
    if not cohort_data_dict or not available_cols_data: return empty_options, None, True, empty_options, [], True, True
    try:
        df = pd.DataFrame(cohort_data_dict)
        if df.empty: return empty_options, None, True, empty_options, [], True, True
        numeric_cols_model = []; boolean_binary_cols_model = []
        all_current_cols = df.columns
        for col in all_current_cols:
            if df[col].notna().any():
                dtype = df[col].dtype
                is_numeric = pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
                is_bool_or_binary = pd.api.types.is_bool_dtype(dtype) or \
                                    (pd.api.types.is_integer_dtype(dtype) and set(df[col].dropna().unique()) <= {0, 1})
                if is_numeric: numeric_cols_model.append(col)
                if is_bool_or_binary: boolean_binary_cols_model.append(col)
        model_y_options = [{'label': col, 'value': col} for col in sorted(list(set(numeric_cols_model + boolean_binary_cols_model)))]
        model_x_options = [{'label': col, 'value': col} for col in sorted(numeric_cols_model)]
        new_y_value = current_y if current_y in [opt['value'] for opt in model_y_options] else None
        valid_x_values = [opt['value'] for opt in model_x_options]
        new_x_value = [x for x in current_x if x in valid_x_values] if current_x else []
        y_disabled = not bool(model_y_options); x_disabled = not bool(model_x_options)
        # Correct logic: button disabled if options missing OR selections missing
        build_disabled = y_disabled or x_disabled
        return model_y_options, new_y_value, y_disabled, model_x_options, new_x_value, x_disabled, build_disabled
    except Exception as e:
        print(f"Error updating model selectors: {e}")
        return empty_options, None, True, empty_options, [], True, True


@callback(
    Output('model-summary-output', 'children', allow_duplicate=True),
    Output('model-plot-output', 'children', allow_duplicate=True),
    Output('model-status', 'children', allow_duplicate=True),
    Input('build-model-button', 'n_clicks'),
    State('cohort-data-store', 'data'),
    State('schema-properties-store', 'data'),
    State('model-y-selector', 'value'),
    State('model-x-selector', 'value'),
    prevent_initial_call=True
)
def build_and_display_model(n_clicks, cohort_data_dict, schema_props_dict, y_col, x_cols):
    empty_plot_div = html.Div("Plot will appear here after model build.", style={'textAlign': 'center', 'padding': '20px', 'color': 'grey'})
    if not cohort_data_dict or not y_col or not x_cols: return no_update, no_update, dbc.Alert("Select Target (Y) and at least one Predictor (X).", color="warning", dismissable=True)
    try:
        df = pd.DataFrame(cohort_data_dict)
        if df.empty: return "Data is empty.", empty_plot_div, dbc.Alert("Data empty.", color="warning", dismissable=True)
        try: model_type = determine_model_type(df, y_col, schema_props_dict)
        except ValueError as type_e: return html.Div([html.P("Model Type Error:"), html.Pre(str(type_e))]), empty_plot_div, dbc.Alert(f"{type_e}", color="danger")
        status_msg = f"Building {model_type.capitalize()} Regression..."
        status_alert = dbc.Alert(status_msg, color="info")
        model_results = None
        if model_type == 'linear':
            model_results = fit_linear_model(df, y_col, x_cols)
            status_msg = f"Linear Regression complete."
        elif model_type == 'logistic':
            try:
                model_results = fit_logistic_model(df, y_col, x_cols)
                status_msg = f"Logistic Regression complete."
            except Exception as fit_e:
                 if "convergence" in str(fit_e).lower(): status_msg = f"Logistic Regression Warning: {fit_e}"; status_alert = dbc.Alert(status_msg, color="warning", dismissable=True)
                 else: raise fit_e
        else: raise ValueError(f"Unsupported model type: {model_type}")
        summary_display = create_model_summary_display(model_results)
        plot_output = empty_plot_div
        if model_type == 'linear':
            plot_result = create_linear_residual_plot(model_results, df, y_col, x_cols)
            if isinstance(plot_result, go.Figure): plot_output = dcc.Graph(figure=plot_result)
            else: plot_output = plot_result; status_msg += " (Residual Plot Error)"
        elif model_type == 'logistic':
            plot_result = create_logistic_roc_plot(model_results, df, y_col, x_cols)
            if isinstance(plot_result, go.Figure): plot_output = dcc.Graph(figure=plot_result)
            else: plot_output = plot_result; status_msg += " (ROC Plot Error)"
        if "Error)" in status_msg: status_alert = dbc.Alert(status_msg, color="warning", dismissable=True)
        elif not hasattr(status_alert, 'color') or status_alert.color != "warning": status_alert = dbc.Alert(status_msg, color="success", dismissable=True, duration=4000)
        return summary_display, plot_output, status_alert
    except ValueError as ve:
        print(f"Model Building Value Error: {ve}")
        return html.Div([html.P("Error:"), html.Pre(str(ve))]), empty_plot_div, dbc.Alert(f"{ve}", color="danger", dismissable=True)
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Unexpected error building model:\n{tb_str}")
        return html.Div([html.P("Unexpected Error:"), html.Pre(str(e))]), empty_plot_div, dbc.Alert(f"Unexpected Error: {e}", color="danger", dismissable=True)