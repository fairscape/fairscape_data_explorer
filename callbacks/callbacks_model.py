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
            if df[col].dropna().shape[0] > 0:
                dtype = df[col].dtype

                is_strictly_numeric = False
                if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                     series_no_na = df[col].dropna()
                     unique_numeric = pd.to_numeric(series_no_na, errors='coerce').dropna().unique()
                     if unique_numeric.size > 0 and not np.isin(unique_numeric, [0, 1, 0.0, 1.0]).all():
                          is_strictly_numeric = True

                if is_strictly_numeric:
                    numeric_cols_for_x.append(col)
                    if col not in potential_y_cols:
                        potential_y_cols.append(col)
                elif col not in potential_y_cols:
                     potential_y_cols.append(col)


        model_y_options = [{'label': col, 'value': col} for col in sorted(potential_y_cols)]
        model_x_options = [{'label': col, 'value': col} for col in sorted(numeric_cols_for_x)]

        valid_y_values = [opt['value'] for opt in model_y_options]
        new_y_value = current_y if current_y in valid_y_values else None

        valid_x_values = [opt['value'] for opt in model_x_options]
        current_x_list = current_x if isinstance(current_x, list) else ([current_x] if current_x else [])
        new_x_value = [x for x in current_x_list if x in valid_x_values]

        y_disabled = not bool(model_y_options)
        x_disabled = not bool(model_x_options)

        build_disabled = y_disabled or x_disabled

        return model_y_options, new_y_value, y_disabled, model_x_options, new_x_value, x_disabled, build_disabled

    except Exception as e:
        print(f"Internal Error updating model selectors: {e}") # Keep internal errors
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
    no_model_summary = html.Div("Build a model to see the summary.", style={'textAlign': 'center', 'padding': '20px', 'color': 'grey'})

    if not n_clicks or n_clicks == 0:
        return no_update, no_update, no_update
    if not cohort_data_dict or not y_col or not x_cols:
        return no_model_summary, empty_plot_div, dbc.Alert("Select Target (Y) and at least one Predictor (X).", color="warning", dismissable=True)

    try:
        df = pd.DataFrame(cohort_data_dict)
        if df.empty:
             return no_model_summary, empty_plot_div, dbc.Alert("Data is empty.", color="warning", dismissable=True)

        try:
            model_type = determine_model_type(df, y_col, schema_props_dict)
        except ValueError as type_e:
             return html.Div([html.P("Model Type Determination Error:", className="text-danger"), html.Pre(str(type_e))]), \
                    empty_plot_div, dbc.Alert(f"Error determining model type: {type_e}", color="danger", dismissable=True)

        status_msg = f"Building {model_type.capitalize()} Regression..."
        status_alert = dbc.Alert(status_msg, color="info")

        model_results = None
        plot_output = empty_plot_div

        if model_type == 'linear':
            try:
                model_results = fit_linear_model(df, y_col, x_cols)
                status_msg = f"Linear Regression complete."
                if model_results:
                    plot_result = create_linear_residual_plot(model_results, df, y_col, x_cols)
                    if isinstance(plot_result, go.Figure): plot_output = dcc.Graph(figure=plot_result)
                    else: plot_output = plot_result; status_msg += f" ({plot_result or 'Plot Error'})"
                else: status_msg = "Linear Regression failed (check data)."
            except (ValueError, RuntimeError) as fit_e:
                 status_msg = f"Linear Regression Error: {fit_e}"
                 status_alert = dbc.Alert(status_msg, color="danger", dismissable=True)


        elif model_type == 'logistic':
            try:
                model_results = fit_logistic_model(df, y_col, x_cols)
                status_msg = f"Logistic Regression complete."
                if model_results:
                    plot_result = create_logistic_roc_plot(model_results, df, y_col, x_cols)
                    if isinstance(plot_result, go.Figure): plot_output = dcc.Graph(figure=plot_result)
                    else: plot_output = plot_result; status_msg += f" ({plot_result or 'Plot Error'})"
                else: status_msg = "Logistic Regression failed (check data/logs)."

            except sm.tools.sm_exceptions.PerfectSeparationError as pse:
                 status_msg = f"Logistic Regression Error (Perfect Separation): {pse}"
                 status_alert = dbc.Alert(status_msg, color="danger", dismissable=True)
            except (ValueError, RuntimeError) as fit_e:
                 status_msg = f"Logistic Regression Error: {fit_e}"
                 status_alert = dbc.Alert(status_msg, color="danger", dismissable=True)
            except Exception as fit_e:
                 status_msg = f"Logistic Regression Fitting Issue: {fit_e}"
                 if "convergence" in str(fit_e).lower():
                      status_alert = dbc.Alert(status_msg, color="warning", dismissable=True)
                      if hasattr(fit_e, 'model') and hasattr(fit_e.model, 'results'):
                           model_results = fit_e.model.results
                           status_msg += " (Results shown despite non-convergence)"
                 else:
                      status_alert = dbc.Alert(status_msg, color="danger", dismissable=True)


        else:
            raise ValueError(f"Unsupported model type determined: {model_type}")

        summary_display = no_model_summary
        if model_results:
            summary_display = create_model_summary_display(model_results)

        if isinstance(status_alert, dbc.Alert):
            is_error = status_alert.color in ["danger", "warning"] or "Error" in status_msg or "failed" in status_msg.lower()
            is_success = not is_error and model_results is not None

            if is_success and status_alert.color == "info":
                status_alert = dbc.Alert(status_msg, color="success", dismissable=True, duration=5000)
            elif is_error and status_alert.color == "info":
                 final_color = "warning" if "warning" in status_alert.className else "danger"
                 status_alert = dbc.Alert(status_msg, color=final_color, dismissable=True)

        return summary_display, plot_output, status_alert

    except ValueError as ve:
        return html.Div([html.P("Model Building Error:", className="text-danger"), html.Pre(str(ve))]), \
               empty_plot_div, dbc.Alert(f"Model Building Error: {ve}", color="danger", dismissable=True)
    except Exception as e:
        print(f"Internal error building model: {e}\n{traceback.format_exc()}") # Keep internal errors
        return html.Div([html.P("Unexpected Error:", className="text-danger"), html.Pre(str(e))]), \
               empty_plot_div, dbc.Alert(f"Unexpected Error building model: {e}", color="danger", dismissable=True)