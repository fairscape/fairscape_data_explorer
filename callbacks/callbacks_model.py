# callbacks_model.py
import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback
import numpy as np # Import numpy needed for model_utils if not already there
import statsmodels.api as sm # Import needed for model_utils if not already there

# Assuming model_utils.py is correctly updated as per previous responses
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
    # State('available-columns-store', 'data'), # Keep this commented out or remove if not used
    State('model-y-selector', 'value'),
    State('model-x-selector', 'value'),
    prevent_initial_call=True
)
def update_model_selectors_on_data_change(cohort_data_dict, current_y, current_x): # Removed available_cols_data from args
    empty_options = []
    # Default to disabled state
    default_return = empty_options, None, True, empty_options, [], True, True
    if not cohort_data_dict:
        return default_return

    try:
        df = pd.DataFrame(cohort_data_dict)
        if df.empty:
            return default_return

        numeric_cols_model = []
        logistic_y_candidates = [] # Use this list specifically for potential Y logistic columns
        all_current_cols = df.columns

        for col in all_current_cols:
            if col == 'Unnamed: 0': continue # Skip default index
            if df[col].notna().any(): # Only consider columns with data
                dtype = df[col].dtype
                series_no_na = df[col].dropna()

                # --- Identify Numeric Columns for X and potentially Y (Linear) ---
                if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                    numeric_cols_model.append(col)

                # --- Identify Candidates for Y (Logistic) ---
                # Reset flag for each column
                is_logistic_y = False
                # 1. Boolean type
                if pd.api.types.is_bool_dtype(dtype):
                    is_logistic_y = True
                # 2. Integer type with only 0/1 values
                elif pd.api.types.is_integer_dtype(dtype):
                    # Ensure unique check handles potential non-numeric values gracefully if dtype is misleading
                    if pd.api.types.is_numeric_dtype(series_no_na):
                        if set(series_no_na.unique()) <= {0, 1}:
                            is_logistic_y = True
                # 3. String/Object type with only "True"/"False" (case-insensitive)
                elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                    # Convert to lowercase string and check unique values
                    # Use .astype(str) for robustness even if object type contains non-strings
                    unique_vals_lower = set(series_no_na.astype(str).str.lower().unique())
                    # Check specifically for {'true', 'false'}
                    if unique_vals_lower == {'true', 'false'}:
                         is_logistic_y = True
                    # Optionally, include other binary pairs if needed, like {'yes', 'no'}
                    # elif unique_vals_lower == {'yes', 'no'}:
                    #     is_logistic_y = True

                if is_logistic_y:
                    # Add to the dedicated list for logistic Y candidates
                    logistic_y_candidates.append(col)

        # Combine candidates for the Y dropdown:
        # - Numeric columns (can be Y for Linear)
        # - Explicitly identified logistic candidates (boolean, 0/1 int, "True"/"False" string)
        y_candidate_cols = sorted(list(set(numeric_cols_model + logistic_y_candidates)))
        model_y_options = [{'label': col, 'value': col} for col in y_candidate_cols]

        # X options remain only the purely numeric columns
        model_x_options = [{'label': col, 'value': col} for col in sorted(numeric_cols_model)]

        # Preserve selections if still valid
        new_y_value = current_y if current_y in y_candidate_cols else None
        valid_x_values = [opt['value'] for opt in model_x_options]
        # Ensure current_x is a list before filtering
        current_x_list = current_x if isinstance(current_x, list) else ([current_x] if current_x else [])
        new_x_value = [x for x in current_x_list if x in valid_x_values]

        # Determine disabled states for dropdowns
        y_disabled = not bool(model_y_options)
        x_disabled = not bool(model_x_options)

        # --- Keep ORIGINAL build button disabled logic ---
        # Disabled if either dropdown is disabled (has no options)
        build_disabled = y_disabled or x_disabled
        # --- End ORIGINAL build button disabled logic ---

        return model_y_options, new_y_value, y_disabled, model_x_options, new_x_value, x_disabled, build_disabled

    except Exception as e:
        print(f"Error updating model selectors: {e}\n{traceback.format_exc()}")
        # Return default disabled state on error
        return empty_options, None, True, empty_options, [], True, True


# --- build_and_display_model remains UNCHANGED from the previous correct version ---
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
    no_model_summary = html.Div("Build a model to see the summary.", style={'textAlign': 'center', 'padding': '20px', 'color': 'grey'}) # Use this for initial/error states

    if not cohort_data_dict or not y_col or not x_cols:
        return no_update, no_update, dbc.Alert("Select Target (Y) and at least one Predictor (X).", color="warning", dismissable=True)

    try:
        df = pd.DataFrame(cohort_data_dict)
        if df.empty:
             return no_model_summary, empty_plot_div, dbc.Alert("Data is empty.", color="warning", dismissable=True)

        try:
            model_type = determine_model_type(df, y_col, schema_props_dict)
        except ValueError as type_e:
             return html.Div([html.P("Model Type Error:"), html.Pre(str(type_e))]), \
                    empty_plot_div, dbc.Alert(f"{type_e}", color="danger", dismissable=True)

        status_msg = f"Building {model_type.capitalize()} Regression..."
        status_alert = dbc.Alert(status_msg, color="info")

        model_results = None
        plot_output = empty_plot_div

        if model_type == 'linear':
            model_results = fit_linear_model(df, y_col, x_cols)
            status_msg = f"Linear Regression complete."
            plot_result = create_linear_residual_plot(model_results, df, y_col, x_cols)
            if isinstance(plot_result, go.Figure):
                 plot_output = dcc.Graph(figure=plot_result)
            else:
                 plot_output = plot_result
                 status_msg += " (Residual Plot Error)"

        elif model_type == 'logistic':
            try:
                model_results = fit_logistic_model(df, y_col, x_cols)
                status_msg = f"Logistic Regression complete."
            except sm.tools.sm_exceptions.PerfectSeparationError as pse:
                 status_msg = f"Logistic Regression Error (Perfect Separation): Cannot fit model. Predictors perfectly separate outcomes. Error: {pse}"
                 status_alert = dbc.Alert(status_msg, color="danger", dismissable=True)
                 return no_model_summary, empty_plot_div, status_alert
            except Exception as fit_e:
                 if "convergence" in str(fit_e).lower():
                      status_msg = f"Logistic Regression Warning: {fit_e}"
                      status_alert = dbc.Alert(status_msg, color="warning", dismissable=True)
                      if hasattr(fit_e, 'model') and hasattr(fit_e.model, 'results'):
                           model_results = fit_e.model.results
                 else:
                      raise fit_e

            if model_results:
                plot_result = create_logistic_roc_plot(model_results, df, y_col, x_cols)
                if isinstance(plot_result, go.Figure):
                     plot_output = dcc.Graph(figure=plot_result)
                else:
                     plot_output = plot_result
                     status_msg += " (ROC Plot Error)"

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        summary_display = no_model_summary
        if model_results:
            summary_display = create_model_summary_display(model_results) # Uses the simple text version from model_utils now

        if isinstance(status_alert, dbc.Alert) and status_alert.color != "danger":
             if "Error)" in status_msg:
                  status_alert = dbc.Alert(status_msg, color="warning", dismissable=True)
             elif status_alert.color == "info":
                  status_alert = dbc.Alert(status_msg, color="success", dismissable=True, duration=4000)

        return summary_display, plot_output, status_alert

    except ValueError as ve:
        print(f"Model Building Value Error: {ve}")
        return html.Div([html.P("Error:"), html.Pre(str(ve))]), \
               empty_plot_div, dbc.Alert(f"{ve}", color="danger", dismissable=True)
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Unexpected error building model:\n{tb_str}")
        return html.Div([html.P("Unexpected Error:"), html.Pre(str(e))]), \
               empty_plot_div, dbc.Alert(f"Unexpected Error: {e}", color="danger", dismissable=True)