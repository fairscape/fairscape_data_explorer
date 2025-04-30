import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import traceback

from utils.cohort_utils import parse_uploaded_csv, join_cohort_data

@callback(
    Output('uploaded-cohort-store', 'data'),
    Output('upload-join-column-selector', 'options', allow_duplicate=True), Output('upload-join-column-selector', 'value', allow_duplicate=True), Output('upload-join-column-selector', 'disabled', allow_duplicate=True),
    Output('upload-cohort-column-selector', 'options', allow_duplicate=True), Output('upload-cohort-column-selector', 'value', allow_duplicate=True), Output('upload-cohort-column-selector', 'disabled', allow_duplicate=True),
    Output('process-upload-button', 'disabled', allow_duplicate=True),
    Output('upload-status', 'children', allow_duplicate=True),
    Input('upload-cohort-csv', 'contents'),
    State('upload-cohort-csv', 'filename'),
    prevent_initial_call=True
)
def handle_cohort_upload(contents, filename):
    if contents is None: return None, no_update, no_update, no_update, no_update, no_update, no_update, True, ""
    try:
        cohort_df = parse_uploaded_csv(contents)
        if cohort_df.empty: raise ValueError("CSV empty/unparseable.")
        cohort_df.columns = cohort_df.columns.str.strip()
        cols = [{'label': c, 'value': c} for c in cohort_df.columns]
        status = dbc.Alert(f"Parsed '{filename}' ({len(cohort_df)}r, {len(cohort_df.columns)}c). Select columns.", color="info")
        return cohort_df.to_dict('records'), cols, None, False, cols, None, False, True, status
    except ValueError as e: status = dbc.Alert(f"Error processing '{filename}': {e}", color="danger"); return None, [], None, True, [], None, True, True, status
    except Exception as e: status = dbc.Alert(f"Error reading '{filename}': {e}", color="danger"); print(f"Upload error: {traceback.format_exc()}"); return None, [], None, True, [], None, True, True, status

@callback(
    Output('process-upload-button', 'disabled', allow_duplicate=True),
    Input('main-join-column-selector', 'value'), Input('upload-join-column-selector', 'value'), Input('upload-cohort-column-selector', 'value'),
    State('uploaded-cohort-store', 'data'),
    prevent_initial_call=True
)
def toggle_process_upload_button(main_join, upload_join, cohort_col, upload_data):
    return not (main_join and upload_join and cohort_col and upload_data)

@callback(
    Output('cohort-data-store', 'data', allow_duplicate=True),
    Output('upload-status', 'children', allow_duplicate=True),
    Input('process-upload-button', 'n_clicks'),
    State('cohort-data-store', 'data'), State('uploaded-cohort-store', 'data'),
    State('main-join-column-selector', 'value'), State('upload-join-column-selector', 'value'), State('upload-cohort-column-selector', 'value'),
    prevent_initial_call=True
)
def process_uploaded_cohorts(n_clicks, cohort_data_dict, uploaded_cohort_dict,
                             main_join_col, upload_join_col, upload_cohort_col):
    alert_error = lambda msg: (no_update, dbc.Alert(msg, color="danger", dismissable=True))
    alert_warning = lambda msg: (no_update, dbc.Alert(msg, color="warning", dismissable=True))
    alert_success = lambda msg: dbc.Alert(msg, color="success", dismissable=True, duration=5000)

    if not cohort_data_dict: return alert_warning("Load main data first.")
    if not uploaded_cohort_dict: return alert_warning("Upload cohort file first.")
    if not all([main_join_col, upload_join_col, upload_cohort_col]): return alert_warning("Select all three join/label columns.")

    try:
        main_df = pd.DataFrame(cohort_data_dict); cohort_df = pd.DataFrame(uploaded_cohort_dict)
        if main_df.empty: return alert_warning("Main data empty.")
        if cohort_df.empty: return alert_warning("Uploaded data empty.")

        merged_df, new_col_name = join_cohort_data(main_df.copy(), cohort_df, main_join_col, upload_join_col, upload_cohort_col)
        if new_col_name in merged_df:
             assigned_count = (merged_df[new_col_name].astype(str) != 'N/A').sum()
             status = alert_success(f"Joined '{new_col_name}'. {assigned_count} rows assigned label (not 'N/A').")
        else: return alert_error("Merge failed: New column not found.")
        return merged_df.to_dict('records'), status
    except ValueError as e: return alert_error(f"Join error: {e}")
    except Exception as e: print(f"Join error: {traceback.format_exc()}"); return alert_error(f"Unexpected join error: {e}")