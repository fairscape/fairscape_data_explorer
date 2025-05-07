# mine/callbacks/callbacks_main.py
import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import requests
import io
import zipfile
from urllib.parse import urlparse
import traceback

# --- Utilities and Configuration (Updated Imports) ---
from utils.s3_utils import get_s3_client, extract_from_zip_in_s3, S3File
from config import FAIRSCAPE_BASE_URL, MINIO_DEFAULT_BUCKET # Relative import for config
from layout import create_rule_row # Relative import for layout
from utils.app_utils import ( # Package import for utils
    update_status, find_schema_ark, update_summary_content,
    create_empty_figure, create_placeholder_plot
)

s3 = get_s3_client()

# --- Main ARK Loading and Processing Callback ---
@callback(
    # --- Core Data Stores ---
    Output('processed-data-store', 'data'),
    Output('cohort-data-store', 'data', allow_duplicate=True), 
    Output('schema-properties-store', 'data'),
    Output('available-columns-store', 'data'),
    # --- UI Feedback & Control ---
    Output('status-alert', 'children'),
    Output('status-alert', 'color'),
    Output('btn-download-summary', 'disabled'),
    Output('btn-download-html', 'disabled'),
    # --- Outputs for Resetting UI elements controlled elsewhere ---
    Output('data-summary', 'children', allow_duplicate=True), 
    Output('data-histogram', 'figure', allow_duplicate=True), 
    Output('column-metadata-display-container', 'children', allow_duplicate=True), # ADDED THIS OUTPUT
    Output('cohort-name-input', 'value'), 
    Output('rule-builder-container', 'children', allow_duplicate=True), 
    Output('rule-apply-status', 'children', allow_duplicate=True), 
    Output('upload-cohort-csv', 'contents'), 
    Output('upload-status', 'children', allow_duplicate=True), 
    Output('uploaded-cohort-store', 'data', allow_duplicate=True), 
    Output('process-upload-button', 'disabled', allow_duplicate=True), 
    Output('model-summary-output', 'children', allow_duplicate=True), 
    Output('model-plot-output', 'children', allow_duplicate=True), 
    Output('model-status', 'children', allow_duplicate=True), 
    # --- Selectors Reset by this callback (as they are cleared on load) ---
    Output('upload-join-column-selector', 'options', allow_duplicate=True),
    Output('upload-join-column-selector', 'value', allow_duplicate=True),
    Output('upload-join-column-selector', 'disabled', allow_duplicate=True),
    Output('upload-cohort-column-selector', 'options', allow_duplicate=True),
    Output('upload-cohort-column-selector', 'value', allow_duplicate=True),
    Output('upload-cohort-column-selector', 'disabled', allow_duplicate=True),
    # --- Inputs ---
    Input('load-button', 'n_clicks'),
    State('data-ark-input', 'value'),
    prevent_initial_call=True
)
def load_and_process_ark(n_clicks, data_ark):
    # --- Initial Reset Values using helpers ---
    status_message, status_color = update_status("Initializing...", "secondary")
    reset_summary = ["Load data to see summary."]
    empty_fig = create_empty_figure("Load data first")
    # ADDED: Initial placeholder for column metadata display
    initial_column_metadata_placeholder = dbc.Alert(
        "Select a numeric column from 'Plot Options' to view its metadata.", 
        color="info", 
        className="small p-2 text-center"
    )
    reset_model_summary = ["Build a model to see results."]
    reset_model_plot = create_placeholder_plot("Build a model to see plot.")
    initial_rule_container_children = [create_rule_row(0)]

    # Define the full list of outputs for the reset/error case
    # Order must match the @callback Output list
    # UPDATED: Added entry for column-metadata-display-container
    initial_outputs = [
        None, None, None, None, # Data stores (4)
        status_message, status_color, True, True, # Status, Download buttons disabled (4)
        reset_summary, empty_fig, initial_column_metadata_placeholder, "", initial_rule_container_children, "", # UI Resets (6)
        None, "", None, True, # Rule/Upload resets (contents, status, store, button) (4)
        reset_model_summary, reset_model_plot, "", # Model resets (3)
        [], None, True, [], None, True, # Upload selector resets (6)
    ] # Total: 4 + 4 + 6 + 4 + 3 + 6 = 27 outputs

    if not data_ark:
        status_message, status_color = update_status("Please provide a Data ARK identifier.", "warning")
        initial_outputs[4] = status_message
        initial_outputs[5] = status_color
        return tuple(initial_outputs)

    schema_props = pd.DataFrame()
    schema_props_dict = None
    raw_data_df = pd.DataFrame()
    typed_data = pd.DataFrame()

    try:
        status_message, status_color = update_status("Fetching Metadata...", "info")
        data_meta_url = f"{FAIRSCAPE_BASE_URL}/{data_ark}"
        response_data = requests.get(data_meta_url, timeout=30)
        response_data.raise_for_status()
        data_metadata = response_data.json()

        status_message, status_color = update_status("Finding Schema...", "info")
        schema_ark_found, schema_msg = find_schema_ark(data_metadata) 
        schema_status_prefix = "Metadata OK."
        if not schema_ark_found:
            status_message = f"{schema_status_prefix} {schema_msg}"
            if status_color != "danger": status_color = "warning"
        else:
            status_message = f"{schema_status_prefix} {schema_msg}"

        status_message, status_color = update_status("Locating Data File...", "info")
        full_path = data_metadata.get("distribution", {}).get("location", {}).get("path")
        if not full_path:
            content_url = data_metadata.get("distribution", {}).get("contentUrl")
            if content_url and urlparse(content_url).scheme == 's3':
                 full_path = urlparse(content_url).path.lstrip('/')
                 print(f"Warning: Using path from contentUrl: {full_path}")
            else:
                 raise ValueError(f"Data location ('distribution.location.path') missing in metadata for {data_ark}.")

        bucket = MINIO_DEFAULT_BUCKET
        key_path = full_path 
        csv_file_name_in_zip = None
        is_zip_file = False
        if ".zip/" in key_path:
            zip_path_parts = key_path.split(".zip/", 1)
            key_path = zip_path_parts[0] + ".zip" 
            csv_file_name_in_zip = zip_path_parts[1] 
            is_zip_file = True
        elif key_path.lower().endswith('.zip'):
            is_zip_file = True 

        status_message, status_color = update_status("Fetching/Extracting Data File...", "info")
        file_content = None
        if is_zip_file:
            try:
                if csv_file_name_in_zip:
                    file_content = extract_from_zip_in_s3(s3, bucket, key_path, csv_file_name_in_zip)
                else:
                    s3_file = S3File(s3, bucket, key_path)
                    with zipfile.ZipFile(s3_file, 'r') as zip_f:
                        csv_files = [f for f in zip_f.namelist() if f.lower().endswith('.csv') and not f.startswith('__MACOSX/')]
                        if not csv_files: raise FileNotFoundError(f"No CSV file found inside ZIP: s3://{bucket}/{key_path}")
                        csv_file_name_in_zip = csv_files[0] 
                        print(f"Auto-selected '{csv_file_name_in_zip}' from zip s3://{bucket}/{key_path}")
                        with zip_f.open(csv_file_name_in_zip) as file_in_zip: file_content = file_in_zip.read()
            except Exception as e: raise ValueError(f"Error processing ZIP s3://{bucket}/{key_path}: {e}") from e
        else:
            try:
                response = s3.get_object(Bucket=bucket, Key=key_path)
                file_content = response['Body'].read()
            except Exception as e: raise ValueError(f"Error downloading file s3://{bucket}/{key_path}: {e}") from e

        status_message, status_color = update_status("Parsing Data File...", "info")
        if not file_content:
             status_message = "Warning: Data file is empty."; status_color = "warning"; raw_data_df = pd.DataFrame()
        else:
            try:
                try: raw_data_df = pd.read_csv(io.BytesIO(file_content))
                except UnicodeDecodeError:
                    try: raw_data_df = pd.read_csv(io.BytesIO(file_content), encoding='latin-1'); status_message += " (Used latin-1)"
                    except Exception as e_enc: raise ValueError(f"CSV reading error after fallback: {e_enc}")
            except Exception as e: raise ValueError(f"Error reading CSV content: {e}")

        conversion_warnings = []
        if schema_ark_found:
            status_message, status_color = update_status("Processing Schema...", "info")
            try:
                schema_url = f"{FAIRSCAPE_BASE_URL}/{schema_ark_found}"
                response_schema = requests.get(schema_url, timeout=30); response_schema.raise_for_status()
                schema_data = response_schema.json()['metadata']; properties = {} 

                if isinstance(schema_data, list):
                    if schema_data and isinstance(schema_data[0], dict) and 'properties' in schema_data[0]:
                       properties = schema_data[0].get('properties', {})
                    else: print("Warning: Schema is a list but structure not recognized for properties.")
                elif isinstance(schema_data, dict): 
                    properties = schema_data.get('properties', {})

                if properties and isinstance(properties, dict):
                    # MODIFIED: Include 'valueURL' in props_list
                    props_list = [
                        {'name': n, 
                         'type': d.get('type'), 
                         'description': d.get('description'),
                         'value-url': d.get('value-url') # Added valueURL
                        } 
                        for n, d in properties.items() if isinstance(d, dict)
                    ]
                    if props_list:
                        schema_props = pd.DataFrame(props_list); schema_props_dict = schema_props.to_dict('records')
                    else: conversion_warnings.append("Schema 'properties' contains no valid definitions."); status_color = "warning"
                else: conversion_warnings.append("Schema found but has no 'properties' dictionary."); status_color = "warning"
            except Exception as e: conversion_warnings.append(f"Failed to process schema {schema_ark_found}: {e}"); status_color = "danger"; print(traceback.format_exc())

        typed_data = raw_data_df.copy()
        if not schema_props.empty and not raw_data_df.empty:
            status_message, status_color = update_status("Applying Schema Types...", "info")
            for _, prop in schema_props.iterrows():
                col_name = prop['name']; schema_type = prop['type']
                if col_name in typed_data.columns:
                    try:
                        original_series = typed_data[col_name]; converted_series = None
                        original_na_mask = original_series.isna() | (original_series.astype(str).str.lower().isin(['na', '<na>', 'none', 'nan', 'null', '']))
                        current_dtype_kind = original_series.dtype.kind
                        is_compatible = False
                        if schema_type == 'integer' and current_dtype_kind in 'iu': is_compatible = True
                        elif schema_type == 'number' and current_dtype_kind in 'iuf': is_compatible = True 
                        elif schema_type == 'boolean' and current_dtype_kind == 'b': is_compatible = True
                        elif schema_type == 'string' and current_dtype_kind in 'OSUV': is_compatible = True 

                        if is_compatible:
                            converted_series = original_series 
                        else:
                            if schema_type == 'integer':
                                num_col = pd.to_numeric(original_series, errors='coerce')
                                if num_col.notna().all() and num_col.dropna().mod(1).eq(0).all():
                                    converted_series = num_col.astype(pd.Int64Dtype())
                                elif num_col.notna().any(): 
                                    converted_series = num_col.astype(pd.Float64Dtype())
                                    conversion_warnings.append(f"'{col_name}' (int schema) has decimals, stored as Float.")
                                else: 
                                    converted_series = num_col.astype(pd.Int64Dtype())
                            elif schema_type == 'number':
                                converted_series = pd.to_numeric(original_series, errors='coerce').astype(pd.Float64Dtype())
                            elif schema_type == 'boolean':
                                 bool_map = {'true': True, 't': True, '1': True, 'yes': True, 'y': True,
                                             'false': False, 'f': False, '0': False, 'no': False, 'n': False}
                                 converted_series = original_series.astype(str).str.lower().map(bool_map).astype(pd.BooleanDtype())
                            elif schema_type == 'string':
                                 converted_series = original_series.astype(str).replace(
                                     {'nan': pd.NA, 'None': pd.NA, '': pd.NA, 'NA': pd.NA, '<NA>': pd.NA, 'null': pd.NA}, regex=False
                                 ).astype(pd.StringDtype())
                        if converted_series is not None:
                             typed_data[col_name] = converted_series.where(~original_na_mask, pd.NA)
                    except Exception as type_e:
                         conversion_warnings.append(f"Type error converting '{col_name}' to '{schema_type}': {type_e}")
        elif not schema_ark_found and status_color != 'danger':
             status_message = "Data Parsed. No schema found or applied."; status_color = "warning"

        if conversion_warnings:
            base_status = status_message.split("...")[0] + "..." if "..." in status_message else status_message
            if status_color != "danger": status_color = "warning"
            max_warn=5; warn_str = "\n".join(conversion_warnings[:max_warn])
            if len(conversion_warnings)>max_warn: warn_str+=f"\n... ({len(conversion_warnings)-max_warn} more)"
            status_message = f"{base_status} Warnings:\n{warn_str}"
        elif status_color != "danger": status_message = "Processing Complete. Explore data or define cohorts."; status_color = "success"

        n_plot, g_plot, n_model, all_ui = [], [], [], []
        for col in typed_data.columns:
            if col=='Unnamed: 0': continue
            dt=typed_data[col].dtype; all_ui.append({'label':col, 'value':col})
            if typed_data[col].notna().any():
                is_numeric = pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_bool_dtype(dt)
                is_groupable = False
                nu = typed_data[col].nunique(dropna=True)
                if pd.api.types.is_string_dtype(dt) or \
                   pd.api.types.is_bool_dtype(dt) or \
                   pd.api.types.is_categorical_dtype(dt) or \
                   ((pd.api.types.is_integer_dtype(dt) or pd.api.types.is_object_dtype(dt)) and nu <= 50):
                       is_groupable = True
                elif pd.api.types.is_object_dtype(dt):
                     try:
                         if pd.to_numeric(typed_data[col], errors='coerce').nunique(dropna=True) > 50: is_groupable = False
                     except: pass 
                if is_numeric:
                    n_model.append(col)
                    if not is_groupable or (pd.api.types.is_numeric_dtype(dt) and nu > 10): 
                         n_plot.append(col)
                if is_groupable:
                    g_plot.append(col)
        available_columns_data = {
            'numeric_plot': sorted(list(set(n_plot))), 
            'grouping_plot': sorted(list(set(g_plot))), 
            'numeric_model': sorted(list(set(n_model))), 
            'all': sorted(all_ui, key=lambda x: x['label'])
        }
        processed_data_dict = typed_data.to_dict('records');
        cohort_data_dict = processed_data_dict 

        # UPDATED: Added initial_column_metadata_placeholder to final_outputs
        final_outputs = [
            processed_data_dict, cohort_data_dict, schema_props_dict, available_columns_data, # Data stores (4)
            status_message, status_color, False, False, # Status, Download buttons enabled (4)
            reset_summary, empty_fig, initial_column_metadata_placeholder, "", initial_rule_container_children, "", # UI Resets (6)
            None, "", None, True, # Reset Upload UI state (4)
            reset_model_summary, reset_model_plot, "", # Reset Model UI placeholders (3)
            [], None, True, [], None, True, # Reset Upload selectors (6)
        ] # Total: 27 outputs
        return tuple(final_outputs)

    except Exception as e:
        tb_str = traceback.format_exc(); print(f"Error processing ARK {data_ark}:\n{tb_str}")
        status_message = f"Error loading/processing ARK: {e}"; status_color = "danger"
        initial_outputs[4] = status_message 
        initial_outputs[5] = status_color 
        initial_outputs[6] = True 
        initial_outputs[7] = True 
        initial_outputs[8] = [html.P(f"Failed to load data: {e}", className="text-danger")] 
        # initial_outputs[10] (column metadata placeholder) will use its default from initial_outputs definition
        return tuple(initial_outputs)


# --- Callback to Populate Initial UI Elements After Load ---
@callback(
    Output('numeric-column-selector', 'options'), Output('numeric-column-selector', 'value'), Output('numeric-column-selector', 'disabled'),
    Output('main-join-column-selector', 'options'), Output('main-join-column-selector', 'value'), Output('main-join-column-selector', 'disabled'),
    Output({'type': 'rule-column', 'index': ALL}, 'options', allow_duplicate=True),
    Output({'type': 'rule-column', 'index': ALL}, 'value', allow_duplicate=True),
    Output({'type': 'rule-column', 'index': ALL}, 'disabled', allow_duplicate=True),
    Input('available-columns-store', 'data'),
    State('rule-builder-container', 'children'),
    prevent_initial_call=True
)
def update_initial_ui_elements(available_cols_data, rule_rows):
    if not available_cols_data:
        num_rules = len(rule_rows) if rule_rows else 0
        return [], None, True, [], None, True, [[]]*num_rules, [None]*num_rules, [True]*num_rules

    num_plot = available_cols_data.get('numeric_plot', [])
    all_cols = available_cols_data.get('all', [])

    num_opts = [{'label': c, 'value': c} for c in num_plot]
    num_val = num_plot[0] if num_plot else None
    num_dis = not bool(num_opts)

    main_join_opts = all_cols
    main_join_val = None
    main_join_dis = not bool(main_join_opts)

    num_rules = len(rule_rows) if rule_rows else 0
    rule_opts = [all_cols] * num_rules
    rule_vals = [None] * num_rules
    rule_dis = [not bool(all_cols)] * num_rules

    return num_opts, num_val, num_dis, main_join_opts, main_join_val, main_join_dis, rule_opts, rule_vals, rule_dis


# --- Sidebar Toggle ---
@callback(
    Output("sidebar-offcanvas", "is_open"),
    Input("open-offcanvas-button", "n_clicks"),
    State("sidebar-offcanvas", "is_open"),
    prevent_initial_call=True,
)
def toggle_offcanvas(n1, is_open):
    return not is_open if n1 else is_open


# --- Summary Update ---
@callback(
    Output('data-summary', 'children', allow_duplicate=True),
    Input('cohort-data-store', 'data'),
    prevent_initial_call='initial_duplicate'
)
def update_summary_display_on_cohort_change(cohort_data_dict):
     print(f"Updating summary triggered by: {ctx.triggered_id}")
     return update_summary_content(cohort_data_dict)


@callback(
    Output("download-summary", "data"),
    Input("btn-download-summary", "n_clicks"),
    State("cohort-data-store", "data"),
    State("data-ark-input", "value"),
    prevent_initial_call=True,
)
def download_summary_csv(n_clicks, data_dict, data_ark): 
    if not data_dict:
        return None 

    try:
        df = pd.DataFrame(data_dict)
        if df.empty:
             return dict(content="Data is empty, cannot generate summary.", filename="summary_error.txt")
        try:
            summary_df = df.describe(include='all', datetime_is_numeric=True)
        except TypeError:
            summary_df = df.describe(include='all')
        summary_csv_string = summary_df.to_csv(index=True, encoding='utf-8')
        safe_ark = "".join(c if c.isalnum() else "_" for c in data_ark or "")[:50] or "data"
        filename = f"{safe_ark}_summary_{pd.Timestamp.now():%Y%m%d_%H%M}.csv"
        return dict(content=summary_csv_string, filename=filename)

    except Exception as e:
        print(f"Summary CSV download error: {e}\n{traceback.format_exc()}")
        error_content = f"Error generating summary CSV: {e}\n\n{traceback.format_exc()}"
        return dict(content=error_content, filename="summary_download_error.txt")
    
import dash
from dash import Input, Output, State, callback, no_update
import plotly.graph_objects as go
from datetime import datetime 
import traceback 
from dash import html

@callback(
    Output("download-exploration-html", "data"),
    Input("btn-download-html", "n_clicks"),
    State("cohort-data-store", "data"), 
    State("data-histogram", "figure"), 
    State("data-ark-input", "value"),   
    prevent_initial_call=True,
)
def download_exploration_html(n_clicks, data_dict, histogram_figure, data_ark): 
    if not n_clicks or n_clicks == 0:
        return no_update

    summary_text_content = "Data not available." 

    try:
        if data_dict:
            df = pd.DataFrame(data_dict)
            if not df.empty:
                try:
                     summary_df = df.describe(include='all', datetime_is_numeric=True)
                except TypeError:
                     summary_df = df.describe(include='all')
                summary_text_content = summary_df.to_string()
            else:
                 summary_text_content = "Data is empty."
        summary_html = f'''
        <div style="margin-bottom: 30px;">
            <h2>Data Summary</h2>
            <pre style="background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre; word-break: break-word;">{summary_text_content}</pre>
        </div>
        '''
        histogram_html = '''
        <div style="margin-bottom: 30px;">
             <h2>Data Histogram</h2>
             <p>Plot not available or could not be generated.</p>
        </div>
        '''
        if histogram_figure and isinstance(histogram_figure, dict) and histogram_figure != {}:
            try:
                fig = go.Figure(histogram_figure)
                plot_div_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                histogram_html = f'''
                <div style="margin-bottom: 30px;">
                    <h2>Data Histogram</h2>
                    {plot_div_html}
                </div>
                '''
            except Exception as e:
                 print(f"Error converting figure to HTML: {e}\n{traceback.format_exc()}")
                 histogram_html = f'''
                 <div style="margin-bottom: 30px;">
                      <h2>Data Histogram</h2>
                      <p style="color: red;">Error generating plot HTML: {e}</p>
                 </div>
                 '''
        safe_ark = "".join(c if c.isalnum() or c in ['-', '_', '.'] else "_" for c in data_ark or "exploration_data")[:100]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_ark}_{timestamp}_exploration.html"

        html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fairscape Data Exploration Report - {data_ark or 'Data Exploration'}</title>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2 {{ color: #005f73; margin-top: 20px; margin-bottom: 10px; }}
                pre {{ background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre; word-break: normal; }}
                .summary-section, .plot-section {{ margin-bottom: 30px; }}
            </style>
            <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script> 
        </head>
        <body>
            <h1>Data Exploration Report</h1>
            <p>Report generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Source Data ARK: {data_ark or 'N/A'}</p>

            {summary_html}
            {histogram_html}

            <div style="margin-top: 40px; font-size: 0.9em; color: #666;">
                <p>Generated by Fairscape Data Explorer.</p>
            </div>
        </body>
        </html>
        '''
        return dict(content=html_content, filename=filename)

    except Exception as e:
        print(f"Error during HTML download generation: {e}\n{traceback.format_exc()}")
        error_content = f"Error generating exploration HTML: {e}\n\n{traceback.format_exc()}"
        error_html = f'''
        <!DOCTYPE html>
        <html>
        <head><title>Download Error</title></head>
        <body>
            <h1>Error Generating Report</h1>
            <p>An error occurred while trying to generate the HTML report:</p>
            <pre>{error_content}</pre>
        </body>
        </html>
        '''
        safe_ark = "".join(c if c.isalnum() or c in ['-', '_', '.'] else "_" for c in data_ark or "error")[:100]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_ark}_{timestamp}_error.html"
        return dict(content=error_html, filename=filename)