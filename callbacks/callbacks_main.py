# mine/callbacks/callbacks_main.py
import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import requests
import io
import zipfile
from urllib.parse import urlparse # Keep for potential future use, but not for core path logic now
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
    Output('cohort-data-store', 'data', allow_duplicate=True), # Initial cohort data
    Output('schema-properties-store', 'data'),
    Output('available-columns-store', 'data'),
    # --- UI Feedback & Control ---
    Output('status-alert', 'children'),
    Output('status-alert', 'color'),
    Output('btn-download-summary', 'disabled'),
    # --- Outputs for Resetting UI elements controlled elsewhere ---
    Output('data-summary', 'children', allow_duplicate=True), # Reset summary view
    Output('data-histogram', 'figure', allow_duplicate=True), # Reset plot view
    Output('cohort-name-input', 'value'), # Reset cohort name
    Output('rule-builder-container', 'children', allow_duplicate=True), # Reset rules
    Output('rule-apply-status', 'children', allow_duplicate=True), # Clear rule status
    Output('upload-cohort-csv', 'contents'), # Clear upload trigger
    Output('upload-status', 'children', allow_duplicate=True), # Clear upload status
    Output('uploaded-cohort-store', 'data', allow_duplicate=True), # Clear uploaded data store (allow_dup in case upload callback clears it too)
    Output('process-upload-button', 'disabled', allow_duplicate=True), # Disable upload process btn
    Output('model-summary-output', 'children', allow_duplicate=True), # Reset model summary
    Output('model-plot-output', 'children', allow_duplicate=True), # Reset model plot
    Output('model-status', 'children', allow_duplicate=True), # Reset model status
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
    reset_model_summary = ["Build a model to see results."]
    reset_model_plot = create_placeholder_plot("Build a model to see plot.")
    initial_rule_container_children = [create_rule_row(0)]

    # Define the full list of outputs for the reset/error case
    # Order must match the @callback Output list
    initial_outputs = [
        None, None, None, None, # Data stores
        status_message, status_color, True, # Status, Download button disabled
        reset_summary, empty_fig, "", initial_rule_container_children, "", # UI Resets
        None, "", None, True, # Rule/Upload resets (contents, status, store, button)
        reset_model_summary, reset_model_plot, "", # Model resets
        [], None, True, [], None, True, # Upload selector resets
    ]

    if not data_ark:
        status_message, status_color = update_status("Please provide a Data ARK identifier.", "warning")
        initial_outputs[4:6] = status_message, status_color
        return tuple(initial_outputs)

    # Initialize variables
    schema_props = pd.DataFrame()
    schema_props_dict = None
    raw_data_df = pd.DataFrame()
    typed_data = pd.DataFrame()

    try:
        # --- Step 1: Fetch Metadata ---
        status_message, status_color = update_status("Fetching Metadata...", "info")
        data_meta_url = f"{FAIRSCAPE_BASE_URL}/{data_ark}"
        response_data = requests.get(data_meta_url, timeout=30)
        response_data.raise_for_status()
        data_metadata = response_data.json()

        # --- Step 2: Find Schema ARK ---
        status_message, status_color = update_status("Finding Schema...", "info")
        schema_ark_found, schema_msg = find_schema_ark(data_metadata) # Using the original find_schema_ark logic from user file
        schema_status_prefix = "Metadata OK."
        if not schema_ark_found:
            status_message = f"{schema_status_prefix} {schema_msg}"
            if status_color != "danger": status_color = "warning"
        else:
            status_message = f"{schema_status_prefix} {schema_msg}"

        # --- Step 3: Locate Data File Path (Reverted to Original Logic) ---
        status_message, status_color = update_status("Locating Data File...", "info")
        # Get path directly from distribution.location.path
        full_path = data_metadata.get("distribution", {}).get("location", {}).get("path")
        if not full_path:
            # Fallback attempt if primary path is missing (optional, based on observed metadata)
            content_url = data_metadata.get("distribution", {}).get("contentUrl")
            if content_url and urlparse(content_url).scheme == 's3': # Basic check if it looks like S3 URL
                 full_path = urlparse(content_url).path.lstrip('/')
                 print(f"Warning: Using path from contentUrl: {full_path}")
            else:
                 raise ValueError(f"Data location ('distribution.location.path') missing in metadata for {data_ark}.")

        # Use default bucket and the extracted path as key
        bucket = MINIO_DEFAULT_BUCKET
        key_path = full_path # The path from metadata is the S3 key
        csv_file_name_in_zip = None
        is_zip_file = False
        # Determine if it's a zip file and extract inner path if necessary
        if ".zip/" in key_path:
            zip_path_parts = key_path.split(".zip/", 1)
            key_path = zip_path_parts[0] + ".zip" # S3 key is the .zip file itself
            csv_file_name_in_zip = zip_path_parts[1] # Name of the file inside the zip
            is_zip_file = True
        elif key_path.lower().endswith('.zip'):
            is_zip_file = True # It's a zip file, but no inner path specified

        # --- Step 4: Fetch/Extract Data File ---
        status_message, status_color = update_status("Fetching/Extracting Data File...", "info")
        file_content = None
        if is_zip_file:
            try:
                if csv_file_name_in_zip:
                    # Specific file requested inside the zip
                    file_content = extract_from_zip_in_s3(s3, bucket, key_path, csv_file_name_in_zip)
                else:
                    # No specific file requested, find the first CSV
                    s3_file = S3File(s3, bucket, key_path)
                    with zipfile.ZipFile(s3_file, 'r') as zip_f:
                        csv_files = [f for f in zip_f.namelist() if f.lower().endswith('.csv') and not f.startswith('__MACOSX/')]
                        if not csv_files: raise FileNotFoundError(f"No CSV file found inside ZIP: s3://{bucket}/{key_path}")
                        csv_file_name_in_zip = csv_files[0] # Take the first one
                        print(f"Auto-selected '{csv_file_name_in_zip}' from zip s3://{bucket}/{key_path}")
                        with zip_f.open(csv_file_name_in_zip) as file_in_zip: file_content = file_in_zip.read()
            except Exception as e: raise ValueError(f"Error processing ZIP s3://{bucket}/{key_path}: {e}") from e
        else:
            # Not a zip file, download directly
            try:
                response = s3.get_object(Bucket=bucket, Key=key_path)
                file_content = response['Body'].read()
            except Exception as e: raise ValueError(f"Error downloading file s3://{bucket}/{key_path}: {e}") from e

        # --- Step 5: Parse Data File ---
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

        # --- Step 6: Process Schema and Apply Types ---
        conversion_warnings = []
        if schema_ark_found:
            status_message, status_color = update_status("Processing Schema...", "info")
            try:
                schema_url = f"{FAIRSCAPE_BASE_URL}/{schema_ark_found}"
                response_schema = requests.get(schema_url, timeout=30); response_schema.raise_for_status()
                schema_data = response_schema.json(); properties = {} # Use schema_data instead of schema_list
                # Handle potential list vs dict structure for schema properties
                if isinstance(schema_data, list):
                    if schema_data and isinstance(schema_data[0], dict) and 'properties' in schema_data[0]:
                       properties = schema_data[0].get('properties', {})
                    else: print("Warning: Schema is a list but structure not recognized for properties.")
                elif isinstance(schema_data, dict): # Standard case
                    properties = schema_data.get('properties', {})

                if properties and isinstance(properties, dict):
                    props_list = [{'name': n, 'type': d.get('type'), 'description': d.get('description')} for n, d in properties.items() if isinstance(d, dict)]
                    if props_list:
                        schema_props = pd.DataFrame(props_list); schema_props_dict = schema_props.to_dict('records')
                    else: conversion_warnings.append("Schema 'properties' contains no valid definitions."); status_color = "warning"
                else: conversion_warnings.append("Schema found but has no 'properties' dictionary."); status_color = "warning"
            except Exception as e: conversion_warnings.append(f"Failed to process schema {schema_ark_found}: {e}"); status_color = "danger"; print(traceback.format_exc())

        typed_data = raw_data_df.copy()
        if not schema_props.empty and not raw_data_df.empty:
            status_message, status_color = update_status("Applying Schema Types...", "info")
            # Type application logic using original approach for reference
            for _, prop in schema_props.iterrows():
                col_name = prop['name']; schema_type = prop['type']
                if col_name in typed_data.columns:
                    try:
                        original_series = typed_data[col_name]; converted_series = None
                        original_na_mask = original_series.isna() | (original_series.astype(str).str.lower().isin(['na', '<na>', 'none', 'nan', 'null', '']))

                        # Check compatibility (simplified - add more checks if needed)
                        current_dtype_kind = original_series.dtype.kind
                        is_compatible = False
                        if schema_type == 'integer' and current_dtype_kind in 'iu': is_compatible = True
                        elif schema_type == 'number' and current_dtype_kind in 'iuf': is_compatible = True # Allow int for number schema
                        elif schema_type == 'boolean' and current_dtype_kind == 'b': is_compatible = True
                        elif schema_type == 'string' and current_dtype_kind in 'OSUV': is_compatible = True # Object, String, Unicode, Void

                        if is_compatible:
                            converted_series = original_series # Keep original if compatible enough
                        else:
                            # Attempt conversion based on schema type (using pandas dtypes)
                            if schema_type == 'integer':
                                num_col = pd.to_numeric(original_series, errors='coerce')
                                # Check if conversion to Int64 is feasible (no non-integer floats)
                                if num_col.notna().all() and num_col.dropna().mod(1).eq(0).all():
                                    converted_series = num_col.astype(pd.Int64Dtype())
                                elif num_col.notna().any(): # Has floats, store as Float64
                                    converted_series = num_col.astype(pd.Float64Dtype())
                                    conversion_warnings.append(f"'{col_name}' (int schema) has decimals, stored as Float.")
                                else: # All NaN or failed conversion
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

                        # Apply conversion, preserving original NAs
                        if converted_series is not None:
                             typed_data[col_name] = converted_series.where(~original_na_mask, pd.NA)
                    except Exception as type_e:
                         conversion_warnings.append(f"Type error converting '{col_name}' to '{schema_type}': {type_e}")
                # else: conversion_warnings.append(f"Schema col '{col_name}' not in data.") # Optional warning

        elif not schema_ark_found and status_color != 'danger':
             status_message = "Data Parsed. No schema found or applied."; status_color = "warning"

        # --- Step 7: Final Status Update ---
        if conversion_warnings:
            base_status = status_message.split("...")[0] + "..." if "..." in status_message else status_message
            if status_color != "danger": status_color = "warning"
            max_warn=5; warn_str = "\n".join(conversion_warnings[:max_warn])
            if len(conversion_warnings)>max_warn: warn_str+=f"\n... ({len(conversion_warnings)-max_warn} more)"
            status_message = f"{base_status} Warnings:\n{warn_str}"
        elif status_color != "danger": status_message = "Processing Complete. Explore data or define cohorts."; status_color = "success"

        # --- Step 8: Column Classification ---
        n_plot, g_plot, n_model, all_ui = [], [], [], []
        for col in typed_data.columns:
            if col=='Unnamed: 0': continue
            dt=typed_data[col].dtype; all_ui.append({'label':col, 'value':col})
            if typed_data[col].notna().any():
                # Simplified check: numeric for plot/model if numeric and not bool
                is_numeric = pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_bool_dtype(dt)

                # Grouping: String, Bool, Category, or Low-Cardinality Integer/Object
                is_groupable = False
                nu = typed_data[col].nunique(dropna=True)
                if pd.api.types.is_string_dtype(dt) or \
                   pd.api.types.is_bool_dtype(dt) or \
                   pd.api.types.is_categorical_dtype(dt) or \
                   ((pd.api.types.is_integer_dtype(dt) or pd.api.types.is_object_dtype(dt)) and nu <= 50):
                       is_groupable = True
                # Avoid grouping high-cardinality floats even if read as object initially
                elif pd.api.types.is_object_dtype(dt):
                     try:
                         if pd.to_numeric(typed_data[col], errors='coerce').nunique(dropna=True) > 50: is_groupable = False
                     except: pass # Ignore conversion errors

                if is_numeric:
                    n_model.append(col)
                    # Only add to numeric plot list if it's NOT likely groupable (e.g. high cardinality float)
                    if not is_groupable or not (pd.api.types.is_integer_dtype(dt) or nu <= 10): # Heuristic: don't plot low card ints as numeric hist
                         n_plot.append(col)

                if is_groupable:
                    g_plot.append(col)

        available_columns_data = {
            'numeric_plot': sorted(list(set(n_plot))), # Ensure unique
            'grouping_plot': sorted(list(set(g_plot))), # Ensure unique
            'numeric_model': sorted(list(set(n_model))), # Ensure unique
            'all': sorted(all_ui, key=lambda x: x['label'])
        }
        processed_data_dict = typed_data.to_dict('records');
        cohort_data_dict = processed_data_dict

        # --- Step 9: Prepare Final Outputs ---
        final_outputs = [
            processed_data_dict, cohort_data_dict, schema_props_dict, available_columns_data, # Data stores
            status_message, status_color, False, # Status, Download button enabled
            reset_summary, empty_fig, "", initial_rule_container_children, "", # Reset UI placeholders
            None, "", None, True, # Reset Upload UI state
            reset_model_summary, reset_model_plot, "", # Reset Model UI placeholders
            [], None, True, [], None, True, # Reset Upload selectors
        ]
        return tuple(final_outputs)

    except Exception as e:
        tb_str = traceback.format_exc(); print(f"Error processing ARK {data_ark}:\n{tb_str}")
        status_message = f"Error loading/processing ARK: {e}"; status_color = "danger"
        # Return the initial_outputs structure but with the error message
        initial_outputs[4:6] = status_message, status_color
        initial_outputs[6] = True # Download disabled
        initial_outputs[7] = [html.P(f"Failed to load data: {e}", className="text-danger")] # Error in summary
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


# --- Summary Download ---
@callback(
    Output("download-summary", "data"),
    Input("btn-download-summary", "n_clicks"),
    State("cohort-data-store", "data"),
    State("data-ark-input", "value"),
    State("available-columns-store", "data"),
    prevent_initial_call=True,
)
def download_summary_text(n_clicks, data_dict, data_ark, available_cols_data):
    if not data_dict: return None
    try:
        df = pd.DataFrame(data_dict)
        buffer = io.StringIO()
        buffer.write(f"Summary for ARK: {data_ark or 'N/A'}\n")
        buffer.write(f"Timestamp: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n")

        orig_cols = [c['value'] for c in available_cols_data['all']] if available_cols_data else []
        cohort_cols = [c for c in df.columns if c not in orig_cols and c != 'Unnamed: 0']
        if cohort_cols: buffer.write(f"Cohort Columns: {', '.join(cohort_cols)}\n")

        buffer.write(f"\nShape: {df.shape}\n\nInfo:\n")
        df.info(buf=buffer)
        buffer.write("\n\nStats:\n")
        try: desc = df.describe(include='all', datetime_is_numeric=True).to_string()
        except: desc = df.describe(include='all').to_string()
        buffer.write(desc)
        summary_text = buffer.getvalue()

        safe_ark = "".join(c if c.isalnum() else "_" for c in data_ark or "")[:50] or "data"
        filename = f"{safe_ark}_summary_{pd.Timestamp.now():%Y%m%d_%H%M}.txt"
        return dict(content=summary_text, filename=filename)
    except Exception as e:
        print(f"Summary download error: {e}")
        return dict(content=f"Error: {e}\n{traceback.format_exc()}", filename="summary_error.txt")