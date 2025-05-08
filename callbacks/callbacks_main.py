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
    Output('column-metadata-display-container', 'children', allow_duplicate=True),
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
    status_message, status_color = update_status("Initializing...", "secondary")
    reset_summary = ["Load data to see summary."]
    empty_fig = create_empty_figure("Load data first")
    initial_column_metadata_placeholder = dbc.Alert(
        "Select a numeric column from 'Plot Options' to view its metadata.",
        color="info", className="small p-2 text-center"
    )
    reset_model_summary = ["Build a model to see results."]
    reset_model_plot = create_placeholder_plot("Build a model to see plot.")
    initial_rule_container_children = [create_rule_row(0)]

    initial_outputs = [
        None, None, None, None,
        status_message, status_color, True, True,
        reset_summary, empty_fig, initial_column_metadata_placeholder, "", initial_rule_container_children, "",
        None, "", None, True,
        reset_model_summary, reset_model_plot, "",
        [], None, True, [], None, True,
    ]

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

        schema_ark_found, schema_msg = find_schema_ark(data_metadata)
        schema_status_prefix = "Metadata OK."
        status_message = f"{schema_status_prefix} {schema_msg}"
        if not schema_ark_found and status_color != "danger": status_color = "warning"


        full_path = data_metadata.get("distribution", {}).get("location", {}).get("path")
        if not full_path:
            content_url = data_metadata.get("distribution", {}).get("contentUrl")
            if content_url and urlparse(content_url).scheme == 's3':
                 full_path = urlparse(content_url).path.lstrip('/')
            else:
                 raise ValueError(f"Data location ('distribution.location.path') missing for {data_ark}.")

        bucket = MINIO_DEFAULT_BUCKET
        key_path = full_path
        csv_file_name_in_zip = None
        is_zip_file = ".zip/" in key_path or key_path.lower().endswith('.zip')

        if ".zip/" in key_path:
            zip_path_parts = key_path.split(".zip/", 1)
            key_path = zip_path_parts[0] + ".zip"
            csv_file_name_in_zip = zip_path_parts[1]

        status_message, status_color = update_status("Fetching/Extracting Data...", "info")
        file_content = None
        if is_zip_file:
            if csv_file_name_in_zip:
                file_content = extract_from_zip_in_s3(s3, bucket, key_path, csv_file_name_in_zip)
            else:
                s3_file = S3File(s3, bucket, key_path)
                with zipfile.ZipFile(s3_file, 'r') as zip_f:
                    csv_files = [f for f in zip_f.namelist() if f.lower().endswith('.csv') and not f.startswith('__MACOSX/')]
                    if not csv_files: raise FileNotFoundError(f"No CSV in ZIP: s3://{bucket}/{key_path}")
                    csv_file_name_in_zip = csv_files[0]
                    with zip_f.open(csv_file_name_in_zip) as file_in_zip: file_content = file_in_zip.read()
        else:
            response = s3.get_object(Bucket=bucket, Key=key_path)
            file_content = response['Body'].read()

        status_message, status_color = update_status("Parsing Data File...", "info")
        if not file_content:
             raw_data_df = pd.DataFrame()
        else:
            try:
                raw_data_df = pd.read_csv(io.BytesIO(file_content))
            except UnicodeDecodeError:
                raw_data_df = pd.read_csv(io.BytesIO(file_content), encoding='latin-1') # Fallback
            # Further specific read errors will be caught by the main try-except

        conversion_warnings = []
        if schema_ark_found:
            status_message, status_color = update_status("Processing Schema...", "info")
            try:
                schema_url = f"{FAIRSCAPE_BASE_URL}/{schema_ark_found}"
                response_schema = requests.get(schema_url, timeout=30); response_schema.raise_for_status()
                schema_data = response_schema.json()['metadata']; properties = {}

                if isinstance(schema_data, list): # Handle schema if it's a list (e.g. from older Fairscape versions)
                    if schema_data and isinstance(schema_data[0], dict) and 'properties' in schema_data[0]:
                       properties = schema_data[0].get('properties', {})
                elif isinstance(schema_data, dict): # Standard case
                    properties = schema_data.get('properties', {})

                if properties and isinstance(properties, dict):
                    props_list = [
                        {'name': n, 'type': d.get('type'), 'description': d.get('description'), 'value-url': d.get('value-url')}
                        for n, d in properties.items() if isinstance(d, dict)
                    ]
                    if props_list:
                        schema_props = pd.DataFrame(props_list); schema_props_dict = schema_props.to_dict('records')
                    else: conversion_warnings.append("Schema 'properties' has no definitions.")
                else: conversion_warnings.append("Schema has no 'properties' dictionary.")
            except Exception as e:
                conversion_warnings.append(f"Failed to process schema {schema_ark_found}: {e}")
                if status_color != "danger": status_color = "warning" # Downgrade unless already critical

        typed_data = raw_data_df.copy()
        if not schema_props.empty and not raw_data_df.empty:
            status_message, status_color = update_status("Applying Schema Types...", "info")
            for _, prop in schema_props.iterrows():
                col_name = prop['name']; schema_type = prop['type']
                if col_name in typed_data.columns:
                    try:
                        original_series = typed_data[col_name]
                        # More robust NA detection for original mask
                        original_na_mask = original_series.isna() | \
                                           (original_series.astype(str).str.strip().str.lower().isin(
                                               ['na', '<na>', 'none', 'nan', 'null', '']))
                        
                        converted_series = None
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
                                    conversion_warnings.append(f"'{col_name}' (int) has decimals; stored as Float.")
                                else:
                                    converted_series = num_col.astype(pd.Int64Dtype()) # All NaN or empty
                            elif schema_type == 'number':
                                converted_series = pd.to_numeric(original_series, errors='coerce').astype(pd.Float64Dtype())
                            elif schema_type == 'boolean':
                                 bool_map = {'true': True, 't': True, '1': True, 'yes': True, 'y': True, '1.0': True,
                                             'false': False, 'f': False, '0': False, 'no': False, 'n': False, '0.0': False}
                                 str_series = original_series.astype(str).str.strip().str.lower()
                                 converted_series = str_series.map(bool_map).astype(pd.BooleanDtype())
                            elif schema_type == 'string':
                                 converted_series = original_series.astype(str).replace(
                                     {'nan': pd.NA, 'None': pd.NA, '': pd.NA, 'NA': pd.NA, '<NA>': pd.NA, 'null': pd.NA},
                                     regex=False).astype(pd.StringDtype())
                        
                        if converted_series is not None:
                             typed_data[col_name] = converted_series.where(~original_na_mask, pd.NA)
                    except Exception as type_e:
                         conversion_warnings.append(f"Type error for '{col_name}' to '{schema_type}': {type_e}")
        elif not schema_ark_found and status_color != 'danger':
             status_message = "Data Parsed. No schema found/applied."; status_color = "warning"

        if conversion_warnings:
            base_status = status_message.split("...")[0] + "..." if "..." in status_message else status_message
            if status_color != "danger": status_color = "warning"
            max_warn=3; warn_str = "\n".join(conversion_warnings[:max_warn])
            if len(conversion_warnings)>max_warn: warn_str+=f"\n... ({len(conversion_warnings)-max_warn} more)"
            status_message = f"{base_status} Warnings:\n{warn_str}"
        elif status_color != "danger":
            status_message = "Processing Complete."; status_color = "success"

        n_plot, g_plot, n_model, all_ui = [], [], [], []
        for col in typed_data.columns:
            if col=='Unnamed: 0': continue # Skip common index column
            dt=typed_data[col].dtype
            all_ui.append({'label':col, 'value':col})
            
            if typed_data[col].notna().any():
                is_suitable_for_model_predictor = pd.api.types.is_numeric_dtype(dt) # Booleans are numeric
                is_numeric_for_plot = pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_bool_dtype(dt)
                
                is_groupable = False
                nu = typed_data[col].nunique(dropna=True)
                if pd.api.types.is_string_dtype(dt) or \
                   pd.api.types.is_bool_dtype(dt) or \
                   pd.api.types.is_categorical_dtype(dt) or \
                   ((pd.api.types.is_integer_dtype(dt) or pd.api.types.is_object_dtype(dt)) and nu <= 50): # Group if few unique ints/objects
                       is_groupable = True
                elif pd.api.types.is_object_dtype(dt):
                     try: # Check if object can be numeric and how many uniques
                         if pd.to_numeric(typed_data[col], errors='coerce').nunique(dropna=True) > 50 : is_groupable = False
                     except: pass # If not convertible, it might remain groupable as string/object if few uniques

                if is_suitable_for_model_predictor:
                    n_model.append(col)
                if is_numeric_for_plot: # For histogram main variable
                    if not is_groupable or (pd.api.types.is_numeric_dtype(dt) and nu > 10): # Don't plot low-cardinality numerics here if they are groupable
                         n_plot.append(col)
                if is_groupable:
                    g_plot.append(col)

        available_columns_data = {
            'numeric_plot': sorted(list(set(n_plot))),
            'grouping_plot': sorted(list(set(g_plot))),
            'numeric_model': sorted(list(set(n_model))),
            'all': sorted(all_ui, key=lambda x: x['label'])
        }
        processed_data_dict = typed_data.to_dict('records')
        cohort_data_dict = processed_data_dict

        final_outputs = [
            processed_data_dict, cohort_data_dict, schema_props_dict, available_columns_data,
            status_message, status_color, False, False,
            reset_summary, empty_fig, initial_column_metadata_placeholder, "", initial_rule_container_children, "",
            None, "", None, True,
            reset_model_summary, reset_model_plot, "",
            [], None, True, [], None, True,
        ]
        return tuple(final_outputs)

    except Exception as e:
        tb_str = traceback.format_exc(); print(f"Error processing ARK {data_ark}:\n{tb_str}") # Keep this for server-side debugging
        status_message = f"Error loading/processing ARK: {e}"; status_color = "danger"
        initial_outputs[4] = status_message
        initial_outputs[5] = status_color
        initial_outputs[6] = True # Disable download buttons
        initial_outputs[7] = True
        initial_outputs[8] = [html.P(f"Failed to load data: {e}", className="text-danger")]
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

    num_plot_cols = available_cols_data.get('numeric_plot', [])
    all_cols_options = available_cols_data.get('all', [])

    num_plot_opts = [{'label': c, 'value': c} for c in num_plot_cols]
    num_plot_val = num_plot_cols[0] if num_plot_cols else None
    num_plot_dis = not bool(num_plot_opts)

    main_join_opts = all_cols_options
    main_join_val = None
    main_join_dis = not bool(main_join_opts)

    num_rules = len(rule_rows) if rule_rows else 0
    rule_opts_all = [all_cols_options] * num_rules
    rule_vals_all = [None] * num_rules
    rule_dis_all = [not bool(all_cols_options)] * num_rules
    
    return num_plot_opts, num_plot_val, num_plot_dis, main_join_opts, main_join_val, main_join_dis, rule_opts_all, rule_vals_all, rule_dis_all


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
     # print(f"Updating summary triggered by: {ctx.triggered_id}") # Optional: keep for understanding triggers
     return update_summary_content(cohort_data_dict)