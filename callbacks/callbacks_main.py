import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import requests
import io
import zipfile
from urllib.parse import urlparse, parse_qs
import traceback

from utils.s3_utils import get_s3_client, extract_from_zip_in_s3, S3File
from config import FAIRSCAPE_BASE_URL, MINIO_DEFAULT_BUCKET
from layout import create_rule_row
from utils.app_utils import (
    update_status, find_schema_ark, update_summary_content,
    create_empty_figure, create_placeholder_plot
)
# Import the default card if you plan to use it for initial state
# from callbacks.model_callbacks import default_equation_card # Optional

s3 = get_s3_client()

@callback(
    Output('processed-data-store', 'data'),
    Output('cohort-data-store', 'data', allow_duplicate=True),
    Output('schema-properties-store', 'data'),
    Output('available-columns-store', 'data'),
    Output('status-alert', 'children'),
    Output('status-alert', 'color'),
    Output('btn-download-summary', 'disabled'),
    Output('btn-download-html', 'disabled'),
    Output('data-summary', 'children', allow_duplicate=True),
    Output('data-histogram', 'figure', allow_duplicate=True),
    Output('column-metadata-display-container', 'children', allow_duplicate=True),
    Output('cohort-name-input', 'value'),
    Output('data-ark-input', 'value', allow_duplicate=True),
    # REMOVED: Output('sidebar-offcanvas', 'is_open', allow_duplicate=True),
    Output('rule-builder-container', 'children', allow_duplicate=True),
    Output('rule-apply-status', 'children', allow_duplicate=True),
    Output('upload-cohort-csv', 'contents'),
    Output('upload-status', 'children', allow_duplicate=True),
    Output('uploaded-cohort-store', 'data', allow_duplicate=True),
    Output('process-upload-button', 'disabled', allow_duplicate=True),
    Output('model-summary-output', 'children', allow_duplicate=True),
    Output('model-plot-output', 'children', allow_duplicate=True),
    Output('model-status', 'children', allow_duplicate=True),
    Output('model-equation-container', 'children', allow_duplicate=True),
    Output('upload-join-column-selector', 'options', allow_duplicate=True),
    Output('upload-join-column-selector', 'value', allow_duplicate=True),
    Output('upload-join-column-selector', 'disabled', allow_duplicate=True),
    Output('upload-cohort-column-selector', 'options', allow_duplicate=True),
    Output('upload-cohort-column-selector', 'value', allow_duplicate=True),
    Output('upload-cohort-column-selector', 'disabled', allow_duplicate=True),
    Input('load-button', 'n_clicks'),
    Input('url', 'search'),
    State('data-ark-input', 'value'),
    prevent_initial_call='initial_duplicate'
)
def load_and_process_ark(n_clicks, search_query, input_ark_from_state):
    triggered_id = ctx.triggered_id
    data_ark_to_load = None
    ark_for_input_field_update = no_update
    # REMOVED: sidebar_status_update = no_update

    ark_from_url = None
    # REMOVED: is_url_trigger = False
    if search_query:
        parsed_url = urlparse(search_query)
        query_params = parse_qs(parsed_url.query)
        ark_from_url = query_params.get('ark', [None])[0]

    if triggered_id == 'load-button' and n_clicks:
        data_ark_to_load = input_ark_from_state
        ark_for_input_field_update = no_update
    elif ark_from_url and (triggered_id == 'url' or (triggered_id is None and not n_clicks)):
        data_ark_to_load = ark_from_url
        ark_for_input_field_update = ark_from_url
        # REMOVED: is_url_trigger = True
    elif triggered_id is None and not n_clicks and input_ark_from_state :
         data_ark_to_load = None
         ark_for_input_field_update = input_ark_from_state

    status_message, status_color = update_status("Initializing...", "secondary")
    reset_summary_content = ["Load data to see summary."]
    empty_fig_placeholder = create_empty_figure("Load data first")
    initial_column_metadata_placeholder = dbc.Alert(
        "Select a numeric column from 'Plot Options' to view its metadata.",
        color="info", className="small p-2 text-center"
    )
    reset_model_summary_text = ["Build a model to see results."]
    reset_model_plot_placeholder = create_placeholder_plot("Build a model to see plot.")
    initial_rule_container_children = [create_rule_row(0)]
    reset_model_equation_content = None

    # Adjust indices for removed output
    initial_outputs_template_list = [
        None, None, None, None,
        status_message, status_color, True, True,
        reset_summary_content, empty_fig_placeholder, initial_column_metadata_placeholder,
        "",
        no_update, # for data-ark-input.value (index 12)
        # Index 13 was sidebar-offcanvas, now rule-builder-container
        initial_rule_container_children, "", # rule-builder, rule-status
        None, "", None, True,
        reset_model_summary_text, reset_model_plot_placeholder, "",
        reset_model_equation_content,
        [], None, True,
        [], None, True,
    ]

    if not data_ark_to_load:
        current_outputs = list(initial_outputs_template_list)
        if triggered_id is None and not n_clicks:
            new_status_message, new_status_color = update_status("Enter a Data ARK or use the URL (e.g., ?ark=your_ark_id) and click Load.", "info")
        else:
            new_status_message, new_status_color = update_status("Please provide a Data ARK identifier.", "warning")
        current_outputs[4] = new_status_message
        current_outputs[5] = new_status_color
        current_outputs[12] = ark_for_input_field_update
        return tuple(current_outputs)

    schema_props = pd.DataFrame()
    schema_props_dict = None
    raw_data_df = pd.DataFrame()
    typed_data = pd.DataFrame()
    available_columns_data = {}

    try:
        status_message, status_color = update_status("Fetching Metadata...", "info")
        data_meta_url = f"{FAIRSCAPE_BASE_URL}/{data_ark_to_load}"
        response_data = requests.get(data_meta_url, timeout=30)
        response_data.raise_for_status()
        data_metadata = response_data.json()

        schema_ark_found, schema_msg = find_schema_ark(data_metadata)
        status_message = f"Metadata OK. {schema_msg}"
        if not schema_ark_found and status_color != "danger": status_color = "warning"

        full_path = data_metadata.get("distribution", {}).get("location", {}).get("path")
        if not full_path:
            content_url = data_metadata.get("distribution", {}).get("contentUrl")
            if content_url and urlparse(content_url).scheme == 's3':
                 full_path = urlparse(content_url).path.lstrip('/')
            else:
                 raise ValueError(f"Data location ('distribution.location.path' or S3 'contentUrl') missing for {data_ark_to_load}.")

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
                    if not csv_files: raise FileNotFoundError(f"No CSV file found in ZIP: s3://{bucket}/{key_path}")
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
                raw_data_df = pd.read_csv(io.BytesIO(file_content), encoding='latin-1')

        conversion_warnings = []
        if schema_ark_found:
            status_message, status_color = update_status("Processing Schema...", "info")
            try:
                schema_url = f"{FAIRSCAPE_BASE_URL}/{schema_ark_found}"
                response_schema = requests.get(schema_url, timeout=30); response_schema.raise_for_status()
                schema_json_data = response_schema.json()
                schema_data_content = schema_json_data.get('metadata', schema_json_data)
                properties = {}

                if isinstance(schema_data_content, list):
                    if schema_data_content and isinstance(schema_data_content[0], dict) and 'properties' in schema_data_content[0]:
                       properties = schema_data_content[0].get('properties', {})
                elif isinstance(schema_data_content, dict):
                    properties = schema_data_content.get('properties', {})

                if properties and isinstance(properties, dict):
                    props_list = [
                        {'name': n, 'type': d.get('type'), 'description': d.get('description'), 'value-url': d.get('value-url')}
                        for n, d in properties.items() if isinstance(d, dict)
                    ]
                    if props_list:
                        schema_props = pd.DataFrame(props_list); schema_props_dict = schema_props.to_dict('records')
                    else: conversion_warnings.append("Schema 'properties' field has no valid column definitions.")
                else: conversion_warnings.append("Schema data does not contain a valid 'properties' dictionary.")
            except Exception as e:
                conversion_warnings.append(f"Failed to process schema {schema_ark_found}: {e}")
                if status_color != "danger": status_color = "warning"

        typed_data = raw_data_df.copy()
        if not schema_props.empty and not raw_data_df.empty:
            status_message, status_color = update_status("Applying Schema Types...", "info")
            for _, prop in schema_props.iterrows():
                col_name = prop['name']; schema_type = prop['type']
                if col_name in typed_data.columns:
                    try:
                        original_series = typed_data[col_name]
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
                                    conversion_warnings.append(f"Column '{col_name}' (schema type: integer) contains decimal values or could not be fully converted to integer; stored as Float64.")
                                else: 
                                    converted_series = num_col.astype(pd.Int64Dtype())
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
                         conversion_warnings.append(f"Type conversion error for column '{col_name}' to schema type '{schema_type}': {type_e}")
        elif not schema_ark_found and status_color != 'danger':
             status_message = "Data Parsed. No schema found or applied."; status_color = "warning"

        if conversion_warnings:
            base_status = status_message.split("...")[0] + "..." if "..." in status_message else status_message
            if status_color != "danger": status_color = "warning"
            max_warn=3; warn_str = "\n".join(conversion_warnings[:max_warn])
            if len(conversion_warnings)>max_warn: warn_str+=f"\n... ({len(conversion_warnings)-max_warn} more warnings)"
            status_message = f"{base_status} Type Conversion Warnings:\n{warn_str}"
        elif status_color != "danger":
            status_message = "Data and Schema Processing Complete."; status_color = "success"

        n_plot_cols, g_plot_cols, n_model_cols, all_ui_cols = [], [], [], []
        for col in typed_data.columns:
            if col=='Unnamed: 0': continue
            col_dtype = typed_data[col].dtype
            all_ui_cols.append({'label':col, 'value':col})
            
            if typed_data[col].notna().any():
                is_numeric_for_plot = pd.api.types.is_numeric_dtype(col_dtype) and not pd.api.types.is_bool_dtype(col_dtype)
                is_suitable_for_model_predictor = pd.api.types.is_numeric_dtype(col_dtype)

                is_groupable = False
                nunique_vals = typed_data[col].nunique(dropna=True)
                if pd.api.types.is_string_dtype(col_dtype) or \
                   pd.api.types.is_bool_dtype(col_dtype) or \
                   pd.api.types.is_categorical_dtype(col_dtype) or \
                   ((pd.api.types.is_integer_dtype(col_dtype) or pd.api.types.is_object_dtype(col_dtype)) and nunique_vals <= 50):
                       is_groupable = True
                elif pd.api.types.is_object_dtype(col_dtype):
                     try:
                         if pd.to_numeric(typed_data[col], errors='coerce').nunique(dropna=True) <= 50 : is_groupable = True
                     except: pass

                if is_suitable_for_model_predictor:
                    n_model_cols.append(col)
                if is_numeric_for_plot:
                    if not is_groupable or (pd.api.types.is_numeric_dtype(col_dtype) and nunique_vals > 10):
                         n_plot_cols.append(col)
                if is_groupable:
                    g_plot_cols.append(col)

        available_columns_data = {
            'numeric_plot': sorted(list(set(n_plot_cols))),
            'grouping_plot': sorted(list(set(g_plot_cols))),
            'numeric_model': sorted(list(set(n_model_cols))),
            'all': sorted(all_ui_cols, key=lambda x: x['label'])
        }
        processed_data_dict = typed_data.to_dict('records') if not typed_data.empty else None
        cohort_data_dict = processed_data_dict

        # REMOVED: if is_url_trigger: sidebar_status_update = False

        success_outputs = list(initial_outputs_template_list)
        success_outputs[0] = processed_data_dict
        success_outputs[1] = cohort_data_dict
        success_outputs[2] = schema_props_dict
        success_outputs[3] = available_columns_data
        success_outputs[4] = status_message
        success_outputs[5] = status_color
        success_outputs[6] = typed_data.empty
        success_outputs[7] = typed_data.empty
        success_outputs[8] = update_summary_content(cohort_data_dict)
        success_outputs[9] = create_empty_figure("Data loaded. Select plot options.") if typed_data.empty else empty_fig_placeholder
        success_outputs[10] = initial_column_metadata_placeholder
        success_outputs[12] = ark_for_input_field_update
        # No longer updating sidebar from here: success_outputs[13] remains as per template (now rule-builder)

        return tuple(success_outputs)

    except Exception as e:
        tb_str = traceback.format_exc(); print(f"Error processing ARK {data_ark_to_load}:\n{tb_str}")
        error_status_message = f"Error loading/processing ARK: {e}"; error_status_color = "danger"
        
        error_outputs_list = list(initial_outputs_template_list)
        error_outputs_list[4] = error_status_message
        error_outputs_list[5] = error_status_color
        error_outputs_list[6] = True
        error_outputs_list[7] = True
        error_outputs_list[8] = [html.P(f"Failed to load data: {e}", className="text-danger")]
        error_outputs_list[12] = ark_for_input_field_update
        
        return tuple(error_outputs_list)

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


@callback(
    Output("sidebar-offcanvas", "is_open"),
    Input("open-offcanvas-button", "n_clicks"),
    State("sidebar-offcanvas", "is_open"),
    prevent_initial_call=True,
)
def toggle_offcanvas(n1, is_open):
    return not is_open if n1 else is_open


@callback(
    Output('data-summary', 'children', allow_duplicate=True),
    Input('cohort-data-store', 'data'),
    prevent_initial_call='initial_duplicate'
)
def update_summary_display_on_cohort_change(cohort_data_dict):
     return update_summary_content(cohort_data_dict)