# callbacks_main.py
import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import io
import os
import zipfile
from urllib.parse import urlparse
import traceback

from s3_utils import get_s3_client, extract_from_zip_in_s3, S3File
from config import FAIRSCAPE_BASE_URL, MINIO_DEFAULT_BUCKET
from layout import create_rule_row
from cohort_utils import parse_uploaded_csv, apply_rules_to_dataframe, join_cohort_data
# Import model utils only if functions called directly here (not currently)

s3 = get_s3_client()

def update_status(message, color):
    return message, color

def find_schema_ark(metadata_dict):
    if not isinstance(metadata_dict, dict):
        return None, "Metadata is not a dictionary."
    potential_keys = ['evi:schema', 'schema']
    for key, value in metadata_dict.items():
        key_norm = key.lower().replace('evi:', '')
        if key_norm == 'schema':
            if isinstance(value, str) and value.startswith('ark:'):
                return value, f"Schema ARK Found ({key})"
            elif isinstance(value, dict) and '@id' in value and value['@id'].startswith('ark:'):
                return value['@id'], f"Schema ARK Found ({key} @id)"
            else:
                 return None, f"Key '{key}' found, but not a valid Schema ARK."
    return None, "No Schema ARK found in metadata."

def update_summary_content(data_dict):
    if data_dict is None:
        return ["Load data to see summary."]
    if not isinstance(data_dict, list) or not data_dict:
         return [html.P("Data loaded, but contains 0 rows or is empty.")]
    try:
        df = pd.DataFrame(data_dict)
        if df.empty:
             return [html.P("Data loaded, but contains 0 rows.")]
        basic_info = html.Div([
            html.P(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns"),
        ], className="mb-3")
        try:
             summary_df = df.describe(include='all', datetime_is_numeric=True).reset_index()
        except TypeError:
             summary_df = df.describe(include='all').reset_index()
        except Exception as desc_e:
            return [basic_info, html.P(f"Error during data description: {desc_e}", className="text-warning")]
        cols = ['index'] + [col for col in summary_df if col != 'index']
        summary_df = summary_df[cols]
        summary_df = summary_df.round(3)
        summary_df.fillna('', inplace=True)
        summary_table = dbc.Table.from_dataframe(
            summary_df, striped=True, bordered=True, hover=True,
            responsive=True, className="small"
        )
        return [basic_info, summary_table]
    except Exception as e:
        return [html.P(f"Error generating summary: {e}", className="text-danger")]


@callback(
    Output("sidebar-offcanvas", "is_open"),
    Input("open-offcanvas-button", "n_clicks"),
    State("sidebar-offcanvas", "is_open"),
    prevent_initial_call=True,
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

@callback(
    Output('processed-data-store', 'data'),
    Output('cohort-data-store', 'data', allow_duplicate=True),
    Output('schema-properties-store', 'data'),
    Output('available-columns-store', 'data'),
    Output('status-alert', 'children'),
    Output('status-alert', 'color'),
    Output('numeric-column-selector', 'options'),
    Output('numeric-column-selector', 'value'),
    Output('numeric-column-selector', 'disabled'),
    Output('group-column-selector', 'options', allow_duplicate=True),
    Output('group-column-selector', 'value', allow_duplicate=True),
    Output('group-column-selector', 'disabled', allow_duplicate=True),
    Output('group-value-filter-row', 'style', allow_duplicate=True),
    Output('group-value-filter', 'options', allow_duplicate=True),
    Output('group-value-filter', 'value', allow_duplicate=True),
    Output('data-summary', 'children', allow_duplicate=True),
    Output('data-histogram', 'figure', allow_duplicate=True),
    Output('btn-download-summary', 'disabled'),
    Output('cohort-name-input', 'value'),
    Output('rule-builder-container', 'children', allow_duplicate=True),
    Output('rule-apply-status', 'children', allow_duplicate=True),
    Output('upload-cohort-csv', 'contents'),
    Output('upload-status', 'children', allow_duplicate=True),
    Output('main-join-column-selector', 'options'),
    Output('main-join-column-selector', 'value'),
    Output('main-join-column-selector', 'disabled'),
    Output('upload-join-column-selector', 'options', allow_duplicate=True),
    Output('upload-join-column-selector', 'value', allow_duplicate=True),
    Output('upload-join-column-selector', 'disabled', allow_duplicate=True),
    Output('upload-cohort-column-selector', 'options', allow_duplicate=True),
    Output('upload-cohort-column-selector', 'value', allow_duplicate=True),
    Output('upload-cohort-column-selector', 'disabled', allow_duplicate=True),
    Output('process-upload-button', 'disabled', allow_duplicate=True),
    Output({'type': 'rule-column', 'index': ALL}, 'options', allow_duplicate=True),
    Output({'type': 'rule-column', 'index': ALL}, 'value', allow_duplicate=True),
    Output({'type': 'rule-column', 'index': ALL}, 'disabled', allow_duplicate=True),
    Output('model-y-selector', 'options'),
    Output('model-y-selector', 'value'),
    Output('model-y-selector', 'disabled'),
    Output('model-x-selector', 'options'),
    Output('model-x-selector', 'value'),
    Output('model-x-selector', 'disabled'),
    Output('build-model-button', 'disabled'),
    Output('model-summary-output', 'children'),
    Output('model-plot-output', 'children'),
    Output('model-status', 'children'),
    Input('load-button', 'n_clicks'),
    State('data-ark-input', 'value'),
    prevent_initial_call=True
)
def load_and_process_ark(n_clicks, data_ark):
    initial_status = "Enter a Data ARK and click Load."
    status_message = "Initializing..."
    status_color = "secondary"
    empty_options = []
    initial_num_rule_rows = 1
    initial_rule_container_children = [create_rule_row(0)]
    reset_plot_children = html.Div("Plot will appear here after model build.", style={'textAlign': 'center', 'padding': '20px', 'color': 'grey'})

    reset_exploration = [
        empty_options, None, True, empty_options, None, True,
        {'display': 'none'}, [], [],
        ["Load data to see summary."], {}, True
    ]
    reset_cohort_ui = [
        "", initial_rule_container_children, "", None, "",
        empty_options, None, True, empty_options, None, True,
        empty_options, None, True, True,
    ]
    reset_rule_cols = [empty_options] * initial_num_rule_rows, [None] * initial_num_rule_rows, [True] * initial_num_rule_rows
    reset_model_ui = [
        empty_options, None, True, empty_options, [], True, True,
        ["Load data and select model options."], reset_plot_children, ""
    ]

    default_outputs = [
        None, None, None, None, status_message, status_color
    ] + reset_exploration + reset_cohort_ui + list(reset_rule_cols) + reset_model_ui

    if not data_ark:
        status_message, status_color = update_status("Please provide a Data ARK identifier.", "warning")
        default_outputs[4:6] = status_message, status_color
        model_plot_output_index = 45
        default_outputs[model_plot_output_index] = reset_plot_children
        return tuple(default_outputs)

    schema_ark_found = None
    schema_props = pd.DataFrame()
    schema_props_dict = None
    typed_data = pd.DataFrame()
    all_cols_info = []
    numeric_cols = []
    grouping_cols = []
    all_cols_for_ui = []

    try:
        status_message, status_color = update_status("Fetching Metadata...", "info")
        data_meta_url = f"{FAIRSCAPE_BASE_URL}/{data_ark}"
        response_data = requests.get(data_meta_url, timeout=30)
        response_data.raise_for_status()
        data_metadata = response_data.json()
        schema_ark_found, schema_msg = find_schema_ark(data_metadata)
        if not schema_ark_found and status_color != "danger":
            status_color = "warning"; status_message = f"{status_message} {schema_msg}"

        status_message = "Locating Data File..."
        full_path = data_metadata.get("distribution", {}).get("location", {}).get("path")
        if not full_path: raise ValueError(f"Data location missing in metadata for {data_ark}.")
        bucket = MINIO_DEFAULT_BUCKET; key_path = full_path
        csv_file_name_in_zip = None; is_zip_file = False
        if ".zip/" in key_path:
            zip_path_parts = key_path.split(".zip/", 1); key_path = zip_path_parts[0] + ".zip"
            csv_file_name_in_zip = zip_path_parts[1]; is_zip_file = True
        elif key_path.lower().endswith('.zip'): is_zip_file = True

        status_message = "Processing Data File..."
        file_content = None
        if is_zip_file:
            try:
                if csv_file_name_in_zip:
                    file_content = extract_from_zip_in_s3(s3, bucket, key_path, csv_file_name_in_zip)
                else:
                    s3_file = S3File(s3, bucket, key_path)
                    with zipfile.ZipFile(s3_file, 'r') as zip_f:
                        csv_files = [f for f in zip_f.namelist() if f.lower().endswith('.csv') and not f.startswith('__MACOSX/')]
                        if not csv_files: raise FileNotFoundError(f"No CSV found in ZIP: s3://{bucket}/{key_path}")
                        csv_file_name_in_zip = csv_files[0]
                        with zip_f.open(csv_file_name_in_zip) as file_in_zip: file_content = file_in_zip.read()
            except Exception as e: raise ValueError(f"Error processing ZIP s3://{bucket}/{key_path}: {e}") from e
        else:
            try:
                response = s3.get_object(Bucket=bucket, Key=key_path)
                file_content = response['Body'].read()
            except Exception as e: raise ValueError(f"Error downloading file s3://{bucket}/{key_path}: {e}") from e

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
            status_message = "Processing Schema..."
            try:
                schema_url = f"{FAIRSCAPE_BASE_URL}/{schema_ark_found}"
                response_schema = requests.get(schema_url, timeout=30); response_schema.raise_for_status()
                schema_list = response_schema.json(); properties = schema_list.get('properties', {})
                if properties:
                    schema_props_list = [{'name': name, 'type': details.get('type'), 'description': details.get('description')}
                                         for name, details in properties.items() if isinstance(details, dict)]
                    schema_props = pd.DataFrame(schema_props_list); schema_props_dict = schema_props.to_dict('records')
                else: conversion_warnings.append("Schema has no 'properties'."); status_color = "warning"
            except Exception as e: conversion_warnings.append(f"Failed to process schema {schema_ark_found}: {e}"); status_color = "danger"

        typed_data = raw_data_df.copy()
        if not schema_props.empty:
            status_message = "Applying Types..."
            for _, prop in schema_props.iterrows():
                col_name = prop['name']; schema_type = prop['type']
                if col_name in typed_data.columns:
                    try:
                        original_series = typed_data[col_name]; converted_series = None; current_dtype = str(original_series.dtype)
                        if (schema_type == 'integer' and 'int' in current_dtype) or \
                           (schema_type == 'number' and ('float' in current_dtype or 'int' in current_dtype)) or \
                           (schema_type == 'boolean' and 'bool' in current_dtype) or \
                           (schema_type == 'string' and ('object' in current_dtype or pd.api.types.is_string_dtype(current_dtype))):
                             converted_series = original_series
                        if converted_series is None:
                            if schema_type == 'integer':
                                num_col = pd.to_numeric(original_series, errors='coerce')
                                if num_col.notna().any() and num_col.dropna().mod(1).eq(0).all(): converted_series = num_col.astype('Int64')
                                elif num_col.notna().any(): converted_series = num_col.astype('Float64'); conversion_warnings.append(f"'{col_name}' (int schema) has decimals.")
                                else: converted_series = num_col.astype('Int64')
                            elif schema_type == 'number': converted_series = pd.to_numeric(original_series, errors='coerce').astype('Float64')
                            elif schema_type == 'boolean':
                                 bool_map = {'true': True, 't': True, '1': True, 'yes': True, 'y': True, 'false': False, 'f': False, '0': False, 'no': False, 'n': False}
                                 converted_series = original_series.astype(str).str.lower().map(bool_map).astype('boolean')
                            elif schema_type == 'string': converted_series = original_series.astype(str).replace({'nan': pd.NA, 'None': pd.NA, '': pd.NA, 'NA': pd.NA, '<NA>': pd.NA})
                        if converted_series is not None:
                             original_na_mask = original_series.isna() | (original_series == '') | (original_series.astype(str).str.lower().isin(['na', '<na>', 'none', 'nan']))
                             typed_data[col_name] = converted_series.where(~original_na_mask, pd.NA)
                    except Exception as type_e: conversion_warnings.append(f"Type error converting '{col_name}' to '{schema_type}': {type_e}")
                else: conversion_warnings.append(f"Schema col '{col_name}' not in data.")
        else: typed_data = raw_data_df.copy()

        if conversion_warnings:
            base_status = status_message.split("...")[0] + "..." if "..." in status_message else status_message
            if status_color != "danger": status_color = "warning"
            status_message = f"{base_status} Completed with Warnings:\n" + "\n".join(conversion_warnings)
        elif status_color != "danger": status_message = "Processing Complete. Select options in tabs."; status_color = "success"

        numeric_cols_for_plot = []; grouping_cols_for_plot = []
        numeric_cols_for_model = []; boolean_or_binary_cols = []
        all_cols_for_ui = []
        for col in typed_data.columns:
            col_dtype = typed_data[col].dtype; all_cols_for_ui.append({'label': col, 'value': col})
            is_numeric = pd.api.types.is_numeric_dtype(col_dtype) and not pd.api.types.is_bool_dtype(col_dtype)
            is_bool_or_binary = pd.api.types.is_bool_dtype(col_dtype) or \
                                (pd.api.types.is_integer_dtype(col_dtype) and set(typed_data[col].dropna().unique()) <= {0, 1})
            is_categorical_like = False
            if pd.api.types.is_string_dtype(col_dtype) or pd.api.types.is_object_dtype(col_dtype) or \
               pd.api.types.is_bool_dtype(col_dtype) or pd.api.types.is_categorical_dtype(col_dtype): is_categorical_like = True
            elif pd.api.types.is_integer_dtype(col_dtype):
                 if typed_data[col].nunique(dropna=True) <= 50: is_categorical_like = True
            if is_numeric and typed_data[col].notna().any(): numeric_cols_for_plot.append(col); numeric_cols_for_model.append(col)
            if is_categorical_like and typed_data[col].notna().any(): grouping_cols_for_plot.append(col)
            if is_bool_or_binary and typed_data[col].notna().any(): boolean_or_binary_cols.append(col)

        numeric_plot_options = [{'label': col, 'value': col} for col in numeric_cols_for_plot]
        initial_numeric_plot_value = numeric_cols_for_plot[0] if numeric_cols_for_plot else None
        grouping_options = [{'label': col, 'value': col} for col in grouping_cols_for_plot]
        model_y_options = [{'label': col, 'value': col} for col in sorted(list(set(numeric_cols_for_model + boolean_or_binary_cols)))]
        model_x_options = [{'label': col, 'value': col} for col in numeric_cols_for_model]
        summary_children = update_summary_content(typed_data.to_dict('records'))
        available_columns_data = {
            'numeric_plot': numeric_cols_for_plot, 'grouping_plot': grouping_cols_for_plot,
            'numeric_model': numeric_cols_for_model, 'boolean_binary_model': boolean_or_binary_cols,
            'all': all_cols_for_ui
        }
        processed_data_dict = typed_data.to_dict('records'); cohort_data_dict = processed_data_dict
        rule_col_opts = [all_cols_for_ui] * initial_num_rule_rows; rule_col_vals = [None] * initial_num_rule_rows
        rule_col_disabled = [False] * initial_num_rule_rows if all_cols_for_ui else [True] * initial_num_rule_rows

        final_outputs = [
            processed_data_dict, cohort_data_dict, schema_props_dict, available_columns_data, status_message, status_color,
            numeric_plot_options, initial_numeric_plot_value, not bool(numeric_plot_options),
            grouping_options, None, not bool(grouping_options), {'display': 'none'}, [], [], summary_children, {}, False,
            "", initial_rule_container_children, "", None, "", all_cols_for_ui, None, not bool(all_cols_for_ui),
            empty_options, None, True, empty_options, None, True, True,
            rule_col_opts, rule_col_vals, rule_col_disabled,
            model_y_options, None, not bool(model_y_options), model_x_options, [], not bool(model_x_options),
            not bool(model_y_options and model_x_options), ["Load data and select model options."], reset_plot_children, ""
        ]
        return tuple(final_outputs)

    except Exception as e:
        tb_str = traceback.format_exc(); print(f"Error processing ARK {data_ark}:\n{tb_str}")
        status_message = f"Error: {e}"; status_color = "danger"
        default_outputs[4:6] = status_message, status_color
        default_outputs[15] = [html.P(f"Failed to load data: {e}", className="text-danger")]
        default_outputs[8] = True; default_outputs[11] = True; default_outputs[17] = True; default_outputs[26] = True
        rule_col_disabled_index = 36
        if len(default_outputs) > rule_col_disabled_index and isinstance(default_outputs[rule_col_disabled_index], list) and len(default_outputs[rule_col_disabled_index]) > 0:
             default_outputs[rule_col_disabled_index] = [True] * len(default_outputs[rule_col_disabled_index])
        default_outputs[39] = True; default_outputs[42] = True; default_outputs[43] = True
        model_plot_output_index = 45
        if len(default_outputs) > model_plot_output_index: default_outputs[model_plot_output_index] = reset_plot_children
        return tuple(default_outputs)

@callback(
    Output('data-summary', 'children', allow_duplicate=True),
    Input('cohort-data-store', 'data'),
    prevent_initial_call=True
)
def update_summary_display_on_cohort_change(cohort_data_dict):
     return update_summary_content(cohort_data_dict)

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
        if df.empty: summary_text = "Data loaded, but contains 0 rows."
        else:
            buffer = io.StringIO()
            buffer.write(f"Summary for ARK: {data_ark}\n")
            buffer.write(f"Data Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            cohort_cols = []
            if available_cols_data and 'all' in available_cols_data:
                 original_col_names = [c['value'] for c in available_cols_data['all']]
                 current_cols = df.columns
                 cohort_cols = [col for col in current_cols if col not in original_col_names]
                 if cohort_cols: buffer.write(f"Applied Cohort Columns: {', '.join(cohort_cols)}\n")
            buffer.write(f"\nShape: {df.shape}\n\n")
            buffer.write("Column Info:\n"); df.info(buf=buffer)
            buffer.write("\n\nDescriptive Statistics:\n")
            try: desc = df.describe(include='all', datetime_is_numeric=True).to_string()
            except TypeError: desc = df.describe(include='all').to_string()
            except Exception as desc_e: desc = f"Error generating description: {desc_e}"
            buffer.write(desc); summary_text = buffer.getvalue()
        safe_ark = "".join(c if c.isalnum() else "_" for c in data_ark)[0:50] if data_ark else "data"
        filename = f"{safe_ark}_summary_cohorts_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt"
        return dict(content=summary_text, filename=filename)
    except Exception as e:
        print(f"Error generating summary for download: {e}")
        error_text = f"Error generating summary file:\n{e}\n\n{traceback.format_exc()}"
        filename = f"summary_error_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt"
        return dict(content=error_text, filename=filename)

@callback(
    Output('rule-builder-container', 'children', allow_duplicate=True),
    Input('add-rule-button', 'n_clicks'),
    State('rule-builder-container', 'children'),
    State('available-columns-store', 'data'),
    prevent_initial_call=True
)
def add_rule_row_callback(n_clicks, existing_rows, available_cols_data):
    if not available_cols_data or not available_cols_data.get('all'): return no_update
    all_cols_options = available_cols_data.get('all', [])
    new_index = len(existing_rows) if existing_rows else 0
    new_row = create_rule_row(new_index)
    try:
        col_dropdown = None; op_dropdown = None
        for component in new_row.children:
            if isinstance(component, dbc.Col) and component.children:
                dropdown = component.children
                if isinstance(dropdown, dcc.Dropdown) and isinstance(dropdown.id, dict):
                     if dropdown.id.get('type') == 'rule-column': col_dropdown = dropdown
                     elif dropdown.id.get('type') == 'rule-operator': op_dropdown = dropdown
        if col_dropdown: col_dropdown.options = all_cols_options; col_dropdown.disabled = not bool(all_cols_options)
        else: print(f"Warning: Could not find 'rule-column' dropdown in new rule row {new_index}")
        if op_dropdown: op_dropdown.disabled = False
        else: print(f"Warning: Could not find 'rule-operator' dropdown in new rule row {new_index}")
    except Exception as e: print(f"Error updating dropdowns in new rule row: {e}")
    if not existing_rows: existing_rows = []
    existing_rows.append(new_row)
    return existing_rows

@callback(
    Output({'type': 'rule-value2', 'index': MATCH}, 'style'),
    Input({'type': 'rule-operator', 'index': MATCH}, 'value'),
     prevent_initial_call=True
)
def toggle_value2_input(operator):
    if operator == 'between': return {'display': 'block'}
    else: return {'display': 'none'}

@callback(
    Output('cohort-data-store', 'data', allow_duplicate=True),
    Output('rule-apply-status', 'children', allow_duplicate=True),
    Input('apply-rules-button', 'n_clicks'),
    State('processed-data-store', 'data'),
    State('cohort-data-store', 'data'),
    State({'type': 'rule-column', 'index': ALL}, 'value'),
    State({'type': 'rule-operator', 'index': ALL}, 'value'),
    State({'type': 'rule-value1', 'index': ALL}, 'value'),
    State({'type': 'rule-value2', 'index': ALL}, 'value'),
    State('cohort-name-input', 'value'),
    State('available-columns-store', 'data'),
    prevent_initial_call=True
)
def apply_rule_cohort(n_clicks, processed_data_dict, cohort_data_dict,
                      rule_cols, rule_ops, rule_vals1, rule_vals2, cohort_name_input,
                      available_cols_data):
    alert_error = lambda msg: (no_update, dbc.Alert(msg, color="danger", dismissable=True))
    alert_warning = lambda msg: (no_update, dbc.Alert(msg, color="warning", dismissable=True))
    current_data_dict = cohort_data_dict if cohort_data_dict else processed_data_dict
    if not current_data_dict: return alert_warning("Load data before applying cohorts.")
    if not cohort_name_input or not cohort_name_input.strip(): return alert_warning("Please enter a name for the cohort.")
    try:
        df = pd.DataFrame(current_data_dict)
        if df.empty: return alert_warning("Data is currently empty, cannot apply rules.")
    except Exception as e: return alert_error(f"Error loading current data: {e}")
    cohort_name = cohort_name_input.strip().replace(' ', '_')
    if not all(c.isalnum() or c == '_' for c in cohort_name): return alert_warning("Cohort name must be alphanumeric/underscores.")
    cohort_col_name = f"cohort_{cohort_name}"
    if cohort_col_name in df.columns: return alert_warning(f"Cohort column '{cohort_col_name}' already exists.")
    rules = []; valid_rules_found = False
    for i, col in enumerate(rule_cols):
        if col is not None and str(col).strip() != "":
            op = rule_ops[i]; val1 = rule_vals1[i]; val2 = rule_vals2[i]
            if op is None or val1 is None or str(val1).strip() == '': return alert_error(f"Rule {i+1} (Col: {col}) is incomplete.")
            if op == 'between' and (val2 is None or str(val2).strip() == ''): return alert_error(f"Rule {i+1} (Col: {col}) 'between' missing second value.")
            rules.append({'column': col, 'op': op, 'value1': val1, 'value2': val2 if op == 'between' else None})
            valid_rules_found = True
    if not valid_rules_found: return alert_warning("No valid rule conditions defined.")
    try:
        cohort_series = apply_rules_to_dataframe(df.copy(), rules, cohort_col_name)
        df[cohort_col_name] = cohort_series; num_members = cohort_series.sum()
        status = dbc.Alert(f"Cohort '{cohort_col_name}' ({num_members} members) applied.", color="success", dismissable=True, duration=4000)
        return df.to_dict('records'), status
    except ValueError as e: return alert_error(f"Error applying rules: {e}")
    except Exception as e: print(f"Unexpected error applying rules: {traceback.format_exc()}"); return alert_error(f"Unexpected error: {e}")

@callback(
    Output('uploaded-cohort-store', 'data'),
    Output('upload-join-column-selector', 'options', allow_duplicate=True),
    Output('upload-join-column-selector', 'value', allow_duplicate=True),
    Output('upload-join-column-selector', 'disabled', allow_duplicate=True),
    Output('upload-cohort-column-selector', 'options', allow_duplicate=True),
    Output('upload-cohort-column-selector', 'value', allow_duplicate=True),
    Output('upload-cohort-column-selector', 'disabled', allow_duplicate=True),
    Output('process-upload-button', 'disabled', allow_duplicate=True),
    Output('upload-status', 'children', allow_duplicate=True),
    Input('upload-cohort-csv', 'contents'),
    State('upload-cohort-csv', 'filename'),
    prevent_initial_call=True
)
def handle_cohort_upload(contents, filename):
    if contents is None: return None, [], None, True, [], None, True, True, ""
    try:
        cohort_df = parse_uploaded_csv(contents)
        if cohort_df.empty: raise ValueError("Uploaded CSV is empty or unparseable.")
        cohort_cols = [{'label': col, 'value': col} for col in cohort_df.columns]
        status = dbc.Alert(f"Parsed '{filename}' ({len(cohort_df)}r, {len(cohort_df.columns)}c). Select columns.", color="info", dismissable=True)
        return cohort_df.to_dict('records'), cohort_cols, None, False, cohort_cols, None, False, False, status
    except ValueError as e:
        status = dbc.Alert(f"Error processing '{filename}': {e}", color="danger", dismissable=True)
        return None, [], None, True, [], None, True, True, status
    except Exception as e:
        status = dbc.Alert(f"Unexpected error reading '{filename}': {e}", color="danger", dismissable=True)
        print(f"Upload error: {traceback.format_exc()}"); return None, [], None, True, [], None, True, True, status

@callback(
    Output('cohort-data-store', 'data', allow_duplicate=True),
    Output('upload-status', 'children', allow_duplicate=True),
    Input('process-upload-button', 'n_clicks'),
    State('cohort-data-store', 'data'),
    State('uploaded-cohort-store', 'data'),
    State('main-join-column-selector', 'value'),
    State('upload-join-column-selector', 'value'),
    State('upload-cohort-column-selector', 'value'),
    State('available-columns-store', 'data'),
    prevent_initial_call=True
)
def process_uploaded_cohorts(n_clicks, cohort_data_dict, uploaded_cohort_dict,
                             main_join_col, upload_join_col, upload_cohort_col,
                             available_cols_data):
    alert_error = lambda msg: (no_update, dbc.Alert(msg, color="danger", dismissable=True))
    alert_warning = lambda msg: (no_update, dbc.Alert(msg, color="warning", dismissable=True))
    if not cohort_data_dict: return alert_warning("Load main data first.")
    if not uploaded_cohort_dict: return alert_warning("Upload cohort file first.")
    if not all([main_join_col, upload_join_col, upload_cohort_col]): return alert_warning("Select all three join/label columns.")
    try:
        main_df = pd.DataFrame(cohort_data_dict); cohort_df = pd.DataFrame(uploaded_cohort_dict)
        if main_df.empty: return alert_warning("Main data is empty.")
        if cohort_df.empty: return alert_warning("Uploaded cohort data is empty.")
        merged_df = join_cohort_data(main_df.copy(), cohort_df, main_join_col, upload_join_col, upload_cohort_col)
        original_main_cols = set(main_df.columns); merged_cols = set(merged_df.columns)
        added_cols = list(merged_cols - original_main_cols)
        if len(added_cols) == 1: actual_new_col_name = added_cols[0]
        elif f"cohort_{upload_cohort_col}" in added_cols: actual_new_col_name = f"cohort_{upload_cohort_col}"
        else: raise ValueError(f"Could not ID added cohort column. Expected ~'cohort_{upload_cohort_col}'. Added: {added_cols}")
        assigned_count = (merged_df[actual_new_col_name] != 'N/A').sum()
        status_msg = f"Joined '{actual_new_col_name}'. {assigned_count} rows assigned."; status = dbc.Alert(status_msg, color="success", dismissable=True, duration=5000)
        return merged_df.to_dict('records'), status
    except ValueError as e: return alert_error(f"Error joining cohort data: {e}")
    except Exception as e: print(f"Unexpected error during cohort join: {traceback.format_exc()}"); return alert_error(f"Unexpected error during join: {e}")