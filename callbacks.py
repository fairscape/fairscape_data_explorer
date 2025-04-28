# callbacks.py
import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import requests
import json
import plotly.express as px
import io
import os
import zipfile
from urllib.parse import urlparse
import traceback

from s3_utils import get_s3_client, extract_from_zip_in_s3, S3File
from config import FAIRSCAPE_BASE_URL, MINIO_DEFAULT_BUCKET
from layout import create_rule_row
from cohort_utils import parse_uploaded_csv, apply_rules_to_dataframe, join_cohort_data

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
    Output('btn-download-summary', 'disabled'),
    Output('data-summary', 'children'),
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
    Output('group-value-filter-row', 'style', allow_duplicate=True), # Reset filter visibility
    Output('group-value-filter', 'options', allow_duplicate=True),  # Reset filter options
    Output('group-value-filter', 'value', allow_duplicate=True),    # Reset filter values
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

    reset_cohort_ui = [
        "", initial_rule_container_children, "", None, "",
        empty_options, None, True,
        empty_options, None, True,
        empty_options, None, True,
        True,
    ]
    reset_rule_cols = [empty_options] * initial_num_rule_rows, [None] * initial_num_rule_rows, [True] * initial_num_rule_rows
    reset_group_filter = {'display': 'none'}, [], []

    default_outputs = [
            None, None, None, None, status_message, status_color,
            empty_options, None, True,
            empty_options, None, True,
            True,
            ["Load data to see summary."],
        ] + reset_cohort_ui + list(reset_rule_cols) + list(reset_group_filter)

    if not data_ark:
        status_message, status_color = update_status("Please provide a Data ARK identifier.", "warning")
        default_outputs[4:6] = status_message, status_color
        return default_outputs

    schema_ark_found = None
    schema_props = pd.DataFrame()
    schema_props_dict = None
    typed_data = pd.DataFrame()
    all_cols_info = []

    try:
        status_message, status_color = update_status("Fetching Metadata...", "info")
        data_meta_url = f"{FAIRSCAPE_BASE_URL}/{data_ark}"
        response_data = requests.get(data_meta_url, timeout=30)
        response_data.raise_for_status()
        data_metadata = response_data.json()

        schema_ark_found, schema_msg = find_schema_ark(data_metadata)
        if not schema_ark_found and status_color != "danger":
            status_color = "warning"

        status_message = "Locating Data File..."
        full_path = data_metadata.get("distribution", {}).get("location", {}).get("path")
        if not full_path:
            raise ValueError(f"Error: Data location ('distribution/location/path') missing in metadata for {data_ark}.")

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
                        if not csv_files:
                            raise FileNotFoundError(f"No CSV found in ZIP: s3://{bucket}/{key_path}")
                        csv_file_name_in_zip = csv_files[0]
                        with zip_f.open(csv_file_name_in_zip) as file_in_zip:
                           file_content = file_in_zip.read()
            except Exception as e:
                 raise ValueError(f"Error processing ZIP s3://{bucket}/{key_path}: {e}") from e
        else:
            try:
                response = s3.get_object(Bucket=bucket, Key=key_path)
                file_content = response['Body'].read()
            except Exception as e:
                raise ValueError(f"Error downloading file s3://{bucket}/{key_path}: {e}") from e

        if not file_content:
             status_message = "Warning: Data file is empty."
             if status_color != "danger": status_color = "warning"
             raw_data_df = pd.DataFrame()
        else:
            try:
                try:
                    raw_data_df = pd.read_csv(io.BytesIO(file_content))
                except UnicodeDecodeError:
                    try:
                        raw_data_df = pd.read_csv(io.BytesIO(file_content), encoding='latin-1')
                        status_message += " (Used latin-1 encoding)"
                    except Exception as e_enc:
                         raise ValueError(f"Error reading CSV content after encoding fallback: {e_enc}")
            except Exception as e:
                raise ValueError(f"Error reading CSV content: {e}")


        conversion_warnings = []
        if schema_ark_found:
            status_message = "Processing Schema..."
            try:
                schema_url = f"{FAIRSCAPE_BASE_URL}/{schema_ark_found}"
                response_schema = requests.get(schema_url, timeout=30)
                response_schema.raise_for_status()
                schema_list = response_schema.json()
                properties = schema_list.get('properties', {})
                if properties:
                    schema_props_list = [{'name': name, 'type': details.get('type'), 'description': details.get('description')}
                                         for name, details in properties.items() if isinstance(details, dict)]
                    schema_props = pd.DataFrame(schema_props_list)
                    schema_props_dict = schema_props.to_dict('records')
                else:
                    conversion_warnings.append("Schema found but has no 'properties'.")
                    if status_color != "danger": status_color = "warning"

            except Exception as e:
                 conversion_warnings.append(f"Failed to process schema {schema_ark_found}: {e}")
                 status_color = "danger"

        typed_data = raw_data_df.copy()
        if not schema_props.empty:
            status_message = "Applying Types..."
            for _, prop in schema_props.iterrows():
                col_name = prop['name']
                schema_type = prop['type']
                if col_name in typed_data.columns:
                    try:
                        original_series = typed_data[col_name]
                        converted_series = None
                        current_dtype = str(original_series.dtype)
                        if (schema_type == 'integer' and 'int' in current_dtype) or \
                           (schema_type == 'number' and ('float' in current_dtype or 'int' in current_dtype)) or \
                           (schema_type == 'boolean' and 'bool' in current_dtype) or \
                           (schema_type == 'string' and 'object' in current_dtype):
                             converted_series = original_series

                        if converted_series is None:
                            if schema_type == 'integer':
                                num_col = pd.to_numeric(original_series, errors='coerce')
                                if num_col.notna().any() and num_col.dropna().mod(1).eq(0).all():
                                    converted_series = num_col.astype('Int64')
                                elif num_col.notna().any():
                                    converted_series = num_col.astype('Float64')
                                    conversion_warnings.append(f"'{col_name}' (integer schema) has decimals.")
                                elif not num_col.notna().any():
                                    converted_series = num_col.astype('Int64')

                            elif schema_type == 'number':
                                converted_series = pd.to_numeric(original_series, errors='coerce').astype('Float64')
                            elif schema_type == 'boolean':
                                 bool_map = {'true': True, 't': True, '1': True, 'yes': True, 'y': True,
                                             'false': False, 'f': False, '0': False, 'no': False, 'n': False}
                                 converted_series = original_series.astype(str).str.lower().map(bool_map).astype('boolean')
                            elif schema_type == 'string':
                                converted_series = original_series.astype(str).replace({'nan': pd.NA, 'None': pd.NA, '': pd.NA, 'NA': pd.NA, '<NA>': pd.NA})


                        if converted_series is not None:
                             original_na = original_series.isna() | (original_series == '')
                             typed_data[col_name] = converted_series.where(~original_na, pd.NA)
                    except Exception as type_e:
                         conversion_warnings.append(f"Type error converting '{col_name}' to '{schema_type}': {type_e}")
                else:
                    conversion_warnings.append(f"Schema col '{col_name}' not in data.")

        if conversion_warnings:
            base_status = status_message.split("...")[0] + "..." if "..." in status_message else status_message
            if status_color != "danger": status_color = "warning"
            status_message = f"{base_status} Completed with Warnings:\n" + "\n".join(conversion_warnings)

        elif status_color != "danger":
             status_message = "Processing Complete."
             status_color = "success"

        numeric_cols = []
        grouping_cols = []
        all_cols_for_ui = []
        all_cols_info = []
        for col in typed_data.columns:
            col_dtype = typed_data[col].dtype
            all_cols_info.append({"name": col, "type": str(col_dtype)})
            all_cols_for_ui.append({'label': col, 'value': col})
            is_numeric = pd.api.types.is_numeric_dtype(col_dtype) and not pd.api.types.is_bool_dtype(col_dtype)
            is_categorical_like = False
            if pd.api.types.is_string_dtype(col_dtype) or pd.api.types.is_object_dtype(col_dtype) or \
               pd.api.types.is_bool_dtype(col_dtype) or pd.api.types.is_categorical_dtype(col_dtype):
                 is_categorical_like = True
            elif pd.api.types.is_integer_dtype(col_dtype):
                 if typed_data[col].nunique(dropna=True) <= 50: # Increased threshold slightly
                     is_categorical_like = True

            if is_numeric and typed_data[col].notna().any(): numeric_cols.append(col)
            if is_categorical_like and typed_data[col].notna().any(): grouping_cols.append(col)

        numeric_options = [{'label': col, 'value': col} for col in numeric_cols]
        initial_numeric_value = numeric_cols[0] if numeric_cols else None

        summary_children = update_summary_content(typed_data.to_dict('records'))

        available_columns_data = {
            'numeric': numeric_cols,
            'grouping': grouping_cols,
            'all': all_cols_for_ui
        }

        processed_data_dict = typed_data.to_dict('records')
        cohort_data_dict = processed_data_dict

        rule_col_opts = [all_cols_for_ui] * initial_num_rule_rows
        rule_col_vals = [None] * initial_num_rule_rows
        rule_col_disabled = [False] * initial_num_rule_rows

        final_outputs_base = [
            processed_data_dict,
            cohort_data_dict,
            schema_props_dict,
            available_columns_data,
            status_message,
            status_color,
            numeric_options,
            initial_numeric_value,
            not bool(numeric_options),
            empty_options,
            None,
            True,
            False,
            summary_children,
        ]

        final_outputs = final_outputs_base + reset_cohort_ui + list(reset_rule_cols) + list(reset_group_filter)

        main_join_options_index = 19
        main_join_disabled_index = 21
        final_outputs[main_join_options_index] = all_cols_for_ui
        final_outputs[main_join_disabled_index] = False

        rule_col_opts_index = 29
        rule_col_vals_index = 30
        rule_col_disabled_index = 31
        final_outputs[rule_col_opts_index] = rule_col_opts
        final_outputs[rule_col_vals_index] = rule_col_vals
        final_outputs[rule_col_disabled_index] = rule_col_disabled

        return tuple(final_outputs)


    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Error processing ARK {data_ark}:\n{tb_str}")
        status_message = f"Error: {e}"
        status_color = "danger"
        default_outputs[4:6] = status_message, status_color
        default_outputs[13] = [html.P(f"Failed to load data: {e}", className="text-danger")]
        default_outputs[8] = True
        default_outputs[11] = True
        default_outputs[12] = True
        return default_outputs


def update_summary_content(data_dict):
    if data_dict is None:
        return ["Load data to see summary."]
    if not data_dict:
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
    Output('data-summary', 'children', allow_duplicate=True),
    Input('cohort-data-store', 'data'),
    prevent_initial_call=True
)
def update_summary_display(cohort_data_dict):
     return update_summary_content(cohort_data_dict)


@callback(
    Output('group-column-selector', 'options', allow_duplicate=True),
    Output('group-column-selector', 'value', allow_duplicate=True),
    Output('group-column-selector', 'disabled', allow_duplicate=True),
    Input('available-columns-store', 'data'),
    Input('cohort-data-store', 'data'),
    State('group-column-selector', 'value'),
    prevent_initial_call=True
)
def update_grouping_dropdown(available_cols_data, cohort_data_dict, current_group_val):
    if not available_cols_data:
        return [], None, True

    original_grouping_cols = available_cols_data.get('grouping', [])
    all_original_cols = [c['value'] for c in available_cols_data.get('all', [])]
    new_cohort_cols = []

    if cohort_data_dict:
         try:
             df_cohort = pd.DataFrame(cohort_data_dict)
             if not df_cohort.empty:
                 new_cohort_cols = [
                     col for col in df_cohort.columns
                     if col not in all_original_cols and df_cohort[col].nunique(dropna=True) <= 50 and df_cohort[col].notna().any()
                 ]
         except Exception as e:
              print(f"Error processing cohort data for grouping dropdown: {e}")
              pass


    combined_grouping_cols = sorted(list(set(original_grouping_cols + new_cohort_cols)))
    options = [{'label': col, 'value': col} for col in combined_grouping_cols]

    new_value = current_group_val if current_group_val in combined_grouping_cols else None
    is_disabled = not bool(options)

    return options, new_value, is_disabled

@callback(
    Output('group-value-filter-row', 'style', allow_duplicate=True),
    Output('group-value-filter', 'options', allow_duplicate=True),
    Output('group-value-filter', 'value', allow_duplicate=True),
    Input('group-column-selector', 'value'),
    State('cohort-data-store', 'data'),
    prevent_initial_call=True
)
def populate_group_filter_checklist(group_col, cohort_data_dict):
    if not group_col or not cohort_data_dict:
        return {'display': 'none'}, [], []

    try:
        df = pd.DataFrame(cohort_data_dict)
        if group_col not in df.columns:
            return {'display': 'none'}, [], []

        unique_values = sorted(df[group_col].dropna().unique())
        # Convert all to string for checklist consistency
        options = [{'label': str(val), 'value': str(val)} for val in unique_values]
        # Default to all values selected
        values = [str(val) for val in unique_values]

        return {'display': 'block'}, options, values

    except Exception as e:
        print(f"Error populating group filter checklist: {e}")
        return {'display': 'none'}, [], []


@callback(
    Output('data-histogram', 'figure'),
    Input('cohort-data-store', 'data'),
    Input('numeric-column-selector', 'value'),
    Input('group-column-selector', 'value'),
    Input('group-value-filter', 'value'), # Listen to the checklist values
    prevent_initial_call=True
)
def update_histogram(data_dict, selected_col, group_col, selected_group_values):
    if not data_dict or not selected_col:
        return {'layout': {'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                           'annotations': [{'text': 'Select a numeric column to plot',
                                           'xref': 'paper', 'yref': 'paper',
                                           'showarrow': False, 'font': {'size': 14}}]}}
    try:
        df = pd.DataFrame(data_dict)
        if df.empty or selected_col not in df.columns:
            return {'layout': {'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                           'annotations': [{'text': 'Selected column not found or data is empty',
                                           'xref': 'paper', 'yref': 'paper',
                                           'showarrow': False, 'font': {'size': 14}}]}}
        if not pd.api.types.is_numeric_dtype(df[selected_col]):
             return {'layout': {'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                           'annotations': [{'text': f"Column '{selected_col}' is not numeric.",
                                           'xref': 'paper', 'yref': 'paper',
                                           'showarrow': False, 'font': {'size': 14}}]}}

        valid_group_col = None
        group_col_warning = ""
        if group_col and group_col in df.columns:
             if df[group_col].notna().any():
                 unique_count = df[group_col].nunique(dropna=True)
                 if unique_count <= 50:
                     valid_group_col = group_col
                 else:
                      group_col_warning = f"'{group_col}' has >50 unique values, grouping disabled."
             else:
                 group_col_warning = f"'{group_col}' contains only missing values."


        plot_df = df.copy()

        # Filter data based on selected group values BEFORE plotting
        if valid_group_col and selected_group_values is not None:
            # Convert column to string for filtering, matching checklist values
            plot_df = plot_df[plot_df[valid_group_col].astype(str).isin(selected_group_values)]
        elif not valid_group_col and group_col:
             # If grouping was attempted but invalid (e.g., too many uniques), don't group
             valid_group_col = None # Ensure it's None for the plot call

        # Handle NA and boolean conversion for the *filtered* data
        if valid_group_col and valid_group_col in plot_df.columns:
            if pd.api.types.is_bool_dtype(plot_df[valid_group_col]):
                 plot_df[valid_group_col] = plot_df[valid_group_col].astype(str)

            plot_df[valid_group_col] = plot_df[valid_group_col].fillna('N/A').astype(str)


        fig = px.histogram(
            plot_df, x=selected_col, color=valid_group_col, marginal="rug",
            histnorm='probability density', opacity=0.7,
            title=f"Distribution of {selected_col}" + (f" by {valid_group_col}" if valid_group_col else ""),
            template="plotly_white", color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            barmode='overlay', legend_title_text=valid_group_col if valid_group_col else '',
            xaxis_title=selected_col, yaxis_title='Density', hovermode="x unified",
            annotations=[{
                'text': group_col_warning, 'align': 'left', 'showarrow': False,
                'xref': 'paper', 'yref': 'paper', 'x': 0.05, 'y': 0.95,
                'bgcolor': 'rgba(255,255,255,0.7)', 'bordercolor': 'rgba(0,0,0,0.5)', 'borderwidth': 1
            }] if group_col_warning else []
        )
        fig.update_traces(marker_line_width=0.5, marker_line_color="black", opacity=0.75 if valid_group_col else 0.8)
        return fig
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Error generating histogram:\n{tb_str}")
        return {'layout': {'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                           'annotations': [{'text': f'Error generating histogram: {e}',
                                           'xref': 'paper', 'yref': 'paper',
                                           'showarrow': False, 'font': {'size': 14}}]}}


@callback(
    Output("download-summary", "data"),
    Input("btn-download-summary", "n_clicks"),
    State("cohort-data-store", "data"),
    State("data-ark-input", "value"),
    State("available-columns-store", "data"),
    prevent_initial_call=True,
)
def download_summary_text(n_clicks, data_dict, data_ark, available_cols_data):
    if not data_dict:
        return None

    try:
        df = pd.DataFrame(data_dict)
        if df.empty:
            summary_text = "Data loaded, but contains 0 rows."
        else:
            buffer = io.StringIO()
            buffer.write(f"Summary for ARK: {data_ark}\n")

            cohort_cols = []
            if available_cols_data and 'all' in available_cols_data:
                 original_col_names = [c['value'] for c in available_cols_data['all']]
                 cohort_cols = [col for col in df.columns if col not in original_col_names]
                 if cohort_cols:
                     buffer.write(f"Applied Cohort Columns: {', '.join(cohort_cols)}\n")

            buffer.write(f"Shape: {df.shape}\n\n")
            buffer.write("Info:\n")
            df.info(buf=buffer)
            buffer.write("\n\nStatistics:\n")
            try:
                desc = df.describe(include='all', datetime_is_numeric=True).to_string()
            except TypeError:
                desc = df.describe(include='all').to_string()
            buffer.write(desc)
            summary_text = buffer.getvalue()

        safe_ark = "".join(c if c.isalnum() else "_" for c in data_ark)[0:50] if data_ark else "data"
        filename = f"{safe_ark}_summary_cohorts_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt"
        return dict(content=summary_text, filename=filename)
    except Exception as e:
        print(f"Error generating summary for download: {e}")
        error_text = f"Error generating summary: {e}"
        filename = f"summary_error_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt"
        return dict(content=error_text, filename=filename)


@callback(
    Output('rule-builder-container', 'children', allow_duplicate=True),
    Input('add-rule-button', 'n_clicks'),
    State('rule-builder-container', 'children'),
    State('available-columns-store', 'data'),
    prevent_initial_call=True
)
def add_rule_row(n_clicks, existing_rows, available_cols_data):
    if not available_cols_data or not available_cols_data.get('all'):
        return no_update

    new_index = len(existing_rows) if existing_rows else 0
    new_row = create_rule_row(new_index)

    try:
        column_dropdown = None
        for col_component in new_row.children:
            if hasattr(col_component, 'children') and isinstance(col_component.children, dcc.Dropdown):
                if isinstance(col_component.children.id, dict) and \
                   col_component.children.id.get('type') == 'rule-column':
                    column_dropdown = col_component.children
                    break

        if column_dropdown:
            column_dropdown.options = available_cols_data.get('all', [])
            column_dropdown.disabled = False
        else:
            print("Warning: Could not find rule-column dropdown in new rule row.")

    except Exception as e:
        print(f"Error setting options for new rule row: {e}")


    if not existing_rows:
        existing_rows = []
    existing_rows.append(new_row)
    return existing_rows

@callback(
    Output({'type': 'rule-value2', 'index': MATCH}, 'style'),
    Input({'type': 'rule-operator', 'index': MATCH}, 'value')
)
def toggle_value2_input(operator):
    if operator == 'between':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@callback(
    Output('cohort-data-store', 'data', allow_duplicate=True),
    Output('rule-apply-status', 'children', allow_duplicate=True),
    Input('apply-rules-button', 'n_clicks'),
    State('cohort-data-store', 'data'),
    State({'type': 'rule-column', 'index': ALL}, 'value'),
    State({'type': 'rule-operator', 'index': ALL}, 'value'),
    State({'type': 'rule-value1', 'index': ALL}, 'value'),
    State({'type': 'rule-value2', 'index': ALL}, 'value'),
    State('cohort-name-input', 'value'),
    prevent_initial_call=True
)
def apply_rule_cohort(n_clicks, cohort_data_dict,
                      rule_cols, rule_ops, rule_vals1, rule_vals2, cohort_name_input):

    if not cohort_data_dict:
        return no_update, dbc.Alert("Load data before applying cohorts.", color="warning", dismissable=True)
    if not cohort_name_input or not cohort_name_input.strip():
        return no_update, dbc.Alert("Please enter a name for the cohort.", color="warning", dismissable=True)

    try:
        df = pd.DataFrame(cohort_data_dict)
    except Exception as e:
        return no_update, dbc.Alert(f"Error loading current data: {e}", color="danger", dismissable=True)

    cohort_name = cohort_name_input.strip().replace(' ', '_')
    if not all(c.isalnum() or c == '_' for c in cohort_name):
         return no_update, dbc.Alert("Cohort name can only contain letters, numbers, and underscores.", color="warning", dismissable=True)
    cohort_name = f"cohort_{cohort_name}"

    if cohort_name in df.columns:
         return no_update, dbc.Alert(f"Cohort '{cohort_name}' already exists. Choose a different name or modify the existing data if intended.", color="warning", dismissable=True)

    rules = []
    valid_rules = True
    rule_error = ""
    for i, col in enumerate(rule_cols):
        if col is not None:
            op = rule_ops[i]
            val1 = rule_vals1[i]
            val2 = rule_vals2[i]

            if op is None or val1 is None or str(val1).strip() == '':
                valid_rules = False
                rule_error = f"Rule {i+1} is incomplete (missing operator or value)."
                break
            if op == 'between' and (val2 is None or str(val2).strip() == ''):
                 valid_rules = False
                 rule_error = f"Rule {i+1} ('between') requires a second value."
                 break

            rules.append({
                'column': col,
                'op': op,
                'value1': val1,
                'value2': val2 if op == 'between' else None
            })


    if not valid_rules:
        return no_update, dbc.Alert(f"Error in rule definition: {rule_error}", color="danger", dismissable=True)
    if not rules:
         return no_update, dbc.Alert("No valid rule conditions defined. Select columns, operators, and values.", color="warning", dismissable=True)


    try:
        cohort_series = apply_rules_to_dataframe(df.copy(), rules, cohort_name)
        df[cohort_name] = cohort_series
        status = dbc.Alert(f"Cohort '{cohort_name}' ({cohort_series.sum()} members) applied successfully.", color="success", dismissable=True, duration=4000)
        return df.to_dict('records'), status
    except ValueError as e:
        status = dbc.Alert(f"Error applying rules: {e}", color="danger", dismissable=True)
        return no_update, status
    except Exception as e:
        status = dbc.Alert(f"An unexpected error occurred: {e}", color="danger", dismissable=True)
        print(f"Error applying rules: {traceback.format_exc()}")
        return no_update, status


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
    if contents is None:
        return None, [], None, True, [], None, True, True, ""

    try:
        cohort_df = parse_uploaded_csv(contents)
        if cohort_df.empty:
            raise ValueError("Uploaded CSV file is empty.")

        cohort_cols = [{'label': col, 'value': col} for col in cohort_df.columns]
        status = dbc.Alert(f"Parsed '{filename}' ({len(cohort_df)} rows, {len(cohort_df.columns)} columns). Select join and cohort columns.", color="info", dismissable=True)
        return cohort_df.to_dict('records'), cohort_cols, None, False, cohort_cols, None, False, False, status
    except ValueError as e:
        status = dbc.Alert(f"Error processing '{filename}': {e}", color="danger", dismissable=True)
        return None, [], None, True, [], None, True, True, status
    except Exception as e:
        status = dbc.Alert(f"Unexpected error reading '{filename}': {e}", color="danger", dismissable=True)
        print(f"Upload error: {traceback.format_exc()}")
        return None, [], None, True, [], None, True, True, status


@callback(
    Output('cohort-data-store', 'data', allow_duplicate=True),
    Output('upload-status', 'children', allow_duplicate=True),
    Input('process-upload-button', 'n_clicks'),
    State('cohort-data-store', 'data'),
    State('uploaded-cohort-store', 'data'),
    State('main-join-column-selector', 'value'),
    State('upload-join-column-selector', 'value'),
    State('upload-cohort-column-selector', 'value'),
    prevent_initial_call=True
)
def process_uploaded_cohorts(n_clicks, cohort_data_dict, uploaded_cohort_dict,
                             main_join_col, upload_join_col, upload_cohort_col):

    if not cohort_data_dict:
         return no_update, dbc.Alert("Load main data first.", color="warning", dismissable=True)
    if not uploaded_cohort_dict:
        return no_update, dbc.Alert("Upload a cohort file first.", color="warning", dismissable=True)
    if not all([main_join_col, upload_join_col, upload_cohort_col]):
        return no_update, dbc.Alert("Select main join, upload join, and cohort label columns.", color="warning", dismissable=True)

    try:
        main_df = pd.DataFrame(cohort_data_dict)
        cohort_df = pd.DataFrame(uploaded_cohort_dict)

        if main_df.empty:
             return no_update, dbc.Alert("Main data is empty, cannot join.", color="warning", dismissable=True)
        if cohort_df.empty:
              return no_update, dbc.Alert("Uploaded cohort data is empty, cannot join.", color="warning", dismissable=True)


        merged_df = join_cohort_data(main_df, cohort_df, main_join_col, upload_join_col, upload_cohort_col)

        expected_new_col_name = f"cohort_{upload_cohort_col}"
        actual_new_col_name = None

        original_main_cols = set(main_df.columns)
        merged_cols = set(merged_df.columns)
        added_cols = list(merged_cols - original_main_cols)

        if len(added_cols) == 1:
            actual_new_col_name = added_cols[0]
            if actual_new_col_name != expected_new_col_name:
                 print(f"Warning: Joined column named '{actual_new_col_name}', expected '{expected_new_col_name}'.")
        elif expected_new_col_name in merged_cols and expected_new_col_name not in original_main_cols:
             actual_new_col_name = expected_new_col_name
        else:
            raise ValueError(f"Could not reliably identify the added cohort column after merge. Added columns: {added_cols}")

        assigned_count = merged_df[actual_new_col_name].notna().sum()
        if 'N/A' in merged_df[actual_new_col_name].unique():
            assigned_count = (merged_df[actual_new_col_name] != 'N/A').sum()


        status = dbc.Alert(f"Joined cohort data using column '{actual_new_col_name}'. {assigned_count} rows received a cohort assignment.", color="success", dismissable=True, duration=5000)
        return merged_df.to_dict('records'), status

    except ValueError as e:
        status = dbc.Alert(f"Error joining cohort data: {e}", color="danger", dismissable=True)
        return no_update, status
    except Exception as e:
        status = dbc.Alert(f"Unexpected error during join: {e}", color="danger", dismissable=True)
        print(f"Cohort join error: {traceback.format_exc()}")
        return no_update, status