import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import traceback

from layout import create_rule_row
from utils.cohort_utils import apply_rules_to_dataframe

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
    if not all_cols_options: return no_update
    new_index = len(existing_rows) if existing_rows else 0
    new_row = create_rule_row(new_index)
    try:
        q = list(new_row.children) if hasattr(new_row, 'children') else []
        found = False
        while q:
            comp = q.pop(0)
            if isinstance(comp, dcc.Dropdown) and isinstance(comp.id, dict) and comp.id.get('type') == 'rule-column':
                 comp.options = all_cols_options; comp.value = None; comp.disabled = False; found = True; break
            if hasattr(comp, 'children'): q.extend(comp.children if isinstance(comp.children, list) else [comp.children])
        if not found: print(f"Warn: Could not find 'rule-column' in new rule row {new_index}")
    except Exception as e: print(f"Error setting options rule row {new_index}: {e}")
    if not existing_rows: existing_rows = []
    existing_rows.append(new_row)
    return existing_rows

@callback(
    Output({'type': 'rule-value2', 'index': MATCH}, 'style'),
    Input({'type': 'rule-operator', 'index': MATCH}, 'value'),
     prevent_initial_call=True
)
def toggle_value2_input(operator):
    return {'display': 'block'} if operator == 'between' else {'display': 'none'}

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
    prevent_initial_call=True
)
def apply_rule_cohort(n_clicks, processed_data_dict, current_cohort_data_dict,
                      rule_cols, rule_ops, rule_vals1, rule_vals2, cohort_name_input):
    alert_error = lambda msg: (no_update, dbc.Alert(msg, color="danger", dismissable=True))
    alert_warning = lambda msg: (no_update, dbc.Alert(msg, color="warning", dismissable=True))
    alert_success = lambda msg: dbc.Alert(msg, color="success", dismissable=True, duration=4000)

    df_mod = None
    if current_cohort_data_dict:
        try: df_mod = pd.DataFrame(current_cohort_data_dict)
        except Exception as e: return alert_error(f"Err reading cohort data: {e}")
    elif processed_data_dict:
         try: df_mod = pd.DataFrame(processed_data_dict)
         except Exception as e: return alert_error(f"Err reading processed data: {e}")
    else: return alert_warning("Load data first.")
    if df_mod is None or df_mod.empty: return alert_warning("Data empty.")
    if not cohort_name_input or not cohort_name_input.strip(): return alert_warning("Enter cohort name.")
    cohort_name = cohort_name_input.strip().replace(' ', '_')
    if not all(c.isalnum() or c == '_' for c in cohort_name): return alert_warning("Cohort name: alphanumeric/underscores only.")
    cohort_col = f"cohort_{cohort_name}"
    if cohort_col in df_mod.columns: return alert_warning(f"Cohort '{cohort_col}' exists.")

    rules = []; valid = False
    for i, col in enumerate(rule_cols):
        if col is not None and str(col).strip() != "":
            op=rule_ops[i]; v1=rule_vals1[i]; v2=rule_vals2[i]
            if op is None: return alert_error(f"Rule {i+1}: Missing operator.")
            if v1 is None or str(v1).strip() == '': return alert_error(f"Rule {i+1}: Missing Value 1.")
            if op == 'between' and (v2 is None or str(v2).strip() == ''): return alert_error(f"Rule {i+1}: Missing Value 2 for 'between'.")
            rules.append({'column': col, 'op': op, 'value1': v1, 'value2': v2 if op == 'between' else None}); valid = True
    if not valid: return alert_warning("No valid rules defined.")

    try:
        bool_series = apply_rules_to_dataframe(df_mod.copy(), rules, cohort_col)
        str_series = bool_series.map({True: "True", False: "False"})
        df_mod[cohort_col] = str_series
        num_members = (str_series == "True").sum()
        status = alert_success(f"Cohort '{cohort_col}' ({num_members} members) applied.")
        return df_mod.to_dict('records'), status
    except ValueError as e: return alert_error(f"Rule error: {e}")
    except Exception as e: print(f"Apply rule error: {traceback.format_exc()}"); return alert_error(f"Unexpected error: {e}")