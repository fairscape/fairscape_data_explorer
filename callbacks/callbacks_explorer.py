import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback

from utils.app_utils import create_empty_figure
from utils.ui_components import create_column_metadata_display_component # ADDED IMPORT

@callback(
    Output('group-column-selector', 'options', allow_duplicate=True), Output('group-column-selector', 'value', allow_duplicate=True), Output('group-column-selector', 'disabled', allow_duplicate=True),
    Input('available-columns-store', 'data'), Input('cohort-data-store', 'data'),
    State('group-column-selector', 'value'),
    prevent_initial_call=True
)
def update_grouping_dropdown(available_cols_data, cohort_data_dict, current_group_val):
    if not available_cols_data: return [], None, True
    orig_grp = available_cols_data.get('grouping_plot', [])
    orig_all = [c['value'] for c in available_cols_data.get('all', [])]
    new_cohort_cols = []
    if cohort_data_dict:
         try:
             df_c = pd.DataFrame(cohort_data_dict)
             if not df_c.empty:
                 added = [c for c in df_c.columns if c not in orig_all]
                 for col in added:
                     if df_c[col].nunique(dropna=True) <= 50 and df_c[col].notna().any():
                         is_float = pd.api.types.is_numeric_dtype(df_c[col]) and not pd.api.types.is_integer_dtype(df_c[col]) and not df_c[col].dropna().apply(float.is_integer).all()
                         if not is_float: new_cohort_cols.append(col)
         except Exception as e: print(f"Error proc cohort data for grouping: {e}")
    combined = sorted(list(set(orig_grp + new_cohort_cols)))
    opts = [{'label': c, 'value': c} for c in combined]
    new_val = current_group_val if current_group_val in combined else None
    is_dis = not bool(opts)
    return opts, new_val, is_dis

@callback(
    Output('group-value-filter-row', 'style', allow_duplicate=True), Output('group-value-filter', 'options', allow_duplicate=True), Output('group-value-filter', 'value', allow_duplicate=True),
    Input('group-column-selector', 'value'),
    State('cohort-data-store', 'data'),
    prevent_initial_call=True
)
def populate_group_filter_checklist(group_col, cohort_data_dict):
    if not group_col or not cohort_data_dict: return {'display': 'none'}, [], []
    try:
        df = pd.DataFrame(cohort_data_dict)
        if group_col not in df.columns: return {'display': 'none'}, [], []
        vals_raw = df[group_col].dropna().unique(); vals_str = sorted([str(v) for v in vals_raw])
        opts = [{'label': s, 'value': s} for s in vals_str]
        return {'display': 'block'}, opts, vals_str
    except Exception as e: print(f"Err populating group filter {group_col}: {e}"); return {'display': 'none'}, [], []

@callback(
    Output('data-histogram', 'figure', allow_duplicate=True),
    Input('cohort-data-store', 'data'), Input('numeric-column-selector', 'value'),
    Input('group-column-selector', 'value'), Input('group-value-filter', 'value'),
    prevent_initial_call=True
)
def update_histogram(data_dict, selected_col, group_col, selected_group_values):
    empty_fig = create_empty_figure("Select numeric column")
    if not data_dict or not selected_col:
        msg = 'Load data and select numeric column.' if not data_dict else 'Select numeric column.'
        empty_fig.layout.annotations[0]['text'] = msg; return empty_fig
    try:
        df = pd.DataFrame(data_dict)
        if df.empty: empty_fig.layout.annotations[0]['text'] = 'Data empty.'; return empty_fig
        if selected_col not in df.columns: empty_fig.layout.annotations[0]['text'] = f"Col '{selected_col}' not found."; return empty_fig
        if not pd.api.types.is_numeric_dtype(df[selected_col]): empty_fig.layout.annotations[0]['text'] = f"Col '{selected_col}' not numeric."; return empty_fig

        valid_grp = None; grp_warn = ""
        if group_col and group_col in df.columns:
             if df[group_col].notna().any():
                 nu = df[group_col].nunique(dropna=True)
                 if nu <= 50: valid_grp = group_col
                 else: grp_warn = f"'{group_col}' >50 unique vals."
             else: grp_warn = f"'{group_col}' has only NAs."
        elif group_col: grp_warn = f"Group col '{group_col}' not found."

        plot_df = df.copy()
        if valid_grp and selected_group_values is not None:
            try: plot_df = plot_df[plot_df[valid_grp].astype(str).isin(selected_group_values)]
            except Exception as e: print(f"Group filter error: {e}"); grp_warn += " (Filter error)"; valid_grp = None
        if plot_df.empty: empty_fig.layout.annotations[0]['text'] = 'No data matches filters.'; return empty_fig
        if valid_grp and valid_grp in plot_df.columns:
            if pd.api.types.is_bool_dtype(plot_df[valid_grp].dtype): plot_df[valid_grp] = plot_df[valid_grp].astype(str)
            plot_df[valid_grp] = plot_df[valid_grp].fillna('N/A').astype(str)
        else: valid_grp = None

        fig = px.histogram(plot_df, x=selected_col, color=valid_grp, histnorm='probability density', nbins=25, opacity=0.7,
                           title=f"Distribution of {selected_col}" + (f" by {valid_grp}" if valid_grp else ""),
                           template="plotly_white", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(barmode='overlay', legend_title_text=valid_grp or '', xaxis_title=selected_col, yaxis_title='Density', hovermode="x unified",
                          annotations=[{'text':grp_warn,'align':'left','showarrow':False,'xref':'paper','yref':'paper','x':0.05,'y':0.95,
                                        'bgcolor':'rgba(255,255,255,0.7)','bordercolor':'rgba(0,0,0,0.5)','borderwidth':1}] if grp_warn else [])
        fig.update_traces(marker_line_width=0.5, marker_line_color="black", opacity=0.75 if valid_grp else 0.8)
        return fig
    except Exception as e: tb_str = traceback.format_exc(); print(f"Hist error:\n{tb_str}"); empty_fig.layout.annotations[0]['text'] = f'Error: {e}'; return empty_fig

# --- NEW CALLBACK FOR COLUMN METADATA DISPLAY ---
@callback(
    Output('column-metadata-display-container', 'children'),
    Input('numeric-column-selector', 'value'),
    Input('schema-properties-store', 'data'),
    prevent_initial_call=True
)
def update_column_metadata_display(selected_numeric_column, schema_props_data):
    return create_column_metadata_display_component(schema_props_data, selected_numeric_column)