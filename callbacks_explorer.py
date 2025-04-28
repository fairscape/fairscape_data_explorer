# callbacks_explorer.py
import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback

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
    if not available_cols_data: return [], None, True
    original_grouping_cols = available_cols_data.get('grouping_plot', [])
    all_original_cols = [c['value'] for c in available_cols_data.get('all', [])]
    new_cohort_cols = []
    if cohort_data_dict:
         try:
             df_cohort = pd.DataFrame(cohort_data_dict)
             if not df_cohort.empty:
                 current_cols = df_cohort.columns
                 added_cols = [col for col in current_cols if col not in all_original_cols]
                 for col in added_cols:
                     if df_cohort[col].nunique(dropna=True) <= 50 and df_cohort[col].notna().any():
                        if not (pd.api.types.is_numeric_dtype(df_cohort[col]) and not pd.api.types.is_integer_dtype(df_cohort[col])):
                            new_cohort_cols.append(col)
         except Exception as e: print(f"Error processing cohort data for grouping dropdown: {e}")
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
    if not group_col or not cohort_data_dict: return {'display': 'none'}, [], []
    try:
        df = pd.DataFrame(cohort_data_dict)
        if group_col not in df.columns: return {'display': 'none'}, [], []
        unique_values_raw = df[group_col].dropna().unique()
        unique_values_str = sorted([str(v) for v in unique_values_raw])
        options = [{'label': val_str, 'value': val_str} for val_str in unique_values_str]
        values = unique_values_str
        return {'display': 'block'}, options, values
    except Exception as e: print(f"Error populating group filter checklist for {group_col}: {e}"); return {'display': 'none'}, [], []

@callback(
    Output('data-histogram', 'figure', allow_duplicate=True),
    Input('cohort-data-store', 'data'),
    Input('numeric-column-selector', 'value'),
    Input('group-column-selector', 'value'),
    Input('group-value-filter', 'value'),
    prevent_initial_call=True
)
def update_histogram(data_dict, selected_col, group_col, selected_group_values):
    empty_fig = go.Figure(layout={'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                                 'annotations': [{'text': 'Select numeric column', 'xref': 'paper', 'yref': 'paper',
                                                'showarrow': False, 'font': {'size': 14}}]})
    empty_fig.update_layout(template="plotly_white")
    if not data_dict or not selected_col: empty_fig.layout.annotations[0]['text'] = 'Load data and select numeric column.'; return empty_fig
    try:
        df = pd.DataFrame(data_dict)
        if df.empty or selected_col not in df.columns: empty_fig.layout.annotations[0]['text'] = 'Column not found or data empty.'; return empty_fig
        if not pd.api.types.is_numeric_dtype(df[selected_col]): empty_fig.layout.annotations[0]['text'] = f"Column '{selected_col}' not numeric."; return empty_fig
        valid_group_col = None; group_col_warning = ""
        if group_col and group_col in df.columns:
             if df[group_col].notna().any():
                 unique_count = df[group_col].nunique(dropna=True)
                 if unique_count <= 50: valid_group_col = group_col
                 else: group_col_warning = f"'{group_col}' >50 unique values, grouping disabled."
             else: group_col_warning = f"'{group_col}' has only missing values."
        elif group_col: group_col_warning = f"Group column '{group_col}' not found."
        plot_df = df.copy()
        if valid_group_col and selected_group_values is not None:
            try: plot_df = plot_df[plot_df[valid_group_col].astype(str).isin(selected_group_values)]
            except Exception as filter_e: print(f"Error applying group filter: {filter_e}"); group_col_warning += " (Filter error)"
        if valid_group_col and valid_group_col in plot_df.columns:
            if pd.api.types.is_bool_dtype(plot_df[valid_group_col].dtype): plot_df[valid_group_col] = plot_df[valid_group_col].astype(str)
            plot_df[valid_group_col] = plot_df[valid_group_col].fillna('N/A').astype(str)
        else: valid_group_col = None
        if plot_df.empty: empty_fig.layout.annotations[0]['text'] = 'No data matches filters.'; return empty_fig
        fig = px.histogram(
            plot_df, x=selected_col, color=valid_group_col, marginal="rug", histnorm='probability density', opacity=0.7,
            title=f"Distribution of {selected_col}" + (f" by {valid_group_col}" if valid_group_col else ""),
            template="plotly_white", color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            barmode='overlay', legend_title_text=valid_group_col if valid_group_col else '',
            xaxis_title=selected_col, yaxis_title='Density', hovermode="x unified",
            annotations=[{'text': group_col_warning, 'align': 'left', 'showarrow': False, 'xref': 'paper', 'yref': 'paper', 'x': 0.05, 'y': 0.95,
                          'bgcolor': 'rgba(255,255,255,0.7)', 'bordercolor': 'rgba(0,0,0,0.5)', 'borderwidth': 1}] if group_col_warning else []
        )
        fig.update_traces(marker_line_width=0.5, marker_line_color="black", opacity=0.75 if valid_group_col else 0.8)
        return fig
    except Exception as e: tb_str = traceback.format_exc(); print(f"Error generating histogram:\n{tb_str}"); empty_fig.layout.annotations[0]['text'] = f'Error: {e}'; return empty_fig