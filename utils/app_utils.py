# app_utils.py
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html
import traceback

# --- General Helper ---
def update_status(message, color):
    """Standard way to update status alert."""
    return message, color

# --- Metadata/Schema Helper ---
def find_schema_ark(metadata_dict):
    """Finds the schema ARK within metadata."""
    if not isinstance(metadata_dict, dict):
        return None, "Metadata is not a dictionary."
    # Prioritize 'evi:schema', then 'schema'
    potential_keys = ['evi:schema', 'schema']
    found_key = None
    schema_val = None

    # Look for specific keys first
    for key in potential_keys:
        if key in metadata_dict:
            found_key = key
            schema_val = metadata_dict[key]
            break # Found preferred key

    # If not found, search case-insensitively (less preferred)
    if not found_key:
        for key, value in metadata_dict.items():
            key_norm = key.lower().replace('evi:', '')
            if key_norm == 'schema':
                found_key = key
                schema_val = value
                break

    if not found_key:
        return None, "No 'schema' or 'evi:schema' key found in metadata."

    # Check the format of the found value
    if isinstance(schema_val, str) and schema_val.startswith('ark:'):
        return schema_val, f"Schema ARK Found ({found_key})"
    elif isinstance(schema_val, dict) and '@id' in schema_val and schema_val['@id'].startswith('ark:'):
        return schema_val['@id'], f"Schema ARK Found ({found_key} @id)"
    else:
        return None, f"Key '{found_key}' found, but value is not a valid Schema ARK string or object with @id."

# --- Data Summary Helper ---
def update_summary_content(data_dict):
    """Generates the content for the data summary section."""
    if data_dict is None:
        return [html.P("Load data to see summary.", className="text-secondary")]
    if not isinstance(data_dict, list) or not data_dict:
         return [html.P("Data loaded, but contains 0 rows or is empty.", className="text-warning")]
    try:
        df = pd.DataFrame(data_dict)
        if df.empty:
             return [html.P("Data loaded, but contains 0 rows.", className="text-warning")]

        basic_info = html.Div([
            html.P(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns"),
        ], className="mb-3")

        # Attempt to generate descriptive statistics
        try:
             summary_df = df.describe(include='all', datetime_is_numeric=True).reset_index()
        except TypeError:
             print("Warning: Falling back on describe() without datetime_is_numeric=True")
             summary_df = df.describe(include='all').reset_index()
        except Exception as desc_e:
            print(f"Error during data description: {desc_e}")
            return [basic_info, html.P(f"Error during data description: {desc_e}", className="text-warning")]

        # Format the summary table
        cols = ['index'] + [col for col in summary_df if col != 'index'] # Ensure 'index' (statistic name) is first
        summary_df = summary_df[cols]
        summary_df = summary_df.round(3) # Round numeric values
        summary_df.fillna('', inplace=True) # Replace NaN with empty string for display

        summary_table = dbc.Table.from_dataframe(
            summary_df, striped=True, bordered=True, hover=True,
            responsive=True, className="small" # Use small class for tighter table
        )
        return [basic_info, summary_table]
    except Exception as e:
        print(f"Error generating summary: {e}\n{traceback.format_exc()}")
        return [html.P(f"Error generating summary: {e}", className="text-danger")]

# --- Placeholder Figures ---
def create_empty_figure(message="Load data first"):
    """Creates a standard empty plotly figure with a message."""
    fig = go.Figure(layout={'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                            'annotations': [{'text': message, 'xref': 'paper', 'yref': 'paper',
                                             'showarrow': False, 'font': {'size': 14}}]})
    fig.update_layout(template="plotly_white", margin=dict(t=20, b=20, l=20, r=20))
    return fig

def create_placeholder_plot(message="Plot will appear here."):
    """Creates a standard placeholder Div for plots."""
    return html.Div(message, style={'textAlign': 'center', 'padding': '20px', 'color': 'grey'})