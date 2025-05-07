# mine/utils/ui_components.py
import dash_bootstrap_components as dbc
from dash import html

def create_column_metadata_display_component(schema_props_list, selected_column_name):
    if not selected_column_name:
        return dbc.Alert(
            "Select a numeric column from 'Plot Options' to view its metadata.",
            color="info",
            className="small p-2 text-center",
            id="column-metadata-content-alert"
        )

    if not schema_props_list:
        return dbc.Alert(
            f"Metadata for '{selected_column_name}': Schema information not available.",
            color="warning",
            className="small p-2",
            id="column-metadata-content-alert-warn"
        )

    column_info = None
    for prop in schema_props_list:
        if isinstance(prop, dict) and prop.get('name') == selected_column_name:
            column_info = prop
            break

    if not column_info:
        return dbc.Alert(
            f"Metadata for '{selected_column_name}': Not found in schema.",
            color="info",
            className="small p-2",
            id="column-metadata-content-alert-info"
        )

    description = column_info.get('description', 'No description available.')
    value_url = column_info.get('value-url')

    # Modified Card Body Content for Styling
    card_content = [
        # Removed "Column:" prefix, added bottom border style to H6
        html.H6(
            f"{selected_column_name}",
            className="card-subtitle mb-2 text-primary pb-1", # pb-1 adds padding below
            style={
                'fontSize': '0.9rem', 
                'borderBottom': '1px solid #dee2e6', # Adds a light grey line below
                'marginBottom': '0.5rem' # Space between border and description
            }
        ),
        html.P([html.Strong("Description: "), description], className="mb-1", style={'fontSize': '0.85rem'}),
    ]
    if value_url:
        card_content.append(html.P([html.Strong("Value URL: "), html.A(value_url, href=value_url, target="_blank", rel="noopener noreferrer")], className="mb-0", style={'fontSize': '0.85rem', 'wordBreak': 'break-all'}))
    else:
        card_content.append(html.P([html.Strong("Value URL: "), html.Span("Not specified.", className="text-muted")], className="mb-0", style={'fontSize': '0.85rem'}))

    return dbc.Card(
        dbc.CardBody(card_content, className="p-2"),
        # Changed bg-light to bg-white
        className="shadow-sm bg-white", 
        id="column-metadata-content-card",
        style={'borderColor': '#e9ecef'}
    )