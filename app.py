# app.py
import dash
import dash_bootstrap_components as dbc
import os

# Import layout and callbacks
from layout import create_layout
import callbacks # Needs to be imported to register callbacks

# Import config (optional, but good practice if callbacks need direct access)
# from config import * # Or specific variables

# --- Initialize Dash App with Bootstrap Theme ---
# Use a theme that aligns with the desired style (e.g., MINTY, LUX, FLATLY)
# Or use the default BOOTSTRAP and rely on custom dbc component styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY],
                suppress_callback_exceptions=True) # Set to True if callbacks are in different files

# Expose server for WSGI deployment (like gunicorn)
server = app.server

# --- Assign Layout ---
app.layout = create_layout()

# --- Run App ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on the network
    # debug=True enables hot reloading and error pages (disable for production)
    app.run(debug=True, host='0.0.0.0', port=8050) # Use a common Dash port like 8050