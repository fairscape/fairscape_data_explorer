# mine/app.py
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# Import layout function
from layout import create_layout

# --- Import Callback Modules ---
# This ensures Dash discovers the callbacks defined in these files
from callbacks import callbacks_main
from callbacks import callbacks_explorer
from callbacks import callbacks_model
from callbacks import callbacks_rules
from callbacks import callbacks_upload

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

app.layout = create_layout() 


if __name__ == '__main__':
    app.run(debug=True)