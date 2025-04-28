# app.py
import dash
import dash_bootstrap_components as dbc
import os

from layout import create_layout
# Import the new callback modules to register the callbacks
import callbacks_main
import callbacks_explorer
import callbacks_model

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.15.4/css/all.css"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.MINTY, FONT_AWESOME],
    suppress_callback_exceptions=True,
)

server = app.server

app.layout = create_layout()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)