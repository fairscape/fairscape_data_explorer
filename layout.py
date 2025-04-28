# layout.py
import dash_bootstrap_components as dbc
from dash import dcc, html

colors = {
    'primary': '#005f73',
    'secondary': '#ee9b00',
    'light': '#f8f9fa',
    'surface': '#ffffff',
    'text': '#212529',
    'border': '#dee2e6'
}

def create_cohort_definition_interface():
    return dbc.Accordion([
        dbc.AccordionItem([
            html.Div([
                html.P("Define cohorts by applying rules to columns.", className="small mb-3"),
                dbc.Input(id="cohort-name-input", placeholder="Enter Cohort Name", type="text", className="mb-2"),
                html.Div(id='rule-builder-container', children=[
                     create_rule_row(0)
                ]),
                dbc.Button("Add Rule Condition", id="add-rule-button", size="sm", color="light", className="me-2 mt-2 border"),
                dbc.Button("Apply Named Cohort", id="apply-rules-button", color="primary", size="sm", className="mt-2"),
                html.Div(id="rule-apply-status", className="mt-2 small")
            ])
        ], title="Define Cohort by Rules"),

        dbc.AccordionItem([
             html.Div([
                html.P("Upload a CSV file with cohort assignments (e.g., subject_id, cohort_label).", className="small mb-3"),
                dcc.Upload(
                    id='upload-cohort-csv',
                    children=html.Div(['Drag and Drop or ', html.A('Select Cohort CSV')]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed',
                        'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0px'
                    },
                    multiple=False
                ),
                dbc.Row([
                    dbc.Col(dbc.Label("Join Column (Main Data):", html_for="main-join-column-selector"), width=12),
                    dbc.Col(dcc.Dropdown(id='main-join-column-selector', options=[], placeholder="Select main data column...", disabled=True), width=12, className="mb-2"),

                    dbc.Col(dbc.Label("Join Column (Uploaded File):", html_for="upload-join-column-selector"), width=12),
                    dbc.Col(dcc.Dropdown(id='upload-join-column-selector', options=[], placeholder="Select uploaded column...", disabled=True), width=12, className="mb-2"),

                    dbc.Col(dbc.Label("Cohort Label Column (Uploaded File):", html_for="upload-cohort-column-selector"), width=12),
                    dbc.Col(dcc.Dropdown(id='upload-cohort-column-selector', options=[], placeholder="Select cohort label column...", disabled=True), width=12, className="mb-2"),
                ], class_name="mb-2"),
                dbc.Button("Process Uploaded Cohorts", id="process-upload-button", color="primary", size="sm", disabled=True),
                html.Div(id="upload-status", className="mt-2 small")
             ])
        ], title="Define Cohort by Upload")
    ], start_collapsed=True, flush=True, id="cohort-accordion")

def create_rule_row(index):
    operators = ['=', '!=', '>', '<', '>=', '<=', 'between']
    return dbc.Row([
        dbc.Col(dcc.Dropdown(id={'type': 'rule-column', 'index': index}, placeholder="Column", options=[], disabled=True), width=12, lg=5, className="mb-1 mb-lg-0"),
        dbc.Col(dcc.Dropdown(id={'type': 'rule-operator', 'index': index}, options=[{'label': op, 'value': op} for op in operators], placeholder="Operator", value='='), width=12, sm=6, lg=2, className="mb-1 mb-lg-0"),
        dbc.Col(dbc.Input(id={'type': 'rule-value1', 'index': index}, placeholder="Value", type="text"), width=12, sm=6, lg=3, className="mb-1 mb-lg-0"),
        dbc.Col(dbc.Input(id={'type': 'rule-value2', 'index': index}, placeholder="Value 2 (for between)", type="text", style={'display': 'none'}), width=12, sm=6, lg=2, className="mb-1 mb-lg-0 rule-value2-col"),
    ], className="mb-2 align-items-center", id={'type': 'rule-row', 'index': index})


def create_layout():
    layout = dbc.Container(fluid=True, className="d-flex flex-column min-vh-100 px-0", style={'backgroundColor': colors['light']}, children=[
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Home", href="https://fairscape.net/", external_link=True)),
                dbc.NavItem(dbc.NavLink("Upload", href="https://fairscape.net/upload", external_link=True)),
                dbc.NavItem(dbc.NavLink("Search", href="https://fairscape.net/search", external_link=True)),
                dbc.NavItem(dbc.NavLink("Documentation", href="https://fairscape.github.io/fairscape-cli/", external_link=True, target="_blank")),
            ],
            brand="Fairscape",
            brand_href="#",
            color="light",
            dark=False,
            className="border-bottom shadow-sm",
            style={'paddingLeft': '0.5rem', 'paddingRight': '0.5rem'} # Minimal padding
        ),

        dbc.Container(fluid=True, className="flex-grow-1 py-4 px-lg-3 px-md-2 px-1", children=[ # Fluid inner container with responsive padding
            dbc.Row([
                dbc.Col(width=12, lg=4, xl=3, className="mb-4 mb-lg-0", children=[
                    dbc.Card([
                        dbc.CardHeader(html.H4("Controls", className="mb-0 card-title"), style={'backgroundColor': colors['primary'], 'color': 'white'}),
                        dbc.CardBody([
                            dbc.Row(class_name="mb-3", children=[
                                dbc.Label("1. Data ARK Identifier", html_for="data-ark-input", width=12, className="fw-bold"),
                                dbc.Col(dbc.Input(id='data-ark-input', placeholder="ark:...", type="text", value="ark:99999/fp10v-jqk40"), width=12),
                            ]),
                            dbc.Button("Load & Process ARK", id='load-button', n_clicks=0, color="secondary", className="w-100 mb-3 fw-bold", style={'backgroundColor': colors['secondary'], 'borderColor': colors['secondary'], 'color': colors['text']}),

                            html.H4("Status", className="mt-4 mb-2 fs-5", style={'color': colors['primary']}),
                            dbc.Alert(
                                "Enter a Data ARK and click Load.", id="status-alert", color="secondary",
                                dismissable=False, is_open=True, className="small",
                                style={"white-space": "pre-wrap", 'maxHeight': '200px', 'overflowY': 'auto'}
                            ),

                            html.Hr(className="my-4"),
                            html.H4("Plot Options", className="mb-3 fs-5", style={'color': colors['primary']}),
                            dbc.Row(class_name="mb-3", children=[
                                dbc.Label("Numeric Column (Histogram):", html_for="numeric-column-selector", width=12, className="fw-bold"),
                                dbc.Col(dcc.Dropdown(id='numeric-column-selector', options=[], value=None, placeholder="Select column...", disabled=True), width=12),
                            ]),
                            dbc.Row(class_name="mb-3", children=[
                                dbc.Label("Group Histogram By:", html_for="group-column-selector", width=12, className="fw-bold"),
                                dbc.Col(dcc.Dropdown(id='group-column-selector', options=[], value=None, placeholder="Optional grouping...", disabled=True, clearable=True), width=12),
                            ]),
                            dbc.Row(class_name="mb-3", children=[
                                dbc.Label("Filter Group Values:", html_for="group-value-filter", width=12, className="fw-bold"),
                                dbc.Col(width=12, children=[
                                    dbc.Spinner(
                                        html.Div(dbc.Checklist(
                                                    id='group-value-filter',
                                                    options=[],
                                                    value=[],
                                                    inline=True,
                                                    className="small mb-1"
                                                ), style={'maxHeight': '150px', 'overflowY': 'auto', 'border': f'1px solid {colors["border"]}', 'borderRadius': '5px', 'padding': '5px'}),
                                        size="sm", color="secondary"
                                    ),
                                    dbc.FormText("Select values to include (defaults to all).", color="secondary", className="mt-1"),
                                ])
                            ], id='group-value-filter-row', style={'display': 'none'}),

                            html.Hr(className="my-4"),
                            html.H4("Define Cohorts", className="mb-3 fs-5", style={'color': colors['primary']}),
                            create_cohort_definition_interface(),

                        ])
                    ])
                ]),

                dbc.Col(width=12, lg=8, xl=9, children=[
                     dbc.Card([
                        dbc.CardHeader(html.H4("Data Summary", className="mb-0 card-title"), style={'backgroundColor': colors['primary'], 'color': 'white'}),
                        dbc.CardBody([
                            dbc.Spinner(html.Pre(id='data-summary', children=["Load data to see summary."], className="bg-white p-3 border rounded small", style={'maxHeight': '300px', 'overflow': 'auto', 'backgroundColor': '#fdfdfe'})),
                            dcc.Download(id="download-summary"),
                            dbc.Button("Download Summary Text", id="btn-download-summary", color="secondary", size="sm", className="mt-2", disabled=True, style={'backgroundColor': colors['secondary'], 'borderColor': colors['secondary'], 'color': colors['text']}),
                        ])
                     ]),

                    html.Hr(className="my-4"),

                    dbc.Card([
                        dbc.CardHeader(html.H4("Data Histogram", className="mb-0 card-title"), style={'backgroundColor': colors['primary'], 'color': 'white'}),
                        dbc.CardBody([
                            dbc.Spinner(dcc.Graph(id='data-histogram', figure={}, style={'height': '450px'}))
                        ])
                    ])
                ])
            ]),
        ]),

        dcc.Store(id='processed-data-store', storage_type='memory'),
        dcc.Store(id='schema-properties-store', storage_type='memory'),
        dcc.Store(id='available-columns-store', storage_type='memory'),
        dcc.Store(id='cohort-data-store', storage_type='memory'),
        dcc.Store(id='uploaded-cohort-store', storage_type='memory'),
    ])
    return layout