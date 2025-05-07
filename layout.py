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

def create_rule_row(index):
    operators = ['=', '!=', '>', '<', '>=', '<=', 'between']
    return dbc.Row([
        dbc.Col(dcc.Dropdown(id={'type': 'rule-column', 'index': index}, placeholder="Column", options=[], disabled=True, className="dbc"), width=12, lg=5, className="mb-1 mb-lg-0"),
        dbc.Col(dcc.Dropdown(id={'type': 'rule-operator', 'index': index}, options=[{'label': op, 'value': op} for op in operators], placeholder="Operator", value='=', className="dbc"), width=12, sm=6, lg=2, className="mb-1 mb-lg-0"),
        dbc.Col(dbc.Input(id={'type': 'rule-value1', 'index': index}, placeholder="Value", type="text", className="form-control-sm"), width=12, sm=6, lg=3, className="mb-1 mb-lg-0"),
        dbc.Col(dbc.Input(id={'type': 'rule-value2', 'index': index}, placeholder="Value 2 (for between)", type="text", style={'display': 'none'}, className="form-control-sm"), width=12, sm=6, lg=2, className="mb-1 mb-lg-0 rule-value2-col"),
    ], className="mb-2 align-items-center", id={'type': 'rule-row', 'index': index})

def create_cohort_definition_interface():
    # Removed 'small' class from H5 for a slightly larger, standard H5 header in sidebar
    return dbc.Card(className="mb-4 shadow-sm", children=[
        dbc.CardHeader(html.H5("Define/Upload Cohorts", className="mb-0 card-title"), style={'backgroundColor': colors['light'], 'color': colors['primary'], 'borderBottom': f'1px solid {colors["border"]}'}),
        dbc.CardBody(style={'padding': '0.75rem'}, children=[ # p-2 equivalent, good for sidebar
            dbc.Accordion([
                dbc.AccordionItem([
                    html.Div([
                        html.P("Define cohorts by applying rules to columns.", className="small mb-2"),
                        dbc.Input(id="cohort-name-input", placeholder="Enter Cohort Name", type="text", className="mb-2 form-control-sm"),
                        html.Div(id='rule-builder-container', children=[
                             create_rule_row(0)
                        ]),
                        dbc.Button("Add Rule Condition", id="add-rule-button", size="sm", color="light", className="me-1 mt-2 border"),
                        dbc.Button("Apply Named Cohort", id="apply-rules-button", color="primary", size="sm", className="mt-2"),
                        html.Div(id="rule-apply-status", className="mt-2 small")
                    ])
                ], title="Define Cohort by Rules"),
                dbc.AccordionItem([
                     html.Div([
                        html.P("Upload a CSV with cohort assignments.", className="small mb-2"),
                        dcc.Upload(
                            id='upload-cohort-csv',
                            children=html.Div(['Drag/Drop or ', html.A('Select CSV')]),
                            style={
                                'width': '100%', 'height': '50px', 'lineHeight': '50px',
                                'borderWidth': '1px', 'borderStyle': 'dashed',
                                'borderRadius': '5px', 'textAlign': 'center', 'margin': '5px 0px'
                            },
                            multiple=False, className="small"
                        ),
                        dbc.Row([
                            dbc.Col(dbc.Label("Join Col (Main):", html_for="main-join-column-selector", className="small fw-bold"), width=12),
                            dbc.Col(dcc.Dropdown(id='main-join-column-selector', options=[], placeholder="Select main...", disabled=True, className="dbc"), width=12, className="mb-2"),
                            dbc.Col(dbc.Label("Join Col (Upload):", html_for="upload-join-column-selector", className="small fw-bold"), width=12),
                            dbc.Col(dcc.Dropdown(id='upload-join-column-selector', options=[], placeholder="Select upload...", disabled=True, className="dbc"), width=12, className="mb-2"),
                            dbc.Col(dbc.Label("Cohort Label Col:", html_for="upload-cohort-column-selector", className="small fw-bold"), width=12),
                            dbc.Col(dcc.Dropdown(id='upload-cohort-column-selector', options=[], placeholder="Select label...", disabled=True, className="dbc"), width=12, className="mb-2"),
                        ], class_name="mb-2 g-1"), # g-1 for smaller gutters
                        dbc.Button("Process Upload", id="process-upload-button", color="primary", size="sm", disabled=True),
                        html.Div(id="upload-status", className="mt-2 small")
                     ])
                ], title="Define Cohort by Upload")
            ], start_collapsed=True, flush=True, id="cohort-accordion", always_open=False)
        ])
    ])

def create_data_exploration_tab():
    return dbc.Row([
         dbc.Col(width=12, lg=4, xl=3, className="mb-4 mb-lg-0", children=[
             dbc.Card(className="shadow-sm", children=[ # Plot Options Card
                 dbc.CardHeader(html.H4("Plot Options", className="mb-0 card-title"), style={'backgroundColor': colors['primary'], 'color': 'white'}),
                 dbc.CardBody(className="p-3", children=[ # Explicit p-3 for consistent padding
                      dbc.Row(class_name="mb-3", children=[
                          dbc.Label("Numeric Column (Histogram):", html_for="numeric-column-selector", width=12, className="fw-bold"),
                          dbc.Col(dcc.Dropdown(id='numeric-column-selector', options=[], value=None, placeholder="Select column...", disabled=True, className="dbc"), width=12),
                      ]),
                      dbc.Row(class_name="mb-3", children=[
                          dbc.Label("Group Histogram By:", html_for="group-column-selector", width=12, className="fw-bold"),
                          dbc.Col(dcc.Dropdown(id='group-column-selector', options=[], value=None, placeholder="Optional grouping...", disabled=True, clearable=True, className="dbc"), width=12),
                      ]),
                      dbc.Row(class_name="mb-3", children=[ # Retained mb-3 for spacing
                          dbc.Label("Filter Group Values:", html_for="group-value-filter", width=12, className="fw-bold"),
                          dbc.Col(width=12, children=[
                              dbc.Spinner(
                                  html.Div(dbc.Checklist(
                                              id='group-value-filter',
                                              options=[],
                                              value=[],
                                              inline=True,
                                              className="small mb-1" # Checklist itself
                                          ), style={'maxHeight': '150px', 'overflowY': 'auto', 'border': f'1px solid {colors["border"]}', 'borderRadius': '5px', 'padding': '5px'}), # Container for checklist
                                  size="sm", color="secondary"
                              ),
                              dbc.FormText("Select values to include.", color="secondary", className="mt-1 small"),
                          ])
                      ], id='group-value-filter-row', style={'display': 'none'}),
                 ])
             ]),
             html.Div(id='dataset-details-links-container', className="mt-3 shadow-sm"), # Added shadow-sm here
         ]),
         dbc.Col(width=12, lg=8, xl=9, children=[
             dbc.Card(className="mb-3 shadow-sm", children=[ # Data Summary Card, changed mb-4 to mb-3
                dbc.CardHeader(html.H4("Data Summary", className="mb-0 card-title"), style={'backgroundColor': colors['primary'], 'color': 'white'}),
                dbc.CardBody(className="p-3", children=[ # Explicit p-3
                    dbc.Spinner(html.Pre(id='data-summary', children=["Load data to see summary."], className="bg-white p-3 border rounded small", style={'maxHeight': '300px', 'overflow': 'auto', 'backgroundColor': '#fdfdfe'})),
                    dbc.Row(className="mt-3", children=[ # Increased margin for download buttons
                        dbc.Col(dbc.Button("Download Summary CSV", id="btn-download-summary", color="secondary", size="sm", className="me-1", disabled=True, style={'backgroundColor': colors['secondary'], 'borderColor': colors['secondary'], 'color': colors['text']}), width="auto"),
                        dbc.Col(dbc.Button("Download Exploration HTML", id="btn-download-html", color="secondary", size="sm", disabled=True, style={'backgroundColor': colors['secondary'], 'borderColor': colors['secondary'], 'color': colors['text']}), width="auto"),
                    ], justify="start"),
                    dcc.Download(id="download-summary"),
                    dcc.Download(id="download-exploration-html"),
                ])
             ]),
             html.Hr(className="my-3"), # Added HR for separation
             html.Div(id='column-metadata-display-container', className="mb-3"), # Changed my-3 to mb-3
             html.Hr(className="my-3"), # Added HR for separation
             dbc.Card(className="shadow-sm",children=[ # Data Histogram Card
                 dbc.CardHeader(html.H4("Data Histogram", className="mb-0 card-title"), style={'backgroundColor': colors['primary'], 'color': 'white'}),
                 dbc.CardBody(className="p-3", children=[ # Explicit p-3
                     dbc.Spinner(dcc.Graph(id='data-histogram', figure={}, style={'height': '450px'}))
                 ])
             ])
         ])
     ])

def create_model_building_tab():
     return dbc.Row([
         dbc.Col(width=12, lg=4, xl=3, className="mb-4 mb-lg-0", children=[
             dbc.Card(className="shadow-sm",children=[ # Model Options Card
                 dbc.CardHeader(html.H4("Model Options", className="mb-0 card-title"), style={'backgroundColor': colors['primary'], 'color': 'white'}),
                 dbc.CardBody(className="p-3", children=[ # Explicit p-3
                     dbc.Row(class_name="mb-3", children=[
                         dbc.Label("Target (Y) Column:", html_for="model-y-selector", width=12, className="fw-bold"),
                         dbc.Col(dcc.Dropdown(id='model-y-selector', options=[], value=None, placeholder="Select target...", disabled=True, clearable=False, className="dbc"), width=12),
                         dbc.FormText("Select numeric (Linear) or boolean/binary (Logistic).", color="secondary", className="mt-1 small"),
                     ]),
                     dbc.Row(class_name="mb-3", children=[
                         dbc.Label("Predictor (X) Columns:", html_for="model-x-selector", width=12, className="fw-bold"),
                         dbc.Col(dcc.Dropdown(id='model-x-selector', options=[], value=[], placeholder="Select predictors...", disabled=True, multi=True, className="dbc"), width=12),
                          dbc.FormText("Select one or more numeric columns.", color="secondary", className="mt-1 small"),
                     ]),
                     dbc.Button("Build Model", id='build-model-button', n_clicks=0, color="secondary", className="w-100 fw-bold mt-2", style={'backgroundColor': colors['secondary'], 'borderColor': colors['secondary'], 'color': colors['text']}, disabled=True), # Added mt-2
                     html.Div(id="model-status", className="mt-2 small")
                 ])
             ])
         ]),
         dbc.Col(width=12, lg=8, xl=9, children=[
             dbc.Card(className="mb-3 shadow-sm", children=[ # Model Summary Card, changed mb-4 to mb-3
                 dbc.CardHeader(html.H4("Model Summary", className="mb-0 card-title"), style={'backgroundColor': colors['primary'], 'color': 'white'}),
                 dbc.CardBody(className="p-3 text-center", children=[ # Explicit p-3
                     dbc.Spinner(html.Div(id='model-summary-output', children=["Build a model to see the summary."]))
                 ])
             ]),
             html.Hr(className="my-3"), # Added HR for separation
             dbc.Card(className="shadow-sm", children=[ # Model Plot Card
                 dbc.CardHeader(html.H4("Model Plot", className="mb-0 card-title"), style={'backgroundColor': colors['primary'], 'color': 'white'}),
                 dbc.CardBody(className="p-3", children=[ # Explicit p-3
                      dbc.Spinner(html.Div(id='model-plot-output', children=[
                          html.Div("Plot will appear here after model build.", style={'textAlign': 'center', 'padding': '20px', 'color': 'grey'})
                     ], style={'minHeight': '450px'})) # minHeight ensures space
                 ])
             ])
         ])
     ])

def create_layout():
    sidebar_content = dbc.Container(fluid=True, children=[ # Removed p-3 from here, let components define their margins
        dbc.Row(class_name="mb-3 align-items-end", children=[ # ARK Input section
             dbc.Col(width=12, children=[
                dbc.Label("Data ARK Identifier", html_for="data-ark-input", className="fw-bold small"),
                dbc.Input(id='data-ark-input', placeholder="ark:...", type="text", value="ark:99999/fp10v-jqk40", className="form-control-sm"),
             ]),
             dbc.Col(width=12, className="mt-2", children=[
                 dbc.Button("Load & Process ARK", id='load-button', n_clicks=0, color="secondary", size="sm", className="w-100 fw-bold", style={'backgroundColor': colors['secondary'], 'borderColor': colors['secondary'], 'color': colors['text']}),
             ]),
        ]),
        dbc.Row(class_name="mb-3", children=[ # Status Alert section
            dbc.Col(width=12, children=[
                dbc.Label("Status", className="fw-bold small"),
                dbc.Alert(
                    "Enter Data ARK and click Load.", id="status-alert", color="secondary",
                    dismissable=False, is_open=True, className="small p-2 mb-0", # mb-0 as row has mb-3
                    style={"white-space": "pre-wrap", 'maxHeight': '150px', 'overflowY': 'auto'}
                ),
            ])
        ]),
        html.Hr(className="my-3"), # Hr before cohort definition
        create_cohort_definition_interface(), # This card already has mb-4
    ])

    nav_links = dbc.Nav(
        [
            dbc.NavItem(dbc.NavLink("Home", href="https://fairscape.net/", external_link=True)),
            dbc.NavItem(dbc.NavLink("Upload", href="https://fairscape.net/upload", external_link=True)),
            dbc.NavItem(dbc.NavLink("Search", href="https://fairscape.net/search", external_link=True)),
            dbc.NavItem(dbc.NavLink("Documentation", href="https://fairscape.github.io/fairscape-cli/", external_link=True, target="_blank")),
        ],
        className="ms-auto",
        navbar=True
    )

    navbar = dbc.Navbar(
        dbc.Container(fluid=True, children=[
            dbc.Button(html.I(className="fas fa-bars"), id="open-offcanvas-button", n_clicks=0, color="light", className="border me-2"),
            dbc.NavbarBrand("Fairscape Data Explorer", href="#"),
            nav_links
        ]),
        color="light", # Bootstrap 'light' background
        dark=False,
        className="border-bottom shadow-sm flex-shrink-0", # Standard navbar styling
        sticky="top",
        style={'padding': '0.25rem 0.5rem'} # Reduced padding for a thinner navbar
    )

    layout = dbc.Container(fluid=True, className="d-flex flex-column vh-100 overflow-hidden", style={'backgroundColor': colors['light']}, children=[
        navbar,
        dbc.Offcanvas(
            sidebar_content,
            id="sidebar-offcanvas",
            title="Data Source & Cohorts", # Offcanvas title
            is_open=True,
            placement="start",
            backdrop=False, # Allows interaction with main page
            scrollable=True,
            style={'width': '450px', 'borderRight': f'1px solid {colors["border"]}'} # Added border for crisper separation
        ),
        dbc.Container(fluid=True, id="page-content", className="flex-grow-1 overflow-auto py-3 px-lg-3 px-md-2 px-1", children=[ # Reduced py-4 to py-3
            dbc.Tabs(id="main-tabs", active_tab="tab-explore", children=[
                dbc.Tab(label="Data Exploration", tab_id="tab-explore", children=[
                    html.Div(create_data_exploration_tab(), className="pt-3") # Reduced pt-4 to pt-3
                ]),
                dbc.Tab(label="Model Building", tab_id="tab-model", children=[
                    html.Div(create_model_building_tab(), className="pt-3") # Reduced pt-4 to pt-3
                ]),
            ]),
        ]),
        # Data stores remain the same
        dcc.Store(id='processed-data-store', storage_type='memory'),
        dcc.Store(id='schema-properties-store', storage_type='memory'),
        dcc.Store(id='available-columns-store', storage_type='memory'),
        dcc.Store(id='cohort-data-store', storage_type='memory'),
        dcc.Store(id='uploaded-cohort-store', storage_type='memory'),
    ])
    return layout