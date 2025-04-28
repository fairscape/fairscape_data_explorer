# model_utils.py
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc
import io
import base64
from dash import html

def determine_model_type(df, y_col, schema_props_dict=None):
    if y_col not in df.columns:
        raise ValueError(f"Target column '{y_col}' not found in data.")

    y_series = df[y_col].dropna()
    if y_series.empty:
        raise ValueError(f"Target column '{y_col}' has no non-missing values.")

    # 1. Check Schema if available
    if schema_props_dict:
        schema_info = next((item for item in schema_props_dict if item['name'] == y_col), None)
        if schema_info:
            schema_type = schema_info.get('type', '').lower()
            if schema_type == 'boolean':
                return 'logistic'
            elif schema_type in ['integer', 'number']:
                # Check if integer looks binary despite schema saying integer/number
                if pd.api.types.is_integer_dtype(y_series) and set(y_series.unique()) <= {0, 1}:
                     return 'logistic'
                return 'linear'
            # Add other schema type handling if needed

    # 2. Infer from data type and values if schema doesn't dictate
    dtype = y_series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return 'logistic'
    elif pd.api.types.is_integer_dtype(dtype):
        # Check if it's binary (0/1)
        unique_values = set(y_series.unique())
        if unique_values <= {0, 1}:
            return 'logistic'
        else:
            return 'linear' # Treat other integers as continuous for linear model
    elif pd.api.types.is_numeric_dtype(dtype): # Includes float
        return 'linear'
    else:
        # Attempt logistic if it looks like binary strings e.g. "yes"/"no", "true"/"false" - basic check
        unique_vals_str = set(y_series.astype(str).str.lower().unique())
        if len(unique_vals_str) == 2:
            # Add more robust checks if needed
            return 'logistic'


    raise ValueError(f"Cannot determine model type for column '{y_col}' with dtype {dtype} and unique values {y_series.unique()}. Consider using a boolean, numeric, or binary (0/1) column.")

def _prepare_data_for_model(df, y_col, x_cols):
    """Handles common data prep: selects columns, drops missing values."""
    if not y_col or not x_cols:
        raise ValueError("Target (Y) and at least one Predictor (X) column must be selected.")

    cols_to_use = [y_col] + x_cols
    missing_cols = [col for col in cols_to_use if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in data: {', '.join(missing_cols)}")

    # Select relevant columns and drop rows with NAs in *any* of these columns
    model_df = df[cols_to_use].copy().dropna()

    if model_df.empty:
        raise ValueError("No data remaining after removing rows with missing values in selected columns.")

    # Ensure X columns are numeric (statsmodels requirement without formula API/dummies)
    non_numeric_x = [col for col in x_cols if not pd.api.types.is_numeric_dtype(model_df[col])]
    if non_numeric_x:
         raise ValueError(f"Predictor (X) columns must be numeric for this simple model. Non-numeric columns found: {', '.join(non_numeric_x)}")

    # Add constant for intercept
    X = sm.add_constant(model_df[x_cols])
    y = model_df[y_col]

    return y, X, model_df # Return model_df in case it's needed later

def fit_linear_model(df, y_col, x_cols):
    y, X, _ = _prepare_data_for_model(df, y_col, x_cols)
    model = sm.OLS(y, X)
    results = model.fit()
    return results

def fit_logistic_model(df, y_col, x_cols):
    y, X, model_df = _prepare_data_for_model(df, y_col, x_cols)

    # Ensure y is 0/1 for Logit
    y_unique = y.unique()
    if not (pd.api.types.is_bool_dtype(y.dtype) or set(y_unique) <= {0, 1}):
         # Attempt conversion if possible (e.g., strings 'true'/'false')
        try:
            bool_map = {'true': 1, 't': 1, '1': 1, 'yes': 1, 'y': 1,
                        'false': 0, 'f': 0, '0': 0, 'no': 0, 'n': 0}
            y = y.astype(str).str.lower().map(bool_map)
            if y.isna().any():
                raise ValueError("Could not convert all Y values to 0/1 for logistic regression.")
            y = y.astype(int)
        except Exception as e:
            raise ValueError(f"Logistic regression target column '{y_col}' must be boolean or contain only 0 and 1. Found unique values: {y_unique}. Error: {e}")


    model = sm.Logit(y, X)
    results = model.fit()
    return results

def create_model_summary_display(model_results):
    """
    Parses the statsmodels summary object and returns a list of
    styled dbc.Table components.
    """
    try:
        summary = model_results.summary()
        summary_tables = []

        # Extract title if available (usually part of the first table's structure)
        if hasattr(summary, 'title'):
             summary_tables.append(html.H5(summary.title, className="text-center mb-3"))

        for i, table in enumerate(summary.tables):
            # Convert SimpleTable to pandas DataFrame
            if isinstance(table, SimpleTable) and hasattr(table, 'as_html'):
                 try:
                     # Use read_html to parse the table, simpler than as_dataframe()
                     table_df = pd.read_html(table.as_html(), header=0, index_col=0)[0]
                     # Special handling for the middle table (coefficients) to reset index
                     if i == 1: # Heuristic: middle table is usually coefficients
                          table_df = table_df.reset_index()
                     # Convert DataFrame to dbc.Table
                     dbc_table = dbc.Table.from_dataframe(
                         table_df,
                         striped=True,
                         bordered=True,
                         hover=True,
                         responsive=True,
                         className="small mb-4" # Add bottom margin
                     )
                     summary_tables.append(dbc_table)
                 except Exception as parse_e:
                     # Fallback: display raw HTML if parsing fails
                     print(f"Warning: Could not parse summary table {i} into DataFrame: {parse_e}")
                     summary_tables.append(html.Div(html.Pre(table.as_html()), className="mb-3"))
            else:
                 # Fallback for non-SimpleTable parts (rare)
                 summary_tables.append(html.Pre(str(table), className="mb-3"))


        if not summary_tables: # If no tables were extracted
            return html.Pre(summary.as_text()) # Fallback to text

        return html.Div(summary_tables) # Return a Div containing the list of tables

    except Exception as e:
        # Broad exception catch if summary generation fails
        print(f"Error generating styled model summary display: {e}")
        # Fallback to plain text summary in a Pre tag
        try:
            return html.Pre(model_results.summary().as_text())
        except Exception as fallback_e:
             return html.Pre(f"Could not generate model summary: {fallback_e}")

# --- UPDATED FUNCTION: create_logistic_roc_plot ---
def create_logistic_roc_plot(model_results, df, y_col, x_cols):
    try:
        y, X, model_df = _prepare_data_for_model(df, y_col, x_cols)

        y_unique = y.unique()
        if not (pd.api.types.is_bool_dtype(y.dtype) or set(y_unique) <= {0, 1}):
            bool_map = {'true': 1, 't': 1, '1': 1, 'yes': 1, 'y': 1,
                        'false': 0, 'f': 0, '0': 0, 'no': 0, 'n': 0}
            y_numeric = y.astype(str).str.lower().map(bool_map)
            if y_numeric.isna().any():
                raise ValueError("Could not convert all Y values to 0/1 for ROC calculation.")
            y_numeric = y_numeric.astype(int)
        else:
            y_numeric = y.astype(int)

        y_pred_prob = model_results.predict(X)
        fpr, tpr, thresholds = roc_curve(y_numeric, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(dash='dash', color='grey')))

        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Sensitivity)',
            # --- Legend Position Update ---
            legend=dict(
                x=0.01,        # Position slightly right of the left edge
                y=0.01,        # Position slightly up from the bottom edge
                traceorder='reversed',
                bgcolor='rgba(255,255,255,0.7)', # Semi-transparent background
                bordercolor='rgba(0,0,0,0.5)',
                borderwidth=1,
                xanchor='left',  # Anchor legend point is bottom-left
                yanchor='bottom'
            ),
            # --- End Legend Position Update ---
            template="plotly_white",
            height=400,
            xaxis=dict(range=[0.0, 1.0]), # Ensure axes cover 0-1 range
            yaxis=dict(range=[0.0, 1.05]) # Ensure axes cover 0-1 range (+ slight margin)
        )
        return fig
    except Exception as e:
         # Return an informative error message within a Div
         return html.Div([
             html.Strong("Error generating ROC plot:"),
             html.Pre(str(e), style={'color': 'red', 'marginTop': '10px'})
         ])

def create_linear_residual_plot(model_results, df, y_col, x_cols):
    try:
        y, X, model_df = _prepare_data_for_model(df, y_col, x_cols) # Get aligned data
        fitted_values = model_results.fittedvalues
        residuals = model_results.resid

        fig = px.scatter(
            x=fitted_values,
            y=residuals,
            labels={'x': 'Fitted Values', 'y': 'Residuals'},
            title='Residuals vs. Fitted Values',
            template="plotly_white",
            trendline="lowess", # Add a lowess smoother to check for patterns
            trendline_color_override="red"
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(height=400)
        return fig
    except Exception as e:
        return html.Div(f"Error generating residual plot: {e}")