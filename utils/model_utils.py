# mine/utils/model_utils.py
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc
import io
from dash import html
import dash_bootstrap_components as dbc # Added dbc
import numpy as np
import traceback # Keep for error printing

# Constants for styling the new card
MODEL_CARD_HEADER_BG = '#005f73'  # Equivalent to colors['primary']
MODEL_CARD_HEADER_TEXT = 'white'


def determine_model_type(df, y_col_name, schema_props_dict=None):
    # ... (existing code)
    if y_col_name not in df.columns:
        raise ValueError(f"Target variable '{y_col_name}' not found in the data.")

    y_series_original = df[y_col_name]
    y_series_no_na = y_series_original.dropna()

    if y_series_no_na.empty:
        # Check if original series has only specific NA-like strings before raising error
        if pd.api.types.is_string_dtype(y_series_original.dtype) or pd.api.types.is_object_dtype(y_series_original.dtype):
            na_strings = {'n/a', 'nan', '<na>', '', 'none', 'null'}
            if set(y_series_original.astype(str).str.lower().unique()).issubset(na_strings):
                 raise ValueError(f"Target variable '{y_col_name}' contains only NA-like string values (e.g., 'N/A', '', 'None').")
        raise ValueError(f"Target variable '{y_col_name}' contains only standard missing values (NA/NaN/None) after initial dropna.")


    # 1. Check Schema Hint (if provided)
    if schema_props_dict:
        schema_info = next((item for item in schema_props_dict if item['name'] == y_col_name), None)
        if schema_info:
            schema_type = schema_info.get('type', '').lower()
            if schema_type == 'boolean': return 'logistic'
            # If schema says numeric, still need to check if it's actually binary
            # Fall through to data checks

    # 2. Check Data Type and Values
    dtype = y_series_no_na.dtype # Use dtype of series after dropping standard NAs

    if pd.api.types.is_bool_dtype(dtype): return 'logistic'
    if pd.api.types.is_integer_dtype(dtype):
        unique_values = set(y_series_no_na.unique())
        if unique_values.issubset({0, 1}): return 'logistic'
        else: return 'linear'
    if pd.api.types.is_float_dtype(dtype):
        unique_values = set(y_series_no_na.unique())
        # Check if all non-NA floats are close to 0 or 1
        is_binary_float = all(np.isclose(val, 0) or np.isclose(val, 1) for val in unique_values)
        if is_binary_float and len(unique_values) <= 2: return 'logistic'
        else: return 'linear'

    # 3. Check String/Object Types (Handle 'N/A' etc.)
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        # Use the original series before dropna to catch strings like 'N/A'
        y_str_series = y_series_original.astype(str).str.lower()
        unique_vals_str_lower = set(y_str_series.unique())

        # Define known missing strings (lowercase) to ignore
        missing_strings = {'n/a', 'nan', '<na>', '', 'none', 'null'}

        # Get the set of unique values EXCLUDING the missing strings
        meaningful_unique_vals = unique_vals_str_lower - missing_strings
        
        if not meaningful_unique_vals: # Only contained missing strings
            raise ValueError(f"Target variable '{y_col_name}' contains only NA-like string values (e.g., 'N/A', '', 'None') after attempting to find meaningful uniques.")


        # Define valid binary patterns (sets of lowercase strings)
        binary_patterns = [
            {'true', 'false'}, {'yes', 'no'}, {'y', 'n'},
            {'t', 'f'}, {'1', '0'}, {'1.0', '0.0'}
        ]

        # Check if the meaningful values match any binary pattern *exactly* or is a subset (e.g., just '1.0')
        for pattern in binary_patterns:
            if meaningful_unique_vals.issubset(pattern) and len(meaningful_unique_vals) > 0:
                return 'logistic'

        # If not binary strings, try converting to numeric to see if it's suitable for linear
        try:
            y_numeric_test = pd.to_numeric(y_series_original, errors='coerce')
            if y_numeric_test.dropna().shape[0] > 0: # Check if *any* value could be converted
                 # Check if the numeric values are non-binary
                 unique_numeric = y_numeric_test.dropna().unique()
                 # Ensure it's not just 0,1 after numeric conversion if it wasn't caught by string patterns
                 if not (np.isin(unique_numeric, [0, 1, 0.0, 1.0]).all() and len(unique_numeric) <=2) :
                      return 'linear'
        except Exception:
            pass # Ignore conversion errors

    # If none of the above conditions met, raise error
    sample_vals = np.unique(y_series_no_na)[:5] if not y_series_no_na.empty else "[]"
    # Check if original series had any non-NA-like strings values
    if pd.api.types.is_string_dtype(y_series_original.dtype) or pd.api.types.is_object_dtype(y_series_original.dtype):
        y_str_series_orig_lower = y_series_original.astype(str).str.lower()
        meaningful_unique_vals_orig = set(y_str_series_orig_lower.unique()) - {'n/a', 'nan', '<na>', '', 'none', 'null'}
        if not meaningful_unique_vals_orig:
            raise ValueError(f"Target variable '{y_col_name}' consists only of various forms of missing data (e.g. 'NA', '', 'None'). Cannot determine model type.")
    
    raise ValueError(f"Cannot determine model type for column '{y_col_name}'. Requires boolean, numeric, binary (0/1 or 0.0/1.0), or binary strings (e.g., 'True'/'False'). Found dtype {df[y_col_name].dtype} with unique non-NA values (sample): {sample_vals}")


def _prepare_data_for_model(df, y_col, x_cols):
    # ... (existing code)
    if not y_col or not x_cols:
        raise ValueError("Target (Y) and at least one Predictor (X) column must be selected.")

    cols_to_use = [y_col] + x_cols
    missing_cols = [col for col in cols_to_use if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in data: {', '.join(missing_cols)}")

    model_df = df[cols_to_use].copy()
    initial_rows = len(model_df)

    non_numeric_x = []
    for col in x_cols:
         if col in model_df:
              try:
                  # Convert X to numeric, coerce errors. NaNs handled by dropna later.
                  model_df[col] = pd.to_numeric(model_df[col], errors='coerce')
                  if model_df[col].isna().all(): # All values became NaN after conversion
                      non_numeric_x.append(f"{col} (all values became NA after numeric conversion)")

              except Exception as e: # Should not happen with errors='coerce'
                   non_numeric_x.append(f"{col} (error during numeric conversion: {e})")
         else: 
              raise ValueError(f"Predictor column '{col}' unexpectedly missing.")

    if non_numeric_x:
        raise ValueError(f"Could not convert Predictor (X) columns to numeric or they became all NA: {', '.join(non_numeric_x)}")

    # Handle Y column specifically for NA-like strings before dropna
    if pd.api.types.is_string_dtype(model_df[y_col].dtype) or pd.api.types.is_object_dtype(model_df[y_col].dtype):
        missing_strings_map = {'n/a': np.nan, 'nan': np.nan, '<na>': np.nan, '': np.nan, 'none': np.nan, 'null': np.nan}
        y_series_lower = model_df[y_col].astype(str).str.lower()
        
        # For logistic: map binary strings to 0/1 then to numeric
        binary_patterns_map = {
            'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'y': 1, 'n': 0,
            't': 1, 'f': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0
        }
        # Combine with missing strings map, ensure binary map takes precedence if a value like '0' is also in missing_strings_map
        combined_map = {**missing_strings_map, **binary_patterns_map}
        
        # Apply mapping: first specific binary/NA strings, then general to_numeric
        # This requires knowing model_type, or determine_model_type has to be more robust
        # For now, _prepare_data_for_model will make Y numeric if possible, or leave as is for determine_model_type to handle.
        # It is critical that determine_model_type is called on data *before* this heavy Y conversion.
        # Let's assume Y is already suitable or determine_model_type confirmed it.
        # The main goal here is to make sure NA-like strings become actual np.nan for dropna.
        
        mapped_y = y_series_lower.map(combined_map)
        # If not in map, try direct pd.to_numeric, else keep original (will likely fail later if not numeric)
        # Coerce unmapped to NaN for numeric types. If it's supposed to be strings that can be binary, they should be in combined_map.
        model_df[y_col] = pd.to_numeric(mapped_y, errors='coerce')
        if model_df[y_col].isna().all() and not y_series_lower.isin(missing_strings_map.keys()).all():
             # All became NaN but not all were originally missing strings - this means some binary strings might not have converted properly
             # Re-evaluate: simpler is to just map explicit NA strings to np.nan for consistent dropna,
             # and let type-specific conversions happen in fit_logistic/linear based on y_clean
             model_df[y_col] = y_series_lower.replace(missing_strings_map).pipe(pd.to_numeric, errors='coerce')


    # Drop rows with NA in Y or ANY X column *after* coercions
    model_df.dropna(subset=cols_to_use, inplace=True)
    cleaned_rows = len(model_df)

    if cleaned_rows == 0:
        raise ValueError(f"No data remaining after removing rows with missing values in Y ('{y_col}') or X ({', '.join(x_cols)}). Original rows: {initial_rows}. Consider checking data quality or variable selection.")

    y_clean = model_df[y_col]
    X_clean = sm.add_constant(model_df[x_cols], has_constant='add') 

    return y_clean, X_clean

def fit_logistic_model(df, y_col, x_cols):
    # ... (existing code)
    y_clean, X_clean = _prepare_data_for_model(df, y_col, x_cols)

    y_numeric = None
    dtype = y_clean.dtype

    if pd.api.types.is_bool_dtype(dtype):
        y_numeric = y_clean.astype(int)
    elif pd.api.types.is_integer_dtype(dtype) and y_clean.isin([0, 1]).all():
        y_numeric = y_clean.astype(int)
    elif pd.api.types.is_float_dtype(dtype):
        # Check if values are close to 0 or 1
        is_binary_float = np.isclose(y_clean, 0) | np.isclose(y_clean, 1)
        if is_binary_float.all() and y_clean.nunique() <=2 : # Ensure only 0 and 1 (or one of them) exist
             y_numeric = y_clean.round().astype(int)
        else:
            unique_non_binary = y_clean[~is_binary_float].unique()
            sample_non_binary = unique_non_binary[:min(len(unique_non_binary), 5)]
            raise ValueError(f"Logistic regression target column '{y_col}' has non-binary float values after cleaning: {sample_non_binary}. Ensure Y is 0/1 or boolean.")
    # This case should ideally be caught by determine_model_type if it made it this far with strings.
    # elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
    #     # This path suggests _prepare_data_for_model didn't fully convert Y, which can happen if original Y was string like "True"/"False"
    #     # and pd.to_numeric(..., errors='coerce') in _prepare_data_for_model made them NaN.
    #     # It's better if determine_model_type ensures Y is convertible and _prepare_data_for_model converts it.
    #     # For now, assume if it's here, it should have been numeric.
    #     temp_y_series_original_for_logistic = df[y_col].dropna() # Re-fetch original to attempt string conversion
    #     str_map = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'y': 1, 'n': 0, 't':1, 'f':0, '1':1, '0':0, '1.0':1, '0.0':0}
    #     y_lower_str = temp_y_series_original_for_logistic.astype(str).str.lower()
    #     if y_lower_str.isin(str_map.keys()).all():
    #         y_numeric = y_lower_str.map(str_map).astype(int)
    #     else:
    #         unmapped_vals = y_lower_str[~y_lower_str.isin(str_map.keys())].unique()[:5]
    #         raise ValueError(f"Logistic regression target column '{y_col}' has unmappable string values: {unmapped_vals} after cleaning. Ensure Y is convertible to 0/1.")
    else:
         raise ValueError(f"Logistic regression target column '{y_col}' has unsuitable type ({dtype}) or values after cleaning for logistic model. Expected 0/1, boolean, or convertible strings.")

    if y_numeric.nunique() < 2:
         raise ValueError(f"Target variable '{y_col}' has only one unique value ({y_numeric.unique()}) after cleaning and conversion to 0/1. Cannot fit logistic model.")

    model = sm.Logit(y_numeric, X_clean)
    results = model.fit(maxiter=100, disp=0) # disp=0 to suppress convergence messages in console
    return results

def fit_linear_model(df, y_col, x_cols):
    # ... (existing code)
    y_clean, X_clean = _prepare_data_for_model(df, y_col, x_cols)

    if not pd.api.types.is_numeric_dtype(y_clean.dtype): # Should be numeric after _prepare_data
         raise ValueError(f"Linear regression target column '{y_col}' must be numeric. Found dtype {y_clean.dtype} after cleaning.")
    if pd.api.types.is_bool_dtype(y_clean.dtype): # bool is numeric, but not for linear
         raise ValueError(f"Linear regression target column '{y_col}' is boolean. Please use logistic regression or ensure Y is continuous numeric.")


    model = sm.OLS(y_clean, X_clean)
    results = model.fit()
    return results

def create_model_summary_display(model_results):
    # ... (existing code)
    try:
        summary_text = model_results.summary().as_text()
        # Use a div with pre-wrap for better control over size than <pre>
        return html.Div(summary_text, className="border rounded p-2 bg-light", style={'fontSize': '11px', 'whiteSpace': 'pre-wrap', 'overflowX': 'auto'})
    except Exception as e:
        print(f"Error generating text model summary display: {e}")
        return html.Pre(f"Could not generate model summary: {e}")

def create_logistic_roc_plot(model_results, df, y_col, x_cols):
    # ... (existing code)
    try:
        # Re-use y_numeric logic from fit_logistic_model for consistency
        y_clean, X_clean = _prepare_data_for_model(df, y_col, x_cols)
        y_numeric = None
        dtype = y_clean.dtype

        if pd.api.types.is_bool_dtype(dtype): y_numeric = y_clean.astype(int)
        elif pd.api.types.is_integer_dtype(dtype) and y_clean.isin([0, 1]).all(): y_numeric = y_clean.astype(int)
        elif pd.api.types.is_float_dtype(dtype):
             is_binary_float = np.isclose(y_clean, 0) | np.isclose(y_clean, 1)
             if is_binary_float.all() and y_clean.nunique() <=2: y_numeric = y_clean.round().astype(int)
             else: raise ValueError(f"Cannot create ROC plot for non-binary float Y column '{y_col}'.")
        # elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        #     temp_y_series_original_for_roc = df[y_col].dropna()
        #     str_map = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'y': 1, 'n': 0, 't':1, 'f':0, '1':1, '0':0, '1.0':1, '0.0':0}
        #     y_lower_str = temp_y_series_original_for_roc.astype(str).str.lower()
        #     if y_lower_str.isin(str_map.keys()).all():
        #         y_numeric = y_lower_str.map(str_map).astype(int)
        #     else:
        #         raise ValueError(f"Y column '{y_col}' has unmappable string values for ROC plot after cleaning.")
        else:
             raise ValueError(f"Y column '{y_col}' has unsuitable type ({dtype}) for ROC plot after cleaning.")

        if y_numeric is None: 
             raise RuntimeError("Failed to prepare Y variable for ROC plot.")
        if y_numeric.nunique() < 2: # Check again after specific conversion for ROC
            raise ValueError(f"Target variable '{y_col}' for ROC plot has only one unique value ({y_numeric.unique()}) after conversion. Cannot compute ROC.")


        y_pred_prob = model_results.predict(X_clean)

        fpr, tpr, thresholds = roc_curve(y_numeric, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.3f})', line=dict(width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance (AUC = 0.5)', line=dict(dash='dash', color='grey')))
        fig.update_layout(
            title=f'ROC Curve for Logistic Regression (Y = {y_col})',
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Sensitivity)',
            legend=dict(x=0.5, y=-0.2, traceorder='reversed', bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.5)', borderwidth=1, orientation="h", xanchor='center', yanchor='top'),
            template="plotly_white", height=450,
            xaxis=dict(range=[-0.02, 1.02]), yaxis=dict(range=[-0.02, 1.02]),
            margin=dict(b=80)
        )
        return fig
    except Exception as e:
         print(f"Error generating ROC plot: {e}\n{traceback.format_exc()}")
         return html.Div([
             html.Strong("Error generating ROC plot:"),
             html.Pre(str(e), style={'color': 'red', 'marginTop': '10px', 'whiteSpace': 'pre-wrap'})
         ], className="border rounded p-2 bg-light")

def create_linear_residual_plot(model_results, df, y_col, x_cols):
    # ... (existing code)
    try:
        y_clean, X_clean = _prepare_data_for_model(df, y_col, x_cols)

        fitted_values = model_results.fittedvalues
        residuals = model_results.resid

        fig = px.scatter(
            x=fitted_values, y=residuals,
            labels={'x': 'Fitted Values', 'y': 'Residuals'},
            title=f'Residuals vs. Fitted Values (Y = {y_col})',
            template="plotly_white",
            trendline="lowess", trendline_color_override="red", opacity=0.7
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
        fig.update_layout(height=450, hovermode='closest')
        return fig
    except Exception as e:
        print(f"Error generating residual plot: {e}\n{traceback.format_exc()}")
        return html.Div([
             html.Strong("Error generating residual plot:"),
             html.Pre(str(e), style={'color': 'red', 'marginTop': '10px', 'whiteSpace': 'pre-wrap'})
        ], className="border rounded p-2 bg-light")


def create_model_equation_display(model_results, model_type, y_col_name, x_cols, schema_props_list):
    """
    Creates a Dash Card component to display the model equation and variable details.
    """
    params = model_results.params
    equation_parts = []
    accordion_items = []

    def _get_metadata(col_name_md, schema_props_list_md):
        if not schema_props_list_md:
            return None, "Schema information not available for this dataset."
        for prop in schema_props_list_md:
            if isinstance(prop, dict) and prop.get('name') == col_name_md:
                desc = prop.get('description', 'No description provided in schema.')
                val_url = prop.get('value-url')
                return val_url, desc
        return None, f"No schema entry found for column '{col_name_md}'."

    # 1. Target Variable (Y)
    y_val_url, y_desc = _get_metadata(y_col_name, schema_props_list)
    
    y_name_component = html.A(y_col_name, href=y_val_url, target="_blank", rel="noopener noreferrer", className="fw-bold text-decoration-none") if y_val_url else html.Span(y_col_name, className="fw-bold")
    
    if model_type == 'logistic':
        equation_parts.append(html.Span("Log-Odds("))
        equation_parts.append(y_name_component)
        equation_parts.append(html.Span(") = "))
    else:  # linear
        equation_parts.append(y_name_component)
        equation_parts.append(html.Span(" = "))

    accordion_items.append(
        dbc.AccordionItem(
            title=html.Span([html.Strong(y_col_name), html.Em(" (Target Variable)")]),
            children=[
                html.P([html.Strong("Description: "), html.Span(y_desc if y_desc else "Not available.")]),
                html.P([html.Strong("Schema Value URL: "), html.A(y_val_url, href=y_val_url, target="_blank", rel="noopener noreferrer") if y_val_url else "Not specified."])
            ]
        )
    )

    # 2. Intercept
    intercept_val = params.get('const', 0)
    equation_parts.append(html.Span(f"{intercept_val:.4f}"))
    
    accordion_items.append(
        dbc.AccordionItem(
            title=html.Span([html.Strong("Intercept"), html.Em(" (Constant Term)")]),
            children=[html.P(f"The estimated intercept (constant) value of the model is {intercept_val:.4f}. "
                             "This is the expected value of the target (or log-odds of target for logistic models) "
                             "when all predictor variables are zero.")]
        )
    )

    # 3. Predictor Variables (X)
    for x_col_name_iter in x_cols:
        if x_col_name_iter == 'const': continue

        coeff_val = params.get(x_col_name_iter)
        if coeff_val is None: # Should not happen if x_cols are from model
            continue

        sign = " + " if coeff_val >= 0 else " - "
        equation_parts.append(html.Span(sign))
        # Multiplication sign: \u00D7 or simply '*'
        equation_parts.append(html.Span(f"{abs(coeff_val):.4f} \u00D7 ")) 

        x_val_url, x_desc = _get_metadata(x_col_name_iter, schema_props_list)
        x_name_component = html.A(x_col_name_iter, href=x_val_url, target="_blank", rel="noopener noreferrer", className="fw-bold text-decoration-none") if x_val_url else html.Span(x_col_name_iter, className="fw-bold")
        equation_parts.append(x_name_component)

        accordion_items.append(
            dbc.AccordionItem(
                title=html.Span([html.Strong(x_col_name_iter), html.Em(" (Predictor Variable)")]),
                children=[
                    html.P([html.Strong("Coefficient: "), f"{coeff_val:.4f}"]),
                    html.P([html.Strong("Description: "), html.Span(x_desc if x_desc else "Not available.")]),
                    html.P([html.Strong("Schema Value URL: "), html.A(x_val_url, href=x_val_url, target="_blank", rel="noopener noreferrer") if x_val_url else "Not specified."])
                ]
            )
        )
    
    equation_style = {
        'fontSize': '1.1rem', 'wordWrap': 'break-word', 'padding': '15px', 
        'border': '1px solid #ced4da', 'borderRadius': '0.25rem', 
        'backgroundColor': '#f8f9fa', 'marginBottom': '20px',
        'fontFamily': 'monospace' # For better alignment of equation
    }
    
    card_body_content = [
        html.Div(equation_parts, style=equation_style),
        html.H6("Learn More About Variables:", className="mt-4 mb-2 text-secondary"),
        dbc.Accordion(accordion_items, start_collapsed=True, flush=True, always_open=False)
    ]

    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5("Model Equation & Variable Interpretation", className="mb-0 card-title"),
                style={'backgroundColor': MODEL_CARD_HEADER_BG, 'color': MODEL_CARD_HEADER_TEXT}
            ),
            dbc.CardBody(card_body_content, className="p-3")
        ],
        className="shadow-sm"
    )