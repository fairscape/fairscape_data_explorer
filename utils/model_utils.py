# model_utils.py
import pandas as pd
import statsmodels.api as sm
# from statsmodels.iolib.table import SimpleTable # No longer needed for simple text summary
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc
import io
# import base64 # Not used here
from dash import html
# import dash_bootstrap_components as dbc # No longer needed for simple text summary
import numpy as np # Import numpy

# --- Updated: determine_model_type ---
def determine_model_type(df, y_col, schema_props_dict=None):
    """Determines if the model should be 'linear' or 'logistic' based on Y column."""
    if y_col not in df.columns:
        raise ValueError(f"Target column '{y_col}' not found in data.")

    y_series = df[y_col].dropna()
    if y_series.empty:
        raise ValueError(f"Target column '{y_col}' has no non-missing values.")

    # 1. Check Schema if available (Hint)
    if schema_props_dict:
        schema_info = next((item for item in schema_props_dict if item['name'] == y_col), None)
        if schema_info:
            schema_type = schema_info.get('type', '').lower()
            if schema_type == 'boolean': return 'logistic'
            elif schema_type in ['integer', 'number']:
                if pd.api.types.is_integer_dtype(y_series.dtype) and set(y_series.unique()) <= {0, 1}: return 'logistic'
                elif pd.api.types.is_float_dtype(y_series.dtype) and set(y_series.unique()) <= {0.0, 1.0}: return 'logistic'
                return 'linear'

    # 2. Infer from data type and values
    dtype = y_series.dtype
    if pd.api.types.is_bool_dtype(dtype): return 'logistic'
    elif pd.api.types.is_integer_dtype(dtype):
        unique_values = set(y_series.unique())
        if unique_values <= {0, 1}: return 'logistic'
        else: return 'linear'
    elif pd.api.types.is_float_dtype(dtype):
        unique_values = set(y_series.unique())
        is_binary_float = all(np.isclose(val, 0) or np.isclose(val, 1) for val in unique_values)
        if is_binary_float and len(unique_values) <= 2: return 'logistic'
        else: return 'linear'
    elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        unique_vals_lower = set(y_series.astype(str).str.lower().unique())
        binary_pairs = [{'true', 'false'}, {'yes', 'no'}, {'y', 'n'}, {'t', 'f'}, {'1', '0'}]
        if any(unique_vals_lower == pair for pair in binary_pairs): return 'logistic'

    raise ValueError(f"Cannot determine model type for column '{y_col}'. Requires boolean, numeric, binary (0/1 or 0.0/1.0), or binary strings (e.g., 'True'/'False'). Found dtype {dtype} with unique values (sample): {np.unique(y_series)[:5]}")


def _prepare_data_for_model(df, y_col, x_cols):
    """Handles common data prep: selects columns, checks types, adds constant."""
    if not y_col or not x_cols:
        raise ValueError("Target (Y) and at least one Predictor (X) column must be selected.")

    cols_to_use = [y_col] + x_cols
    missing_cols = [col for col in cols_to_use if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in data: {', '.join(missing_cols)}")

    model_df = df[cols_to_use].copy()

    # Ensure X columns are numeric *before* dropna
    non_numeric_x = []
    for col in x_cols:
         # Check if the column exists and if it's not numeric after dropping NAs just for the check
         if col in model_df and not pd.api.types.is_numeric_dtype(model_df[col].dropna()):
              # Try conversion
              try:
                  model_df[col] = pd.to_numeric(model_df[col], errors='raise')
              except (ValueError, TypeError):
                  non_numeric_x.append(col) # Add to list if conversion failed

    if non_numeric_x:
        raise ValueError(f"Predictor (X) columns must be numeric. Non-numeric columns found: {', '.join(non_numeric_x)}")

    # Add constant for intercept BEFORE dropna
    X_with_const = sm.add_constant(model_df[x_cols], has_constant='add')
    y = model_df[y_col]

    # Combine y and X_with_const for consistent NA dropping based on selected variables
    final_df_for_na_drop = pd.concat([y.rename('__target_y__'), X_with_const], axis=1)

    # Drop rows with NA in EITHER the target OR any predictor (including the added constant if somehow NA)
    final_df_clean = final_df_for_na_drop.dropna(subset=['__target_y__'] + list(X_with_const.columns))

    if final_df_clean.empty:
        raise ValueError("No data remaining after removing rows with missing values in Y or selected X columns.")

    # Separate cleaned Y and X
    y_clean = final_df_clean['__target_y__']
    X_clean = final_df_clean[list(X_with_const.columns)] # Ensure constant column is included

    return y_clean, X_clean


# --- Updated: fit_logistic_model ---
def fit_logistic_model(df, y_col, x_cols):
    """Fits a logistic regression model, handling various Y column formats."""
    y_clean_initial, X_clean = _prepare_data_for_model(df, y_col, x_cols)

    # --- Convert Cleaned Y to 0/1 for Logit ---
    y_numeric = None
    dtype = y_clean_initial.dtype
    unique_vals = y_clean_initial.unique() # Use unique on already cleaned data

    if pd.api.types.is_bool_dtype(dtype):
        y_numeric = y_clean_initial.astype(int)
    elif pd.api.types.is_integer_dtype(dtype) and set(unique_vals) <= {0, 1}:
        y_numeric = y_clean_initial.astype(int)
    elif pd.api.types.is_float_dtype(dtype):
        # Check if floats are essentially 0.0/1.0 after cleaning
        is_binary_float = all(np.isclose(val, 0) or np.isclose(val, 1) for val in unique_vals)
        if is_binary_float and len(unique_vals) <= 2:
             y_numeric = y_clean_initial.astype(int)
        else:
            # This condition should ideally be caught by determine_model_type, but raise error here if not
            raise ValueError(f"Linear regression target column '{y_col}' cannot be non-binary float. Found values like: {unique_vals[:5]}")
    elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        y_str_lower = y_clean_initial.astype(str).str.lower()
        bool_map = {'true': 1, 't': 1, '1': 1, 'yes': 1, 'y': 1, 'false': 0, 'f': 0, '0': 0, 'no': 0, 'n': 0}
        y_mapped = y_str_lower.map(bool_map)
        # After mapping, check if ONLY 0 and 1 exist (mapping handles NAs implicitly if not in map)
        if y_mapped.isin([0, 1]).all():
            y_numeric = y_mapped.astype(int) # Convert to int after successful mapping
        else:
             # Check which values failed conversion
             failed_values = y_str_lower[~y_mapped.isin([0, 1])].unique()
             raise ValueError(f"Could not convert all string values in '{y_col}' to binary 0/1 for logistic regression. Problematic values found: {failed_values[:5]}")
    else:
         raise ValueError(f"Logistic regression target column '{y_col}' has unsuitable type ({dtype}) or values after cleaning. Requires boolean, binary int/float, or binary strings.")

    # Fit the model using the cleaned and converted data
    model = sm.Logit(y_numeric, X_clean) # Use y_numeric and X_clean
    results = model.fit() # Add fit_method='bfgs' or others if needed
    return results


# --- fit_linear_model - simplified NA handling via _prepare_data_for_model ---
def fit_linear_model(df, y_col, x_cols):
    """Fits a linear regression model."""
    y_clean, X_clean = _prepare_data_for_model(df, y_col, x_cols)

    # Ensure cleaned Y is numeric for OLS (should be caught by determine_model_type, but double check)
    if not pd.api.types.is_numeric_dtype(y_clean):
         try:
              # Attempt conversion one last time on cleaned data
              y_clean = pd.to_numeric(y_clean, errors='raise')
         except (ValueError, TypeError):
              raise ValueError(f"Linear regression target column '{y_col}' must be numeric. Found dtype {y_clean.dtype} after cleaning.")

    model = sm.OLS(y_clean, X_clean)
    results = model.fit()
    return results


# --- Reverted: create_model_summary_display ---
def create_model_summary_display(model_results):
    """
    Returns the standard statsmodels summary as preformatted text.
    """
    try:
        # Get the summary as text
        summary_text = model_results.summary().as_text()
        # Wrap it in an html.Pre component for proper display
        return html.Pre(summary_text, className="border rounded p-2 bg-light small")
    except Exception as e:
        print(f"Error generating text model summary display: {e}")
        # Fallback if summary generation fails
        return html.Pre(f"Could not generate model summary: {e}")


# --- Updated: create_logistic_roc_plot ---
def create_logistic_roc_plot(model_results, df, y_col, x_cols):
    """Generates ROC plot, handling various Y column formats using cleaned data."""
    try:
        # Get the cleaned data used for the model fit
        y_clean_initial, X_clean = _prepare_data_for_model(df, y_col, x_cols)

        # --- Convert Cleaned Y to 0/1 for ROC calculation (same logic as fit_logistic_model) ---
        y_numeric = None
        dtype = y_clean_initial.dtype
        unique_vals = y_clean_initial.unique()

        if pd.api.types.is_bool_dtype(dtype): y_numeric = y_clean_initial.astype(int)
        elif pd.api.types.is_integer_dtype(dtype) and set(unique_vals) <= {0, 1}: y_numeric = y_clean_initial.astype(int)
        elif pd.api.types.is_float_dtype(dtype):
             is_binary_float = all(np.isclose(val, 0) or np.isclose(val, 1) for val in unique_vals)
             if is_binary_float and len(unique_vals) <= 2: y_numeric = y_clean_initial.astype(int)
             else: raise ValueError(f"Cannot create ROC plot for non-binary float Y column '{y_col}'.")
        elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            y_str_lower = y_clean_initial.astype(str).str.lower()
            bool_map = {'true': 1, 't': 1, '1': 1, 'yes': 1, 'y': 1, 'false': 0, 'f': 0, '0': 0, 'no': 0, 'n': 0}
            y_mapped = y_str_lower.map(bool_map)
            if y_mapped.isin([0, 1]).all(): y_numeric = y_mapped.astype(int)
            else:
                 failed_values = y_str_lower[~y_mapped.isin([0, 1])].unique()
                 raise ValueError(f"Cannot convert Y column '{y_col}' to binary 0/1 for ROC plot. Problematic values: {failed_values[:5]}")
        else:
             raise ValueError(f"Y column '{y_col}' has unsuitable type ({dtype}) for ROC plot after cleaning.")

        # --- Get Predictions using the same X data used for fitting ---
        y_pred_prob = model_results.predict(X_clean) # Predict on X_clean

        # --- Calculate ROC using the numeric Y aligned with X_clean ---
        fpr, tpr, thresholds = roc_curve(y_numeric, y_pred_prob) # Use y_numeric
        roc_auc = auc(fpr, tpr)

        # --- Create Plot (remains the same) ---
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
    """Generates Residual vs Fitted plot for linear models using cleaned data."""
    try:
        # Get the cleaned data used for the model fit
        y_clean, X_clean = _prepare_data_for_model(df, y_col, x_cols)

        # Use the results directly, as they are aligned with the cleaned data used for fitting
        fitted_values = model_results.fittedvalues
        residuals = model_results.resid

        # --- Create Plot (remains the same) ---
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