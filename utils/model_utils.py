import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc
import io
from dash import html
import numpy as np
import traceback # Keep for error printing


def determine_model_type(df, y_col_name, schema_props_dict=None):
    if y_col_name not in df.columns:
        raise ValueError(f"Target variable '{y_col_name}' not found in the data.")

    y_series_original = df[y_col_name]
    y_series_no_na = y_series_original.dropna()

    if y_series_no_na.empty:
        raise ValueError(f"Target variable '{y_col_name}' contains only standard missing values (NA/NaN/None).")

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
                 if not np.isin(unique_numeric, [0, 1, 0.0, 1.0]).all():
                      return 'linear'
        except Exception:
            pass # Ignore conversion errors

    # If none of the above conditions met, raise error
    sample_vals = np.unique(y_series_no_na)[:5] if not y_series_no_na.empty else "[]"
    raise ValueError(f"Cannot determine model type for column '{y_col_name}'. Requires boolean, numeric, binary (0/1 or 0.0/1.0), or binary strings (e.g., 'True'/'False'). Found dtype {df[y_col_name].dtype} with unique values (sample): {sample_vals}")


def _prepare_data_for_model(df, y_col, x_cols):
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
              except Exception as e:
                   # Should not happen with errors='coerce', but catch just in case
                   non_numeric_x.append(f"{col} (error: {e})")
         else: # Should have been caught earlier
              raise ValueError(f"Predictor column '{col}' unexpectedly missing.")

    if non_numeric_x:
        # This error should ideally not be reached if check is done after coerce
        raise ValueError(f"Could not convert Predictor (X) columns to numeric: {', '.join(non_numeric_x)}")

    # Handle Y column specifically for NA-like strings before dropna
    # Convert common NA strings in Y to actual np.nan for consistent dropping
    if pd.api.types.is_string_dtype(model_df[y_col].dtype) or pd.api.types.is_object_dtype(model_df[y_col].dtype):
        missing_strings_map = {'n/a': np.nan, 'nan': np.nan, '<na>': np.nan, '': np.nan, 'none': np.nan, 'null': np.nan}
        model_df[y_col] = model_df[y_col].astype(str).str.lower().replace(missing_strings_map)
        # Attempt numeric conversion for Y after replacing NA strings, but keep errors as NaN
        model_df[y_col] = pd.to_numeric(model_df[y_col], errors='coerce')


    # Drop rows with NA in Y or ANY X column *after* coercions
    model_df.dropna(subset=cols_to_use, inplace=True)
    cleaned_rows = len(model_df)

    if cleaned_rows == 0:
        raise ValueError(f"No data remaining after removing rows with missing values in Y ('{y_col}') or X ({', '.join(x_cols)}). Original rows: {initial_rows}.")

    y_clean = model_df[y_col]
    X_clean = sm.add_constant(model_df[x_cols], has_constant='add') # Add constant AFTER dropna

    return y_clean, X_clean


def fit_logistic_model(df, y_col, x_cols):
    y_clean, X_clean = _prepare_data_for_model(df, y_col, x_cols)

    y_numeric = None
    dtype = y_clean.dtype

    # Convert cleaned Y (which is now numeric or boolean after _prepare) to 0/1 integer
    if pd.api.types.is_bool_dtype(dtype):
        y_numeric = y_clean.astype(int)
    elif pd.api.types.is_integer_dtype(dtype) and y_clean.isin([0, 1]).all():
        y_numeric = y_clean.astype(int)
    elif pd.api.types.is_float_dtype(dtype):
        is_binary_float = np.isclose(y_clean, 0) | np.isclose(y_clean, 1)
        if is_binary_float.all():
             y_numeric = y_clean.round().astype(int) # Round might be safer than just astype(int)
        else:
            raise ValueError(f"Logistic regression target column '{y_col}' has non-binary float values after cleaning: {y_clean[~is_binary_float].unique()[:5]}")
    else:
         # Should have been caught by determine_model_type or _prepare, but raise error
         raise ValueError(f"Logistic regression target column '{y_col}' has unsuitable type ({dtype}) or values after cleaning.")

    if y_numeric.nunique() < 2:
         raise ValueError(f"Target variable '{y_col}' has only one unique value ({y_numeric.unique()}) after cleaning. Cannot fit logistic model.")

    model = sm.Logit(y_numeric, X_clean)
    results = model.fit(maxiter=100) # Add reasonable maxiter
    return results


def fit_linear_model(df, y_col, x_cols):
    y_clean, X_clean = _prepare_data_for_model(df, y_col, x_cols)

    if not pd.api.types.is_numeric_dtype(y_clean.dtype):
         raise ValueError(f"Linear regression target column '{y_col}' must be numeric. Found dtype {y_clean.dtype} after cleaning.")

    model = sm.OLS(y_clean, X_clean)
    results = model.fit()
    return results


def create_model_summary_display(model_results):
    try:
        summary_text = model_results.summary().as_text()
        return html.Pre(summary_text, className="border rounded p-2 bg-light small", style={'fontSize': '11px'}) # Smaller font
    except Exception as e:
        print(f"Error generating text model summary display: {e}")
        return html.Pre(f"Could not generate model summary: {e}")


def create_logistic_roc_plot(model_results, df, y_col, x_cols):
    try:
        y_clean, X_clean = _prepare_data_for_model(df, y_col, x_cols)

        y_numeric = None
        dtype = y_clean.dtype
        if pd.api.types.is_bool_dtype(dtype): y_numeric = y_clean.astype(int)
        elif pd.api.types.is_integer_dtype(dtype) and y_clean.isin([0, 1]).all(): y_numeric = y_clean.astype(int)
        elif pd.api.types.is_float_dtype(dtype):
             is_binary_float = np.isclose(y_clean, 0) | np.isclose(y_clean, 1)
             if is_binary_float.all(): y_numeric = y_clean.round().astype(int)
             else: raise ValueError(f"Cannot create ROC plot for non-binary float Y column '{y_col}'.")
        else:
             raise ValueError(f"Y column '{y_col}' has unsuitable type ({dtype}) for ROC plot after cleaning.")

        if y_numeric is None: # Should not happen if checks above are correct
             raise RuntimeError("Failed to prepare Y variable for ROC plot.")

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