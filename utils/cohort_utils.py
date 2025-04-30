# cohort_utils.py
import pandas as pd
import io
import base64

def parse_uploaded_csv(contents):
    """Parses the content string from dcc.Upload component for a CSV."""
    if contents is None:
        raise ValueError("No file content provided.")

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        # Try UTF-8 first
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            df = pd.read_csv(io.StringIO(decoded.decode('latin-1')))
        return df
    except Exception as e:
        raise ValueError(f"Error parsing uploaded CSV: {e}")

def apply_rules_to_dataframe(df, rules, cohort_name):
    """
    Applies a set of rules (AND logic) to filter a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        rules (list): A list of rule dictionaries, e.g.,
                      [{'column': 'age', 'op': '>', 'value1': 30, 'value2': None}, ...]
        cohort_name (str): The base name for the resulting boolean column.

    Returns:
        pd.Series: A boolean Series indicating which rows match all rules.
                   The Series name will be `cohort_name`.
    """
    if not rules:
        # Return Series of False if no rules provided
        return pd.Series([False] * len(df), index=df.index, name=cohort_name)

    combined_mask = pd.Series([True] * len(df), index=df.index) # Start with all True

    for rule in rules:
        col = rule['column']
        op = rule['op']
        val1 = rule['value1']
        val2 = rule['value2'] # Used for 'between'

        if col not in df.columns:
            raise ValueError(f"Rule column '{col}' not found in the data.")

        series = df[col] # Get the column to apply rule on

        # --- Type Handling for Comparisons ---
        is_numeric_op = op in ['>', '<', '>=', '<=', 'between']
        is_equality_op = op in ['=', '!=']

        # Use original dtype for equality unless it fails
        if is_equality_op:
            try:
                if op == '=': mask = (series == val1)
                else: mask = (series != val1) # op == '!='
            except TypeError: # If direct comparison fails (e.g., int vs str), compare as strings
                if op == '=': mask = (series.astype(str) == str(val1))
                else: mask = (series.astype(str) != str(val1))
            except Exception as e: # Catch other comparison errors
                raise ValueError(f"Error comparing column '{col}' ({series.dtype}) with value '{val1}' using op '{op}': {e}")

        elif is_numeric_op:
            # For numeric operations, coerce both series and rule values to numeric.
            # Errors during coercion will result in NaN.
            series_numeric = pd.to_numeric(series, errors='coerce')
            val1_numeric = pd.to_numeric(val1, errors='coerce')
            if val1_numeric is pd.NA:
                 raise ValueError(f"Rule value '{val1}' for numeric operator '{op}' on column '{col}' is not numeric.")

            if op == '>': mask = (series_numeric > val1_numeric)
            elif op == '<': mask = (series_numeric < val1_numeric)
            elif op == '>=': mask = (series_numeric >= val1_numeric)
            elif op == '<=': mask = (series_numeric <= val1_numeric)
            elif op == 'between':
                val2_numeric = pd.to_numeric(val2, errors='coerce')
                if val2_numeric is pd.NA:
                     raise ValueError(f"Rule value 2 '{val2}' for 'between' operator on column '{col}' is not numeric.")
                # Ensure val1 is the lower bound, val2 is upper
                lower = min(val1_numeric, val2_numeric)
                upper = max(val1_numeric, val2_numeric)
                mask = (series_numeric >= lower) & (series_numeric <= upper)
        else:
            # Handle other potential operators here if added later (e.g., 'contains', 'startswith')
            # For now, assume string comparison for unhandled ops if needed, or raise error
             raise ValueError(f"Unsupported operator '{op}' used for column '{col}'.")


        # --- Important: Handle NaNs resulting from coercion or comparison ---
        # Comparisons involving NaN generally result in False.
        # We explicitly fill NaNs in the resulting mask with False to ensure clean boolean logic.
        mask.fillna(False, inplace=True)

        # Combine the mask for this rule with the overall mask using AND logic
        combined_mask &= mask

    # Return the final boolean mask with the specified cohort name
    return combined_mask.rename(cohort_name)


def join_cohort_data(main_df, cohort_df, main_join_col, cohort_join_col, cohort_assignment_col):
    """
    Joins cohort assignments from cohort_df to main_df, ensuring the resulting
    cohort column uses strings ("True", "False", original strings, or "N/A").

    Args:
        main_df (pd.DataFrame): The main data.
        cohort_df (pd.DataFrame): The dataframe with cohort assignments.
        main_join_col (str): Column name in main_df to join on.
        cohort_join_col (str): Column name in cohort_df to join on.
        cohort_assignment_col (str): Column name in cohort_df that contains the cohort labels.

    Returns:
        tuple: (pd.DataFrame, str)
            - The main_df with an added column containing cohort assignments as strings.
            - The name of the newly added cohort column.
    """
    if main_join_col not in main_df.columns:
        raise ValueError(f"Main join column '{main_join_col}' not found in main data.")
    if cohort_join_col not in cohort_df.columns:
        raise ValueError(f"Cohort join column '{cohort_join_col}' not found in uploaded cohort data.")
    if cohort_assignment_col not in cohort_df.columns:
        raise ValueError(f"Cohort assignment column '{cohort_assignment_col}' not found in uploaded cohort data.")

    # Prepare cohort subset: select join and assignment columns, drop duplicates on join key
    # Convert join keys to string upfront to avoid merge issues with mixed types
    cohort_subset = cohort_df[[cohort_join_col, cohort_assignment_col]].astype({cohort_join_col: str}).copy()
    cohort_subset.drop_duplicates(subset=[cohort_join_col], inplace=True)

    # Prepare main_df join key as string
    main_df_copy = main_df.copy()
    main_df_copy[main_join_col] = main_df_copy[main_join_col].astype(str)

    # Perform the left merge
    merged_df = pd.merge(
        main_df_copy,
        cohort_subset,
        left_on=main_join_col,
        right_on=cohort_join_col,
        how='left'
    )

    # Define the new cohort column name (prevent conflicts)
    new_cohort_col_name = f"cohort_{cohort_assignment_col}"
    # Handle potential name collisions if 'cohort_assignment_col' was already prefixed
    temp_suffix = "_upload"
    while new_cohort_col_name in main_df.columns:
         new_cohort_col_name += temp_suffix

    # Rename the merged assignment column
    if cohort_assignment_col in merged_df.columns:
        merged_df.rename(columns={cohort_assignment_col: new_cohort_col_name}, inplace=True)
    else:
        # This case should not happen with a successful merge
        raise ValueError("Merge failed - cohort assignment column missing after merge.")

    # Drop the potentially redundant join column from the cohort_df if names differed
    if main_join_col != cohort_join_col and cohort_join_col in merged_df.columns:
        merged_df.drop(columns=[cohort_join_col], inplace=True)

    # --- Convert cohort column to String Type ("True"/"False"/"N/A"/Original) ---
    target_col = merged_df[new_cohort_col_name]

    # Check if the original column *looked* boolean (before merge potentially converted it)
    original_assignment_series = cohort_df[cohort_assignment_col].dropna()
    looks_boolean = False
    if pd.api.types.is_bool_dtype(original_assignment_series.dtype):
        looks_boolean = True
    elif pd.api.types.is_numeric_dtype(original_assignment_series.dtype):
        if set(original_assignment_series.unique()) <= {0, 1}:
             looks_boolean = True
    elif pd.api.types.is_object_dtype(original_assignment_series.dtype) or pd.api.types.is_string_dtype(original_assignment_series.dtype):
         unique_vals_lower = set(original_assignment_series.astype(str).str.lower().unique())
         if unique_vals_lower <= {'true', 'false', 't', 'f', 'yes', 'no', 'y', 'n', '1', '0'}:
              looks_boolean = True

    if looks_boolean:
        # Map boolean-like values explicitly to "True" / "False" strings
        bool_map = {'true': "True", 't': "True", '1': "True", 'yes': "True", 'y': "True", 1: "True", True: "True",
                    'false': "False", 'f': "False", '0': "False", 'no': "False", 'n': "False", 0: "False", False: "False"}
        # Apply map robustly: convert to string, lower, then map
        merged_df[new_cohort_col_name] = target_col.astype(str).str.lower().map(bool_map)
    else:
        # Otherwise, just convert the column to string type
        merged_df[new_cohort_col_name] = target_col.astype(str)

    # Fill any remaining NaNs (rows in main_df without a match) with 'N/A' string
    merged_df[new_cohort_col_name].fillna('N/A', inplace=True)
    # Replace any 'nan' strings possibly introduced during conversion
    merged_df[new_cohort_col_name].replace({'nan': 'N/A', '<NA>': 'N/A', 'None':'N/A'}, inplace=True)


    # Restore original data types for columns other than the new cohort column and the join key
    for col in main_df.columns:
        if col != main_join_col: # Don't restore join key if it was converted
             original_dtype = main_df[col].dtype
             try:
                 # Avoid converting Int64 back if it's now float due to merge NAs
                 if pd.api.types.is_integer_dtype(original_dtype) and pd.api.types.is_float_dtype(merged_df[col].dtype):
                     # Try converting back to nullable Int, otherwise leave as float
                      merged_df[col] = merged_df[col].astype(pd.Int64Dtype())
                 elif merged_df[col].dtype != original_dtype:
                     merged_df[col] = merged_df[col].astype(original_dtype)
             except Exception as e:
                 print(f"Warning: Could not restore dtype for column '{col}'. Error: {e}")


    return merged_df, new_cohort_col_name