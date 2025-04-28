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
        # Assume UTF-8 encoding, provide option to change if needed later
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
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
        cohort_name (str): The name for the resulting boolean column.

    Returns:
        pd.Series: A boolean Series indicating which rows match all rules.
    """
    if not rules:
        return pd.Series([False] * len(df), index=df.index, name=cohort_name) # No rules means no match

    combined_mask = pd.Series([True] * len(df), index=df.index)

    for rule in rules:
        col = rule['column']
        op = rule['op']
        val1 = rule['value1']
        val2 = rule['value2'] # Used for 'between'

        if col not in df.columns:
            raise ValueError(f"Rule column '{col}' not found in the data.")

        # Ensure column is numeric for numeric operations
        is_numeric_op = op in ['>', '<', '>=', '<=', 'between']
        if is_numeric_op:
            try:
                # Attempt conversion, coercing errors to NaN
                series = pd.to_numeric(df[col], errors='coerce')
                # Convert rule values to numeric as well
                val1 = pd.to_numeric(val1, errors='coerce')
                if val2 is not None:
                    val2 = pd.to_numeric(val2, errors='coerce')
            except Exception as e:
                 raise ValueError(f"Cannot apply numeric operator '{op}' to non-numeric rule value or column '{col}'. Error: {e}")
        else:
            series = df[col] # Use original series for string/equality checks

        # Apply mask based on operator
        mask = pd.Series([False] * len(df), index=df.index) # Default to false
        try:
            if op == '=':
                # Try numeric comparison first if possible, else string
                try:
                    mask = (series == val1)
                except TypeError: # Handle potential type mismatches
                     mask = (series.astype(str) == str(val1))
            elif op == '!=':
                 try:
                    mask = (series != val1)
                 except TypeError:
                     mask = (series.astype(str) != str(val1))
            elif op == '>':
                mask = (series > val1)
            elif op == '<':
                mask = (series < val1)
            elif op == '>=':
                mask = (series >= val1)
            elif op == '<=':
                mask = (series <= val1)
            elif op == 'between':
                if val1 is None or val2 is None:
                    raise ValueError("Both values required for 'between' operator.")
                 # Ensure val1 is the lower bound
                lower, upper = min(val1, val2), max(val1, val2)
                mask = (series >= lower) & (series <= upper)
            # 'in list' might be added later if needed, requires different value parsing
            # elif op == 'in list':
            #     values_list = [v.strip() for v in str(val1).split(',')]
            #     mask = series.isin(values_list)

            # Handle NaNs: comparisons with NaN usually result in False, which is often desired.
            # If specific NaN handling is needed, it can be added here.
            # For example, if '!=' NaN should be True: mask = mask | series.isna() if op == '!=' else mask
            mask.fillna(False, inplace=True) # Ensure NaNs in result become False

        except Exception as e:
            raise ValueError(f"Error applying rule ({col} {op} {val1}): {e}")

        combined_mask &= mask

    return combined_mask.rename(cohort_name)


def join_cohort_data(main_df, cohort_df, main_join_col, cohort_join_col, cohort_assignment_col):
    """
    Joins cohort assignments from cohort_df to main_df.

    Args:
        main_df (pd.DataFrame): The main data.
        cohort_df (pd.DataFrame): The dataframe with cohort assignments.
        main_join_col (str): Column name in main_df to join on.
        cohort_join_col (str): Column name in cohort_df to join on.
        cohort_assignment_col (str): Column name in cohort_df that contains the cohort labels.

    Returns:
        pd.DataFrame: main_df with an added column containing cohort assignments.
                      The new column name is derived from cohort_assignment_col.
    """
    if main_join_col not in main_df.columns:
        raise ValueError(f"Main join column '{main_join_col}' not found in main data.")
    if cohort_join_col not in cohort_df.columns:
        raise ValueError(f"Cohort join column '{cohort_join_col}' not found in uploaded cohort data.")
    if cohort_assignment_col not in cohort_df.columns:
        raise ValueError(f"Cohort assignment column '{cohort_assignment_col}' not found in uploaded cohort data.")

    # Select only the necessary columns from cohort_df and drop duplicates based on the join key
    cohort_subset = cohort_df[[cohort_join_col, cohort_assignment_col]].drop_duplicates(subset=[cohort_join_col])

    # Perform the merge
    merged_df = pd.merge(
        main_df,
        cohort_subset,
        left_on=main_join_col,
        right_on=cohort_join_col,
        how='left' # Keep all rows from main_df
    )

    # The merge might duplicate the join columns if names differ; we only need the assignment.
    # The cohort assignment column from cohort_df is now in merged_df.
    # We might want to rename it to avoid conflicts if the main_df already had that name.
    new_cohort_col_name = f"cohort_{cohort_assignment_col}" # Create a distinct name
    if cohort_assignment_col in merged_df.columns:
         merged_df.rename(columns={cohort_assignment_col: new_cohort_col_name}, inplace=True)
    else:
        # This case should not happen with standard pandas merge, but handle defensively
        raise Exception("Merge failed unexpectedly - cohort assignment column missing.")

    # Drop the potentially redundant join column from the cohort_df if names were different
    if main_join_col != cohort_join_col and cohort_join_col in merged_df.columns:
        merged_df.drop(columns=[cohort_join_col], inplace=True)

    # Fill NaNs in the new cohort column (for main_df rows that didn't have a match)
    # with a specific category like 'No Cohort Assigned' or keep as NaN.
    merged_df[new_cohort_col_name] = merged_df[new_cohort_col_name].fillna('N/A') # Or pd.NA

    return merged_df