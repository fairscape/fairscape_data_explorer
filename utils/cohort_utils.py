# cohort_utils.py
import pandas as pd
import io
import base64
import traceback

def parse_uploaded_csv(contents):
    if contents is None:
        raise ValueError("No file content provided.")

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        df = None
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(io.StringIO(decoded.decode('latin-1')))
            except Exception as e_latin:
                raise ValueError(f"Error parsing uploaded CSV (latin-1 fallback failed): {e_latin}")
        except Exception as e_utf8:
             raise ValueError(f"Error parsing uploaded CSV (UTF-8 failed): {e_utf8}")

        if df is None:
             raise ValueError("Could not parse CSV content.")

        return df

    except Exception as e:
        # print(f"--- ERROR: Exception in parse_uploaded_csv: {e}") # Keep error logging if desired
        # print(traceback.format_exc())
        raise ValueError(f"Error parsing uploaded file: {e}")


def apply_rules_to_dataframe(df, rules, cohort_name):
    if not rules:
        return pd.Series([False] * len(df), index=df.index, name=cohort_name)
    if df.empty:
         return pd.Series(dtype=bool, name=cohort_name)

    combined_mask = pd.Series(True, index=df.index)

    for i, rule in enumerate(rules):
        col = rule.get('column')
        op = rule.get('op')
        val1 = rule.get('value1')
        val2 = rule.get('value2')

        if not col or not op:
             continue

        if col not in df.columns:
            raise ValueError(f"Rule column '{col}' not found in the data.")

        series = df[col]
        mask = pd.Series(False, index=df.index)

        try:
            is_numeric_op = op in ['>', '<', '>=', '<=', 'between']
            is_equality_op = op in ['=', '!=']
            is_string_op = op in ['contains', 'starts_with', 'ends_with']

            if is_equality_op:
                try:
                    val1_converted = pd.Series([val1], dtype=series.dtype).iloc[0]
                    if op == '=': mask = (series == val1_converted)
                    else: mask = (series != val1_converted)
                except Exception:
                    mask = series.astype(str).str.fullmatch(str(val1), na=False) if op == '=' else ~series.astype(str).str.fullmatch(str(val1), na=True)

            elif is_numeric_op:
                series_numeric = pd.to_numeric(series, errors='coerce')
                val1_numeric = pd.to_numeric(val1, errors='coerce')

                if pd.isna(val1_numeric):
                    mask = pd.Series(False, index=df.index)
                else:
                    if op == '>': mask = (series_numeric > val1_numeric)
                    elif op == '<': mask = (series_numeric < val1_numeric)
                    elif op == '>=': mask = (series_numeric >= val1_numeric)
                    elif op == '<=': mask = (series_numeric <= val1_numeric)
                    elif op == 'between':
                        val2_numeric = pd.to_numeric(val2, errors='coerce')
                        if pd.isna(val2_numeric):
                             mask = pd.Series(False, index=df.index)
                        else:
                            lower = min(val1_numeric, val2_numeric)
                            upper = max(val1_numeric, val2_numeric)
                            mask = (series_numeric >= lower) & (series_numeric <= upper)

            elif is_string_op:
                 series_str = series.astype(str).fillna('')
                 val1_str = str(val1) if val1 is not None else ''
                 if op == 'contains': mask = series_str.str.contains(val1_str, case=False, na=False)
                 # Add other string ops if needed

            else:
                continue

            mask = mask.fillna(False).astype(bool)
            combined_mask &= mask

        except Exception as e:
             # print(f"--- ERROR: Failed to apply rule {i+1}: {e}") # Keep error logging if desired
             # print(traceback.format_exc())
             combined_mask = pd.Series(False, index=df.index)
             break

    return combined_mask.rename(cohort_name)


def join_cohort_data(main_df, cohort_df, main_join_col, cohort_join_col, cohort_assignment_col):
    if main_df is None or main_df.empty:
        raise ValueError("Main data is missing or empty.")
    if cohort_df is None or cohort_df.empty:
        raise ValueError("Uploaded cohort data is missing or empty.")

    if main_join_col not in main_df.columns:
        raise ValueError(f"Main join column '{main_join_col}' not found in main data.")
    if cohort_join_col not in cohort_df.columns:
        raise ValueError(f"Cohort join column '{cohort_join_col}' not found in uploaded cohort data.")
    if cohort_assignment_col not in cohort_df.columns:
        raise ValueError(f"Cohort assignment column '{cohort_assignment_col}' not found in uploaded cohort data.")

    try:
        main_df_copy = main_df.copy()
        cohort_subset = cohort_df[[cohort_join_col, cohort_assignment_col]].copy()

        main_df_copy[main_join_col] = main_df_copy[main_join_col].astype(str).str.strip()
        cohort_subset[cohort_join_col] = cohort_subset[cohort_join_col].astype(str).str.strip()

        cohort_subset.drop_duplicates(subset=[cohort_join_col], inplace=True)

        merged_df = pd.merge(
            main_df_copy,
            cohort_subset,
            left_on=main_join_col,
            right_on=cohort_join_col,
            how='left',
            suffixes=('', '_cohort_upload')
        )

        potential_new_col_name = cohort_assignment_col
        if cohort_assignment_col in main_df.columns and cohort_assignment_col != cohort_join_col:
             potential_new_col_name = f"{cohort_assignment_col}_cohort_upload"

        if potential_new_col_name not in merged_df.columns:
             if cohort_assignment_col in merged_df.columns:
                 potential_new_col_name = cohort_assignment_col
             else:
                raise ValueError(f"Merge failed - cohort assignment column '{cohort_assignment_col}' (or suffixed version) missing after merge.")

        final_cohort_col_name = f"cohort_{cohort_assignment_col}"
        temp_suffix = "_upload"
        while final_cohort_col_name in main_df.columns:
            final_cohort_col_name += temp_suffix

        merged_df.rename(columns={potential_new_col_name: final_cohort_col_name}, inplace=True)

        if main_join_col != cohort_join_col and cohort_join_col in merged_df.columns:
            merged_df.drop(columns=[cohort_join_col], inplace=True)

        # Convert the final joined column to string and handle NAs
        merged_df[final_cohort_col_name] = merged_df[final_cohort_col_name].astype(str)
        # Fill NAs resulting from merge and replace common missing string representations
        merged_df[final_cohort_col_name].fillna('N/A', inplace=True)
        merged_df[final_cohort_col_name].replace({'nan': 'N/A', '<NA>': 'N/A', 'None': 'N/A', '': 'N/A'}, inplace=True)

        # Skipping dtype restoration for simplicity

        return merged_df, final_cohort_col_name

    except Exception as e:
        # print(f"--- ERROR: Exception in join_cohort_data: {e}") # Keep error logging if desired
        # print(traceback.format_exc())
        raise