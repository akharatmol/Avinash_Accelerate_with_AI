import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

def clean_bronze_csv_v2(file_name: str, bronze_dir: str = "data/1_bronze", silver_dir: str = "data/2_silver") -> Dict[str, Any]:
    """
    Loads a bronze CSV, applies cleaning, logs every action,
    saves the result to Silver, and provides a before/after diff.
    """
    bronze_path = Path(bronze_dir) / file_name
    if not bronze_path.exists():
        raise FileNotFoundError(f"Bronze CSV not found: {bronze_path}")

    df = pd.read_csv(bronze_path)
    before_df = df.copy(deep=True) 
    after_df = df.copy()

    audit_log = []

    # 1. Standardize Column Names 
    original_columns = list(after_df.columns)
    cleaned_columns = [str(col).strip().lower().replace(" ", "_") for col in original_columns]
    if original_columns != cleaned_columns:
        after_df.columns = cleaned_columns
        audit_log.append(f"Standardized {len(cleaned_columns)} column name(s) (lowercased, stripped whitespace, replaced spaces with underscores).")

    # 2. Trim Whitespace from all String Columns 
    str_cols = after_df.select_dtypes(include=['object']).columns
    trimmed_cells = 0
    if len(str_cols) > 0:
        for col in str_cols:
            original_series = after_df[col]
            after_df[col] = after_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            trimmed_cells += (original_series != after_df[col]).sum()
        if trimmed_cells > 0:
            audit_log.append(f"Applied TRIM() to remove leading/trailing whitespace from {trimmed_cells:,} text cell(s).")

    # 3. Handle Missing Values Safely 
    for col in after_df.columns:
        if after_df[col].isna().any():
            null_count = after_df[col].isna().sum()
            if pd.api.types.is_numeric_dtype(after_df[col]):
                after_df[col] = after_df[col].fillna(0)
                audit_log.append(f"Filled {null_count} missing numeric value(s) in '{col}' with 0.")
            else:
                after_df[col] = after_df[col].fillna("Unknown")
                audit_log.append(f"Filled {null_count} missing text/other value(s) in '{col}' with 'Unknown'.")

    # 4. Final Audit Log Check (Forces a message if clean)
    if not audit_log:
        audit_log.append("Data was already perfectly clean; no transformations were necessary.")

    # 5. Save the Cleaned Dataframe 
    silver_path = Path(silver_dir)
    silver_path.mkdir(parents=True, exist_ok=True)
    output_path = silver_path / file_name
    after_df.to_csv(output_path, index=False)

    # 6. Create a Secure Before/After Diff
    comparison_rows = []
    if not before_df.empty:
        col_map = dict(zip(original_columns, cleaned_columns))
        row_index = 0
        before_row = before_df.iloc[row_index]
        after_row = after_df.iloc[row_index]
        
        for orig_col, clean_col in col_map.items():
            before_val = before_row.get(orig_col)
            after_val = after_row.get(clean_col)
            
            before_str = str(before_val) if pd.notna(before_val) else "Null"
            after_str = str(after_val) if pd.notna(after_val) else "Null"
            
            if (orig_col != clean_col) or (before_str != after_str):
                label = orig_col if orig_col == clean_col else f"{orig_col} ➔ {clean_col}"
                comparison_rows.append({
                    "Column Name": label,
                    "Bronze Value (Before)": before_str,
                    "Silver Value (After)": after_str
                })

    comparison_df = pd.DataFrame(comparison_rows)

    return {
        "audit_log": audit_log,
        "comparison_dataframe": comparison_df,
    }
