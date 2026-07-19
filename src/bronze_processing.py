from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

def profile_uploaded_csv(uploaded_file, save_dir: str = "data/1_bronze") -> Dict[str, Any]:
    """Save an uploaded CSV to the bronze directory and build a basic data profile."""
    if uploaded_file is None:
        raise ValueError("uploaded_file must not be None")
        
    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = target_dir / uploaded_file.name
    raw_bytes = uploaded_file.getvalue()
    raw_path.write_bytes(raw_bytes)
    
    df = pd.read_csv(raw_path)
    
    # Capture original data types and missing values
    inferred_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_values = {col: int(count) for col, count in df.isna().sum().items()}
    
    def check_basic_format_consistency(series: pd.Series) -> Dict[str, Any]:
        checks: Dict[str, Any] = {}
        non_null = series.dropna()
        if non_null.empty:
            checks["empty"] = True
            return checks
            
        checks["numeric_like"] = bool(pd.api.types.is_numeric_dtype(non_null))
        checks["datetime_like"] = bool(pd.api.types.is_datetime64_any_dtype(non_null))
        checks["date_format_like"] = bool(non_null.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").any())
        
        return checks

    format_consistency = {
        col: check_basic_format_consistency(df[col]) for col in df.columns
    }
    
    profile = {
        "source_file": uploaded_file.name,
        "saved_path": str(raw_path),
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "columns_with_inferred_dtypes": inferred_dtypes,
        "missing_values_per_column": missing_values,
        "basic_format_consistency_checks": format_consistency,
    }
    
    return profile

def collect_bronze_profiles(uploaded_files: List[Any], save_dir: str = "data/1_bronze") -> Dict[str, Any]:
    """Profile each uploaded file and return a simple bundle of profiles."""
    if not uploaded_files:
        raise ValueError("uploaded_files must not be empty")

    profiles: List[Dict[str, Any]] = []

    for uploaded_file in uploaded_files:
        profile = profile_uploaded_csv(uploaded_file, save_dir=save_dir)
        profiles.append(profile)

    return {
        "profiles": profiles,
        "source_files": [p["source_file"] for p in profiles]
    }
