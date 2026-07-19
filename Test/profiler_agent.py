import io
import json
import os
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    data_bytes = uploaded_file.getvalue()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data_bytes))

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data_bytes))

    if name.endswith(".json"):
        payload = json.loads(data_bytes.decode("utf-8"))
        if isinstance(payload, dict):
            if isinstance(payload.get("records"), list):
                return pd.DataFrame(payload["records"])
            return pd.DataFrame([payload])
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        raise ValueError("Unsupported JSON structure")

    raise ValueError("Unsupported file type. Upload a CSV, Excel, or JSON file.")


def build_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def build_missing_summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    missing_counts = df.isna().sum()
    total_rows = len(df)
    for col, count in missing_counts.items():
        pct = round((count / total_rows) * 100, 2) if total_rows else 0.0
        rows.append({"column": col, "missing_count": int(count), "missing_percentage": pct})
    return rows


def build_duplicate_summary(df: pd.DataFrame) -> Dict[str, Any]:
    duplicated = df.duplicated(keep=False)
    return {
        "duplicate_rows": int(duplicated.sum()),
        "duplicate_examples": df[duplicated].head(5).to_dict(orient="records"),
    }


def generate_profile(uploaded_file) -> Dict[str, Any]:
    df = read_uploaded_file(uploaded_file)
    basic_stats = build_basic_stats(df)
    missing_summary = build_missing_summary(df)
    duplicate_summary = build_duplicate_summary(df)

    client = OpenAI(
        base_url=os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com"),
        api_key=os.getenv("GITHUB_TOKEN"),
    )
    model = os.getenv("GITHUB_MODEL", "gpt-4.1-mini")

    prompt = f"""
You are a senior data analyst. Review the dataset and provide a concise profiling summary.

Basic stats:
- Rows: {basic_stats['rows']}
- Columns: {basic_stats['columns']}
- Columns: {', '.join(basic_stats['column_names'])}
- Types: {json.dumps(basic_stats['dtypes'], indent=2)}

Missing values:
{json.dumps(missing_summary, indent=2)}

Duplicate rows:
{json.dumps(duplicate_summary, indent=2)}

Return 4 short bullet points covering:
1. A key insight
2. A likely trend or pattern
3. A likely anomaly or data quality issue
4. A practical recommendation
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )

    return {
        "basic_stats": basic_stats,
        "missing_summary": missing_summary,
        "duplicate_summary": duplicate_summary,
        "ai_insights": response.choices[0].message.content.strip(),
    }
