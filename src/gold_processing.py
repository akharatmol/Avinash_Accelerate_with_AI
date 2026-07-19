from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Infer date, numeric, and categorical columns dynamically."""
    date_columns: List[str] = []
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []

    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue

        try:
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().sum() > 0:
                date_columns.append(column)
            else:
                categorical_columns.append(column)
        except Exception:
            categorical_columns.append(column)

    return {
        "date_columns": date_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


def build_business_summary(
    file_name: str,
    silver_dir: str = "data/2_silver",
    gold_dir: str = "data/3_gold",
) -> Dict[str, Any]:
    """Load a silver CSV, infer column types, aggregate a business-level summary, and save it to gold."""
    silver_path = Path(silver_dir) / file_name
    if not silver_path.exists():
        raise FileNotFoundError(f"Silver CSV not found: {silver_path}")

    df = pd.read_csv(silver_path)
    inferred_types = detect_column_types(df)

    numeric_columns = inferred_types["numeric_columns"]
    categorical_columns = inferred_types["categorical_columns"]
    date_columns = inferred_types["date_columns"]

    primary_numeric_column = numeric_columns[0] if numeric_columns else None
    grouping_column: Optional[str] = None
    if categorical_columns:
        grouping_column = max(
            categorical_columns,
            key=lambda col: (df[col].dropna().count(), df[col].dropna().nunique(dropna=True)),
        )
    elif date_columns:
        grouping_column = date_columns[0]

    if primary_numeric_column is not None:
        if grouping_column is not None:
            summary_df = (
                df.groupby(grouping_column, dropna=False)[primary_numeric_column]
                .agg(sum="sum", average="mean", count="count")
                .reset_index()
            )
            summary_df = summary_df.sort_values(["sum", "average"], ascending=[False, False]).reset_index(drop=True)
        else:
            summary_df = pd.DataFrame(
                [
                    {
                        "metric": "total",
                        "sum": float(df[primary_numeric_column].sum()),
                        "average": float(df[primary_numeric_column].mean()),
                        "count": int(df[primary_numeric_column].count()),
                    }
                ]
            )
    else:
        if grouping_column is not None:
            summary_df = (
                df[grouping_column]
                .dropna()
                .value_counts()
                .rename_axis(grouping_column)
                .reset_index(name="row_count")
            )
            summary_df = summary_df.sort_values("row_count", ascending=False).head(10).reset_index(drop=True)
        else:
            summary_df = pd.DataFrame([{"category": "all", "row_count": int(len(df))}])

    gold_path = Path(gold_dir)
    gold_path.mkdir(parents=True, exist_ok=True)
    output_path = gold_path / f"{Path(file_name).stem}_summary.csv"
    summary_df.to_csv(output_path, index=False)

    summary_metrics = {
        "source_file": file_name,
        "silver_path": str(silver_path),
        "gold_summary_path": str(output_path),
        "date_columns": date_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "primary_numeric_column": primary_numeric_column,
        "grouping_column": grouping_column,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "total_sum": float(df[primary_numeric_column].sum()) if primary_numeric_column is not None else None,
        "average_value": float(df[primary_numeric_column].mean()) if primary_numeric_column is not None else None,
    }

    return {
        "summary_metrics": summary_metrics,
        "summary_dataframe": summary_df,
        "source_dataframe": df,
    }


def build_plotly_charts(
    df: pd.DataFrame,
    summary_df: Optional[pd.DataFrame] = None,
    primary_numeric_column: Optional[str] = None,
    grouping_column: Optional[str] = None,
    date_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Create Plotly bar, line, and treemap charts for display in Streamlit."""
    if summary_df is None:
        summary_result = build_business_summary(Path(df.name).name if hasattr(df, "name") else "data.csv")
        summary_df = summary_result["summary_dataframe"]
        primary_numeric_column = summary_result["summary_metrics"].get("primary_numeric_column")
        grouping_column = summary_result["summary_metrics"].get("grouping_column")
        date_column = summary_result["summary_metrics"].get("date_columns", [None])[0] if summary_result["summary_metrics"].get("date_columns") else None

    metrics: Dict[str, Any] = {}
    bar_chart = None
    line_chart = None
    treemap_chart = None

    if grouping_column is not None and summary_df is not None and grouping_column in summary_df.columns:
        chart_df = summary_df.head(10).copy()
        if "sum" in chart_df.columns:
            y_column = "sum"
        elif "average" in chart_df.columns:
            y_column = "average"
        elif "row_count" in chart_df.columns:
            y_column = "row_count"
        else:
            y_column = chart_df.columns[-1]

        bar_chart = px.bar(
            chart_df,
            x=grouping_column,
            y=y_column,
            text=y_column,
            title=f"Top categories in {grouping_column} by {primary_numeric_column or y_column}",
        )
        metrics["top_categories"] = chart_df[[grouping_column, y_column]].to_dict(orient="records")

    if date_column is not None and primary_numeric_column is not None:
        trend_df = (
            df.groupby(date_column, dropna=False)[primary_numeric_column]
            .agg(["sum", "mean", "count"])
            .reset_index()
        )
        trend_df.columns = [date_column, "sum", "average", "count"]
        trend_df = trend_df.sort_values(date_column).reset_index(drop=True)
        line_chart = px.line(
            trend_df,
            x=date_column,
            y="sum",
            markers=True,
            title=f"Trend of {primary_numeric_column} over {date_column}",
        )

    location_columns = [col for col in ["region", "state", "city"] if col in df.columns]
    if location_columns:
        location_df = df[location_columns].copy()
        location_df = location_df.dropna(how="all")
        if len(location_columns) >= 2:
            location_df["row_count"] = 1
            location_df = location_df.groupby(location_columns, dropna=False).size().reset_index(name="row_count")
            if len(location_columns) >= 3:
                treemap_chart = px.treemap(
                    location_df,
                    path=[px.Constant("All"), "region", "state", "city"],
                    values="row_count",
                    title="Geographic distribution",
                )
            else:
                treemap_chart = px.treemap(
                    location_df,
                    path=[px.Constant("All"), "region", "state"],
                    values="row_count",
                    title="Geographic distribution",
                )

    return {
        "summary_metrics": metrics,
        "bar_chart": bar_chart,
        "treemap_chart": treemap_chart,
        "line_chart": line_chart,
    }
