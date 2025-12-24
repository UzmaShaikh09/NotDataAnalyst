import os
import json
import pandas as pd
import sqlalchemy
import re
from typing import Optional, Dict, Any, List
from pathlib import Path

# ==========================================================
# 📊 DATA ANALYSIS TOOLS
# ==========================================================

def analyze_df(df: pd.DataFrame):
    """
    Produce a concise Markdown summary of a DataFrame for LLM consumption.
    """
    lines = []
    lines.append("## Table Summary")
    lines.append("")
    lines.append(f"- Rows: {len(df)}")
    lines.append(f"- Columns: {len(df.columns)}")
    lines.append(f"- Index is unique: {bool(df.index.is_unique)}")
    lines.append(f"- Duplicate rows: {int(df.duplicated().sum())}")
    lines.append("")

    lines.append("### Columns")
    lines.append("")
    for col in df.columns:
        ser = df[col]
        dtype = str(ser.dtype)
        null_count = int(ser.isna().sum())
        
        if pd.api.types.is_numeric_dtype(ser):
            mn = ser.min()
            mx = ser.max()
            sample_vals = ser.dropna().unique()[:5].tolist()
            lines.append(f"- `{col}` — {dtype}, nulls={null_count}, min={mn}, max={mx}, sample={sample_vals}")
        else:
            vc = ser.value_counts(dropna=True)
            top = vc.index[0] if len(vc) else None
            freq = int(vc.iloc[0]) if len(vc) else None
            sample_vals = ser.dropna().unique()[:5].tolist()
            lines.append(f"- `{col}` — {dtype}, nulls={null_count}, top={top} ({freq}), sample={sample_vals}")
    lines.append("")

    # Head Preview
    try:
        head_csv = df.head().to_csv(index=False)
        lines.append("### Head Preview (CSV)")
        lines.append("```")
        lines.append(head_csv.strip())
        lines.append("```")
        lines.append("")
    except Exception:
        pass

    return "\n".join(lines)


def get_data_context(source):
    """
    Ingests a source (File Path, URL, Folder, DB String) and returns 
    a Markdown string analyzing the data found.
    """
    results = {}
    pieces = []

    # CASE 1: URL
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        ext = source.split(".")[-1].lower()
        if ext == "csv":
            return {"file": analyze_df(pd.read_csv(source))}
        if ext in ("xlsx", "xls"):
            sheets = pd.read_excel(source, sheet_name=None)
            return {name: analyze_df(df) for name, df in sheets.items()}
        if ext == "parquet":
            return {"file": analyze_df(pd.read_parquet(source))}

    # CASE 2: FILE PATH
    if os.path.isfile(source):
        ext = source.split(".")[-1].lower()
        filename = os.path.basename(source)
        
        if ext == "csv":
            txt = analyze_df(pd.read_csv(source))
            pieces.append(f"### {filename}\n" + txt)
            results[filename] = txt
        elif ext in ("xlsx", "xls"):
            sheets = pd.read_excel(source, sheet_name=None)
            for s, df in sheets.items():
                txt = analyze_df(df)
                results[f"{filename}::{s}"] = txt
                pieces.append(f"### {filename}::{s}\n" + txt)
        elif ext == "parquet":
            txt = analyze_df(pd.read_parquet(source))
            pieces.append(f"### {filename}\n" + txt)
            results[filename] = txt
            
        if pieces:
            return "\n\n".join(pieces)

    # CASE 3: FOLDER
    if os.path.isdir(source):
        for file in os.listdir(source):
            path = os.path.join(source, file)
            if file.endswith(".csv"):
                txt = analyze_df(pd.read_csv(path))
                pieces.append(f"### {file}\n" + txt)
            elif file.endswith((".xlsx", ".xls")):
                sheets = pd.read_excel(path, sheet_name=None)
                for s, df in sheets.items():
                    txt = analyze_df(df)
                    pieces.append(f"### {file}::{s}\n" + txt)
            elif file.endswith(".parquet"):
                txt = analyze_df(pd.read_parquet(path))
                pieces.append(f"### {file}\n" + txt)
        if pieces:
            return "\n\n".join(pieces)

    # CASE 4: DATABASE
    try:
        engine = sqlalchemy.create_engine(source)
        insp = sqlalchemy.inspect(engine)
        tables = insp.get_table_names()
        for table in tables:
            df = pd.read_sql(f"SELECT * FROM {table}", engine)
            txt = analyze_df(df)
            pieces.append(f"### {table}\n" + txt)
        if pieces:
            return "\n\n".join(pieces)
    except:
        pass

    raise ValueError("Unknown source type. Not a URL, file, folder, or valid DB string.")