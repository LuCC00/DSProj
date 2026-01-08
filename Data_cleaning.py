#!/usr/bin/env python3


from __future__ import annotations

import re
import numpy as np
import pandas as pd



# Config

INPUT_XLSX = "Table.xlsx"

COL_METADATA_USED = "Metadata used?"
COL_METADATA_TYPES = "Metadata types used"
COL_TIME_WINDOW = "Study time window"
COL_CONCLUSION = "Conclusion"
COL_DS_METHODS = "DS Methods"
COL_CV_TASKS = "Computer vision task(s)"
COL_DATASET_SIZE = "Dataset size (images)"
COL_COLLECTION_MODE = "Data collection mode"


# Helpers

def norm_str(x) -> str:
    """Normalize cell to a trimmed string; empty/NaN -> ''."""
    if pd.isna(x):
        return ""
    return str(x).strip()


def is_blank(x) -> bool:
    return norm_str(x) == ""


def yes_no(x) -> str:
    """Normalize Yes/No strings; returns 'Yes', 'No', or original trimmed."""
    s = norm_str(x).lower()
    if s in {"yes", "y", "true"}:
        return "Yes"
    if s in {"no", "n", "false"}:
        return "No"
    return norm_str(x)


def is_none_like(x) -> bool:
    """Treat None/Not applicable/blank as none-like."""
    s = norm_str(x).lower()
    return s in {"", "none", "not applicable", "na", "n/a", "null"}


# Map dataset-size bins to a numeric midpoint / lower bound for logic checks
DATASET_BIN_TO_APPROX = {
    "<100": 50,
    "100–999": 500,
    "100-999": 500,
    "1k–9,999": 5000,
    "1k-9,999": 5000,
    "10k–99,999": 50000,
    "10k-99,999": 50000,
    "100k–999,999": 500000,
    "100k-999,999": 500000,
    "≥1M": 1_000_000,
    ">=1M": 1_000_000,
    "Not reported": np.nan,
    "Not mentioned": np.nan,
}


def approx_dataset_size(value: str) -> float:
    """Convert dataset size bin to approx numeric; unknown -> NaN."""
    v = norm_str(value)
    if v in DATASET_BIN_TO_APPROX:
        return DATASET_BIN_TO_APPROX[v]
    # Recognize raw numbers
    s = v.lower().replace(",", "").replace(" ", "")
    if s == "":
        return np.nan
    m = re.fullmatch(r"(\d+(\.\d+)?)(m|k)?", s)
    if m:
        num = float(m.group(1))
        suffix = m.group(3)
        if suffix == "k":
            num *= 1_000
        elif suffix == "m":
            num *= 1_000_000
        return num
    return np.nan


def normalize_time_window(value: str) -> tuple[str, bool]:
    """
    Normalize to 'YYYY' or 'YYYY-YYYY'.
    Returns (normalized_string, ok_bool). If not ok, normalized may still be best-effort.
    """
    raw = norm_str(value)
    if raw == "":
        return ("", False)

    # Find all 4-digit years 1900-2099
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", raw)]
    years = sorted(set(years))

    if len(years) == 1:
        return (str(years[0]), True)
    if len(years) >= 2:
        # Use min/max; covers "2019, 2020, 2021"
        return (f"{years[0]}-{years[-1]}", True)

    # No 4-digit year found -> cannot normalize
    return (raw, False)


def ensure_columns_exist(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing expected columns in Table.xlsx:\n"
            + "\n".join(f"- {c}" for c in missing)
            + "\n\nTip: check spelling/case in your sheet and update COL_* constants."
        )


# Main cleaning

def main():
    df = pd.read_excel(INPUT_XLSX)

    required_cols = [
        COL_METADATA_USED,
        COL_METADATA_TYPES,
        COL_TIME_WINDOW,
        COL_CONCLUSION,
        COL_DS_METHODS,
        COL_CV_TASKS,
        COL_DATASET_SIZE,
        COL_COLLECTION_MODE,
    ]
    ensure_columns_exist(df, required_cols)

    # Normalize some key columns to consistent strings
    df[COL_METADATA_USED] = df[COL_METADATA_USED].apply(yes_no)
    df[COL_CONCLUSION] = df[COL_CONCLUSION].apply(lambda x: norm_str(x).title())

    # Create a flag column (to accumulate)
    FLAG_COL = "QC Flags"
    df[FLAG_COL] = ""

    def add_flag(mask: pd.Series, reason: str):
        df.loc[mask, FLAG_COL] = df.loc[mask, FLAG_COL].where(
            df.loc[mask, FLAG_COL].eq(""),
            df.loc[mask, FLAG_COL] + " | "
        ) + reason

    # ---- Rule 8: Dataset size missing -> Not mentioned
    df[COL_DATASET_SIZE] = df[COL_DATASET_SIZE].apply(norm_str)
    df.loc[df[COL_DATASET_SIZE].eq(""), COL_DATASET_SIZE] = "Not mentioned"

    # ---- Rule 1: Metadata consistency check (flag)
    mask_meta_yes_types_blank = (df[COL_METADATA_USED].eq("Yes")) & (df[COL_METADATA_TYPES].apply(is_blank))
    add_flag(mask_meta_yes_types_blank, "Metadata used=Yes but Metadata types empty")

    # ---- Rule 2: Metadata contradiction fix
    mask_meta_no_types_present = (df[COL_METADATA_USED].eq("No")) & (~df[COL_METADATA_TYPES].apply(is_blank))
    # If meta no but types not empty, set to Not applicable
    df.loc[mask_meta_no_types_present, COL_METADATA_TYPES] = "Not applicable"

    # ---- Rule 3: Study time window normalization + flag if cannot normalize
    normalized_vals = []
    ok_vals = []
    for v in df[COL_TIME_WINDOW].apply(norm_str).tolist():
        normed, ok = normalize_time_window(v)
        normalized_vals.append(normed)
        ok_vals.append(ok)
    df[COL_TIME_WINDOW] = normalized_vals
    add_flag(pd.Series([not ok and norm_str(x) != "" for ok, x in zip(ok_vals, normalized_vals)], index=df.index),
             "Study time window not in YYYY or YYYY-YYYY (manual check)")

    # ---- Rule 4: Final inclusion filter (Include only)
    # Keep original df for flags export first; create filtered after.
    valid_conclusions = {"Include"}
    mask_keep = df[COL_CONCLUSION].isin(valid_conclusions)

    # ---- Rule 5: DS Methods=None -> CV task(s)=None
    mask_ds_none = df[COL_DS_METHODS].apply(lambda x: norm_str(x).lower() == "none")
    df.loc[mask_ds_none, COL_CV_TASKS] = "None"

    # ---- Rule 6: CV != None -> DS != None (flag)
    mask_cv_present = ~df[COL_CV_TASKS].apply(is_none_like)
    mask_ds_is_none = df[COL_DS_METHODS].apply(lambda x: norm_str(x).lower() == "none")
    mask_violate_cv_implies_ds = mask_cv_present & mask_ds_is_none
    add_flag(mask_violate_cv_implies_ds, "CV task present but DS Methods=None")

    # ---- Rule 7: Dataset size <100 AND DS != None -> flag
    approx_sizes = df[COL_DATASET_SIZE].apply(approx_dataset_size)
    mask_size_lt_100 = approx_sizes.notna() & (approx_sizes < 100)
    mask_ds_not_none = ~df[COL_DS_METHODS].apply(lambda x: norm_str(x).lower() == "none")
    mask_violate_small_ds = mask_size_lt_100 & mask_ds_not_none
    add_flag(mask_violate_small_ds, "Dataset size <100 but DS Methods not None (check plausibility)")

    # ---- Rule 9: Dataset size >=100k AND collection mode=Manual -> flag
    mask_size_ge_100k = approx_sizes.notna() & (approx_sizes >= 100_000)
    mask_manual = df[COL_COLLECTION_MODE].apply(lambda x: norm_str(x).lower() == "manual")
    mask_violate_manual_huge = mask_size_ge_100k & mask_manual
    add_flag(mask_violate_manual_huge, "Dataset size >=100k but Data collection=Manual (check)")

  
    # Outputs
 
    # Flags file
    df_flags = df.loc[df[FLAG_COL].apply(lambda x: norm_str(x) != "")].copy()

    # Filtered cleaned table
    df_clean = df.loc[mask_keep].copy()

    # Save
    df_clean.to_excel("Table_cleaned.xlsx", index=False)
    df_clean.to_csv("Table_cleaned.csv", index=False)
    df_flags.to_excel("Table_flags.xlsx", index=False)

    print("Saved: Table_cleaned.xlsx, Table_cleaned.csv, Table_flags.xlsx")
    print(f"Rows in original: {len(df)}")
    print(f"Rows kept (Include/Maybe): {len(df_clean)}")
    print(f"Rows flagged for review: {len(df_flags)}")


if __name__ == "__main__":
    main()
