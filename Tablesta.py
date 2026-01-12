#!/usr/bin/env python3
"""
多列统计小工具：对指定列做分词/年份解析，输出频次和百分比到控制台和 CSV。
- 支持多个目标列配置（列名或序号）。
- mode="tokens": 用分隔符 ; , / | 拆分计数。
- mode="year": 提取年份（支持 YYYY 或 YYYY-YYYY）。
纯标准库，无需依赖。
"""

import csv
import os
import re
import sys
from collections import Counter
from typing import List, Dict, Any

TABLE_CSV = "Table_cleaned.csv"

# 配置多个统计任务
# mode: "tokens" 按分隔符拆分计数；"year" 解析年份/年份区间；"single" 不拆分
TARGETS = [
    {"name": "year", "index": None, "mode": "year", "out": "year_counts.csv"},
    {
        "name": "Social media platform(s)",
        "index": None,
        "mode": "tokens",
        "out": "platform_counts.csv",
        # 百分比分母：如果需要固定分母（如 68 行），设置 percent_base
        "percent_base": 68,
    },
    {
        "name": "Dataset size (images)",
        "index": None,  # 可用 name 匹配，或填 16 指定列 P
        "mode": "single",  # 每行单值，不拆分
        "out": "dataset_size_counts.csv",
        "percent_base": 68,
    },
    {
        "name": "DS Methods",
        "index": None,  # 可用 name，或填 23 指定列 W
        "mode": "tokens",
        "out": "ds_methods_counts.csv",
        "percent_base": 68,
        "dedupe_per_row": True,  # 每行去重，避免多值重复放大
    },
    {
        "name": "Analysis method (primary/secondary)",
        "indices": [21, 22],  # U, V 列；1-based
        "mode": "tokens",
        "out": "analysis_methods_counts.csv",
        "percent_base": 68,
        "dedupe_per_row": True,
    },
    {
        "name": "Metadata used?",
        "index": 18,  # R 列
        "mode": "single",
        "out": "metadata_used_counts.csv",
        "percent_base": 68,
    },
]


def load_rows(path: str) -> List[List[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def find_col(headers: List[str], cfg: Dict[str, Any]) -> List[int]:
    # Config: multiple indices (1-based)
    if cfg.get("indices"):
        idxs = []
        for idx in cfg["indices"]:
            idx0 = idx - 1
            if 0 <= idx0 < len(headers):
                idxs.append(idx0)
            else:
                raise KeyError(f"Configured column index {idx} out of range (1..{len(headers)}).")
        return idxs
    # Config: column name
    if cfg.get("name"):
        for idx, h in enumerate(headers):
            if h.strip().lower() == cfg["name"].strip().lower():
                return [idx]
        raise KeyError(f"Configured column name '{cfg['name']}' not found.")
    # Config: column index (1-based)
    if cfg.get("index"):
        idx0 = cfg["index"] - 1
        if 0 <= idx0 < len(headers):
            return [idx0]
        raise KeyError(f"Configured column index {cfg['index']} out of range (1..{len(headers)}).")
    # Default: look for 'year'
    for idx, h in enumerate(headers):
        if h.strip().lower() == "year":
            return [idx]
    # Fallback: 3rd column
    if len(headers) >= 3:
        return [2]  # column C (0-based index)
    raise KeyError("No target column found and table has fewer than 3 columns.")


def split_tokens(val: str) -> List[str]:
    """Split a cell on common delimiters into tokens."""
    if not isinstance(val, str):
        return []
    s = val.strip()
    if not s:
        return []
    parts = re.split(r"[;,/|]+", s)
    tokens = [p.strip() for p in parts if p.strip()]
    return tokens


def extract_years(val: str) -> List[str]:
    """Extract years as strings, expand ranges like 2010-2012."""
    if not isinstance(val, str):
        return []
    s = val.strip()
    if not s:
        return []
    s = s.replace("–", "-").replace("—", "-")
    years = []
    m = re.fullmatch(r"((?:19|20)\d{2})\s*-\s*((?:19|20)\d{2})", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a <= b:
            years.extend(str(y) for y in range(a, b + 1))
        return years
    years.extend(re.findall(r"(?:19|20)\d{2}", s))
    return years


def process_target(headers: List[str], data_rows: List[List[str]], cfg: Dict[str, Any]):
    col_idxs = find_col(headers, cfg)
    mode = cfg.get("mode", "tokens")
    out_csv = cfg.get("out", "value_counts.csv")
    dedupe_per_row = cfg.get("dedupe_per_row", False)

    counter = Counter()
    multi_count = 0  # rows with >1 token (used for platform mode)
    for row in data_rows:
        tokens = []
        for col_idx in col_idxs:
            if col_idx >= len(row):
                continue
            val = row[col_idx]
            if mode == "year":
                tokens.extend(extract_years(val))
            elif mode == "single":
                if isinstance(val, str) and val.strip():
                    tokens.append(val.strip())
            else:
                tokens.extend(split_tokens(val))
        if len(tokens) > 1:
            multi_count += 1
            if cfg.get("name", "").lower().startswith("social media platform"):
                counter["Multiple"] += 1  # also count as a platform token
        row_tokens = set(tokens) if dedupe_per_row else tokens
        for token in row_tokens:
            counter[token] += 1

    if not counter:
        print(f"[{cfg.get('name','col')}] No tokens found; check column content.")
        return

    total = sum(counter.values())
    percent_base = cfg.get("percent_base", total) or total  # 避免 0
    all_numeric = all(token.isdigit() for token in counter)
    if all_numeric and counter:
        sorted_items = sorted(counter.items(), key=lambda x: int(x[0]))
    else:
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["value", "count", "percent"])
        for token, cnt in sorted_items:
            pct = (cnt / percent_base * 100) if percent_base else 0.0
            writer.writerow([token, cnt, f"{pct:.2f}%"])

    print(f"[{cfg.get('name','col')}] Value counts (saved to {out_csv}):")
    for token, cnt in sorted_items:
        pct = (cnt / percent_base * 100) if percent_base else 0.0
        print(f"{token}: {cnt} ({pct:.2f}%)")


def main():
    if not os.path.exists(TABLE_CSV):
        print(f"File not found: {TABLE_CSV}")
        sys.exit(1)

    rows = load_rows(TABLE_CSV)
    if not rows:
        print("Empty file.")
        sys.exit(1)

    headers, data_rows = rows[0], rows[1:]

    for cfg in TARGETS:
        try:
            process_target(headers, data_rows, cfg)
        except Exception as e:
            print(f"[{cfg.get('name','col')}] Error: {e}")


if __name__ == "__main__":
    main()
