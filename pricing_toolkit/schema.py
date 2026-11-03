"""The canonical table every dataset is converted into.

One row per item per store per week. Adapters (visuelle, simulated, faro) produce
this table; everything downstream (audit, curves, models, ...) only reads it.
"""
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"

KEY = ["item_id", "store_id", "week"]
REQUIRED = KEY + [
    "lot_id",
    "units_sold",
    "stock_start",
    "stock_end",
    "ref_price",     # reference price (RRP)
    "price_paid",    # price actually charged that week
]
OPTIONAL = ["cost", "condition", "quantity"]  # empty for datasets that lack them
DERIVED = ["discount_depth"]                  # 1 - price_paid / ref_price


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def validate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Check a table against the canonical schema; return a sorted copy with
    discount_depth added. Raises ValueError on hard failures."""
    needed = REQUIRED + list(cfg["groups"])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    problems = []

    if out.duplicated(KEY).any():
        problems.append(f"{int(out.duplicated(KEY).sum())} duplicate rows on {KEY}")
    for col in ["units_sold", "stock_start", "stock_end"]:
        if (out[col] < 0).any():
            problems.append(f"negative values in {col}")
    if (out["ref_price"] <= 0).any() or (out["price_paid"] <= 0).any():
        problems.append("non-positive prices")
    if (out["week"] < 0).any():
        problems.append("negative week numbers")
    if out[REQUIRED].isna().any().any():
        bad = out[REQUIRED].isna().sum()
        problems.append(f"missing values: {bad[bad > 0].to_dict()}")
    if problems:
        raise ValueError("Schema validation failed: " + "; ".join(problems))

    out["discount_depth"] = 1 - out["price_paid"] / out["ref_price"]
    return out.sort_values(KEY).reset_index(drop=True)
