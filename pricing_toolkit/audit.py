"""Stage 1 audit: how much does the opening discount vary, and is it enough
to learn price response from?  This checks SPREAD only. Whether the spread is
tangled up with brand strength (confounding) is checked in the exploration stage.
"""
import pandas as pd


def opening_discounts(df: pd.DataFrame, groups) -> pd.DataFrame:
    """One row per item: its discount in its first week on the floor."""
    first_week = df.groupby("item_id")["week"].transform("min")
    first = df[df["week"] == first_week]
    agg = {"opening_discount": ("discount_depth", "mean")}
    agg.update({g: (g, "first") for g in groups})
    return first.groupby("item_id").agg(**agg).reset_index()


def markdown_share(df: pd.DataFrame) -> float:
    """Share of items whose discount changes at any point."""
    n_levels = df.assign(d=df["discount_depth"].round(2)).groupby("item_id")["d"].nunique()
    return float((n_levels > 1).mean())


def discount_spread(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    groups = cfg["audit_groups"]
    od = opening_discounts(df, groups)
    g = od.groupby(groups)["opening_discount"].agg(
        n_items="size", lowest="min", highest="max", typical="median", std="std")
    g["range"] = g["highest"] - g["lowest"]
    return g.round(3)


def audit(df: pd.DataFrame, cfg: dict) -> dict:
    a = cfg["audit"]
    od = opening_discounts(df, cfg["audit_groups"])
    p5, p95 = od["opening_discount"].quantile([0.05, 0.95])
    spread = float(p95 - p5)
    by_group = discount_spread(df, cfg)
    smallest = int(by_group["n_items"].min())
    stockouts = float((df["stock_end"] == 0).mean())

    if spread < a["flat_discount_range"]:
        verdict = "Test first: the opening discount is effectively flat"
    elif spread < a["min_discount_range"] or smallest < a["min_items_per_group"]:
        verdict = "Go with caution: spread or group sizes are thin"
    else:
        verdict = "Go on spread (confounding still to be checked)"

    return {
        "n_items": int(od.shape[0]),
        "opening_discount_range_5_95": round(spread, 3),
        "smallest_group_n_items": smallest,
        "share_items_marked_down": round(markdown_share(df), 3),
        "share_rows_out_of_stock": round(stockouts, 3),
        "verdict": verdict,
    }
