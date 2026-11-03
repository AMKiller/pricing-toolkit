"""Synthetic fashion data in the canonical schema.

Three ways the opening discount can be set:
  flat           everything at one discount, never marked down
  observational  weak brands get deeper discounts (confounding), slow lots get
                 marked down at week 4: like a real, rule-and-judgement history
  experiment     opening discount randomised, independent of brand strength

True price sensitivity is known, so later estimation code can be checked
against the truth.
"""
import numpy as np
import pandas as pd

from .schema import load_config

BRANDS = {f"B{i:02d}": t for i, t in enumerate(["premium"] * 3 + ["mid"] * 5 + ["value"] * 4)}
STRENGTH = {"premium": 1.0, "mid": 0.0, "value": -1.0}     # demand strength
ELASTICITY = {"premium": 1.2, "mid": 1.8, "value": 2.4}    # price sensitivity (positive = sales fall as price rises)
TYPE_ADJ = {"running": 0.0, "lifestyle": -0.2, "kids": 0.3}
EXPERIMENT_ARMS = [0.50, 0.55, 0.60, 0.65, 0.70]


def simulate(mode="observational", n_items=300, n_stores=10, n_weeks=12,
             seed=0, flat_discount=0.60, markdown_week=4, cfg=None):
    """Return (table, truth). `table` is in the canonical schema;
    `truth` holds each item's true elasticity."""
    if mode not in ("flat", "observational", "experiment"):
        raise ValueError("mode must be flat, observational or experiment")
    cfg = cfg or load_config()
    d_ref = cfg["simulator"]["reference_discount"]
    decay = cfg["simulator"]["weekly_decay"]
    rng = np.random.default_rng(seed)

    # --- items ---
    items = pd.DataFrame({"item_id": np.arange(n_items)})
    items["brand"] = rng.choice(list(BRANDS), n_items)
    items["brand_tier"] = items["brand"].map(BRANDS)
    items["product_type"] = rng.choice(list(TYPE_ADJ), n_items)
    items["lot_id"] = items["item_id"] // 3
    items["ref_price"] = np.round(rng.lognormal(np.log(1500), 0.3, n_items), -1)
    strength = items["brand_tier"].map(STRENGTH).to_numpy()
    elast = (items["brand_tier"].map(ELASTICITY) + items["product_type"].map(TYPE_ADJ)).to_numpy()
    base_rate = np.exp(rng.normal(0, 0.3, n_items) + 0.5 * strength) * 2.0  # units/store/week at reference discount
    store_factor = rng.uniform(0.6, 1.4, n_stores)

    # --- opening discount ---
    if mode == "flat":
        disc = np.full(n_items, flat_discount)
    elif mode == "observational":
        disc = np.clip(0.60 - 0.06 * strength + rng.normal(0, 0.03, n_items), 0.30, 0.80)
    else:
        disc = rng.choice(EXPERIMENT_ARMS, n_items)

    # --- stock per item per store ---
    stock = np.maximum(3, np.round(base_rate[:, None] * store_factor[None, :] * 6
                                   * rng.uniform(0.7, 1.4, (n_items, 1)))).astype(int)
    stock0_total = stock.sum(axis=1)

    # --- week by week ---
    weeks = []
    for w in range(n_weeks):
        if mode == "observational" and w == markdown_week:
            sell_through = 1 - stock.sum(axis=1) / stock0_total
            disc = np.where(sell_through < 0.35, np.minimum(disc + 0.10, 0.85), disc)
        price_ratio = (1 - disc) / (1 - d_ref)
        lam = (base_rate[:, None] * store_factor[None, :] * decay ** w
               * price_ratio[:, None] ** (-elast[:, None]))
        sold = np.minimum(rng.poisson(lam), stock)
        start = stock.copy()
        stock = stock - sold
        weeks.append(pd.DataFrame({
            "item_id": np.repeat(np.arange(n_items), n_stores),
            "store_id": np.tile(np.arange(n_stores), n_items),
            "week": w,
            "units_sold": sold.ravel(),
            "stock_start": start.ravel(),
            "stock_end": stock.ravel(),
            "discount": np.repeat(disc, n_stores),
        }))

    df = pd.concat(weeks, ignore_index=True)
    df = df.merge(items[["item_id", "lot_id", "brand", "brand_tier", "product_type", "ref_price"]],
                  on="item_id")
    df["price_paid"] = np.round(df["ref_price"] * (1 - df["discount"]), 0)
    df = df.drop(columns="discount")
    truth = pd.DataFrame({"item_id": items["item_id"], "true_elasticity": elast})
    return df, truth
