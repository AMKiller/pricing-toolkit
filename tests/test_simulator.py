import numpy as np

from pricing_toolkit.schema import load_config
from pricing_toolkit.simulator import EXPERIMENT_ARMS, simulate

cfg = load_config()


def test_same_seed_gives_same_data():
    a, _ = simulate("observational", seed=5, cfg=cfg)
    b, _ = simulate("observational", seed=5, cfg=cfg)
    assert a.equals(b)


def test_stock_is_consistent():
    df, _ = simulate("observational", seed=2, cfg=cfg)
    assert (df["units_sold"] <= df["stock_start"]).all()
    assert (df["stock_end"] == df["stock_start"] - df["units_sold"]).all()
    assert (df["stock_end"] >= 0).all()


def test_flat_mode_has_one_discount_and_no_markdowns():
    df, _ = simulate("flat", seed=3, cfg=cfg)
    disc = (1 - df["price_paid"] / df["ref_price"]).round(1)
    assert disc.nunique() == 1


def test_experiment_mode_uses_all_arms():
    df, _ = simulate("experiment", seed=4, cfg=cfg)
    wk0 = df[df["week"] == 0].copy()
    wk0["disc"] = (1 - wk0["price_paid"] / wk0["ref_price"]).round(2)
    per_item = wk0.groupby("item_id")["disc"].mean().round(2)
    assert set(per_item.unique()) == set(np.round(EXPERIMENT_ARMS, 2))
