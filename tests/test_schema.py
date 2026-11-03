import pandas as pd
import pytest

from pricing_toolkit.schema import load_config, validate
from pricing_toolkit.simulator import simulate

cfg = load_config()


def test_simulated_data_passes_validation():
    df, _ = simulate("observational", seed=0, cfg=cfg)
    out = validate(df, cfg)
    assert "discount_depth" in out.columns
    assert out["discount_depth"].between(0, 1).all()


def test_missing_column_is_rejected():
    df, _ = simulate("flat", seed=0, cfg=cfg)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate(df.drop(columns="units_sold"), cfg)


def test_negative_units_rejected():
    df, _ = simulate("flat", seed=0, cfg=cfg)
    df.loc[0, "units_sold"] = -1
    with pytest.raises(ValueError, match="negative"):
        validate(df, cfg)


def test_duplicate_rows_rejected():
    df, _ = simulate("flat", seed=0, cfg=cfg)
    with pytest.raises(ValueError, match="duplicate"):
        validate(pd.concat([df, df.iloc[[0]]]), cfg)
