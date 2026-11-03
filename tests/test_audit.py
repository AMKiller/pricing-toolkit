from pricing_toolkit.audit import audit
from pricing_toolkit.schema import load_config, validate
from pricing_toolkit.simulator import simulate

cfg = load_config()


def run(mode, **kw):
    df, _ = simulate(mode, seed=1, cfg=cfg, **kw)
    return audit(validate(df, cfg), cfg)


def test_flat_discount_is_flagged_test_first():
    assert run("flat")["verdict"].startswith("Test first")


def test_experiment_data_is_not_flagged_flat():
    assert not run("experiment")["verdict"].startswith("Test first")


def test_flat_data_has_no_markdowns():
    assert run("flat")["share_items_marked_down"] == 0
