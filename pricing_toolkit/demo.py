"""Run:  python -m pricing_toolkit.demo"""
from .audit import audit
from .schema import load_config, validate
from .simulator import simulate

if __name__ == "__main__":
    cfg = load_config()
    for mode in ("flat", "observational", "experiment"):
        df, _ = simulate(mode, seed=1, cfg=cfg)
        df = validate(df, cfg)
        print(f"\n=== {mode} ===")
        for k, v in audit(df, cfg).items():
            print(f"  {k}: {v}")
