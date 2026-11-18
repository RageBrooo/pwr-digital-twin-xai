"""explain_models.py -- global SHAP importance of a trained baseline on HELD-OUT test rows."""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_models import prepare                                      # noqa: E402
from src.ai_engine.explainer import RFExplainer, save_importance_plot         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="B_full", choices=["B_full", "A_hardware_only"])
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    args = ap.parse_args()
    _, split, feat = prepare(Path(args.raw), ROOT / "data" / "processed" / "split.json", 0)
    FN = joblib.load(ROOT / "models" / "feature_names.joblib")
    model = joblib.load(ROOT / "models" / f"rf_{args.model}.joblib")
    te = feat[feat.run_id.isin(split["test"]) & feat.post]
    X = te.sample(min(args.rows, len(te)), random_state=0)
    ex = RFExplainer(model, FN)
    imp = ex.global_importance(X)
    out = ROOT / "data" / "results"
    out.mkdir(parents=True, exist_ok=True)
    imp.round(5).to_csv(out / f"shap_importance_{args.model}.csv")
    save_importance_plot(imp, out / f"shap_importance_{args.model}.png")
    print(imp.head(10).round(4).to_string())
    print("\nlocal explanation of one held-out row:")
    r = ex.explain(X.iloc[[0]])[0]
    print(" true class:", X.label.iloc[0], "| predicted:", r["predicted"], f"(p={r['probability']:.2f})")
    for f, v, x in r["top"][:5]:
        print(f"   {f:14s} shap={v:+.3f}  value={x:.4g}")


if __name__ == "__main__":
    main()
