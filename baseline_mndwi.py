"""Baseline B1: physical rule, no learning.

MNDWI = (Green - SWIR1) / (Green + SWIR1)   (S2 bands B3 and B11)
Pixel is water if MNDWI > tau. Chip ALERT if water fraction > alert threshold.
tau and the alert threshold are chosen on VAL, then reported once on TEST.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import alert_metrics, log_result  # noqa: E402

B3, B11 = 2, 11  # 0-based indices in Sen1Floods11 S2 chips


def mndwi(s2: np.ndarray) -> np.ndarray:
    g, s = s2[B3].astype("float32"), s2[B11].astype("float32")
    return (g - s) / (g + s + 1e-6)


def water_fraction(s2, tau):
    m = mndwi(s2)
    valid = np.isfinite(m) & (s2[B3] > 0)
    return float((m[valid] > tau).mean()) if valid.any() else 0.0


def load_s2(root, cid):
    with rasterio.open(Path(root) / "S2Hand" / f"{cid}_S2Hand.tif") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--splits", default="data/splits.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.labels)))
    split = json.load(open(args.splits))
    sets = {s: [r for r in rows if r["country"] in split[s]] for s in split}
    s2 = {r["chip_id"]: load_s2(args.root, r["chip_id"]) for r in rows}

    # tune on val: maximise recall subject to false-alarm rate <= 5 %
    best = None
    for tau in np.arange(-0.2, 0.41, 0.05):
        for thr in (0.005, 0.01, 0.02, 0.05, 0.1):
            pred = [int(water_fraction(s2[r["chip_id"]], tau) > thr) for r in sets["val"]]
            m = alert_metrics([int(r["label"]) for r in sets["val"]], pred)
            if m["false_alarm_rate"] <= 0.05 and (best is None or m["recall"] > best[2]["recall"]):
                best = (tau, thr, m)
    if best is None:
        sys.exit("no setting met the false-alarm target on val")
    tau, thr, _ = best

    pred = [int(water_fraction(s2[r["chip_id"]], tau) > thr) for r in sets["test"]]
    m = alert_metrics([int(r["label"]) for r in sets["test"]], pred)
    print(f"B1 MNDWI  tau={tau:.2f} thr={thr}  TEST: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
    log_result({"approach": "B1_mndwi", "params": f"tau={tau:.2f};thr={thr}", **m, "size_mb": 0.0})


if __name__ == "__main__":
    main()
