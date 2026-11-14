"""Demo: analyse a new Sentinel-2 image and show the decision.

    python src/predict.py --image my_scene.tif --model outputs/one_look_model.joblib --out decision.png

The image must be a 13-band Sentinel-2 L1C GeoTIFF (bands B1..B12 incl. B8A and B10, 10 m).
A small chip (<= 512 px) gets one decision. A larger scene is cut into 512x512 tiles and every
tile gets its own decision in one pass; the scene is ALERT if any tile is ALERT.
Decisions: ALERT / NO ALERT / UNKNOWN (scene unlike the training data -> the model abstains).
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import chip_features, mndwi, read_s2, rgb, valid_mask  # noqa: E402

TILE = 512
COLORS = {"ALERT": "red", "NO ALERT": "limegreen", "UNKNOWN": "gold", "EMPTY": "grey"}


def decide(bundle, x):
    f = chip_features(x)
    if f is None:
        return "EMPTY", float("nan"), float("nan")
    f = f[None]
    d = float(bundle["ood_cov"].mahalanobis(bundle["ood_scaler"].transform(f))[0])
    p = float(bundle["model"].predict_proba(f)[0, 1])
    if d > bundle["ood_threshold"]:
        return "UNKNOWN", p, d
    return ("ALERT" if p >= bundle["threshold"] else "NO ALERT"), p, d


def tiles(x):
    _, h, w = x.shape
    if h <= TILE and w <= TILE:
        yield 0, 0, x
        return
    for r in range(0, h - TILE // 2, TILE):
        for c in range(0, w - TILE // 2, TILE):
            yield r, c, x[:, r:r + TILE, c:c + TILE]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="outputs/one_look_model.joblib")
    ap.add_argument("--out", default="decision.png")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    x = read_s2(a.image)
    t0 = time.perf_counter()
    res = [(r, c, t.shape[1], t.shape[2], *decide(bundle, t)) for r, c, t in tiles(x)]
    ms = (time.perf_counter() - t0) * 1000
    decs = [d for *_, d, _, _ in res]
    overall = ("ALERT" if "ALERT" in decs else "UNKNOWN" if "UNKNOWN" in decs
               else "NO ALERT" if "NO ALERT" in decs else "EMPTY")

    print(f"Image: {a.image}  ({x.shape[2]}x{x.shape[1]} px, {len(res)} tile(s), {ms:.0f} ms total)")
    for r, c, h, w, d, p, dist in res:
        print(f"  tile row {r:5d} col {c:5d}: {d:9s} p(flood)={p:.2f}  unknown-score={dist:.1f}"
              f" (limit {bundle['ood_threshold']:.1f})")
    print(f"DECISION: {overall}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(rgb(x))
    for r, c, h, w, d, p, _ in res:
        ax[0].add_patch(Rectangle((c, r), w, h, fill=False, lw=2, ec=COLORS[d]))
        ax[0].text(c + 8, r + 30, f"{d}\np={p:.2f}", color=COLORS[d], fontsize=8, weight="bold")
    ax[0].set_title(f"Decision: {overall}", color=COLORS[overall], fontsize=14, weight="bold")
    m = np.where(valid_mask(x), mndwi(x), np.nan)
    ax[1].imshow(m > 0.2, cmap="Blues")
    ax[1].set_title("Water-like pixels (MNDWI > 0.2), for context")
    for axis in ax:
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(a.out, dpi=90)
    print(f"Picture saved to {a.out}")


if __name__ == "__main__":
    main()
