"""Turn Sen1Floods11 pixel masks into image-level labels.

ALERT (1) if flood water (water minus permanent water) covers more than
`threshold` of the valid pixels in the chip, else NO ALERT (0).

Expected file names (Sen1Floods11 v1.1, hand-labelled):
    LabelHand/<Country>_<id>_LabelHand.tif     (-1 invalid, 0 not water, 1 water)
    JRCWaterHand/<Country>_<id>_JRCWaterHand.tif  (1 = permanent water)  [optional]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio


def flood_fraction(label: np.ndarray, permanent: np.ndarray | None = None) -> float:
    """Share of valid pixels that are flood water (water that is not permanent)."""
    valid = label != -1
    if valid.sum() == 0:
        return float("nan")
    water = label == 1
    if permanent is not None:
        water &= permanent != 1
    return float(water[valid].sum() / valid.sum())


def chip_label(frac: float, threshold: float) -> int:
    return int(frac > threshold)


def chip_id(path: Path) -> str:
    return path.name.replace("_LabelHand.tif", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing LabelHand/ (and JRCWaterHand/)")
    ap.add_argument("--out", default="data/labels.csv")
    ap.add_argument("--threshold", type=float, default=0.01)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for lp in sorted((root / "LabelHand").glob("*_LabelHand.tif")):
        cid = chip_id(lp)
        with rasterio.open(lp) as f:
            label = f.read(1)
        jp = root / "JRCWaterHand" / f"{cid}_JRCWaterHand.tif"
        permanent = None
        if jp.exists():
            with rasterio.open(jp) as f:
                permanent = f.read(1)
        frac = flood_fraction(label, permanent)
        rows.append({
            "chip_id": cid,
            "country": cid.split("_")[0],
            "flood_fraction": round(frac, 5),
            "label": chip_label(frac, args.threshold),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    n_alert = sum(r["label"] for r in rows)
    print(f"{len(rows)} chips -> {n_alert} ALERT / {len(rows) - n_alert} NO ALERT  (threshold {args.threshold})")


if __name__ == "__main__":
    main()
