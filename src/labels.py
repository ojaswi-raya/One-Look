"""Convert Sen1Floods11 pixel masks into image-level flood labels."""
import argparse
import csv
from pathlib import Path
import numpy as np
import rasterio

def flood_fraction(label: np.ndarray, permanent: np.ndarray | None = None) -> float:
    """Fraction of valid pixels labelled as non-permanent water."""
    valid = label != -1
    if not valid.any():
        return float("nan")
    flood = label == 1
    if permanent is not None:
        if permanent.shape != label.shape:
            raise ValueError(f"Mask shape mismatch: label={label.shape}, permanent={permanent.shape}")
        flood &= permanent != 1
    return float(flood[valid].sum() / valid.sum())

def chip_label(frac: float, threshold: float) -> int:
    """Return 1 for ALERT, 0 for NO ALERT; reject missing fractions."""
    if not np.isfinite(frac):
        raise ValueError("Cannot label a chip with no valid pixels")
    return int(frac > threshold)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="Folder containing LabelHand/ and optionally JRCWaterHand/")
    ap.add_argument("--out", default="data/labels.csv")
    ap.add_argument("--threshold", type=float, default=0.01)
    args = ap.parse_args()
    if not 0 <= args.threshold <= 1:
        ap.error("--threshold must be between 0 and 1")
    root = Path(args.root)
    label_dir = root / "LabelHand"
    files = sorted(label_dir.glob("*_LabelHand.tif"))
    if not files:
        raise SystemExit(f"No *_LabelHand.tif files found in {label_dir}. Check the dataset folder.")
    rows=[]
    for path in files:
        cid = path.name.removesuffix("_LabelHand.tif")
        with rasterio.open(path) as ds:
            label=ds.read(1)
        jp=root/"JRCWaterHand"/f"{cid}_JRCWaterHand.tif"
        permanent=None
        if jp.exists():
            with rasterio.open(jp) as ds: permanent=ds.read(1)
        frac=flood_fraction(label, permanent)
        if not np.isfinite(frac):
            print(f"Skipping {cid}: no valid pixels")
            continue
        rows.append({"chip_id":cid,"country":cid.split("_")[0],"flood_fraction":round(frac,6),"label":chip_label(frac,args.threshold)})
    if not rows:
        raise SystemExit("No valid labelled chips were found.")
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    positives=sum(r["label"] for r in rows)
    print(f"Saved {len(rows)} chips to {out}: {positives} ALERT, {len(rows)-positives} NO ALERT (threshold={args.threshold:.1%})")

if __name__=="__main__": main()
