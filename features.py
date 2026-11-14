"""Reading Sentinel-2 images and turning one chip into a short list of numbers (features).

Every model in this project looks at these features, never at raw pixels, which keeps
the models tiny and fast (embeddable). One chip -> 55 numbers:
  - for each of the 13 bands: mean, standard deviation, 25th and 75th percentile (52)
  - fraction of pixels that look like water by MNDWI at three cut-offs (3)
"""
from pathlib import Path

import numpy as np
import rasterio

N_BANDS = 13
B2, B3, B4, B8, B11 = 1, 2, 3, 7, 11      # 0-based band indices in Sen1Floods11 S2 chips
MNDWI_CUTS = (0.0, 0.2, 0.4)
FEATURE_VERSION = "v1: 13x(mean,std,p25,p75) + MNDWI water fraction at 0.0/0.2/0.4"


def read_s2(path):
    """Read a 13-band Sentinel-2 GeoTIFF as float32 in Sen1Floods11 units (reflectance x 10000)."""
    with rasterio.open(path) as ds:
        x = ds.read().astype("float32")
    if x.shape[0] != N_BANDS:
        raise ValueError(f"{Path(path).name}: expected {N_BANDS} Sentinel-2 bands "
                         f"(B1..B12 incl. B8A and B10), found {x.shape[0]}")
    x[~np.isfinite(x)] = 0
    if 0 < np.nanmax(x) <= 2.0:          # e.g. Copernicus Browser float reflectance 0-1
        x = x * 10000
    return x


def valid_mask(x):
    """Pixels with real data (no-data pixels are 0 in every band)."""
    return x.sum(axis=0) > 0


def mndwi(x):
    g, s = x[B3], x[B11]
    return (g - s) / (g + s + 1e-6)


def chip_features(x):
    v = valid_mask(x)
    if v.sum() < 0.05 * v.size:       # almost empty chip
        return None
    feats = []
    for band in x:
        b = band[v]
        feats += [b.mean(), b.std(), np.percentile(b, 25), np.percentile(b, 75)]
    m = mndwi(x)[v]
    feats += [(m > c).mean() for c in MNDWI_CUTS]
    return np.asarray(feats, dtype="float32")


def rgb(x):
    """True-colour picture for display."""
    return np.clip(x[[B4, B3, B2]].transpose(1, 2, 0) / 3000, 0, 1)
