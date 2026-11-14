# One Look — flood alerting from a single satellite image

**Intelligence Lab · Makers Piscine · 16 October – 16 November 2026**
**Author:** Ojaswi Rayamajhi · **Mentor:** Lilian Rage

A lightweight model that looks at one Sentinel-2 image chip **once** and returns one of three decisions:

| Decision | Meaning |
|---|---|
| **ALERT** | Flood water present: non-permanent water covers more than 1 % of the chip |
| **NO ALERT** | Normal scene, including permanent water (rivers, lakes) |
| **UNKNOWN** | The scene is unlike anything seen in training, so the model abstains instead of guessing |

## 1. What deserves an alert, and what doesn't

**Target anomaly: floods.** They are time-critical (onboard detection saves the most downlink delay), and a public,
hand-labelled dataset covering 11 countries exists, which allows an honest test on countries never seen in training.
One anomaly type done carefully fits the timeline.

**Alert rule:** a chip is ALERT if water that is **not** permanent water covers more than **1 %** of its valid pixels.
Permanent water comes from the JRC Global Surface Water map and is subtracted first.

**Must not alert:** rivers, lakes, sea (permanent water); clouds and their shadows; wet fields; normal scenes.

**UNKNOWN:** scenes far from the training data (thick cloud, snow, desert, sensor faults) are flagged instead of decided.

## 2. Data

Sen1Floods11 v1.1, hand-labelled subset: **446 Sentinel-2 chips** (512 × 512 px at 10 m, about 5 × 5 km) from
**11 flood events**, with hand-drawn water masks and the JRC permanent-water mask. With the 1 % rule:
**273 ALERT / 173 NO ALERT**. Details, licences, composition and leakage risks: [`docs/data_card.md`](docs/data_card.md).

**Geographic split. All chips of a country stay in one subset**, because neighbouring chips of the same flood are
near-duplicates and a random split would leak:

| Subset | Countries | ALERT | NO ALERT | Used for |
|---|---|---|---|---|
| train | India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, USA | 197 | 109 | fitting models |
| val | Spain, Sri-Lanka | 46 | 26 | choosing thresholds and the final model |
| test | Bolivia, Ghana | 30 | 38 | measured once, never used for any choice |

Bolivia is held out as in the dataset paper. Ghana is added because Bolivia alone has only 4 NO ALERT chips,
too few to measure the false-alarm rate.

## 3. Approach: simple first, each step measured

Every model sees the same **55 features per chip** (`src/features.py`): for each of the 13 bands the mean,
standard deviation, 25th and 75th percentile, plus the share of water-like pixels by MNDWI at three cut-offs.
Working on features rather than raw pixels keeps every model tiny and fast.

| # | Approach | Why |
|---|---|---|
| B1 | **MNDWI rule** (no learning): pixel is water if (Green − SWIR1)/(Green + SWIR1) > τ; ALERT if enough water | Physical baseline: water reflects green and absorbs shortwave infrared. Sets the floor. |
| M1 | **Logistic regression** | Smallest learned model (a few KB) |
| M2 | **Random forest, small** (50 trees, depth 6) | Non-linear, still small |
| M3 | **Random forest, large** (300 trees) | Does a bigger model help? (size/speed trade-off) |

**Protocol (identical for every approach):** train on *train* → choose the alert threshold on *val* to get the highest
recall while keeping the **false-alarm rate ≤ 5 %** → measure **once** on *test*. The final model is chosen on *val* only.

**UNKNOWN detector:** Mahalanobis distance between a chip's features and the training data (Ledoit-Wolf covariance).
If it is further than 99 % of train+val chips, the decision is UNKNOWN. It is evaluated on the real test chips (should
rarely trigger) and on synthetic out-of-distribution versions of them (thick cloud, snow, desert sand, sensor noise,
dead bands).

**Embeddability:** size of the saved model (KB) and CPU time per chip, including feature extraction.

## 4. Results

Produced by `notebooks/03_run_everything.ipynb` on the real dataset; full tables in [`results/`](results/).

> **To fill in:** paste `results/results.md` here after running notebook 03.

## 5. Reproduce

Google Colab (recommended):
1. `notebooks/01_setup_data.ipynb`: downloads LabelHand, S2Hand and JRCWaterHand (446 files each) to Google Drive.
2. `notebooks/03_run_everything.ipynb`: labels, split, all experiments, failure analysis and demo.

Locally (Python 3.10+):

```bash
pip install -r requirements.txt
python src/labels.py --root data/HandLabeled --out data/labels.csv
python src/split.py  --labels data/labels.csv --out data/splits.json
python src/run_experiments.py --root data/HandLabeled --labels data/labels.csv --splits data/splits.json --out results
python -m pytest tests
```

## 6. Demo on a new image

```bash
python src/predict.py --image my_scene.tif --model results/one_look_model.joblib --out decision.png
```

Input: a Sentinel-2 L1C GeoTIFF with all 13 bands (B1…B12, including B8A and B10) at 10 m, for example from
[Copernicus Browser](https://browser.dataspace.copernicus.eu/). A chip gets one decision; a larger scene is cut into
512 × 512 tiles, each decided in one pass, and the scene is ALERT if any tile is ALERT. The output picture shows the
image with coloured tiles (red ALERT, green NO ALERT, yellow UNKNOWN) next to a water-index map.

## 7. Repository

```text
notebooks/01_setup_data.ipynb      download the data to Google Drive
notebooks/02_labels_split.ipynb    look at the data, labels, split (step by step)
notebooks/03_run_everything.ipynb  all experiments + demo, in one run
src/labels.py                      pixel masks -> one label per chip
src/split.py                       geographic split (test Bolivia+Ghana, val Spain+Sri-Lanka)
src/features.py                    reading Sentinel-2 and the 55 features
src/baseline_mndwi.py              standalone version of the MNDWI baseline
src/run_experiments.py             comparison, UNKNOWN detector, size/speed, failure analysis
src/predict.py                     demo on a new image
eval/metrics.py                    precision, recall, F1, false-alarm rate
tests/                             unit tests (labels, split)
docs/data_card.md                  data sources, licences, composition, biases, leakage
results/                           outputs of the final run
```

## 8. Limitations

- Sentinel-2 is optical: under thick cloud it cannot see the ground. Cloudy scenes should end up UNKNOWN rather than
  NO ALERT, but a flood under cloud will be missed. Sentinel-1 radar (sees through cloud) is the natural next step.
- 11 flood events only: results on 2 test countries carry large uncertainty (68 test chips).
- The NO ALERT chips all come from flood-affected regions, so many kinds of normal scene (cities, desert, snow) are
  missing. This is why the UNKNOWN detector matters.
- UNKNOWN detection is tested on synthetic shifts, not on real imagery of those conditions.
- Image-level decisions only: the model says *whether* a chip is flooded, not *where* in the chip.
- The 1 % alert rule is a project choice, not an operational warning threshold.
- Models are classical (logistic regression, random forest) on hand-made features, not a deep CNN: chosen for size,
  speed and the time available.

## 9. Tools and AI use

Built with Python, rasterio, scikit-learn, Google Colab and GitHub. No pre-trained model is used. I used Claude
(Anthropic) as a tutor and coding assistant to explain concepts and draft code. I reviewed and ran all of it, and I
made and can justify the design decisions (scope, split, false-alarm target).
