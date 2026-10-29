# One Look — onboard flood alerting from a single satellite image

Intelligence Lab · Makers Piscine (Oct 16 → Nov 16, 2026) · Ojaswi Rayamajhi · Mentor: Lilian Rage

A lightweight model that looks at one satellite image chip **once** and returns one of three decisions:

| Decision | Meaning |
|---|---|
| **ALERT** | Flood water present beyond the threshold (see below) |
| **NO ALERT** | Normal scene, including permanent water, clouds, wet fields |
| **UNKNOWN** | Scene unlike anything seen in training → do not guess, flag it |

## 1. What deserves an alert

**Target anomaly: floods.** Chosen because:
- Floods are high-impact and time-critical, so onboard detection saves the most downlink delay.
- A good public labelled dataset exists (Sen1Floods11), covering many countries, which makes an honest geographic split possible.
- One well-done anomaly type fits the timeline. Burned areas (CaBuAr) are a possible extension.

**Alert rule (image-level label derived from pixel masks):**
`ALERT` if flood-water pixels (water in the event mask **minus** permanent water from the JRC layer) cover more than **T %** of the valid pixels in the chip. T starts at 1 %; it will be tuned and discussed with the mentor.

**Must NOT trigger an alert:**
- Permanent water: rivers, lakes, sea, reservoirs
- Clouds and cloud shadows (dark shadows look like water in optical bands)
- Wet fields, irrigated crops, normal seasonal change

**UNKNOWN:** out-of-distribution scenes (snow/ice, desert, burn scars, heavy cloud, sensor artefacts), detected with a confidence or feature-distance score on top of the classifier.

## 2. Data

- **Main:** Sen1Floods11 v1.1. 512×512 chips at 10 m, Sentinel-1 (VV, VH) and Sentinel-2 (13 bands), 446 hand-labelled chips from 11 flood events, plus weakly labelled chips.
- **Hard negatives:** CloudSEN12 (cloudy scenes) and chips dominated by permanent water.
- **Demo images:** new scenes downloaded from Copernicus Browser for recent floods listed by Copernicus EMS.

Details, licences and leakage risks are in [`docs/data_card.md`](docs/data_card.md).

**Split = by country/event, never random.** Neighbouring chips from the same flood are near-duplicates, so a random split would leak and inflate scores. Bolivia is kept fully out as a geographic test set, following the dataset authors.

## 3. Approach (simple → better, each step measured)

| # | Approach | Why |
|---|---|---|
| B1 | MNDWI threshold rule (no ML) | Physical baseline; sets the floor |
| B2 | Logistic regression / random forest on per-band statistics | Cheap ML baseline |
| M1 | Small CNN (MobileNetV3-Small / ResNet-18, TorchGeo pretrained weights) | Main model |
| M2 | + threshold tuned for a target false-alarm rate, hard negatives, S1+S2 fusion | Reliability |
| M3 | + OOD score (max-softmax → energy → Mahalanobis) → UNKNOWN | "Flag the unknown" |
| M4 | ONNX export + int8 quantization | Embeddability |

All runs are logged to `results.csv`: precision, recall, false-alarm rate on normal images, AUROC (unknown), size in MB, CPU latency in ms.

## 4. Repo layout

```
src/labels.py        mask -> image-level label (ALERT / NO ALERT), permanent water removed
src/split.py         country/event-based train/val/test split, saved to data/splits.json
src/baseline_mndwi.py  baseline B1
eval/metrics.py      shared metrics: precision, recall, false-alarm rate, size, latency
docs/data_card.md    sources, licences, composition, biases, leakage risks
docs/plan.md         7-day plan and status
```

## 5. Reproduce

```bash
pip install -r requirements.txt
# download Sen1Floods11 v1.1 (hand-labelled subset) into data/sen1floods11/
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled data/sen1floods11/HandLabeled
python src/labels.py --root data/sen1floods11/HandLabeled --out data/labels.csv
python src/split.py  --labels data/labels.csv --out data/splits.json
python src/baseline_mndwi.py --root data/sen1floods11/HandLabeled --labels data/labels.csv --splits data/splits.json
```

## Status
See [`docs/plan.md`](docs/plan.md).
