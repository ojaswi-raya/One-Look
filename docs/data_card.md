# Data card — One Look

## Sources and licences

| Source | What is used | Licence / terms | Attribution |
|---|---|---|---|
| **Sen1Floods11 v1.1** (Bonafilia et al., CVPR Workshops 2020), hand-labelled subset, from `gs://sen1floods11` | `S2Hand` (Sentinel-2 chips), `LabelHand` (hand-drawn water masks) | **CC BY 4.0**, as listed on the dataset's Hugging Face mirror. The official GitHub repository does not state a licence; *to confirm with the mentor*. | Cite: Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* CVPR Workshops. |
| **Copernicus Sentinel-2** imagery (inside Sen1Floods11 and in demo scenes) | 13 spectral bands, Level-1C, 10 m | Free, full and open Copernicus data policy | "Contains modified Copernicus Sentinel data [year]" |
| **JRC Global Surface Water** (distributed with Sen1Floods11 as `JRCWaterHand`) | Permanent-water mask, subtracted from the flood labels | Free of charge, without restriction of use (Copernicus Programme) | Source: EC JRC/Google; cite Pekel et al. (2016), *Nature* 540 |
| **Demo scenes** from Copernicus Browser | New Sentinel-2 L1C scenes, never used for training | Copernicus open data policy | As above |

No pre-trained model or weights are used.

## Data types

- `S2Hand`: Sentinel-2 L1C top-of-atmosphere reflectance × 10000 (UInt16), 13 bands
  (B1 coastal, B2 blue, B3 green, B4 red, B5–B7 red edge, B8 NIR, B8A narrow NIR, B9 water vapour, B10 cirrus,
  B11 SWIR-1, B12 SWIR-2), 512 × 512 px at 10 m (≈ 5 × 5 km), EPSG:4326.
- `LabelHand`: −1 no data / invalid, 0 not water, 1 water. Drawn by trained analysts.
- `JRCWaterHand`: 1 = permanent water.

## From pixels to one label per chip

flood pixels = water in `LabelHand` **and not** permanent water in `JRCWaterHand`; no-data pixels ignored.
**ALERT (1)** if flood pixels > 1 % of valid pixels, otherwise **NO ALERT (0)**. The 1 % is a project choice: lower
catches smaller floods but turns tiny wet patches into alerts.

## Composition (446 chips, 11 countries)

| Country | Chips | ALERT | NO ALERT | Subset |
|---|---|---|---|---|
| Bolivia | 15 | 11 | 4 | test |
| Ghana | 53 | 19 | 34 | test |
| Spain | 30 | 21 | 9 | val |
| Sri-Lanka | 42 | 25 | 17 | val |
| India | 68 | 58 | 10 | train |
| Mekong | 30 | 24 | 6 | train |
| Nigeria | 18 | 12 | 6 | train |
| Pakistan | 28 | 14 | 14 | train |
| Paraguay | 67 | 38 | 29 | train |
| Somalia | 26 | 15 | 11 | train |
| USA | 69 | 36 | 33 | train |
| **Total** | **446** | **273** | **173** | |

| Subset | Chips | ALERT | NO ALERT |
|---|---|---|---|
| train | 306 | 197 | 109 |
| val | 72 | 46 | 26 |
| test | 68 | 30 | 38 |

*(From the labelling run of 2 Nov 2026. Check against the output of `src/split.py` in notebook 03.)*

## Split protocol and leakage

| Risk | Mitigation |
|---|---|
| Neighbouring chips of one flood are near-duplicates | Split by **country**: a country is in exactly one subset (checked by an assertion and a unit test) |
| Choosing thresholds or the model on test | Thresholds and the final model are chosen on **val** only; test is measured once |
| Test too small to measure false alarms | Ghana added to test: 38 NO ALERT chips instead of 4 |
| Demo images overlapping training areas | Demo scenes come from floods/places not in Sen1Floods11 |
| Same place seen at several dates | Not checked chip by chip. Splitting by country removes it across subsets |

## Biases and limitations

- **Geography:** 11 events, mostly tropical and sub-tropical river floods. Few arid, snowy or urban scenes.
- **"Normal" is narrow:** every NO ALERT chip comes from a flood-affected region at flood time. A model can still meet
  scenes it has never seen (desert, snow, cities), hence the UNKNOWN decision.
- **Class balance:** 61 % ALERT overall, but India and Mekong are mostly ALERT, while Ghana and USA are more balanced.
- **Optical sensor:** clouds and cloud shadows hide or imitate water. Floods often happen under cloud.
- **Labels:** human masks contain errors at water edges. Chips near the 1 % line are the most ambiguous (see the failure analysis).
- **Size:** 68 test chips, so one extra error changes recall by about 3 points and the false-alarm rate by about 3 points.
