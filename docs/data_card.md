# Data card (draft v0, 29 Oct 2026)

## Sources

| Source | Content | Use | Licence | Status |
|---|---|---|---|---|
| Sen1Floods11 v1.1 (Bonafilia et al., CVPR-W 2020) | 512×512 chips, 10 m; S1 (VV, VH, dB) + S2 (13 bands, TOA ×10000); 446 hand-labelled chips from 11 flood events; JRC permanent-water layer | Train / val / test | To confirm from the official repo / bucket README before submission | Planned |
| CloudSEN12 | Sentinel-2 chips with cloud / shadow masks | Hard negatives (clouds must not alert) | To confirm (dataset page) | Planned |
| Copernicus Sentinel data (via Copernicus Browser) | New S1/S2 scenes | Demo on unseen images | Free and open Copernicus data policy; attribution "Contains modified Copernicus Sentinel data [year]" | Planned |
| Copernicus EMS | Activation list for recent floods | Choosing demo scenes (not used for training) | Check per-product terms | Planned |
| Pre-trained weights (TorchGeo) | ImageNet or Sentinel-2 self-supervised weights | Backbone initialisation | Per-weight licence, to record | Planned |

## Label meaning (Sen1Floods11)
`-1` = no data / invalid · `0` = not water · `1` = water.
Permanent water is removed using the JRC layer **before** computing the flood fraction, so that rivers and lakes do not cause alerts.

## Split protocol
- Unit of splitting = **country / flood event**, never single chips.
- **Test:** Bolivia (fully held out, geographic generalisation) + 1–2 other events.
- **Val:** 1–2 events, used to tune thresholds only.
- **Train:** remaining events.
- The split is saved to `data/splits.json` and committed, so every experiment uses the same split.

## Leakage risks and mitigations
| Risk | Mitigation |
|---|---|
| Neighbouring chips from the same event are near-duplicates | Split by event / country |
| Same location seen at different dates | Check chip footprints across splits |
| Tuning thresholds on test | Thresholds tuned on val only; test touched once per final model |
| Pretrained weights trained on overlapping scenes | Record weight provenance |
| Demo images overlapping training areas | Choose demo scenes from events not in Sen1Floods11 |

## Known biases and limitations (to complete)
- Only 11 events, so the climate and land cover coverage is limited.
- Optical (S2) bands fail under cloud; S1 radar handles this but has speckle and is confused by smooth surfaces (tarmac, sand).
- Hand labels contain annotator errors at water edges.
- Class balance (alert vs no alert) depends on the threshold T; this will be reported once the data is downloaded.

## Composition
To be filled after download: number of chips per country, per split, per class.
