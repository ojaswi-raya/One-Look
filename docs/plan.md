# Plan and status

Deadline: **16 Nov 2026**. Core work: 30 Oct → 5 Nov. Buffer: 6 → 16 Nov.

| Day | Date | Goal | Status |
|---|---|---|---|
| 0 | 29 Oct | Scope, alert definition, data plan, split protocol, repo + label/split/baseline/metrics code | ✅ Done |
| 1 | 30 Oct | Download Sen1Floods11 (hand-labelled), run labels + split, fill data card composition, hard negatives | ⏳ |
| 2 | 31 Oct | Baselines B1 (MNDWI) + B2 (RF on band stats), first rows in results.csv | ⏳ |
| 3 | 1 Nov | Small CNN (MobileNetV3-Small, TorchGeo weights), train on Colab | ⏳ |
| 4 | 2 Nov | Threshold for target false-alarm rate, hard negatives, S1+S2 fusion, compare | ⏳ |
| 5 | 3 Nov | UNKNOWN detection (max-softmax / energy / Mahalanobis), AUROC on OOD scenes | ⏳ |
| 6 | 4 Nov | ONNX + int8, size and latency; failure-case analysis | ⏳ |
| 7 | 5 Nov | Gradio demo + Grad-CAM, test on new Copernicus scenes, README, data card final | ⏳ |
| — | 6–16 Nov | Buffer: mentor feedback, optional 2nd anomaly (burned areas, CaBuAr), polish | |

## Open questions for mentor
1. Is one anomaly type (floods) enough, or are several expected?
2. What format will the organisers' unseen test images use (S2 multispectral, RGB, S1)?
3. Any embedding targets (max MB, max ms, target hardware)?
4. Image-level decision vs segmentation: is image-level acceptable?
5. Meeting rhythm and what to show next time.
