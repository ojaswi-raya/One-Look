"""Run every experiment of the project in one go and write all results.

Steps
 1. Read each chip once: features (src/features.py), MNDWI water fractions, permanent-water share.
 2. Approaches, simplest first, all evaluated the same way:
      B1  MNDWI rule            no learning (physical water index)
      M1  Logistic regression   smallest learned model
      M2  Random forest, small  50 trees, depth 6
      M3  Random forest, large  300 trees
    Each one: trained on TRAIN, thresholds chosen on VAL (max recall with
    false-alarm rate <= 5 %), then measured once on TEST.
 3. The final model is chosen on VAL only (best F1 among those meeting the false-alarm target).
 4. UNKNOWN detector: Mahalanobis distance of a chip's features from the training data.
    Far from everything seen in training -> UNKNOWN. Evaluated on real test chips and on
    synthetic out-of-distribution versions of them (cloud, snow, sand, noise, dead bands).
 5. Embeddability: model size (bytes on disk) and CPU time per chip.
 6. Failure analysis: every wrong test decision with clues, plus a picture grid.

Outputs (in --out): results.csv, results.md, ood.csv, failures.csv, failures.png,
summary.json, one_look_model.joblib (used by src/predict.py)
"""
import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import alert_metrics  # noqa: E402
from src.features import (FEATURE_VERSION, chip_features, mndwi, read_s2,  # noqa: E402
                          rgb, valid_mask)

from sklearn.covariance import LedoitWolf  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

FAR_TARGET = 0.05
TAUS = np.round(np.arange(-0.2, 0.61, 0.05), 2)          # MNDWI cut-offs tried by B1
AREA_THRS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)           # share of water pixels for B1 ALERT
OOD_QUANTILE = 0.99


# ---------------------------------------------------------------- helpers
def choose_threshold(y, score, candidates):
    """Highest recall with false-alarm rate <= target; if impossible, lowest false-alarm rate."""
    best, fallback = None, None
    for t in candidates:
        m = alert_metrics(y, (score >= t).astype(int))
        key_ok = (m["recall"], m["precision"])
        if m["false_alarm_rate"] <= FAR_TARGET and (best is None or key_ok > best[0]):
            best = (key_ok, t)
        key_fb = (-m["false_alarm_rate"], m["recall"])
        if fallback is None or key_fb > fallback[0]:
            fallback = (key_fb, t)
    return float((best or fallback)[1]), best is not None


def nbytes(obj):
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.tell()


def per_chip_ms(fn, repeats=200):
    fn()
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t) / repeats * 1000


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


# ---------------------------------------------------------------- synthetic OOD
def make_shift(x, kind, rng):
    """Synthetic stand-ins for scenes the model never saw. Units: reflectance x 10000."""
    x = x.copy()
    v = valid_mask(x)
    if kind == "thick_cloud":
        cloud = np.array([6500, 6800, 7000, 7100, 7200, 7200, 7300, 7300, 7300, 3000, 1500, 4500, 3500],
                         "float32")[:, None, None]
        x = 0.15 * x + 0.85 * cloud
    elif kind == "snow":
        snow = np.array([8500, 8800, 9000, 8900, 8800, 8600, 8400, 8200, 8000, 2500, 100, 900, 600],
                        "float32")[:, None, None]
        x = 0.1 * x + 0.9 * snow
    elif kind == "desert_sand":
        sand = np.array([1800, 2200, 3000, 3900, 4300, 4600, 4800, 5000, 5100, 1200, 30, 6200, 5300],
                        "float32")[:, None, None]
        x = 0.1 * x + 0.9 * sand
    elif kind == "sensor_noise":
        x = x + rng.normal(0, 2500, x.shape).astype("float32")
    elif kind == "dead_bands":
        x[rng.choice(13, 5, replace=False)] = 0
    x = np.clip(x, 0, None)
    x[:, ~v] = 0
    return x


SHIFTS = ("thick_cloud", "snow", "desert_sand", "sensor_noise", "dead_bands")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder with S2Hand/, LabelHand/, JRCWaterHand/")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--ood-quantile", type=float, default=OOD_QUANTILE,
                    help="share of train+val chips that must stay below the UNKNOWN limit (default 0.99)")
    a = ap.parse_args()
    q = a.ood_quantile
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    rows = list(csv.DictReader(open(a.labels, encoding="utf-8")))
    split = json.loads(Path(a.splits).read_text(encoding="utf-8"))
    owner = {c: s for s, cs in split.items() for c in cs}

    # 1. read every chip once
    print(f"Reading {len(rows)} chips ...")
    data = []
    t_feat = []
    for i, r in enumerate(rows):
        x = read_s2(root / "S2Hand" / f"{r['chip_id']}_S2Hand.tif")
        t0 = time.perf_counter()
        f = chip_features(x)
        t_feat.append(time.perf_counter() - t0)
        if f is None:
            print(f"  skipping {r['chip_id']}: almost no valid pixels")
            continue
        v = valid_mask(x)
        m = mndwi(x)[v]
        wf = np.array([(m > t).mean() for t in TAUS], "float32")
        jp = root / "JRCWaterHand" / f"{r['chip_id']}_JRCWaterHand.tif"
        perm = 0.0
        if jp.exists():
            with rasterio.open(jp) as ds:
                perm = float((ds.read(1) == 1)[v].mean())
        data.append(dict(id=r["chip_id"], country=r["country"], split=owner.get(r["country"]),
                         y=int(r["label"]), flood=float(r["flood_fraction"]), f=f, wf=wf, perm=perm,
                         blue=float(x[1][v].mean())))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}")
    feat_ms = float(np.median(t_feat) * 1000)

    S = {s: [d for d in data if d["split"] == s] for s in ("train", "val", "test")}
    for s, ds in S.items():
        print(f"{s:5s}: {len(ds)} chips, ALERT={sum(d['y'] for d in ds)}, NO ALERT={sum(1 - d['y'] for d in ds)}")
    X = {s: np.stack([d["f"] for d in ds]) for s, ds in S.items()}
    Y = {s: np.array([d["y"] for d in ds]) for s, ds in S.items()}
    WF = {s: np.stack([d["wf"] for d in ds]) for s, ds in S.items()}

    results, finals = [], {}

    # 2a. B1: MNDWI rule (tau and area threshold chosen on val)
    best = None
    for ti, tau in enumerate(TAUS):
        for thr in AREA_THRS:
            m = alert_metrics(Y["val"], (WF["val"][:, ti] > thr).astype(int))
            ok = m["false_alarm_rate"] <= FAR_TARGET
            key = (ok, m["recall"] if ok else -m["false_alarm_rate"], m["precision"])
            if best is None or key > best[0]:
                best = (key, ti, thr)
    _, ti, thr = best
    pred = (WF["test"][:, ti] > thr).astype(int)
    m_val = alert_metrics(Y["val"], (WF["val"][:, ti] > thr).astype(int))
    m = alert_metrics(Y["test"], pred)
    m["roc_auc"] = float(roc_auc_score(Y["test"], WF["test"][:, ti])) if len(set(Y["test"])) == 2 else float("nan")
    results.append(dict(approach="B1 MNDWI rule (no learning)", params=f"MNDWI>{TAUS[ti]}, area>{thr}",
                        val_f1=m_val["f1"], val_far=m_val["false_alarm_rate"], **m,
                        size_kb=0.0, ms_per_chip=feat_ms, met_far_target_on_val=best[0][0]))

    # 2b. learned models
    models = {
        "M1 Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=5000)),
        "M2 Random forest small (50 trees, depth 6)": RandomForestClassifier(
            n_estimators=50, max_depth=6, class_weight="balanced", random_state=0, n_jobs=1),
        "M3 Random forest large (300 trees)": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=1),
    }
    for name, model in models.items():
        model.fit(X["train"], Y["train"])
        pv = model.predict_proba(X["val"])[:, 1]
        t, ok = choose_threshold(Y["val"], pv, np.unique(np.r_[0.02:0.99:0.01, pv]))
        m_val = alert_metrics(Y["val"], (pv >= t).astype(int))
        pt = model.predict_proba(X["test"])[:, 1]
        m = alert_metrics(Y["test"], (pt >= t).astype(int))
        m["roc_auc"] = float(roc_auc_score(Y["test"], pt)) if len(set(Y["test"])) == 2 else float("nan")
        one = X["test"][:1]
        ms = feat_ms + per_chip_ms(lambda: model.predict_proba(one), repeats=50 if "large" in name else 200)
        results.append(dict(approach=name, params=f"alert if p>={t:.2f}", val_f1=m_val["f1"],
                            val_far=m_val["false_alarm_rate"], **m, size_kb=nbytes(model) / 1024,
                            ms_per_chip=ms, met_far_target_on_val=ok))
        finals[name] = (model, t, pt, ok, m_val["f1"])

    # 3. choose final model on VAL only
    ranked = sorted(finals.items(), key=lambda kv: (kv[1][3], kv[1][4], -nbytes(kv[1][0])), reverse=True)
    final_name, (final_model, final_t, final_pt, _, _) = ranked[0]
    print(f"Final model (chosen on validation): {final_name}")

    # 4. UNKNOWN detector
    scaler = StandardScaler().fit(X["train"])
    lw = LedoitWolf().fit(scaler.transform(X["train"]))
    dist = lambda F: lw.mahalanobis(scaler.transform(F))  # noqa: E731
    t_ood = float(np.quantile(np.r_[dist(X["train"]), dist(X["val"])], q))
    d_test = dist(X["test"])
    ood_rows = [dict(scene="real test chips (in-distribution)", n=len(d_test),
                     flagged_unknown=float((d_test > t_ood).mean()), auroc_vs_real=float("nan"))]
    shift_decisions = {}
    test_ids = [d["id"] for d in S["test"]]
    for kind in SHIFTS:
        F = []
        for cid in test_ids:
            fs = chip_features(make_shift(read_s2(root / "S2Hand" / f"{cid}_S2Hand.tif"), kind, rng))
            if fs is not None:
                F.append(fs)
        F = np.stack(F)
        d_s = dist(F)
        auroc = roc_auc_score(np.r_[np.zeros(len(d_test)), np.ones(len(d_s))], np.r_[d_test, d_s])
        unknown = d_s > t_ood
        alerts = (final_model.predict_proba(F)[:, 1] >= final_t)
        shift_decisions[kind] = dict(unknown=float(unknown.mean()),
                                     alert_without_unknown_check=float(alerts.mean()),
                                     alert_after_unknown_check=float((alerts & ~unknown).mean()))
        ood_rows.append(dict(scene=f"synthetic: {kind}", n=len(d_s), flagged_unknown=float(unknown.mean()),
                             auroc_vs_real=float(auroc)))

    # 3-way decisions on real test chips
    decision = np.where(d_test > t_ood, "UNKNOWN", np.where(final_pt >= final_t, "ALERT", "NO ALERT"))
    three_way = {f"{lab}->{dec}": int(((Y['test'] == (lab == 'ALERT')) & (decision == dec)).sum())
                 for lab in ("ALERT", "NO ALERT") for dec in ("ALERT", "NO ALERT", "UNKNOWN")}

    # 5. final bundle (what goes on the satellite) + its size and speed
    bundle = dict(model=final_model, threshold=final_t, ood_scaler=scaler, ood_cov=lw, ood_threshold=t_ood,
                  name=final_name, feature_version=FEATURE_VERSION,
                  label_rule="ALERT if non-permanent water > 1% of valid pixels")
    bpath = out / "one_look_model.joblib"
    joblib.dump(bundle, bpath)
    one = X["test"][:1]
    bundle_ms = feat_ms + per_chip_ms(lambda: (final_model.predict_proba(one), dist(one)), repeats=50)

    # 6. failure analysis
    fails = []
    for d, p, dt, dec in zip(S["test"], final_pt, d_test, decision):
        pred = int(p >= final_t)
        if pred == d["y"]:
            continue
        clues = []
        if d["y"] == 1 and d["flood"] < 0.05:
            clues.append("small flood (<5% of chip)")
        if d["y"] == 0 and d["flood"] > 0.002:
            clues.append("some flood water just under the 1% rule")
        if d["perm"] > 0.05:
            clues.append(f"permanent water {d['perm']:.0%} of chip (river/lake)")
        if d["blue"] > 2000:
            clues.append("bright/hazy scene - possible cloud")
        if dec == "UNKNOWN":
            clues.append("also flagged UNKNOWN")
        fails.append(dict(chip_id=d["id"], country=d["country"],
                          error="FALSE ALARM" if pred else "MISSED FLOOD",
                          true_flood_fraction=round(d["flood"], 4), alert_probability=round(float(p), 3),
                          permanent_water_fraction=round(d["perm"], 4), unknown_score=round(float(dt), 1),
                          clues="; ".join(clues) or "no obvious clue - inspect image"))
    with open(out / "failures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fails[0]) if fails else ["chip_id"])
        w.writeheader()
        w.writerows(fails)
    save_failure_grid(fails[:12], root, out / "failures.png")

    # write tables
    keys = ["approach", "params", "precision", "recall", "f1", "false_alarm_rate", "roc_auc",
            "tp", "fp", "fn", "tn", "val_f1", "val_far", "met_far_target_on_val", "size_kb", "ms_per_chip"]
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    with open(out / "ood.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ood_rows[0]))
        w.writeheader()
        w.writerows(ood_rows)

    md = ["## Comparison of approaches (TEST = " + ", ".join(split["test"]) + ", never seen in training)", "",
          "| Approach | Precision | Recall | F1 | False-alarm rate | ROC-AUC | Size (KB) | ms / chip (CPU) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['approach']} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} | "
                  f"{r['false_alarm_rate']:.2f} | {r['roc_auc']:.2f} | {r['size_kb']:.0f} | {r['ms_per_chip']:.1f} |")
    md += ["", f"Thresholds chosen on validation ({', '.join(split['val'])}) to keep false alarms <= "
           f"{FAR_TARGET:.0%}. Final model chosen on validation: **{final_name}**. "
           f"Full bundle (model + UNKNOWN detector): {bpath.stat().st_size / 1024:.0f} KB, "
           f"{bundle_ms:.1f} ms per chip on CPU (feature extraction {feat_ms:.1f} ms).", "",
           "## UNKNOWN detection", "",
           f"UNKNOWN if Mahalanobis distance > {t_ood:.1f} (the {q:.1%} quantile on train+val).", "",
           "| Scenes | n | Flagged UNKNOWN | AUROC vs real test chips |", "|---|---|---|---|"]
    for r in ood_rows:
        md.append(f"| {r['scene']} | {r['n']} | {r['flagged_unknown']:.0%} | {fmt(r['auroc_vs_real'])} |")
    md += ["", "Synthetic scenes are test chips with simulated conditions, not real imagery.", "",
           "## Three-way decisions on real test chips", "", "| True \\ Decision | ALERT | NO ALERT | UNKNOWN |",
           "|---|---|---|---|"]
    for lab in ("ALERT", "NO ALERT"):
        md.append(f"| {lab} | " + " | ".join(str(three_way[f'{lab}->{d}']) for d in ("ALERT", "NO ALERT", "UNKNOWN")) + " |")
    md += ["", f"## Failure cases: {len(fails)} wrong test decisions (see failures.csv / failures.png)"]
    (out / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    summary = dict(final_model=final_name, alert_threshold=final_t, unknown_threshold=t_ood,
                   bundle_kb=bpath.stat().st_size / 1024, bundle_ms_per_chip=bundle_ms,
                   feature_ms_per_chip=feat_ms, split=split, three_way=three_way,
                   synthetic_shift_decisions=shift_decisions, n_failures=len(fails))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nAll outputs in {out}/")


def save_failure_grid(fails, root, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not fails:
        return
    fig, axes = plt.subplots(len(fails), 2, figsize=(6, 3 * len(fails)), squeeze=False)
    for (a1, a2), fl in zip(axes, fails):
        x = read_s2(root / "S2Hand" / f"{fl['chip_id']}_S2Hand.tif")
        with rasterio.open(root / "LabelHand" / f"{fl['chip_id']}_LabelHand.tif") as ds:
            lab = ds.read(1)
        a1.imshow(rgb(x))
        a1.set_title(f"{fl['error']}: {fl['chip_id']}\np={fl['alert_probability']}", fontsize=8)
        a2.imshow(lab, vmin=-1, vmax=1, cmap="Blues")
        a2.set_title(f"true flood {fl['true_flood_fraction']:.1%}", fontsize=8)
        a1.axis("off")
        a2.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
