"""Shared evaluation used for every approach, so results are comparable."""
import csv
import os
import time
from pathlib import Path

import numpy as np


def alert_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_alarm_rate = fp / (fp + tn) if fp + tn else 0.0  # on normal images
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "false_alarm_rate": false_alarm_rate, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def latency_ms(fn, x, repeats=50):
    fn(x)  # warm-up
    t = time.perf_counter()
    for _ in range(repeats):
        fn(x)
    return (time.perf_counter() - t) / repeats * 1000


def size_mb(path):
    return os.path.getsize(path) / 1e6 if path and Path(path).exists() else 0.0


def log_result(row: dict, path="results.csv"):
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)
