"""Country/event-based split. Never split chips of the same event across sets.

Default: Bolivia is fully held out as test (as in the Sen1Floods11 paper).
Remaining countries are assigned to val/test/train with a fixed seed.
"""
import argparse
import csv
import json
import random
from collections import Counter


def make_split(countries, test_fixed=("Bolivia",), n_val=2, n_test_extra=1, seed=0):
    rest = sorted(set(countries) - set(test_fixed))
    rng = random.Random(seed)
    rng.shuffle(rest)
    val = rest[:n_val]
    test = list(test_fixed) + rest[n_val:n_val + n_test_extra]
    train = rest[n_val + n_test_extra:]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--out", default="data/splits.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.labels) as f:
        rows = list(csv.DictReader(f))
    split = make_split([r["country"] for r in rows], seed=args.seed)

    by_country = {c: s for s, cs in split.items() for c in cs}
    stats = Counter((by_country[r["country"]], r["label"]) for r in rows)
    for s in ("train", "val", "test"):
        print(f"{s:5s} countries={split[s]}  ALERT={stats[(s, '1')]}  NO_ALERT={stats[(s, '0')]}")

    assert not (set(split["train"]) & set(split["test"])), "leak: country in train and test"
    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)


if __name__ == "__main__":
    main()
