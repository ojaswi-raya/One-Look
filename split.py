"""Geographic split: every chip from a country stays in one subset.

Default split (fixed, not random, so every run uses the same one):
  test  = Bolivia + Ghana    -> never seen in training; Bolivia follows the dataset
                                 authors, Ghana adds enough NO ALERT chips to measure
                                 the false-alarm rate (Bolivia alone has only 4).
  val   = Spain + Sri-Lanka  -> used only to choose thresholds and the final model
  train = all other countries
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DEFAULT_TEST = ("Bolivia", "Ghana")
DEFAULT_VAL = ("Spain", "Sri-Lanka")


def make_split(countries, test=DEFAULT_TEST, val=DEFAULT_VAL):
    countries = sorted(set(countries))
    test = [c for c in test if c in countries]
    val = [c for c in val if c in countries and c not in test]
    train = [c for c in countries if c not in test and c not in val]
    if not test or not train:
        raise ValueError(f"Need at least one train and one test country; found {countries}")
    return {"train": train, "val": sorted(val), "test": sorted(test)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--out", default="data/splits.json")
    ap.add_argument("--test", nargs="+", default=list(DEFAULT_TEST))
    ap.add_argument("--val", nargs="+", default=list(DEFAULT_VAL))
    a = ap.parse_args()

    with open(a.labels, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Labels CSV is empty.")
    split = make_split([r["country"] for r in rows], a.test, a.val)

    owner = {c: s for s, cs in split.items() for c in cs}
    assert not (set(split["train"]) & set(split["test"])), "leak: country in train and test"
    counts = Counter((owner[r["country"]], int(r["label"])) for r in rows)
    for s in ("train", "val", "test"):
        print(f"{s:5s} countries={split[s]} | ALERT={counts[(s, 1)]} | NO ALERT={counts[(s, 0)]}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Saved geographic split to {out}")


if __name__ == "__main__":
    main()
