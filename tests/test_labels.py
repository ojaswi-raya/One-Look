import numpy as np
from src.labels import flood_fraction, chip_label

def test_flood_fraction_excludes_permanent_water():
    label=np.array([[1,1,0],[-1,0,1]])
    permanent=np.array([[1,0,0],[0,0,0]])
    assert abs(flood_fraction(label,permanent)-2/5)<1e-9

def test_threshold():
    assert chip_label(0.02,0.01)==1
    assert chip_label(0.005,0.01)==0


def test_split_keeps_countries_apart():
    from src.split import make_split
    countries = ["Bolivia", "Ghana", "India", "Spain", "Sri-Lanka", "USA"]
    s = make_split(countries)
    assert s["test"] == ["Bolivia", "Ghana"] and s["val"] == ["Spain", "Sri-Lanka"]
    assert not set(s["train"]) & set(s["test"]) and not set(s["train"]) & set(s["val"])
