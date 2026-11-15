"""The three UCI datasets of the simulation study (Table 2 of the paper)."""

import os
import xml.etree.ElementTree as ET
import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()


def _default_data_dir():
    """$EMPUTATION_DATA_DIR, else a local simulation/data folder, else the data
    folder of the R simulation: Code/simulation/data when this folder sits next to
    the R repository Code/, or simulation/data when it sits inside it."""
    if os.environ.get("EMPUTATION_DATA_DIR"):
        return Path(os.environ["EMPUTATION_DATA_DIR"])
    root = _HERE.parents[3]
    candidates = (_HERE.parents[1] / "data", root / "Code" / "simulation" / "data", root / "simulation" / "data")
    return next((p for p in candidates if (p / "concrete_data.csv").exists()), candidates[0])


DATA_DIR = _default_data_dir()


@lru_cache(maxsize=None)
def load_dataset(name, data_dir=None):
    """Load a complete dataset.

    Args:
        name: "concrete" (n=1030, d=9), "wine" (white wine quality, n=4898,
            d=12) or "ccpp" (combined cycle power plant, n=9568, d=5; the
            first sheet of Folds_pp.xlsx, as read by the R code).
        data_dir: folder with the raw files. Defaults to ``DATA_DIR`` (the R simulation's data folder).

    Returns:
        tuple: (float64 array of shape (n, d), list of column names)
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    if name == "concrete":
        df = pd.read_csv(data_dir / "concrete_data.csv")
    elif name == "wine":
        df = pd.read_csv(data_dir / "winequality-white.csv", sep=";")
    elif name == "ccpp":
        df = read_xlsx_first_sheet(data_dir / "Folds_pp.xlsx")
    else:
        raise ValueError(f"Unknown dataset {name!r}; expected concrete, wine or ccpp.")
    df.columns = [c.strip() for c in df.columns]
    X = df.to_numpy(dtype=float)
    X.setflags(write=False)
    return X, list(df.columns)


_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
       "rel": "http://schemas.openxmlformats.org/package/2006/relationships"}


def _column(ref):
    """Zero-based column of a cell reference, e.g. "C12" -> 2."""
    letters = "".join(ch for ch in ref if ch.isalpha())
    index = 0
    for ch in letters.upper():
        index = index * 26 + ord(ch) - 64
    return index - 1


def read_xlsx_first_sheet(path):
    """First worksheet of an .xlsx file as a DataFrame, header in the first row.

    Uses only the standard library (an .xlsx file is a zip archive of XML
    files), so no Excel package is needed; for the CCPP table (text header,
    numeric cells) it returns the same values as ``readxl::read_xlsx`` and
    ``pandas.read_excel``.
    """
    with zipfile.ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            sst = ET.fromstring(z.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")) for si in sst.findall("m:si", _NS)]
        rid = ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets/m:sheet", _NS).get(f"{{{_NS['r']}}}id")
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = next(rel.get("Target") for rel in rels.findall("rel:Relationship", _NS) if rel.get("Id") == rid)
        sheet = ET.fromstring(z.read(target.lstrip("/") if target.startswith("/") else "xl/" + target))
    rows = []
    for row in sheet.iter(f"{{{_NS['m']}}}row"):
        cells = {}
        for c in row.findall("m:c", _NS):
            v = c.find("m:v", _NS)
            if v is not None:
                col = _column(c.get("r")) if c.get("r") else len(cells)
                cells[col] = strings[int(v.text)] if c.get("t") == "s" else float(v.text)
        rows.append(cells)
    ncol = max(max(r) + 1 for r in rows if r)
    table = [[r.get(j, float("nan")) for j in range(ncol)] for r in rows]
    return pd.DataFrame(table[1:], columns=table[0])
