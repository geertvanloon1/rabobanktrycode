from pathlib import Path
import pandas as pd

FOLDER = Path("data/vintages_nl")
SPEC_PATH = Path("data/Spec_NL.xlsx")

DATE_CANDIDATES = ["date", "time", "period"]

def find_date_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.strip().lower() in DATE_CANDIDATES:
            return c
    raise ValueError(f"No date/time column found in {list(df.columns)}")

def get_required_cols(spec_path: Path) -> list[str]:
    spec = pd.read_excel(spec_path)
    col = next(c for c in spec.columns if c.lower() == "seriesid")
    return spec[col].dropna().astype(str).str.strip().tolist()

def fix_df(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    date_col = find_date_column(df)

    # Add missing spec columns (numeric)
    for c in required:
        if c not in df.columns:
            df[c] = pd.NA

    # Reorder: Date first, everything else after
    rest = [c for c in df.columns if c != date_col]
    return df[[date_col] + rest]

def process_file(path: Path, required_cols: list[str]) -> None:
    xls = pd.ExcelFile(path)
    out = {}

    for sh in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sh)
        out[sh] = fix_df(df, required_cols)

    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as w:
        for sh, df in out.items():
            df.to_excel(w, sheet_name=sh, index=False)

    print(f"Fixed: {path}")

required_cols = get_required_cols(SPEC_PATH)

for p in sorted(FOLDER.glob("*.xlsx")):
    process_file(p, required_cols)
