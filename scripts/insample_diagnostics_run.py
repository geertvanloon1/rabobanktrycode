import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathlib import Path
import subprocess
from shutil import which
import json
import time

import numpy as np
import pandas as pd

from zhangnowcast.data.data import build_zhang_data, slice_rolling_10y_window
from zhangnowcast.data.zhang_buckets_from_vintages import (
    infer_buckets_month_by_month,
    list_vintages,
    pick_vintage_for_month_q,
)
from zhangnowcast.inference.insamplesampler import run_bay
from zhangnowcast.results.io import init_run_dir, finalize_run_dir, save_subrun_outputs


# paste this into terminal: nohup /usr/local/bin/python3 "/Users/geertvanloon/Documents/erasmus/2025-2026/seminar financial case study /Structured Dutch Zhang 2 copy/scripts/insample_diagnostics_run.py" > run_full.log 2>&1 &



# CONFIG (edit values here)
MODEL_TYPES = ["bay", "bay_sv"]  # ["bay"] or ["bay_sv"] or both
DATA_DIR_REL = Path("data/NL")
SPEC_FILE_REL = Path("data/Spec_NL.xlsx")  # adjust if yours lives elsewhere
CACHE_DIR_REL = Path("output/replication")
RESULTS_DIR_REL = Path("outputs")

VINTAGE_START = "2025-11-14"
VINTAGE_END = "2025-11-14"  # Use the most complete vintage
GDP_SERIES_ID = "Real GDP"
SAMPLE_START = "2002-01-01"
ROLLING_WINDOW_MONTHS = 120
ALLOW_SHORT_START = True

# Faster computation (pilot defaults)
R_MAX = 4
N_ITER = 8000
BURN = 4000
THIN = 1

STANDARDIZE = True
SEED = 123
PAPER_DEFAULTS = True

SAVE_DRAWS = False  # raw draws are big; keep False by default
RESUME = True  # skip (month,q) if already saved
DEBUG = False  # set True for verbose diagnostics


# Path / IO helpers
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _dbg(tag, D):
    last_date = pd.to_datetime(D.dates_m[-1]).date()
    last_obs = int(np.sum(~np.isnan(D.X_m[-1, :])))
    sec_obs = int(np.sum(~np.isnan(D.X_m[-2, :]))) if D.X_m.shape[0] >= 2 else -1
    print(f"DEBUG {tag}: last_date={last_date} last_obs={last_obs} second_last_obs={sec_obs}")


def have_xlrd() -> bool:
    try:
        import xlrd  # noqa: F401
        return True
    except Exception:
        return False


def soffice_exists() -> bool:
    return which("soffice") is not None


def convert_xls_to_xlsx(infile: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{infile.stem}.xlsx"
    if outfile.exists():
        return outfile

    if not soffice_exists():
        raise RuntimeError(
            "Need to read .xls but xlrd is not available and 'soffice' was not found. "
            "Either provide .xlsx files or ensure LibreOffice is available."
        )

    cmd = [
        "soffice",
        "--headless",
        "--convert-to", "xlsx",
        "--outdir", str(outdir),
        str(infile),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not outfile.exists():
        raise RuntimeError(f"Conversion failed; expected {outfile}")
    return outfile


def prepare_xlsx_cache(root: Path) -> tuple[Path, Path, str]:
    """
    If xlrd is missing, convert:
      - spec .xls -> .xlsx
      - all vintages in [VINTAGE_START, VINTAGE_END] -> .xlsx
    Returns: (vintage_dir, spec_file, pattern)
    """
    data_dir = root / DATA_DIR_REL
    spec_xls = root / SPEC_FILE_REL

    if have_xlrd():
        return data_dir, spec_xls, "*.xlsx"


    cache_dir = root / CACHE_DIR_REL / "_xlsx_cache"
    cache_vint = cache_dir / "NL"
    cache_spec_dir = cache_dir / "spec"

    spec_xlsx = convert_xls_to_xlsx(spec_xls, cache_spec_dir)

    start = pd.Timestamp(VINTAGE_START)
    end = pd.Timestamp(VINTAGE_END)

    # We don't have list_vintage_files() in this repo; glob is enough.
    for p in sorted(data_dir.glob("*.xlsx")):
        try:
            d = pd.Timestamp(p.stem)
        except Exception:
            continue
        if start <= d <= end:
            convert_xls_to_xlsx(p, cache_vint)

    return cache_vint, spec_xlsx, "*.xlsx"


def model_use_sv(model_type: str) -> bool:
    mt = model_type.lower().strip()
    if mt == "bay":
        return False
    if mt in ("bay_sv", "baysv", "bay-sv"):
        return True
    raise ValueError(f"Unknown MODEL_TYPE '{model_type}' (expected 'bay' or 'bay_sv').")


# Core correctness helper: TRIM to month_T
def trim_data_to_month_T(D, month_T: pd.Timestamp):
    """
    Ensure the monthly panel ends at month_T (inclusive).
    This is critical: 'T' defines the end of the information set in Zhang.
    If the chosen vintage is later, D may include future-month rows; we must drop them.
    """
    month_T = pd.to_datetime(month_T).normalize().replace(day=1)
    dm = pd.to_datetime(D.dates_m).normalize()

    idx_arr = np.where(dm == month_T)[0]
    if idx_arr.size == 0:
        raise ValueError(f"month_T={month_T.date()} not found in D.dates_m (unexpected).")
    end_idx = int(idx_arr[0])

    D.X_m = D.X_m[: end_idx + 1, :]
    D.dates_m = D.dates_m[: end_idx + 1]
    D.month_pos_in_quarter = D.month_pos_in_quarter[: end_idx + 1]
    D.quarter_of_month = D.quarter_of_month[: end_idx + 1]
    D.obs_idx = list(D.obs_idx[: end_idx + 1])
    return D


def month_in_quarter(month_T: pd.Timestamp) -> int:
    m = int(pd.to_datetime(month_T).month)
    return ((m - 1) % 3) + 1


# Main
def main():
    root = project_root()
    vintage_dir, spec_file, pattern = prepare_xlsx_cache(root)

    start = pd.Timestamp(VINTAGE_START)
    end = pd.Timestamp(VINTAGE_END)

    # Vintage objects (date + path), filtered to replication window
    vintages_all = list_vintages(Path(vintage_dir), pattern=pattern)
    vintages = [v for v in vintages_all if start <= v.date <= end]
    if not vintages:
        raise RuntimeError(f"No vintage files found in {vintage_dir} for [{start.date()}..{end.date()}]")

    # Get the most complete vintage (i.e., the last vintage)
    latest_vintage = vintages[-1]
    print(f"Using the latest vintage: {latest_vintage.date}")

    # Get the path of the most complete vintage
    vintage_path = latest_vintage.path
    print(f"Using vintage data from: {vintage_path}")

    # Build dataset from vintage
    D_v = build_zhang_data(
        vintage_file=vintage_path,
        spec_file=spec_file,
        gdp_series_id=GDP_SERIES_ID,
        sample_start=SAMPLE_START,
    )

    # Skip rolling window slicing and use the full data
    D_v = trim_data_to_month_T(D_v, latest_vintage.date)

    # Results-ready run directory (outputs/<run_id>/...)
    batch_config = dict(
        model_types=MODEL_TYPES,
        R_max=R_MAX,
        n_iter=N_ITER,
        burn=BURN,
        thin=THIN,
        standardize=STANDARDIZE,
        seed=SEED,
        paper_defaults=PAPER_DEFAULTS,
        rolling_window_months=ROLLING_WINDOW_MONTHS,
        allow_short_start=ALLOW_SHORT_START,
        vintage_start=VINTAGE_START,
        vintage_end=VINTAGE_END,
        save_raw_draws=bool(SAVE_DRAWS),
        resume=bool(RESUME),
    )
    data_config = dict(
        data_dir=str(DATA_DIR_REL),
        spec_file=str(SPEC_FILE_REL),
        gdp_series_id=GDP_SERIES_ID,
        sample_start=SAMPLE_START,
        pattern=pattern,
    )

    t0 = time.time()
    run_dir = init_run_dir(root / RESULTS_DIR_REL, batch_config=batch_config, data_config=data_config)

    all_rows = []

    for model_type in MODEL_TYPES:
        use_sv = model_use_sv(model_type)
        per_run_dir = run_dir / "nowcasts" / "per_run" / model_type
        per_run_dir.mkdir(parents=True, exist_ok=True)

        rows = []

        # Skip the month and quarter loop as you only have one vintage
        tag = f"vintage={latest_vintage.date}"
        out_row_path = per_run_dir / f"{tag}.csv"

        if RESUME and out_row_path.exists():
            row = pd.read_csv(out_row_path).iloc[0].to_dict()
            rows.append(row)
            all_rows.append(row)
            if not DEBUG:
                print(f"[{model_type}] skip {tag} (exists)")
        else:
            print(f"Running model for vintage: {vintage_path.name}")

            # Run the model
            res = run_bay(
                D_v,
                R_max=R_MAX,
                n_iter=N_ITER,
                burn=BURN,
                thin=THIN,
                standardize=STANDARDIZE,
                seed=SEED,
                paper_defaults=PAPER_DEFAULTS,
                use_sv=use_sv,
                do_selection=False,
            )

            # Check if result is empty
            if not res:
                print(f"[{model_type}] Warning: No results for vintage {vintage_path.name}")
                continue

            # Save results
            now_rows, _diag = save_subrun_outputs(
                run_dir=run_dir,
                model_type=model_type,
                tag=tag,
                D=D_v,
                res=res,
                month_T=latest_vintage.date,
                q=1,
                vintage_file=vintage_path.name,
                target_quarter=str(pd.Period(latest_vintage.date, freq="Q") + 1),
            )

            for row in now_rows:
                row["run_id"] = run_dir.name
                row.setdefault("month_in_quarter", int(month_in_quarter(latest_vintage.date)))

                # Save per-run row immediately (resume-safe)
                pd.DataFrame([row]).to_csv(out_row_path, index=False)

                rows.append(row)
                all_rows.append(row)


        # Store the results as a DataFrame
        if rows:
            df = pd.DataFrame(rows).sort_values(["month", "q"]).reset_index(drop=True)
            out_csv = run_dir / "nowcasts" / f"nowcasts_{model_type}.csv"
            df.to_csv(out_csv, index=False)
            print(f"Saved {out_csv}")

    if all_rows:
        df_all = pd.DataFrame(all_rows).sort_values(["model", "month", "q"]).reset_index(drop=True)
        (run_dir / "nowcasts").mkdir(parents=True, exist_ok=True)
        df_all.to_csv(run_dir / "nowcasts" / "nowcasts.csv", index=False)

    # Diagnostics summary
    diag_files = sorted((run_dir / "posterior" / "per_run").glob("*/*/diagnostics.json"))
    diag_rows = []
    for f in diag_files:
        try:
            d = json.loads(f.read_text())
            model = f.parents[1].name
            tag = f.parent.name
            d["model"] = model
            d["subrun_tag"] = tag
            diag_rows.append(d)
        except Exception:
            continue
    if diag_rows:
        pd.DataFrame(diag_rows).to_csv(run_dir / "diagnostics" / "diagnostics_summary.csv", index=False)

    finalize_run_dir(run_dir, start_time=t0)
    print(f"Done. Results saved under: {run_dir}")


if __name__ == "__main__":
    main()



