import os

# -----------------------------------------------------------------------------
# IMPORTANT: set these BEFORE numpy/pandas import to avoid oversubscription.
# This helps ensure "2 processes -> ~2 cores" instead of each process using many
# threads internally.
# -----------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import subprocess
from shutil import which
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from zhangnowcast.data.data import build_zhang_data, slice_rolling_10y_window
from zhangnowcast.data.zhang_buckets_from_vintages import (
    infer_buckets_month_by_month,
    list_vintages,
    pick_vintage_for_month_q,
)
from zhangnowcast.inference.sampler import run_bay
from zhangnowcast.results.io import init_run_dir, finalize_run_dir, save_subrun_outputs


# =============================================================================
# CONFIG (edit here)
# =============================================================================

MODEL_TYPES = ["bay", "bay_sv"]  # ["bay"] or ["bay_sv"] or both

DATA_DIR_REL = Path("data/NL")
SPEC_FILE_REL = Path("data/Spec_NL.xlsx")

CACHE_DIR_REL = Path("output/replication")
RESULTS_DIR_REL = Path("outputs")

# Long horizon
VINTAGE_START = "2015-01-01"
VINTAGE_END = "2025-11-14"

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

SAVE_DRAWS = True
RESUME = True
DEBUG = False

do_selection = False

# -------------------------------------------------------------------------
# NEW: choose frequency of *saved* nowcasts
# -------------------------------------------------------------------------
RUN_MODE = "eom"   # "eom" or "eoq"
Q_FIXED = 3        # end-of-month / last-release snapshot


# =============================================================================
# Path / IO helpers
# =============================================================================

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def trim_data_to_month_T(D, month_T: pd.Timestamp):
    """
    Ensure the monthly panel ends at month_T (inclusive).
    This is critical: 'T' defines the end of the information set.
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


def pick_months(start: pd.Timestamp, end: pd.Timestamp, run_mode: str) -> list[pd.Timestamp]:
    start_month = start.normalize().replace(day=1)
    end_month = end.normalize().replace(day=1)
    months_all = list(pd.date_range(start_month, end_month, freq="MS"))
    if run_mode.lower().strip() == "eom":
        return months_all
    if run_mode.lower().strip() == "eoq":
        return [m for m in months_all if month_in_quarter(m) == 3]
    raise ValueError("RUN_MODE must be 'eom' or 'eoq'")


# =============================================================================
# Worker: run one model_type exactly as before
# =============================================================================

def run_one_model_type(
    model_type: str,
    months: list[pd.Timestamp],
    vintages,
    run_dir: Path,
    spec_file: Path,
) -> list[dict]:
    use_sv = model_use_sv(model_type)

    per_run_dir = run_dir / "nowcasts" / "per_run" / model_type
    per_run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for month_T in months:
        month_T0 = pd.to_datetime(month_T).normalize().replace(day=1)
        q = int(Q_FIXED)

        tag = f"month={month_T0.date().isoformat()}_q={q}"
        out_row_path = per_run_dir / f"{tag}.csv"

        # RESUME: read *all* rows for this (month,q)
        if RESUME and out_row_path.exists():
            df_prev = pd.read_csv(out_row_path)
            prev_rows = df_prev.to_dict(orient="records")
            rows.extend(prev_rows)
            if not DEBUG:
                print(f"[{model_type}] skip {tag} (exists; {len(prev_rows)} target rows)")
            continue

        v = pick_vintage_for_month_q(vintages, month_T0, q)
        if v is None:
            if DEBUG:
                print(f"[{model_type}] no vintage for month={month_T0.date()} q={q}")
            continue

        vintage_path = v.path
        if DEBUG:
            print("\n" + "=" * 80)
            print(f"DEBUG LOOP: month_T={month_T0.date()} q={q} chosen_vintage={vintage_path.name}")
            print("=" * 80)

        # 1) Build dataset from vintage
        D_v = build_zhang_data(
            vintage_file=vintage_path,
            spec_file=spec_file,
            gdp_series_id=GDP_SERIES_ID,
            sample_start=SAMPLE_START,
        )

        # 2) Trim to month_T (CRITICAL correctness step)
        D_v = trim_data_to_month_T(D_v, month_T0)

        # 3) Masking (optional; currently bypassed)
        Dq_full = D_v

        # 4) Rolling 10-year window ending at T
        Dq = slice_rolling_10y_window(
            Dq_full,
            month_T=month_T0,
            window_months=ROLLING_WINDOW_MONTHS,
            allow_short_start=ALLOW_SHORT_START,
        )

        # 5) Estimate + nowcast (multi-target in res)
        res = run_bay(
            Dq,
            R_max=R_MAX,
            n_iter=N_ITER,
            burn=BURN,
            thin=THIN,
            standardize=STANDARDIZE,
            seed=SEED,
            paper_defaults=PAPER_DEFAULTS,
            use_sv=use_sv,
            do_selection=do_selection,
        )

        # 6) Save outputs (returns LIST of rows)
        now_rows, _diag = save_subrun_outputs(
            run_dir=run_dir,
            model_type=model_type,
            tag=tag,
            D=Dq,
            res=res,
            month_T=month_T0,
            q=q,
            vintage_file=vintage_path.name,
            target_quarter=None,  # legacy; ignored
        )

        for r in now_rows:
            r["run_id"] = run_dir.name
            r.setdefault("month_in_quarter", int(month_in_quarter(month_T0)))

        pd.DataFrame(now_rows).to_csv(out_row_path, index=False)

        rows.extend(now_rows)

        # simple progress: show the within-quarter row if present, else last row
        log_row = now_rows[-1] if now_rows else None
        if log_row:
            print(
                f"[{model_type}] month={month_T0.date()} q={q} "
                f"vintage={vintage_path.name} "
                f"asofQ={log_row.get('asof_quarter')} targetQ={log_row.get('target_quarter')} "
                f"mean={float(log_row.get('nowcast_mean', float('nan'))):.4f} "
                f"p05={float(log_row.get('nowcast_p05', float('nan'))):.4f} "
                f"p95={float(log_row.get('nowcast_p95', float('nan'))):.4f} "
                f"(J={len(now_rows)})"
            )
        else:
            print(f"[{model_type}] month={month_T0.date()} q={q} vintage={vintage_path.name} (no rows?)")

    # Write per-model CSV exactly as before
    if rows:
        df = pd.DataFrame(rows).copy()
        if "month" in df.columns:
            df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
        df = df.sort_values(["month", "q", "target_quarter"]).reset_index(drop=True)

        out_csv = run_dir / "nowcasts" / f"nowcasts_{model_type}.csv"
        (run_dir / "nowcasts").mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")

    return rows


# =============================================================================
# Main
# =============================================================================

def main():
    root = project_root()

    vintage_dir, spec_file, pattern = prepare_xlsx_cache(root)

    start = pd.Timestamp(VINTAGE_START)
    end = pd.Timestamp(VINTAGE_END)

    vintages_all = list_vintages(Path(vintage_dir), pattern=pattern)
    vintages = [v for v in vintages_all if start <= v.date <= end]
    if not vintages:
        raise RuntimeError(f"No vintage files found in {vintage_dir} for [{start.date()}..{end.date()}]")

    months = pick_months(start, end, RUN_MODE)

    # Infer buckets once (still useful if you later re-enable make_Xq_for_month)
    _buckets = infer_buckets_month_by_month(
        vintage_dir=Path(vintage_dir),
        spec_file=Path(spec_file),
        gdp_series_id=GDP_SERIES_ID,
        sample_start=SAMPLE_START,
        pattern=pattern,
        vintage_start=VINTAGE_START,
        vintage_end=VINTAGE_END,
    )

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
        run_mode=RUN_MODE,
        q_fixed=Q_FIXED,
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

    # -------------------------------------------------------------------------
    # PARALLELIZE ONLY OVER MODEL_TYPES (2 models -> 2 cores)
    # -------------------------------------------------------------------------
    all_rows: list[dict] = []

    max_workers = min(2, len(MODEL_TYPES))  # ensures 2 cores max for your request
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(run_one_model_type, model_type, months, vintages, run_dir, spec_file): model_type
            for model_type in MODEL_TYPES
        }
        for fut in as_completed(futs):
            all_rows.extend(fut.result())

    # Combined nowcasts.csv (same as before)
    if all_rows:
        df_all = pd.DataFrame(all_rows).copy()
        if "month" in df_all.columns:
            df_all["month"] = pd.to_datetime(df_all["month"]).dt.to_period("M").dt.to_timestamp()
        df_all = df_all.sort_values(["model", "month", "q", "target_quarter"]).reset_index(drop=True)

        (run_dir / "nowcasts").mkdir(parents=True, exist_ok=True)
        df_all.to_csv(run_dir / "nowcasts" / "nowcasts.csv", index=False)

    # diagnostics summary (scan per-subrun diagnostics.json) (same as before)
    diag_files = sorted((run_dir / "posterior" / "per_run").glob("*/*/*/diagnostics.json"))
    diag_rows = []
    for f in diag_files:
        try:
            d = json.loads(f.read_text())
            model = f.parents[1].name      # .../<model>/<tag>/diagnostics.json
            tag = f.parent.name
            d["model"] = model
            d["subrun_tag"] = tag
            diag_rows.append(d)
        except Exception:
            continue
    if diag_rows:
        (run_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diag_rows).to_csv(run_dir / "diagnostics" / "diagnostics_summary.csv", index=False)

    finalize_run_dir(run_dir, start_time=t0)
    print(f"Done. Results saved under: {run_dir}")


if __name__ == "__main__":
    main()