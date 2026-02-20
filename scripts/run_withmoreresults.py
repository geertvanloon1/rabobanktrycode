import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
from zhangnowcast.data.zhang_xq_builder import make_Xq_for_month  # noqa: F401 (kept for future)
from zhangnowcast.inference.sampler import run_bay
from zhangnowcast.results.io import init_run_dir, finalize_run_dir, save_subrun_outputs


# paste this into terminal:
# nohup /usr/local/bin/python3 "/path/to/scripts/run_withmoreresults.py" > run_full.log 2>&1 &


# CONFIG (no argparse; edit values here)

MODEL_TYPES = ["bay", "bay_sv"]  # ["bay"] or ["bay_sv"] or both

DATA_DIR_REL = Path("data/NL")
SPEC_FILE_REL = Path("data/Spec_NL.xlsx")

CACHE_DIR_REL = Path("output/replication")
RESULTS_DIR_REL = Path("outputs")

VINTAGE_START = "2016-01-01"
VINTAGE_END = "2017-01-01"

GDP_SERIES_ID = "Real GDP"
SAMPLE_START = "2002-01-01"

ROLLING_WINDOW_MONTHS = 120
ALLOW_SHORT_START = True

# Faster computation (pilot defaults)
R_MAX = 3
N_ITER = 80
BURN = 40
THIN = 1

STANDARDIZE = True
SEED = 123
PAPER_DEFAULTS = True

SAVE_DRAWS = False
RESUME = True
DEBUG = False

do_selection = True  # currently not used; kept for easy toggling


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


def _pick_log_row(now_rows: list[dict]) -> dict:
    """
    For printing progress:
    Prefer the row whose target_quarter == asof_quarter (within-quarter nowcast),
    else fall back to the last target row.
    """
    if not now_rows:
        return {}
    for r in now_rows:
        if bool(r.get("asof_in_target", False)):
            return r
    return now_rows[-1]


# Main

def main():
    root = project_root()

    vintage_dir, spec_file, pattern = prepare_xlsx_cache(root)

    start = pd.Timestamp(VINTAGE_START)
    end = pd.Timestamp(VINTAGE_END)

    vintages_all = list_vintages(Path(vintage_dir), pattern=pattern)
    vintages = [v for v in vintages_all if start <= v.date <= end]
    if not vintages:
        raise RuntimeError(f"No vintage files found in {vintage_dir} for [{start.date()}..{end.date()}]")

    start_month = start.normalize().replace(day=1)
    end_month = end.normalize().replace(day=1)
    months = list(pd.date_range(start_month, end_month, freq="MS"))

    # Infer buckets once (still useful if you later re-enable make_Xq_for_month)
    buckets = infer_buckets_month_by_month(
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

    all_rows: list[dict] = []

    for model_type in MODEL_TYPES:
        use_sv = model_use_sv(model_type)

        per_run_dir = run_dir / "nowcasts" / "per_run" / model_type
        per_run_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []

        for month_T in months:
            month_T0 = pd.to_datetime(month_T).normalize().replace(day=1)

            for q in (1, 2, 3):
                tag = f"month={month_T0.date().isoformat()}_q={q}"
                out_row_path = per_run_dir / f"{tag}.csv"

                # RESUME: read *all* rows for this (month,q)
                if RESUME and out_row_path.exists():
                    df_prev = pd.read_csv(out_row_path)
                    prev_rows = df_prev.to_dict(orient="records")
                    rows.extend(prev_rows)
                    all_rows.extend(prev_rows)
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
                # Dq_full = make_Xq_for_month(D_v, buckets, month_T0, q=q)
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
                    do_selection=False,
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
                all_rows.extend(now_rows)

                log_row = _pick_log_row(now_rows)
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

        if rows:
            df = pd.DataFrame(rows).copy()
            if "month" in df.columns:
                df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
            df = df.sort_values(["month", "q", "target_quarter"]).reset_index(drop=True)

            out_csv = run_dir / "nowcasts" / f"nowcasts_{model_type}.csv"
            df.to_csv(out_csv, index=False)
            print(f"Saved {out_csv}")

    if all_rows:
        df_all = pd.DataFrame(all_rows).copy()
        if "month" in df_all.columns:
            df_all["month"] = pd.to_datetime(df_all["month"]).dt.to_period("M").dt.to_timestamp()
        df_all = df_all.sort_values(["model", "month", "q", "target_quarter"]).reset_index(drop=True)

        (run_dir / "nowcasts").mkdir(parents=True, exist_ok=True)
        df_all.to_csv(run_dir / "nowcasts" / "nowcasts.csv", index=False)

    # diagnostics summary (scan per-subrun diagnostics.json)
    diag_files = sorted((run_dir / "posterior" / "per_run").glob("*/*/*/diagnostics.json"))  # <-- FIXED
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
