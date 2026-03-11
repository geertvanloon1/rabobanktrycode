from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zhangnowcast.data.data import build_zhang_data


# python3 scripts/make_densityplots.py --run_dir 

# =============================================================================
# Utilities
# =============================================================================

def _have_jinja2() -> bool:
    try:
        import jinja2  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(fig, outbase: Path) -> None:
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=200)
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _coerce_month(x) -> pd.Timestamp:
    return pd.to_datetime(x).to_period("M").to_timestamp()


def _nice(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_=." else "_" for c in str(s))


def _to_latex_table(df: pd.DataFrame, path: Path, caption: str = "", label: str = "") -> None:
    if not _have_jinja2():
        print(f"[make_artifacts] jinja2 not installed → skipping LaTeX table {path.name}")
        return

    latex = df.to_latex(index=False, escape=True)

    if caption or label:
        latex = (
            "\\begin{table}[!htbp]\n\\centering\n"
            + latex
            + (f"\\caption{{{caption}}}\n" if caption else "")
            + (f"\\label{{{label}}}\n" if label else "")
            + "\\end{table}\n"
        )

    path.write_text(latex)


def _period_str_to_periodQ(s: str) -> Optional[pd.Period]:
    try:
        return pd.Period(str(s), freq="Q")
    except Exception:
        return None


# =============================================================================
# Units conversion
# =============================================================================

def ann_to_qoq_pct(x_ann: pd.Series) -> pd.Series:
    x = pd.to_numeric(x_ann, errors="coerce") / 100.0
    return 100.0 * (np.power(1.0 + x, 1.0 / 4.0) - 1.0)


def qoq_to_ann_pct(x_qoq: pd.Series) -> pd.Series:
    x = pd.to_numeric(x_qoq, errors="coerce") / 100.0
    return 100.0 * (np.power(1.0 + x, 4.0) - 1.0)


def ann_draws_to_qoq(draws_ann: np.ndarray) -> np.ndarray:
    """
    Elementwise transform for draws in percent annualized -> percent QoQ.
    """
    x = np.asarray(draws_ann, dtype=float) / 100.0
    out = 100.0 * (np.power(1.0 + x, 1.0 / 4.0) - 1.0)
    return out


def qoq_draws_to_ann(draws_qoq: np.ndarray) -> np.ndarray:
    x = np.asarray(draws_qoq, dtype=float) / 100.0
    out = 100.0 * (np.power(1.0 + x, 4.0) - 1.0)
    return out


def _apply_display_units(now: pd.DataFrame, display_units: str, base_units: str = "annualized") -> pd.DataFrame:
    """
    Create *_disp columns without touching originals.
    base_units: "annualized" (current Zhang-style output) or "qoq"
    display_units: "annualized" or "qoq"
    """
    df = now.copy()

    cols = [
        "nowcast_mean",
        "nowcast_p05",
        "nowcast_p50",
        "nowcast_p95",
        "nowcast_sd",
        "actual",
        "rw_nowcast",
    ]

    def convert(s: pd.Series) -> pd.Series:
        if base_units == display_units:
            return pd.to_numeric(s, errors="coerce")
        if base_units == "annualized" and display_units == "qoq":
            return ann_to_qoq_pct(s)
        if base_units == "qoq" and display_units == "annualized":
            return qoq_to_ann_pct(s)
        raise ValueError(f"Unsupported unit conversion {base_units} -> {display_units}")

    for c in cols:
        if c in df.columns:
            df[f"{c}_disp"] = convert(df[c])
        else:
            df[f"{c}_disp"] = np.nan

    return df


# =============================================================================
# Actual GDP + RW benchmark helpers
# =============================================================================

def build_actual_gdp_lookup(
    project_root: Path,
    actual_vintage_rel: str,
    spec_file_rel: str,
    gdp_series_id: str,
    sample_start: str,
) -> pd.Series:
    """
    Map quarter label (e.g., '2020Q1') -> realized GDP value from a complete vintage.
    Uses build_zhang_data() so transforms match the runner.
    """
    actual_vintage = (project_root / Path(actual_vintage_rel)).resolve()
    spec_file = (project_root / Path(spec_file_rel)).resolve()

    if not actual_vintage.exists():
        raise FileNotFoundError(f"Actual vintage not found: {actual_vintage}")
    if not spec_file.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_file}")

    D = build_zhang_data(
        vintage_file=actual_vintage,
        spec_file=spec_file,
        gdp_series_id=gdp_series_id,
        sample_start=sample_start,
    )

    if not hasattr(D, "dates_q"):
        raise AttributeError("ZhangData missing 'dates_q'; cannot build actual GDP lookup.")

    dates_q = pd.to_datetime(np.asarray(D.dates_q))
    q_labels = dates_q.to_period("Q").astype(str)

    y = None
    for name in ("y_q", "Y_q", "gdp_q", "GDP_q"):
        if hasattr(D, name):
            y = np.asarray(getattr(D, name), dtype=float)
            break
    if y is None:
        raise AttributeError("Could not find quarterly GDP array on ZhangData (tried y_q, Y_q, gdp_q, GDP_q).")

    s = pd.Series(y, index=q_labels)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def attach_actual_and_rw(now: pd.DataFrame, actual_by_q: pd.Series) -> pd.DataFrame:
    """
    Adds:
      - actual: realized GDP for target_quarter
      - rw_nowcast: random-walk forecast = actual of (target_quarter - 1)
    """
    df = now.copy()
    if "target_quarter" not in df.columns:
        df["actual"] = np.nan
        df["rw_nowcast"] = np.nan
        return df

    df["actual"] = df["target_quarter"].astype(str).map(actual_by_q)

    tq = df["target_quarter"].astype(str).apply(_period_str_to_periodQ)
    prev = tq.apply(lambda p: (p - 1).strftime("%YQ%q") if p is not None else None)
    df["rw_nowcast"] = prev.map(actual_by_q)

    return df


# =============================================================================
# Posterior pack loading (density-aware)
# =============================================================================

@dataclass
class PosteriorPack:
    model: str
    tag: str
    month: pd.Timestamp
    q: int
    month_in_quarter: int
    vintage_file: str
    path: Path

    # Posterior predictive for targets:
    nowcast_targets: np.ndarray            # (J,)
    nowcast_draws_matrix: np.ndarray       # (S,J)  NEW (preferred)

    # (optional) older single-target key:
    nowcast_draws_legacy: np.ndarray       # (S,) or empty

    # Some posterior params (optional; kept)
    F_mean: np.ndarray
    F_sd: np.ndarray
    Lambda_mean: np.ndarray
    Lambda_sd: np.ndarray
    mu_mean: np.ndarray
    z_mean: np.ndarray
    z_draws: np.ndarray
    r_draws: np.ndarray
    beta_mean: np.ndarray
    a_mean: np.ndarray
    sigma2_mean: np.ndarray
    eta2_mean: float
    Psi_mean: np.ndarray
    omega_mean: np.ndarray
    omega_sd: np.ndarray
    tau2_mean: float

    dates_m: np.ndarray
    dates_q: np.ndarray
    series: np.ndarray

    standardize_mean: np.ndarray
    standardize_sd: np.ndarray


def load_posterior_pack(model: str, subdir: Path) -> Optional[PosteriorPack]:
    npz_path = subdir / "posterior.npz"
    meta_path = subdir / "posterior_meta.json"
    if not npz_path.exists() or not meta_path.exists():
        return None

    meta = _safe_read_json(meta_path)
    tag = meta.get("subrun_tag", subdir.name)
    month = pd.to_datetime(meta.get("month_T", "1970-01-01"))
    q = int(meta.get("q", -1))
    miq = int(meta.get("month_in_quarter", -1))
    vint = str(meta.get("vintage_file", ""))

    z = np.load(npz_path, allow_pickle=True)

    def get(name, default=None):
        return z[name] if name in z.files else default

    m0 = _coerce_month(month)
    miq_use = miq if miq > 0 else (((int(m0.month) - 1) % 3) + 1)

    # --- density payload (preferred keys from your saver) ---
    nowcast_draws_matrix = get("nowcast_draws_matrix")
    nowcast_targets = get("nowcast_targets")

    # --- fallback: legacy single-target draw vector stored as "nowcast_draws" ---
    nowcast_draws_legacy = get("nowcast_draws")
    if nowcast_draws_matrix is None or np.asarray(nowcast_draws_matrix).size == 0:
        if nowcast_draws_legacy is not None and np.asarray(nowcast_draws_legacy).ndim == 1:
            nowcast_draws_matrix = np.asarray(nowcast_draws_legacy, dtype=float)[:, None]
        else:
            nowcast_draws_matrix = np.asarray([], dtype=float)

    if nowcast_targets is None or np.asarray(nowcast_targets).size == 0:
        # try meta
        tqs = meta.get("nowcast_targets", [])
        nowcast_targets = np.asarray([str(x) for x in tqs], dtype=object)

    return PosteriorPack(
        model=model,
        tag=tag,
        month=m0,
        q=q,
        month_in_quarter=miq_use,
        vintage_file=vint,
        path=subdir,

        nowcast_targets=np.asarray(nowcast_targets, dtype=object) if nowcast_targets is not None else np.asarray([], dtype=object),
        nowcast_draws_matrix=np.asarray(nowcast_draws_matrix, dtype=float) if nowcast_draws_matrix is not None else np.asarray([], dtype=float),
        nowcast_draws_legacy=np.asarray(nowcast_draws_legacy, dtype=float) if nowcast_draws_legacy is not None else np.asarray([], dtype=float),

        F_mean=get("F_mean") if get("F_mean") is not None else np.asarray([]),
        F_sd=get("F_sd") if get("F_sd") is not None else np.asarray([]),
        Lambda_mean=get("Lambda_mean") if get("Lambda_mean") is not None else np.asarray([]),
        Lambda_sd=get("Lambda_sd") if get("Lambda_sd") is not None else np.asarray([]),
        mu_mean=get("mu_mean") if get("mu_mean") is not None else np.asarray([]),
        z_mean=get("z_mean") if get("z_mean") is not None else np.asarray([]),
        z_draws=get("z_draws") if get("z_draws") is not None else np.asarray([]),
        r_draws=get("r_draws") if get("r_draws") is not None else np.asarray([]),
        beta_mean=get("beta_mean") if get("beta_mean") is not None else np.asarray([]),
        a_mean=get("a_mean") if get("a_mean") is not None else np.asarray([]),
        sigma2_mean=get("sigma2_mean") if get("sigma2_mean") is not None else np.asarray([]),
        eta2_mean=float(get("eta2_mean")) if get("eta2_mean") is not None else float("nan"),
        Psi_mean=get("Psi_mean") if get("Psi_mean") is not None else np.asarray([]),
        omega_mean=get("omega_mean") if get("omega_mean") is not None else np.asarray([]),
        omega_sd=get("omega_sd") if get("omega_sd") is not None else np.asarray([]),
        tau2_mean=float(get("tau2_mean")) if get("tau2_mean") is not None else float("nan"),

        dates_m=get("dates_m") if get("dates_m") is not None else np.asarray([]),
        dates_q=get("dates_q") if get("dates_q") is not None else np.asarray([]),
        series=get("series") if get("series") is not None else np.asarray([]),

        standardize_mean=get("standardize_mean") if get("standardize_mean") is not None else np.asarray([]),
        standardize_sd=get("standardize_sd") if get("standardize_sd") is not None else np.asarray([]),
    )


def scan_run(run_dir: Path) -> Tuple[pd.DataFrame, List[PosteriorPack], dict]:
    meta = _safe_read_json(run_dir / "run_metadata.json")

    nowcasts_path = run_dir / "nowcasts" / "nowcasts_with_actual_rw.csv"
    if not nowcasts_path.exists():
        nowcasts_path = run_dir / "nowcasts" / "nowcasts.csv"

    if nowcasts_path.exists():
        now = pd.read_csv(nowcasts_path)
    else:
        parts = list((run_dir / "nowcasts").glob("nowcasts_*.csv"))
        if not parts:
            raise FileNotFoundError(f"Could not find nowcasts.csv under {run_dir / 'nowcasts'}")
        now = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

    # Normalize types
    if "month" in now.columns:
        now["month"] = pd.to_datetime(now["month"]).dt.to_period("M").dt.to_timestamp()
    if "q" in now.columns:
        now["q"] = now["q"].astype(int)

    # month_in_quarter
    if "month_in_quarter" not in now.columns and "month" in now.columns:
        now["month_in_quarter"] = ((now["month"].dt.month - 1) % 3) + 1

    # asof_quarter default from month if missing
    if "asof_quarter" not in now.columns and "month" in now.columns:
        now["asof_quarter"] = now["month"].dt.to_period("Q").astype(str)

    # asof_in_target
    if "asof_in_target" not in now.columns:
        if "asof_quarter" in now.columns and "target_quarter" in now.columns:
            now["asof_in_target"] = now["target_quarter"].astype(str) == now["asof_quarter"].astype(str)
        else:
            now["asof_in_target"] = False

    posterior_root = run_dir / "posterior" / "per_run"
    packs: List[PosteriorPack] = []
    if posterior_root.exists():
        for model_dir in sorted(posterior_root.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for tag_dir in sorted(model_dir.iterdir()):
                if not tag_dir.is_dir():
                    continue
                pack = load_posterior_pack(model, tag_dir)
                if pack is not None:
                    packs.append(pack)

    return now, packs, meta


# =============================================================================
# Derived columns for within-quarter logic
# =============================================================================

def _quarter_start_month(tq: str) -> Optional[pd.Timestamp]:
    p = _period_str_to_periodQ(tq)
    if p is None:
        return None
    return p.start_time.to_period("M").to_timestamp()


def _months_diff(m1: pd.Timestamp, m0: pd.Timestamp) -> int:
    return (m1.year - m0.year) * 12 + (m1.month - m0.month)


def _rel_step_label(step: int) -> str:
    """
    step 0..8  => m1-q1 .. m3-q3
    step 9..   => +1m-q1, +1m-q2, ... (months after quarter end)
    """
    if step < 0:
        return f"pre({step})"
    month_index = step // 3
    q = (step % 3) + 1
    if month_index <= 2:
        return f"m{month_index + 1}-q{q}"
    extra = month_index - 2
    return f"+{extra}m-q{q}"


def add_release_step_columns(now: pd.DataFrame) -> pd.DataFrame:
    df = now.copy()
    if "target_quarter" not in df.columns or "month" not in df.columns or "q" not in df.columns:
        df["rel_step_in_target"] = np.nan
        df["rel_step_label"] = ""
        return df

    df["target_q_start_month"] = df["target_quarter"].astype(str).map(_quarter_start_month)
    df["target_q_start_month"] = pd.to_datetime(df["target_q_start_month"], errors="coerce")

    steps = []
    for m, q, q0 in zip(df["month"], df["q"], df["target_q_start_month"]):
        if pd.isna(q0) or pd.isna(m):
            steps.append(np.nan)
            continue
        md = _months_diff(pd.Timestamp(m), pd.Timestamp(q0))
        steps.append(int(md * 3 + (int(q) - 1)))
    df["rel_step_in_target"] = steps
    df["rel_step_label"] = df["rel_step_in_target"].apply(lambda x: _rel_step_label(int(x)) if np.isfinite(x) else "")
    return df


def add_release_date(now: pd.DataFrame) -> pd.DataFrame:
    """
    Adds release_date (calendar) from vintage_file if possible, else month + q offset.
    """
    df = now.copy()

    if "vintage_file" in df.columns:
        stem = df["vintage_file"].astype(str).str.replace(".xlsx", "", regex=False)
        rd = pd.to_datetime(stem, errors="coerce")
    else:
        rd = pd.Series(pd.NaT, index=df.index)

    if "month" in df.columns and "q" in df.columns:
        fallback = pd.to_datetime(df["month"]) + pd.to_timedelta((df["q"].astype(int) - 1) * 10, unit="D")
        rd = rd.fillna(fallback)

    df["release_date"] = rd
    return df


# =============================================================================
# Mean-error tables (yours, kept)
# =============================================================================

def table_run_summary(meta: dict) -> pd.DataFrame:
    batch = meta.get("batch_config", {})
    data = meta.get("data", {})
    runtime = meta.get("runtime", {})
    rows = [
        ("run_id", meta.get("run_id", "")),
        ("created_at_utc", meta.get("created_at_utc", "")),
        ("seconds", runtime.get("seconds", "")),
        ("model_types", ",".join(batch.get("model_types", [])) if isinstance(batch.get("model_types", []), list) else str(batch.get("model_types", ""))),
        ("R_max", batch.get("R_max", "")),
        ("n_iter", batch.get("n_iter", "")),
        ("burn", batch.get("burn", "")),
        ("thin", batch.get("thin", "")),
        ("standardize", batch.get("standardize", "")),
        ("seed", batch.get("seed", "")),
        ("paper_defaults", batch.get("paper_defaults", "")),
        ("rolling_window_months", batch.get("rolling_window_months", "")),
        ("allow_short_start", batch.get("allow_short_start", "")),
        ("vintage_start", batch.get("vintage_start", "")),
        ("vintage_end", batch.get("vintage_end", "")),
        ("data_dir", data.get("data_dir", "")),
        ("spec_file", data.get("spec_file", "")),
        ("gdp_series_id", data.get("gdp_series_id", "")),
        ("sample_start", data.get("sample_start", "")),
        ("pattern", data.get("pattern", "")),
        ("save_raw_draws", batch.get("save_raw_draws", "")),
    ]
    return pd.DataFrame(rows, columns=["key", "value"])


def table_nowcast_errors_within_quarter(now: pd.DataFrame) -> pd.DataFrame:
    """
    Within-quarter mean nowcast errors:
      - Uses rows where asof_in_target==True
      - Groups by model and rel_step_label (m1-q1 ... m3-q3)
    """
    df = now.copy()
    if "actual_disp" not in df.columns or df["actual_disp"].isna().all():
        return pd.DataFrame([{"note": "No realized GDP values attached (actual_disp all NaN). Cannot compute errors."}])

    d = df[(df["asof_in_target"] == True) & (df["actual_disp"].notna())].copy()
    if d.empty:
        return pd.DataFrame([{"note": "No within-quarter rows with non-NaN actual_disp. Cannot compute errors."}])

    if "rel_step_in_target" in d.columns:
        d = d[d["rel_step_in_target"].between(0, 8)]
    if d.empty:
        return pd.DataFrame([{"note": "No rows remaining after filtering to within-quarter steps 0..8."}])

    d["err"] = d["nowcast_mean_disp"] - d["actual_disp"]
    d["abs_err"] = d["err"].abs()
    d["sq_err"] = d["err"] ** 2

    if "rw_nowcast_disp" not in d.columns:
        d["rw_nowcast_disp"] = np.nan

    d["rw_err"] = d["rw_nowcast_disp"] - d["actual_disp"]
    d["rw_abs_err"] = d["rw_err"].abs()
    d["rw_sq_err"] = d["rw_err"] ** 2

    out = []
    group_cols = ["model", "rel_step_label"]
    for keys, g in d.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        mae = float(g["abs_err"].mean())
        rmsfe = float(np.sqrt(g["sq_err"].mean()))
        rw_mae = float(g["rw_abs_err"].mean()) if g["rw_abs_err"].notna().any() else float("nan")
        rw_rmsfe = float(np.sqrt(g["rw_sq_err"].mean())) if g["rw_sq_err"].notna().any() else float("nan")
        row.update(dict(
            n=int(len(g)),
            MAE=mae,
            RMSFE=rmsfe,
            RW_MAE=rw_mae,
            RW_RMSFE=rw_rmsfe,
            MANE=(mae / rw_mae) if (np.isfinite(rw_mae) and rw_mae > 0) else float("nan"),
        ))
        out.append(row)

    order = [f"m{m}-q{q}" for m in (1, 2, 3) for q in (1, 2, 3)]
    out_df = pd.DataFrame(out)
    out_df["__ord"] = out_df["rel_step_label"].apply(lambda s: order.index(s) if s in order else 999)
    out_df = out_df.sort_values(["model", "__ord"]).drop(columns="__ord").reset_index(drop=True)
    return out_df


def table_nowcast_errors_final_before_release(now: pd.DataFrame) -> pd.DataFrame:
    """
    Final-before-release mean nowcast errors:
      - For each (model, target_quarter), take last nowcast row in time (month,q).
      - Aggregate by model.
    """
    df = now.copy()
    if "actual_disp" not in df.columns or df["actual_disp"].isna().all():
        return pd.DataFrame([{"note": "No realized GDP values attached (actual_disp all NaN). Cannot compute errors."}])

    if "target_quarter" not in df.columns:
        return pd.DataFrame([{"note": "No target_quarter column present. Cannot compute final-before-release errors."}])

    d = df[df["actual_disp"].notna()].copy()
    if d.empty:
        return pd.DataFrame([{"note": "No rows with non-NaN actual_disp. Cannot compute errors."}])

    d = d.sort_values(["model", "target_quarter", "month", "q"])
    last_rows = d.groupby(["model", "target_quarter"], as_index=False).tail(1)

    last_rows["err"] = last_rows["nowcast_mean_disp"] - last_rows["actual_disp"]
    last_rows["abs_err"] = last_rows["err"].abs()
    last_rows["sq_err"] = last_rows["err"] ** 2

    if "rw_nowcast_disp" not in last_rows.columns:
        last_rows["rw_nowcast_disp"] = np.nan

    last_rows["rw_err"] = last_rows["rw_nowcast_disp"] - last_rows["actual_disp"]
    last_rows["rw_abs_err"] = last_rows["rw_err"].abs()
    last_rows["rw_sq_err"] = last_rows["rw_err"] ** 2

    out = []
    for model, g in last_rows.groupby(["model"]):
        mae = float(g["abs_err"].mean())
        rmsfe = float(np.sqrt(g["sq_err"].mean()))
        rw_mae = float(g["rw_abs_err"].mean()) if g["rw_abs_err"].notna().any() else float("nan")
        rw_rmsfe = float(np.sqrt(g["rw_sq_err"].mean())) if g["rw_sq_err"].notna().any() else float("nan")
        out.append(dict(
            model=model,
            n=int(len(g)),
            MAE=mae,
            RMSFE=rmsfe,
            RW_MAE=rw_mae,
            RW_RMSFE=rw_rmsfe,
            MANE=(mae / rw_mae) if (np.isfinite(rw_mae) and rw_mae > 0) else float("nan"),
        ))

    return pd.DataFrame(out).sort_values(["model"]).reset_index(drop=True)


# =============================================================================
# Density scoring
# =============================================================================

def crps_from_draws(draws: np.ndarray, y: float) -> float:
    """
    CRPS from Monte Carlo draws:
      CRPS = E|X-y| - 0.5 E|X-X'|
    Using O(S log S) exact formula for E|X-X'| via sorting.
    """
    x = np.asarray(draws, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20 or not np.isfinite(y):
        return float("nan")
    term1 = float(np.mean(np.abs(x - y)))
    xs = np.sort(x)
    S = xs.size
    i = np.arange(1, S + 1)
    e_abs_xx = float((2.0 / (S * S)) * np.sum((2 * i - S - 1) * xs))
    return term1 - 0.5 * e_abs_xx


def pit_from_draws(draws: np.ndarray, y: float, rng: np.random.Generator) -> float:
    x = np.asarray(draws, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20 or not np.isfinite(y):
        return float("nan")
    lt = np.mean(x < y)
    eq = np.mean(x == y)
    return float(lt + rng.random() * eq)


def hist_logscore(draws: np.ndarray, y: float, nbins: int = 60, eps: float = 1e-12) -> float:
    """
    Approximate log predictive density at y using a histogram density estimate.
    (Stable and dependency-free.)
    """
    x = np.asarray(draws, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 50 or not np.isfinite(y):
        return float("nan")

    lo, hi = np.quantile(x, [0.001, 0.999])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")

    pad = 0.02 * (hi - lo)
    lo -= pad
    hi += pad

    counts, edges = np.histogram(x, bins=nbins, range=(lo, hi), density=False)
    binw = float(edges[1] - edges[0])
    dens = (counts / max(1, x.size)) / max(binw, eps)

    if y < lo or y > hi:
        return float(np.log(eps))

    b = int(np.clip(np.searchsorted(edges, y, side="right") - 1, 0, nbins - 1))
    return float(np.log(max(dens[b], eps)))


def _pack_index(packs: List[PosteriorPack]) -> Dict[Tuple[str, str], PosteriorPack]:
    return {(p.model, p.tag): p for p in packs}


def _get_draws_for_row(
    r: pd.Series,
    pack: PosteriorPack,
    display_units: str,
    base_units: str = "annualized",
) -> Optional[np.ndarray]:
    """
    Return predictive draws for THIS row's (target_quarter) from pack, transformed to display_units.
    """
    if pack is None or pack.nowcast_draws_matrix is None or pack.nowcast_draws_matrix.size == 0:
        return None

    tqs = list(map(str, np.asarray(pack.nowcast_targets).tolist()))
    tq = str(r["target_quarter"])
    if tq not in tqs:
        return None
    j = tqs.index(tq)

    draws = np.asarray(pack.nowcast_draws_matrix)[:, j].astype(float)
    draws = draws[np.isfinite(draws)]
    if draws.size < 20:
        return None

    if base_units == display_units:
        return draws
    if base_units == "annualized" and display_units == "qoq":
        return ann_draws_to_qoq(draws)
    if base_units == "qoq" and display_units == "annualized":
        return qoq_draws_to_ann(draws)
    return draws


def density_scores_current_quarter(now: pd.DataFrame, packs: List[PosteriorPack], display_units: str) -> pd.DataFrame:
    """
    Density scores for the true nowcasting task: asof_in_target==True.
    """
    df = now[(now["asof_in_target"] == True) & (now["actual_disp"].notna())].copy()
    if df.empty:
        return pd.DataFrame([{"note": "No rows with asof_in_target==True and non-NaN actual_disp."}])

    df = df.sort_values(["model", "release_date", "q"]).copy()
    pidx = _pack_index(packs)
    rng = np.random.default_rng(123)

    out = []
    for _, r in df.iterrows():
        model = str(r["model"])
        tag = str(r.get("subrun_tag", ""))
        pack = pidx.get((model, tag))
        draws = _get_draws_for_row(r, pack, display_units=display_units)
        if draws is None:
            continue

        y = float(r["actual_disp"])
        lo90, hi90 = np.quantile(draws, [0.05, 0.95])

        out.append(dict(
            model=model,
            subrun_tag=tag,
            release_date=pd.to_datetime(r.get("release_date", pd.NaT)),
            month=pd.to_datetime(r.get("month", pd.NaT)),
            q=int(r.get("q", -1)),
            target_quarter=str(r["target_quarter"]),
            rel_step_label=str(r.get("rel_step_label", "")),
            crps=crps_from_draws(draws, y),
            logscore=hist_logscore(draws, y),
            pit=pit_from_draws(draws, y, rng=rng),
            cover90=float(lo90 <= y <= hi90),
            width90=float(hi90 - lo90),
            draws_n=int(draws.size),
        ))

    return pd.DataFrame(out)


def density_scores_final_before_release(now: pd.DataFrame, packs: List[PosteriorPack], display_units: str) -> pd.DataFrame:
    """
    Density scores for "final before release": last snapshot per (model, target_quarter).
    """
    df = now[now["actual_disp"].notna()].copy()
    if df.empty:
        return pd.DataFrame([{"note": "No rows with non-NaN actual_disp."}])
    if "target_quarter" not in df.columns:
        return pd.DataFrame([{"note": "Missing target_quarter; cannot compute final-before-release density scores."}])

    df = df.sort_values(["model", "target_quarter", "month", "q"]).copy()
    last_rows = df.groupby(["model", "target_quarter"], as_index=False).tail(1).copy()

    pidx = _pack_index(packs)
    rng = np.random.default_rng(456)

    out = []
    for _, r in last_rows.iterrows():
        model = str(r["model"])
        tag = str(r.get("subrun_tag", ""))
        pack = pidx.get((model, tag))
        draws = _get_draws_for_row(r, pack, display_units=display_units)
        if draws is None:
            continue

        y = float(r["actual_disp"])
        lo90, hi90 = np.quantile(draws, [0.05, 0.95])

        out.append(dict(
            model=model,
            subrun_tag=tag,
            release_date=pd.to_datetime(r.get("release_date", pd.NaT)),
            month=pd.to_datetime(r.get("month", pd.NaT)),
            q=int(r.get("q", -1)),
            target_quarter=str(r["target_quarter"]),
            crps=crps_from_draws(draws, y),
            logscore=hist_logscore(draws, y),
            pit=pit_from_draws(draws, y, rng=rng),
            cover90=float(lo90 <= y <= hi90),
            width90=float(hi90 - lo90),
            draws_n=int(draws.size),
        ))

    return pd.DataFrame(out)


def summarize_density_scores(scores: pd.DataFrame, by_step: bool = False) -> pd.DataFrame:
    if "note" in scores.columns:
        return scores.copy()

    group_cols = ["model"]
    if by_step and "rel_step_label" in scores.columns:
        group_cols = ["model", "rel_step_label"]

    g = scores.groupby(group_cols, as_index=False).agg(
        n=("crps", "count"),
        mean_crps=("crps", "mean"),
        mean_logscore=("logscore", "mean"),
        mean_cover90=("cover90", "mean"),
        mean_width90=("width90", "mean"),
        mean_pit=("pit", "mean"),
    )
    return g


# =============================================================================
# DM test (Diebold–Mariano) for score differences
# =============================================================================

def _newey_west_var(d: np.ndarray, L: int) -> float:
    """
    Newey–West variance estimate of mean(d) with lag L.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    T = d.size
    if T < 3:
        return float("nan")

    d0 = d - d.mean()
    gamma0 = np.mean(d0 * d0)

    var = gamma0
    for l in range(1, min(L, T - 1) + 1):
        w = 1.0 - l / (L + 1.0)
        gamma = np.mean(d0[l:] * d0[:-l])
        var += 2.0 * w * gamma

    # variance of sample mean
    return float(var / T)


def dm_test(scores_a: pd.DataFrame, scores_b: pd.DataFrame, key_cols: List[str], score_col: str, hac_lag: int = 1) -> pd.DataFrame:
    """
    DM test on d_t = score_a - score_b aligned on key_cols.
    Lower is better for CRPS; higher is better for logscore.
    You interpret sign accordingly.
    """
    a = scores_a[key_cols + [score_col]].rename(columns={score_col: "a"})
    b = scores_b[key_cols + [score_col]].rename(columns={score_col: "b"})
    m = a.merge(b, on=key_cols, how="inner")
    if m.empty:
        return pd.DataFrame([{"note": f"No overlap for DM test on {score_col}"}])

    d = (m["a"] - m["b"]).to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 8:
        return pd.DataFrame([{"note": f"Too few paired observations for DM test on {score_col}", "n": int(d.size)}])

    mean_d = float(np.mean(d))
    var_mean = _newey_west_var(d, L=hac_lag)
    if not np.isfinite(var_mean) or var_mean <= 0:
        return pd.DataFrame([{"note": f"NW variance failed for DM test on {score_col}", "n": int(d.size)}])

    dm_stat = mean_d / np.sqrt(var_mean)
    # normal approx p-value
    from math import erf, sqrt
    pval = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm_stat) / sqrt(2.0))))

    return pd.DataFrame([{
        "score": score_col,
        "n": int(d.size),
        "hac_lag": int(hac_lag),
        "mean_diff_a_minus_b": mean_d,
        "dm_stat": float(dm_stat),
        "p_value": float(pval),
    }])


# =============================================================================
# Density plots
# =============================================================================

def fig_fan_chart_current_target(now: pd.DataFrame, packs: List[PosteriorPack], model: str, outdir: Path, ylab: str,
                                 display_units: str, bands=(0.5, 0.7, 0.9), base_units: str = "annualized") -> None:
    """
    Fan chart for current-quarter nowcasts (calendar time):
      shaded credible intervals + mean + actual.
    """
    df = now[(now["model"] == model) & (now["asof_in_target"] == True)].copy()
    if df.empty or "release_date" not in df.columns:
        return

    df = df.sort_values(["release_date", "q"]).copy()
    pidx = _pack_index(packs)

    rows = []
    for _, r in df.iterrows():
        tag = str(r.get("subrun_tag", ""))
        pack = pidx.get((model, tag))
        draws = _get_draws_for_row(r, pack, display_units=display_units, base_units=base_units)
        if draws is None or draws.size < 50:
            continue

        rec = dict(
            release_date=pd.to_datetime(r["release_date"]),
            mean=float(np.mean(draws)),
            actual=float(r["actual_disp"]) if pd.notna(r.get("actual_disp", np.nan)) else np.nan,
        )
        for b in bands:
            lo = (1 - b) / 2
            hi = 1 - lo
            rec[f"lo{int(b*100)}"] = float(np.quantile(draws, lo))
            rec[f"hi{int(b*100)}"] = float(np.quantile(draws, hi))
        rows.append(rec)

    if not rows:
        return

    qdf = pd.DataFrame(rows).sort_values("release_date")
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()

    for b in sorted(bands, reverse=True):
        ax.fill_between(qdf["release_date"], qdf[f"lo{int(b*100)}"], qdf[f"hi{int(b*100)}"], alpha=0.2,
                        label=f"{int(b*100)}% band")

    ax.plot(qdf["release_date"], qdf["mean"], marker="o", linewidth=1.5, label="mean")
    if qdf["actual"].notna().any():
        ax.plot(qdf["release_date"], qdf["actual"], marker="o", linewidth=1.5, label="actual")

    ax.set_title(f"{model} | Current-quarter density nowcast (fan chart)")
    ax.set_xlabel("Release date")
    ax.set_ylabel(ylab)
    ax.legend(fontsize=8, ncol=4)
    _savefig(fig, outdir / f"fan_chart_current_target__{model}")


def fig_density_heatmap_current_target(now: pd.DataFrame, packs: List[PosteriorPack], model: str, outdir: Path, ylab: str,
                                       display_units: str, grid_n: int = 160, base_units: str = "annualized") -> None:
    """
    Heatmap of predictive density over time for current-quarter nowcasts.
    """
    df = now[(now["model"] == model) & (now["asof_in_target"] == True)].copy()
    if df.empty or "release_date" not in df.columns:
        return

    df = df.sort_values(["release_date", "q"]).copy()
    pidx = _pack_index(packs)

    series = []
    dates = []
    for _, r in df.iterrows():
        tag = str(r.get("subrun_tag", ""))
        pack = pidx.get((model, tag))
        draws = _get_draws_for_row(r, pack, display_units=display_units, base_units=base_units)
        if draws is None or draws.size < 50:
            continue
        series.append(draws)
        dates.append(pd.to_datetime(r["release_date"]))

    if len(series) < 3:
        return

    allx = np.concatenate(series)
    lo, hi = np.quantile(allx, [0.01, 0.99])
    ygrid = np.linspace(lo, hi, grid_n)

    dens = np.zeros((grid_n - 1, len(series)))
    for i, d in enumerate(series):
        c, _ = np.histogram(d, bins=ygrid, density=True)
        dens[:, i] = c

    fig = plt.figure(figsize=(12, 5))
    ax = plt.gca()
    im = ax.imshow(
        dens, aspect="auto", origin="lower",
        extent=[0, len(dates) - 1, ygrid[0], ygrid[-1]]
    )
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([pd.Timestamp(x).date().isoformat() for x in dates], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylab)
    ax.set_title(f"{model} | Current-quarter predictive density (heatmap)")
    fig.colorbar(im, ax=ax, label="density")
    _savefig(fig, outdir / f"density_heatmap_current_target__{model}")


def fig_pit_hist(scores: pd.DataFrame, model: str, outdir: Path, title_suffix: str) -> None:
    d = scores[scores["model"] == model].copy()
    if d.empty or "pit" not in d.columns:
        return
    pit = pd.to_numeric(d["pit"], errors="coerce").dropna().to_numpy()
    if pit.size < 20:
        return

    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()
    ax.hist(pit, bins=10, density=True)
    ax.set_xlabel("PIT")
    ax.set_ylabel("density")
    ax.set_title(f"{model} | PIT histogram ({title_suffix})")
    _savefig(fig, outdir / f"pit_hist__{_nice(title_suffix)}__{model}")


# =============================================================================
# Existing mean figures (yours, kept as-is)
# =============================================================================

def fig_revision_paths_from_qstart_until_release(now: pd.DataFrame, model: str, outdir: Path, ylab: str, last_k_quarters: int = 6) -> None:
    df = now[now["model"] == model].copy()
    if df.empty or "target_quarter" not in df.columns:
        return

    df = df.sort_values(["target_quarter", "month", "q"]).copy()
    tq_list = df["target_quarter"].dropna().astype(str).unique().tolist()
    if not tq_list:
        return

    tq_periods = [(t, _period_str_to_periodQ(t)) for t in tq_list]
    tq_periods = [(t, p) for t, p in tq_periods if p is not None]
    tq_periods.sort(key=lambda x: x[1])
    tq_use = [t for (t, _) in tq_periods][-last_k_quarters:]

    fig = plt.figure(figsize=(11, 5))
    ax = plt.gca()
    max_step = 0

    for t in tq_use:
        g = df[df["target_quarter"].astype(str) == str(t)].copy()
        if g.empty:
            continue

        g = g[np.isfinite(g["rel_step_in_target"]) & (g["rel_step_in_target"] >= 0)].copy()
        if g.empty:
            continue

        g = g.sort_values(["rel_step_in_target", "month", "q"]).groupby("rel_step_in_target", as_index=False).last()
        g = g.sort_values("rel_step_in_target")

        xs = g["rel_step_in_target"].astype(int).to_numpy()
        ys = g["nowcast_mean_disp"].to_numpy(dtype=float)

        max_step = max(max_step, int(xs.max()))
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=str(t))

        if "actual_disp" in g.columns and g["actual_disp"].notna().any():
            a = float(g["actual_disp"].dropna().iloc[0])
            ax.hlines(a, xmin=int(xs.min()), xmax=int(xs.max()), linestyles="dashed", linewidth=1)

    ticks = list(range(0, max_step + 1))
    labels = [_rel_step_label(s) for s in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlim(-0.5, max_step + 2.2)
    ax.set_xlabel("Release step since target-quarter start")
    ax.set_ylabel(ylab)
    ax.set_title(f"{model} | Nowcast revision paths (from quarter start until release)")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, outdir / f"revision_paths_until_release__{model}")


def fig_within_quarter_nowcast_paths(now: pd.DataFrame, model: str, outdir: Path, ylab: str, last_k_quarters: int = 8) -> None:
    df = now[(now["model"] == model) & (now["asof_in_target"] == True)].copy()
    if df.empty or "target_quarter" not in df.columns:
        return

    df = df[np.isfinite(df["rel_step_in_target"]) & df["rel_step_in_target"].between(0, 8)].copy()
    if df.empty:
        return

    tq_list = df["target_quarter"].dropna().astype(str).unique().tolist()
    tq_periods = [(t, _period_str_to_periodQ(t)) for t in tq_list]
    tq_periods = [(t, p) for t, p in tq_periods if p is not None]
    tq_periods.sort(key=lambda x: x[1])
    tq_use = [t for (t, _) in tq_periods][-last_k_quarters:]

    fig = plt.figure(figsize=(11, 5))
    ax = plt.gca()

    for t in tq_use:
        g = df[df["target_quarter"].astype(str) == str(t)].copy()
        if g.empty:
            continue

        g = g.sort_values(["rel_step_in_target", "month", "q"]).groupby("rel_step_in_target", as_index=False).last()
        g = g.sort_values("rel_step_in_target")

        xs = g["rel_step_in_target"].astype(int).to_numpy()
        ys = g["nowcast_mean_disp"].to_numpy(dtype=float)

        ax.plot(xs, ys, marker="o", linewidth=1.5, label=str(t))

        if "actual_disp" in g.columns and g["actual_disp"].notna().any():
            a = float(g["actual_disp"].dropna().iloc[0])
            ax.hlines(a, xmin=0, xmax=8, linestyles="dashed", linewidth=1)

    ticks = list(range(0, 9))
    labels = [_rel_step_label(s) for s in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlim(-0.5, 10.0)
    ax.set_xlabel("Within-quarter release step")
    ax.set_ylabel(ylab)
    ax.set_title(f"{model} | Within-quarter nowcast paths (sanity)")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, outdir / f"within_quarter_paths__{model}")


def fig_actual_vs_final_nowcast_over_time(now: pd.DataFrame, model: str, outdir: Path, ylab: str, last_k_quarters: int = 12) -> None:
    df = now[now["model"] == model].copy()
    if df.empty or "target_quarter" not in df.columns:
        return
    if "actual_disp" not in df.columns or df["actual_disp"].isna().all():
        return

    df = df[df["actual_disp"].notna()].copy()
    if df.empty:
        return

    df = df.sort_values(["target_quarter", "month", "q"]).copy()
    last_rows = df.groupby("target_quarter", as_index=False).tail(1).copy()

    last_rows["tq_period"] = last_rows["target_quarter"].astype(str).apply(_period_str_to_periodQ)
    last_rows = last_rows[last_rows["tq_period"].notna()].copy()
    last_rows = last_rows.sort_values("tq_period")
    if len(last_rows) > last_k_quarters:
        last_rows = last_rows.iloc[-last_k_quarters:].copy()

    fig = plt.figure(figsize=(11, 4))
    x = pd.PeriodIndex(last_rows["target_quarter"].astype(str), freq="Q").to_timestamp(how="end")
    plt.plot(x, last_rows["actual_disp"], label="actual", marker="o")
    plt.plot(x, last_rows["nowcast_mean_disp"], label="final nowcast (before release)", marker="o")
    plt.xlabel("Target quarter")
    plt.ylabel(ylab)
    plt.title(f"{model} | Actual vs final nowcast (last {len(last_rows)} quarters)")
    plt.legend()
    _savefig(fig, outdir / f"actual_vs_final_nowcast__{model}")


def fig_all_nowcasts_calendar_current_target(now: pd.DataFrame, model: str, outdir: Path, ylab: str) -> None:
    df = now[(now["model"] == model) & (now["asof_in_target"] == True)].copy()
    if df.empty:
        return
    if "release_date" not in df.columns:
        return

    df = df.sort_values(["release_date", "q"]).copy()

    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.plot(df["release_date"], df["nowcast_mean_disp"], marker="o", linewidth=1.5, label="current-quarter nowcast")

    if "actual_disp" in df.columns and df["actual_disp"].notna().any():
        for tq, g in df.groupby(df["target_quarter"].astype(str)):
            if g["actual_disp"].notna().any():
                a = float(g["actual_disp"].dropna().iloc[0])
                ax.hlines(a, xmin=g["release_date"].min(), xmax=g["release_date"].max(),
                          linestyles="dashed", linewidth=1)

    ax.set_xlabel("Release date (calendar)")
    ax.set_ylabel(ylab)
    ax.set_title(f"{model} | Current-quarter nowcasts over time (calendar dates)")
    ax.legend()
    _savefig(fig, outdir / f"calendar_current_target__{model}")


def fig_all_nowcasts_calendar_by_target(now: pd.DataFrame, model: str, outdir: Path, ylab: str, last_k_quarters: int = 12) -> None:
    df = now[now["model"] == model].copy()
    if df.empty:
        return
    if "release_date" not in df.columns or "target_quarter" not in df.columns:
        return

    tq_list = df["target_quarter"].dropna().astype(str).unique().tolist()
    tq_periods = [(t, _period_str_to_periodQ(t)) for t in tq_list]
    tq_periods = [(t, p) for t, p in tq_periods if p is not None]
    tq_periods.sort(key=lambda x: x[1])
    tq_use = [t for (t, _) in tq_periods][-last_k_quarters:]

    df = df[df["target_quarter"].astype(str).isin(set(tq_use))].copy()
    df = df.sort_values(["target_quarter", "release_date", "q"]).copy()

    fig = plt.figure(figsize=(12, 5))
    ax = plt.gca()

    for tq in tq_use:
        g = df[df["target_quarter"].astype(str) == str(tq)].copy()
        if g.empty:
            continue
        g = g.sort_values(["release_date", "q"]).groupby("release_date", as_index=False).last()

        ax.plot(g["release_date"], g["nowcast_mean_disp"], marker="o", linewidth=1.3, label=str(tq))

        if "actual_disp" in g.columns and g["actual_disp"].notna().any():
            a = float(g["actual_disp"].dropna().iloc[0])
            ax.hlines(a, xmin=g["release_date"].min(), xmax=g["release_date"].max(),
                      linestyles="dashed", linewidth=1)

    ax.set_xlabel("Release date (calendar)")
    ax.set_ylabel(ylab)
    ax.set_title(f"{model} | All targets over time (calendar dates) | last {len(tq_use)} quarters")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, outdir / f"calendar_all_targets__{model}")


# =============================================================================
# Orchestration
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Path to outputs/<run_id>")

    ap.add_argument("--actual_vintage", type=str, default="data/NL/2025-11-14.xlsx",
                    help="Relative path to most complete vintage used as 'actual' GDP")
    ap.add_argument("--spec_file", type=str, default="data/Spec_NL.xlsx",
                    help="Relative path to Spec file")
    ap.add_argument("--gdp_series_id", type=str, default="Real GDP")
    ap.add_argument("--sample_start", type=str, default="1985-01-01")

    ap.add_argument("--display_units", type=str, default="annualized", choices=["qoq", "annualized"],
                    help="Units for plots/tables. 'qoq' converts from annualized via compounding.")

    ap.add_argument("--last_k_quarters", type=int, default=12)
    ap.add_argument("--last_k_revision_quarters", type=int, default=6)

    # density options
    ap.add_argument("--hac_lag", type=int, default=1, help="HAC lag for DM test (Newey–West).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    project_root = Path.cwd()

    fig_dir = run_dir / "figures"
    tab_dir = run_dir / "tables"
    _ensure_dir(fig_dir)
    _ensure_dir(tab_dir)

    now, packs, meta = scan_run(run_dir)

    # Attach realized GDP + RW benchmark
    try:
        actual_by_q = build_actual_gdp_lookup(
            project_root=project_root,
            actual_vintage_rel=args.actual_vintage,
            spec_file_rel=args.spec_file,
            gdp_series_id=args.gdp_series_id,
            sample_start=args.sample_start,
        )
        now = attach_actual_and_rw(now, actual_by_q)
    except Exception as e:
        print(f"[make_artifacts] WARNING: could not attach actual GDP / RW benchmark: {e}")
        now = now.copy()
        now["actual"] = np.nan
        now["rw_nowcast"] = np.nan

    # Display scale (assumes base is annualized)
    ylab = "Real GDP growth (QoQ, %)" if args.display_units == "qoq" else "Real GDP growth (annualized, %)"
    now = _apply_display_units(now, display_units=args.display_units, base_units="annualized")

    # Derived columns
    now = add_release_step_columns(now)
    now = add_release_date(now)

    # Save enriched nowcasts
    (run_dir / "nowcasts").mkdir(parents=True, exist_ok=True)
    now.to_csv(run_dir / "nowcasts" / "nowcasts_with_actual_rw.csv", index=False)

    # ----------------------------
    # Tables: run summary + mean error metrics (existing)
    # ----------------------------
    run_summary = table_run_summary(meta)
    run_summary.to_csv(tab_dir / "run_summary.csv", index=False)
    _to_latex_table(run_summary, tab_dir / "run_summary.tex", caption="Run summary", label="tab:run_summary")

    err_wq = table_nowcast_errors_within_quarter(now)
    err_wq.to_csv(tab_dir / "nowcast_errors_within_quarter.csv", index=False)
    _to_latex_table(
        err_wq,
        tab_dir / "nowcast_errors_within_quarter.tex",
        caption=f"Within-quarter mean error metrics (units: {args.display_units})",
        label="tab:nowcast_errors_within_quarter",
    )

    err_final = table_nowcast_errors_final_before_release(now)
    err_final.to_csv(tab_dir / "nowcast_errors_final_before_release.csv", index=False)
    _to_latex_table(
        err_final,
        tab_dir / "nowcast_errors_final_before_release.tex",
        caption=f"Final-before-release mean error metrics (units: {args.display_units})",
        label="tab:nowcast_errors_final_before_release",
    )

    # ----------------------------
    # Density tables: scores + summaries + DM tests
    # ----------------------------
    dens_cur = density_scores_current_quarter(now, packs, display_units=args.display_units)
    dens_cur.to_csv(tab_dir / "density_scores_current_quarter.csv", index=False)

    dens_final = density_scores_final_before_release(now, packs, display_units=args.display_units)
    dens_final.to_csv(tab_dir / "density_scores_final_before_release.csv", index=False)

    dens_cur_sum = summarize_density_scores(dens_cur, by_step=False)
    dens_cur_sum.to_csv(tab_dir / "density_summary_current_quarter.csv", index=False)

    dens_cur_by_step = summarize_density_scores(dens_cur, by_step=True)
    dens_cur_by_step.to_csv(tab_dir / "density_summary_current_quarter_by_step.csv", index=False)

    dens_final_sum = summarize_density_scores(dens_final, by_step=False)
    dens_final_sum.to_csv(tab_dir / "density_summary_final_before_release.csv", index=False)

    # DM tests: only if at least 2 models exist
    models = sorted(now["model"].dropna().unique())
    if len(models) >= 2 and "note" not in dens_final.columns:
        # choose first two models for DM (you can extend easily)
        mA, mB = models[0], models[1]
        A = dens_final[dens_final["model"] == mA].copy()
        B = dens_final[dens_final["model"] == mB].copy()

        # align by target_quarter for final-before-release
        dm_crps = dm_test(A, B, key_cols=["target_quarter"], score_col="crps", hac_lag=args.hac_lag)
        dm_ls = dm_test(A, B, key_cols=["target_quarter"], score_col="logscore", hac_lag=args.hac_lag)

        dm_out = pd.concat([dm_crps.assign(model_A=mA, model_B=mB),
                            dm_ls.assign(model_A=mA, model_B=mB)], ignore_index=True)
        dm_out.to_csv(tab_dir / "dm_tests_final_before_release.csv", index=False)
    else:
        pd.DataFrame([{"note": "DM tests skipped (need >=2 models and density scores)."}]).to_csv(
            tab_dir / "dm_tests_final_before_release.csv", index=False
        )

    # ----------------------------
    # Figures: mean + density
    # ----------------------------
    if not now.empty:
        for model in models:
            # existing mean plots
            fig_within_quarter_nowcast_paths(now, model, fig_dir, ylab=ylab, last_k_quarters=min(8, args.last_k_quarters))
            fig_revision_paths_from_qstart_until_release(now, model, fig_dir, ylab=ylab, last_k_quarters=args.last_k_revision_quarters)
            fig_actual_vs_final_nowcast_over_time(now, model, fig_dir, ylab=ylab, last_k_quarters=args.last_k_quarters)
            fig_all_nowcasts_calendar_current_target(now, model, fig_dir, ylab=ylab)
            fig_all_nowcasts_calendar_by_target(now, model, fig_dir, ylab=ylab, last_k_quarters=args.last_k_quarters)

            # NEW density plots
            fig_fan_chart_current_target(now, packs, model, fig_dir, ylab=ylab, display_units=args.display_units)
            fig_density_heatmap_current_target(now, packs, model, fig_dir, ylab=ylab, display_units=args.display_units)

            # PIT histograms
            if "note" not in dens_cur.columns:
                fig_pit_hist(dens_cur, model, fig_dir, title_suffix="current_quarter")
            if "note" not in dens_final.columns:
                fig_pit_hist(dens_final, model, fig_dir, title_suffix="final_before_release")

    # ----------------------------
    # Index file
    # ----------------------------
    index_lines = []
    index_lines.append(f"Run: {run_dir.name}")
    index_lines.append("")
    index_lines.append("Figures (PNG/PDF):")
    for p in sorted(fig_dir.glob("*.png")):
        index_lines.append(f"  - {p.name}")
    index_lines.append("")
    index_lines.append("Tables (CSV/TEX):")
    for p in sorted(tab_dir.glob("*.csv")):
        index_lines.append(f"  - {p.name}")
    index_lines.append("")
    index_lines.append("Nowcasts:")
    index_lines.append("  - nowcasts_with_actual_rw.csv (enriched + display-scale + rel-step + release_date)")
    index_lines.append("")
    index_lines.append("Density scoring notes:")
    index_lines.append("  - Requires runs saved with posterior draws (SAVE_DRAWS=True).")
    index_lines.append("  - Uses nowcast_draws_matrix + nowcast_targets from posterior.npz written by save_subrun_outputs().")

    (run_dir / "ARTIFACTS_INDEX.txt").write_text("\n".join(index_lines))

    print(f"Artifacts written to:\n  {fig_dir}\n  {tab_dir}\nIndex:\n  {run_dir/'ARTIFACTS_INDEX.txt'}")


if __name__ == "__main__":
    main()