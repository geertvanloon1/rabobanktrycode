#!/usr/bin/env python3
"""
plottablebothselection.py

Cleaned results script for runs with 4 DFM variants, e.g.
    bay, bay_fs, bay_sv, bay_sv_fs

What this script does:
1) loads the run outputs
2) attaches realized GDP from a full "actual" vintage
3) evaluates FINAL within-quarter point nowcasts
4) evaluates FINAL within-quarter density nowcasts
5) adds AR(1) benchmark for both point and density
6) adds RW benchmark for point forecasts
7) creates a small set of tables and plots
8) keeps fan charts for all DFM models found in the run
9) creates nowcasting-grid tables by month-in-quarter x release-within-month
   relative to RW, in the style of standard nowcasting papers

Outputs:
- tables/point_errors_final_within_quarter.csv
- tables/density_summary_final_within_quarter.csv
- tables/dm_tests_point_final_within_quarter.csv
- tables/dm_tests_density_final_within_quarter.csv
- tables/density_scores_final_within_quarter.csv
- tables/point_errors_by_month_release.csv
- tables/relative_mae_by_month_release_vs_rw.csv
- tables/relative_rmsfe_by_month_release_vs_rw.csv
- tables/relative_mae_grid__<model>__vs_rw.csv
- tables/relative_rmsfe_grid__<model>__vs_rw.csv
- nowcasts/nowcasts_with_actual_rw.csv

Figures:
- figures/actual_vs_nowcasts_final_within_quarter.png
- figures/fan_chart_current_target__<model>.png
- figures/fan_chart_final_within_quarter__<model>.png

Notes:
- No period analysis
- No PIT histograms
- No loss-difference plots
- No LaTeX export
- RW is only for point forecast comparison
- AR(1) is used for both point and density comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from math import erf, sqrt
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from zhangnowcast.data.data import build_zhang_data


# python3 scripts/plottablebothselection.py --run_dir outputs/20260304_152004_4b325e02

# =============================================================================
# Matplotlib style
# =============================================================================

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
        "font.size": 10,
    }
)

# =============================================================================
# Utilities
# =============================================================================


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(fig, outbase: Path) -> None:
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=200)
    plt.close(fig)


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _coerce_month(x) -> pd.Timestamp:
    return pd.to_datetime(x, format="mixed", errors="coerce").to_period("M").to_timestamp()


def _nice(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_=. " else "_" for c in str(s))


def _period_str_to_periodQ(s: str) -> Optional[pd.Period]:
    try:
        return pd.Period(str(s), freq="Q")
    except Exception:
        return None


def _padded_ylim(ymin: float, ymax: float, pad_frac: float = 0.05) -> Optional[Tuple[float, float]]:
    if not (np.isfinite(ymin) and np.isfinite(ymax)):
        return None
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    rng = ymax - ymin
    if not np.isfinite(rng):
        return None
    if rng <= 0:
        pad = max(0.5, abs(ymin) * pad_frac)
        return (ymin - pad, ymax + pad)
    pad = pad_frac * rng
    return (ymin - pad, ymax + pad)


def _newey_west_var(d: np.ndarray, L: int) -> float:
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

    return float(var / T)


def _norm_pval_from_z(z: float) -> float:
    return float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0)))))


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
    x = np.asarray(draws_ann, dtype=float) / 100.0
    return 100.0 * (np.power(1.0 + x, 1.0 / 4.0) - 1.0)


def qoq_draws_to_ann(draws_qoq: np.ndarray) -> np.ndarray:
    x = np.asarray(draws_qoq, dtype=float) / 100.0
    return 100.0 * (np.power(1.0 + x, 4.0) - 1.0)


def _apply_display_units(now: pd.DataFrame, display_units: str, base_units: str = "annualized") -> pd.DataFrame:
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
# Actual GDP + RW benchmark
# =============================================================================


def build_actual_gdp_lookup(
    project_root: Path,
    actual_vintage_rel: str,
    spec_file_rel: str,
    gdp_series_id: str,
    sample_start: str,
) -> pd.Series:
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
        raise AttributeError("Could not find quarterly GDP array on ZhangData.")

    s = pd.Series(y, index=q_labels)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def attach_actual_and_rw(now: pd.DataFrame, actual_by_q: pd.Series) -> pd.DataFrame:
    df = now.copy()
    if "target_quarter" not in df.columns:
        df["actual"] = np.nan
        df["rw_nowcast"] = np.nan
        return df

    df["actual"] = df["target_quarter"].astype(str).map(actual_by_q)

    tq = df["target_quarter"].astype(str).apply(_period_str_to_periodQ)
    prev = tq.apply(lambda p: str(p - 1) if p is not None else None)
    df["rw_nowcast"] = prev.map(actual_by_q)

    return df


# =============================================================================
# Posterior pack loading
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
    nowcast_targets: np.ndarray
    nowcast_draws_matrix: np.ndarray
    nowcast_draws_legacy: np.ndarray


def load_posterior_pack(model: str, subdir: Path) -> Optional[PosteriorPack]:
    npz_path = subdir / "posterior.npz"
    meta_path = subdir / "posterior_meta.json"
    if not npz_path.exists() or not meta_path.exists():
        return None

    meta = _safe_read_json(meta_path)
    tag = meta.get("subrun_tag", subdir.name)
    month = pd.to_datetime(meta.get("month_T", "1970-01-01"), format="mixed", errors="coerce")
    q = int(meta.get("q", -1))
    miq = int(meta.get("month_in_quarter", -1))
    vint = str(meta.get("vintage_file", ""))

    z = np.load(npz_path, allow_pickle=True)

    def get(name, default=None):
        return z[name] if name in z.files else default

    m0 = _coerce_month(month)
    miq_use = miq if miq > 0 else (((int(m0.month) - 1) % 3) + 1)

    nowcast_draws_matrix = get("nowcast_draws_matrix")
    nowcast_targets = get("nowcast_targets")

    nowcast_draws_legacy = get("nowcast_draws")
    if nowcast_draws_matrix is None or np.asarray(nowcast_draws_matrix).size == 0:
        if nowcast_draws_legacy is not None and np.asarray(nowcast_draws_legacy).ndim == 1:
            nowcast_draws_matrix = np.asarray(nowcast_draws_legacy, dtype=float)[:, None]
        else:
            nowcast_draws_matrix = np.asarray([], dtype=float)

    if nowcast_targets is None or np.asarray(nowcast_targets).size == 0:
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
        nowcast_targets=np.asarray(nowcast_targets, dtype=object),
        nowcast_draws_matrix=np.asarray(nowcast_draws_matrix, dtype=float),
        nowcast_draws_legacy=np.asarray(nowcast_draws_legacy, dtype=float)
        if nowcast_draws_legacy is not None
        else np.asarray([], dtype=float),
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

    if "month" in now.columns:
        now["month"] = pd.to_datetime(now["month"], format="mixed", errors="coerce").dt.to_period("M").dt.to_timestamp()
    if "q" in now.columns:
        now["q"] = pd.to_numeric(now["q"], errors="coerce").astype("Int64")

    if "month_in_quarter" not in now.columns and "month" in now.columns:
        now["month_in_quarter"] = ((now["month"].dt.month - 1) % 3) + 1

    if "asof_quarter" not in now.columns and "month" in now.columns:
        now["asof_quarter"] = now["month"].dt.to_period("Q").astype(str)

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
# Derived columns
# =============================================================================


def _quarter_start_month(tq: str) -> Optional[pd.Timestamp]:
    p = _period_str_to_periodQ(tq)
    if p is None:
        return None
    return p.start_time.to_period("M").to_timestamp()


def _months_diff(m1: pd.Timestamp, m0: pd.Timestamp) -> int:
    return (m1.year - m0.year) * 12 + (m1.month - m0.month)


def _rel_step_label(step: int) -> str:
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
        if pd.isna(q0) or pd.isna(m) or pd.isna(q):
            steps.append(np.nan)
            continue
        md = _months_diff(pd.Timestamp(m), pd.Timestamp(q0))
        steps.append(int(md * 3 + (int(q) - 1)))

    df["rel_step_in_target"] = steps
    df["rel_step_label"] = df["rel_step_in_target"].apply(
        lambda x: _rel_step_label(int(x)) if np.isfinite(x) else ""
    )
    return df


def add_release_date(now: pd.DataFrame) -> pd.DataFrame:
    df = now.copy()

    if "vintage_file" in df.columns:
        stem = df["vintage_file"].astype(str).str.replace(".xlsx", "", regex=False)
        rd = pd.to_datetime(stem, errors="coerce")
    else:
        rd = pd.Series(pd.NaT, index=df.index)

    if "month" in df.columns and "q" in df.columns:
        fallback = pd.to_datetime(df["month"], errors="coerce") + pd.to_timedelta(
            (pd.to_numeric(df["q"], errors="coerce").fillna(1).astype(int) - 1) * 10, unit="D"
        )
        rd = rd.fillna(fallback)

    df["release_date"] = rd
    return df


# =============================================================================
# Final within-quarter row selection
# =============================================================================


def select_final_within_quarter(now: pd.DataFrame) -> pd.DataFrame:
    df = now.copy()

    if "asof_in_target" in df.columns:
        df = df[df["asof_in_target"] == True].copy()  # noqa: E712

    if "rel_step_in_target" in df.columns:
        df = df[df["rel_step_in_target"].between(0, 8)].copy()

    if df.empty:
        return df

    sort_cols = ["model", "target_quarter"]
    if "rel_step_in_target" in df.columns:
        sort_cols += ["rel_step_in_target"]
    if "month" in df.columns:
        sort_cols += ["month"]
    if "q" in df.columns:
        sort_cols += ["q"]

    df = df.sort_values(sort_cols)
    return df.groupby(["model", "target_quarter"], as_index=False).tail(1).reset_index(drop=True)


# =============================================================================
# Within-quarter grid selection
# =============================================================================


def select_within_target_grid(now: pd.DataFrame) -> pd.DataFrame:
    """
    Keep all rows that belong to the target quarter and fall in the standard
    3x3 nowcasting grid:
      columns -> month_in_quarter in {1,2,3}
      rows    -> q in {1,2,3}
    """
    df = now.copy()

    if "asof_in_target" in df.columns:
        df = df[df["asof_in_target"] == True].copy()  # noqa: E712

    if "rel_step_in_target" in df.columns:
        df = df[df["rel_step_in_target"].between(0, 8)].copy()

    if "month_in_quarter" in df.columns:
        df = df[df["month_in_quarter"].isin([1, 2, 3])].copy()

    if "q" in df.columns:
        df = df[df["q"].isin([1, 2, 3])].copy()

    return df.reset_index(drop=True)


# =============================================================================
# AR(1) helpers
# =============================================================================


def _build_actual_series_from_lookup(
    actual_by_q: pd.Series,
    display_units: str,
    base_units: str = "annualized",
) -> pd.Series:
    if actual_by_q is None or len(actual_by_q) == 0:
        return pd.Series(dtype=float)

    s = pd.Series(actual_by_q).copy()
    s.index = pd.PeriodIndex([str(x) for x in s.index], freq="Q")
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()

    if base_units != display_units:
        if base_units == "annualized" and display_units == "qoq":
            s = ann_to_qoq_pct(s)
        elif base_units == "qoq" and display_units == "annualized":
            s = qoq_to_ann_pct(s)
        else:
            raise ValueError(f"Unsupported unit conversion {base_units} -> {display_units}")

    s = s[~s.index.duplicated(keep="last")]
    return s


def _estimate_ar1_normal(y: pd.Series) -> Optional[dict]:
    y = pd.to_numeric(y, errors="coerce")
    y = y[y.notna()]
    if y.size < 10:
        return None

    y_t = y.iloc[1:].to_numpy(dtype=float)
    y_lag = y.iloc[:-1].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(y_lag), y_lag])

    beta_hat, _, _, _ = np.linalg.lstsq(X, y_t, rcond=None)
    alpha_hat = float(beta_hat[0])
    phi_hat = float(beta_hat[1])

    resid = y_t - (alpha_hat + phi_hat * y_lag)
    sigma2_hat = float(np.mean(resid ** 2))

    return {
        "alpha": alpha_hat,
        "phi": phi_hat,
        "sigma2": sigma2_hat,
        "y_series": y,
    }


def _ar1_one_step_params(ar1: dict, tq: str) -> Tuple[float, float]:
    if ar1 is None:
        return float("nan"), float("nan")

    p = _period_str_to_periodQ(tq)
    if p is None:
        return float("nan"), float("nan")

    y_series: pd.Series = ar1["y_series"]
    prev = p - 1
    if prev not in y_series.index:
        return float("nan"), float("nan")

    y_prev = float(y_series.loc[prev])
    mu = ar1["alpha"] + ar1["phi"] * y_prev
    sigma2 = ar1["sigma2"]
    return mu, sigma2


def _ar1_point_forecast_for_target(ar1: dict, tq: str) -> float:
    mu, _ = _ar1_one_step_params(ar1, tq)
    return float(mu) if np.isfinite(mu) else float("nan")


def _ar1_draws_for_target(ar1: dict, tq: str, S: int = 5000) -> Optional[np.ndarray]:
    mu, sigma2 = _ar1_one_step_params(ar1, tq)
    if not np.isfinite(mu) or not np.isfinite(sigma2) or sigma2 <= 0:
        return None
    return np.random.default_rng(789).normal(loc=mu, scale=np.sqrt(sigma2), size=S)


def _add_ar1_point_rows(
    now: pd.DataFrame,
    actual_by_q: Optional[pd.Series],
    display_units: str,
) -> pd.DataFrame:
    base = now.copy()
    dff = select_final_within_quarter(base)
    if dff.empty or "actual_disp" not in dff.columns:
        return base

    dff = dff[dff["actual_disp"].notna()].copy()
    if dff.empty:
        return base

    y_series = _build_actual_series_from_lookup(
        actual_by_q=actual_by_q,
        display_units=display_units,
        base_units="annualized",
    )
    ar1 = _estimate_ar1_normal(y_series)
    if ar1 is None:
        return base

    rows = []
    dff_unique = dff.sort_values("target_quarter").drop_duplicates(subset=["target_quarter"]).copy()

    for _, r in dff_unique.iterrows():
        tq = str(r["target_quarter"])
        mu = _ar1_point_forecast_for_target(ar1, tq)
        if not np.isfinite(mu):
            continue
        rr = r.copy()
        rr["model"] = "ar1"
        rr["nowcast_mean_disp"] = float(mu)
        rows.append(rr)

    if not rows:
        return base

    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True)


def _ensure_rw_rows_for_final(now: pd.DataFrame) -> pd.DataFrame:
    base = now.copy()
    df_final = select_final_within_quarter(base)
    if df_final.empty or "rw_nowcast_disp" not in df_final.columns:
        return base
    rw = df_final[df_final["rw_nowcast_disp"].notna()].copy()
    if rw.empty:
        return base
    rw["model"] = "RW"
    rw["nowcast_mean_disp"] = rw["rw_nowcast_disp"]
    return pd.concat([base, rw], ignore_index=True)


# =============================================================================
# Point forecast tables and DM tests
# =============================================================================


def table_point_errors_final_within_quarter(now: pd.DataFrame) -> pd.DataFrame:
    df = select_final_within_quarter(now)
    if df.empty:
        return pd.DataFrame([{"note": "No final-within-quarter rows available."}])

    if "actual_disp" not in df.columns or df["actual_disp"].isna().all():
        return pd.DataFrame([{"note": "No realized GDP attached (actual_disp all NaN)."}])

    df = df[df["actual_disp"].notna()].copy()
    if df.empty:
        return pd.DataFrame([{"note": "No rows with non-NaN actual_disp after selection."}])

    df["err"] = df["nowcast_mean_disp"] - df["actual_disp"]
    df["abs_err"] = df["err"].abs()
    df["sq_err"] = df["err"] ** 2

    out = []
    for model, g in df.groupby("model"):
        out.append(
            dict(
                model=str(model),
                n=int(len(g)),
                MAE=float(g["abs_err"].mean()),
                RMSFE=float(np.sqrt(g["sq_err"].mean())),
            )
        )

    order = ["RW", "ar1", "bay", "bay_fs", "bay_sv", "bay_sv_fs"]
    out_df = pd.DataFrame(out)
    out_df["order"] = out_df["model"].map(lambda x: order.index(x) if x in order else 999)
    out_df = out_df.sort_values(["order", "model"]).drop(columns=["order"]).reset_index(drop=True)
    return out_df


def dm_test_point_final_within_quarter(
    now: pd.DataFrame,
    model_A: str,
    model_B: str,
    loss: str = "ae",
    hac_lag: int = 1,
) -> pd.DataFrame:
    dff = select_final_within_quarter(now)
    dff = dff[dff["actual_disp"].notna()].copy()
    if dff.empty:
        return pd.DataFrame([{"note": "No final-within-quarter rows with actual_disp for point DM."}])

    A = dff[dff["model"] == model_A][["target_quarter", "nowcast_mean_disp", "actual_disp"]].copy()
    B = dff[dff["model"] == model_B][["target_quarter", "nowcast_mean_disp", "actual_disp"]].copy()
    if A.empty or B.empty:
        return pd.DataFrame([{"note": f"Missing models for point DM: {model_A} or {model_B}"}])

    A = A.rename(columns={"nowcast_mean_disp": "fc_A"})
    B = B.rename(columns={"nowcast_mean_disp": "fc_B"})
    m = A.merge(B, on=["target_quarter", "actual_disp"], how="inner")
    if m.empty:
        return pd.DataFrame([{"note": f"No overlap quarters for point DM: {model_A} vs {model_B}"}])

    eA = (m["fc_A"] - m["actual_disp"]).to_numpy(dtype=float)
    eB = (m["fc_B"] - m["actual_disp"]).to_numpy(dtype=float)

    if loss.lower() == "se":
        d = eA**2 - eB**2
        loss_name = "SE"
    elif loss.lower() == "ae":
        d = np.abs(eA) - np.abs(eB)
        loss_name = "AE"
    else:
        raise ValueError("loss must be 'ae' or 'se'")

    d = d[np.isfinite(d)]
    if d.size < 8:
        return pd.DataFrame([{"note": "Too few paired observations for point DM", "n": int(d.size)}])

    mean_d = float(np.mean(d))
    var_mean = _newey_west_var(d, L=hac_lag)
    if not np.isfinite(var_mean) or var_mean <= 0:
        return pd.DataFrame([{"note": "NW variance failed for point DM", "n": int(d.size)}])

    dm_stat = mean_d / np.sqrt(var_mean)
    pval = _norm_pval_from_z(dm_stat)

    return pd.DataFrame(
        [
            {
                "comparison": f"{model_A} vs {model_B}",
                "loss": loss_name,
                "n": int(d.size),
                "hac_lag": int(hac_lag),
                "mean_diff_LA_minus_LB": mean_d,
                "dm_stat": float(dm_stat),
                "p_value": float(pval),
            }
        ]
    )


# =============================================================================
# Nowcasting grid tables: month-in-quarter x release-within-month
# =============================================================================


def point_errors_by_month_release(now: pd.DataFrame) -> pd.DataFrame:
    """
    Long table with MAE and RMSFE for each model at each cell:
      (month_in_quarter, q)
    Also includes RW benchmark computed from rw_nowcast_disp on the same rows.
    """
    df = select_within_target_grid(now)
    if df.empty:
        return pd.DataFrame([{"note": "No within-target rows available for month-release point evaluation."}])

    if "actual_disp" not in df.columns or df["actual_disp"].isna().all():
        return pd.DataFrame([{"note": "No realized GDP attached (actual_disp all NaN)."}])

    df = df[df["actual_disp"].notna()].copy()
    if df.empty:
        return pd.DataFrame([{"note": "No rows with non-NaN actual_disp after within-target selection."}])

    out = []

    # Model rows
    for (model, miq, q), g in df.groupby(["model", "month_in_quarter", "q"]):
        fc = pd.to_numeric(g["nowcast_mean_disp"], errors="coerce")
        y = pd.to_numeric(g["actual_disp"], errors="coerce")
        ok = fc.notna() & y.notna()
        if ok.sum() == 0:
            continue

        err = fc[ok] - y[ok]
        out.append(
            {
                "model": str(model),
                "month_in_quarter": int(miq),
                "q": int(q),
                "n": int(ok.sum()),
                "MAE": float(np.mean(np.abs(err))),
                "RMSFE": float(np.sqrt(np.mean(err**2))),
            }
        )

    # RW rows computed on same cell rows
    if "rw_nowcast_disp" in df.columns:
        for (miq, q), g in df.groupby(["month_in_quarter", "q"]):
            fc = pd.to_numeric(g["rw_nowcast_disp"], errors="coerce")
            y = pd.to_numeric(g["actual_disp"], errors="coerce")
            ok = fc.notna() & y.notna()
            if ok.sum() == 0:
                continue

            err = fc[ok] - y[ok]
            out.append(
                {
                    "model": "RW",
                    "month_in_quarter": int(miq),
                    "q": int(q),
                    "n": int(ok.sum()),
                    "MAE": float(np.mean(np.abs(err))),
                    "RMSFE": float(np.sqrt(np.mean(err**2))),
                }
            )

    if not out:
        return pd.DataFrame([{"note": "No point errors computed for month-release grid."}])

    out_df = pd.DataFrame(out)
    order = ["RW", "ar1", "bay", "bay_fs", "bay_sv", "bay_sv_fs"]
    out_df["order"] = out_df["model"].map(lambda x: order.index(x) if x in order else 999)
    out_df = out_df.sort_values(["order", "model", "month_in_quarter", "q"]).drop(columns=["order"]).reset_index(drop=True)
    return out_df


def relative_score_by_month_release_vs_benchmark(
    point_table: pd.DataFrame,
    score_col: str = "MAE",
    benchmark_model: str = "RW",
) -> pd.DataFrame:
    """
    Computes percentage difference relative to the benchmark:
        100 * (score_model - score_benchmark) / score_benchmark

    Negative values mean improvement relative to the benchmark.
    """
    if point_table.empty or "note" in point_table.columns:
        return point_table.copy()

    needed = {"model", "month_in_quarter", "q", score_col}
    if not needed.issubset(point_table.columns):
        return pd.DataFrame([{"note": f"Point table missing columns needed for relative {score_col} table."}])

    bench = point_table[point_table["model"] == benchmark_model][["month_in_quarter", "q", "n", score_col]].copy()
    if bench.empty:
        return pd.DataFrame([{"note": f"Benchmark model '{benchmark_model}' not found for relative {score_col} table."}])

    bench = bench.rename(
        columns={
            "n": "n_benchmark",
            score_col: f"{score_col}_{benchmark_model}",
        }
    )

    models = point_table[point_table["model"] != benchmark_model].copy()
    m = models.merge(bench, on=["month_in_quarter", "q"], how="left")
    if m.empty:
        return pd.DataFrame([{"note": f"No overlap with benchmark '{benchmark_model}' for relative {score_col} table."}])

    denom = pd.to_numeric(m[f"{score_col}_{benchmark_model}"], errors="coerce")
    num = pd.to_numeric(m[score_col], errors="coerce") - denom
    rel = 100.0 * num / denom.replace(0.0, np.nan)

    out = m.copy()
    out["benchmark_model"] = benchmark_model
    out[f"relative_{score_col.lower()}_pct"] = rel

    cols = [
        "model",
        "benchmark_model",
        "month_in_quarter",
        "q",
        "n",
        "n_benchmark",
        score_col,
        f"{score_col}_{benchmark_model}",
        f"relative_{score_col.lower()}_pct",
    ]
    out = out[cols].copy()

    order = ["bay", "bay_fs", "bay_sv", "bay_sv_fs", "ar1"]
    out["order"] = out["model"].map(lambda x: order.index(x) if x in order else 999)
    out = out.sort_values(["order", "model", "q", "month_in_quarter"]).drop(columns=["order"]).reset_index(drop=True)
    return out


def relative_grid_for_model(
    rel_table: pd.DataFrame,
    model: str,
    rel_col: str,
) -> pd.DataFrame:
    """
    Returns a 3x3 grid table:
      rows    -> release q
      columns -> month_in_quarter
    """
    if rel_table.empty or "note" in rel_table.columns:
        return rel_table.copy()

    df = rel_table[rel_table["model"] == model].copy()
    if df.empty:
        return pd.DataFrame([{"note": f"No relative grid rows found for model '{model}'."}])

    pivot = (
        df.pivot_table(
            index="q",
            columns="month_in_quarter",
            values=rel_col,
            aggfunc="mean",
        )
        .reindex(index=[1, 2, 3], columns=[1, 2, 3])
    )

    pivot.index = ["1st", "2nd", "3rd"]
    pivot.columns = ["1st Month", "2nd Month", "3rd Month"]
    pivot.index.name = "Release"

    return pivot.reset_index()


# =============================================================================
# Density scoring and DM tests
# =============================================================================


def crps_from_draws(draws: np.ndarray, y: float) -> float:
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


def hist_logscore(draws: np.ndarray, y: float, nbins: int = 60, eps: float = 1e-12) -> float:
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


def density_scores_final_within_quarter(
    now: pd.DataFrame,
    packs: List[PosteriorPack],
    display_units: str,
    actual_by_q: Optional[pd.Series] = None,
) -> pd.DataFrame:
    df = select_final_within_quarter(now)
    if df.empty:
        return pd.DataFrame([{"note": "No final-within-quarter rows available for density scoring."}])

    df = df[df["actual_disp"].notna()].copy()
    if df.empty:
        return pd.DataFrame([{"note": "No final-within-quarter rows with non-NaN actual_disp."}])

    df = df.sort_values(["model", "target_quarter", "month", "q"]).copy()
    pidx = _pack_index(packs)

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

        out.append(
            dict(
                model=model,
                subrun_tag=tag,
                release_date=pd.to_datetime(r.get("release_date", pd.NaT)),
                month=pd.to_datetime(r.get("month", pd.NaT)),
                q=int(r.get("q", -1)),
                target_quarter=str(r["target_quarter"]),
                crps=crps_from_draws(draws, y),
                logscore=hist_logscore(draws, y),
                cover90=float(lo90 <= y <= hi90),
                width90=float(hi90 - lo90),
                draws_n=int(draws.size),
            )
        )

    y_series = _build_actual_series_from_lookup(
        actual_by_q=actual_by_q,
        display_units=display_units,
        base_units="annualized",
    )
    ar1 = _estimate_ar1_normal(y_series)
    if ar1 is not None:
        df_ar = select_final_within_quarter(now)
        df_ar = df_ar[df_ar["actual_disp"].notna()].copy()
        df_ar = df_ar.sort_values("target_quarter").drop_duplicates(subset=["target_quarter"])

        for _, r in df_ar.iterrows():
            tq = str(r["target_quarter"])
            y = float(r["actual_disp"])
            draws = _ar1_draws_for_target(ar1, tq, S=5000)
            if draws is None:
                continue
            lo90, hi90 = np.quantile(draws, [0.05, 0.95])

            out.append(
                dict(
                    model="ar1",
                    subrun_tag="ar1_benchmark",
                    release_date=pd.to_datetime(r.get("release_date", pd.NaT)),
                    month=pd.to_datetime(r.get("month", pd.NaT)),
                    q=int(r.get("q", -1)),
                    target_quarter=tq,
                    crps=crps_from_draws(draws, y),
                    logscore=hist_logscore(draws, y),
                    cover90=float(lo90 <= y <= hi90),
                    width90=float(hi90 - lo90),
                    draws_n=int(draws.size),
                )
            )

    if not out:
        return pd.DataFrame([{"note": "No final-within-quarter density scores computed (missing draws)."}])

    return pd.DataFrame(out)


def summarize_density_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if "note" in scores.columns:
        return scores.copy()

    g = scores.groupby(["model"], as_index=False).agg(
        n=("crps", "count"),
        mean_crps=("crps", "mean"),
        mean_logscore=("logscore", "mean"),
        mean_cover90=("cover90", "mean"),
        mean_width90=("width90", "mean"),
    )

    order = ["ar1", "bay", "bay_fs", "bay_sv", "bay_sv_fs"]
    g["order"] = g["model"].map(lambda x: order.index(x) if x in order else 999)
    g = g.sort_values(["order", "model"]).drop(columns=["order"]).reset_index(drop=True)
    return g



# =============================================================================
# Cumulative log predictive score
# =============================================================================


def cumulative_logscore_differences(
    scores: pd.DataFrame,
    benchmark_model: str = "ar1",
) -> pd.DataFrame:
    """
    Build cumulative log predictive score differences relative to a benchmark.

    For each model m and target quarter t:
        diff_t = logscore_m,t - logscore_benchmark,t
        cumdiff_t = sum_{s <= t} diff_s

    Positive values mean model m beats the benchmark in density forecasting.
    """
    if scores.empty or "note" in scores.columns:
        return pd.DataFrame([{"note": "No density scores available for cumulative logscore plot."}])

    needed = {"model", "target_quarter", "logscore"}
    if not needed.issubset(scores.columns):
        return pd.DataFrame([{"note": "Density scores missing required columns for cumulative logscore plot."}])

    bench = scores[scores["model"] == benchmark_model][["target_quarter", "logscore"]].copy()
    if bench.empty:
        return pd.DataFrame([{"note": f"Benchmark model '{benchmark_model}' not found in density scores."}])

    bench = bench.rename(columns={"logscore": "logscore_benchmark"})

    models = sorted([m for m in scores["model"].dropna().astype(str).unique() if m != benchmark_model])

    out = []
    for model in models:
        g = scores[scores["model"] == model][["target_quarter", "logscore"]].copy()
        if g.empty:
            continue

        m = g.merge(bench, on="target_quarter", how="inner")
        if m.empty:
            continue

        m["tq_period"] = pd.PeriodIndex(m["target_quarter"].astype(str), freq="Q")
        m = m.sort_values("tq_period").reset_index(drop=True)

        m["benchmark_model"] = benchmark_model
        m["model"] = model
        m["logscore_diff"] = m["logscore"] - m["logscore_benchmark"]
        m["cum_logscore_diff"] = m["logscore_diff"].cumsum()

        out.append(
            m[
                [
                    "model",
                    "benchmark_model",
                    "target_quarter",
                    "tq_period",
                    "logscore",
                    "logscore_benchmark",
                    "logscore_diff",
                    "cum_logscore_diff",
                ]
            ].copy()
        )

    if not out:
        return pd.DataFrame([{"note": "No overlapping model/benchmark quarters for cumulative logscore plot."}])

    out_df = pd.concat(out, ignore_index=True)

    preferred_order = ["RW", "ar1", "bay", "bay_fs", "bay_sv", "bay_sv_fs"]
    out_df["order"] = out_df["model"].map(lambda x: preferred_order.index(x) if x in preferred_order else 999)
    out_df = out_df.sort_values(["order", "model", "tq_period"]).drop(columns=["order"]).reset_index(drop=True)
    return out_df


def fig_cumulative_logscore(
    cum_df: pd.DataFrame,
    outdir: Path,
    benchmark_model: str,
    model_order: Optional[List[str]] = None,
    filename_suffix: Optional[str] = None,
) -> None:
    """
    Plot cumulative log predictive score differences relative to benchmark.
    Positive values mean the model outperforms the benchmark.
    """
    if cum_df.empty or "note" in cum_df.columns:
        return

    df = cum_df.copy()
    df["tq_period"] = pd.PeriodIndex(df["target_quarter"].astype(str), freq="Q")
    df["tq_end"] = df["tq_period"].dt.to_timestamp(how="end")

    if model_order is None:
        model_order = sorted(df["model"].dropna().astype(str).unique())

    fig, ax = plt.subplots(figsize=(10, 5))

    for model in model_order:
        g = df[df["model"] == model].copy()
        if g.empty:
            continue
        g = g.sort_values("tq_period")
        ax.plot(
            g["tq_end"],
            g["cum_logscore_diff"],
            marker="o",
            linewidth=2.0,
            label=model,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Target quarter")
    ax.set_ylabel(f"Cumulative log predictive score over {benchmark_model}")
    ax.set_title(f"Cumulative log predictive score relative to {benchmark_model}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    suffix = f"__vs_{benchmark_model}" if filename_suffix is None else filename_suffix
    _savefig(fig, outdir / f"cumulative_logscore{suffix}")

def dm_test(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    key_cols: List[str],
    score_col: str,
    hac_lag: int = 1,
) -> pd.DataFrame:
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
    pval = _norm_pval_from_z(dm_stat)

    return pd.DataFrame(
        [
            {
                "score": score_col,
                "n": int(d.size),
                "hac_lag": int(hac_lag),
                "mean_diff_a_minus_b": mean_d,
                "dm_stat": float(dm_stat),
                "p_value": float(pval),
            }
        ]
    )


# =============================================================================
# Figures
# =============================================================================


def fig_actual_vs_models_final_within_quarter(
    now: pd.DataFrame,
    outdir: Path,
    ylab: str,
    model_order: List[str],
) -> None:
    df = select_final_within_quarter(now)
    if df.empty or "actual_disp" not in df.columns or df["actual_disp"].isna().all():
        return

    df = df[df["model"].isin(model_order)].copy()
    if df.empty:
        return

    df["tq_period"] = pd.PeriodIndex(df["target_quarter"].astype(str), freq="Q")
    df = df.sort_values(["tq_period", "model"])

    actual = df.groupby("tq_period", as_index=False)["actual_disp"].first()
    x_q = actual["tq_period"].dt.to_timestamp(how="end")
    y_q = actual["actual_disp"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_q, y_q, "k--o", label="Actual", markersize=5)

    for model in model_order:
        g = df[df["model"] == model].copy()
        if g.empty:
            continue
        g = g.sort_values("tq_period")
        xx = g["tq_period"].dt.to_timestamp(how="end")
        yy = g["nowcast_mean_disp"].to_numpy(dtype=float)
        ax.plot(xx, yy, linewidth=2.0, marker="o", markersize=5, label=model)

    ax.set_xlabel("Target quarter")
    ax.set_ylabel(ylab)
    ax.set_title("Actual vs nowcasts (final within-quarter)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _savefig(fig, outdir / "actual_vs_nowcasts_final_within_quarter")


def _fan_chart_data_current_target(
    now: pd.DataFrame,
    packs: List[PosteriorPack],
    model: str,
    display_units: str,
    bands=(0.5, 0.7, 0.9),
    base_units: str = "annualized",
) -> pd.DataFrame:
    df = now[(now["model"] == model) & (now["asof_in_target"] == True)].copy()  # noqa: E712
    if df.empty or "release_date" not in df.columns:
        return pd.DataFrame()

    df = df.sort_values(["release_date", "q"]).copy()
    pidx = _pack_index(packs)

    rows = []
    for _, r in df.iterrows():
        tag = str(r.get("subrun_tag", ""))
        pack = pidx.get((model, tag))
        draws = _get_draws_for_row(r, pack, display_units=display_units, base_units=base_units)
        if draws is None or draws.size < 50:
            continue

        rec = {
            "release_date": pd.to_datetime(r["release_date"]),
            "mean": float(np.mean(draws)),
            "actual": float(r["actual_disp"]) if pd.notna(r.get("actual_disp", np.nan)) else np.nan,
        }
        for b in bands:
            lo = (1 - b) / 2
            hi = 1 - lo
            rec[f"lo{int(b*100)}"] = float(np.quantile(draws, lo))
            rec[f"hi{int(b*100)}"] = float(np.quantile(draws, hi))
        rows.append(rec)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("release_date").reset_index(drop=True)


def _fan_chart_data_final_within_quarter(
    now: pd.DataFrame,
    packs: List[PosteriorPack],
    model: str,
    display_units: str,
    bands=(0.5, 0.7, 0.9),
    base_units: str = "annualized",
) -> pd.DataFrame:
    df_all = select_final_within_quarter(now)
    df = df_all[df_all["model"] == model].copy()
    if df.empty or "actual_disp" not in df.columns:
        return pd.DataFrame()

    df["tq_period"] = pd.PeriodIndex(df["target_quarter"].astype(str), freq="Q")
    df = df.sort_values("tq_period").copy()
    pidx = _pack_index(packs)

    rows = []
    for _, r in df.iterrows():
        tag = str(r.get("subrun_tag", ""))
        pack = pidx.get((model, tag))
        draws = _get_draws_for_row(r, pack, display_units=display_units, base_units=base_units)
        if draws is None or draws.size < 50:
            continue

        tq_end = r["tq_period"].to_timestamp(how="end")
        rec = {
            "tq_end": tq_end,
            "mean": float(np.mean(draws)),
            "actual": float(r["actual_disp"]) if pd.notna(r["actual_disp"]) else np.nan,
        }
        for b in bands:
            lo = (1 - b) / 2
            hi = 1 - lo
            rec[f"lo{int(b*100)}"] = float(np.quantile(draws, lo))
            rec[f"hi{int(b*100)}"] = float(np.quantile(draws, hi))
        rows.append(rec)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("tq_end").reset_index(drop=True)


def _compute_common_ylim_from_fan_data(dfs: List[pd.DataFrame], bands=(0.5, 0.7, 0.9)) -> Optional[Tuple[float, float]]:
    ymin = float("inf")
    ymax = float("-inf")

    lo_cols = [f"lo{int(b*100)}" for b in bands]
    hi_cols = [f"hi{int(b*100)}" for b in bands]

    for df in dfs:
        if df is None or df.empty:
            continue
        cols = []
        for c in ["mean", "actual"] + lo_cols + hi_cols:
            if c in df.columns:
                cols.append(c)
        if not cols:
            continue
        vals = pd.to_numeric(df[cols].stack(), errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        ymin = min(ymin, float(np.min(vals)))
        ymax = max(ymax, float(np.max(vals)))

    if not (np.isfinite(ymin) and np.isfinite(ymax)):
        return None

    return _padded_ylim(ymin, ymax, pad_frac=0.05)


def fig_fan_chart_current_target(
    now: pd.DataFrame,
    packs: List[PosteriorPack],
    model: str,
    outdir: Path,
    ylab: str,
    display_units: str,
    bands=(0.5, 0.7, 0.9),
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    qdf = _fan_chart_data_current_target(
        now=now,
        packs=packs,
        model=model,
        display_units=display_units,
        bands=bands,
    )
    if qdf.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 3.5))

    for b in sorted(bands, reverse=True):
        ax.fill_between(
            qdf["release_date"],
            qdf[f"lo{int(b*100)}"],
            qdf[f"hi{int(b*100)}"],
            alpha=0.2,
            label=f"{int(b*100)}% band" if b == max(bands) else None,
            color="#1f77b4",
        )

    ax.plot(qdf["release_date"], qdf["mean"], marker="o", linewidth=1.5, label="Posterior mean", color="#1f77b4")
    if qdf["actual"].notna().any():
        ax.plot(
            qdf["release_date"],
            qdf["actual"],
            marker="o",
            linewidth=1.5,
            label="Actual GDP",
            color="black",
            linestyle="--",
        )

    if ylim is not None and np.isfinite(ylim[0]) and np.isfinite(ylim[1]):
        ax.set_ylim(ylim)

    ax.set_title(f"{model} — current-quarter density nowcasts")
    ax.set_xlabel("Release date")
    ax.set_ylabel(ylab)
    ax.legend(frameon=False, ncol=3)
    _savefig(fig, outdir / f"fan_chart_current_target__{model}")


def fig_fan_chart_final_within_quarter(
    now: pd.DataFrame,
    packs: List[PosteriorPack],
    model: str,
    outdir: Path,
    ylab: str,
    display_units: str,
    bands=(0.5, 0.7, 0.9),
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    qdf = _fan_chart_data_final_within_quarter(
        now=now,
        packs=packs,
        model=model,
        display_units=display_units,
        bands=bands,
    )
    if qdf.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 3.5))

    for b in sorted(bands, reverse=True):
        ax.fill_between(
            qdf["tq_end"],
            qdf[f"lo{int(b*100)}"],
            qdf[f"hi{int(b*100)}"],
            alpha=0.2,
            label=f"{int(b*100)}% band" if b == max(bands) else None,
            color="#1f77b4",
        )

    ax.plot(qdf["tq_end"], qdf["mean"], color="#1f77b4", marker="o", linewidth=1.8, label="Posterior mean")

    if qdf["actual"].notna().any():
        ax.plot(qdf["tq_end"], qdf["actual"], color="black", marker="o", linestyle="--", linewidth=1.2, label="Actual GDP")

    if ylim is not None and np.isfinite(ylim[0]) and np.isfinite(ylim[1]):
        ax.set_ylim(ylim)

    ax.set_title(f"{model} — final within-quarter density nowcasts")
    ax.set_xlabel("Target quarter")
    ax.set_ylabel(ylab)
    ax.legend(frameon=False, ncol=3)
    _savefig(fig, outdir / f"fan_chart_final_within_quarter__{model}")


# =============================================================================
# Factor-selection analysis
# =============================================================================


def _fs_models_from_packs(packs: List[PosteriorPack]) -> List[str]:
    models = sorted({p.model for p in packs})
    return [m for m in models if m.endswith("_fs")]


def _load_selection_arrays(pack: PosteriorPack) -> Optional[dict]:
    """
    Read factor-selection arrays from posterior.npz.

    Expected:
    - z_draws: (S, K) binary inclusion indicators
    - r_draws: (S,) number of active factors
    - Lambda_mean: (N, K) factor loadings
    - series: (N,) series names
    """
    npz_path = pack.path / "posterior.npz"
    if not npz_path.exists():
        return None

    z = np.load(npz_path, allow_pickle=True)

    needed = ["z_draws", "r_draws", "Lambda_mean", "series"]
    for k in needed:
        if k not in z.files:
            return None

    z_draws = np.asarray(z["z_draws"])
    r_draws = np.asarray(z["r_draws"])
    lambda_mean = np.asarray(z["Lambda_mean"])
    series = np.asarray(z["series"], dtype=object)

    if z_draws.ndim != 2:
        return None
    if r_draws.ndim != 1:
        return None
    if lambda_mean.ndim != 2:
        return None

    return {
        "z_draws": z_draws,
        "r_draws": r_draws,
        "Lambda_mean": lambda_mean,
        "series": series,
    }


def factor_selection_by_vintage(packs: List[PosteriorPack]) -> pd.DataFrame:
    """
    Long table with posterior inclusion probability per factor per vintage.
    """
    rows = []
    for pack in packs:
        if not str(pack.model).endswith("_fs"):
            continue

        arr = _load_selection_arrays(pack)
        if arr is None:
            continue

        z_draws = arr["z_draws"]
        K = z_draws.shape[1]

        pips = np.mean(z_draws, axis=0)
        for k in range(K):
            rows.append(
                {
                    "model": str(pack.model),
                    "month": pd.to_datetime(pack.month),
                    "factor": f"F{k + 1}",
                    "factor_index": int(k + 1),
                    "pip": float(pips[k]),
                    "subrun_tag": str(pack.tag),
                }
            )

    if not rows:
        return pd.DataFrame([{"note": "No factor-selection posterior arrays found."}])

    out = pd.DataFrame(rows).sort_values(["model", "month", "factor_index"]).reset_index(drop=True)
    return out


# =============================================================================
# Factor inclusion distribution by nowcasting month
# =============================================================================


def factor_inclusion_distribution_by_month(packs: List[PosteriorPack]) -> pd.DataFrame:
    """
    For *_fs models, compute average inclusion rate for each factor by
    month_in_quarter.

    Output columns:
      model, month_in_quarter, factor, factor_index, inclusion_rate,
      n_draws_total, n_vintages
    """
    rows = []

    fs_models = sorted({p.model for p in packs if str(p.model).endswith("_fs")})

    for model in fs_models:
        model_packs = [p for p in packs if p.model == model]

        for miq in [1, 2, 3]:
            month_packs = [p for p in model_packs if int(p.month_in_quarter) == miq]
            if not month_packs:
                continue

            z_all = []
            n_vintages = 0

            for pack in month_packs:
                arr = _load_selection_arrays(pack)
                if arr is None:
                    continue

                z_draws = np.asarray(arr["z_draws"], dtype=float)
                if z_draws.ndim != 2 or z_draws.size == 0:
                    continue

                z_all.append(z_draws)
                n_vintages += 1

            if not z_all:
                continue

            z_cat = np.concatenate(z_all, axis=0)   # (total_draws, K)
            K = z_cat.shape[1]
            inc = np.mean(z_cat, axis=0)

            for k in range(K):
                rows.append(
                    {
                        "model": str(model),
                        "month_in_quarter": int(miq),
                        "factor": f"F{k + 1}",
                        "factor_index": int(k + 1),
                        "inclusion_rate": float(inc[k]),
                        "n_draws_total": int(z_cat.shape[0]),
                        "n_vintages": int(n_vintages),
                    }
                )

    if not rows:
        return pd.DataFrame([{"note": "No factor inclusion distributions could be computed for *_fs models."}])

    out = pd.DataFrame(rows).sort_values(["model", "month_in_quarter", "factor_index"]).reset_index(drop=True)
    return out


def factor_inclusion_probabilities_table(packs: List[PosteriorPack]) -> pd.DataFrame:
    """
    Average posterior inclusion probability over vintages.
    Wide table: one row per factor, columns are fs-models.
    """
    long_df = factor_selection_by_vintage(packs)
    if "note" in long_df.columns:
        return long_df

    wide = (
        long_df.groupby(["model", "factor", "factor_index"], as_index=False)["pip"]
        .mean()
        .pivot(index=["factor_index", "factor"], columns="model", values="pip")
        .reset_index()
        .sort_values("factor_index")
        .drop(columns=["factor_index"])
        .reset_index(drop=True)
    )
    return wide


def active_factors_by_vintage(packs: List[PosteriorPack]) -> pd.DataFrame:
    """
    Posterior distribution summary for number of active factors over time.
    """
    rows = []
    for pack in packs:
        if not str(pack.model).endswith("_fs"):
            continue

        arr = _load_selection_arrays(pack)
        if arr is None:
            continue

        r_draws = np.asarray(arr["r_draws"], dtype=float)
        r_draws = r_draws[np.isfinite(r_draws)]
        if r_draws.size == 0:
            continue

        rows.append(
            {
                "model": str(pack.model),
                "month": pd.to_datetime(pack.month),
                "subrun_tag": str(pack.tag),
                "mean_active_factors": float(np.mean(r_draws)),
                "median_active_factors": float(np.median(r_draws)),
                "p25_active_factors": float(np.quantile(r_draws, 0.25)),
                "p75_active_factors": float(np.quantile(r_draws, 0.75)),
            }
        )

    if not rows:
        return pd.DataFrame([{"note": "No r_draws found for factor-selection models."}])

    out = pd.DataFrame(rows).sort_values(["model", "month"]).reset_index(drop=True)
    return out


def active_factors_summary_table(packs: List[PosteriorPack]) -> pd.DataFrame:
    """
    Overall summary of active factors across vintages.
    """
    long_df = active_factors_by_vintage(packs)
    if "note" in long_df.columns:
        return long_df

    g = (
        long_df.groupby("model", as_index=False)
        .agg(
            n_vintages=("month", "count"),
            mean_active_factors=("mean_active_factors", "mean"),
            median_active_factors=("mean_active_factors", "median"),
            p25_active_factors=("mean_active_factors", lambda x: float(np.quantile(x, 0.25))),
            p75_active_factors=("mean_active_factors", lambda x: float(np.quantile(x, 0.75))),
        )
        .sort_values("model")
        .reset_index(drop=True)
    )
    return g

# =============================================================================
# Distribution of estimated number of factors by month-in-quarter
# =============================================================================


def factor_count_distribution_by_month(
    packs: List[PosteriorPack],
) -> pd.DataFrame:
    """
    For *_fs models, compute posterior probabilities P(r = k) by model and
    month_in_quarter, pooling across vintages.

    Output columns:
      model, month_in_quarter, r, prob, n_draws_total, n_vintages
    """
    rows = []

    for model in sorted({p.model for p in packs if str(p.model).endswith("_fs")}):
        model_packs = [p for p in packs if p.model == model]
        if not model_packs:
            continue

        for miq in [1, 2, 3]:
            month_packs = [p for p in model_packs if int(p.month_in_quarter) == miq]
            if not month_packs:
                continue

            r_all = []
            n_vintages = 0

            for pack in month_packs:
                arr = _load_selection_arrays(pack)
                if arr is None:
                    continue

                r_draws = np.asarray(arr["r_draws"], dtype=float)
                r_draws = r_draws[np.isfinite(r_draws)]
                if r_draws.size == 0:
                    continue

                r_all.append(r_draws.astype(int))
                n_vintages += 1

            if not r_all:
                continue

            r_cat = np.concatenate(r_all)
            vals, counts = np.unique(r_cat, return_counts=True)
            probs = counts / counts.sum()

            for r, p in zip(vals, probs):
                rows.append(
                    {
                        "model": str(model),
                        "month_in_quarter": int(miq),
                        "r": int(r),
                        "prob": float(p),
                        "n_draws_total": int(r_cat.size),
                        "n_vintages": int(n_vintages),
                    }
                )

    if not rows:
        return pd.DataFrame([{"note": "No factor-count distributions could be computed for *_fs models."}])

    out = pd.DataFrame(rows).sort_values(["model", "month_in_quarter", "r"]).reset_index(drop=True)
    return out


def fig_factor_count_distribution(
    factor_dist: pd.DataFrame,
    model: str,
    outdir: Path,
    max_r: Optional[int] = None,
) -> None:
    """
    3-panel bar chart:
      First Month Nowcast / Second Month Nowcast / Third Month Nowcast
    showing posterior probabilities of r for one *_fs model.
    """
    if factor_dist.empty or "note" in factor_dist.columns:
        return

    df = factor_dist[factor_dist["model"] == model].copy()
    if df.empty:
        return

    if max_r is None:
        max_r = int(df["r"].max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)

    month_titles = {
        1: "First Month Nowcast",
        2: "Second Month Nowcast",
        3: "Third Month Nowcast",
    }

    ymax = max(0.8, float(df["prob"].max()) * 1.15)

    for ax, miq in zip(axes, [1, 2, 3]):
        g = df[df["month_in_quarter"] == miq].copy()

        x = np.arange(1, max_r + 1)
        y = np.zeros_like(x, dtype=float)

        if not g.empty:
            rr = g["r"].to_numpy(dtype=int)
            pp = g["prob"].to_numpy(dtype=float)
            for r, p in zip(rr, pp):
                if 1 <= r <= max_r:
                    y[r - 1] = p

        ax.bar(x, y, edgecolor="black", alpha=0.5)
        for xi, yi in zip(x, y):
            if yi > 0:
                ax.text(xi, yi + 0.015, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(month_titles[miq])
        ax.set_xlabel("Number of Factors")
        ax.set_ylabel("Proportion")
        ax.set_xticks(x)
        ax.set_ylim(0, ymax)
        ax.grid(False)

    fig.suptitle(f"Distribution of estimated number of latent factors — {model}", y=1.03)
    _savefig(fig, outdir / f"factor_count_distribution__{model}")


def fig_factor_inclusion_distribution_by_month(
    factor_dist: pd.DataFrame,
    model: str,
    outdir: Path,
) -> None:
    """
    3-panel bar chart:
      First Month Nowcast / Second Month Nowcast / Third Month Nowcast
    showing average inclusion rates for factors F1, F2, ..., FK.
    """
    if factor_dist.empty or "note" in factor_dist.columns:
        return

    df = factor_dist[factor_dist["model"] == model].copy()
    if df.empty:
        return

    max_k = int(df["factor_index"].max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)

    month_titles = {
        1: "First Month Nowcast",
        2: "Second Month Nowcast",
        3: "Third Month Nowcast",
    }

    for ax, miq in zip(axes, [1, 2, 3]):
        g = df[df["month_in_quarter"] == miq].copy()

        x = np.arange(1, max_k + 1)
        y = np.zeros_like(x, dtype=float)

        if not g.empty:
            idx = g["factor_index"].to_numpy(dtype=int)
            vals = g["inclusion_rate"].to_numpy(dtype=float)
            for k, v in zip(idx, vals):
                if 1 <= k <= max_k:
                    y[k - 1] = v

        ax.bar(x, y, edgecolor="black", alpha=0.5)
        for xi, yi in zip(x, y):
            if yi > 0:
                ax.text(xi, yi + 0.015, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(month_titles[miq])
        ax.set_xlabel("Factor")
        ax.set_ylabel("Inclusion rate")
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{k}" for k in x])
        ax.set_ylim(0, 1)
        ax.grid(False)

    fig.suptitle(f"Factor inclusion by nowcasting month — {model}", y=1.03)
    _savefig(fig, outdir / f"factor_inclusion_distribution_by_month__{model}")



def top_series_per_factor_table(packs: List[PosteriorPack], top_n: int = 5) -> pd.DataFrame:
    """
    Economic interpretation table:
    top loading series per factor, averaged over vintages, for each fs-model.
    """
    rows = []

    fs_models = _fs_models_from_packs(packs)
    for model in fs_models:
        model_packs = [p for p in packs if p.model == model]
        if not model_packs:
            continue

        lambda_list = []
        series_ref = None

        for pack in model_packs:
            arr = _load_selection_arrays(pack)
            if arr is None:
                continue

            L = np.asarray(arr["Lambda_mean"], dtype=float)
            s = np.asarray(arr["series"], dtype=object)

            if series_ref is None:
                series_ref = s
            elif len(series_ref) != len(s) or not np.all(series_ref == s):
                continue

            lambda_list.append(np.abs(L))

        if not lambda_list or series_ref is None:
            continue

        Lbar = np.mean(np.stack(lambda_list, axis=0), axis=0)  # (N, K)
        N, K = Lbar.shape

        for k in range(K):
            idx = np.argsort(-Lbar[:, k])[:top_n]
            top_series = [str(series_ref[i]) for i in idx]
            top_loadings = [float(Lbar[i, k]) for i in idx]

            rows.append(
                {
                    "model": model,
                    "factor": f"F{k + 1}",
                    "top_series": " | ".join(top_series),
                    "top_abs_loadings": " | ".join(f"{x:.3f}" for x in top_loadings),
                }
            )

    if not rows:
        return pd.DataFrame([{"note": "Could not compute top-series table for factors."}])

    return pd.DataFrame(rows).sort_values(["model", "factor"]).reset_index(drop=True)


def fig_factor_inclusion_over_time(
    factor_long: pd.DataFrame,
    model: str,
    outdir: Path,
) -> None:
    if factor_long.empty or "note" in factor_long.columns:
        return

    df = factor_long[factor_long["model"] == model].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for factor in sorted(df["factor"].unique(), key=lambda x: int(str(x).replace("F", ""))):
        g = df[df["factor"] == factor].sort_values("month")
        ax.plot(g["month"], g["pip"], marker="o", label=factor)

    ax.set_title(f"Factor inclusion over time — {model}")
    ax.set_xlabel("Vintage month")
    ax.set_ylabel("Posterior inclusion probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, ncol=4)
    ax.grid(True, alpha=0.3)
    _savefig(fig, outdir / f"factor_inclusion_over_time__{model}")


def fig_active_factors_over_time(
    active_long: pd.DataFrame,
    outdir: Path,
) -> None:
    if active_long.empty or "note" in active_long.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    for model in sorted(active_long["model"].unique()):
        g = active_long[active_long["model"] == model].sort_values("month")
        ax.plot(g["month"], g["mean_active_factors"], marker="o", label=model)

    ax.set_title("Mean number of active factors over time")
    ax.set_xlabel("Vintage month")
    ax.set_ylabel("Mean active factors")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _savefig(fig, outdir / "active_factors_over_time")


def fig_factor_inclusion_heatmap(
    factor_long: pd.DataFrame,
    model: str,
    outdir: Path,
) -> None:
    if factor_long.empty or "note" in factor_long.columns:
        return

    df = factor_long[factor_long["model"] == model].copy()
    if df.empty:
        return

    pivot = (
        df.pivot_table(index="month", columns="factor", values="pip", aggfunc="mean")
        .sort_index()
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.22 * len(pivot))))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        interpolation="nearest",
        norm=Normalize(vmin=0.0, vmax=1.0),
        cmap="Blues",
    )
    ax.set_title(f"Factor inclusion heatmap — {model}")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Vintage month")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([pd.Timestamp(x).strftime("%Y-%m") for x in pivot.index])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Posterior inclusion probability")

    _savefig(fig, outdir / f"factor_inclusion_heatmap__{model}")

def fig_ar1_vs_actual_gdp(
    actual_by_q: Optional[pd.Series],
    outdir: Path,
    ylab: str,
    display_units: str,
    base_units: str = "annualized",
) -> Optional[dict]:
    """
    Plot realized GDP together with the AR(1) one-step-ahead forecast
    constructed exactly as in the benchmark code.

    Returns the estimated AR(1) coefficients so you can inspect them:
        {"alpha": ..., "phi": ..., "sigma2": ...}
    """
    if actual_by_q is None or len(actual_by_q) == 0:
        return None

    y = _build_actual_series_from_lookup(
        actual_by_q=actual_by_q,
        display_units=display_units,
        base_units=base_units,
    )
    if y.empty or y.size < 10:
        return None

    ar1 = _estimate_ar1_normal(y)
    if ar1 is None:
        return None

    # Compute standard error, t-stat, and p-value for phi from the same AR(1) regression
    y_t = y.iloc[1:].to_numpy(dtype=float)
    y_lag = y.iloc[:-1].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(y_lag), y_lag])
    beta_hat = np.array([ar1["alpha"], ar1["phi"]], dtype=float)

    resid = y_t - X @ beta_hat
    n, k = X.shape
    if n > k:
        sigma2_ols = float((resid @ resid) / (n - k))
        xtx_inv = np.linalg.inv(X.T @ X)
        vcov = sigma2_ols * xtx_inv
        se_phi = float(np.sqrt(vcov[1, 1])) if vcov[1, 1] >= 0 else np.nan
        t_phi = float(ar1["phi"] / se_phi) if np.isfinite(se_phi) and se_phi > 0 else np.nan
        p_phi = _norm_pval_from_z(t_phi) if np.isfinite(t_phi) else np.nan
    else:
        se_phi = np.nan
        t_phi = np.nan
        p_phi = np.nan

    rows = []
    for tq in y.index[1:]:
        mu, sigma2 = _ar1_one_step_params(ar1, str(tq))
        if not np.isfinite(mu):
            continue

        rows.append(
            {
                "target_quarter": str(tq),
                "tq_end": tq.to_timestamp(how="end"),
                "actual": float(y.loc[tq]),
                "ar1_forecast": float(mu),
                "lagged_actual": float(y.loc[tq - 1]),
                "sigma": float(np.sqrt(sigma2)) if np.isfinite(sigma2) and sigma2 >= 0 else np.nan,
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("tq_end").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        df["tq_end"],
        df["actual"],
        "k--o",
        label="Actual GDP",
        markersize=5,
        linewidth=1.8,
    )
    ax.plot(
        df["tq_end"],
        df["ar1_forecast"],
        marker="o",
        linewidth=2.0,
        label="AR(1) one-step forecast",
    )

    ax.set_xlabel("Target quarter")
    ax.set_ylabel(ylab)
    ax.set_title("Actual GDP vs AR(1) benchmark")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    subtitle = (
        f"AR(1): alpha = {ar1['alpha']:.6f}, "
        f"phi = {ar1['phi']:.6f}, "
        f"p(phi) = {p_phi:.6f}, "
        f"sigma = {np.sqrt(ar1['sigma2']):.6f}"
    )
    ax.text(
        0.01,
        0.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    _savefig(fig, outdir / "actual_vs_ar1_gdp")

    return {
        "alpha": float(ar1["alpha"]),
        "phi": float(ar1["phi"]),
        "sigma2": float(ar1["sigma2"]),
        "se_phi": float(se_phi) if np.isfinite(se_phi) else np.nan,
        "t_phi": float(t_phi) if np.isfinite(t_phi) else np.nan,
        "p_phi": float(p_phi) if np.isfinite(p_phi) else np.nan,
    }
# =============================================================================
# Main
# =============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Path to outputs/<run_id>")
    ap.add_argument(
        "--actual_vintage",
        type=str,
        default="data/NL/2025-11-14.xlsx",
        help="Relative path to most complete vintage used as actual GDP",
    )
    ap.add_argument(
        "--spec_file",
        type=str,
        default="data/Spec_NL.xlsx",
        help="Relative path to Spec file",
    )
    ap.add_argument("--gdp_series_id", type=str, default="Real GDP")
    ap.add_argument("--sample_start", type=str, default="1985-01-01")
    ap.add_argument(
        "--display_units",
        type=str,
        default="annualized",
        choices=["qoq", "annualized"],
        help="Units for plots/tables.",
    )
    ap.add_argument("--hac_lag", type=int, default=1, help="HAC lag for DM tests.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    project_root = Path.cwd()

    fig_dir = run_dir / "figures"
    tab_dir = run_dir / "tables"
    _ensure_dir(fig_dir)
    _ensure_dir(tab_dir)

    now, packs, _meta = scan_run(run_dir)

    actual_by_q = None
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
        print(f"[plottablebothselection] WARNING: could not attach actual GDP / RW benchmark: {e}")
        now = now.copy()
        now["actual"] = np.nan
        now["rw_nowcast"] = np.nan

    ylab = "Real GDP growth (QoQ, %)" if args.display_units == "qoq" else "Real GDP growth (annualized, %)"
    now = _apply_display_units(now, display_units=args.display_units, base_units="annualized")
    now = add_release_step_columns(now)
    now = add_release_date(now)

    (run_dir / "nowcasts").mkdir(parents=True, exist_ok=True)
    now.to_csv(run_dir / "nowcasts" / "nowcasts_with_actual_rw.csv", index=False)

    models_all = sorted(now["model"].dropna().astype(str).unique())
    dfm_models = [m for m in models_all if m not in ("RW", "ar1")]
    preferred_order = ["bay", "bay_fs", "bay_sv", "bay_sv_fs"]
    dfm_models = sorted(dfm_models, key=lambda x: preferred_order.index(x) if x in preferred_order else 999)

    # ----------------------------
    # Point evaluation: final within-quarter
    # ----------------------------
    now_point = _add_ar1_point_rows(
        _ensure_rw_rows_for_final(now),
        actual_by_q=actual_by_q,
        display_units=args.display_units,
    )
    pt_final = table_point_errors_final_within_quarter(now_point)
    pt_final.to_csv(tab_dir / "point_errors_final_within_quarter.csv", index=False)

    point_comp_models = [m for m in pt_final["model"].tolist() if m in ["RW", "ar1"] + dfm_models]
    dm_point_list = []
    for A, B in combinations(point_comp_models, 2):
        dm_point_list.append(dm_test_point_final_within_quarter(now_point, A, B, loss="ae", hac_lag=args.hac_lag))
        dm_point_list.append(dm_test_point_final_within_quarter(now_point, A, B, loss="se", hac_lag=args.hac_lag))

    dm_point = (
        pd.concat(dm_point_list, ignore_index=True)
        if dm_point_list
        else pd.DataFrame([{"note": "Point DM tests skipped."}])
    )
    dm_point.to_csv(tab_dir / "dm_tests_point_final_within_quarter.csv", index=False)

    # ----------------------------
    # Point evaluation: by month-in-quarter x release
    # ----------------------------
    pt_grid = point_errors_by_month_release(now)
    pt_grid.to_csv(tab_dir / "point_errors_by_month_release.csv", index=False)

    rel_mae_vs_rw = relative_score_by_month_release_vs_benchmark(
        pt_grid,
        score_col="MAE",
        benchmark_model="RW",
    )
    rel_mae_vs_rw.to_csv(tab_dir / "relative_mae_by_month_release_vs_rw.csv", index=False)

    rel_rmsfe_vs_rw = relative_score_by_month_release_vs_benchmark(
        pt_grid,
        score_col="RMSFE",
        benchmark_model="RW",
    )
    rel_rmsfe_vs_rw.to_csv(tab_dir / "relative_rmsfe_by_month_release_vs_rw.csv", index=False)

    for model in dfm_models:
        g_mae = relative_grid_for_model(
            rel_mae_vs_rw,
            model=model,
            rel_col="relative_mae_pct",
        )
        g_mae.to_csv(tab_dir / f"relative_mae_grid__{model}__vs_rw.csv", index=False)

        g_rmsfe = relative_grid_for_model(
            rel_rmsfe_vs_rw,
            model=model,
            rel_col="relative_rmsfe_pct",
        )
        g_rmsfe.to_csv(tab_dir / f"relative_rmsfe_grid__{model}__vs_rw.csv", index=False)

    # ----------------------------
    # Density evaluation
    # ----------------------------
    dens_final = density_scores_final_within_quarter(
        now,
        packs,
        display_units=args.display_units,
        actual_by_q=actual_by_q,
    )
    dens_final.to_csv(tab_dir / "density_scores_final_within_quarter.csv", index=False)

    dens_summary = summarize_density_scores(dens_final)
    dens_summary.to_csv(tab_dir / "density_summary_final_within_quarter.csv", index=False)

    # ----------------------------
    # Cumulative log predictive score
    # ----------------------------
    cum_logscore_vs_ar1 = cumulative_logscore_differences(
        dens_final,
        benchmark_model="ar1",
    )
    cum_logscore_vs_ar1.to_csv(tab_dir / "cumulative_logscore_vs_ar1.csv", index=False)

    dm_out_list = []
    if "note" not in dens_final.columns:
        density_models = sorted(
            dens_final["model"].dropna().astype(str).unique(),
            key=lambda x: (["ar1"] + preferred_order).index(x) if x in ["ar1"] + preferred_order else 999,
        )
        for A, B in combinations(density_models, 2):
            AA = dens_final[dens_final["model"] == A].copy()
            BB = dens_final[dens_final["model"] == B].copy()
            dm_crps = dm_test(
                AA, BB, key_cols=["target_quarter"], score_col="crps", hac_lag=args.hac_lag
            ).assign(model_A=A, model_B=B)
            dm_ls = dm_test(
                AA, BB, key_cols=["target_quarter"], score_col="logscore", hac_lag=args.hac_lag
            ).assign(model_A=A, model_B=B)
            dm_out_list.extend([dm_crps, dm_ls])

    if dm_out_list:
        dm_density = pd.concat(dm_out_list, ignore_index=True)
    else:
        dm_density = pd.DataFrame([{"note": "Density DM tests skipped."}])

    dm_density.to_csv(tab_dir / "dm_tests_density_final_within_quarter.csv", index=False)

    # ----------------------------
    # Factor-selection analysis
    # ----------------------------
    factor_long = factor_selection_by_vintage(packs)
    factor_long.to_csv(tab_dir / "factor_inclusion_by_vintage.csv", index=False)

    factor_avg = factor_inclusion_probabilities_table(packs)
    factor_avg.to_csv(tab_dir / "factor_inclusion_probabilities.csv", index=False)

    factor_dist_month = factor_inclusion_distribution_by_month(packs)
    factor_dist_month.to_csv(tab_dir / "factor_inclusion_distribution_by_month.csv", index=False)


    active_long = active_factors_by_vintage(packs)
    active_long.to_csv(tab_dir / "active_factors_by_vintage.csv", index=False)

    active_summary = active_factors_summary_table(packs)
    active_summary.to_csv(tab_dir / "active_factors_summary.csv", index=False)

    top_series = top_series_per_factor_table(packs, top_n=5)
    top_series.to_csv(tab_dir / "top_series_per_factor.csv", index=False)


    factor_dist = factor_count_distribution_by_month(packs)
    factor_dist.to_csv(tab_dir / "factor_count_distribution_by_month.csv", index=False)

    # ----------------------------
    # Figures
    # ----------------------------
    if not now.empty:
        fig_actual_vs_models_final_within_quarter(now, fig_dir, ylab=ylab, model_order=dfm_models)

        current_fan_dfs = [
            _fan_chart_data_current_target(now, packs, model, display_units=args.display_units)
            for model in dfm_models
        ]
        final_fan_dfs = [
            _fan_chart_data_final_within_quarter(now, packs, model, display_units=args.display_units)
            for model in dfm_models
        ]
        common_ylim_current = _compute_common_ylim_from_fan_data(current_fan_dfs)
        common_ylim_final = _compute_common_ylim_from_fan_data(final_fan_dfs)

        for model in dfm_models:
            fig_fan_chart_current_target(
                now,
                packs,
                model,
                fig_dir,
                ylab=ylab,
                display_units=args.display_units,
                ylim=common_ylim_current,
            )
            fig_fan_chart_final_within_quarter(
                now,
                packs,
                model,
                fig_dir,
                ylab=ylab,
                display_units=args.display_units,
                ylim=common_ylim_final,
            )

        # cumulative log predictive score relative to AR(1)
        if "note" not in cum_logscore_vs_ar1.columns:
            fig_cumulative_logscore(
                cum_logscore_vs_ar1,
                fig_dir,
                benchmark_model="ar1",
                model_order=[m for m in dfm_models if m in cum_logscore_vs_ar1["model"].unique()],
            )

        fs_models = [m for m in dfm_models if m.endswith("_fs")]

        if "note" not in factor_long.columns:
            for model in fs_models:
                fig_factor_inclusion_over_time(factor_long, model, fig_dir)
                fig_factor_inclusion_heatmap(factor_long, model, fig_dir)

        if "note" not in active_long.columns:
            fig_active_factors_over_time(active_long, fig_dir)

        if "note" not in factor_dist_month.columns:
            for model in fs_models:
                fig_factor_inclusion_distribution_by_month(
                    factor_dist_month,
                    model=model,
                    outdir=fig_dir,
                )


        # factor-count distribution plots for *_fs models
        if "note" not in factor_dist.columns:
            fs_models = [m for m in dfm_models if m.endswith("_fs")]
            for model in fs_models:
                fig_factor_count_distribution(
                    factor_dist,
                    model=model,
                    outdir=fig_dir,
                )

        if not now.empty:
            fig_ar1_vs_actual_gdp(
                actual_by_q=actual_by_q,
                outdir=fig_dir,
                ylab=ylab,
                display_units=args.display_units,
            )
        ar1_coef = fig_ar1_vs_actual_gdp(
            actual_by_q=actual_by_q,
            outdir=fig_dir,
            ylab=ylab,
            display_units=args.display_units,
        )

        print(f"alpha   = {ar1_coef['alpha']}")
        print(f"phi     = {ar1_coef['phi']}")
        print(f"sigma2  = {ar1_coef['sigma2']}")
        print(f"p_phi   = {ar1_coef['p_phi']}")
        fig_actual_vs_models_final_within_quarter(now, fig_dir, ylab=ylab, model_order=dfm_models)

    # ----------------------------
    # Index
    # ----------------------------
    index_lines = []
    index_lines.append(f"Run: {run_dir.name}")
    index_lines.append("")
    index_lines.append("Models found:")
    for m in dfm_models:
        index_lines.append(f"  - {m}")
    index_lines.append("")
    index_lines.append("Figures (PNG):")
    for p in sorted(fig_dir.glob("*.png")):
        index_lines.append(f"  - {p.name}")
    index_lines.append("")
    index_lines.append("Tables (CSV):")
    for p in sorted(tab_dir.glob("*.csv")):
        index_lines.append(f"  - {p.name}")
    index_lines.append("")
    index_lines.append("Notes:")
    index_lines.append("  - Point benchmarks: RW and AR(1).")
    index_lines.append("  - Density benchmark: AR(1).")
    index_lines.append("  - Final within-quarter selection = last snapshot with asof_in_target==True and rel_step_in_target in [0..8].")
    index_lines.append("  - Month-release grid tables use all within-target rows with month_in_quarter in {1,2,3} and q in {1,2,3}.")
    index_lines.append("  - Relative grid formula = 100 * (score_model - score_RW) / score_RW, so negative values mean improvement relative to RW.")
    index_lines.append("  - Fan charts are created for all DFM models found in the run.")
    index_lines.append("  - Factor-selection tables are created for *_fs models only.")
    index_lines.append("  - factor_inclusion_probabilities.csv reports average posterior inclusion probabilities.")
    index_lines.append("  - active_factors_summary.csv reports how many factors are active on average.")
    index_lines.append("  - top_series_per_factor.csv helps interpret the economic meaning of each factor.")
    index_lines.append("  - No period analysis, PIT histograms, or loss-difference plots in this cleaned script.")
    index_lines.append("  - cumulative_logscore_vs_ar1.csv reports quarter-by-quarter and cumulative log predictive score differences relative to AR(1).")
    index_lines.append("  - factor_count_distribution_by_month.csv reports pooled posterior probabilities for the number of active factors by nowcasting month.")

    (run_dir / "ARTIFACTS_INDEX.txt").write_text("\n".join(index_lines))

    print(f"Artifacts written to:\n  {fig_dir}\n  {tab_dir}\nIndex:\n  {run_dir/'ARTIFACTS_INDEX.txt'}")


if __name__ == "__main__":
    main()