#!/usr/bin/env python3
"""
make_densityplots.py (extended: point DM + period/regime analysis)

Adds:
1) DM tests for POINT nowcasts (AE and SE losses) on FINAL within-quarter snapshots
2) Period/regime analysis for BOTH point and density nowcasts:
   - period tables (point + density)
   - DM tests by period (point + density)
   - "Does SV help more in crisis?" tests via loss-differential regressions
   - plots:
        * loss differential time series with shaded crisis window(s)
        * bar charts of period means (point + density)

Design choices:
- Statistical testing uses FINAL within-quarter (1 obs per target quarter) to keep alignment clean.
- Periods default to:
    pre_covid: 2015Q1–2019Q4
    covid:     2020Q1–2021Q4
    post:      2022Q1–2025Q3
  (You can override via --periods_json).

Outputs (new CSVs):
- dm_tests_point_final_within_quarter.csv
- point_errors_by_period_final_within_quarter.csv
- dm_tests_point_by_period_final_within_quarter.csv
- density_summary_by_period_final_within_quarter.csv
- dm_tests_density_by_period_final_within_quarter.csv
- diff_in_diff_tests.csv

Outputs (new PNGs):
- lossdiff_point_se__bay_sv_minus_bay.png
- lossdiff_point_ae__bay_sv_minus_bay.png
- lossdiff_density_crps__bay_sv_minus_bay.png
- lossdiff_density_logscore__bay_sv_minus_bay.png
- period_bars_point_mae.png, period_bars_point_rmsfe.png
- period_bars_density_crps.png, period_bars_density_logscore.png

Other notes:
- Only PNG figures (no PDF).
- RW is used only in tables/tests, never plotted.

SCALE FIX (requested):
- Ensure bay and bay_sv fan charts use the SAME y-axis scale.
- The common scale is chosen as the union (min/max) across BOTH models for the same plot type.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zhangnowcast.data.data import build_zhang_data


# python3 scripts/completeresults.py --run_dir outputs/20260223_171049_18754b93
# =============================================================================
# Matplotlib style: clean and consistent
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
# Default evaluation periods (for OOS nowcast performance tables/tests)
# =============================================================================

DEFAULT_PERIODS = [
    dict(name="pre_covid", label="Pre-COVID (2015Q1–2019Q4)", start="2015Q1", end="2019Q4"),
    dict(name="covid", label="COVID (2020Q1–2021Q4)", start="2020Q1", end="2021Q4"),
    dict(name="post", label="Post-COVID (2022Q1–2025Q3)", start="2022Q1", end="2025Q3"),
]


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
    """PNG only."""
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=200)
    plt.close(fig)


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _coerce_month(x) -> pd.Timestamp:
    return pd.to_datetime(x).to_period("M").to_timestamp()


def _nice(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_=. " else "_" for c in str(s))


def _to_latex_table(df: pd.DataFrame, path: Path, caption: str = "", label: str = "") -> None:
    """Optional .tex export (kept, but harmless if you ignore .tex)."""
    if not _have_jinja2():
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


def _period_in_range(tq: str, start_q: str, end_q: str) -> bool:
    p = _period_str_to_periodQ(tq)
    s = _period_str_to_periodQ(start_q)
    e = _period_str_to_periodQ(end_q)
    if p is None or s is None or e is None:
        return False
    return (p >= s) and (p <= e)


def _assign_period_label(target_quarter: str, periods: List[dict]) -> str:
    tq = str(target_quarter)
    for pr in periods:
        if _period_in_range(tq, pr["start"], pr["end"]):
            return pr.get("name", "")
    return "other"


def _period_order_map(periods: List[dict]) -> Dict[str, int]:
    return {p["name"]: i for i, p in enumerate(periods)}


def _padded_ylim(ymin: float, ymax: float, pad_frac: float = 0.05) -> Optional[Tuple[float, float]]:
    """Return a slightly padded ylim for nicer plots, preserving union scale."""
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
# Actual GDP + RW benchmark (RW used ONLY for tests/tables)
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

    nowcast_targets: np.ndarray  # (J,)
    nowcast_draws_matrix: np.ndarray  # (S,J)
    nowcast_draws_legacy: np.ndarray  # (S,) or empty


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
        nowcast_targets=np.asarray(nowcast_targets, dtype=object)
        if nowcast_targets is not None
        else np.asarray([], dtype=object),
        nowcast_draws_matrix=np.asarray(nowcast_draws_matrix, dtype=float)
        if nowcast_draws_matrix is not None
        else np.asarray([], dtype=float),
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
    df["rel_step_label"] = df["rel_step_in_target"].apply(
        lambda x: _rel_step_label(int(x)) if np.isfinite(x) else ""
    )
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
        fallback = pd.to_datetime(df["month"]) + pd.to_timedelta(
            (df["q"].astype(int) - 1) * 10, unit="D"
        )
        rd = rd.fillna(fallback)

    df["release_date"] = rd
    return df


def add_period_columns(now: pd.DataFrame, periods: List[dict]) -> pd.DataFrame:
    df = now.copy()
    if "target_quarter" not in df.columns:
        df["period"] = "other"
        return df
    df["period"] = df["target_quarter"].astype(str).apply(lambda x: _assign_period_label(x, periods))
    return df


# =============================================================================
# Row selection: FINAL within-quarter (one per target_quarter)
# =============================================================================


def select_final_within_quarter(now: pd.DataFrame) -> pd.DataFrame:
    """
    Pick one row per (model, target_quarter): the LAST within-quarter snapshot.
    """
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
# Tables: point forecast errors vs RW
# =============================================================================


def table_point_errors_final_within_quarter(now: pd.DataFrame) -> pd.DataFrame:
    """
    Point errors using FINAL within-quarter snapshot per target_quarter.
    Outputs MAE/RMSFE and RW baselines + MANE.
    """
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

    if "rw_nowcast_disp" in df.columns and df["rw_nowcast_disp"].notna().any():
        df["rw_err"] = df["rw_nowcast_disp"] - df["actual_disp"]
        df["rw_abs_err"] = df["rw_err"].abs()
        df["rw_sq_err"] = df["rw_err"] ** 2
    else:
        df["rw_abs_err"] = np.nan
        df["rw_sq_err"] = np.nan

    out = []
    for model, g in df.groupby("model"):
        mae = float(g["abs_err"].mean())
        rmsfe = float(np.sqrt(g["sq_err"].mean()))
        rw_mae = float(g["rw_abs_err"].mean()) if g["rw_abs_err"].notna().any() else float("nan")
        rw_rmsfe = float(np.sqrt(g["rw_sq_err"].mean())) if g["rw_sq_err"].notna().any() else float("nan")
        out.append(
            dict(
                model=str(model),
                n=int(len(g)),
                MAE=mae,
                RMSFE=rmsfe,
                RW_MAE=rw_mae,
                RW_RMSFE=rw_rmsfe,
                MANE=(mae / rw_mae) if (np.isfinite(rw_mae) and rw_mae > 0) else float("nan"),
            )
        )

    # Add RW row explicitly
    if df["rw_abs_err"].notna().any():
        g_rw = df[df["rw_abs_err"].notna()].copy()
        mae_rw = float(g_rw["rw_abs_err"].mean())
        rmsfe_rw = float(np.sqrt(g_rw["rw_sq_err"].mean()))
        out.append(
            dict(
                model="RW",
                n=int(len(g_rw)),
                MAE=mae_rw,
                RMSFE=rmsfe_rw,
                RW_MAE=np.nan,
                RW_RMSFE=np.nan,
                MANE=np.nan,
            )
        )

    return pd.DataFrame(out).sort_values("model").reset_index(drop=True)


def table_point_errors_by_period_final_within_quarter(now: pd.DataFrame, periods: List[dict]) -> pd.DataFrame:
    """
    Period breakdown of point errors using FINAL within-quarter snapshot per target_quarter.
    """
    df = select_final_within_quarter(now)
    if df.empty or "actual_disp" not in df.columns:
        return pd.DataFrame([{"note": "No final-within-quarter rows available for point period table."}])

    df = df[df["actual_disp"].notna()].copy()
    if df.empty:
        return pd.DataFrame([{"note": "No rows with actual_disp for point period table."}])

    if "period" not in df.columns:
        df["period"] = df["target_quarter"].astype(str).apply(lambda x: _assign_period_label(x, periods))

    df["err"] = df["nowcast_mean_disp"] - df["actual_disp"]
    df["abs_err"] = df["err"].abs()
    df["sq_err"] = df["err"] ** 2

    # RW
    if "rw_nowcast_disp" in df.columns and df["rw_nowcast_disp"].notna().any():
        df["rw_err"] = df["rw_nowcast_disp"] - df["actual_disp"]
        df["rw_abs_err"] = df["rw_err"].abs()
        df["rw_sq_err"] = df["rw_err"] ** 2
    else:
        df["rw_abs_err"] = np.nan
        df["rw_sq_err"] = np.nan

    rows = []
    for (period, model), g in df.groupby(["period", "model"]):
        rows.append(
            dict(
                period=str(period),
                model=str(model),
                n=int(len(g)),
                MAE=float(g["abs_err"].mean()),
                RMSFE=float(np.sqrt(g["sq_err"].mean())),
            )
        )

    # Add RW rows per period (if possible)
    if df["rw_abs_err"].notna().any():
        for period, g in df.groupby(["period"]):
            gg = g[g["rw_abs_err"].notna()]
            if gg.empty:
                continue
            rows.append(
                dict(
                    period=str(period),
                    model="RW",
                    n=int(len(gg)),
                    MAE=float(gg["rw_abs_err"].mean()),
                    RMSFE=float(np.sqrt(gg["rw_sq_err"].mean())),
                )
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame([{"note": "Point period table produced no rows."}])

    ord_map = _period_order_map(periods)
    out["period_order"] = out["period"].map(lambda x: ord_map.get(x, 999))
    out = out.sort_values(["period_order", "model"]).drop(columns=["period_order"])
    return out.reset_index(drop=True)


# =============================================================================
# AR(1) normal density benchmark (one forecast per target quarter)
# =============================================================================


def _build_actual_series_from_now(now: pd.DataFrame) -> pd.Series:
    """
    Build a time-ordered series of realized GDP (display units) by target_quarter.
    Uses actual_disp from select_final_within_quarter(now).
    """
    df = select_final_within_quarter(now)
    if df.empty or "actual_disp" not in df.columns:
        return pd.Series(dtype=float)

    df = df[df["actual_disp"].notna()].copy()
    if df.empty:
        return pd.Series(dtype=float)

    df["tq_period"] = pd.PeriodIndex(df["target_quarter"].astype(str), freq="Q")
    df = df.sort_values("tq_period")
    s = pd.Series(df["actual_disp"].to_numpy(dtype=float), index=df["tq_period"])
    s = s[~s.index.duplicated(keep="last")]
    return s


def _estimate_ar1_normal(y: pd.Series) -> Optional[dict]:
    """
    Estimate AR(1) with intercept on a quarterly series y_t (display units):

        y_t = alpha + phi * y_{t-1} + eps_t,   eps_t ~ N(0, sigma2)

    Returns dict with alpha, phi, sigma2, and the index of y (PeriodIndex).
    """
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
    """
    For target_quarter tq (e.g. '2020Q1'), return (mu_t, sigma2) for the AR(1)
    one-step-ahead predictive density, using y_{t-1} from the realized series.
    """
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


def _ar1_draws_for_target(ar1: dict, tq: str, S: int = 5000) -> Optional[np.ndarray]:
    """
    Generate S draws from the AR(1) normal predictive density for target_quarter tq.
    """
    mu, sigma2 = _ar1_one_step_params(ar1, tq)
    if not np.isfinite(mu) or not np.isfinite(sigma2) or sigma2 <= 0:
        return None
    return np.random.default_rng(789).normal(loc=mu, scale=np.sqrt(sigma2), size=S)


# =============================================================================
# Density scoring
# =============================================================================


def crps_from_draws(draws: np.ndarray, y: float) -> float:
    """
    CRPS = E|X-y| - 0.5 E|X-X'|
    O(S log S) formula for E|X-X'| using sorting.
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
    Return predictive draws for this row's target_quarter from pack, transformed to display_units.
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
    Density scores for the true nowcasting task: ALL within-quarter snapshots (asof_in_target==True).
    """
    df = now[(now["asof_in_target"] == True) & (now["actual_disp"].notna())].copy()  # noqa: E712
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

        out.append(
            dict(
                model=model,
                subrun_tag=tag,
                release_date=pd.to_datetime(r.get("release_date", pd.NaT)),
                month=pd.to_datetime(r.get("month", pd.NaT)),
                q=int(r.get("q", -1)),
                target_quarter=str(r["target_quarter"]),
                rel_step_in_target=float(r.get("rel_step_in_target", np.nan)),
                rel_step_label=str(r.get("rel_step_label", "")),
                crps=crps_from_draws(draws, y),
                logscore=hist_logscore(draws, y),
                pit=pit_from_draws(draws, y, rng=rng),
                cover90=float(lo90 <= y <= hi90),
                width90=float(hi90 - lo90),
                draws_n=int(draws.size),
            )
        )

    if not out:
        return pd.DataFrame([{"note": "No density scores computed (missing draws for rows)."}])
    return pd.DataFrame(out)


def density_scores_final_within_quarter(
    now: pd.DataFrame, packs: List[PosteriorPack], display_units: str
) -> pd.DataFrame:
    """
    Density scores for FINAL within-quarter snapshot: one per (model, target_quarter).
    Includes:
      - DFM models based on posterior draws
      - AR(1) normal benchmark ("ar1") based on realized actual_disp
    """
    df = select_final_within_quarter(now)
    if df.empty:
        return pd.DataFrame([{"note": "No final-within-quarter rows available for density scoring."}])
    df = df[df["actual_disp"].notna()].copy()
    if df.empty:
        return pd.DataFrame([{"note": "No final-within-quarter rows with non-NaN actual_disp."}])

    df = df.sort_values(["model", "target_quarter", "month", "q"]).copy()
    pidx = _pack_index(packs)
    rng = np.random.default_rng(456)

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
                pit=pit_from_draws(draws, y, rng=rng),
                cover90=float(lo90 <= y <= hi90),
                width90=float(hi90 - lo90),
                draws_n=int(draws.size),
            )
        )

    # AR(1) benchmark part
    y_series = _build_actual_series_from_now(now)
    ar1 = _estimate_ar1_normal(y_series)
    if ar1 is not None:
        df_ar = select_final_within_quarter(now)
        df_ar = df_ar[df_ar["actual_disp"].notna()].copy()
        df_ar = df_ar.sort_values(["target_quarter"]).copy()
        rng_ar = np.random.default_rng(987)

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
                    pit=pit_from_draws(draws, y, rng=rng_ar),
                    cover90=float(lo90 <= y <= hi90),
                    width90=float(hi90 - lo90),
                    draws_n=int(draws.size),
                )
            )

    if not out:
        return pd.DataFrame([{"note": "No final-within-quarter density scores computed (missing draws)."}])
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


def summarize_density_scores_by_period(scores: pd.DataFrame, periods: List[dict]) -> pd.DataFrame:
    if "note" in scores.columns:
        return scores.copy()

    df = scores.copy()
    if "period" not in df.columns:
        df["period"] = df["target_quarter"].astype(str).apply(lambda x: _assign_period_label(x, periods))

    g = df.groupby(["period", "model"], as_index=False).agg(
        n=("crps", "count"),
        mean_crps=("crps", "mean"),
        mean_logscore=("logscore", "mean"),
        mean_cover90=("cover90", "mean"),
        mean_width90=("width90", "mean"),
        mean_pit=("pit", "mean"),
    )

    ord_map = _period_order_map(periods)
    g["period_order"] = g["period"].map(lambda x: ord_map.get(x, 999))
    g = g.sort_values(["period_order", "model"]).drop(columns=["period_order"])
    return g.reset_index(drop=True)


# =============================================================================
# DM test (Diebold–Mariano) for score differences
# =============================================================================


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


def dm_test(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    key_cols: List[str],
    score_col: str,
    hac_lag: int = 1,
) -> pd.DataFrame:
    """
    DM test on d_t = score_a - score_b aligned on key_cols.
    Lower is better for CRPS; higher is better for logscore.
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

    from math import erf, sqrt

    pval = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm_stat) / sqrt(2.0))))

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


def _ensure_rw_rows_for_final(now: pd.DataFrame) -> pd.DataFrame:
    """
    Append synthetic model='RW' rows at FINAL within-quarter selection level
    so we can treat RW like a model in DM tests.
    """
    base = now.copy()
    df_final = select_final_within_quarter(base)
    if df_final.empty or "rw_nowcast_disp" not in df_final.columns:
        return base
    rw = df_final[df_final["rw_nowcast_disp"].notna()].copy()
    if rw.empty:
        return base
    rw["model"] = "RW"
    rw["nowcast_mean_disp"] = rw["rw_nowcast_disp"]
    # keep needed columns; extra columns fine
    return pd.concat([base, rw], ignore_index=True)


def dm_test_point_final_within_quarter(
    now: pd.DataFrame,
    model_A: str,
    model_B: str,
    loss: str = "ae",  # "ae" or "se"
    hac_lag: int = 1,
) -> pd.DataFrame:
    """
    DM test for point nowcasts using FINAL within-quarter snapshots.
    d_t = L(e_A) - L(e_B), aligned by target_quarter.
    """
    df = _ensure_rw_rows_for_final(now)
    dff = select_final_within_quarter(df)
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

    from math import erf, sqrt

    pval = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm_stat) / sqrt(2.0))))

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


def dm_tests_point_by_period(
    now: pd.DataFrame, periods: List[dict], comparisons: List[Tuple[str, str]], hac_lag: int
) -> pd.DataFrame:
    """
    DM tests for point forecasts by period, for AE and SE losses.
    """
    df = add_period_columns(now, periods)
    out = []
    for pr in periods:
        pname = pr["name"]
        sub = df[df["period"] == pname].copy()
        if sub.empty:
            continue
        for A, B in comparisons:
            out.append(dm_test_point_final_within_quarter(sub, A, B, loss="ae", hac_lag=hac_lag).assign(period=pname))
            out.append(dm_test_point_final_within_quarter(sub, A, B, loss="se", hac_lag=hac_lag).assign(period=pname))
    if not out:
        return pd.DataFrame([{"note": "No point DM-by-period tests computed."}])
    ord_map = _period_order_map(periods)
    res = pd.concat(out, ignore_index=True)
    res["period_order"] = res["period"].map(lambda x: ord_map.get(x, 999))
    res = res.sort_values(["period_order", "comparison", "loss"]).drop(columns=["period_order"])
    return res.reset_index(drop=True)


def dm_tests_density_by_period(
    dens_final_wq: pd.DataFrame,
    periods: List[dict],
    comparisons: List[Tuple[str, str]],
    hac_lag: int,
) -> pd.DataFrame:
    """
    DM tests for density scores (CRPS and logscore) by period.
    """
    if "note" in dens_final_wq.columns:
        return dens_final_wq.copy()

    df = dens_final_wq.copy()
    if "period" not in df.columns:
        df["period"] = df["target_quarter"].astype(str).apply(lambda x: _assign_period_label(x, periods))

    out = []
    for pr in periods:
        pname = pr["name"]
        sub = df[df["period"] == pname].copy()
        if sub.empty:
            continue
        for A, B in comparisons:
            if (A in sub["model"].unique()) and (B in sub["model"].unique()):
                AA = sub[sub["model"] == A].copy()
                BB = sub[sub["model"] == B].copy()
                dm_crps = dm_test(AA, BB, key_cols=["target_quarter"], score_col="crps", hac_lag=hac_lag).assign(
                    model_A=A, model_B=B, period=pname
                )
                dm_ls = dm_test(AA, BB, key_cols=["target_quarter"], score_col="logscore", hac_lag=hac_lag).assign(
                    model_A=A, model_B=B, period=pname
                )
                out.extend([dm_crps, dm_ls])

    if not out:
        return pd.DataFrame([{"note": "No density DM-by-period tests computed."}])

    ord_map = _period_order_map(periods)
    res = pd.concat(out, ignore_index=True)
    res["period_order"] = res["period"].map(lambda x: ord_map.get(x, 999))
    res = res.sort_values(["period_order", "model_A", "model_B", "score"]).drop(columns=["period_order"])
    return res.reset_index(drop=True)


# =============================================================================
# "Does SV help more in crisis?" tests (difference-in-differences style)
# =============================================================================


def _ols_gamma_and_se(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Simple OLS for y = X b, return (b[1], se[1]) assuming X includes intercept and 1 regressor.
    Uses White (HC0) robust SE.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    if n < 8 or k != 2:
        return float("nan"), float("nan")

    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_hat

    XtX_inv = np.linalg.inv(X.T @ X)
    # HC0 robust: X' diag(e^2) X
    S = (X.T * (resid**2)) @ X
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return float(beta_hat[1]), float(se[1])


def _two_sided_pval_norm(z: float) -> float:
    from math import erf, sqrt

    return float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0)))))


def diff_in_diff_tests(
    point_final: pd.DataFrame,
    dens_final: pd.DataFrame,
    periods: List[dict],
    crisis_name: str = "covid",
) -> pd.DataFrame:
    """
    Create a small set of "interaction" tests using loss differentials:
      d_t = L_sv - L_bay  (negative => SV better for loss metrics)
      for logscore we use d_t = logscore_sv - logscore_bay (positive => SV better)

    Regression:
      d_t = alpha + gamma * 1[crisis] + u_t
    gamma tests whether SV advantage differs in crisis vs non-crisis.
    """
    out = []

    # --- Point: AE and SE differentials (sv - bay) ---
    if point_final is not None and not point_final.empty:
        pf = point_final.copy()
        if "period" not in pf.columns:
            pf["period"] = pf["target_quarter"].astype(str).apply(lambda x: _assign_period_label(x, periods))
        # align bay and bay_sv
        A = pf[pf["model"] == "bay"][["target_quarter", "actual_disp", "nowcast_mean_disp", "period"]].copy()
        S = pf[pf["model"] == "bay_sv"][["target_quarter", "actual_disp", "nowcast_mean_disp"]].copy()
        if not A.empty and not S.empty:
            A = A.rename(columns={"nowcast_mean_disp": "fc_bay"})
            S = S.rename(columns={"nowcast_mean_disp": "fc_sv"})
            m = A.merge(S, on=["target_quarter", "actual_disp"], how="inner")
            if not m.empty:
                e_b = (m["fc_bay"] - m["actual_disp"]).to_numpy(float)
                e_s = (m["fc_sv"] - m["actual_disp"]).to_numpy(float)
                d_ae = np.abs(e_s) - np.abs(e_b)
                d_se = e_s**2 - e_b**2
                crisis = (m["period"].astype(str) == crisis_name).to_numpy(int)

                for name, d in [("point_AE_diff_sv_minus_bay", d_ae), ("point_SE_diff_sv_minus_bay", d_se)]:
                    mask = np.isfinite(d) & np.isfinite(crisis)
                    dd = d[mask]
                    cc = crisis[mask]
                    if dd.size >= 8:
                        X = np.column_stack([np.ones(dd.size), cc])
                        gamma, se = _ols_gamma_and_se(X, dd)
                        z = gamma / se if np.isfinite(se) and se > 0 else float("nan")
                        out.append(
                            dict(
                                test=name,
                                n=int(dd.size),
                                gamma_crisis_minus_noncrisis=float(gamma),
                                se=float(se),
                                z=float(z),
                                p_value=_two_sided_pval_norm(z) if np.isfinite(z) else float("nan"),
                                sign_interpretation="negative => SV improves more in crisis",
                            )
                        )

    # --- Density: CRPS (loss) and logscore (higher better) ---
    if dens_final is not None and (not dens_final.empty) and ("note" not in dens_final.columns):
        df = dens_final.copy()
        if "period" not in df.columns:
            df["period"] = df["target_quarter"].astype(str).apply(lambda x: _assign_period_label(x, periods))

        B = df[df["model"] == "bay"][["target_quarter", "crps", "logscore", "period"]].copy()
        S = df[df["model"] == "bay_sv"][["target_quarter", "crps", "logscore"]].copy()
        if not B.empty and not S.empty:
            B = B.rename(columns={"crps": "crps_bay", "logscore": "ls_bay"})
            S = S.rename(columns={"crps": "crps_sv", "logscore": "ls_sv"})
            m = B.merge(S, on=["target_quarter"], how="inner")
            if not m.empty:
                d_crps = m["crps_sv"].to_numpy(float) - m["crps_bay"].to_numpy(float)  # negative => SV better
                d_ls = m["ls_sv"].to_numpy(float) - m["ls_bay"].to_numpy(float)        # positive => SV better
                crisis = (m["period"].astype(str) == crisis_name).to_numpy(int)

                for name, d, interp in [
                    ("density_CRPS_diff_sv_minus_bay", d_crps, "negative => SV improves more in crisis"),
                    ("density_LOGSCORE_diff_sv_minus_bay", d_ls, "positive => SV improves more in crisis"),
                ]:
                    mask = np.isfinite(d) & np.isfinite(crisis)
                    dd = d[mask]
                    cc = crisis[mask]
                    if dd.size >= 8:
                        X = np.column_stack([np.ones(dd.size), cc])
                        gamma, se = _ols_gamma_and_se(X, dd)
                        z = gamma / se if np.isfinite(se) and se > 0 else float("nan")
                        out.append(
                            dict(
                                test=name,
                                n=int(dd.size),
                                gamma_crisis_minus_noncrisis=float(gamma),
                                se=float(se),
                                z=float(z),
                                p_value=_two_sided_pval_norm(z) if np.isfinite(z) else float("nan"),
                                sign_interpretation=interp,
                            )
                        )

    if not out:
        return pd.DataFrame([{"note": "Diff-in-diff style tests not computed (missing overlap or too few obs)."}])
    return pd.DataFrame(out)


# =============================================================================
# Figures (existing + new period/loss-diff plots)
# =============================================================================


def fig_actual_vs_both_models_final_within_quarter(
    now: pd.DataFrame,
    outdir: Path,
    ylab: str,
    title_suffix: str = "",
) -> None:
    """
    Actual vs bay vs bay_sv using FINAL within-quarter nowcast per target quarter.
    """
    df = select_final_within_quarter(now)
    if df.empty:
        return
    if "actual_disp" not in df.columns or df["actual_disp"].isna().all():
        return

    df["tq_period"] = pd.PeriodIndex(df["target_quarter"].astype(str), freq="Q")
    df = df.sort_values(["tq_period", "model"])

    actual = df.groupby("tq_period", as_index=False)["actual_disp"].first()
    x_q = actual["tq_period"].dt.to_timestamp(how="end")
    y_q = actual["actual_disp"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_q, y_q, "k--o", label="Actual", markersize=5)

    for model, color in [("bay", "blue"), ("bay_sv", "deepskyblue")]:
        g = df[df["model"] == model].copy()
        if g.empty:
            continue
        g = g.sort_values("tq_period")
        xx = g["tq_period"].dt.to_timestamp(how="end")
        yy = g["nowcast_mean_disp"].to_numpy(dtype=float)
        ax.plot(xx, yy, color=color, linewidth=2.0, marker="o", markersize=5, label=f"DFM ({model})")

    ax.set_xlabel("Target quarter")
    ax.set_ylabel(ylab)
    title_main = "Actual vs nowcasts (final within-quarter)"
    if title_suffix:
        title_main += f" — {title_suffix}"
    ax.set_title(title_main)
    ax.grid(True, alpha=0.3)
    ax.legend()

    _savefig(fig, outdir / f"actual_vs_nowcast__both_models{('__' + _nice(title_suffix)) if title_suffix else ''}")


# -----------------------------
# SCALE FIX HELPERS (NEW)
# -----------------------------


def _fan_chart_data_current_target(
    now: pd.DataFrame,
    packs: List[PosteriorPack],
    model: str,
    display_units: str,
    bands=(0.5, 0.7, 0.9),
    base_units: str = "annualized",
) -> pd.DataFrame:
    """
    Build the data used by fig_fan_chart_current_target, so we can compute common y-limits
    across bay and bay_sv, and then plot each with the same scale.
    """
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
    """
    Build the data used by fig_fan_chart_final_within_quarter, so we can compute common y-limits
    across bay and bay_sv, and then plot each with the same scale.
    """
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
        rec = dict(
            tq_end=tq_end,
            mean=float(np.mean(draws)),
            actual=float(r["actual_disp"]) if pd.notna(r["actual_disp"]) else np.nan,
        )
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
    """
    Compute union y-limits across multiple fan-chart datasets.
    Uses min over all loXX/mean/actual, max over all hiXX/mean/actual.
    """
    if not dfs:
        return None

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


# -----------------------------
# Fan chart plotting (UPDATED)
# -----------------------------


def fig_fan_chart_current_target(
    now: pd.DataFrame,
    packs: List[PosteriorPack],
    model: str,
    outdir: Path,
    ylab: str,
    display_units: str,
    bands=(0.5, 0.7, 0.9),
    base_units: str = "annualized",
    title_suffix: str = "",
    ylim: Optional[Tuple[float, float]] = None,  # <-- NEW
) -> None:
    qdf = _fan_chart_data_current_target(
        now=now,
        packs=packs,
        model=model,
        display_units=display_units,
        bands=bands,
        base_units=base_units,
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

    title = f"{model} — current-quarter density nowcasts"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("Release date")
    ax.set_ylabel(ylab)
    ax.legend(frameon=False, ncol=3)
    _savefig(fig, outdir / f"fan_chart_current_target__{model}{('__' + _nice(title_suffix)) if title_suffix else ''}")


def fig_fan_chart_final_within_quarter(
    now: pd.DataFrame,
    packs: List[PosteriorPack],
    model: str,
    outdir: Path,
    ylab: str,
    display_units: str,
    bands=(0.5, 0.7, 0.9),
    base_units: str = "annualized",
    title_suffix: str = "",
    ylim: Optional[Tuple[float, float]] = None,  # <-- NEW
) -> None:
    qdf = _fan_chart_data_final_within_quarter(
        now=now,
        packs=packs,
        model=model,
        display_units=display_units,
        bands=bands,
        base_units=base_units,
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

    ax.plot(
        qdf["tq_end"],
        qdf["mean"],
        color="#1f77b4",
        marker="o",
        linewidth=1.8,
        label="Posterior mean",
    )

    if qdf["actual"].notna().any():
        ax.plot(
            qdf["tq_end"],
            qdf["actual"],
            color="black",
            marker="o",
            linestyle="--",
            linewidth=1.2,
            label="Actual GDP",
        )

    if ylim is not None and np.isfinite(ylim[0]) and np.isfinite(ylim[1]):
        ax.set_ylim(ylim)

    title = f"{model} — final within-quarter density nowcasts"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("Target quarter")
    ax.set_ylabel(ylab)
    ax.legend(frameon=False, ncol=3)
    _savefig(fig, outdir / f"fan_chart_final_within_quarter__{model}{('__' + _nice(title_suffix)) if title_suffix else ''}")


def fig_pit_hist(scores: pd.DataFrame, model: str, outdir: Path, title_suffix: str) -> None:
    d = scores[scores["model"] == model].copy()
    if d.empty or "pit" not in d.columns:
        return
    pit = pd.to_numeric(d["pit"], errors="coerce").dropna().to_numpy()
    if pit.size < 20:
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(pit, bins=10, density=True, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Uniform(0,1)")
    ax.set_xlabel("PIT")
    ax.set_ylabel("Density")
    ax.set_title(f"{model} — PIT histogram ({title_suffix})")
    ax.legend(frameon=False)
    _savefig(fig, outdir / f"pit_hist__{_nice(title_suffix)}__{model}")


def _shade_periods(ax, periods: List[dict], color="grey", alpha=0.15):
    """
    Shade the period windows (on datetime x-axis). Uses period end timestamps.
    """
    for pr in periods:
        s = _period_str_to_periodQ(pr["start"])
        e = _period_str_to_periodQ(pr["end"])
        if s is None or e is None:
            continue
        x0 = s.to_timestamp(how="start")
        x1 = e.to_timestamp(how="end")
        ax.axvspan(x0, x1, alpha=alpha, color=color)


def fig_lossdiff_timeseries_point(
    now: pd.DataFrame,
    outdir: Path,
    periods: List[dict],
    loss: str = "se",
    A: str = "bay_sv",
    B: str = "bay",
    title_suffix: str = "",
) -> None:
    """
    Plot loss differential over target quarters for point nowcasts:
      d_t = L_A - L_B, where L is AE or SE.
    Negative d_t => A better (lower loss).
    """
    df = _ensure_rw_rows_for_final(now)
    dff = select_final_within_quarter(df)
    dff = dff[dff["actual_disp"].notna()].copy()
    if dff.empty:
        return

    A_df = dff[dff["model"] == A][["target_quarter", "nowcast_mean_disp", "actual_disp"]].copy()
    B_df = dff[dff["model"] == B][["target_quarter", "nowcast_mean_disp", "actual_disp"]].copy()
    if A_df.empty or B_df.empty:
        return

    A_df = A_df.rename(columns={"nowcast_mean_disp": "fc_A"})
    B_df = B_df.rename(columns={"nowcast_mean_disp": "fc_B"})
    m = A_df.merge(B_df, on=["target_quarter", "actual_disp"], how="inner")
    if m.empty:
        return

    m["tq"] = pd.PeriodIndex(m["target_quarter"].astype(str), freq="Q")
    m = m.sort_values("tq")
    x = m["tq"].dt.to_timestamp(how="end")

    eA = (m["fc_A"] - m["actual_disp"]).to_numpy(float)
    eB = (m["fc_B"] - m["actual_disp"]).to_numpy(float)

    if loss.lower() == "se":
        d = eA**2 - eB**2
        ylab = "SE diff (A − B)"
        fname = "lossdiff_point_se"
    else:
        d = np.abs(eA) - np.abs(eB)
        ylab = "AE diff (A − B)"
        fname = "lossdiff_point_ae"

    fig, ax = plt.subplots(figsize=(10, 3.5))
    _shade_periods(ax, [p for p in periods if p["name"] == "covid"], alpha=0.18)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.plot(x, d, marker="o", linewidth=1.5)
    ax.set_title(f"Point loss differential: {A} − {B} ({loss.upper()}){(' — ' + title_suffix) if title_suffix else ''}")
    ax.set_xlabel("Target quarter")
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    _savefig(fig, outdir / f"{fname}__{A}_minus_{B}")


def fig_lossdiff_timeseries_density(
    dens_final_wq: pd.DataFrame,
    outdir: Path,
    periods: List[dict],
    score: str = "crps",
    A: str = "bay_sv",
    B: str = "bay",
    title_suffix: str = "",
) -> None:
    """
    Plot density score differential over target quarters for final within-quarter:
      d_t = score_A - score_B.
    For CRPS: negative => A better. For logscore: positive => A better (but we still plot A-B).
    """
    if "note" in dens_final_wq.columns:
        return

    df = dens_final_wq.copy()
    A_df = df[df["model"] == A][["target_quarter", score]].copy()
    B_df = df[df["model"] == B][["target_quarter", score]].copy()
    if A_df.empty or B_df.empty:
        return

    A_df = A_df.rename(columns={score: "a"})
    B_df = B_df.rename(columns={score: "b"})
    m = A_df.merge(B_df, on=["target_quarter"], how="inner")
    if m.empty:
        return

    m["tq"] = pd.PeriodIndex(m["target_quarter"].astype(str), freq="Q")
    m = m.sort_values("tq")
    x = m["tq"].dt.to_timestamp(how="end")
    d = (m["a"] - m["b"]).to_numpy(float)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    _shade_periods(ax, [p for p in periods if p["name"] == "covid"], alpha=0.18)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.plot(x, d, marker="o", linewidth=1.5)
    ax.set_title(f"Density score differential: {A} − {B} ({score}){(' — ' + title_suffix) if title_suffix else ''}")
    ax.set_xlabel("Target quarter")
    ax.set_ylabel(f"{score} diff (A − B)")
    ax.grid(True, alpha=0.3)
    _savefig(fig, outdir / f"lossdiff_density_{score}__{A}_minus_{B}")


def fig_period_bars_point(point_period: pd.DataFrame, outdir: Path, metric: str, periods: List[dict]) -> None:
    """
    Bar chart of point metric by period and model.
    """
    if point_period.empty or "note" in point_period.columns:
        return

    df = point_period.copy()
    # keep only bay/bay_sv/RW if present
    df = df[df["model"].isin(["bay", "bay_sv", "RW"])].copy()
    if df.empty:
        return

    ord_map = _period_order_map(periods)
    df["period_order"] = df["period"].map(lambda x: ord_map.get(x, 999))
    df = df.sort_values(["period_order", "model"])

    pivot = df.pivot_table(index="period", columns="model", values=metric, aggfunc="mean")
    pivot = pivot.reindex([p["name"] for p in periods if p["name"] in pivot.index])

    fig, ax = plt.subplots(figsize=(9, 3.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"Point {metric} by period (final within-quarter)")
    ax.set_xlabel("Period")
    ax.set_ylabel(metric)
    ax.legend(title="Model", frameon=False, ncol=3)
    ax.grid(True, alpha=0.3)
    _savefig(fig, outdir / f"period_bars_point_{metric.lower()}")


def fig_period_bars_density(dens_period: pd.DataFrame, outdir: Path, metric: str, periods: List[dict]) -> None:
    """
    Bar chart of density metric by period and model (final within-quarter).
    metric should be one of: mean_crps, mean_logscore
    """
    if dens_period.empty or "note" in dens_period.columns:
        return

    df = dens_period.copy()
    df = df[df["model"].isin(["bay", "bay_sv", "ar1"])].copy()
    if df.empty:
        return

    ord_map = _period_order_map(periods)
    df["period_order"] = df["period"].map(lambda x: ord_map.get(x, 999))
    df = df.sort_values(["period_order", "model"])

    pivot = df.pivot_table(index="period", columns="model", values=metric, aggfunc="mean")
    pivot = pivot.reindex([p["name"] for p in periods if p["name"] in pivot.index])

    fig, ax = plt.subplots(figsize=(9, 3.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"Density {metric} by period (final within-quarter)")
    ax.set_xlabel("Period")
    ax.set_ylabel(metric)
    ax.legend(title="Model", frameon=False, ncol=3)
    ax.grid(True, alpha=0.3)
    _savefig(fig, outdir / f"period_bars_density_{metric.lower()}")


# =============================================================================
# Orchestration
# =============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Path to outputs/<run_id>")

    ap.add_argument(
        "--actual_vintage",
        type=str,
        default="data/NL/2025-11-14.xlsx",
        help="Relative path to most complete vintage used as 'actual' GDP",
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
        help="Units for plots/tables. 'qoq' converts from annualized via compounding.",
    )

    ap.add_argument("--hac_lag", type=int, default=1, help="HAC lag for DM test (Newey–West).")

    ap.add_argument(
        "--periods_json",
        type=str,
        default="",
        help="Optional JSON string to override DEFAULT_PERIODS. "
             "Format: [{'name':'pre', 'label':'..', 'start':'2015Q1','end':'2019Q4'}, ...]",
    )

    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    project_root = Path.cwd()

    fig_dir = run_dir / "figures"
    tab_dir = run_dir / "tables"
    _ensure_dir(fig_dir)
    _ensure_dir(tab_dir)

    # periods
    if args.periods_json.strip():
        try:
            periods = json.loads(args.periods_json)
            if not isinstance(periods, list):
                raise ValueError("periods_json must decode to a list")
        except Exception as e:
            raise ValueError(f"Could not parse --periods_json: {e}")
    else:
        periods = DEFAULT_PERIODS

    now, packs, meta = scan_run(run_dir)

    # Attach realized GDP + RW benchmark (RW used ONLY for tables/tests)
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
        print(f"[make_densityplots] WARNING: could not attach actual GDP / RW benchmark: {e}")
        now = now.copy()
        now["actual"] = np.nan
        now["rw_nowcast"] = np.nan

    # Display scale (assumes base is annualized)
    ylab = "Real GDP growth (QoQ, %)" if args.display_units == "qoq" else "Real GDP growth (annualized, %)"
    now = _apply_display_units(now, display_units=args.display_units, base_units="annualized")

    # Derived columns
    now = add_release_step_columns(now)
    now = add_release_date(now)
    now = add_period_columns(now, periods)

    # Save enriched nowcasts
    (run_dir / "nowcasts").mkdir(parents=True, exist_ok=True)
    now.to_csv(run_dir / "nowcasts" / "nowcasts_with_actual_rw.csv", index=False)

    # ----------------------------
    # Tables: Run summary
    # ----------------------------
    batch = meta.get("batch_config", {})
    data = meta.get("data", {})
    runtime = meta.get("runtime", {})
    run_summary = pd.DataFrame(
        [
            ("run_id", meta.get("run_id", "")),
            ("created_at_utc", meta.get("created_at_utc", "")),
            ("seconds", runtime.get("seconds", "")),
            (
                "model_types",
                ",".join(batch.get("model_types", []))
                if isinstance(batch.get("model_types", []), list)
                else str(batch.get("model_types", "")),
            ),
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
        ],
        columns=["key", "value"],
    )
    run_summary.to_csv(tab_dir / "run_summary.csv", index=False)
    _to_latex_table(run_summary, tab_dir / "run_summary.tex", caption="Run summary", label="tab:run_summary")

    # ----------------------------
    # Point errors (FINAL within-quarter)
    # ----------------------------
    pt_final_wq = table_point_errors_final_within_quarter(now)
    pt_final_wq.to_csv(tab_dir / "point_errors_final_within_quarter.csv", index=False)
    _to_latex_table(
        pt_final_wq,
        tab_dir / "point_errors_final_within_quarter.tex",
        caption=f"Point forecast errors (final within-quarter, units: {args.display_units})",
        label="tab:point_errors_final_within_quarter",
    )

    # Point errors by period
    pt_by_period = table_point_errors_by_period_final_within_quarter(now, periods)
    pt_by_period.to_csv(tab_dir / "point_errors_by_period_final_within_quarter.csv", index=False)
    _to_latex_table(
        pt_by_period,
        tab_dir / "point_errors_by_period_final_within_quarter.tex",
        caption=f"Point forecast errors by period (final within-quarter, units: {args.display_units})",
        label="tab:point_errors_by_period_final_within_quarter",
    )

    # ----------------------------
    # Density scores
    # ----------------------------
    dens_cur = density_scores_current_quarter(now, packs, display_units=args.display_units)
    dens_cur.to_csv(tab_dir / "density_scores_current_quarter.csv", index=False)

    dens_final_wq = density_scores_final_within_quarter(now, packs, display_units=args.display_units)
    dens_final_wq.to_csv(tab_dir / "density_scores_final_within_quarter.csv", index=False)

    dens_cur_sum = summarize_density_scores(dens_cur, by_step=False)
    dens_cur_sum.to_csv(tab_dir / "density_summary_current_quarter.csv", index=False)

    dens_final_wq_sum = summarize_density_scores(dens_final_wq, by_step=False)
    dens_final_wq_sum.to_csv(tab_dir / "density_summary_final_within_quarter.csv", index=False)

    # Density summary by period (final within-quarter)
    dens_final_wq_by_period = summarize_density_scores_by_period(dens_final_wq, periods)
    dens_final_wq_by_period.to_csv(tab_dir / "density_summary_by_period_final_within_quarter.csv", index=False)
    _to_latex_table(
        dens_final_wq_by_period,
        tab_dir / "density_summary_by_period_final_within_quarter.tex",
        caption="Density summary by period (final within-quarter)",
        label="tab:density_summary_by_period_final_within_quarter",
    )

    # ----------------------------
    # DM tests: density (full sample final within-quarter)
    # ----------------------------
    models = sorted(now["model"].dropna().unique())
    dm_out_list = []

    if "note" not in dens_final_wq.columns:

        def _maybe_dm_density(model_A: str, model_B: str):
            if (model_A in dens_final_wq["model"].unique()) and (model_B in dens_final_wq["model"].unique()):
                A = dens_final_wq[dens_final_wq["model"] == model_A].copy()
                B = dens_final_wq[dens_final_wq["model"] == model_B].copy()
                dm_crps = dm_test(A, B, key_cols=["target_quarter"], score_col="crps", hac_lag=args.hac_lag)
                dm_ls = dm_test(A, B, key_cols=["target_quarter"], score_col="logscore", hac_lag=args.hac_lag)
                dm_crps = dm_crps.assign(model_A=model_A, model_B=model_B)
                dm_ls = dm_ls.assign(model_A=model_A, model_B=model_B)
                dm_out_list.extend([dm_crps, dm_ls])

        _maybe_dm_density("bay", "bay_sv")
        _maybe_dm_density("bay", "ar1")
        _maybe_dm_density("bay_sv", "ar1")

    if dm_out_list:
        dm_out = pd.concat(dm_out_list, ignore_index=True)
        dm_out.to_csv(tab_dir / "dm_tests_density_final_within_quarter.csv", index=False)
    else:
        pd.DataFrame([{"note": "DM tests skipped (need density scores and overlapping models)."}]).to_csv(
            tab_dir / "dm_tests_density_final_within_quarter.csv", index=False
        )

    # DM tests: density by period
    dens_dm_by_period = dm_tests_density_by_period(
        dens_final_wq,
        periods=periods,
        comparisons=[("bay", "bay_sv"), ("bay", "ar1"), ("bay_sv", "ar1")],
        hac_lag=args.hac_lag,
    )
    dens_dm_by_period.to_csv(tab_dir / "dm_tests_density_by_period_final_within_quarter.csv", index=False)

    # ----------------------------
    # DM tests: point (full sample final within-quarter)
    # ----------------------------
    dm_point_list = []
    for A, B in [("bay", "bay_sv"), ("bay", "RW"), ("bay_sv", "RW")]:
        dm_point_list.append(dm_test_point_final_within_quarter(now, A, B, loss="ae", hac_lag=args.hac_lag))
        dm_point_list.append(dm_test_point_final_within_quarter(now, A, B, loss="se", hac_lag=args.hac_lag))
    dm_point = pd.concat(dm_point_list, ignore_index=True) if dm_point_list else pd.DataFrame(
        [{"note": "Point DM tests skipped."}]
    )
    dm_point.to_csv(tab_dir / "dm_tests_point_final_within_quarter.csv", index=False)

    # DM tests: point by period
    point_dm_by_period = dm_tests_point_by_period(
        now,
        periods=periods,
        comparisons=[("bay", "bay_sv"), ("bay", "RW"), ("bay_sv", "RW")],
        hac_lag=args.hac_lag,
    )
    point_dm_by_period.to_csv(tab_dir / "dm_tests_point_by_period_final_within_quarter.csv", index=False)

    # ----------------------------
    # Diff-in-diff style tests (SV advantage differs in crisis?)
    # ----------------------------
    point_final = select_final_within_quarter(_ensure_rw_rows_for_final(now))
    did = diff_in_diff_tests(point_final=point_final, dens_final=dens_final_wq, periods=periods, crisis_name="covid")
    did.to_csv(tab_dir / "diff_in_diff_tests.csv", index=False)

    # ----------------------------
    # Figures (PNG only) - existing + new
    # ----------------------------
    if not now.empty:
        # main plot: actual vs nowcasts (full sample)
        fig_actual_vs_both_models_final_within_quarter(now, fig_dir, ylab=ylab)

        # ----------------------------
        # SCALE FIX: compute common y-lims for bay + bay_sv fan charts
        # ----------------------------
        common_ylim_current = None
        common_ylim_final = None
        if "bay" in models or "bay_sv" in models:
            # current-quarter fan charts
            d_bay_c = _fan_chart_data_current_target(now, packs, "bay", display_units=args.display_units) if "bay" in models else pd.DataFrame()
            d_sv_c = _fan_chart_data_current_target(now, packs, "bay_sv", display_units=args.display_units) if "bay_sv" in models else pd.DataFrame()
            common_ylim_current = _compute_common_ylim_from_fan_data([d_bay_c, d_sv_c])

            # final-within-quarter fan charts
            d_bay_f = _fan_chart_data_final_within_quarter(now, packs, "bay", display_units=args.display_units) if "bay" in models else pd.DataFrame()
            d_sv_f = _fan_chart_data_final_within_quarter(now, packs, "bay_sv", display_units=args.display_units) if "bay_sv" in models else pd.DataFrame()
            common_ylim_final = _compute_common_ylim_from_fan_data([d_bay_f, d_sv_f])

        # fan charts (now with common y-scale for bay vs bay_sv)
        for model in ["bay", "bay_sv"]:
            if model in models:
                fig_fan_chart_current_target(
                    now, packs, model, fig_dir, ylab=ylab, display_units=args.display_units, ylim=common_ylim_current
                )
                fig_fan_chart_final_within_quarter(
                    now, packs, model, fig_dir, ylab=ylab, display_units=args.display_units, ylim=common_ylim_final
                )

        # PIT histograms
        if "note" not in dens_cur.columns:
            for model in ["bay", "bay_sv"]:
                if model in models:
                    fig_pit_hist(dens_cur, model, fig_dir, title_suffix="current-quarter")
        if "note" not in dens_final_wq.columns:
            for model in ["bay", "bay_sv"]:
                if model in models:
                    fig_pit_hist(dens_final_wq, model, fig_dir, title_suffix="final-within-quarter")

        # New: loss differential time series (SV - bay)
        fig_lossdiff_timeseries_point(now, fig_dir, periods, loss="se", A="bay_sv", B="bay")
        fig_lossdiff_timeseries_point(now, fig_dir, periods, loss="ae", A="bay_sv", B="bay")
        fig_lossdiff_timeseries_density(dens_final_wq, fig_dir, periods, score="crps", A="bay_sv", B="bay")
        fig_lossdiff_timeseries_density(dens_final_wq, fig_dir, periods, score="logscore", A="bay_sv", B="bay")

        # New: period bar charts
        if isinstance(pt_by_period, pd.DataFrame) and ("note" not in pt_by_period.columns):
            fig_period_bars_point(pt_by_period, fig_dir, metric="MAE", periods=periods)
            fig_period_bars_point(pt_by_period, fig_dir, metric="RMSFE", periods=periods)

        if isinstance(dens_final_wq_by_period, pd.DataFrame) and ("note" not in dens_final_wq_by_period.columns):
            fig_period_bars_density(dens_final_wq_by_period, fig_dir, metric="mean_crps", periods=periods)
            fig_period_bars_density(dens_final_wq_by_period, fig_dir, metric="mean_logscore", periods=periods)

    # ----------------------------
    # Index file
    # ----------------------------
    index_lines = []
    index_lines.append(f"Run: {run_dir.name}")
    index_lines.append("")
    index_lines.append("Periods:")
    for pr in periods:
        index_lines.append(f"  - {pr.get('name')} : {pr.get('start')}..{pr.get('end')} ({pr.get('label','')})")
    index_lines.append("")
    index_lines.append("Figures (PNG):")
    for p in sorted(fig_dir.glob("*.png")):
        index_lines.append(f"  - {p.name}")
    index_lines.append("")
    index_lines.append("Tables (CSV/TEX):")
    for p in sorted(tab_dir.glob("*.csv")):
        index_lines.append(f"  - {p.name}")
    index_lines.append("")
    index_lines.append("Nowcasts:")
    index_lines.append("  - nowcasts_with_actual_rw.csv (enriched + display-scale + rel-step + release_date + period)")
    index_lines.append("")
    index_lines.append("Notes:")
    index_lines.append("  - RW is computed and used ONLY for tables/tests (not plotted).")
    index_lines.append("  - Density scoring requires posterior draws saved (SAVE_DRAWS=True).")
    index_lines.append(
        "  - Final within-quarter selection = last snapshot with asof_in_target==True and rel_step_in_target in [0..8]."
    )
    index_lines.append("  - Point DM tests use AE and SE losses; negative mean_diff => model_A better.")
    index_lines.append("  - Density DM tests use score differences A-B (CRPS lower better; logscore higher better).")
    index_lines.append("  - Diff-in-diff tests regress SV-bay differential on crisis dummy (period=='covid').")
    index_lines.append("  - SCALE FIX: bay and bay_sv fan charts share common y-limits (union across both).")

    (run_dir / "ARTIFACTS_INDEX.txt").write_text("\n".join(index_lines))

    print(f"Artifacts written to:\n  {fig_dir}\n  {tab_dir}\nIndex:\n  {run_dir/'ARTIFACTS_INDEX.txt'}")


if __name__ == "__main__":
    main()