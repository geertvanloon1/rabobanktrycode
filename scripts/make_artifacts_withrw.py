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

# python3 scripts/make_artifacts_withrw.py --run_dir 
# ----------------------------
# Utilities
# ----------------------------

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


def ann_to_qoq_pct(x_ann: pd.Series) -> pd.Series:
    x = pd.to_numeric(x_ann, errors="coerce") / 100.0
    return 100.0 * (np.power(1.0 + x, 1.0 / 4.0) - 1.0)


def qoq_to_ann_pct(x_qoq: pd.Series) -> pd.Series:
    x = pd.to_numeric(x_qoq, errors="coerce") / 100.0
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


def _place_right_label(ax, x, y, text, used_y: List[float], min_sep: float, **kwargs) -> None:
    """
    Place labels on the right side while avoiding heavy overlap by nudging y upward.
    """
    yy = float(y)
    for _ in range(40):
        if all(abs(yy - uy) > min_sep for uy in used_y):
            break
        yy += min_sep
    used_y.append(yy)
    ax.text(x, yy, text, **kwargs)


# ----------------------------
# Actual GDP + RW benchmark helpers
# ----------------------------

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


# ----------------------------
# Posterior pack loading (optional; kept for compatibility)
# ----------------------------

@dataclass
class PosteriorPack:
    model: str
    tag: str
    month: pd.Timestamp
    q: int
    month_in_quarter: int
    vintage_file: str
    path: Path

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
    nowcast_draws: np.ndarray

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

    return PosteriorPack(
        model=model,
        tag=tag,
        month=m0,
        q=q,
        month_in_quarter=miq_use,
        vintage_file=vint,
        path=subdir,

        F_mean=get("F_mean"),
        F_sd=get("F_sd"),
        Lambda_mean=get("Lambda_mean"),
        Lambda_sd=get("Lambda_sd"),
        mu_mean=get("mu_mean"),
        z_mean=get("z_mean"),
        z_draws=get("z_draws"),
        r_draws=get("r_draws"),
        beta_mean=get("beta_mean"),
        a_mean=get("a_mean"),
        sigma2_mean=get("sigma2_mean"),
        eta2_mean=float(get("eta2_mean")) if get("eta2_mean") is not None else float("nan"),
        Psi_mean=get("Psi_mean"),
        omega_mean=get("omega_mean"),
        omega_sd=get("omega_sd"),
        tau2_mean=float(get("tau2_mean")) if get("tau2_mean") is not None else float("nan"),
        nowcast_draws=get("nowcast_draws") if get("nowcast_draws") is not None else np.asarray([]),

        dates_m=get("dates_m") if get("dates_m") is not None else np.asarray([]),
        dates_q=get("dates_q") if get("dates_q") is not None else np.asarray([]),
        series=get("series") if get("series") is not None else np.asarray([]),

        standardize_mean=get("standardize_mean") if get("standardize_mean") is not None else np.asarray([]),
        standardize_sd=get("standardize_sd") if get("standardize_sd") is not None else np.asarray([]),
    )


def scan_run(run_dir: Path) -> Tuple[pd.DataFrame, List[PosteriorPack], dict]:
    meta = _safe_read_json(run_dir / "run_metadata.json")

    # Prefer enriched if already exists
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


# ----------------------------
# Derived columns for new logic
# ----------------------------

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


# ----------------------------
# Tables
# ----------------------------

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
    ]
    return pd.DataFrame(rows, columns=["key", "value"])


def table_nowcast_errors_within_quarter(now: pd.DataFrame) -> pd.DataFrame:
    """
    Within-quarter nowcast errors:
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
    Final-before-release errors:
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


# ----------------------------
# Figures (updated to new logic)
# ----------------------------

def fig_revision_paths_from_qstart_until_release(
    now: pd.DataFrame,
    model: str,
    outdir: Path,
    ylab: str,
    last_k_quarters: int = 6,
) -> None:
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

    # label spacing
    yvals = pd.to_numeric(df.get("nowcast_mean_disp", pd.Series(dtype=float)), errors="coerce")
    yr = float(np.nanmax(yvals) - np.nanmin(yvals)) if yvals.notna().any() else 10.0
    min_sep = max(0.25, 0.03 * yr)

    fig = plt.figure(figsize=(11, 5))
    ax = plt.gca()

    max_step = 0
    used_y: List[float] = []

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
            ax.text(
                float(xs.max()) + 0.15, a, f"{t} actual",
                fontsize=8, va="center"
            )

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


def fig_within_quarter_nowcast_paths(
    now: pd.DataFrame,
    model: str,
    outdir: Path,
    ylab: str,
    last_k_quarters: int = 8,
) -> None:
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

    yvals = pd.to_numeric(df.get("nowcast_mean_disp", pd.Series(dtype=float)), errors="coerce")
    yr = float(np.nanmax(yvals) - np.nanmin(yvals)) if yvals.notna().any() else 10.0
    min_sep = max(0.25, 0.03 * yr)

    fig = plt.figure(figsize=(11, 5))
    ax = plt.gca()
    used_y: List[float] = []

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
            ax.text(
                float(xs.max()) + 0.15, a, f"{t} actual",
                fontsize=8, va="center"
            )

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


def fig_actual_vs_final_nowcast_over_time(
    now: pd.DataFrame,
    model: str,
    outdir: Path,
    ylab: str,
    last_k_quarters: int = 12,
) -> None:
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


def fig_all_nowcasts_calendar_current_target(
    now: pd.DataFrame,
    model: str,
    outdir: Path,
    ylab: str,
) -> None:
    """
    One calendar-time graph (x-axis = release_date) plotting ONLY the current-target nowcasts:
      rows where asof_in_target==True.
    """
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
        # show actual per target quarter as horizontal segments across its date span
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


def fig_all_nowcasts_calendar_by_target(
    now: pd.DataFrame,
    model: str,
    outdir: Path,
    ylab: str,
    last_k_quarters: int = 12,
) -> None:
    """
    One calendar-time graph (x-axis = release_date) with one line per target_quarter.
    """
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


# ----------------------------
# Orchestration
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Path to outputs/<run_id>")

    ap.add_argument("--actual_vintage", type=str, default="data/NL/2025-10-16.xlsx",
                    help="Relative path to most complete vintage used as 'actual' GDP")
    ap.add_argument("--spec_file", type=str, default="data/Spec_NL.xlsx",
                    help="Relative path to Spec file")
    ap.add_argument("--gdp_series_id", type=str, default="Real GDP")
    ap.add_argument("--sample_start", type=str, default="1985-01-01")

    ap.add_argument("--display_units", type=str, default="annualized", choices=["qoq", "annualized"],
                    help="Units for plots/tables. 'qoq' converts from annualized via compounding.")

    ap.add_argument("--last_k_quarters", type=int, default=12,
                    help="How many recent quarters to show in time plots.")
    ap.add_argument("--last_k_revision_quarters", type=int, default=6,
                    help="How many recent quarters to show in revision plots.")

    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    project_root = Path.cwd()

    fig_dir = run_dir / "figures"
    tab_dir = run_dir / "tables"
    _ensure_dir(fig_dir)
    _ensure_dir(tab_dir)

    now, _packs, meta = scan_run(run_dir)

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

    # New columns for new logic
    now = add_release_step_columns(now)
    now = add_release_date(now)

    # Save enriched nowcasts
    (run_dir / "nowcasts").mkdir(parents=True, exist_ok=True)
    now.to_csv(run_dir / "nowcasts" / "nowcasts_with_actual_rw.csv", index=False)

    # Tables
    run_summary = table_run_summary(meta)
    run_summary.to_csv(tab_dir / "run_summary.csv", index=False)
    _to_latex_table(run_summary, tab_dir / "run_summary.tex", caption="Run summary", label="tab:run_summary")

    err_wq = table_nowcast_errors_within_quarter(now)
    err_wq.to_csv(tab_dir / "nowcast_errors_within_quarter.csv", index=False)
    _to_latex_table(
        err_wq,
        tab_dir / "nowcast_errors_within_quarter.tex",
        caption=f"Within-quarter nowcast error metrics (units: {args.display_units})",
        label="tab:nowcast_errors_within_quarter",
    )

    err_final = table_nowcast_errors_final_before_release(now)
    err_final.to_csv(tab_dir / "nowcast_errors_final_before_release.csv", index=False)
    _to_latex_table(
        err_final,
        tab_dir / "nowcast_errors_final_before_release.tex",
        caption=f"Final-before-release nowcast error metrics (units: {args.display_units})",
        label="tab:nowcast_errors_final_before_release",
    )

    # Figures
    if not now.empty:
        for model in sorted(now["model"].dropna().unique()):
            fig_within_quarter_nowcast_paths(
                now, model, fig_dir, ylab=ylab,
                last_k_quarters=min(8, args.last_k_quarters)
            )
            fig_revision_paths_from_qstart_until_release(
                now, model, fig_dir, ylab=ylab,
                last_k_quarters=args.last_k_revision_quarters
            )
            fig_actual_vs_final_nowcast_over_time(
                now, model, fig_dir, ylab=ylab,
                last_k_quarters=args.last_k_quarters
            )

            # NEW: calendar date plots
            fig_all_nowcasts_calendar_current_target(now, model, fig_dir, ylab=ylab)
            fig_all_nowcasts_calendar_by_target(
                now, model, fig_dir, ylab=ylab,
                last_k_quarters=args.last_k_quarters
            )

    # Index file
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

    (run_dir / "ARTIFACTS_INDEX.txt").write_text("\n".join(index_lines))

    print(f"Artifacts written to:\n  {fig_dir}\n  {tab_dir}\nIndex:\n  {run_dir/'ARTIFACTS_INDEX.txt'}")


if __name__ == "__main__":
    main()
