from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zhangnowcast.data.data import build_zhang_data

# python3 scripts/insample_diagnostics_plots.py --run_dir outputs/20260301_135623_99d2920a

# paths relative to project root (same as runner)
DATA_DIR_REL = Path("data/NL")
SPEC_FILE_REL = Path("data/Spec_NL.xlsx")


# =========================
# SCALE HELPERS (NEW)
# =========================

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


def _common_ylim(arrays: List[np.ndarray], pad_frac: float = 0.05) -> Optional[Tuple[float, float]]:
    vals = []
    for a in arrays:
        if a is None:
            continue
        x = np.asarray(a, dtype=float).ravel()
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)
    if not vals:
        return None
    x = np.concatenate(vals)
    return _padded_ylim(float(np.min(x)), float(np.max(x)), pad_frac=pad_frac)


def ann_to_qoq_pct(x_ann: pd.Series) -> pd.Series:
    """Annualized QoQ percent -> QoQ percent via compounding."""
    x = pd.to_numeric(x_ann, errors="coerce") / 100.0
    return 100.0 * (np.power(1.0 + x, 1.0 / 4.0) - 1.0)


def extract_actual_gdp(D) -> tuple:
    """Extract quarterly GDP LEVELS (not growth) from ZhangData."""
    y = None
    for name in ("y_q", "Y_q", "gdp_q", "GDP_q"):
        if hasattr(D, name):
            y = np.asarray(getattr(D, name), dtype=float)
            break

    if y is None:
        return np.array([]), np.array([])

    if not hasattr(D, "dates_q"):
        return np.array([]), np.array([])

    dates_q = pd.to_datetime(np.asarray(D.dates_q))
    valid_idx = ~np.isnan(y)
    return dates_q[valid_idx], y[valid_idx]  # levels


# Utilities
def _have_jinja2() -> bool:
    try:
        import jinja2  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(fig, outbase: Path) -> None:
    """Save only .png with tight layout."""
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=200)
    plt.close(fig)


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _coerce_dt(x) -> pd.Timestamp:
    return pd.to_datetime(x).to_period("M").to_timestamp()


def _nice(s: str) -> str:
    """File-safe slug-ish string."""
    return "".join(c if c.isalnum() or c in "-_=." else "_" for c in s)


def _to_latex_table(df: pd.DataFrame, path: Path, caption: str = "", label: str = "") -> None:
    """
    Write a LaTeX table if jinja2 is available; otherwise skip silently.
    CSV output is always written elsewhere, so LaTeX is optional.
    """
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


@dataclass
class PosteriorPack:
    model: str
    tag: str
    month: pd.Timestamp
    q: int
    month_in_quarter: int
    target_quarter: str
    vintage_file: str
    path: Path

    # arrays
    F_mean: np.ndarray      # (T, R)
    F_sd: np.ndarray        # (T, R)
    Lambda_mean: np.ndarray # (N, R)
    Lambda_sd: np.ndarray   # (N, R)
    mu_mean: np.ndarray     # (N,)
    z_mean: np.ndarray      # (R,)
    z_draws: np.ndarray     # (S, R)
    r_draws: np.ndarray     # (S,)
    beta_mean: np.ndarray   # (1+3R+1,)
    a_mean: np.ndarray      # (R,)
    sigma2_mean: np.ndarray # (R,)
    eta2_mean: float
    Psi_mean: np.ndarray    # (N, N)
    omega_mean: np.ndarray  # (T, N)
    omega_sd: np.ndarray    # (T, N)
    tau2_mean: float
    nowcast_draws: np.ndarray  # (S,)

    dates_m: np.ndarray     # (T,)
    dates_q: np.ndarray     # (?)
    series: np.ndarray      # (N,)

    standardize_mean: np.ndarray # (N,)
    standardize_sd: np.ndarray   # (N,)


def load_posterior_pack(model: str, subdir: Path) -> Optional[PosteriorPack]:
    """
    subdir is outputs/<run_id>/posterior/per_run/<model>/<tag>/
    """
    npz_path = subdir / "posterior.npz"
    meta_path = subdir / "posterior_meta.json"
    if not npz_path.exists() or not meta_path.exists():
        return None

    meta = _safe_read_json(meta_path)
    tag = meta.get("subrun_tag", subdir.name)
    month = pd.to_datetime(meta.get("month_T", "1970-01-01"))
    q = int(meta.get("q", -1))
    miq = int(meta.get("month_in_quarter", -1))
    tgt = str(meta.get("target_quarter", ""))
    vint = str(meta.get("vintage_file", ""))

    z = np.load(npz_path, allow_pickle=True)

    def get(name, default=None):
        return z[name] if name in z.files else default

    return PosteriorPack(
        model=model,
        tag=tag,
        month=_coerce_dt(month),
        q=q,
        month_in_quarter=miq if miq > 0 else (((int(_coerce_dt(month).month) - 1) % 3) + 1),
        target_quarter=tgt,
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
        nowcast_draws=get("nowcast_draws"),
        dates_m=get("dates_m"),
        dates_q=get("dates_q"),
        series=get("series"),
        standardize_mean=get("standardize_mean"),
        standardize_sd=get("standardize_sd"),
    )


def load_final_vintage_posterior(run_dir: Path, model: str, vintage_name: str) -> Optional[PosteriorPack]:
    """
    Load posterior results for the specified model and vintage.
    """
    model_dir = run_dir / "posterior" / "per_run" / model / vintage_name
    print(f"Debug: Trying to access directory: {model_dir}")

    if not model_dir.exists():
        print(f"Error: Directory does not exist: {model_dir}")
        return None

    npz_path = model_dir / "posterior.npz"
    meta_path = model_dir / "posterior_meta.json"
    print(f"Debug: Checking for files: {npz_path} and {meta_path}")

    if not npz_path.exists():
        print(f"Error: Missing posterior.npz file: {npz_path}")
        return None
    if not meta_path.exists():
        print(f"Error: Missing posterior_meta.json file: {meta_path}")
        return None

    return load_posterior_pack(model, model_dir)


def backcast(D, rep: PosteriorPack):
    """
    Backcasting using factor model: GDP ~ intercept + current factors + 3 lags of all factors.
    We use the actual length of beta_mean.
    """
    F_mean = np.asarray(rep.F_mean)  # (T, R)
    R = F_mean.shape[1]
    T = F_mean.shape[0]

    beta = np.asarray(rep.beta_mean).ravel()
    print(f"Debug: R={R}, beta_mean.shape={beta.shape}, len={len(beta)}")

    X = np.zeros((T, len(beta)))
    X[:, 0] = 1.0  # intercept

    for t in range(T):
        # current factors
        X[t, 1:1 + R] = F_mean[t, :]
        # lag 1
        if t >= 1:
            X[t, 1 + R:1 + 2 * R] = F_mean[t - 1, :]
        # lag 2
        if t >= 2:
            X[t, 1 + 2 * R:1 + 3 * R] = F_mean[t - 2, :]
        # possible extra coefficient (e.g. lag 3 of factor 1)
        if t >= 3 and len(beta) > 1 + 3 * R:
            X[t, -1] = F_mean[t - 3, 0]

    backcasted_values = X @ beta
    return backcasted_values


# ========= STOCHASTIC VOLATILITY FIGURES / TABLES =========

def fig_sv_aggregate(
    rep: PosteriorPack,
    outdir: Path,
    title_suffix: str = "",
    ylim: Optional[Tuple[float, float]] = None,  # NEW
) -> None:
    """
    Aggregate stochastic volatility index: median omega across series with 10–90 band.
    """
    if rep.omega_mean is None:
        return

    Om = np.asarray(rep.omega_mean)  # (T, N)
    if Om.ndim != 2 or Om.size == 0:
        return

    dates = pd.to_datetime(rep.dates_m)
    med = np.nanmedian(Om, axis=1)
    p10 = np.nanpercentile(Om, 10, axis=1)
    p90 = np.nanpercentile(Om, 90, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, med, color="navy", linewidth=2, label="Median log volatility")
    ax.fill_between(dates, p10, p90, color="lightsteelblue", alpha=0.4,
                    label="10–90 percentile across indicators")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel("Month")
    ax.set_ylabel("Log standard deviation (omega)")
    title = "Aggregate stochastic volatility index"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    _savefig(fig, outdir / f"sv_aggregate_{_nice(rep.model)}"
                    f"{('__' + _nice(title_suffix)) if title_suffix else ''}")


def fig_sv_selected_series(
    rep: PosteriorPack,
    outdir: Path,
    top_n: int = 6,
    ylim: Optional[Tuple[float, float]] = None,  # NEW (shared scale)
) -> None:
    """
    Plot stochastic volatility paths for indicators whose log volatility varies most.
    """
    if rep.omega_mean is None:
        return

    Om = np.asarray(rep.omega_mean)  # (T, N)
    if Om.ndim != 2 or Om.size == 0:
        return

    dates = pd.to_datetime(rep.dates_m)
    series_names = np.asarray(rep.series, dtype=object)

    # Time-series variance of omega per series
    var_omega = np.nanvar(Om, axis=0)
    idx = np.argsort(-var_omega)[: min(top_n, Om.shape[1])]

    fig, axes = plt.subplots(len(idx), 1, figsize=(10, 2.3 * len(idx)), sharex=True)
    if len(idx) == 1:
        axes = [axes]

    for ax, j in zip(axes, idx):
        omega_j = Om[:, j]
        name_j = str(series_names[j])

        ax.plot(dates, omega_j, color="darkred", linewidth=1.5)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel(f"{name_j}\nlog sd")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Month")
    fig.suptitle("Stochastic volatility for selected indicators", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    _savefig(fig, outdir / f"sv_selected_series_{_nice(rep.model)}")


def sv_summary_table(rep: PosteriorPack, out_dir: Path) -> None:
    """
    Simple summary statistics of stochastic volatility:
    average omega by period (pre-GFC, GFC, COVID, post-COVID).
    """
    if rep.omega_mean is None:
        return

    Om = np.asarray(rep.omega_mean)
    dates = pd.to_datetime(rep.dates_m)

    med = np.nanmedian(Om, axis=1)

    periods = {
        "Pre-GFC (2002–2006)": ((dates >= "2002-01-01") & (dates < "2007-01-01")),
        "GFC (2007–2010)": ((dates >= "2007-01-01") & (dates <= "2010-12-31")),
        "Post-GFC / pre-COVID (2011–2018)": ((dates >= "2011-01-01") & (dates <= "2018-12-31")),
        "COVID period (2019–2021)": ((dates >= "2019-01-01") & (dates <= "2021-12-31")),
        "Post-COVID (2022–2025)": ((dates >= "2022-01-01") & (dates <= "2025-12-31")),
    }

    rows = []
    for label, mask in periods.items():
        vals = med[mask]
        if vals.size == 0:
            avg = np.nan
            sd = np.nan
        else:
            avg = float(np.nanmean(vals))
            sd = float(np.nanstd(vals))
        rows.append({"Period": label,
                     "Mean median omega": avg,
                     "SD median omega": sd})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"sv_summary_{_nice(rep.model)}.csv", index=False)
    _to_latex_table(
        df,
        out_dir / f"sv_summary_{_nice(rep.model)}.tex",
        caption="Summary statistics of aggregate stochastic volatility (median log standard deviation) across periods.",
        label=f"tab:sv_summary_{_nice(rep.model)}",
    )


# ==================== FIGURES / TABLES ====================

def fig_backcast_growth(
    backcasted_values,
    dates_m,
    actual_gdp_growth,
    actual_dates_q,
    outdir: Path,
    suffix: str = "",
    ylim: Optional[Tuple[float, float]] = None,  # NEW
) -> None:
    """Clean single-panel: Backcast vs Actual GDP Growth."""
    dates_m_pd = pd.to_datetime(dates_m)

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(
        dates_m_pd, backcasted_values,
        label=f"{suffix} Backcast (QoQ ann. growth %)",
        color="blue", linewidth=2.5, alpha=0.9
    )
    ax.plot(
        actual_dates_q, actual_gdp_growth, "ro",
        label="Actual GDP (QoQ ann. growth %)",
        markersize=12, linewidth=2.5
    )

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel("Date")
    ax.set_ylabel("GDP Growth (QoQ annualized, %)")
    ax.set_title(
        f"{suffix.upper()} Factor Model Backcast vs Actual Quarterly GDP Growth",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    name = f"backcast_vs_actual_growth_{suffix}" if suffix else "backcast_vs_actual_growth"
    _savefig(fig, outdir / name)


def generate_parameter_table(rep: PosteriorPack, out_dir: Path, suffix: str = "") -> None:
    beta = np.asarray(rep.beta_mean).ravel()
    R = rep.F_mean.shape[1]

    rows = []
    rows.append({"Parameter": "beta0_intercept", "Value": float(beta[0])})

    for r in range(R):
        rows.append({"Parameter": f"beta1_Fm3_factor{r+1}", "Value": float(beta[1 + r])})

    for r in range(R):
        rows.append({"Parameter": f"beta2_Fm2_factor{r+1}", "Value": float(beta[1 + R + r])})

    for r in range(R):
        rows.append({"Parameter": f"beta3_Fm1_factor{r+1}", "Value": float(beta[1 + 2 * R + r])})

    rows.append({"Parameter": "beta4_lagged_GDP", "Value": float(beta[-1])})
    rows.append({"Parameter": "eta2_mean", "Value": float(rep.eta2_mean)})
    rows.append({"Parameter": "tau2_mean", "Value": float(rep.tau2_mean)})

    df_params = pd.DataFrame(rows)
    csv_name = f"model_parameters_{suffix}.csv" if suffix else "model_parameters.csv"
    tex_name = f"model_parameters_{suffix}.tex" if suffix else "model_parameters.tex"
    label = f"tab:model_parameters_{suffix}" if suffix else "tab:model_parameters"

    df_params.to_csv(out_dir / csv_name, index=False)
    _to_latex_table(
        df_params,
        out_dir / tex_name,
        caption="Model parameters (bridge coefficients and variances).",
        label=label,
    )


def fig_factor_timeseries(
    rep: PosteriorPack,
    outdir: Path,
    max_factors: int = 4,
    ylim_by_factor: Optional[Dict[int, Tuple[float, float]]] = None,  # NEW
) -> None:
    """
    Plot factors. IMPORTANT: we keep the same factor indices (1..max_factors) across models
    so bay vs bay_sv are comparable one-to-one.
    """
    dates = pd.to_datetime(rep.dates_m)
    Fm = np.asarray(rep.F_mean)
    Fs = np.asarray(rep.F_sd) if rep.F_sd is not None else None

    R = Fm.shape[1]
    use = list(range(min(max_factors, R)))  # factor indices 0..K-1

    for r in use:
        fig, ax = plt.subplots(figsize=(10, 4))

        mu = Fm[:, r]
        sd = Fs[:, r] if Fs is not None else np.zeros_like(mu)
        lo = mu - 1.645 * sd
        hi = mu + 1.645 * sd

        ax.plot(dates, mu)
        ax.fill_between(dates, lo, hi, alpha=0.25)
        ax.axhline(0.0, linewidth=1)

        if ylim_by_factor is not None and (r + 1) in ylim_by_factor:
            ax.set_ylim(ylim_by_factor[r + 1])

        ax.set_xlabel("Month")
        ax.set_ylabel("Factor value")
        ax.set_title(f"{rep.model}: Factor {r+1} time series with 90% band")
        _savefig(fig, outdir / f"factor_ts_F{r+1}_{rep.model}")


def fig_loadings_topbars(
    rep: PosteriorPack,
    outdir: Path,
    top_k: int = 12,
    max_factors: int = 4,
    ylim_by_factor: Optional[Dict[int, Tuple[float, float]]] = None,  # NEW
) -> None:
    """
    Top loadings bar charts. IMPORTANT: we keep same factor indices (1..max_factors)
    across models so bay vs bay_sv are comparable.
    """
    Lam = np.asarray(rep.Lambda_mean)
    series = np.asarray(rep.series, dtype=object)

    R = Lam.shape[1]
    use = list(range(min(max_factors, R)))  # factor indices 0..K-1

    for r in use:
        vals = Lam[:, r]
        idx = np.argsort(-np.abs(vals))[: min(top_k, len(vals))]

        fig, ax = plt.subplots(figsize=(10, 5))
        y = vals[idx]
        xlab = [str(series[i]) for i in idx]

        ax.bar(range(len(idx)), y)
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels(xlab, rotation=60, ha="right")
        ax.axhline(0.0, linewidth=1)
        ax.set_ylabel("Loading")
        ax.set_title(f"{rep.model}: Top {len(idx)} loadings for factor {r+1}")

        if ylim_by_factor is not None and (r + 1) in ylim_by_factor:
            ax.set_ylim(ylim_by_factor[r + 1])

        _savefig(fig, outdir / f"loadings_top_F{r+1}_{rep.model}")


# ==================== MAIN ARTIFACT GENERATION (REFLOWED FOR SHARED SCALES) ====================

@dataclass
class ModelArtifacts:
    rep: PosteriorPack
    backcasted_values: Optional[np.ndarray]
    backcast_q_levels: Optional[np.ndarray]
    actual_dates_q: np.ndarray
    actual_gdp_levels: np.ndarray


def _compute_model_artifacts(run_dir: Path, model: str, vintage_name: str, tab_dir: Path) -> Optional[ModelArtifacts]:
    rep = load_final_vintage_posterior(run_dir, model, vintage_name)
    if rep is None:
        print(f"No posterior pack found for model={model}, tag={vintage_name}!")
        return None

    project_root = PROJECT_ROOT
    vintage_path = project_root / DATA_DIR_REL / rep.vintage_file
    spec_path = project_root / SPEC_FILE_REL

    D = build_zhang_data(
        vintage_file=vintage_path,
        spec_file=spec_path,
        gdp_series_id="Real GDP",
        sample_start="2002-01-01",
    )

    backcasted_values = None
    backcast_q_levels = None

    actual_dates_q, actual_gdp_levels = extract_actual_gdp(D)

    # Backcast + interpolate to quarters + write CSV
    try:
        backcasted_values = backcast(D, rep)

        dates_m_pd = pd.to_datetime(rep.dates_m)
        from scipy import interpolate

        backcast_interp = interpolate.interp1d(
            dates_m_pd.to_numpy().astype(np.int64) // 10**9,
            backcasted_values,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        q_times = actual_dates_q.to_numpy().astype(np.int64) // 10**9
        backcast_q_levels = backcast_interp(q_times)

        comparison_df = pd.DataFrame({
            "Date": actual_dates_q,
            "Actual GDP (levels)": actual_gdp_levels,
            "Backcast GDP (levels)": backcast_q_levels,
        })
        comparison_df.to_csv(tab_dir / f"backcast_vs_actual_levels_{model}.csv", index=False)

        print(f"✓ [{model}] Backcast vs Actual GDP (levels) successful!")
    except Exception as e:
        print(f"[{model}] Backcast failed: {e}")

    return ModelArtifacts(
        rep=rep,
        backcasted_values=backcasted_values,
        backcast_q_levels=backcast_q_levels,
        actual_dates_q=actual_dates_q,
        actual_gdp_levels=actual_gdp_levels,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--run_dir", type=str, required=True,
                    help="Path to outputs/<run_id>")

    ap.add_argument("--max_factor_plots", type=int, default=4)
    ap.add_argument("--top_loadings_k", type=int, default=12)
    ap.add_argument("--sv_top_n", type=int, default=6)

    ap.add_argument("--actual_vintage", type=str,
                    default="data/NL/2025-11-14.xlsx",
                    help="Path to most complete vintage used as 'actual' GDP")
    ap.add_argument("--spec_file", type=str,
                    default="data/Spec_NL.xlsx",
                    help="Path to Spec file")
    ap.add_argument("--gdp_series_id", type=str,
                    default="Real GDP")
    ap.add_argument("--sample_start", type=str,
                    default="2002-01-01")

    ap.add_argument("--display_units", type=str,
                    default="annualized",
                    choices=["qoq", "annualized"],
                    help="Units for plots/tables.")

    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    fig_dir = run_dir / "figures"
    tab_dir = run_dir / "tables"
    _ensure_dir(fig_dir)
    _ensure_dir(tab_dir)

    # hard-code tag; matches your run
    tag = "vintage=2025-11-14 00:00:00"

    # ---- Pass 1: load both models + compute backcasts (so we can set shared y-lims)
    artifacts: Dict[str, ModelArtifacts] = {}
    for model in ["bay", "bay_sv"]:
        art = _compute_model_artifacts(run_dir, model, tag, tab_dir)
        if art is not None:
            artifacts[model] = art

    if "bay" not in artifacts or "bay_sv" not in artifacts:
        print("WARNING: Could not load both bay and bay_sv; shared scaling may be incomplete.")

    # ---- Compute shared scales (union across bay and bay_sv)
    # Backcast plot (monthly backcasted_values + quarterly actual points)
    backcast_ylim = None
    if "bay" in artifacts and "bay_sv" in artifacts:
        vals = []
        for m in ["bay", "bay_sv"]:
            if artifacts[m].backcasted_values is not None:
                vals.append(np.asarray(artifacts[m].backcasted_values))
            # actual series (whatever you plot as "actual" here)
            if artifacts[m].actual_gdp_levels is not None:
                vals.append(np.asarray(artifacts[m].actual_gdp_levels))
        backcast_ylim = _common_ylim(vals)

    # Factor time series: same factor index => same y-limits
    factor_ylim_by_factor: Dict[int, Tuple[float, float]] = {}
    if "bay" in artifacts and "bay_sv" in artifacts:
        F_b = np.asarray(artifacts["bay"].rep.F_mean)
        F_s = np.asarray(artifacts["bay_sv"].rep.F_mean)
        R = min(F_b.shape[1], F_s.shape[1], args.max_factor_plots)
        for k in range(1, R + 1):  # factor number (1-based)
            fb = F_b[:, k - 1]
            fs = F_s[:, k - 1]
            ylim = _common_ylim([fb, fs])
            if ylim is not None:
                factor_ylim_by_factor[k] = ylim

    # Loadings bars: same factor index => same y-limits
    loadings_ylim_by_factor: Dict[int, Tuple[float, float]] = {}
    if "bay" in artifacts and "bay_sv" in artifacts:
        L_b = np.asarray(artifacts["bay"].rep.Lambda_mean)
        L_s = np.asarray(artifacts["bay_sv"].rep.Lambda_mean)
        R = min(L_b.shape[1], L_s.shape[1], args.max_factor_plots)
        for k in range(1, R + 1):
            lb = L_b[:, k - 1]
            ls = L_s[:, k - 1]
            ylim = _common_ylim([lb, ls])
            if ylim is not None:
                loadings_ylim_by_factor[k] = ylim

    # SV aggregate: shared y-limits across bay vs bay_sv
    sv_agg_ylim = None
    sv_sel_ylim = None
    if "bay" in artifacts and "bay_sv" in artifacts:
        rep_b = artifacts["bay"].rep
        rep_s = artifacts["bay_sv"].rep
        if rep_b.omega_mean is not None and rep_s.omega_mean is not None:
            Om_b = np.asarray(rep_b.omega_mean)
            Om_s = np.asarray(rep_s.omega_mean)
            med_b = np.nanmedian(Om_b, axis=1)
            p10_b = np.nanpercentile(Om_b, 10, axis=1)
            p90_b = np.nanpercentile(Om_b, 90, axis=1)
            med_s = np.nanmedian(Om_s, axis=1)
            p10_s = np.nanpercentile(Om_s, 10, axis=1)
            p90_s = np.nanpercentile(Om_s, 90, axis=1)
            sv_agg_ylim = _common_ylim([med_b, p10_b, p90_b, med_s, p10_s, p90_s])

            # For selected-series panels: use a single common ylim across BOTH models' omega series
            # (so the multi-panel figures are comparable in overall scale).
            sv_sel_ylim = _common_ylim([Om_b, Om_s])

    # ---- Pass 2: produce plots and tables, injecting shared scales
    for model in ["bay", "bay_sv"]:
        if model not in artifacts:
            continue
        rep = artifacts[model].rep

        print(f"[{model}] beta_mean shape:", rep.beta_mean.shape)
        print(f"[{model}] beta_mean:", rep.beta_mean)

        # Backcast vs Actual GDP Growth
        if artifacts[model].backcasted_values is not None:
            fig_backcast_growth(
                artifacts[model].backcasted_values,
                rep.dates_m,
                artifacts[model].actual_gdp_levels,
                artifacts[model].actual_dates_q,
                fig_dir,
                suffix=model,
                ylim=backcast_ylim,  # SHARED SCALE
            )

        # Factor and loadings plots (shared scales by factor index)
        fig_factor_timeseries(
            rep,
            fig_dir,
            max_factors=args.max_factor_plots,
            ylim_by_factor=factor_ylim_by_factor if factor_ylim_by_factor else None,
        )
        fig_loadings_topbars(
            rep,
            fig_dir,
            top_k=args.top_loadings_k,
            max_factors=args.max_factor_plots,
            ylim_by_factor=loadings_ylim_by_factor if loadings_ylim_by_factor else None,
        )

        # Stochastic volatility: aggregate index + selected series + summary table (shared scales)
        if rep.omega_mean is not None:
            fig_sv_aggregate(
                rep,
                fig_dir,
                title_suffix="Dutch indicators, 2002–2025",
                ylim=sv_agg_ylim,  # SHARED SCALE
            )
            fig_sv_selected_series(
                rep,
                fig_dir,
                top_n=args.sv_top_n,
                ylim=sv_sel_ylim,  # SHARED SCALE
            )
            sv_summary_table(rep, tab_dir)

        # Parameter table
        generate_parameter_table(rep, tab_dir, suffix=model)

    print(f"Figures saved to {fig_dir}")
    print(f"Tables saved to {tab_dir}")


if __name__ == "__main__":
    main()