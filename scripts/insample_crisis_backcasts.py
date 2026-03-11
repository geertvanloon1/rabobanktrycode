from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zhangnowcast.data.data import build_zhang_data


# python3 scripts/insample_crisis_backcasts.py --run_dir outputs/20260301_135623_99d2920a
# paths relative to project root (same as runner)
DATA_DIR_REL = Path("data/NL")
SPEC_FILE_REL = Path("data/Spec_NL.xlsx")


def extract_actual_gdp(D) -> tuple[np.ndarray, np.ndarray]:
    """Extract quarterly GDP series (whatever units y_q is in) from ZhangData."""
    y = None
    for name in ("y_q", "Y_q", "gdp_q", "GDP_q"):
        if hasattr(D, name):
            y = np.asarray(getattr(D, name), dtype=float)
            break

    if y is None or not hasattr(D, "dates_q"):
        return np.array([]), np.array([])

    dates_q = pd.to_datetime(np.asarray(D.dates_q))
    valid_idx = ~np.isnan(y)
    return dates_q[valid_idx], y[valid_idx]


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


def _coerce_dt(x) -> pd.Timestamp:
    return pd.to_datetime(x).to_period("M").to_timestamp()


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
    beta_mean: np.ndarray   # (1+3R+1,)
    dates_m: np.ndarray     # (T,)


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
        beta_mean=get("beta_mean"),
        dates_m=get("dates_m"),
    )


def load_final_vintage_posterior(run_dir: Path, model: str, vintage_name: str) -> Optional[PosteriorPack]:
    model_dir = run_dir / "posterior" / "per_run" / model / vintage_name
    print(f"Debug: Trying to access directory: {model_dir}")
    if not model_dir.exists():
        print(f"Error: Directory does not exist: {model_dir}")
        return None
    return load_posterior_pack(model, model_dir)


def backcast(rep: PosteriorPack) -> np.ndarray:
    """
    Backcasting using factor model: GDP ~ intercept + current factors + lags.
    beta_mean shape (1 + 3R + 1,): [intercept, F_t, F_{t-1}, F_{t-2}, F0_{t-3}]
    """
    F_mean = np.asarray(rep.F_mean)  # (T, R)
    R = F_mean.shape[1]
    T = F_mean.shape[0]

    beta = np.asarray(rep.beta_mean).ravel()
    print(f"Debug: R={R}, beta_mean.shape={beta.shape}")

    X = np.zeros((T, len(beta)))
    X[:, 0] = 1.0  # intercept

    for t in range(T):
        X[t, 1:1 + R] = F_mean[t, :]
        if t >= 1:
            X[t, 1 + R:1 + 2 * R] = F_mean[t - 1, :]
        if t >= 2:
            X[t, 1 + 2 * R:1 + 3 * R] = F_mean[t - 2, :]
        if t >= 3 and len(beta) > 1 + 3 * R:
            X[t, -1] = F_mean[t - 3, 0]

    return X @ beta


def fig_backcast_two(backcast_bay, backcast_baysv, dates_m,
                     actual_series, actual_dates_q, outdir: Path) -> None:
    dates_m_pd = pd.to_datetime(dates_m)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(dates_m_pd, backcast_bay, label="DFM (bay)", color="blue", linewidth=2.0)
    ax.plot(dates_m_pd, backcast_baysv, label="DFM (bay_sv)", color="deepskyblue", linewidth=2.0)
    ax.plot(actual_dates_q, actual_series, "k--o", label="Actual", markersize=6)

    ax.set_xlabel("Date")
    ax.set_ylabel("Real GDP growth (annualized QoQ, %)")
    ax.set_title("Factor model backcasts vs actual GDP (bay & bay_sv)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _savefig(fig, outdir / "backcast_vs_actual_bay_and_baysv")


def plot_window(dates_m, back_bay_m, back_baysv_m,
                actual_dates_q, actual_gdp,
                start: str, end: str,
                outpath: Path, title: str) -> None:
    dates_m = pd.to_datetime(dates_m)
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    mask_m = (dates_m >= start_dt) & (dates_m <= end_dt)
    mask_q = (actual_dates_q >= start_dt) & (actual_dates_q <= end_dt)

    print(f"{title}: monthly points={mask_m.sum()}, quarterly points={mask_q.sum()}")

    if mask_m.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dates_m[mask_m], back_bay_m[mask_m], label="DFM (bay)", color="blue")
    ax.plot(dates_m[mask_m], back_baysv_m[mask_m], label="DFM (bay_sv)", color="deepskyblue")
    ax.plot(actual_dates_q[mask_q], actual_gdp[mask_q], "k--o", label="Actual", markersize=5)

    ax.set_xlabel("Date")
    ax.set_ylabel("GDP (QoQ%\ annualized)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    _savefig(fig, outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True,
                    help="Path to outputs/<run_id>")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    fig_dir = run_dir / "figures"
    _ensure_dir(fig_dir)

    tag = "vintage=2025-11-14 00:00:00"

    rep_bay = load_final_vintage_posterior(run_dir, "bay", tag)
    rep_baysv = load_final_vintage_posterior(run_dir, "bay_sv", tag)
    if rep_bay is None or rep_baysv is None:
        print("Missing bay or bay_sv posterior for this vintage!")
        return

    # Load data only to get actual GDP (quarterly)
    project_root = PROJECT_ROOT
    vintage_path = project_root / DATA_DIR_REL / rep_bay.vintage_file
    spec_path = project_root / SPEC_FILE_REL

    D = build_zhang_data(
        vintage_file=vintage_path,
        spec_file=spec_path,
        gdp_series_id="Real GDP",
        sample_start="2002-01-01",
    )
    actual_dates_q, actual_gdp = extract_actual_gdp(D)

    # Backcasts (monthly) from posterior packs
    back_bay_m = backcast(rep_bay)
    back_baysv_m = backcast(rep_baysv)

    # Full‑sample plot
    fig_backcast_two(back_bay_m, back_baysv_m, rep_bay.dates_m,
                     actual_gdp, actual_dates_q, fig_dir)

    # Three zoomed windows
    plot_window(rep_bay.dates_m, back_bay_m, back_baysv_m,
                actual_dates_q, actual_gdp,
                "2008-01-01", "2010-12-31",
                fig_dir / "backcasts_GR_2008_2010",
                "Backcasts 2008–2010")

    plot_window(rep_bay.dates_m, back_bay_m, back_baysv_m,
                actual_dates_q, actual_gdp,
                "2013-01-01", "2015-12-31",
                fig_dir / "backcasts_stable_2013_2015",
                "Backcasts 2013–2015")

    plot_window(rep_bay.dates_m, back_bay_m, back_baysv_m,
                actual_dates_q, actual_gdp,
                "2019-01-01", "2021-12-31",
                fig_dir / "backcasts_Covid_2019_2021",
                "Backcasts 2019–2021")

    print(f"Figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
