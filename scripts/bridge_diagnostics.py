from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zhangnowcast.data.data import build_zhang_data

# python3 scripts/bridge_diagnostics.py --run_dir outputs/20260301_135623_99d2920a

# --- basic helpers ---

def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _coerce_dt(x) -> pd.Timestamp:
    return pd.to_datetime(x).to_period("M").to_timestamp()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(fig, outbase: Path) -> None:
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=200)
    plt.close(fig)


# --- posterior container & loader (minimal fields) ---

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

    F_mean: np.ndarray      # (T, R)
    beta_mean: np.ndarray   # (1+3R+1,)
    dates_m: np.ndarray     # (T,)
    z_mean: np.ndarray      # (R,)


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

    zfile = np.load(npz_path, allow_pickle=True)

    def get(name, default=None):
        return zfile[name] if name in zfile.files else default

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
        z_mean=get("z_mean"),
    )


def load_final_vintage_posterior(run_dir: Path, model: str, vintage_name: str) -> Optional[PosteriorPack]:
    model_dir = run_dir / "posterior" / "per_run" / model / vintage_name
    print(f"[{model}] Trying to access {model_dir}")
    if not model_dir.exists():
        print(f"[{model}] Error: Directory does not exist: {model_dir}")
        return None
    return load_posterior_pack(model, model_dir)


# --- data access: y_q and alignment helpers ---

@dataclass
class ZhangDataMinimal:
    y_q: np.ndarray
    dates_q: np.ndarray
    quarter_of_month: np.ndarray
    month_pos_in_quarter: np.ndarray


def build_zhang_data_minimal(vintage_file: Path,
                             spec_file: Path,
                             gdp_series_id: str,
                             sample_start: str) -> ZhangDataMinimal:
    from zhangnowcast.data.data import build_zhang_data as _build
    D = _build(
        vintage_file=vintage_file,
        spec_file=spec_file,
        gdp_series_id=gdp_series_id,
        sample_start=sample_start,
    )

    y = None
    for name in ("y_q", "Y_q", "gdp_q", "GDP_q"):
        if hasattr(D, name):
            y = np.asarray(getattr(D, name), dtype=float)
            break
    if y is None:
        raise RuntimeError("No y_q / gdp_q found in ZhangData.")

    dates_q = pd.to_datetime(np.asarray(D.dates_q))
    valid = ~np.isnan(y)
    y_q = y[valid]
    dates_q = dates_q[valid]

    q_of_m = np.asarray(D.quarter_of_month)
    mpq = np.asarray(D.month_pos_in_quarter)

    return ZhangDataMinimal(
        y_q=y_q,
        dates_q=dates_q,
        quarter_of_month=q_of_m,
        month_pos_in_quarter=mpq,
    )


# --- core: reconstruct quarterly X and fitted y_q ---

def bridge_quarterly_backcast(D: ZhangDataMinimal,
                              rep: PosteriorPack) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Rebuild the quarterly regression data (yreg, Xreg) exactly as in sample_beta_eta2,
    but using posterior means (F_mean, z_mean, beta_mean).

    Returns:
      yreg  : (n_obs,) actual y_q used in the regression
      yhat  : (n_obs,) fitted y_q = Xreg @ beta_mean
      idx_k : list of quarter indices k (matching positions in D.y_q) used in yreg
    """
    y_q = np.asarray(D.y_q, float)
    F = np.asarray(rep.F_mean, float)            # (T, R)
    z = np.asarray(rep.z_mean, float)           # (R,)

    q_of_m = np.asarray(D.quarter_of_month)     # (T,)
    mpq = np.asarray(D.month_pos_in_quarter)    # (T,)

    T, R = F.shape
    K = len(y_q)
    Z = z.astype(float)

    rows = []
    targ = []
    idx_k = []

    for k in range(1, K):
        ts = np.where(q_of_m == k)[0]
        if ts.size < 3:
            continue
        ts = np.sort(ts)

        t3_c = ts[mpq[ts] == 3]
        t2_c = ts[mpq[ts] == 2]
        t1_c = ts[mpq[ts] == 1]

        if t3_c.size > 0 and t2_c.size > 0 and t1_c.size > 0:
            t3 = int(t3_c[-1])
            t2 = int(t2_c[-1])
            t1 = int(t1_c[-1])
        else:
            t1, t2, t3 = map(int, ts[-3:])

        if not (0 <= t1 < T and 0 <= t2 < T and 0 <= t3 < T):
            continue

        xrow = np.r_[
            1.0,
            Z * F[t3, :],
            Z * F[t2, :],
            Z * F[t1, :],
            y_q[k - 1],
        ]
        rows.append(xrow)
        targ.append(y_q[k])
        idx_k.append(k)

    if not rows:
        return np.array([]), np.array([]), []

    Xreg = np.vstack(rows)
    yreg = np.asarray(targ, float)
    beta = np.asarray(rep.beta_mean).ravel()

    yhat = Xreg @ beta
    return yreg, yhat, idx_k


# --- diagnostics for one model ---

def run_diagnostics_for_model(run_dir: Path,
                              model: str,
                              tag: str,
                              out_dir: Path,
                              data_dir_rel: Path,
                              spec_file_rel: Path,
                              gdp_series_id: str,
                              sample_start: str) -> pd.DataFrame:
    rep = load_final_vintage_posterior(run_dir, model, tag)
    if rep is None:
        print(f"[{model}] No posterior pack; skipping.")
        return pd.DataFrame()

    project_root = PROJECT_ROOT
    vintage_path = project_root / data_dir_rel / rep.vintage_file
    spec_path = project_root / spec_file_rel

    Dmin = build_zhang_data_minimal(
        vintage_file=vintage_path,
        spec_file=spec_path,
        gdp_series_id=gdp_series_id,
        sample_start=sample_start,
    )

    yreg, yhat, idx_k = bridge_quarterly_backcast(Dmin, rep)
    if yreg.size == 0:
        print(f"[{model}] No usable quarters in bridge_quarterly_backcast; skipping.")
        return pd.DataFrame()

    dates_q_used = Dmin.dates_q[idx_k]
    resid = yreg - yhat

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "Date": dates_q_used,
        "Actual_y": yreg,
        "Fitted_y": yhat,
        "Residual": resid,
    })
    df.to_csv(out_dir / f"bridge_residuals_quarterly_{model}.csv", index=False)

    # residual TS
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.plot(dates_q_used, resid, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Residual (model y units)")
    ax.set_title(f"{model}: Bridge residuals (quarterly)")
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir / f"bridge_residuals_quarterly_ts_{model}")

    # histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(resid, bins=20, color="grey", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Residual (model y units)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{model}: Bridge residuals histogram (quarterly)")
    _savefig(fig, out_dir / f"bridge_residuals_quarterly_hist_{model}")

    # ACF
    try:
        from statsmodels.graphics.tsaplots import plot_acf
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_acf(resid, lags=8, zero=False, ax=ax)
        ax.set_title(f"{model}: Bridge residuals ACF (quarterly)")
        _savefig(fig, out_dir / f"bridge_residuals_quarterly_acf_{model}")
    except Exception as e:
        print(f"[{model}] Could not plot ACF (maybe statsmodels missing): {e}")

    # detailed summary stats
    from scipy.stats import skew, kurtosis, jarque_bera
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except Exception:
        acorr_ljungbox = None

    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    denom = np.sum((yreg - np.mean(yreg)) ** 2)
    r2 = float(1.0 - np.sum(resid ** 2) / denom) if denom > 0 else np.nan

    mean_resid = float(np.mean(resid))
    sd_resid = float(np.std(resid, ddof=1))
    skew_resid = float(skew(resid, bias=False)) if resid.size > 2 else np.nan
    kurt_resid = float(kurtosis(resid, fisher=True, bias=False)) if resid.size > 3 else np.nan

    jb_stat = np.nan
    jb_p = np.nan
    try:
        jb_stat, jb_p = jarque_bera(resid)
        jb_stat = float(jb_stat)
        jb_p = float(jb_p)
    except Exception:
        pass

    lbq4_stat = lbq4_p = lbq8_stat = lbq8_p = np.nan
    if acorr_ljungbox is not None:
        lb = acorr_ljungbox(resid, lags=[4, 8], return_df=True)
        # rows indexed by lags, columns: 'lb_stat', 'lb_pvalue'
        lbq4_stat = float(lb.loc[4, "lb_stat"])
        lbq4_p    = float(lb.loc[4, "lb_pvalue"])
        lbq8_stat = float(lb.loc[8, "lb_stat"])
        lbq8_p    = float(lb.loc[8, "lb_pvalue"])



    summary_row = {
        "model": model,
        "n_obs": int(len(resid)),
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "mean_resid": mean_resid,
        "sd_resid": sd_resid,
        "skew_resid": skew_resid,
        "kurt_resid": kurt_resid,
        "LBQ4_stat": lbq4_stat,
        "LBQ4_p": lbq4_p,
        "LBQ8_stat": lbq8_stat,
        "LBQ8_p": lbq8_p,
        "JB_stat": jb_stat,
        "JB_p": jb_p,
    }

    summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(out_dir / f"bridge_residuals_quarterly_summary_{model}.csv", index=False)

    print(f"[{model}] Quarterly bridge diagnostics written to {out_dir}")
    return summary_df


# --- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True,
                    help="Path to outputs/<run_id>")
    ap.add_argument("--tag", type=str,
                    default="vintage=2025-11-14 00:00:00",
                    help="Subrun tag, e.g. vintage=YYYY-MM-DD 00:00:00")
    ap.add_argument("--gdp_series_id", type=str,
                    default="Real GDP")
    ap.add_argument("--sample_start", type=str,
                    default="2002-01-01")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    out_dir = run_dir / "bridge_diagnostics_quarterly"
    _ensure_dir(out_dir)

    all_summaries = []

    for model in ["bay", "bay_sv"]:
        summary_df = run_diagnostics_for_model(
            run_dir=run_dir,
            model=model,
            tag=args.tag,
            out_dir=out_dir,
            data_dir_rel=Path("data/NL"),
            spec_file_rel=Path("data/Spec_NL.xlsx"),
            gdp_series_id=args.gdp_series_id,
            sample_start=args.sample_start,
        )
        if not summary_df.empty:
            all_summaries.append(summary_df)

    if all_summaries:
        df_all = pd.concat(all_summaries, ignore_index=True)
        df_all.to_csv(out_dir / "bridge_residuals_quarterly_summary_all.csv", index=False)

    print(f"All done. Diagnostics saved under {out_dir}")


if __name__ == "__main__":
    main()
