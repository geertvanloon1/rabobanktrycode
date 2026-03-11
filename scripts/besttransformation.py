import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from zhangnowcast.data.load_spec import load_spec
from zhangnowcast.data.load_data import read_data_excel


# =========================
# CONFIG
# =========================
VINTAGE_FILE = Path("data/NL/2025-11-14.xlsx")
SPEC_FILE = Path("data/Spec_NL.xlsx")

# Evaluate monthly + quarterly predictors (exclude GDP in a filter if you want)
ONLY_FREQ = {"m", "q"}

# Candidate transforms supported by your transform logic
CANDIDATES = ["lin", "log", "chg", "ch1", "pch", "pc1", "pca", "cch", "cca"]

# Minimum usable points AFTER compressing to the native frequency
MIN_OBS_M = 80          # months
MIN_OBS_Q = 30          # quarters (adjust if you want stricter)

OUT_CSV = Path("outputs/best_transforms_stationarity_freqaware.csv")


# =========================
# Optional: ADF/KPSS tests
# =========================
try:
    from statsmodels.tsa.stattools import adfuller, kpss  # type: ignore
    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False


# =========================
# Helpers
# =========================
def _safe_float(x: np.ndarray) -> np.ndarray:
    x = x.astype(float, copy=False)
    x[~np.isfinite(x)] = np.nan
    return x


def align_to_spec_columns(
    Z: np.ndarray, mnem: List[str], spec_series: List[str]
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Align raw Z columns to spec.SeriesID order.
    If a SeriesID is missing from the vintage mnemonic list, keep it as all-NaN.
    """
    mnem_arr = np.array(mnem, dtype=object)
    Z = _safe_float(Z)

    out = np.full((Z.shape[0], len(spec_series)), np.nan, dtype=float)
    found = np.zeros(len(spec_series), dtype=bool)

    idx_map: Dict[str, int] = {str(k): int(i) for i, k in enumerate(mnem_arr)}

    for j, sid in enumerate(spec_series):
        if sid in idx_map:
            out[:, j] = Z[:, idx_map[sid]]
            found[j] = True

    return out, list(spec_series), found


def _step_for_freq(freq: str) -> int:
    freq = freq.lower().strip()
    if freq == "m":
        return 1
    if freq == "q":
        return 3
    return 0


def compress_to_native_frequency(z: np.ndarray, freq: str) -> np.ndarray:
    """
    Convert series on the monthly grid to a contiguous series at its native frequency.

    - freq="m": return z as-is (monthly).
    - freq="q": take values every 3 months, aligned with your transformation logic:
        t1 = step-1 => for q: t1=2 -> indices 2,5,8,... (quarter-end month if aligned)
    """
    z = z.astype(float, copy=False)
    step = _step_for_freq(freq)
    if step == 0:
        return np.asarray([], dtype=float)

    t1 = step - 1
    return z[t1::step].copy()


def apply_transform_native(z_native: np.ndarray, freq: str, formula: str) -> np.ndarray:
    """
    Apply transforms on the *compressed* (native frequency) series.

    Important: once compressed, step=1 always (contiguous in its own time).
    But we keep formula definitions consistent:
      - "ch1"/"pc1" = 12-month difference for monthly, 4-quarter difference for quarterly.
    """
    z = z_native.astype(float, copy=True)
    T = z.shape[0]
    x = np.full(T, np.nan, dtype=float)

    freq = freq.lower().strip()
    if freq not in ("m", "q"):
        return x

    # In compressed space, observations are contiguous: step_native = 1
    step_native = 1

    # For annual diffs in native units:
    # - monthly: 12 lags
    # - quarterly: 4 lags
    annual_lag = 12 if freq == "m" else 4

    if formula == "lin":
        x[:] = z

    elif formula == "log":
        with np.errstate(invalid="ignore", divide="ignore"):
            x[:] = np.log(z)

    elif formula == "chg":
        prev = np.arange(0, T - step_native)
        nxt = np.arange(step_native, T)
        x[nxt] = z[nxt] - z[prev]

    elif formula == "ch1":
        if T > annual_lag:
            x[annual_lag:] = z[annual_lag:] - z[:-annual_lag]

    elif formula == "pch":
        prev = np.arange(0, T - step_native)
        nxt = np.arange(step_native, T)
        with np.errstate(invalid="ignore", divide="ignore"):
            x[nxt] = 100.0 * (z[nxt] / z[prev] - 1.0)

    elif formula == "pc1":
        if T > annual_lag:
            with np.errstate(invalid="ignore", divide="ignore"):
                x[annual_lag:] = 100.0 * (z[annual_lag:] / z[:-annual_lag] - 1.0)

    elif formula == "pca":
        # annualized growth from one period to next
        # monthly: n=1/12; quarterly: n=1/4
        n = (1.0 / 12.0) if freq == "m" else (1.0 / 4.0)
        prev = np.arange(0, T - step_native)
        nxt = np.arange(step_native, T)
        with np.errstate(invalid="ignore", divide="ignore"):
            x[nxt] = 100.0 * ((z[nxt] / z[prev]) ** (1.0 / n) - 1.0)

    elif formula == "cch":
        prev = np.arange(0, T - step_native)
        nxt = np.arange(step_native, T)
        valid = (z[nxt] > 0) & (z[prev] > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            x[nxt[valid]] = 100.0 * (np.log(z[nxt[valid]]) - np.log(z[prev[valid]]))

    elif formula == "cca":
        prev = np.arange(0, T - step_native)
        nxt = np.arange(step_native, T)
        # monthly: 1200, quarterly: 400
        annual_factor = 1200.0 if freq == "m" else 400.0
        valid = (z[nxt] > 0) & (z[prev] > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            x[nxt[valid]] = annual_factor * (np.log(z[nxt[valid]]) - np.log(z[prev[valid]]))

    return x


def stationarity_score(y: np.ndarray, min_obs: int) -> Tuple[float, float, float, int]:
    """
    Returns (score, adf_p, kpss_p, n_obs). Lower score is better.
    ADF: want small p (reject unit root)
    KPSS: want large p (fail to reject stationarity)
    """
    yy = pd.Series(y).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    n = int(len(yy))
    if n < min_obs:
        return float("inf"), np.nan, np.nan, n

    if float(np.nanstd(yy.values)) < 1e-12:
        return float("inf"), 1.0, 0.0, n

    if not HAVE_STATSMODELS:
        ac1 = float(pd.Series(yy.values).autocorr(lag=1))
        return abs(ac1), np.nan, np.nan, n

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            adf_p = float(adfuller(yy.values, autolag="AIC")[1])
        except Exception:
            adf_p = np.nan
        try:
            kpss_p = float(kpss(yy.values, regression="c", nlags="auto")[1])
        except Exception:
            kpss_p = np.nan

    adf_term = adf_p if not np.isnan(adf_p) else 0.5
    kpss_term = (1.0 - kpss_p) if not np.isnan(kpss_p) else 0.5
    score = adf_term + kpss_term
    return float(score), adf_p, kpss_p, n


def main():
    if not VINTAGE_FILE.exists():
        raise FileNotFoundError(f"Vintage file not found: {VINTAGE_FILE}")

    print(f"Using vintage: {VINTAGE_FILE}")

    spec = load_spec(SPEC_FILE)
    Z_raw, Time, mnem = read_data_excel(VINTAGE_FILE)

    Z_aligned, sids, found_mask = align_to_spec_columns(Z_raw, mnem, spec.SeriesID)

    rows = []
    for j, sid in enumerate(sids):
        freq = str(spec.Frequency[j]).lower().strip()
        if freq not in ONLY_FREQ:
            continue

        found = bool(found_mask[j])
        zcol_monthly_grid = Z_aligned[:, j]

        if not found:
            rows.append({
                "SeriesID": sid,
                "SeriesName": spec.SeriesName[j],
                "Frequency": freq,
                "status": "NOT_FOUND_IN_VINTAGE",
                "best_transform": "",
                "best_score": np.nan,
                "best_adf_p": np.nan,
                "best_kpss_p": np.nan,
                "best_n_obs": 0,
                "current_spec_transform": spec.Transformation[j],
            })
            continue

        # Compress to native frequency BEFORE scoring/transforming
        z_native = compress_to_native_frequency(zcol_monthly_grid, freq=freq)
        n_raw_native = int(pd.Series(z_native).replace([np.inf, -np.inf], np.nan).notna().sum())

        min_obs = MIN_OBS_M if freq == "m" else MIN_OBS_Q
        if n_raw_native < min_obs:
            rows.append({
                "SeriesID": sid,
                "SeriesName": spec.SeriesName[j],
                "Frequency": freq,
                "status": f"TOO_SHORT_RAW_NATIVE(n={n_raw_native})",
                "best_transform": "",
                "best_score": np.nan,
                "best_adf_p": np.nan,
                "best_kpss_p": np.nan,
                "best_n_obs": n_raw_native,
                "current_spec_transform": spec.Transformation[j],
            })
            continue

        best = None
        for tcode in CANDIDATES:
            y = apply_transform_native(z_native, freq=freq, formula=tcode)
            score, adf_p, kpss_p, n_obs = stationarity_score(y, min_obs=min_obs)

            cand = {
                "tcode": tcode,
                "score": score,
                "adf_p": adf_p,
                "kpss_p": kpss_p,
                "n_obs": n_obs,
            }
            if best is None or cand["score"] < best["score"]:
                best = cand

        if best is None or not np.isfinite(best["score"]):
            rows.append({
                "SeriesID": sid,
                "SeriesName": spec.SeriesName[j],
                "Frequency": freq,
                "status": "NO_VALID_TRANSFORM",
                "best_transform": "",
                "best_score": np.nan,
                "best_adf_p": np.nan,
                "best_kpss_p": np.nan,
                "best_n_obs": 0,
                "current_spec_transform": spec.Transformation[j],
            })
        else:
            rows.append({
                "SeriesID": sid,
                "SeriesName": spec.SeriesName[j],
                "Frequency": freq,
                "status": "OK",
                "best_transform": best["tcode"],
                "best_score": best["score"],
                "best_adf_p": best["adf_p"],
                "best_kpss_p": best["kpss_p"],
                "best_n_obs": best["n_obs"],
                "current_spec_transform": spec.Transformation[j],
            })

    out = pd.DataFrame(rows).sort_values(["status", "best_score"], na_position="last")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}")

    if not HAVE_STATSMODELS:
        print("Note: statsmodels not installed; ADF/KPSS are NaN and a fallback heuristic is used.")
        print("Install: pip install statsmodels")


if __name__ == "__main__":
    main()