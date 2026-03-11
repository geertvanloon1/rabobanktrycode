from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# import your existing loaders
from zhangnowcast.data.load_spec import load_spec, Spec
from zhangnowcast.data.load_data import load_data


@dataclass
class ZhangData:
    # Monthly panel for factor model
    X_m: np.ndarray                 # (T, nM) monthly panel (transformed), NaNs allowed
    dates_m: np.ndarray             # (T,) datetime64[ns]
    series_m: List[str]             # monthly series IDs in X_m order

    # Quarterly GDP target
    y_q: np.ndarray                 # (K,) GDP (transformed), quarterly
    dates_q: np.ndarray             # (K,) quarter-end dates (datetime64[ns])

    # Alignment helpers
    quarter_of_month: np.ndarray    # (T,) int in [0..K] mapping month t -> quarter index
    month_pos_in_quarter: np.ndarray  # (T,) in {1,2,3}

    # Missingness structure for Eq (12) indicator-matrix idea:
    # obs_idx[t] gives which monthly series are observed at month t
    obs_idx: List[np.ndarray]       # length T, each array of indices into monthly dimension


def _month_pos_in_quarter(dates_m: np.ndarray) -> np.ndarray:
    """
    Returns 1,2,3 based on calendar month: Jan/Apr/Jul/Oct -> 1; Feb/May/Aug/Nov -> 2; else 3.
    This matches quarter month positions for standard quarterly aggregation.
    """
    dt = pd.to_datetime(dates_m)
    m = dt.month.to_numpy()
    # month position: 1 if (m-1)%3==0, 2 if ==1, 3 if ==2
    return ((m - 1) % 3) + 1


def _build_quarter_mapping(dates_m: np.ndarray, dates_q: np.ndarray) -> np.ndarray:
    """
    Map each monthly date to the index k of the most recent quarter-end in dates_q that is >= that month’s quarter.
    We assume dates_q are quarter-end timestamps that appear in the dataset.

    Returns quarter_of_month[t] in [0..K-1] for months that belong to a known quarter,
    and -1 if the month is outside the y_q range (e.g., early months before first quarter observation).
    """
    dm = pd.to_datetime(dates_m)
    dq = pd.to_datetime(dates_q)

    # Compute quarter period for each monthly date and quarter date
    qm = dm.to_period("Q")
    qq = dq.to_period("Q")

    # Map quarter period -> index in dates_q
    q_to_idx: Dict[pd.Period, int] = {}
    for k, qper in enumerate(qq):
        # If duplicates exist (rare), keep the first
        if qper not in q_to_idx:
            q_to_idx[qper] = k

    out = np.full(len(dm), -1, dtype=int)
    for t, qper in enumerate(qm):
        if qper in q_to_idx:
            out[t] = q_to_idx[qper]
    return out


def build_zhang_data(
    vintage_file: Union[str, Path],
    spec_file: Union[str, Path],
    gdp_series_id: str,
    sample_start: Optional[Union[str, pd.Timestamp, np.datetime64]] = None,
) -> ZhangData:
    """
    Loads one vintage and constructs the ZhangData object needed for BAY / BAY-SV.
    """
    spec = load_spec(spec_file)
    X, dates_m, _Zraw = load_data(vintage_file, spec, sample=sample_start)

    vintage_month = pd.Timestamp(Path(vintage_file).stem).normalize().replace(day=1)

    dm = pd.to_datetime(dates_m).normalize().to_period("M").to_timestamp()
    if dm.size == 0:
        raise ValueError("Empty dates_m returned by load_data")

    start_m = pd.Timestamp(dm.min()).replace(day=1)
    end_m = max(pd.Timestamp(dm.max()).replace(day=1), vintage_month)

    full_dm = pd.date_range(start=start_m, end=end_m, freq="MS")  # month starts
    if len(full_dm) != len(dm) or not np.all(full_dm.values == dm.values):
        # Reindex X to full_dm: missing months become all-NaN rows
        X_df = pd.DataFrame(X, index=dm)
        X_df = X_df.reindex(full_dm)
        X = X_df.to_numpy(dtype=float)
        dates_m = full_dm.to_numpy(dtype="datetime64[ns]")

    # Identify GDP quarterly column
    sid = np.array(spec.SeriesID, dtype=object)
    freq = np.array([f.lower() for f in spec.Frequency], dtype=object)

    if gdp_series_id not in sid:
        raise ValueError(f"gdp_series_id='{gdp_series_id}' not found in Spec.SeriesID")

    gdp_col = int(np.where(sid == gdp_series_id)[0][0])
    if freq[gdp_col] != "q":
        raise ValueError(f"Series '{gdp_series_id}' must have Frequency 'q' in spec; got '{freq[gdp_col]}'")

    # Predictors panel = monthly series + quarterly indicators (excluding GDP target)
    monthly_cols = np.where(freq == "m")[0]
    quarterly_cols = np.where(freq == "q")[0]

    panel_cols = np.concatenate([monthly_cols, quarterly_cols])

    # Exclude GDP from predictors (avoid leakage / double-use)
    panel_cols = np.array([i for i in panel_cols if i != gdp_col], dtype=int)

    X_m = X[:, panel_cols]
    series_m = [spec.SeriesID[i] for i in panel_cols]

    # Extract quarterly GDP y_q from its column: take non-NaN entries only
    y_full = X[:, gdp_col]
    gdp_mask = ~np.isnan(y_full)
    y_q = y_full[gdp_mask]

    # Quarter dates: use the monthly timestamps at which GDP is observed (usually quarter-end month)
    dates_q = dates_m[gdp_mask]

    # Build month position in quarter and quarter mapping
    mpq = _month_pos_in_quarter(dates_m)
    q_of_m = _build_quarter_mapping(dates_m, dates_q)

    # Build obs_idx per month for monthly panel (indicator-matrix equivalent)
    obs_idx: List[np.ndarray] = []
    for t in range(X_m.shape[0]):
        obs = ~np.isnan(X_m[t])
        obs_idx.append(np.where(obs)[0].astype(int))

    return ZhangData(
        X_m=X_m,
        dates_m=dates_m,
        series_m=series_m,
        y_q=y_q,
        dates_q=dates_q,
        quarter_of_month=q_of_m,
        month_pos_in_quarter=mpq.astype(int),
        obs_idx=obs_idx,
    )


def slice_rolling_10y_window(
    D: ZhangData,
    month_T: pd.Timestamp,
    window_months: int = 120,
    allow_short_start: bool = True,
) -> ZhangData:
    """
    Enforce Zhang (2022) 10-year rolling estimation window for each (T, q).

    Paper design: each nowcast at calendar month T (release q) is estimated using only
    the most recent 120 months: [T-119, ..., T]. Older data must NOT enter estimation.

    IMPORTANT: Apply this AFTER constructing X^q (ragged-edge masking), and BEFORE run_bay(...).
    """
    dm = pd.to_datetime(D.dates_m)
    month_T = pd.to_datetime(month_T).normalize().replace(day=1)

    # Find the index of the nowcast month T in the monthly sample
    idx_end_arr = np.where(dm == month_T)[0]
    if idx_end_arr.size == 0:
        raise ValueError(f"month_T={month_T.date()} not found in D.dates_m.")
    idx_end = int(idx_end_arr[0])

    # Drop any months after T (vintage files sometimes include trailing months)
    # and enforce a rolling window ending at T.
    if idx_end + 1 < 1:
        raise ValueError("Invalid idx_end computed for month_T.")

    if (idx_end + 1) < window_months and not allow_short_start:
        # Option: skip early T where <120 months exist
        raise ValueError(
            f"Only {idx_end+1} months available up to T={month_T.date()} (<{window_months})."
        )

    idx_start = max(0, idx_end - (window_months - 1))
    sl = slice(idx_start, idx_end + 1)

    # ---- Slice monthly objects ----
    X_m = D.X_m[sl, :].copy()
    dates_m = D.dates_m[sl]
    quarter_of_month = D.quarter_of_month[sl].copy()
    month_pos_in_quarter = D.month_pos_in_quarter[sl].copy()
    obs_idx = D.obs_idx[sl]  # list slice keeps alignment with X_m

    # ---- Slice quarterly objects consistently ----
    # Include one extra lag quarter (k0-1) so the GDP bridge term y_{k-1} is available
    # for the first modeled quarter in the window.
    qidx = quarter_of_month
    used = qidx[qidx >= 0]
    if used.size == 0:
        y_q = D.y_q[:0].copy()
        dates_q = D.dates_q[:0].copy()
    else:
        k0_used = int(np.min(used))
        k1_used = int(np.max(used))

        # Option A: keep one additional quarter before k0_used if possible
        k0_keep = max(k0_used - 1, 0)

        y_q = D.y_q[k0_keep : k1_used + 1].copy()
        dates_q = D.dates_q[k0_keep : k1_used + 1].copy()

        # Remap quarter_of_month to new indexing starting at 0
        remap = qidx.copy()
        remap[remap >= 0] = remap[remap >= 0] - k0_keep
        quarter_of_month = remap


    return ZhangData(
        X_m=X_m,
        dates_m=dates_m,
        series_m=D.series_m,
        y_q=y_q,
        dates_q=dates_q,
        quarter_of_month=quarter_of_month,
        month_pos_in_quarter=month_pos_in_quarter,
        obs_idx=obs_idx,
    )