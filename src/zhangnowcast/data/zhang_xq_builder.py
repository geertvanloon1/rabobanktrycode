from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from zhangnowcast.data.data import ZhangData


def latest_available_month(month_T: pd.Timestamp, q: int, set_id: int) -> pd.Timestamp:
    """Table-2 ragged-edge rule for the latest available month.

    This implements the paper's release-date information set (Zhang, 2022, Table 2)
    used to construct X^q.

    Args:
        month_T: Nowcast month T as a pandas Timestamp. Day is ignored (normalized to 1).
        q: Release within month T (1, 2, or 3).
        set_id: Release set for the series (1, 2, or 3).

    Returns:
        A month-start Timestamp representing the latest month whose observation is
        available at (T, q) for that series.
    """
    if q not in (1, 2, 3):
        raise ValueError("q must be 1, 2, or 3")
    if set_id not in (1, 2, 3):
        raise ValueError("set_id must be 1, 2, or 3")

    month_T = month_T.normalize().replace(day=1)

    # Set 1: latest month = T-2 for all q
    if set_id == 1:
        return (month_T - pd.offsets.MonthBegin(2)).normalize()

    # Set 2: latest month = T-2 if q=1; else T-1
    if set_id == 2:
        return (month_T - pd.offsets.MonthBegin(2)).normalize() if q == 1 else (month_T - pd.offsets.MonthBegin(1)).normalize()

    # Set 3: latest month = T-1 if q∈{1,2}; else T
    return (month_T - pd.offsets.MonthBegin(1)).normalize() if q in (1, 2) else month_T


def make_Xq_for_month(
    D_full: ZhangData,
    series_bucket: Dict[str, int],
    month_M: pd.Timestamp,
    q: int,
) -> ZhangData:
    """
    Construct the release-q ragged-edge information set X^q for nowcast month T (=month_M).

    Paper requirement (Zhang, 2022, Table 2): monthly series are split into 3 release
    sets (Set 1/2/3). For a given nowcast month T and release q∈{1,2,3}, the latest
    available month depends on (set_id, q). All observations after the latest
    available month are treated as not-yet-released and must be set to NaN.

    Existing NaNs (true missing data) are preserved; we only add additional NaNs
    for "not yet released" observations.

    Input notes:
      - `series_bucket` is interpreted as a mapping {series_name -> set_id in {1,2,3}}.
        (The parameter name is kept for compatibility with older code.)
      - `month_M` corresponds to paper's nowcast month T. Indexing in code is 0-based
        (row t=0 is the first month in `D_full.dates_m`).
    """
    dates_m = pd.to_datetime(D_full.dates_m)
    month_M = month_M.normalize().replace(day=1)

    # Sanity: month_M should be in the sample (it is "T" in the paper)
    if np.where(dates_m == month_M)[0].size == 0:
        raise ValueError(f"Nowcast month {month_M.date()} not in sample.")

    Xq = D_full.X_m.copy()

    # For each series i, mask all months strictly AFTER its latest available month.
    for i, sname in enumerate(D_full.series_m):
        set_id = int(series_bucket.get(sname, 3))
        latest_m = latest_available_month(month_M, q=q, set_id=set_id)
        mask_rows = np.where(dates_m > latest_m)[0]
        if mask_rows.size:
            Xq[mask_rows, i] = np.nan

    obs_idx = [np.where(~np.isnan(Xq[t]))[0].astype(int) for t in range(Xq.shape[0])]

    return ZhangData(
        X_m=Xq,
        dates_m=D_full.dates_m,
        series_m=D_full.series_m,
        y_q=D_full.y_q,
        dates_q=D_full.dates_q,
        quarter_of_month=D_full.quarter_of_month,
        month_pos_in_quarter=D_full.month_pos_in_quarter,
        obs_idx=obs_idx,
    )
