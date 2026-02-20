from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from zhangnowcast.data.data import build_zhang_data, ZhangData


@dataclass(frozen=True)
class Vintage:
    path: Path
    date: pd.Timestamp  # YYYY-MM-DD


def parse_vintage_date(p: Path) -> pd.Timestamp:
    # expects filenames like 2016-06-29.xls
    return pd.to_datetime(p.stem, format="%Y-%m-%d").normalize()


def list_vintages(vintage_dir: Path, pattern: str = "*.xls") -> List[Vintage]:
    vint = []
    for p in vintage_dir.glob(pattern):
        try:
            vint.append(Vintage(p, parse_vintage_date(p)))
        except Exception:
            pass
    vint.sort(key=lambda v: v.date)
    return vint


def pick_vintage_for_month_q(
    vintages: list[Vintage],
    month_T: pd.Timestamp,
    q: int
) -> Vintage:
    month_T = pd.to_datetime(month_T).normalize().replace(day=1)
    month_end = (month_T + pd.offsets.MonthEnd(1)).normalize()
    target_day = {1: 10, 2: 20, 3: month_end.day}[q]
    target = pd.Timestamp(year=month_T.year, month=month_T.month, day=target_day)

    same_month = [v for v in vintages if (v.date.year == month_T.year and v.date.month == month_T.month)]
    same_month.sort(key=lambda v: v.date)

    if same_month:
        candidates = [v for v in same_month if v.date <= target]
        if candidates:
            return candidates[-1]
        return same_month[0]

    candidates_all = [v for v in vintages if v.date <= target]
    if candidates_all:
        return candidates_all[-1]
    return vintages[0]



def bucket_from_day(day: int) -> int:
    if day <= 10:
        return 1
    if day <= 20:
        return 2
    return 3


def infer_buckets_month_by_month(
    vintage_dir: Path,
    spec_file: Path,
    gdp_series_id: str,
    sample_start: str,
    pattern: str = "*.xls",
    min_observations_per_series: int = 2,
    vintage_start: str | None = None,
    vintage_end: str | None = None,
) -> Dict[str, int]:
    """
    For each calendar month M, look at daily vintages in that month.
    For each monthly series i, record the first day in month M when the value for month (M-1)
    becomes non-NaN. Take median over months and map to bucket 1/2/3 (<=10, <=20, else 3).

    Returns mapping {series_name: bucket}.
    """
    vintages = list_vintages(vintage_dir, pattern=pattern)
    if len(vintages) < 2:
        raise ValueError("Need at least 2 vintage files to infer buckets.")

        # Optional: restrict bucket inference to a specific vintage date window
    if vintage_start is not None:
        vs = pd.Timestamp(vintage_start).normalize()
        vintages = [v for v in vintages if v.date >= vs]
    if vintage_end is not None:
        ve = pd.Timestamp(vintage_end).normalize()
        vintages = [v for v in vintages if v.date <= ve]

    if len(vintages) < 2:
        raise ValueError("Need at least 2 vintage files in the selected window to infer buckets.")

    # group vintages by calendar month (YYYY-MM)
    by_month: Dict[Tuple[int, int], List[Vintage]] = {}
    for v in vintages:
        by_month.setdefault((v.date.year, v.date.month), []).append(v)

    # We'll infer using the series list from any loaded vintage
    D0 = build_zhang_data(vintages[0].path, spec_file, gdp_series_id, sample_start=sample_start)
    series_names = list(D0.series_m)
    n = len(series_names)

    # record release day observations per series
    release_days: List[List[int]] = [[] for _ in range(n)]

    for (yy, mm), vs in sorted(by_month.items()):
        vs = sorted(vs, key=lambda x: x.date)

        # Use last vintage in month to locate the row for (M-1)
        D_last = build_zhang_data(vs[-1].path, spec_file, gdp_series_id, sample_start=sample_start)
        dates_m = pd.to_datetime(D_last.dates_m)
        month_M = pd.Timestamp(year=yy, month=mm, day=1)
        month_prev = (month_M - pd.offsets.MonthBegin(1)).normalize()

        # Find index of previous month row in D_last
        try:
            t_prev = int(np.where(dates_m == month_prev)[0][0])
        except Exception:
            # if not in sample, skip this month
            continue

        # For each series, find first day in month M where X[t_prev, i] becomes available
        first_day = np.full(n, fill_value=np.nan)

        for v in vs:
            Dv = build_zhang_data(v.path, spec_file, gdp_series_id, sample_start=sample_start)

            # Align by time index: assume same dates_m in this month; if not, match by date
            dates_v = pd.to_datetime(Dv.dates_m)
            idx = np.where(dates_v == month_prev)[0]
            if idx.size == 0:
                continue
            t_prev_v = int(idx[0])

            row = Dv.X_m[t_prev_v, :]
            newly_available = ~np.isnan(row) & np.isnan(first_day)
            if newly_available.any():
                first_day[newly_available] = v.date.day

        for i in range(n):
            if not np.isnan(first_day[i]):
                release_days[i].append(int(first_day[i]))

    buckets: Dict[str, int] = {}
    for i, name in enumerate(series_names):
        if len(release_days[i]) < min_observations_per_series:
            buckets[name] = 3  # conservative default: late
        else:
            med_day = int(np.median(release_days[i]))
            buckets[name] = bucket_from_day(med_day)

    return buckets
