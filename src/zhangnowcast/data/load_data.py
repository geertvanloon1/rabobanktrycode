from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---- Spec container (matches MATLAB struct fields used) ----
@dataclass
class Spec:
    SeriesID: List[str]            # e.g. ["CPIAUCSL", "INDPRO", ...]
    SeriesName: List[str]          # human-readable names
    Transformation: List[str]     # ["lin","pch",...]
    Frequency: List[str]          # ["m","q",...] monthly/quarterly


def load_data(
    datafile: Union[str, Path],
    spec: Spec,
    sample: Optional[Union[str, pd.Timestamp, np.datetime64]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load vintage of data from Excel file and format as arrays.

    Parameters
    ----------
    datafile : str | Path
        Path like "data/US/2016-07-08.xls"
    spec : Spec
        Model specification with SeriesID ordering and transformations.
    sample : optional
        If provided, drop observations with Time < sample.

    Returns
    -------
    X : (T, N) np.ndarray
        Transformed dataset.
    Time : (T,) np.ndarray
        Observation dates (datetime64).
    Z : (T, N) np.ndarray
        Raw (untransformed) dataset (aligned/sorted and trimmed like X).
    """
    

    datafile = Path(datafile)
    ext = datafile.suffix.lower()

    if ext not in [".xlsx", ".xls"]:
        raise ValueError("Only Microsoft Excel workbook files supported.")

    # --- Read raw data from Excel ---
    Z, Time, mnem = read_data_excel(datafile)

    # --- Sort data based on model specification ---
    Z, mnem = sort_data(Z, mnem, spec)

    # --- Transform data based on model specification ---
    X, Time, Z = transform_data(Z, Time, spec)

    # --- Drop data not in estimation sample ---
    if sample is not None:
        X, Time, Z = drop_data(X, Time, Z, sample)

    return X, Time, Z


def read_data_excel(datafile: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Read raw data from Excel file 

    MATLAB behavior:
      - TEXT(1,2:end)   -> mnemonics
      - TEXT(2:end,1)  -> dates (Windows)
      - DATA(:,2:end)  -> values
    """
    if datafile.suffix.lower() == ".xls":
        df = pd.read_excel(datafile, sheet_name="data", header=None, engine="xlrd")
    else:
        df = pd.read_excel(datafile, sheet_name="data", header=None, engine="openpyxl")


    # First row, columns 2..end are mnemonics
    mnem = df.iloc[0, 1:].astype(str).tolist()

    # Dates: column 1, rows 2:end
    time_col = df.iloc[1:, 0]
    values = df.iloc[1:, 1:]

    # Parse dates
    if np.issubdtype(time_col.dtype, np.number):
        # Excel serial dates
        Time = pd.to_datetime(time_col, unit="D", origin="1899-12-30")
    else:
        # Text dates like 'mm/dd/yyyy'
        Time = pd.to_datetime(time_col.astype(str), format="%m/%d/%Y", errors="coerce")
        if Time.isna().any():
            Time = pd.to_datetime(time_col.astype(str), errors="coerce")

    Z = values.to_numpy(dtype=float)

    return Z, Time.to_numpy(dtype="datetime64[ns]"), mnem


def sort_data(Z: np.ndarray, mnem: List[str], spec: Spec) -> Tuple[np.ndarray, List[str]]:
    """
    Drop series not in Spec and sort columns to match Spec.SeriesID ordering.
    """
    mnem_arr = np.array(mnem, dtype=object)

    # Drop series not in Spec
    in_spec = np.isin(mnem_arr, np.array(spec.SeriesID, dtype=object))
    mnem_arr = mnem_arr[in_spec]
    Z = Z[:, in_spec]

    # Sort by Spec ordering
    perm = []
    for sid in spec.SeriesID:
        matches = np.where(mnem_arr == sid)[0]
        if len(matches) == 0:
            raise ValueError(f"SeriesID '{sid}' in spec not found in datafile columns.")
        perm.append(int(matches[0]))

    mnem_sorted = mnem_arr[perm].tolist()
    Z_sorted = Z[:, perm]

    return Z_sorted, mnem_sorted


def transform_data(Z: np.ndarray, Time: np.ndarray, spec: Spec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply transformations and drop first quarter of observations (first 3 months),
    exactly like the MATLAB code.
    """
    T, N = Z.shape
    X = np.full((T, N), np.nan, dtype=float)

    for i in range(N):
        formula = spec.Transformation[i]
        freq = spec.Frequency[i].lower()

        # Step for different frequencies
        if freq == "m":
            step = 1
        elif freq == "q":
            step = 3
        else:
            raise ValueError(f"Unknown frequency '{spec.Frequency[i]}' for series {spec.SeriesID[i]}")

        t1 = step - 1          # MATLAB t1 = step (1-indexed) → Python 0-index
        n = step / 12.0       # number of years for annualization
        series_name = spec.SeriesName[i]

        if formula == "lin":
            X[:, i] = Z[:, i]

        elif formula == "chg":
            idx = np.arange(t1, T, step)
            prev = idx[:-1]
            nxt = idx[1:]
            X[nxt, i] = Z[nxt, i] - Z[prev, i]

        elif formula == "ch1":
            start = 12 + t1
            idx2 = np.arange(start, T, step)
            idx0 = np.arange(t1, T - 12, step)
            L = min(len(idx2), len(idx0))
            X[idx2[:L], i] = Z[idx2[:L], i] - Z[idx0[:L], i]

        elif formula == "pch":
            idx = np.arange(t1, T, step)
            prev = idx[:-1]
            nxt = idx[1:]
            X[nxt, i] = 100.0 * (Z[nxt, i] / Z[prev, i] - 1.0)

        elif formula == "pc1":
            start = 12 + t1
            idx2 = np.arange(start, T, step)
            idx0 = np.arange(t1, T - 12, step)
            L = min(len(idx2), len(idx0))
            X[idx2[:L], i] = 100.0 * (Z[idx2[:L], i] / Z[idx0[:L], i] - 1.0)

        elif formula == "pca":
            idx = np.arange(t1, T, step)
            prev = idx[:-1]
            nxt = idx[1:]
            X[nxt, i] = 100.0 * ((Z[nxt, i] / Z[prev, i]) ** (1.0 / n) - 1.0)

        elif formula == "log":
            X[:, i] = np.log(Z[:, i])
            
        elif formula == "cch":
            # Continuously compounded rate of change: 100 * Δ log Z
            idx = np.arange(t1, T, step)
            prev = idx[:-1]
            nxt = idx[1:]

            # Guard against nonpositive values for log
            valid = (Z[nxt, i] > 0) & (Z[prev, i] > 0)
            X[nxt[valid], i] = 100.0 * (np.log(Z[nxt[valid], i]) - np.log(Z[prev[valid], i]))

        elif formula == "cca":
            # Continuously compounded ANNUAL rate of change: (1200/step) * Δ log Z
            idx = np.arange(t1, T, step)
            prev = idx[:-1]
            nxt = idx[1:]

            annual_factor = 1200.0 / step  # step=1 -> 1200, step=3 -> 400
            valid = (Z[nxt, i] > 0) & (Z[prev, i] > 0)
            X[nxt[valid], i] = annual_factor * (np.log(Z[nxt[valid], i]) - np.log(Z[prev[valid], i]))

        else:
            print(f"Warning: Transformation '{formula}' not found for {series_name}. Using untransformed data.")
            X[:, i] = Z[:, i]

    # Drop first quarter (MATLAB: Time = Time(4:end))
    Time = Time[3:]
    Z = Z[3:, :]
    X = X[3:, :]

    return X, Time, Z


def drop_data(
    X: np.ndarray,
    Time: np.ndarray,
    Z: np.ndarray,
    sample: Union[str, pd.Timestamp, np.datetime64],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove data not in estimation sample (Time < sample).
    """
    sample_ts = pd.to_datetime(sample).to_datetime64()
    idx_keep = Time >= sample_ts
    return X[idx_keep, :], Time[idx_keep], Z[idx_keep, :]
