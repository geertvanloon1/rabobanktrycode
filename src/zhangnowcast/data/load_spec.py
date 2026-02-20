from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import re


@dataclass
class Spec:
    SeriesID: List[str]
    SeriesName: List[str]
    Frequency: List[str]
    Units: List[str]
    Transformation: List[str]
    Category: List[str]
    Blocks: np.ndarray               # shape (N, B)
    BlockNames: List[str]
    UnitsTransformed: List[str]


def load_spec(
    specfile: Union[str, Path],
    sheet_name: Union[str, int, None] = 0,
) -> Spec:
    """
    Load model specification from an Excel workbook, matching MATLAB load_spec.m.
    """
    specfile = Path(specfile)

    # Read the sheet raw: no header inference, keep everything as-is
    raw_df = pd.read_excel(specfile, sheet_name=sheet_name, header=None, dtype=object)

    # MATLAB: header = strrep(raw(1,:),' ',''); raw = raw(2:end,:);
    header = raw_df.iloc[0, :].astype(str).str.replace(" ", "", regex=False).tolist()
    data = raw_df.iloc[1:, :].copy()

    # Helper: find column index by name (case-insensitive)
    def col_idx(name: str) -> int:
        name_l = name.lower()
        for j, h in enumerate(header):
            if str(h).lower() == name_l:
                return j
        return -1

    # Filter out rows not in model: Model column must exist
    j_model = col_idx("Model")
    if j_model == -1:
        raise ValueError("Model column missing from model specification.")

    model_vals = pd.to_numeric(data.iloc[:, j_model], errors="coerce").fillna(0).to_numpy()
    data = data.loc[model_vals != 0].reset_index(drop=True)

    # Parse required fields
    required = ["SeriesID", "SeriesName", "Frequency", "Units", "Transformation", "Category"]
    parsed: Dict[str, List[Any]] = {}

    for fld in required:
        j = col_idx(fld)
        if j == -1:
            raise ValueError(f"{fld} column missing from model specification.")
        parsed[fld] = data.iloc[:, j].tolist()

    # Parse block columns: MATLAB jColBlock = strncmpi('Block',header,length('Block'));
    block_cols = [j for j, h in enumerate(header) if str(h).lower().startswith("block")]
    if len(block_cols) == 0:
        raise ValueError("No Block* columns found in model specification.")

    # ---- FIX: ensure blocks is a writable array ----
    blocks = (
        data.iloc[:, block_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
        .copy()
    )
    blocks[np.isnan(blocks)] = 0.0

    # MATLAB requires all blocks(:,1)==1 i.e., first block column must be 1 for all variables
    if not np.allclose(blocks[:, 0], 1.0):
        raise ValueError("All variables must load on global block (first Block column must be 1).")

    # Sort in order of decreasing frequency: d, w, m, q, sa, a
    freq_order = ["d", "w", "m", "q", "sa", "a"]
    freq_series = [str(x).strip().lower() for x in parsed["Frequency"]]

    permutation: List[int] = []
    for f in freq_order:
        permutation.extend([i for i, fx in enumerate(freq_series) if fx == f])

    # Append any remaining (unexpected) frequencies at the end
    remaining = [i for i in range(len(freq_series)) if i not in permutation]
    permutation.extend(remaining)

    # Apply permutation to all fields + blocks
    for k in parsed:
        parsed[k] = [parsed[k][i] for i in permutation]
    blocks = blocks[permutation, :]

    # BlockNames: MATLAB regexprep(header(jColBlock),'Block\d-','');
    block_headers = [header[j] for j in block_cols]
    block_names = [re.sub(r"Block\d-", "", str(bh)) for bh in block_headers]

    # UnitsTransformed mappings (same as MATLAB string replaces)
    units_trans = [str(x) for x in parsed["Transformation"]]
    repl = {
        "lin": "Levels (No Transformation)",
        "chg": "Change (Difference)",
        "ch1": "Year over Year Change (Difference)",
        "pch": "Percent Change",
        "pc1": "Year over Year Percent Change",
        "pca": "Percent Change (Annual Rate)",
        "cch": "Continuously Compounded Rate of Change",
        "cca": "Continuously Compounded Annual Rate of Change",
        "log": "Natural Log",
    }
    units_trans = [repl.get(u, u) for u in units_trans]

    spec = Spec(
        SeriesID=[str(x) for x in parsed["SeriesID"]],
        SeriesName=[str(x) for x in parsed["SeriesName"]],
        Frequency=[str(x).strip().lower() for x in parsed["Frequency"]],
        Units=[str(x) for x in parsed["Units"]],
        Transformation=[str(x) for x in parsed["Transformation"]],
        Category=[str(x) for x in parsed["Category"]],
        Blocks=blocks,
        BlockNames=block_names,
        UnitsTransformed=units_trans,
    )

    ## Summarize model specification (like MATLAB)
    #print("Table 1: Model specification")
    #try:
    #    summary = pd.DataFrame(
    #        {
    #            "SeriesID": spec.SeriesID,
    #            "SeriesName": spec.SeriesName,
    #           "Units": spec.Units,
    #           "Transformation": spec.UnitsTransformed,
    #       }
    #    )
    #    print(summary.to_string(index=False))
    #except Exception:
    #    pass

    return spec
