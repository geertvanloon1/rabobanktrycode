from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from typing import List, Optional, Tuple  # add Optional, List if not already imported



def _json_default(x: Any):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return str(x)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_run_id(batch_config: Dict[str, Any]) -> str:
    payload = json.dumps(batch_config, sort_keys=True, default=_json_default).encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()[:8]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{h}"


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))
    tmp.replace(path)


def init_run_dir(outputs_root: Path, *, batch_config: Dict[str, Any], data_config: Dict[str, Any]) -> Path:
    run_id = make_run_id(batch_config)
    run_dir = outputs_root / run_id

    (run_dir / "nowcasts" / "per_run").mkdir(parents=True, exist_ok=True)
    (run_dir / "posterior" / "per_run").mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    meta = dict(
        run_id=run_id,
        created_at_utc=_utc_now_iso(),
        environment=dict(
            python=sys.version.split()[0],
            platform=platform.platform(),
        ),
        batch_config=batch_config,
        data=data_config,
        runtime=dict(start_utc=_utc_now_iso()),
    )
    write_json(run_dir / "run_metadata.json", meta)
    return run_dir


def finalize_run_dir(run_dir: Path, *, start_time: float) -> None:
    meta_path = run_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["runtime"]["end_utc"] = _utc_now_iso()
    meta["runtime"]["seconds"] = float(time.time() - start_time)
    write_json(meta_path, meta)


def _safe_series_names(D) -> np.ndarray:
    n = int(D.X_m.shape[1])
    for attr in ("series", "series_ids", "series_names", "x_names", "var_names"):
        if hasattr(D, attr):
            v = getattr(D, attr)
            try:
                if v is not None and len(v) == n:
                    return np.asarray([str(s) for s in v], dtype=object)
            except Exception:
                pass
    return np.asarray([f"series_{i}" for i in range(n)], dtype=object)



import numpy as np

def _extract_series_labels(D) -> np.ndarray:
    """
    Extract labels aligned with the N monthly series in D.X_m.
    In this codebase, build_zhang_data stores them as D.series_m.
    """
    # Most likely correct in your project:
    if hasattr(D, "series_m") and getattr(D, "series_m") is not None:
        s = np.asarray(getattr(D, "series_m"), dtype=object)
        if s.size:
            return s

    # Backwards/alternative attribute names (just in case)
    for attr in ["series", "series_names", "series_labels", "var_names", "names"]:
        if hasattr(D, attr) and getattr(D, attr) is not None:
            s = np.asarray(getattr(D, attr), dtype=object)
            if s.size:
                return s

    # Fallback: aligned placeholders
    N = getattr(D, "X_m", np.empty((0, 0))).shape[1]
    return np.asarray([f"series_{i+1}" for i in range(N)], dtype=object)


def save_subrun_outputs(
    *,
    run_dir: Path,
    model_type: str,
    tag: str,
    D,
    res,
    month_T: pd.Timestamp,
    q: int,
    vintage_file: str,
    target_quarter: str | None = None,  # legacy; ignored (we compute targets from res)
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    NEW CONTRACT:
      - returns (now_rows, diag) where now_rows is a LIST of rows
        (one per target quarter in res.nowcast_targets).
      - saves posterior.npz including nowcast_targets + nowcast_draws_matrix.

    Backward compatibility:
      - If res does not have nowcast_targets/matrix, we fall back to single-target mode.
    """
    month_T0 = pd.to_datetime(month_T).normalize().replace(day=1)
    month_in_quarter = ((int(month_T0.month) - 1) % 3) + 1
    asof_quarter = str(month_T0.to_period("Q"))  # <-- FIXES your Period.astype error

    sub_dir = run_dir / "posterior" / "per_run" / model_type / tag
    sub_dir.mkdir(parents=True, exist_ok=True)

    dates_m = pd.to_datetime(D.dates_m).strftime("%Y-%m-%d").to_numpy(dtype=object)

    try:
        dates_q = pd.to_datetime(D.dates_q).to_period("Q").astype(str).to_numpy(dtype=object)
    except Exception:
        dates_q = np.asarray([], dtype=object)

    series_labels = _extract_series_labels(D)

    # ---- detect multi-target vs single-target ----
    has_panel = hasattr(res, "nowcast_targets") and hasattr(res, "nowcast_draws_matrix")
    if has_panel and res.nowcast_targets is not None and res.nowcast_draws_matrix is not None:
        nowcast_targets = [str(x) for x in res.nowcast_targets]
        nowcast_draws_matrix = np.asarray(res.nowcast_draws_matrix, dtype=float)  # (S,J)
    else:
        # fallback: single target = legacy fields
        nowcast_targets = [str(target_quarter) if target_quarter is not None else asof_quarter]
        nowcast_draws_matrix = np.asarray(res.nowcast_draws, dtype=float)[:, None]  # (S,1)

    # save posterior arrays
    np.savez_compressed(
        sub_dir / "posterior.npz",
        F_mean=res.F_mean,
        F_sd=res.F_sd,
        Lambda_mean=res.Theta_mean,
        Lambda_sd=res.Theta_sd,
        mu_mean=res.mu_mean,
        z_mean=res.z_mean,
        z_draws=res.z_draws,
        r_draws=res.r_draws,
        beta_mean=res.beta_mean,
        a_mean=res.a_mean,
        sigma2_mean=res.sigma2_mean,
        eta2_mean=np.asarray(res.eta2_mean, dtype=float),
        Psi_mean=res.Psi_mean,
        omega_mean=res.omega_mean,
        omega_sd=res.omega_sd,
        tau2_mean=np.asarray(res.tau2_mean, dtype=float),

        # NEW:
        nowcast_targets=np.asarray(nowcast_targets, dtype=object),
        nowcast_draws_matrix=nowcast_draws_matrix,

        # keep old key too (useful if old plotting expects it)
        nowcast_draws=np.asarray(res.nowcast_draws, dtype=float),

        dates_m=dates_m,
        dates_q=dates_q,
        series=series_labels,
        standardize_mean=res.standardize_mean,
        standardize_sd=res.standardize_sd,
    )

    posterior_meta = dict(
        model_type=model_type,
        subrun_tag=tag,
        month_T=month_T0.strftime("%Y-%m-%d"),
        asof_quarter=asof_quarter,
        q=int(q),
        month_in_quarter=int(month_in_quarter),
        vintage_file=vintage_file,
        T=int(D.X_m.shape[0]),
        N=int(D.X_m.shape[1]),
        R_max=int(res.config.get("R_max", res.F_mean.shape[1])),
        draws_kept=int(res.draws_kept),
        use_sv=bool(res.config.get("use_sv", False)),
        nowcast_targets=nowcast_targets,
        nowcast_draws_shape=list(nowcast_draws_matrix.shape),
    )
    write_json(sub_dir / "posterior_meta.json", posterior_meta)

    diag = dict(
        model=model_type,
        subrun_tag=tag,
        month=month_T0.strftime("%Y-%m-%d"),
        asof_quarter=asof_quarter,
        q=int(q),
        psi_accept_rate=float(res.diagnostics.get("psi_accept_rate", float("nan"))),
        psi_accepts=int(res.diagnostics.get("psi_accepts", -1)),
        psi_trials=int(res.diagnostics.get("psi_trials", -1)),
        nan_F_mean=int(np.isnan(res.F_mean).sum()),
        nan_Lambda_mean=int(np.isnan(res.Theta_mean).sum()),
        nan_omega_mean=int(np.isnan(res.omega_mean).sum()),
        draws_kept=int(res.draws_kept),
    )
    write_json(sub_dir / "diagnostics.json", diag)

    # ---- build one output row per target ----
    now_rows: List[Dict[str, Any]] = []
    S, J = nowcast_draws_matrix.shape

    for j, tq in enumerate(nowcast_targets):
        draws = nowcast_draws_matrix[:, j]

        p05 = float(np.nanpercentile(draws, 5)) if draws.size else float("nan")
        p50 = float(np.nanpercentile(draws, 50)) if draws.size else float("nan")
        p95 = float(np.nanpercentile(draws, 95)) if draws.size else float("nan")
        mean = float(np.nanmean(draws)) if draws.size else float("nan")
        sd = float(np.nanstd(draws, ddof=0)) if draws.size else float("nan")

        now_rows.append(dict(
            model=model_type,
            month=month_T0.strftime("%Y-%m-%d"),
            asof_quarter=asof_quarter,
            month_in_quarter=int(month_in_quarter),
            q=int(q),
            vintage_file=vintage_file,

            # key fields for the new logic:
            target_quarter=str(tq),
            asof_in_target=(str(tq) == asof_quarter),

            # moments
            nowcast_mean=mean,
            nowcast_p05=p05,
            nowcast_p50=p50,
            nowcast_p95=p95,
            nowcast_sd=sd,

            actual=float("nan"),       # filled later by artifacts script
            draws_kept=int(res.draws_kept),
            xT_obs_count=int(np.sum(~np.isnan(D.X_m[-1, :]))),
            subrun_tag=tag,

            # helpful auditing
            target_index=int(j),
            n_targets=int(J),
        ))

    return now_rows, diag