"""Verify the current code produces outputs identical to the pre-change baselines.

Run AFTER every optimization change. Exits with code 0 if every pipeline matches
the captured baseline, non-zero with diff output otherwise.

Run:  python -m tests.golden.verify_outputs
"""

from __future__ import annotations

# Install the streamlit shim FIRST, before any src.modules imports.
from tests.golden import _streamlit_shim  # noqa: F401

import json
import os
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd

# Comma-separated suffixes (e.g. "_str") whose matching columns are dropped from
# BOTH baseline and current frames before comparison. Use during refactors that
# intentionally remove columns; reset to "" once baselines are re-captured.
_IGNORE_SUFFIXES = tuple(s for s in os.environ.get("HMIS_IGNORE_COL_SUFFIXES", "").split(",") if s)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SAMPLE_CSV = Path(__file__).parent / "sample_hmis.csv"
BASELINES = Path(__file__).parent / "baselines"

REPORT_START = pd.Timestamp("2024-01-01")
REPORT_END = pd.Timestamp("2024-12-31")
LOOKBACK_DAYS = 730
DASHBOARD_PERIOD_START = pd.Timestamp("2024-01-01")
DASHBOARD_PERIOD_END = pd.Timestamp("2024-12-31")


def _load_uploaded_file() -> BytesIO:
    with open(SAMPLE_CSV, "rb") as f:
        buf = BytesIO(f.read())
    buf.name = "sample_hmis.csv"
    buf.seek(0)
    return buf


def _unwrap(fn):
    return getattr(fn, "__wrapped__", fn)


def _normalize_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Same normalization used in capture: cast categoricals/object to string and
    coerce row order so comparison ignores irrelevant differences (categorical
    metadata, row order). The values themselves must match exactly."""
    out = df.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype("string")
        elif out[col].dtype == "object":
            out[col] = out[col].astype("string")
    sort_cols = [c for c in ("ClientID", "EnrollmentID", "ProjectStart", "ProjectExit") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def _drop_ignored(df: pd.DataFrame) -> pd.DataFrame:
    if not _IGNORE_SUFFIXES:
        return df
    cols = [c for c in df.columns if not any(c.endswith(s) for s in _IGNORE_SUFFIXES)]
    return df[cols]


def _assert_frames_equal(name: str, current: pd.DataFrame, baseline_path: Path) -> None:
    baseline = pd.read_parquet(baseline_path)
    cur_norm = _drop_ignored(_normalize_for_compare(current))
    base_norm = _drop_ignored(_normalize_for_compare(baseline))
    # Align column order before comparing (categorical conversion can shuffle).
    common = [c for c in base_norm.columns if c in cur_norm.columns]
    missing_in_cur = set(base_norm.columns) - set(cur_norm.columns)
    extra_in_cur = set(cur_norm.columns) - set(base_norm.columns)
    if missing_in_cur or extra_in_cur:
        print(f"  FAIL {name}: column mismatch")
        if missing_in_cur:
            print(f"    missing in current: {sorted(missing_in_cur)}")
        if extra_in_cur:
            print(f"    extra in current:   {sorted(extra_in_cur)}")
        raise AssertionError(f"{name}: columns differ")
    cur_norm = cur_norm[common]
    base_norm = base_norm[common]
    try:
        pd.testing.assert_frame_equal(
            cur_norm,
            base_norm,
            check_exact=True,
            check_categorical=False,
            check_dtype=False,
        )
    except AssertionError as e:
        print(f"  FAIL {name}: DataFrame differs")
        print(f"    current shape={cur_norm.shape}  baseline shape={base_norm.shape}")
        # Print first ~5 rows of diff for fast diagnosis
        for col in common:
            if not cur_norm[col].equals(base_norm[col]):
                # find first index where they differ
                diff_mask = (cur_norm[col].fillna("__NA__") != base_norm[col].fillna("__NA__"))
                if diff_mask.any():
                    first = diff_mask.idxmax()
                    print(f"    col {col!r}: first diff at row {first}: "
                          f"current={cur_norm[col].iloc[first]!r}  "
                          f"baseline={base_norm[col].iloc[first]!r}")
                    break
        raise
    print(f"  OK   {name}")


def verify_loaded_df() -> pd.DataFrame:
    from src.core.data.loader import load_and_preprocess_data
    df = _unwrap(load_and_preprocess_data)(_load_uploaded_file())
    _assert_frames_equal("loaded_df", df, BASELINES / "loaded_df.parquet")
    return df


def verify_spm2(df: pd.DataFrame) -> None:
    from src.modules.spm2.calculator import run_spm2
    out = _unwrap(run_spm2)(df, REPORT_START, REPORT_END, lookback_value=730,
                            lookback_unit="Days", return_period=730)
    _assert_frames_equal("spm2_default", out, BASELINES / "spm2_default.parquet")


def verify_inbound(df: pd.DataFrame) -> None:
    from src.modules.recidivism.inbound_calculator import run_return_analysis
    out = _unwrap(run_return_analysis)(
        df, REPORT_START, REPORT_END, LOOKBACK_DAYS,
        None, None, None, None, None, None, None, None, None, None, None, None,
    )
    _assert_frames_equal("inbound_default", out, BASELINES / "inbound_default.parquet")


def verify_outbound(df: pd.DataFrame) -> None:
    from src.modules.recidivism.outbound_calculator import run_outbound_recidivism
    out = _unwrap(run_outbound_recidivism)(df, REPORT_START, REPORT_END)
    _assert_frames_equal("outbound_default", out, BASELINES / "outbound_default.parquet")


def _verify_spm2_parity() -> None:
    """Run the per-(client, exit) parity check that ensures
    ``_find_earliest_return_fast`` matches ``_find_earliest_return_legacy``
    byte-for-byte across every exit in the fixture. Imported from the
    dedicated script so we share a single source of truth."""
    import importlib
    spm2_parity = importlib.import_module("tests.golden.spm2_parity")
    # The script's main() exits via sys.exit(1) on mismatch; wrap to raise
    # AssertionError instead so verify_outputs can aggregate the failure.
    import io
    import contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            spm2_parity.main()
    except SystemExit as e:
        if e.code != 0:
            raise AssertionError(
                f"spm2_parity: {buf.getvalue().splitlines()[-1] if buf.getvalue() else 'failed'}"
            )
    # Echo the count line for visibility.
    for line in buf.getvalue().splitlines():
        if "Compared" in line:
            print(f"  {line.strip()}")
    print("  OK   spm2_parity")


def _verify_duckdb_parity(df: pd.DataFrame) -> None:
    """Force the DuckDB filter path (bypassing the row-count threshold) and
    confirm it returns the same row set as the pandas mask path. Keeps the
    DuckDB code reachable in CI even when sample fixtures are small."""
    from src.modules.dashboard.filters import _apply_filters_duckdb
    # Temporarily lower the threshold so the small fixture exercises DuckDB.
    import src.modules.dashboard.filters as _filters_mod
    original = _filters_mod._DUCKDB_MIN_ROWS
    _filters_mod._DUCKDB_MIN_ROWS = 0
    try:
        filter_tuple = (
            ("ProjectTypeCode", ("Emergency Shelter – Entry Exit", "PH – Rapid Re-Housing")),
            ("Gender", ("Woman", "Man")),
        )
        ddb = _apply_filters_duckdb(df, filter_tuple)
    finally:
        _filters_mod._DUCKDB_MIN_ROWS = original
    if ddb is None:
        raise AssertionError("duckdb_parity: DuckDB path returned None (unavailable?)")
    # Compare row set against the pandas baseline by EnrollmentID multiset.
    baseline = pd.read_parquet(BASELINES / "dashboard_filtered.parquet")
    ddb_ids = sorted(int(x) for x in ddb["EnrollmentID"])
    base_ids = sorted(int(x) for x in baseline["EnrollmentID"])
    if ddb_ids != base_ids:
        sym = set(ddb_ids) ^ set(base_ids)
        raise AssertionError(
            f"duckdb_parity: row sets differ. duckdb={len(ddb_ids)} "
            f"baseline={len(base_ids)} sym_diff={len(sym)}"
        )
    print("  OK   duckdb_parity")


def verify_dashboard(df: pd.DataFrame) -> None:
    from src.modules.dashboard.data_utils import ClientMetrics, PHMetrics
    from src.modules.dashboard.filters import _apply_filters_cached

    client = _unwrap(ClientMetrics.batch_calculate_metrics)(
        df, DASHBOARD_PERIOD_START, DASHBOARD_PERIOD_END
    )
    ph = _unwrap(PHMetrics.batch_calculate_ph_metrics)(
        df, DASHBOARD_PERIOD_START, DASHBOARD_PERIOD_END
    )
    current = {
        "client_metrics": {
            "served_clients": sorted(int(x) for x in client["served_clients"]),
            "households_served": int(client["households_served"]),
            "inflow": sorted(int(x) for x in client["inflow"]),
            "outflow": sorted(int(x) for x in client["outflow"]),
        },
        "ph_metrics": {
            k: (sorted(int(x) for x in v) if isinstance(v, set) else int(v))
            for k, v in ph.items()
        },
    }
    with open(BASELINES / "dashboard_metrics.json") as fh:
        baseline = json.load(fh)
    if current != baseline:
        print("  FAIL dashboard_metrics")
        for section in ("client_metrics", "ph_metrics"):
            for key in current.get(section, {}):
                if current[section].get(key) != baseline[section].get(key):
                    cv = current[section][key]
                    bv = baseline[section].get(key)
                    if isinstance(cv, list) and isinstance(bv, list):
                        print(f"    {section}.{key}: current_len={len(cv)} baseline_len={len(bv)} "
                              f"sym_diff={len(set(cv) ^ set(bv))}")
                    else:
                        print(f"    {section}.{key}: current={cv}  baseline={bv}")
        raise AssertionError("dashboard_metrics: dicts differ")
    print("  OK   dashboard_metrics")

    filter_tuple = (
        ("ProjectTypeCode", ("Emergency Shelter – Entry Exit", "PH – Rapid Re-Housing")),
        ("Gender", ("Woman", "Man")),
    )
    filtered = _unwrap(_apply_filters_cached)(df, filter_tuple)
    _assert_frames_equal("dashboard_filtered", filtered, BASELINES / "dashboard_filtered.parquet")


def main() -> None:
    if not BASELINES.exists():
        sys.exit(f"No baselines at {BASELINES}. Run capture_baseline first.")
    print("Verifying current outputs match baseline ...")
    failures: list[str] = []
    # loaded_df has to succeed enough to pass `df` to downstream verifications.
    from src.core.data.loader import load_and_preprocess_data
    df = _unwrap(load_and_preprocess_data)(_load_uploaded_file())
    for name, fn in [
        ("loaded_df", lambda: _assert_frames_equal("loaded_df", df, BASELINES / "loaded_df.parquet")),
        ("spm2_default", lambda: verify_spm2(df)),
        ("inbound_default", lambda: verify_inbound(df)),
        ("outbound_default", lambda: verify_outbound(df)),
        ("dashboard", lambda: verify_dashboard(df)),
        ("duckdb_parity", lambda: _verify_duckdb_parity(df)),
        ("spm2_parity", _verify_spm2_parity),
    ]:
        try:
            fn()
        except AssertionError as e:
            failures.append(name)
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failures.append(name)
    if failures:
        print(f"\n{len(failures)} pipeline(s) failed: {failures}")
        sys.exit(1)
    print("All pipelines match baseline.")


if __name__ == "__main__":
    main()
