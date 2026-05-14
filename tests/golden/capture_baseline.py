"""Capture pre-change parquet/JSON snapshots of every analysis pipeline run on
the deterministic sample CSV. Run BEFORE any optimization change. The same
sample fed through the same pipelines after a change must produce byte-for-byte
equivalent outputs (modulo categorical-ordering nuances).

Run:  python -m tests.golden.capture_baseline
Output:
  tests/golden/baselines/loaded_df.parquet
  tests/golden/baselines/spm2_default.parquet
  tests/golden/baselines/inbound_default.parquet
  tests/golden/baselines/outbound_default.parquet
  tests/golden/baselines/dashboard_metrics.json
  tests/golden/baselines/dashboard_filtered.parquet
"""

from __future__ import annotations

# Install the streamlit shim FIRST, before any src.modules imports.
from tests.golden import _streamlit_shim  # noqa: F401

import json
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd

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
    """Return a BytesIO that mimics a Streamlit UploadedFile."""
    with open(SAMPLE_CSV, "rb") as f:
        buf = BytesIO(f.read())
    buf.name = "sample_hmis.csv"
    buf.seek(0)
    return buf


def _unwrap(fn):
    """Return the original function inside a @st.cache_data decorator, so we
    skip the cache layer entirely. Safe whether the function is decorated."""
    return getattr(fn, "__wrapped__", fn)


def _normalize_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet does not preserve every pandas dtype quirk; normalize for
    deterministic round-tripping in the verify step."""
    out = df.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            # Cast to string so categorical order/membership shifts don't
            # spuriously fail equality. We compare values, not category metadata.
            out[col] = out[col].astype("string")
        elif out[col].dtype == "object":
            out[col] = out[col].astype("string")
    return out


def _dump_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_for_storage(df).to_parquet(path, index=False)
    print(f"  wrote {path.name}  rows={len(df)}  cols={len(df.columns)}")


def capture_loaded_df() -> pd.DataFrame:
    print("[1/5] loaded_df")
    from src.core.data.loader import load_and_preprocess_data

    df = _unwrap(load_and_preprocess_data)(_load_uploaded_file())
    _dump_parquet(df, BASELINES / "loaded_df.parquet")
    return df


def capture_spm2(df: pd.DataFrame) -> None:
    print("[2/5] spm2_default")
    from src.modules.spm2.calculator import run_spm2

    result = _unwrap(run_spm2)(
        df,
        REPORT_START,
        REPORT_END,
        lookback_value=730,
        lookback_unit="Days",
        return_period=730,
    )
    _dump_parquet(result, BASELINES / "spm2_default.parquet")


def capture_inbound(df: pd.DataFrame) -> None:
    print("[3/5] inbound_default")
    from src.modules.recidivism.inbound_calculator import run_return_analysis

    result = _unwrap(run_return_analysis)(
        df,
        REPORT_START,
        REPORT_END,
        LOOKBACK_DAYS,
        allowed_cocs=None,
        allowed_localcocs=None,
        allowed_programs=None,
        allowed_agencies=None,
        entry_project_types=None,
        entry_ssvf_rrh=None,
        allowed_cocs_exit=None,
        allowed_localcocs_exit=None,
        allowed_programs_exit=None,
        allowed_agencies_exit=None,
        exit_project_types=None,
        exit_ssvf_rrh=None,
    )
    _dump_parquet(result, BASELINES / "inbound_default.parquet")


def capture_outbound(df: pd.DataFrame) -> None:
    print("[4/5] outbound_default")
    from src.modules.recidivism.outbound_calculator import run_outbound_recidivism

    result = _unwrap(run_outbound_recidivism)(df, REPORT_START, REPORT_END)
    _dump_parquet(result, BASELINES / "outbound_default.parquet")


def capture_dashboard(df: pd.DataFrame) -> None:
    print("[5/5] dashboard_metrics + dashboard_filtered")
    from src.modules.dashboard.data_utils import ClientMetrics, PHMetrics
    from src.modules.dashboard.filters import _apply_filters_cached

    client = _unwrap(ClientMetrics.batch_calculate_metrics)(
        df, DASHBOARD_PERIOD_START, DASHBOARD_PERIOD_END
    )
    ph = _unwrap(PHMetrics.batch_calculate_ph_metrics)(
        df, DASHBOARD_PERIOD_START, DASHBOARD_PERIOD_END
    )

    payload = {
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
    BASELINES.mkdir(parents=True, exist_ok=True)
    with open(BASELINES / "dashboard_metrics.json", "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"  wrote dashboard_metrics.json  served={len(payload['client_metrics']['served_clients'])}")

    filter_tuple = (
        ("ProjectTypeCode", ("Emergency Shelter – Entry Exit", "PH – Rapid Re-Housing")),
        ("Gender", ("Woman", "Man")),
    )
    filtered = _unwrap(_apply_filters_cached)(df, filter_tuple)
    _dump_parquet(filtered, BASELINES / "dashboard_filtered.parquet")


def main() -> None:
    if not SAMPLE_CSV.exists():
        sys.exit(f"Sample CSV not found at {SAMPLE_CSV}. Run generate_sample first.")
    BASELINES.mkdir(parents=True, exist_ok=True)
    df = capture_loaded_df()
    capture_spm2(df)
    capture_inbound(df)
    capture_outbound(df)
    capture_dashboard(df)
    print("Done. Baselines in", BASELINES)


if __name__ == "__main__":
    main()
