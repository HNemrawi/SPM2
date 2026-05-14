"""Per-(client, exit) parity check: ``_find_earliest_return_fast`` MUST
return the same row (by EnrollmentID) and type as ``_find_earliest_return_legacy``
for every exit in the test fixture. This is the stronger safety net behind
the SPM2 vectorization (F9): the end-to-end harness verifies the aggregate
DataFrame, but this script verifies the inner-loop invariant directly.

Run:  python -m tests.golden.spm2_parity
"""

from __future__ import annotations

# Streamlit shim first.
from tests.golden import _streamlit_shim  # noqa: F401

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SAMPLE_CSV = Path(__file__).parent / "sample_hmis.csv"


def _load_df() -> pd.DataFrame:
    with open(SAMPLE_CSV, "rb") as f:
        buf = BytesIO(f.read())
    buf.name = "sample_hmis.csv"
    from src.core.data.loader import load_and_preprocess_data
    return load_and_preprocess_data.__wrapped__(buf)


def _summarize(result):
    if result is None:
        return None
    row, kind = result
    eid = getattr(row, "EnrollmentID", None)
    ps = getattr(row, "ProjectStart", None)
    return (kind, int(eid) if eid is not None else None, ps)


def main() -> None:
    from src.modules.spm2.calculator import (
        DEFAULT_PROJECT_TYPES,
        _find_earliest_return_fast,
        _find_earliest_return_legacy,
        _prep_client_returns_for_scan,
    )

    df = _load_df()
    print(f"Loaded {len(df):,} rows.")

    report_start = pd.Timestamp("2024-01-01")
    report_end = pd.Timestamp("2024-12-31")
    lookback_days = 730
    return_period = 730

    # Reproduce the same exit-selection the SPM2 pipeline uses.
    exit_min = report_start - pd.Timedelta(days=lookback_days)
    exit_max = report_end - pd.Timedelta(days=lookback_days)
    mask = (
        df["ProjectExit"].notna()
        & df["ProjectTypeCode"].isin(DEFAULT_PROJECT_TYPES)
        & (df["ProjectExit"] >= exit_min)
        & (df["ProjectExit"] <= exit_max)
    )
    df_exits = (
        df[mask]
        .sort_values(["ClientID", "ProjectExit", "EnrollmentID"])
        .drop_duplicates(subset=["ClientID"], keep="first")
    )
    df_for_scan = df[df["ProjectTypeCode"].isin(DEFAULT_PROJECT_TYPES)].sort_values(
        ["ClientID", "ProjectStart", "EnrollmentID"]
    )

    grouped_returns = df_for_scan.groupby("ClientID", sort=False)
    grouped_exits = df_exits.groupby("ClientID", sort=False)

    total_calls = 0
    mismatches = []

    for cid, group_ex in grouped_exits:
        group_ret = (
            grouped_returns.get_group(cid)
            if cid in grouped_returns.groups
            else pd.DataFrame()
        )
        rec_list = _prep_client_returns_for_scan(group_ret)

        for row in group_ex.itertuples(index=False):
            exit_date = row.ProjectExit
            if pd.isna(exit_date):
                continue
            cutoff = min(
                exit_date + pd.Timedelta(days=return_period), report_end
            )
            exit_eid = row.EnrollmentID

            legacy_res = _find_earliest_return_legacy(
                group_ret, exit_date, cutoff, report_end, exit_eid
            )
            if rec_list is None:
                fast_res = None
            else:
                fast_res = _find_earliest_return_fast(
                    rec_list, exit_date, cutoff, report_end, exit_eid
                )

            total_calls += 1
            if _summarize(legacy_res) != _summarize(fast_res):
                mismatches.append(
                    {
                        "cid": int(cid),
                        "exit_eid": int(exit_eid),
                        "exit_date": str(exit_date),
                        "legacy": _summarize(legacy_res),
                        "fast": _summarize(fast_res),
                    }
                )

    print(f"Compared {total_calls} (client, exit) pairs.")
    if mismatches:
        print(f"FAIL: {len(mismatches)} mismatches found:")
        for m in mismatches[:10]:
            print(f"  {m}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
        sys.exit(1)
    print("OK   spm2_parity: legacy and fast match on every exit.")


if __name__ == "__main__":
    main()
