"""
Outbound Recidivism Analysis Logic
----------------------------------
Implements analysis of clients returning to homelessness after exit.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

from src.core.constants import NON_HOMELESS_PROJECTS, PH_PROJECTS


# Internal helper functions
def _find_earliest_return_any(
    client_df: pd.DataFrame,
    exit_date: pd.Timestamp,
    exit_enrollment_id: int,
) -> Optional[Tuple[pd.Series, int]]:
    """
    Return the first enrollment (any project type) that begins strictly
    after *exit_date* (skipping the reference enrollment).
    """
    try:
        df_next = (
            client_df.copy()
            .dropna(subset=["ProjectStart"])
            .loc[lambda d: d["ProjectStart"] > exit_date]
            .sort_values(["ProjectStart", "EnrollmentID"])
        )
        for row in df_next.itertuples(index=False):
            if row.EnrollmentID == exit_enrollment_id:
                continue
            gap_days: int = (row.ProjectStart - exit_date).days
            return (pd.Series(row._asdict()), gap_days)
        return None
    except Exception as exc:
        st.error(f"Error in _find_earliest_return_any: {exc}")
        return None


def _find_earliest_return_homeless(
    client_df: pd.DataFrame,
    exit_date: pd.Timestamp,
    exit_enrollment_id: int,
) -> Optional[Tuple[pd.Series, bool, int]]:
    """
    HUD-compliant "Return to Homelessness" finder.

    Scans a client's history after a given exit for the earliest qualifying return:
      - Excludes non-homeless projects.
      - Builds exclusion windows for short (<15d) PH stays.
      - **New**: skips PH records where ProjectStart == HouseholdMoveInDate.

    Parameters
    ----------
    client_df : pd.DataFrame
        Enrollment records with columns including:
        ['ProjectTypeCode', 'ProjectStart', 'ProjectExit',
         'EnrollmentID', 'HouseholdMoveInDate', ...].
    exit_date : pd.Timestamp
        The date of the exit event to compare against.
    exit_enrollment_id : int
        The EnrollmentID of the exit record (to skip).

    Returns
    -------
    Optional[Tuple[pd.Series, bool, int]]
        (row_series, True, gap_days) for the first valid return, else None.
    """
    try:
        # Prepare next enrollments after exit_date
        df_next = (
            client_df.dropna(subset=["ProjectStart"])
            .loc[lambda d: ~d["ProjectTypeCode"].isin(NON_HOMELESS_PROJECTS)]
            .loc[lambda d: d["ProjectStart"] > exit_date]
            .sort_values(["ProjectStart", "EnrollmentID"])
        )
        if df_next.empty:
            return None

        exclusion_windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

        def _merge_window(
            new_win: Tuple[pd.Timestamp, pd.Timestamp],
            windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
        ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
            new_start, new_end = new_win
            merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
            for ws, we in windows:
                if new_start <= we and new_end >= ws:
                    new_start = min(new_start, ws)
                    new_end = max(new_end, we)
                else:
                    merged.append((ws, we))
            merged.append((new_start, new_end))
            merged.sort(key=lambda w: w[0])
            return merged

        # Iterate chronologically
        for row in df_next.itertuples(index=False):
            if row.EnrollmentID == exit_enrollment_id:
                continue

            is_ph = row.ProjectTypeCode in PH_PROJECTS

            # Skip PH entries that coincide with move-in date
            if is_ph and hasattr(row, "HouseholdMoveInDate"):
                hmi = row.HouseholdMoveInDate
                if pd.notna(hmi) and row.ProjectStart == hmi:
                    continue

            gap_days = (row.ProjectStart - exit_date).days

            if is_ph:
                # Exclude short stays (≤14d)
                if gap_days <= 14:
                    if pd.notna(row.ProjectExit):
                        exclusion_windows = _merge_window(
                            (
                                row.ProjectStart + pd.Timedelta(days=1),
                                row.ProjectExit + pd.Timedelta(days=14),
                            ),
                            exclusion_windows,
                        )
                    continue

                # Exclude if within any prior exclusion window
                if any(
                    ws <= row.ProjectStart <= we
                    for ws, we in exclusion_windows
                ):
                    if pd.notna(row.ProjectExit):
                        exclusion_windows = _merge_window(
                            (
                                row.ProjectStart + pd.Timedelta(days=1),
                                row.ProjectExit + pd.Timedelta(days=14),
                            ),
                            exclusion_windows,
                        )
                    continue

                # Qualifies as a PH return
                return (pd.Series(row._asdict()), True, gap_days)

            # Non-PH return qualifies immediately
            return (pd.Series(row._asdict()), True, gap_days)

        return None

    except Exception as exc:
        st.error(f"Error in _find_earliest_return_homeless: {exc}")
        return None


@st.cache_data(show_spinner=False)
def run_outbound_recidivism(
    df: pd.DataFrame,
    report_start: pd.Timestamp,
    report_end: pd.Timestamp,
    *,
    exit_cocs: Optional[List[str]] = None,
    exit_localcocs: Optional[List[str]] = None,
    exit_agencies: Optional[List[str]] = None,
    exit_programs: Optional[List[str]] = None,
    exit_ssvf_rrh: Optional[List[str]] = None,
    return_cocs: Optional[List[str]] = None,
    return_localcocs: Optional[List[str]] = None,
    return_agencies: Optional[List[str]] = None,
    return_programs: Optional[List[str]] = None,
    return_ssvf_rrh: Optional[List[str]] = None,
    allowed_continuum: Optional[List[str]] = None,
    allowed_exit_dest_cats: Optional[List[str]] = None,
    allowed_exit_destinations: Optional[List[str]] = None,
    exiting_projects: Optional[List[str]] = None,
    return_projects: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    End‑to‑end ETL for Outbound Recidivism.

    Parameters:
        df (pd.DataFrame): Input DataFrame with client records
        report_start (pd.Timestamp): Start date of the reporting period
        report_end (pd.Timestamp): End date of the reporting period
        exit_cocs (Optional[List[str]]): CoC filter for exits
        exit_localcocs (Optional[List[str]]): Local CoC filter for exits
        exit_agencies (Optional[List[str]]): Agency filter for exits
        exit_programs (Optional[List[str]]): Program filter for exits
        return_cocs (Optional[List[str]]): CoC filter for returns
        return_localcocs (Optional[List[str]]): Local CoC filter for returns
        return_agencies (Optional[List[str]]): Agency filter for returns
        return_programs (Optional[List[str]]): Program filter for returns
        allowed_continuum (Optional[List[str]]): Continuum filter
        allowed_exit_dest_cats (Optional[List[str]]): Exit destination category filter
        exiting_projects (Optional[List[str]]): Project types for exits
        return_projects (Optional[List[str]]): Project types for returns

    Returns:
        pd.DataFrame: Results with exit and return information
    """
    try:
        req_cols: Set[str] = {
            "ClientID",
            "ProjectStart",
            "ProjectExit",
            "ProjectTypeCode",
            "EnrollmentID",
        }
        missing = sorted(req_cols.difference(df.columns))
        if missing:
            st.error(f"Missing required columns: {missing}")
            return pd.DataFrame()

        exiting_projects = exiting_projects or sorted(
            df["ProjectTypeCode"].dropna().unique()
        )
        return_projects = return_projects or exiting_projects
        if not exiting_projects:
            st.error("No project types selected for Exits.")
            return pd.DataFrame()

        def _opt_filter(
            frame: pd.DataFrame, col: str, allowed: Optional[List[str]]
        ) -> pd.DataFrame:
            return (
                frame
                if not allowed or col not in frame.columns
                else frame[frame[col].isin(allowed)]
            )

        # EXIT filter
        df_exit = df.copy()
        df_exit = _opt_filter(df_exit, "ProgramSetupCoC", exit_cocs)
        df_exit = _opt_filter(df_exit, "LocalCoCCode", exit_localcocs)
        df_exit = _opt_filter(df_exit, "AgencyName", exit_agencies)
        df_exit = _opt_filter(df_exit, "ProgramName", exit_programs)
        df_exit = _opt_filter(df_exit, "SSVF_RRH", exit_ssvf_rrh)
        df_exit = _opt_filter(
            df_exit, "ProgramsContinuumProject", allowed_continuum
        )
        df_exit = _opt_filter(
            df_exit, "ExitDestinationCat", allowed_exit_dest_cats
        )
        df_exit = _opt_filter(
            df_exit, "ExitDestination", allowed_exit_destinations
        )

        df_exit = (
            df_exit.dropna(subset=["ProjectExit"])
            .loc[lambda d: d["ProjectTypeCode"].isin(exiting_projects)]
            .loc[
                lambda d: (d["ProjectExit"] >= report_start)
                & (d["ProjectExit"] <= report_end)
            ]
        )
        if df_exit.empty:
            return pd.DataFrame()

        df_exit = (
            df_exit.sort_values(["ClientID", "ProjectExit", "EnrollmentID"])
            .groupby("ClientID", as_index=False)
            .tail(1)
        )

        # RETURN filter
        df_return = df.copy()
        df_return = _opt_filter(df_return, "ProgramSetupCoC", return_cocs)
        df_return = _opt_filter(df_return, "LocalCoCCode", return_localcocs)
        df_return = _opt_filter(df_return, "AgencyName", return_agencies)
        df_return = _opt_filter(df_return, "ProgramName", return_programs)
        df_return = _opt_filter(df_return, "SSVF_RRH", return_ssvf_rrh)
        df_return = _opt_filter(
            df_return, "ProgramsContinuumProject", allowed_continuum
        )
        df_return = df_return.loc[
            df_return["ProjectTypeCode"].isin(return_projects)
        ]

        grouped_return = df_return.groupby("ClientID")

        rows: List[Dict[str, Any]] = []
        for exit_row in df_exit.itertuples(index=False):
            cid = exit_row.ClientID
            exit_date = exit_row.ProjectExit
            exit_enroll_id = exit_row.EnrollmentID

            row_dict = {
                f"Exit_{col}": getattr(exit_row, col, pd.NA)
                for col in df_exit.columns
            }
            for col in df_return.columns:
                row_dict[f"Return_{col}"] = pd.NA

            row_dict.update(
                {
                    "HasReturn": False,
                    "ReturnToHomelessness": False,
                    "DaysToReturnEnrollment": pd.NA,
                }
            )

            if cid not in grouped_return.groups:
                rows.append(row_dict)
                continue

            client_enrolls = grouped_return.get_group(cid)

            # Any Return
            result_any = _find_earliest_return_any(
                client_enrolls, exit_date, exit_enroll_id
            )
            if result_any:
                next_row_any, gap_any = result_any
                row_dict["HasReturn"] = True
                row_dict["DaysToReturnEnrollment"] = gap_any
                for col in df_return.columns:
                    row_dict[f"Return_{col}"] = next_row_any.get(col, pd.NA)

            # Return to Homelessness
            result_rth = _find_earliest_return_homeless(
                client_enrolls, exit_date, exit_enroll_id
            )
            if result_rth:
                _, is_return, _ = result_rth
                row_dict["ReturnToHomelessness"] = is_return

            rows.append(row_dict)

        final_df = pd.DataFrame(rows)
        final_df["PH_Exit"] = final_df["Exit_ExitDestinationCat"].eq(
            "Permanent Housing Situations"
        )

        # Age buckets
        if {"Exit_DOB", "Exit_ProjectExit"}.issubset(final_df.columns):
            age_years = (
                final_df["Exit_ProjectExit"] - final_df["Exit_DOB"]
            ).dt.days / 365.25

            def _bucket(age):
                if pd.isna(age):
                    return "Unknown"
                for bound, label in zip(
                    [18, 25, 35, 45, 55, 65],
                    ["0–17", "18–24", "25–34", "35–44", "45–54", "55–64"],
                ):
                    if age < bound:
                        return label
                return "65+"

            final_df["AgeAtExitRange"] = age_years.apply(_bucket)
        else:
            final_df["AgeAtExitRange"] = "Unknown"

        return final_df

    except Exception as exc:
        st.error(f"Error in run_outbound_recidivism: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def compute_summary_metrics(final_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute key summary metrics for outbound recidivism.

    Parameters:
        final_df (pd.DataFrame): Results DataFrame

    Returns:
        Dict[str, Any]: Dictionary of metrics
    """
    if final_df.empty:
        return {
            "Number of Relevant Exits": 0,
            "Total Exits to PH": 0,
            "Return": 0,
            "Return %": 0.0,
            "Return to Homelessness": 0,
            "% Return to Homelessness": 0.0,
            "Median Days": 0,
            "Average Days": 0,
            "Max Days": 0,
        }
    total_exits = len(final_df)
    total_ph_exits = final_df["PH_Exit"].sum()
    return_count = final_df["HasReturn"].sum()
    return_pct = return_count / total_exits * 100
    rth_count = final_df.loc[final_df["PH_Exit"], "ReturnToHomelessness"].sum()
    rth_pct = (rth_count / total_ph_exits * 100) if total_ph_exits else 0.0
    rtn_days = final_df["DaysToReturnEnrollment"].dropna()
    return {
        "Number of Relevant Exits": total_exits,
        "Total Exits to PH": int(total_ph_exits),
        "Return": int(return_count),
        "Return %": float(return_pct),
        "Return to Homelessness": int(rth_count),
        "% Return to Homelessness": float(rth_pct),
        "Median Days": float(rtn_days.median() if not rtn_days.empty else 0),
        "Average Days": float(rtn_days.mean() if not rtn_days.empty else 0),
        "Max Days": float(rtn_days.max() if not rtn_days.empty else 0),
    }


@st.cache_data(show_spinner=False)
def breakdown_by_columns(
    final_df: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    """
    Group outbound recidivism data by selected columns and compute metrics.

    Parameters:
        final_df (pd.DataFrame): Results DataFrame
        columns (List[str]): Columns to group by

    Returns:
        pd.DataFrame: Aggregated results
    """
    if final_df.empty or not columns:
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    for keys, grp in final_df.groupby(columns):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(columns, keys))
        m = compute_summary_metrics(grp)
        row.update(
            {
                "Relevant Exits": m["Number of Relevant Exits"],
                "Exits to Permanent Housing": m["Total Exits to PH"],
                "Return": f"{m['Return']} ({m['Return %']:.1f}%)",
                # CHANGED
                "Returns to Homelessness (From PH)": f"{m['Return to Homelessness']} ({m['% Return to Homelessness']:.1f}%)",
                "Median Days to Return": f"{m['Median Days']:.1f}",
                "Average Days to Return": f"{m['Average Days']:.1f}",
                "Max Days to Return": f"{m['Max Days']:.0f}",
            }
        )
        records.append(row)
    return pd.DataFrame(records).sort_values("Relevant Exits", ascending=False)
