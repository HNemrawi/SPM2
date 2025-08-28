"""
SPM2 Core Analysis Logic
------------------------
Implements the System Performance Measure 2 analysis for housing stability assessment.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.core.constants import DEFAULT_PROJECT_TYPES, PH_PROJECTS


# Helper function for finding the earliest return enrollment
def _find_earliest_return(
    client_df: pd.DataFrame,
    earliest_exit: pd.Timestamp,
    cutoff: pd.Timestamp,
    report_end: pd.Timestamp,
    exit_enrollment_id: int,
) -> Optional[Tuple[pd.Series, str]]:
    """
    Identify the earliest valid return enrollment after an exit enrollment.

    For PH projects, ensure at least a 14-day gap and consider exclusion windows.

    Parameters:
        client_df (pd.DataFrame): Client's enrollment records.
        earliest_exit (pd.Timestamp): The exit date.
        cutoff (pd.Timestamp): Maximum ProjectStart date to consider.
        report_end (pd.Timestamp): End date of the report.
        exit_enrollment_id (int): EnrollmentID to exclude from scanning.

    Returns:
        Optional[Tuple[pd.Series, str]]: (row, type) for a valid return enrollment,
                                         or None if not found.
    """
    if client_df.empty or "ProjectStart" not in client_df.columns:
        return None

    client_df = (
        client_df.dropna(subset=["ProjectStart"])
        .copy()
        .sort_values(["ProjectStart", "EnrollmentID"])
    )
    exclusion_windows = []

    def add_or_extend_window(
        new_window: Tuple[pd.Timestamp, pd.Timestamp],
        windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Merge overlapping windows or add a new one."""
        new_start, new_end = new_window
        updated_windows = []
        for window in windows:
            win_start, win_end = window
            if new_start <= win_end and new_end >= win_start:
                new_start = min(new_start, win_start)
                new_end = max(new_end, win_end)
            else:
                updated_windows.append(window)
        updated_windows.append((new_start, new_end))
        updated_windows.sort(key=lambda w: w[0])
        return updated_windows

    for row in client_df.itertuples(index=False):
        if (
            getattr(row, "EnrollmentID", None) is not None
            and row.EnrollmentID == exit_enrollment_id
        ):
            continue
        if row.ProjectStart < earliest_exit:
            if row.ProjectTypeCode in PH_PROJECTS:
                if (
                    pd.notna(row.ProjectExit)
                    and row.ProjectExit > earliest_exit
                ):
                    new_window = (
                        earliest_exit + pd.Timedelta(days=1),
                        min(
                            row.ProjectExit + pd.Timedelta(days=14), report_end
                        ),
                    )
                    exclusion_windows = add_or_extend_window(
                        new_window, exclusion_windows
                    )
            continue
        if row.ProjectStart > cutoff:
            break
        if row.ProjectTypeCode not in PH_PROJECTS:
            return (row, "Non-PH")
        else:
            gap = (row.ProjectStart - earliest_exit).days
            new_window = (
                row.ProjectStart + pd.Timedelta(days=1),
                min(
                    (
                        (row.ProjectExit + pd.Timedelta(days=14))
                        if pd.notna(row.ProjectExit)
                        else report_end
                    ),
                    report_end,
                ),
            )
            if gap <= 14:
                exclusion_windows = add_or_extend_window(
                    new_window, exclusion_windows
                )
                continue
            else:
                if any(
                    win_start <= row.ProjectStart <= win_end
                    for win_start, win_end in exclusion_windows
                ):
                    exclusion_windows = add_or_extend_window(
                        new_window, exclusion_windows
                    )
                    continue
                return (row, "PH")
    return None


@st.cache_data(show_spinner=False)
def _validate_spm2_inputs(
    df: pd.DataFrame, exiting_projects: Optional[List[str]]
) -> bool:
    """Validate required inputs for SPM2 analysis.

    Returns:
        True if validation passes, False otherwise
    """
    required_cols = [
        "ProjectStart",
        "ProjectExit",
        "ProjectTypeCode",
        "ClientID",
        "EnrollmentID",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(
            f"Error during SPM2 processing: Missing required columns: {missing_cols}"
        )
        return False

    if not exiting_projects:
        st.error(
            "Error during SPM2 processing: No project types selected for exit identification."
        )
        return False

    return True


def _apply_data_filters(
    df: pd.DataFrame,
    exit_cocs: Optional[List[str]],
    exit_localcocs: Optional[List[str]],
    exit_agencies: Optional[List[str]],
    exit_programs: Optional[List[str]],
    exit_ssvf_rrh: Optional[List[str]],
    return_cocs: Optional[List[str]],
    return_localcocs: Optional[List[str]],
    return_agencies: Optional[List[str]],
    return_programs: Optional[List[str]],
    return_ssvf_rrh: Optional[List[str]],
    allowed_continuum: Optional[List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply filters to create exit and return dataframes.

    Returns:
        Tuple of (exit_df, return_df)
    """

    def filter_by_column(
        frame: pd.DataFrame, col: str, values: Optional[List[str]]
    ) -> pd.DataFrame:
        return (
            frame
            if not values or col not in frame.columns
            else frame[frame[col].isin(values)]
        )

    # Filter for exits
    df_exit_base = df.copy()
    df_exit_base = filter_by_column(df_exit_base, "ProgramSetupCoC", exit_cocs)
    df_exit_base = filter_by_column(
        df_exit_base, "LocalCoCCode", exit_localcocs
    )
    df_exit_base = filter_by_column(df_exit_base, "AgencyName", exit_agencies)
    df_exit_base = filter_by_column(df_exit_base, "ProgramName", exit_programs)
    df_exit_base = filter_by_column(df_exit_base, "SSVF_RRH", exit_ssvf_rrh)

    # Filter for returns
    df_return_base = df.copy()
    df_return_base = filter_by_column(
        df_return_base, "ProgramSetupCoC", return_cocs
    )
    df_return_base = filter_by_column(
        df_return_base, "LocalCoCCode", return_localcocs
    )
    df_return_base = filter_by_column(
        df_return_base, "AgencyName", return_agencies
    )
    df_return_base = filter_by_column(
        df_return_base, "ProgramName", return_programs
    )
    df_return_base = filter_by_column(
        df_return_base, "SSVF_RRH", return_ssvf_rrh
    )

    # Apply continuum filter if specified
    if allowed_continuum and "ProgramsContinuumProject" in df.columns:
        df_exit_base = df_exit_base[
            df_exit_base["ProgramsContinuumProject"].isin(allowed_continuum)
        ]
        df_return_base = df_return_base[
            df_return_base["ProgramsContinuumProject"].isin(allowed_continuum)
        ]

    return df_exit_base, df_return_base


def _identify_exit_enrollments(
    df_exit_base: pd.DataFrame,
    report_start: pd.Timestamp,
    report_end: pd.Timestamp,
    lookback_value: int,
    lookback_unit: str,
    exiting_projects: List[str],
    allowed_exit_dest_cats: Optional[List[str]],
    allowed_exit_destinations: Optional[List[str]],
) -> pd.DataFrame:
    """Identify valid exit enrollments within the analysis window.

    Returns:
        DataFrame of valid exit enrollments
    """
    # Calculate exit analysis window
    if lookback_unit == "Days":
        exit_min = report_start - pd.Timedelta(days=lookback_value)
        exit_max = report_end - pd.Timedelta(days=lookback_value)
    else:
        exit_min = report_start - pd.DateOffset(months=lookback_value)
        exit_max = report_end - pd.DateOffset(months=lookback_value)

    # Filter for valid exits
    df_exits = df_exit_base.dropna(subset=["ProjectExit"]).copy()
    df_exits = df_exits[df_exits["ProjectTypeCode"].isin(exiting_projects)]
    df_exits = df_exits[
        (df_exits["ProjectExit"] >= exit_min)
        & (df_exits["ProjectExit"] <= exit_max)
    ]

    # Apply exit destination filters
    if allowed_exit_dest_cats and "ExitDestinationCat" in df_exits.columns:
        df_exits = df_exits[
            df_exits["ExitDestinationCat"].isin(allowed_exit_dest_cats)
        ]

    if (
        allowed_exit_destinations is not None
        and "ExitDestination" in df_exits.columns
    ):
        df_exits = df_exits[
            df_exits["ExitDestination"].isin(allowed_exit_destinations)
        ]

    # Deduplicate to first exit per client
    df_exits = df_exits.sort_values(
        ["ClientID", "ProjectExit", "EnrollmentID"]
    )
    df_exits = df_exits.drop_duplicates(subset=["ClientID"], keep="first")

    return df_exits


def _process_client_exits_and_returns(
    df_exits: pd.DataFrame,
    df_return_base: pd.DataFrame,
    return_projects: List[str],
    return_period: int,
    report_end: pd.Timestamp,
) -> List[Dict[str, Any]]:
    """Process each client's exits and find corresponding returns.

    Returns:
        List of dictionaries containing exit and return information
    """
    # Prepare return data for scanning
    df_for_scan = df_return_base[df_return_base.columns.tolist()].copy()
    df_for_scan = df_for_scan[
        df_for_scan["ProjectTypeCode"].isin(return_projects)
    ]
    grouped_returns = df_for_scan.groupby("ClientID")

    exit_rows = []
    for cid, group_ex in df_exits.groupby("ClientID"):
        group_ex = group_ex.sort_values(["ProjectExit", "EnrollmentID"])
        group_ret = (
            grouped_returns.get_group(cid)
            if cid in grouped_returns.groups
            else pd.DataFrame()
        )

        for row in group_ex.itertuples(index=False):
            row_dict = _process_single_exit(
                row,
                group_ex,
                group_ret,
                df_for_scan,
                return_period,
                report_end,
            )
            exit_rows.append(row_dict)

    return exit_rows


def _process_single_exit(
    row,
    group_ex: pd.DataFrame,
    group_ret: pd.DataFrame,
    df_for_scan: pd.DataFrame,
    return_period: int,
    report_end: pd.Timestamp,
) -> Dict[str, Any]:
    """Process a single exit enrollment and find its return.

    Returns:
        Dictionary containing exit and return data
    """
    # Initialize row dictionary
    row_dict = {f"Exit_{c}": getattr(row, c, None) for c in group_ex.columns}
    earliest_exit_date = getattr(row, "ProjectExit", pd.NaT)

    if pd.isna(earliest_exit_date):
        row_dict["DaysToReturn"] = pd.NA
        row_dict["ReturnCategory"] = "No Return"
        row_dict["ReturnedToHomelessness"] = False
        return row_dict

    # Find return enrollment
    cutoff_date = min(
        earliest_exit_date + pd.Timedelta(days=return_period),
        report_end,
    )

    found_result = _find_earliest_return(
        group_ret,
        earliest_exit_date,
        cutoff_date,
        report_end,
        exit_enrollment_id=row.EnrollmentID,
    )

    if found_result is not None:
        found, _ = found_result
        for cc in df_for_scan.columns:
            row_dict[f"Return_{cc}"] = getattr(found, cc, None)
        row_dict["DaysToReturn"] = (
            found.ProjectStart - earliest_exit_date
        ).days
    else:
        for cc in df_for_scan.columns:
            row_dict[f"Return_{cc}"] = None
        row_dict["DaysToReturn"] = pd.NA

    # Classify return timing
    _classify_return_timing(row_dict)
    return row_dict


def _classify_return_timing(row_dict: Dict[str, Any]) -> None:
    """Classify the return timing and set flags."""
    if pd.isna(row_dict["DaysToReturn"]):
        row_dict["ReturnCategory"] = "No Return"
    else:
        days = row_dict["DaysToReturn"]
        if days <= 180:
            row_dict["ReturnCategory"] = "Return < 6 Months"
        elif days <= 365:
            row_dict["ReturnCategory"] = "Return 6–12 Months"
        elif days <= 730:
            row_dict["ReturnCategory"] = "Return 12–24 Months"
        else:
            row_dict["ReturnCategory"] = "Return > 24 Months"

    row_dict["ReturnedToHomelessness"] = (
        row_dict["ReturnCategory"] != "No Return"
    )


def _add_derived_fields(final_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived fields to the final dataframe.

    Returns:
        DataFrame with derived fields added
    """
    # Add PH Exit flag
    if "Exit_ExitDestinationCat" in final_df.columns:
        final_df["PH_Exit"] = final_df["Exit_ExitDestinationCat"].apply(
            lambda x: x == "Permanent Housing Situations"
        )
    else:
        final_df["PH_Exit"] = False

    # Add age range calculation
    if (
        "Exit_DOB" in final_df.columns
        and "Exit_ProjectExit" in final_df.columns
    ):
        age_days = (
            final_df["Exit_ProjectExit"] - final_df["Exit_DOB"]
        ).dt.days
        age_years = age_days / 365.25

        def age_range(age):
            if pd.isna(age):
                return "Unknown"
            if age < 18:
                return "0 to 17"
            elif age < 25:
                return "18 to 24"
            elif age < 35:
                return "25 to 34"
            elif age < 45:
                return "35 to 44"
            elif age < 55:
                return "45 to 54"
            elif age < 65:
                return "55 to 64"
            return "65 or Above"

        final_df["AgeAtExitRange"] = age_years.apply(age_range)
    else:
        final_df["AgeAtExitRange"] = "Unknown"

    # Add custom program type mapping
    if "Exit_ProjectTypeCode" in final_df.columns:
        final_df["Exit_CustomProgramType"] = (
            final_df["Exit_ProjectTypeCode"]
            .map(
                {
                    "Emergency Shelter – Entry Exit": "Exit was from ES",
                    "Emergency Shelter – Night-by-Night": "Exit was from ES",
                    "PH – Housing Only": "Exit was from PH",
                    "PH – Housing with Services (no disability required for entry)": "Exit was from PH",
                    "PH – Permanent Supportive Housing (disability required for entry)": "Exit was from PH",
                    "PH – Rapid Re-Housing": "Exit was from PH",
                    "Transitional Housing": "Exit was from TH",
                    "Street Outreach": "Exit was from SO",
                    "Safe Haven": "Exit was from SH",
                }
            )
            .fillna("Other")
        )

    return final_df


def run_spm2(
    df: pd.DataFrame,
    report_start: pd.Timestamp,
    report_end: pd.Timestamp,
    lookback_value: int = 730,
    lookback_unit: str = "Days",
    exit_cocs: Optional[List[str]] = None,
    exit_localcocs: Optional[List[str]] = None,
    exit_agencies: Optional[List[str]] = None,
    exit_programs: Optional[List[str]] = None,
    return_cocs: Optional[List[str]] = None,
    return_localcocs: Optional[List[str]] = None,
    return_agencies: Optional[List[str]] = None,
    return_programs: Optional[List[str]] = None,
    allowed_continuum: Optional[List[str]] = None,
    allowed_exit_dest_cats: Optional[List[str]] = None,
    exiting_projects: Optional[List[str]] = None,
    return_projects: Optional[List[str]] = None,
    return_period: int = 730,
    exit_ssvf_rrh: Optional[List[str]] = None,
    return_ssvf_rrh: Optional[List[str]] = None,
    allowed_exit_destinations: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Main logic for SPM2 analysis.

    Orchestrates the complete SPM2 analysis by:
    1. Validating inputs and data structure
    2. Filtering data for exit and return criteria
    3. Identifying valid exit enrollments
    4. Processing exits to find corresponding returns
    5. Adding derived fields and classifications

    Parameters:
        df (pd.DataFrame): Input DataFrame with HMIS data
        report_start (pd.Timestamp): Start date of the reporting period
        report_end (pd.Timestamp): End date of the reporting period
        lookback_value (int): Number of days or months to look back
        lookback_unit (str): Unit for lookback ("Days" or "Months")
        [Additional parameters for filtering exits and returns by various criteria]

    Returns:
        pd.DataFrame: Results DataFrame with exits and matched returns
    """
    exiting_projects = exiting_projects or DEFAULT_PROJECT_TYPES
    return_projects = return_projects or exiting_projects

    if not _validate_spm2_inputs(df, exiting_projects):
        return pd.DataFrame()

    try:
        # Apply filters to create exit and return dataframes
        df_exit_base, df_return_base = _apply_data_filters(
            df,
            exit_cocs,
            exit_localcocs,
            exit_agencies,
            exit_programs,
            exit_ssvf_rrh,
            return_cocs,
            return_localcocs,
            return_agencies,
            return_programs,
            return_ssvf_rrh,
            allowed_continuum,
        )

        # Identify valid exit enrollments
        df_exits = _identify_exit_enrollments(
            df_exit_base,
            report_start,
            report_end,
            lookback_value,
            lookback_unit,
            exiting_projects,
            allowed_exit_dest_cats,
            allowed_exit_destinations,
        )

        # Process exits to find returns
        exit_rows = _process_client_exits_and_returns(
            df_exits,
            df_return_base,
            return_projects,
            return_period,
            report_end,
        )

        # Create final dataframe and add derived fields
        final_df = pd.DataFrame(exit_rows)
        final_df = _add_derived_fields(final_df)

        return final_df

    except Exception as e:
        st.error(f"Error during SPM2 processing: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def compute_summary_metrics(
    final_df: pd.DataFrame, return_period: int = 730
) -> Dict[str, Any]:
    """
    Compute key summary metrics for SPM2 analysis.

    Parameters:
        final_df (pd.DataFrame): Analysis results dataframe
        return_period (int): Maximum days to consider for returns

    Returns:
        Dict[str, Any]: A dictionary with counts, percentages, and day statistics
    """
    total_exited = len(final_df)
    ph_exits = (
        final_df["PH_Exit"].sum() if "PH_Exit" in final_df.columns else 0
    )

    cat_counts = final_df["ReturnCategory"].value_counts(dropna=False)
    r6 = cat_counts.get("Return < 6 Months", 0)
    r6_12 = cat_counts.get("Return 6–12 Months", 0)
    r12_24 = cat_counts.get("Return 12–24 Months", 0)
    r_gt_24 = cat_counts.get("Return > 24 Months", 0)

    total_return = r6 + r6_12 + r12_24 + r_gt_24
    pct_return = (total_return / total_exited * 100) if total_exited else 0
    pct_r6 = (r6 / total_exited * 100) if total_exited else 0
    pct_r6_12 = (r6_12 / total_exited * 100) if total_exited else 0
    pct_r12_24 = (r12_24 / total_exited * 100) if total_exited else 0
    pct_r_gt_24 = (r_gt_24 / total_exited * 100) if total_exited else 0
    pct_ph_exits = (ph_exits / total_exited * 100) if total_exited else 0

    # Calculate return timing metrics
    returned_df = final_df[
        (final_df["ReturnedToHomelessness"])
        & (final_df["DaysToReturn"].notna())
        & (final_df["DaysToReturn"] <= return_period)
    ]
    if not returned_df.empty:
        median_days = returned_df["DaysToReturn"].median()
        avg_days = returned_df["DaysToReturn"].mean()
        p25 = returned_df["DaysToReturn"].quantile(0.25)
        p75 = returned_df["DaysToReturn"].quantile(0.75)
        max_days = returned_df["DaysToReturn"].max()
    else:
        median_days = avg_days = p25 = p75 = max_days = 0

    return {
        "Number of Relevant Exits": total_exited,
        "PH Exits": ph_exits,
        "% PH Exits": pct_ph_exits,
        "Return < 6 Months": r6,
        "% Return < 6M": pct_r6,
        "Return 6–12 Months": r6_12,
        "% Return 6–12M": pct_r6_12,
        "Return 12–24 Months": r12_24,
        "% Return 12–24M": pct_r12_24,
        "Return > 24 Months": r_gt_24,
        "% Return > 24M": pct_r_gt_24,
        "Total Return": total_return,
        "% Return": pct_return,
        "Median Days (<=period)": median_days,
        "Average Days (<=period)": avg_days,
        "DaysToReturn 25th Pctl": p25,
        "DaysToReturn 75th Pctl": p75,
        "DaysToReturn Max": max_days,
    }


@st.cache_data(show_spinner=False)
def breakdown_by_columns(
    final_df: pd.DataFrame, columns: List[str], return_period: int = 730
) -> pd.DataFrame:
    """
    Group the SPM2 DataFrame by the specified columns and compute summary metrics.

    Parameters:
        final_df (pd.DataFrame): Analysis results DataFrame
        columns (List[str]): Columns to group by
        return_period (int): Maximum days to consider for returns

    Returns:
        pd.DataFrame: Aggregated metrics with a combined "count (%)" format
    """
    grouped = final_df.groupby(columns)
    rows = []
    for group_vals, subdf in grouped:
        group_vals = (
            group_vals if isinstance(group_vals, tuple) else (group_vals,)
        )
        row_data = {col: val for col, val in zip(columns, group_vals)}
        m = compute_summary_metrics(subdf, return_period)

        if return_period <= 730:
            m["Return > 24 Months"] = None
            m["% Return > 24M"] = None

        row_data["Number of Relevant Exits"] = m["Number of Relevant Exits"]
        row_data["Total Return"] = m["Total Return"]
        row_data["% Return"] = f"{m['% Return']:.1f}%"

        pairs = [
            ("PH Exits", "% PH Exits"),
            ("Return < 6 Months", "% Return < 6M"),
            ("Return 6–12 Months", "% Return 6–12M"),
            ("Return 12–24 Months", "% Return 12–24M"),
            ("Return > 24 Months", "% Return > 24M"),
        ]
        for count_col, pct_col in pairs:
            cnt = m.get(count_col)
            pct = m.get(pct_col)
            if cnt is not None and pct is not None:
                row_data[count_col] = f"{cnt} ({pct:.1f}%)"
            else:
                row_data[count_col] = None

        row_data["Median Days"] = f"{m['Median Days (<=period)']:.1f}"
        row_data["Average Days"] = f"{m['Average Days (<=period)']:.1f}"

        rows.append(row_data)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(
        by="Number of Relevant Exits",
        key=lambda x: x.astype(int),
        ascending=False,
    )
    return out_df
