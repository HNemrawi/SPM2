"""
Inbound Recidivism Analysis Logic
---------------------------------
Implements the core analysis for tracking clients returning to homelessness programs.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

from config.constants import PH_CATEGORY


def bucket_return_period(days: Optional[float]) -> str:
    """
    Bucket the number of days into standard return period categories.

    Parameters
    ----------
    days : float or None
        Days since last exit. NaN/None → "New".

    Returns
    -------
    str
        One of "New", "Return < 6 Months", "Return 6–12 Months", etc.
    """
    if pd.isna(days):
        return "New"
    if days <= 180:
        return "Return < 6 Months"
    if days <= 365:
        return "Return 6–12 Months"
    if days <= 730:
        return "Return 12–24 Months"
    return "Return > 24 Months"


def get_last_exit_record(
    row: pd.Series,
    exits: pd.DataFrame,
    lookback_days: int
) -> pd.Series:
    """
    Find the most recent exit within `lookback_days` prior to the entry.

    Classify as:
      - "Returning From Housing (<n> Days Lookback)" if ExitDestinationCat == PH_CATEGORY
      - "Returning (<n> Days Lookback)" otherwise
      - "New" if no exit found

    Parameters
    ----------
    row : pd.Series
        Entry record with at least 'ClientID' and 'ProjectStart'.
    exits : pd.DataFrame
        Exits with 'ClientID', 'ProjectExit', 'ExitDestinationCat'.
    lookback_days : int
        Days before entry to search for exits.

    Returns
    -------
    pd.Series
        All exit columns prefixed with "Exit_", plus "ReturnCategory".
    """
    try:
        client_id = row["ClientID"]
        entry_date = row["ProjectStart"]
        window_start = entry_date - pd.Timedelta(days=lookback_days)

        candidates = exits[
            (exits["ClientID"] == client_id) &
            (exits["ProjectExit"] < entry_date) &
            (exits["ProjectExit"] >= window_start)
        ]

        if candidates.empty:
            # No prior exit → New
            data = {f"Exit_{col}": None for col in exits.columns}
            data["ReturnCategory"] = "New"
            return pd.Series(data)

        # Take the single most recent exit
        last_idx = candidates["ProjectExit"].idxmax()
        last_exit = candidates.loc[last_idx]
        exit_series = last_exit.add_prefix("Exit_")

        # Label based on category
        dest = last_exit["ExitDestinationCat"]
        if dest == PH_CATEGORY:
            label = f"Returning From Housing ({lookback_days} Days Lookback)"
        else:
            label = f"Returning ({lookback_days} Days Lookback)"
        exit_series["ReturnCategory"] = label

        return exit_series

    except Exception as e:
        raise RuntimeError(f"Error in get_last_exit_record for ClientID={row.get('ClientID', 'unknown')}: {e}")


@st.cache_data(show_spinner=False)
def run_return_analysis(
    df: pd.DataFrame,
    report_start: pd.Timestamp,
    report_end: pd.Timestamp,
    days_lookback: int,
    allowed_cocs: Optional[List[str]],
    allowed_localcocs: Optional[List[str]],
    allowed_programs: Optional[List[str]],
    allowed_agencies: Optional[List[str]],
    entry_project_types: Optional[List[str]],
    allowed_cocs_exit: Optional[List[str]],
    allowed_localcocs_exit: Optional[List[str]],
    allowed_programs_exit: Optional[List[str]],
    allowed_agencies_exit: Optional[List[str]],
    exit_project_types: Optional[List[str]]
) -> pd.DataFrame:
    """
    Perform inbound recidivism analysis with entry & exit filtering and lookback.

    Steps
    -----
    1. Filter entries and keep first per client in report window.
    2. Filter exits, drop NaN ProjectExit.
    3. For each entry, call get_last_exit_record.
    4. Prefix entry columns "Enter_", exit columns "Exit_".
    5. Compute days_since_last_exit.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with client records
    report_start : pd.Timestamp
        Start date of the reporting period
    report_end : pd.Timestamp
        End date of the reporting period
    days_lookback : int
        Days before entry to look for exits
    allowed_cocs : Optional[List[str]]
        CoC filter for entries
    allowed_localcocs : Optional[List[str]]
        Local CoC filter for entries
    allowed_programs : Optional[List[str]]
        Program filter for entries
    allowed_agencies : Optional[List[str]]
        Agency filter for entries
    entry_project_types : Optional[List[str]]
        Project types for entries
    allowed_cocs_exit : Optional[List[str]]
        CoC filter for exits
    allowed_localcocs_exit : Optional[List[str]]
        Local CoC filter for exits
    allowed_programs_exit : Optional[List[str]]
        Program filter for exits
    allowed_agencies_exit : Optional[List[str]]
        Agency filter for exits
    exit_project_types : Optional[List[str]]
        Project types for exits

    Returns
    -------
    pd.DataFrame
        One row per entry, with Enter_*/Exit_* fields, ReturnCategory, and days_since_last_exit.
    """
    try:
        # Entry filtering
        raw = df.copy()
        if allowed_cocs and "ProgramSetupCoC" in raw:
            raw = raw[raw["ProgramSetupCoC"].isin(allowed_cocs)]
        if allowed_localcocs and "LocalCoCCode" in raw:
            raw = raw[raw["LocalCoCCode"].isin(allowed_localcocs)]
        if allowed_programs and "ProgramName" in raw:
            raw = raw[raw["ProgramName"].isin(allowed_programs)]
        if allowed_agencies and "AgencyName" in raw:
            raw = raw[raw["AgencyName"].isin(allowed_agencies)]
        if entry_project_types and "ProjectTypeCode" in raw:
            raw = raw[raw["ProjectTypeCode"].isin(entry_project_types)]

        entries = (
            raw.dropna(subset=["ProjectStart"])
               .loc[lambda d: (d["ProjectStart"] >= report_start) & (d["ProjectStart"] <= report_end)]
               .sort_values("ProjectStart")
               .drop_duplicates("ClientID", keep="first")
               .copy()
        )

        # Exit filtering
        exits = df.copy()
        if allowed_cocs_exit and "ProgramSetupCoC" in exits:
            exits = exits[exits["ProgramSetupCoC"].isin(allowed_cocs_exit)]
        if allowed_localcocs_exit and "LocalCoCCode" in exits:
            exits = exits[exits["LocalCoCCode"].isin(allowed_localcocs_exit)]
        if allowed_programs_exit and "ProgramName" in exits:
            exits = exits[exits["ProgramName"].isin(allowed_programs_exit)]
        if allowed_agencies_exit and "AgencyName" in exits:
            exits = exits[exits["AgencyName"].isin(allowed_agencies_exit)]
        if exit_project_types and "ProjectTypeCode" in exits:
            exits = exits[exits["ProjectTypeCode"].isin(exit_project_types)]
        exits = exits.dropna(subset=["ProjectExit"])

        # Apply helper
        exit_info = (
            entries
            .apply(lambda r: get_last_exit_record(r, exits, days_lookback), axis=1)
            .reset_index(drop=True)
        )
        entry_info = entries.add_prefix("Enter_").reset_index(drop=True)

        # Merge and compute days since exit
        merged = pd.concat([entry_info, exit_info], axis=1)
        merged["days_since_last_exit"] = merged.apply(
            lambda r: (r["Enter_ProjectStart"] - r["Exit_ProjectExit"]).days
            if pd.notna(r["Exit_ProjectExit"]) else None,
            axis=1
        )
        return merged

    except Exception as e:
        st.error(f"Error during Return Analysis: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def compute_return_metrics(final_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary metrics for inbound recidivism analysis.
    
    Parameters:
        final_df (pd.DataFrame): Results from inbound recidivism analysis
        
    Returns:
        Dict[str, Any]: Dictionary of metrics
    """
    total = len(final_df)
    
    # "New" is straightforward
    new_count = (final_df["ReturnCategory"] == "New").sum()
    
    # Specifically count rows labeled as "Returning From Housing (X Days Lookback)"
    returning_housing_count = final_df["ReturnCategory"].str.contains("Returning From Housing").sum()
    
    # Count rows that say "Returning (X Days Lookback)" but NOT "Returning From Housing"
    returning_only_mask = (
        final_df["ReturnCategory"].str.contains("Returning") &
        ~final_df["ReturnCategory"].str.contains("From Housing")
    )
    returning_count = returning_only_mask.sum()
    
    # Percentages
    pct_new = (new_count / total * 100) if total else 0
    pct_returning = (returning_count / total * 100) if total else 0
    pct_returning_housing = (returning_housing_count / total * 100) if total else 0
    
    # Timing metrics for return entries
    returned_entries = final_df[final_df["ReturnCategory"] != "New"]
    if not returned_entries.empty and "days_since_last_exit" in returned_entries.columns:
        valid_times = returned_entries.dropna(subset=["days_since_last_exit"])
        median_days = valid_times["days_since_last_exit"].median() if not valid_times.empty else 0
        avg_days = valid_times["days_since_last_exit"].mean() if not valid_times.empty else 0
        max_days = valid_times["days_since_last_exit"].max() if not valid_times.empty else 0
    else:
        median_days = avg_days = max_days = 0
    
    return {
        "Total Entries": total,
        "New": new_count,
        "New (%)": pct_new,
        "Returning": returning_count,
        "Returning (%)": pct_returning,
        "Returning From Housing": returning_housing_count,
        "Returning From Housing (%)": pct_returning_housing,
        "Median Days": median_days,
        "Average Days": avg_days,
        "Max Days": max_days
    }


@st.cache_data(show_spinner=False)
def return_breakdown_analysis(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Group inbound recidivism data by selected columns and compute summary metrics.
    
    Parameters:
        df (pd.DataFrame): Results DataFrame
        group_cols (List[str]): Columns to group by
    
    Returns:
        pd.DataFrame: Aggregated results sorted by total entries.
    """
    grouped = df.groupby(group_cols)
    rows = []
    for group_vals, subdf in grouped:
        group_vals = group_vals if isinstance(group_vals, tuple) else (group_vals,)
        row_data = dict(zip(group_cols, group_vals))
        total = len(subdf)
        new_count = (subdf["ReturnCategory"] == "New").sum()
        ret = (subdf["ReturnCategory"].str.contains("Returning") & ~subdf["ReturnCategory"].str.contains("From Housing")).sum()
        ret_housing = subdf["ReturnCategory"].str.contains("Returning From Housing").sum()

        valid_time = subdf.dropna(subset=["days_since_last_exit"])
        median_days = valid_time["days_since_last_exit"].median() if not valid_time.empty else 0
        avg_days = valid_time["days_since_last_exit"].mean() if not valid_time.empty else 0

        row_data["Total Entries"] = total
        row_data["New (%)"] = f"{new_count} ({(new_count/total*100 if total else 0):.1f}%)"
        row_data["Returning (%)"] = f"{ret} ({(ret/total*100 if total else 0):.1f}%)"
        row_data["Returning From Housing (%)"] = f"{ret_housing} ({(ret_housing/total*100 if total else 0):.1f}%)"
        row_data["Median Days"] = f"{median_days:.1f}"
        row_data["Average Days"] = f"{avg_days:.1f}"
        rows.append(row_data)
    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("Total Entries", ascending=False)
    return out_df