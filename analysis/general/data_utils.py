"""
Data processing utilities for HMIS dashboard.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
from pandas import DataFrame, Series, Timestamp
from core.data_loader import load_and_preprocess_data

# ────────────────────────── Constants ────────────────────────── #

REQUIRED_BASE_COLS: List[str] = ["ClientID", "ProjectStart", "ProjectExit"]

# Project type classifications
PH_PROJECTS: Set[str] = {
    "PH – Housing Only",
    "PH – Housing with Services (no disability required for entry)",
    "PH – Permanent Supportive Housing (disability required for entry)",
    "PH – Rapid Re-Housing",
}

NON_HOMELESS_PROJECTS: Set[str] = {
    "Coordinated Entry",
    "Day Shelter",
    "Homelessness Prevention",
    "Other",
    "Services Only",
}

# Demographic dimensions for breakdowns
DEMOGRAPHIC_DIMENSIONS: List[Tuple[str, str]] = [
    ("Race / Ethnicity", "RaceEthnicity"),
    ("Gender", "Gender"),
    ("Entry Age Tier", "AgeTieratEntry"),
    ("Program CoC", "ProgramSetupCoC"),
    ("Local CoC", "LocalCoCCode"),
    ("Agency Name", "AgencyName"),
    ("Program Name", "ProgramName"),
    ("Project Type", "ProjectTypeCode"),
    ("Household Type", "HouseholdType"),
    ("Head of Household", "IsHeadOfHousehold"),
    ("Veteran Status", "VeteranStatus"),
    ("Chronic Homelessness", "CHStartHousehold"),
    ("Currently Fleeing DV", "CurrentlyFleeingDV"),
]

# Time frequency mapping for trend analysis
FREQUENCY_MAP: Dict[str, str] = {
    "Days": "D",
    "Weeks": "W",
    "Months": "M",
    "Quarters": "Q",
    "Years": "Y",
}

# ────────────────────────── Helper Functions ────────────────────────── #

def _need(df: DataFrame, cols: List[str]) -> None:
    """
    Check that required columns exist in DataFrame.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame to check
    cols : list
        Required column names
        
    Raises:
    -------
    KeyError
        If any required columns are missing
    """
    missing: List[str] = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s): {', '.join(missing)}")

def _safe_div(a: Union[int, float], b: Union[int, float], 
             default: float = 0.0, multiplier: float = 1.0) -> float:
    """
    Safely divide two numbers with optional scaling.
    
    Parameters:
    -----------
    a : numeric
        Numerator
    b : numeric
        Denominator
    default : float
        Default value if denominator is zero
    multiplier : float
        Value to multiply the result by (e.g., 100 for percentages)
        
    Returns:
    --------
    float
        Result of division
    """
    return round((a / b) * multiplier if b else default, 1)

def calc_delta(current: float, previous: float) -> Tuple[float, float]:
    """
    Calculate absolute and percentage change.
    
    Parameters:
    -----------
    current : float
        Current value
    previous : float
        Previous value
        
    Returns:
    --------
    tuple
        (absolute_change, percentage_change)
    """
    try:
        change = current - previous
        pct = round(change / previous * 100, 1) if previous else 0.0
        return change, pct
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0, 0.0

# ────────────────────────── Data Loading ────────────────────────── #

# Data Loading & Caching
@st.cache_data(show_spinner="Loading and preprocessing data…")
def cached_load(upload_or_df) -> pd.DataFrame:
    if isinstance(upload_or_df, pd.DataFrame):
        df = upload_or_df.copy()
    else:
        df = load_and_preprocess_data(upload_or_df)
    
    # Coerce any existing datetime columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
    return df

# ────────────────────────── Core Metrics ────────────────────────── #

@st.cache_data(show_spinner=False)
def served_clients(df: DataFrame, start: Timestamp, end: Timestamp) -> Set[int]:
    """
    Get unique clients active within date range.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame of client enrollments
    start : Timestamp
        Start date
    end : Timestamp
        End date
        
    Returns:
    --------
    set
        Set of ClientIDs
    """
    _need(df, REQUIRED_BASE_COLS)
    mask = ((df["ProjectExit"] >= start) | df["ProjectExit"].isna()) & (
        df["ProjectStart"] <= end
    )
    return set(df.loc[mask, "ClientID"])

@st.cache_data(show_spinner=False)
def households_served(df: DataFrame, start: Timestamp, end: Timestamp) -> int:
    """
    Count unique households (heads only) active in date range.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame of client enrollments
    start : Timestamp
        Start date
    end : Timestamp
        End date
        
    Returns:
    --------
    int
        Count of unique households
    """
    _need(df, REQUIRED_BASE_COLS + ["IsHeadOfHousehold"])
    mask = ((df["ProjectExit"] >= start) | df["ProjectExit"].isna()) & (
        df["ProjectStart"] <= end
    )
    mask &= df["IsHeadOfHousehold"].astype(str).str.strip().str.lower() == "yes"
    return df.loc[mask, "ClientID"].nunique()

@st.cache_data(show_spinner=False)
def inflow(df: DataFrame, start: Timestamp, end: Timestamp) -> Set[int]:
    """
    Get clients entering during date range who weren't in any programs 
    the day before the report start.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame of client enrollments
    start : Timestamp
        Start date
    end : Timestamp
        End date
        
    Returns:
    --------
    set
        Set of ClientIDs
    """
    _need(df, REQUIRED_BASE_COLS)
    
    # Get entries during the period
    entries_mask = df["ProjectStart"].between(start, end)
    entries_df = df[entries_mask]
    
    # Check who was active the day before
    day_before = start - pd.Timedelta(days=1)
    active_before_mask = (
        ((df["ProjectExit"] >= day_before) | df["ProjectExit"].isna()) & 
        (df["ProjectStart"] <= day_before)
    )
    active_before_ids = set(df.loc[active_before_mask, "ClientID"])
    
    # Return entries who weren't active before
    entry_ids = set(entries_df["ClientID"])
    return entry_ids - active_before_ids

# Replace the existing outflow function
@st.cache_data(show_spinner=False)
def outflow(df: DataFrame, start: Timestamp, end: Timestamp) -> Set[int]:
    """
    Get clients exiting during date range who aren't in any programs 
    on the last day of the period.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame of client enrollments
    start : Timestamp
        Start date
    end : Timestamp
        End date
        
    Returns:
    --------
    set
        Set of ClientIDs
    """
    _need(df, REQUIRED_BASE_COLS)
    
    # Get exits during the period
    exits_mask = df["ProjectExit"].between(start, end)
    exits_df = df[exits_mask]
    
    # Check who is still active on the last day
    active_on_end_mask = (
        ((df["ProjectExit"] > end) | df["ProjectExit"].isna()) & 
        (df["ProjectStart"] <= end)
    )
    still_active_ids = set(df.loc[active_on_end_mask, "ClientID"])
    
    # Return exits who aren't still active
    exit_ids = set(exits_df["ClientID"])
    return exit_ids - still_active_ids

@st.cache_data(show_spinner=False)
def ph_exit_clients(df: DataFrame, start: Timestamp, end: Timestamp) -> Set[int]:
    """
    Get clients exiting to permanent housing during date range.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame of client enrollments
    start : Timestamp
        Start date
    end : Timestamp
        End date
        
    Returns:
    --------
    set
        Set of ClientIDs
    """
    _need(df, ["ClientID", "ProjectExit", "ExitDestinationCat"])
    
    # Get PH exits during the period - using same mask construction as ph_exit_rate
    ph_exits_mask = (
        (df["ExitDestinationCat"] == "Permanent Housing Situations") & 
        df["ProjectExit"].between(start, end)
    )
    
    # Get unique client IDs, excluding any NaN values
    client_ids = df.loc[ph_exits_mask, "ClientID"].dropna().unique()
    
    return set(client_ids)

@st.cache_data(show_spinner=False)
def ph_exit_rate(df: DataFrame, start: Timestamp, end: Timestamp) -> float:
    """
    Calculate PH exits ÷ total exits as percentage.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame of client enrollments
    start : Timestamp
        Start date of reporting period
    end : Timestamp
        End date of reporting period
        
    Returns:
    --------
    float
        Percentage of all exits that were to permanent housing
    """
    _need(df, ["ClientID", "ProjectExit", "ExitDestinationCat"])
    
    # Get all exits during the period
    all_exits_mask = df["ProjectExit"].between(start, end)
    total_exits = df.loc[all_exits_mask, "ClientID"].nunique()
    
    # Get PH exits during the period
    ph_exits_mask = (
        (df["ExitDestinationCat"] == "Permanent Housing Situations") & 
        df["ProjectExit"].between(start, end)
    )
    ph_exits = df.loc[ph_exits_mask, "ClientID"].nunique()
    
    return _safe_div(ph_exits, total_exits, multiplier=100)

@st.cache_data(show_spinner=False)
def period_comparison(
    df: DataFrame, 
    current_start: Timestamp, 
    current_end: Timestamp,
    previous_start: Timestamp,
    previous_end: Timestamp
) -> Dict[str, Set[int]]:
    """
    Compare client populations between two time periods.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame of client enrollments
    current_start : Timestamp
        Start of current period
    current_end : Timestamp
        End of current period
    previous_start : Timestamp
        Start of previous period
    previous_end : Timestamp
        End of previous period
        
    Returns:
    --------
    dict
        Dictionary with:
        - 'current_clients': Clients active in current period
        - 'previous_clients': Clients active in previous period
        - 'carryover': Clients active in both periods
        - 'new': Clients in current but not previous
        - 'exited': Clients in previous but not current
    """
    # Get clients active in each period
    current_clients = served_clients(df, current_start, current_end)
    previous_clients = served_clients(df, previous_start, previous_end)
    
    # Calculate the different groups
    carryover = current_clients.intersection(previous_clients)
    new = current_clients - previous_clients
    exited = previous_clients - current_clients
    
    return {
        'current_clients': current_clients,
        'previous_clients': previous_clients,
        'carryover': carryover,
        'new': new,
        'exited': exited
    }

@st.cache_data(show_spinner=False)
def return_after_exit(
    df_filtered: pd.DataFrame,
    full_df: pd.DataFrame,
    s: pd.Timestamp,
    e: pd.Timestamp,
    return_window: int = 730
) -> Set[int]:
    """
    Identify clients who exited to PH in the reporting period and returned
    to homelessness within return_window days using HUD-compliant logic.
    
    Parameters:
    -----------
    df_filtered : DataFrame
        The filtered dataset (used to select PH exits in the reporting window)
    full_df : DataFrame
        The full dataset (used to scan for re-enrollments after exit)
    s : Timestamp
        Start of the reporting period
    e : Timestamp
        End of the reporting period
    return_window : int, optional
        Number of days after exit to check for returns (default is 730)
        
    Returns:
    --------
    Set[int]
        Set of ClientIDs who returned to homelessness within return_window days
        of a PH exit
    """
    _need(df_filtered, REQUIRED_BASE_COLS + ["ExitDestinationCat"])
    _need(full_df, REQUIRED_BASE_COLS + ["ProjectTypeCode", "HouseholdMoveInDate", "EnrollmentID"])
    
    # Get PH exits during the reporting period
    exits = df_filtered[
        (df_filtered["ProjectExit"].between(s, e)) &
        (df_filtered["ExitDestinationCat"] == "Permanent Housing Situations")
    ].copy()
    
    if exits.empty:
        return set()
    
    # Get unique clients who had qualifying exits
    exited_clients = set(exits["ClientID"])
    returners = set()
    
    # Check each client for returns using HUD logic
    for client_id in exited_clients:
        # Get all enrollments for this client
        client_enrollments = full_df[full_df["ClientID"] == client_id].copy()
        
        # Get this client's qualifying exits
        client_exits = exits[exits["ClientID"] == client_id].sort_values("ProjectExit")
        
        # For each qualifying exit, scan forward for returns
        for _, exit_row in client_exits.iterrows():
            exit_date = exit_row["ProjectExit"]
            exit_enrollment_id = exit_row.get("EnrollmentID", -1)
            
            # Check if client returned using HUD logic
            if _check_return_to_homelessness(
                client_enrollments, 
                exit_date, 
                exit_enrollment_id, 
                return_window
            ):
                returners.add(client_id)
                break  # Stop after first valid return
    
    return returners

def _check_return_to_homelessness(
    client_df: pd.DataFrame,
    exit_date: pd.Timestamp,
    exit_enrollment_id: int,
    max_days: int = 730
) -> bool:
    """
    Check if a client returns to homelessness after an exit using HUD logic.
    
    Parameters:
    -----------
    client_df : DataFrame
        All enrollments for a single client
    exit_date : Timestamp
        The PH exit date to check returns after
    exit_enrollment_id : int
        The enrollment ID of the exit (to skip)
    max_days : int
        Maximum days to check for returns
        
    Returns:
    --------
    bool
        True if client returned to homelessness, False otherwise
    """
    # Filter to enrollments after exit date within window
    df_next = client_df[
        (client_df["ProjectStart"] > exit_date) &
        (client_df["ProjectStart"] <= exit_date + pd.Timedelta(days=max_days)) &
        (~client_df["ProjectTypeCode"].isin(NON_HOMELESS_PROJECTS))
    ].sort_values(["ProjectStart", "EnrollmentID"])
    
    if df_next.empty:
        return False
    
    # Track exclusion windows for short PH stays
    exclusion_windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    
    def _merge_window(
        new_win: Tuple[pd.Timestamp, pd.Timestamp],
        windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Merge overlapping exclusion windows."""
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
    
    # Check each potential return enrollment
    for _, row in df_next.iterrows():
        if row.get("EnrollmentID", -1) == exit_enrollment_id:
            continue
            
        is_ph = row["ProjectTypeCode"] in PH_PROJECTS
        gap_days = (row["ProjectStart"] - exit_date).days
        
        if is_ph:
            # Skip PH entries where ProjectStart == HouseholdMoveInDate
            if pd.notna(row.get("HouseholdMoveInDate")) and row["ProjectStart"] == row["HouseholdMoveInDate"]:
                continue
                
            # Exclude short stays (≤14d)
            if gap_days <= 14:
                if pd.notnull(row.get("ProjectExit")):
                    exclusion_windows = _merge_window(
                        (row["ProjectStart"] + pd.Timedelta(days=1),
                         row["ProjectExit"] + pd.Timedelta(days=14)),
                        exclusion_windows
                    )
                continue
                
            # Exclude if within any prior exclusion window
            if any(ws <= row["ProjectStart"] <= we for ws, we in exclusion_windows):
                if pd.notnull(row.get("ProjectExit")):
                    exclusion_windows = _merge_window(
                        (row["ProjectStart"] + pd.Timedelta(days=1),
                         row["ProjectExit"] + pd.Timedelta(days=14)),
                        exclusion_windows
                    )
                continue
                
            # This is a valid PH return to homelessness
            return True
        else:
            # Non-PH enrollment is an immediate return to homelessness
            return True
    
    return False

# ────────────────────────── Demographic Helpers ────────────────────────── #

def category_counts(df: DataFrame, ids: Set[int], group_col: str, name: str) -> Series:
    """
    Return value counts for group_col over ids.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame of client enrollments
    ids : set
        Set of ClientIDs to filter by
    group_col : str
        Column name to group by
    name : str
        Name for the resulting Series
        
    Returns:
    --------
    Series
        Count of clients in each group
    """
    if group_col not in df.columns:
        st.warning(f"Column '{group_col}' not found.")
        return pd.Series(dtype=int, name=name)

    mask = df["ClientID"].isin(ids)
    return (
        df[mask]
        .groupby(group_col)["ClientID"]
        .nunique()
        .rename(name)
    )

# ────────────────────────── Time Series Helpers ────────────────────────── #

def recalculated_metric_time_series(
    df: DataFrame,
    metric_func,
    start: Timestamp,
    end: Timestamp,
    freq: str = "M",
) -> DataFrame:
    """
    Calculate metric for each time period between start and end.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame of client enrollments
    metric_func : callable
        Function that takes (df, start, end) and returns a set of ClientIDs
    start : Timestamp
        Start date
    end : Timestamp
        End date
    freq : str, optional
        Frequency for time bucketing ("D" for daily, "W" for weekly, etc.)
        
    Returns:
    --------
    DataFrame
        DataFrame with columns: bucket, count
    """
    periods = pd.period_range(start=start, end=end, freq=freq)
    rows: List[Dict[str, Any]] = []
    for period in periods:
        period_start = period.start_time
        period_end = period.end_time
        ids = metric_func(df, period_start, period_end)
        rows.append({"bucket": period_start, "count": len(ids)})
    return pd.DataFrame(rows)

def recalculated_metric_time_series_by_group(
    df: DataFrame,
    metric_func, 
    group_col: str,
    start: Timestamp,
    end: Timestamp,
    freq: str = "M",
) -> DataFrame:
    """
    Calculate metric for each time period, broken down by group.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame of client enrollments
    metric_func : callable
        Function that takes (df, start, end) and returns a set of ClientIDs
    group_col : str
        Column name to group by
    start : Timestamp
        Start date
    end : Timestamp
        End date
    freq : str, optional
        Frequency for time bucketing ("D" for daily, "W" for weekly, etc.)
        
    Returns:
    --------
    DataFrame
        DataFrame with columns: bucket, group, count
    """
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found.")

    periods = pd.period_range(start=start, end=end, freq=freq)
    rows: List[Dict[str, Any]] = []

    for grp, sub in df.groupby(group_col, dropna=True):
        for period in periods:
            period_start = period.start_time
            period_end = period.end_time
            ids = metric_func(sub, period_start, period_end)
            rows.append(
                {"bucket": period_start, "group": grp, "count": len(ids)}
            )

    return pd.DataFrame(rows)

def calculate_demographic_growth(df):
    """
    Calculate growth rates for demographic groups.
    
    Parameters:
    -----------
    df : DataFrame
        Time series data with demographic breakdown
        
    Returns:
    --------
    DataFrame
        Growth metrics by demographic group
    """
    try:
        # Get first and last periods for each group
        periods = df["bucket"].unique()
        if len(periods) < 2:
            return pd.DataFrame()  # Not enough data
            
        first_period = periods.min()
        last_period = periods.max()
        
        # Get values from first and last periods
        first_df = df[df["bucket"] == first_period][["group", "count"]]
        last_df = df[df["bucket"] == last_period][["group", "count"]]
        
        # Join the data
        growth_df = pd.merge(
            first_df, 
            last_df, 
            on="group", 
            suffixes=("_first", "_last")
        )
        
        # Calculate growth metrics
        growth_df["first_count"] = growth_df["count_first"]
        growth_df["last_count"] = growth_df["count_last"]
        growth_df["growth"] = growth_df["last_count"] - growth_df["first_count"]
        
        # Calculate percentage growth carefully to avoid division by zero
        def safe_pct_change(row):
            if row["first_count"] == 0:
                if row["last_count"] == 0:
                    return 0.0  # No change if both zero
                else:
                    return 100.0  # 100% growth if from zero to non-zero
            else:
                return (row["growth"] / row["first_count"]) * 100
                
        growth_df["growth_pct"] = growth_df.apply(safe_pct_change, axis=1)
        
        # Drop intermediary columns
        growth_df = growth_df[["group", "first_count", "last_count", "growth", "growth_pct"]]
        
        return growth_df
        
    except Exception as e:
        import traceback
        print(f"Error in growth calculation: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()  # Return empty dataframe on error
