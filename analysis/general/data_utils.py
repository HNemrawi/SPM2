"""
Data processing utilities for HMIS dashboard.

This module provides core data processing functions for the HMIS dashboard including:
- Client and household metrics calculation
- Demographic analysis utilities
- Time series data processing
- Return to homelessness tracking
- Data validation and caching
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
import streamlit as st
from pandas import DataFrame, Series, Timestamp
from core.ph_destinations import apply_custom_ph_destinations

# ==================== CONSTANTS & CONFIGURATIONS ====================

# Required columns for basic operations
REQUIRED_BASE_COLS: List[str] = ["ClientID", "ProjectStart", "ProjectExit"]

# Project type classifications
class ProjectTypes:
    """Project type classifications for HMIS data."""
    
    PERMANENT_HOUSING: Set[str] = {
        "PH – Housing Only",
        "PH – Housing with Services (no disability required for entry)",
        "PH – Permanent Supportive Housing (disability required for entry)",
        "PH – Rapid Re-Housing",
    }
    
    NON_HOMELESS_SERVICES: Set[str] = {
        "Coordinated Entry",
        "Day Shelter",
        "Homelessness Prevention",
        "Other",
        "Services Only",
    }
    
    @classmethod
    def is_permanent_housing(cls, project_type: str) -> bool:
        """Check if a project type is permanent housing."""
        return project_type in cls.PERMANENT_HOUSING
    
    @classmethod
    def is_homeless_project(cls, project_type: str) -> bool:
        """Check if a project type serves homeless individuals."""
        return project_type not in cls.NON_HOMELESS_SERVICES

# Backward compatibility
PH_PROJECTS = ProjectTypes.PERMANENT_HOUSING
NON_HOMELESS_PROJECTS = ProjectTypes.NON_HOMELESS_SERVICES

# Demographic dimensions for analysis
DEMOGRAPHIC_DIMENSIONS: List[Tuple[str, str]] = [
    # Basic Demographics
    ("Race / Ethnicity", "RaceEthnicity"),
    ("Gender", "Gender"),
    ("Entry Age Tier", "AgeTieratEntry"),
    
    # Client Characteristics
    ("Veteran Status", "VeteranStatus"),
    ("Has Income", "HasIncome"),
    ("Has Disability", "HasDisability"),
    ("Currently Fleeing DV", "CurrentlyFleeingDV"),
    
    # Household Information
    ("Household Type", "HouseholdType"),
    ("Head of Household", "IsHeadOfHousehold"),
    
    # Housing Status
    ("Prior Living Situation", "PriorLivingCat"),
    ("Chronic Homelessness", "CHStartHousehold"),
    
    # Program Information
    ("Program CoC", "ProgramSetupCoC"),
    ("Local CoC", "LocalCoCCode"),
    ("Agency Name", "AgencyName"),
    ("Program Name", "ProgramName"),
    ("Project Type", "ProjectTypeCode"),
    ("SSVF RRH", "SSVF_RRH"),
    ("Continuum Project", "ProgramsContinuumProject"),
    
    # Exit Information
    ("Exit Destination Category", "ExitDestinationCat"),
    ("Exit Destination", "ExitDestination"),
]

# Time frequency mappings
FREQUENCY_MAP: Dict[str, str] = {
    "Days": "D",
    "Weeks": "W",
    "Months": "M",
    "Quarters": "Q",
    "Years": "Y",
}

# ==================== UTILITY FUNCTIONS ====================

def validate_columns(df: DataFrame, required_cols: List[str]) -> None:
    """
    Validate that required columns exist in DataFrame.
    
    Args:
        df: DataFrame to check
        required_cols: List of required column names
        
    Raises:
        KeyError: If any required columns are missing
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")

def safe_divide(
    numerator: Union[int, float],
    denominator: Union[int, float],
    default: float = 0.0,
    multiplier: float = 1.0,
    precision: int = 1
) -> float:
    """
    Safely divide two numbers with optional scaling and rounding.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if denominator is zero
        multiplier: Value to multiply result by (e.g., 100 for percentages)
        precision: Number of decimal places to round to
        
    Returns:
        Result of division, scaled and rounded
    """
    if denominator == 0:
        return default
    return round((numerator / denominator) * multiplier, precision)

def calculate_change(
    current: Union[int, float],
    previous: Union[int, float]
) -> Tuple[float, float]:
    """
    Calculate absolute and percentage change between two values.
    
    Args:
        current: Current value
        previous: Previous value
        
    Returns:
        Tuple of (absolute_change, percentage_change)
    """
    try:
        absolute_change = current - previous
        pct_change = safe_divide(absolute_change, abs(previous), multiplier=100)
        return absolute_change, pct_change
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0, 0.0

# Maintain backward compatibility
_need = validate_columns
_safe_div = safe_divide
calc_delta = calculate_change

# ==================== DATA LOADING & CACHING ====================

@st.cache_data(show_spinner="Loading and preprocessing data…")
def cached_load(upload_or_df: Union[Any, pd.DataFrame]) -> pd.DataFrame:
    """
    Load and cache data with preprocessing.
    
    Args:
        upload_or_df: Either an uploaded file or existing DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    if isinstance(upload_or_df, pd.DataFrame):
        df = upload_or_df.copy()
    else:
        df = load_and_preprocess_data(upload_or_df)
    
    # Ensure datetime columns are properly typed
    datetime_cols = [col for col in df.columns 
                     if pd.api.types.is_datetime64_any_dtype(df[col])]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col])
    
    return df

# ==================== CORE METRICS CALCULATIONS ====================

class ClientMetrics:
    """Core client and household metrics calculations."""
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def served_clients(
        df: DataFrame,
        start_date: Timestamp,
        end_date: Timestamp
    ) -> Set[int]:
        """
        Get unique clients active within date range.
        
        Args:
            df: DataFrame of client enrollments
            start_date: Start of period
            end_date: End of period
            
        Returns:
            Set of unique ClientIDs
        """
        validate_columns(df, REQUIRED_BASE_COLS)
        
        # Active if: exit after start (or no exit) AND entry before end
        active_mask = (
            ((df["ProjectExit"] >= start_date) | df["ProjectExit"].isna()) &
            (df["ProjectStart"] <= end_date)
        )
        
        return set(df.loc[active_mask, "ClientID"].unique())
    
    @staticmethod
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
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def inflow(
        df: DataFrame,
        start_date: Timestamp,
        end_date: Timestamp
    ) -> Set[int]:
        """
        Get clients entering during period who weren't in any programs
        the day before the report start.
        
        Args:
            df: DataFrame of client enrollments
            start_date: Start of period
            end_date: End of period
            
        Returns:
            Set of ClientIDs who are new entries
        """
        validate_columns(df, REQUIRED_BASE_COLS)
        
        # Get entries during the period
        entries_mask = df["ProjectStart"].between(start_date, end_date)
        entry_ids = set(df.loc[entries_mask, "ClientID"])
        
        # Check who was active the day before
        day_before = start_date - pd.Timedelta(days=1)
        active_before_mask = (
            ((df["ProjectExit"] >= day_before) | df["ProjectExit"].isna()) &
            (df["ProjectStart"] <= day_before)
        )
        active_before_ids = set(df.loc[active_before_mask, "ClientID"])
        
        # Return new entries (not active before)
        return entry_ids - active_before_ids
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def outflow(
        df: DataFrame,
        start_date: Timestamp,
        end_date: Timestamp
    ) -> Set[int]:
        """
        Get clients exiting during period who aren't in any programs
        on the last day of the period.
        
        Args:
            df: DataFrame of client enrollments
            start_date: Start of period
            end_date: End of period
            
        Returns:
            Set of ClientIDs who exited completely
        """
        validate_columns(df, REQUIRED_BASE_COLS)
        
        # Get exits during the period
        exits_mask = df["ProjectExit"].between(start_date, end_date)
        exit_ids = set(df.loc[exits_mask, "ClientID"])
        
        # Check who is still active on the last day
        active_on_end_mask = (
            ((df["ProjectExit"] > end_date) | df["ProjectExit"].isna()) &
            (df["ProjectStart"] <= end_date)
        )
        still_active_ids = set(df.loc[active_on_end_mask, "ClientID"])
        
        # Return exits who aren't still active
        return exit_ids - still_active_ids

# Maintain backward compatibility
served_clients = ClientMetrics.served_clients
households_served = ClientMetrics.households_served
inflow = ClientMetrics.inflow
outflow = ClientMetrics.outflow

# ==================== PERMANENT HOUSING METRICS ====================

class PHMetrics:
    """Permanent Housing exit metrics and analysis."""
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def exit_clients(
        df: DataFrame,
        start_date: Timestamp,
        end_date: Timestamp
    ) -> Set[int]:
        """
        Get clients exiting to permanent housing during date range.
        
        Args:
            df: DataFrame of client enrollments
            start_date: Start of period
            end_date: End of period
            
        Returns:
            Set of ClientIDs who exited to PH
        """
        validate_columns(df, ["ClientID", "ProjectExit", "ExitDestinationCat"])
        
        # PH exits during the period
        ph_exits_mask = (
            (df["ExitDestinationCat"] == "Permanent Housing Situations") &
            df["ProjectExit"].between(start_date, end_date)
        )
        
        # Get unique client IDs
        client_ids = df.loc[ph_exits_mask, "ClientID"].dropna().unique()
        return set(client_ids)
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def exit_rate(
        df: DataFrame,
        start_date: Timestamp,
        end_date: Timestamp
    ) -> float:
        """
        Calculate PH exits ÷ total exits as percentage.
        
        Args:
            df: DataFrame of client enrollments
            start_date: Start of period
            end_date: End of period
            
        Returns:
            Percentage of exits that were to permanent housing
        """
        validate_columns(df, ["ClientID", "ProjectExit", "ExitDestinationCat"])
        
        # All exits during the period
        all_exits_mask = df["ProjectExit"].between(start_date, end_date)
        total_exits = df.loc[all_exits_mask, "ClientID"].nunique()
        
        # PH exits during the period
        ph_exits_mask = (
            (df["ExitDestinationCat"] == "Permanent Housing Situations") &
            df["ProjectExit"].between(start_date, end_date)
        )
        ph_exits = df.loc[ph_exits_mask, "ClientID"].nunique()
        
        return safe_divide(ph_exits, total_exits, multiplier=100)

# Backward compatibility
ph_exit_clients = PHMetrics.exit_clients
ph_exit_rate = PHMetrics.exit_rate

# ==================== PERIOD COMPARISON ====================

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
    
    Args:
        df: DataFrame of client enrollments
        current_start: Start of current period
        current_end: End of current period
        previous_start: Start of previous period
        previous_end: End of previous period
        
    Returns:
        Dictionary with client sets for different categories
    """
    # Get clients active in each period
    current_clients = served_clients(df, current_start, current_end)
    previous_clients = served_clients(df, previous_start, previous_end)
    
    # Calculate different groups
    return {
        'current_clients': current_clients,
        'previous_clients': previous_clients,
        'carryover': current_clients.intersection(previous_clients),
        'new': current_clients - previous_clients,
        'exited': previous_clients - current_clients
    }

# ==================== RETURN TO HOMELESSNESS TRACKING ====================

class ReturnToHomelessness:
    """Track and analyze returns to homelessness after PH exits."""
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def identify_returners(
        df_filtered: pd.DataFrame,
        full_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        return_window_days: int = 730
    ) -> Set[int]:
        """
        Identify clients who exited to PH and returned to homelessness
        within the specified window using HUD-compliant logic.
        
        Args:
            df_filtered: Filtered dataset for PH exits in reporting period
            full_df: Full dataset to scan for re-enrollments
            start_date: Start of reporting period
            end_date: End of reporting period
            return_window_days: Days after exit to check for returns
            
        Returns:
            Set of ClientIDs who returned to homelessness
        """
        # Apply custom PH destinations to both dataframes
        df_filtered = apply_custom_ph_destinations(df_filtered, force=True)
        full_df = apply_custom_ph_destinations(full_df, force=True)
        
        required_exit_cols = REQUIRED_BASE_COLS + ["ExitDestinationCat"]
        required_full_cols = REQUIRED_BASE_COLS + ["ProjectTypeCode", 
                                                   "HouseholdMoveInDate", 
                                                   "EnrollmentID"]
        
        validate_columns(df_filtered, required_exit_cols)
        validate_columns(full_df, required_full_cols)
        
        # Get PH exits during the reporting period
        ph_exits = df_filtered[
            (df_filtered["ProjectExit"].between(start_date, end_date)) &
            (df_filtered["ExitDestinationCat"] == "Permanent Housing Situations")
        ].copy()
        
        if ph_exits.empty:
            return set()
        
        # Check each client for returns
        exited_clients = set(ph_exits["ClientID"])
        returners = set()
        
        for client_id in exited_clients:
            client_enrollments = full_df[full_df["ClientID"] == client_id].copy()
            client_exits = ph_exits[ph_exits["ClientID"] == client_id].sort_values("ProjectExit")
            
            # Check each qualifying exit
            for _, exit_row in client_exits.iterrows():
                if ReturnToHomelessness._check_return(
                    client_enrollments,
                    exit_row["ProjectExit"],
                    exit_row.get("EnrollmentID", -1),
                    return_window_days
                ):
                    returners.add(client_id)
                    break
        
        return returners
    
    @staticmethod
    def _check_return(
        client_df: pd.DataFrame,
        exit_date: pd.Timestamp,
        exit_enrollment_id: int,
        max_days: int
    ) -> bool:
        """
        Check if a client returns to homelessness after a PH exit.
        
        Uses HUD logic including exclusion windows for short PH stays.
        """
        # Get enrollments after exit within window
        window_end = exit_date + pd.Timedelta(days=max_days)
        next_enrollments = client_df[
            (client_df["ProjectStart"] > exit_date) &
            (client_df["ProjectStart"] <= window_end) &
            (~client_df["ProjectTypeCode"].isin(NON_HOMELESS_PROJECTS))
        ].sort_values(["ProjectStart", "EnrollmentID"])
        
        if next_enrollments.empty:
            return False
        
        # Track exclusion windows for short PH stays
        exclusion_windows = []
        
        # Check each potential return
        for _, enrollment in next_enrollments.iterrows():
            if enrollment.get("EnrollmentID", -1) == exit_enrollment_id:
                continue
            
            is_ph = enrollment["ProjectTypeCode"] in PH_PROJECTS
            gap_days = (enrollment["ProjectStart"] - exit_date).days
            
            if is_ph:
                # Skip if move-in equals project start (immediate housing)
                if (pd.notna(enrollment.get("HouseholdMoveInDate")) and 
                    enrollment["ProjectStart"] == enrollment["HouseholdMoveInDate"]):
                    continue
                
                # Handle short stays (≤14 days)
                if gap_days <= 14:
                    if pd.notnull(enrollment.get("ProjectExit")):
                        exclusion_windows = ReturnToHomelessness._merge_windows(
                            (enrollment["ProjectStart"] + pd.Timedelta(days=1),
                             enrollment["ProjectExit"] + pd.Timedelta(days=14)),
                            exclusion_windows
                        )
                    continue
                
                # Check if within exclusion window
                if any(start <= enrollment["ProjectStart"] <= end 
                       for start, end in exclusion_windows):
                    if pd.notnull(enrollment.get("ProjectExit")):
                        exclusion_windows = ReturnToHomelessness._merge_windows(
                            (enrollment["ProjectStart"] + pd.Timedelta(days=1),
                             enrollment["ProjectExit"] + pd.Timedelta(days=14)),
                            exclusion_windows
                        )
                    continue
                
                # Valid PH return to homelessness
                return True
            else:
                # Non-PH enrollment is immediate return
                return True
        
        return False
    
    @staticmethod
    def _merge_windows(
        new_window: Tuple[pd.Timestamp, pd.Timestamp],
        existing_windows: List[Tuple[pd.Timestamp, pd.Timestamp]]
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Merge overlapping exclusion windows."""
        new_start, new_end = new_window
        merged = []
        
        for window_start, window_end in existing_windows:
            if new_start <= window_end and new_end >= window_start:
                # Overlapping - extend the window
                new_start = min(new_start, window_start)
                new_end = max(new_end, window_end)
            else:
                # Non-overlapping - keep as is
                merged.append((window_start, window_end))
        
        merged.append((new_start, new_end))
        return sorted(merged, key=lambda w: w[0])

# Backward compatibility
return_after_exit = ReturnToHomelessness.identify_returners
_check_return_to_homelessness = ReturnToHomelessness._check_return

# ==================== DEMOGRAPHIC ANALYSIS ====================

def category_counts(
    df: DataFrame,
    client_ids: Set[int],
    group_column: str,
    series_name: str = "Count"
) -> Series:
    """
    Count unique clients by category for a given column.
    
    Args:
        df: DataFrame of client enrollments
        client_ids: Set of ClientIDs to filter by
        group_column: Column name to group by
        series_name: Name for the resulting Series
        
    Returns:
        Series with count of clients in each category
    """
    if group_column not in df.columns:
        st.warning(f"Column '{group_column}' not found in data.")
        return pd.Series(dtype=int, name=series_name)
    
    # Filter to specified clients and count by group
    filtered_df = df[df["ClientID"].isin(client_ids)]
    
    return (
        filtered_df
        .groupby(group_column, observed=True)["ClientID"]
        .nunique()
        .rename(series_name)
        .sort_values(ascending=False)
    )

# ==================== TIME SERIES ANALYSIS ====================

class TimeSeriesAnalysis:
    """Time series analysis utilities for metrics."""
    
    @staticmethod
    def calculate_metric_series(
        df: DataFrame,
        metric_function: Callable,
        start_date: Timestamp,
        end_date: Timestamp,
        frequency: str = "M"
    ) -> DataFrame:
        """
        Calculate a metric for each time period.
        
        Args:
            df: DataFrame of client enrollments
            metric_function: Function that takes (df, start, end) -> Set[int]
            start_date: Start of analysis period
            end_date: End of analysis period
            frequency: Pandas frequency string (D, W, M, Q, Y)
            
        Returns:
            DataFrame with columns: bucket, count
        """
        periods = pd.period_range(start=start_date, end=end_date, freq=frequency)
        
        results = []
        for period in periods:
            period_start = period.start_time
            period_end = period.end_time
            
            client_ids = metric_function(df, period_start, period_end)
            results.append({
                "bucket": period_start,
                "count": len(client_ids)
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def calculate_metric_series_by_group(
        df: DataFrame,
        metric_function: Callable,
        group_column: str,
        start_date: Timestamp,
        end_date: Timestamp,
        frequency: str = "M"
    ) -> DataFrame:
        """
        Calculate a metric for each time period, broken down by group.
        
        Args:
            df: DataFrame of client enrollments
            metric_function: Function that takes (df, start, end) -> Set[int]
            group_column: Column to group by
            start_date: Start of analysis period
            end_date: End of analysis period
            frequency: Pandas frequency string
            
        Returns:
            DataFrame with columns: bucket, group, count
        """
        if group_column not in df.columns:
            raise KeyError(f"Column '{group_column}' not found in data.")
        
        periods = pd.period_range(start=start_date, end=end_date, freq=frequency)
        results = []
        
        # Calculate for each group
        for group_value, group_df in df.groupby(group_column, dropna=True):
            for period in periods:
                period_start = period.start_time
                period_end = period.end_time
                
                client_ids = metric_function(group_df, period_start, period_end)
                results.append({
                    "bucket": period_start,
                    "group": group_value,
                    "count": len(client_ids)
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def calculate_growth_rates(df: DataFrame) -> DataFrame:
        """
        Calculate growth rates for demographic groups over time.
        
        Args:
            df: Time series DataFrame with demographic breakdown
            
        Returns:
            DataFrame with growth metrics by group
        """
        try:
            periods = df["bucket"].unique()
            if len(periods) < 2:
                return pd.DataFrame()
            
            # Get first and last period data
            first_period = periods.min()
            last_period = periods.max()
            
            first_data = df[df["bucket"] == first_period][["group", "count"]]
            last_data = df[df["bucket"] == last_period][["group", "count"]]
            
            # Merge and calculate growth
            growth_df = pd.merge(
                first_data,
                last_data,
                on="group",
                suffixes=("_first", "_last")
            )
            
            # Calculate growth metrics
            growth_df["first_count"] = growth_df["count_first"]
            growth_df["last_count"] = growth_df["count_last"]
            growth_df["growth"] = growth_df["last_count"] - growth_df["first_count"]
            
            # Safe percentage calculation
            growth_df["growth_pct"] = growth_df.apply(
                lambda row: safe_divide(
                    row["growth"],
                    row["first_count"],
                    default=100.0 if row["last_count"] > 0 else 0.0,
                    multiplier=100
                ),
                axis=1
            )
            
            # Return cleaned dataframe
            return growth_df[["group", "first_count", "last_count", 
                            "growth", "growth_pct"]].sort_values(
                                "growth_pct", ascending=False
                            )
            
        except Exception as e:
            st.error(f"Error calculating growth rates: {str(e)}")
            return pd.DataFrame()

# Backward compatibility
recalculated_metric_time_series = TimeSeriesAnalysis.calculate_metric_series
recalculated_metric_time_series_by_group = TimeSeriesAnalysis.calculate_metric_series_by_group
calculate_demographic_growth = TimeSeriesAnalysis.calculate_growth_rates

# ==================== EXPORT ALL PUBLIC ITEMS ====================

__all__ = [
    # Constants
    'REQUIRED_BASE_COLS',
    'ProjectTypes',
    'PH_PROJECTS',
    'NON_HOMELESS_PROJECTS',
    'DEMOGRAPHIC_DIMENSIONS',
    'FREQUENCY_MAP',
    
    # Utility functions
    'validate_columns',
    'safe_divide',
    'calculate_change',
    '_need',  # Backward compatibility
    '_safe_div',  # Backward compatibility
    'calc_delta',  # Backward compatibility
    
    # Data loading
    'cached_load',
    
    # Client metrics
    'ClientMetrics',
    'served_clients',
    'households_served',
    'inflow',
    'outflow',
    
    # PH metrics
    'PHMetrics',
    'ph_exit_clients',
    'ph_exit_rate',
    
    # Period comparison
    'period_comparison',
    
    # Return to homelessness
    'ReturnToHomelessness',
    'return_after_exit',
    '_check_return_to_homelessness',  # Backward compatibility
    
    # Demographics
    'category_counts',
    
    # Time series
    'TimeSeriesAnalysis',
    'recalculated_metric_time_series',
    'recalculated_metric_time_series_by_group',
    'calculate_demographic_growth',
]