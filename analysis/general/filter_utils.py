"""
Filter utilities for HMIS dashboard.
"""

import hashlib
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from pandas import DataFrame, Timestamp

# ────────────────────────── State Management ────────────────────────── #

def hash_data(data: Any) -> str:
    """Return a short MD5 hash of data for widget keys / cache keys."""
    return hashlib.md5(str(data).encode()).hexdigest()[:8]

def init_section_state(key: str, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialize or retrieve a per-section state dictionary.
    
    Parameters:
    -----------
    key : str
        Unique key for the section state
    defaults : dict, optional
        Default values to initialize state with
        
    Returns:
    --------
    dict
        Section state dictionary
    """
    state_key = f"state_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = defaults or {}
    return st.session_state[state_key]

def get_filter_timestamp() -> str:
    """Get the current filter change timestamp from session state."""
    return st.session_state.get("last_filter_change", "")

def is_cache_valid(state: Dict[str, Any], timestamp: Optional[str] = None) -> bool:
    """
    Check if cached values in a section state are still valid.
    
    Parameters:
    -----------
    state : dict
        Section state dictionary
    timestamp : str, optional
        Filter timestamp to check against (default: use session state)
        
    Returns:
    --------
    bool
        True if cache is valid, False otherwise
    """
    filter_ts = timestamp or get_filter_timestamp()
    return state.get("last_updated") == filter_ts

def invalidate_cache(state: Dict[str, Any], timestamp: Optional[str] = None) -> None:
    """
    Mark a section's cache as invalid and update timestamp.
    
    Parameters:
    -----------
    state : dict
        Section state dictionary
    timestamp : str, optional
        New timestamp value (default: use session state)
    """
    filter_ts = timestamp or get_filter_timestamp()
    state["last_updated"] = filter_ts

# ────────────────────────── Filter Rendering ────────────────────────── #

def render_filter_form(df: DataFrame) -> bool:
    """
    Render a sidebar filter form with date ranges and other filters.
    
    Parameters:
    -----------
    df : DataFrame
        The full dataset
        
    Returns:
    --------
    bool
        True if filters were applied, False otherwise
    """
    # Define constants
    FILTER_FORM_KEY = "filter_form"
    DEFAULT_CURRENT_START = date(2023, 10, 1)
    DEFAULT_CURRENT_END = date(2024, 9, 30)
    
    # Retrieve or initialize per-section state
    state: Dict[str, Any] = init_section_state(FILTER_FORM_KEY)

    # Defaults for the current reporting window
    win_start_default: date = state.get("win_start", DEFAULT_CURRENT_START)
    win_end_default: date = state.get("win_end", DEFAULT_CURRENT_END)
    
    # Header outside form
    st.sidebar.header("🗓️ Reporting Window & Filters")
    st.sidebar.markdown("Set the analysis window and apply optional filters.")
    
    # Custom previous checkbox OUTSIDE the form for immediate interactivity
    custom_prev: bool = st.sidebar.checkbox(
        "✏️ Custom previous window",
        value=state.get("custom_prev", False),
        key="custom_prev_checkbox",
        help="Enable custom comparison window",
    )
    
    # Store checkbox state immediately
    state["custom_prev"] = custom_prev
    
    # Render the form
    with st.sidebar.form("filters_form", clear_on_submit=False):
        # Main window inputs
        win_start, win_end = st.date_input(
            "Current Reporting Period",
            value=(win_start_default, win_end_default),
            help="Primary analysis reporting period",
        )
        st.caption("📌 **Note:** The selected end date will be included in the analysis period.")

        if win_start >= win_end:
            st.error("Start date must be before end date.")
        
        # Calculate the auto-generated previous window based on current selection
        current_period_days = (pd.Timestamp(win_end) - pd.Timestamp(win_start)).days + 1
        auto_prev_start = win_start - pd.Timedelta(days=current_period_days)
        auto_prev_end = win_start - pd.Timedelta(days=1)
        
        # Show auto-generated period info
        if not custom_prev:
            st.markdown(f"**Previous period (auto-generated):**  \n"
                       f"{auto_prev_start.strftime('%b %d, %Y')} to {auto_prev_end.strftime('%b %d, %Y')}  \n"
                       f"({current_period_days} days, matching current period length)")
        
        # Custom previous window inputs (only shown when checkbox is checked)
        if custom_prev:
            st.markdown("**Custom previous period:**")
            
            # Determine default values for custom period
            if "custom_prev_start" in state and "custom_prev_end" in state:
                # Use previously saved custom dates
                prev_start_default = state["custom_prev_start"]
                prev_end_default = state["custom_prev_end"]
            else:
                # First time enabling custom, use auto-generated as starting point
                prev_start_default = auto_prev_start
                prev_end_default = auto_prev_end
                
            prev_start, prev_end = st.date_input(
                "Previous window (Start / End)",
                value=(prev_start_default, prev_end_default),
                help="Comparison period for changes",
            )
            
            if prev_start >= prev_end:
                st.error("Previous start date must be before end date.")
                
            # Show comparison info
            custom_days = (pd.Timestamp(prev_end) - pd.Timestamp(prev_start)).days + 1
            if custom_days != current_period_days:
                st.caption(f"⚠️ Custom period is {custom_days} days vs {current_period_days} days for current period")
        else:
            # Use auto-generated values
            prev_start, prev_end = auto_prev_start, auto_prev_end

        st.divider()

        # Demographic / program filters
        filter_map: Dict[str, str] = {
            "Program CoC": "ProgramSetupCoC",
            "Local CoC": "LocalCoCCode",
            "Agency Name": "AgencyName",
            "Program Name": "ProgramName",
            "Project Type": "ProjectTypeCode",
            "Race / Ethnicity": "RaceEthnicity",
            "Gender": "Gender",
            "Household Type": "HouseholdType",
            "Veteran Status": "VeteranStatus",
            "Entry Age Tier": "AgeTieratEntry",
            "Currently Fleeing DV": "CurrentlyFleeingDV",
            "Head of Household": "IsHeadOfHousehold",
            "Chronic Homelessness Household": "CHStartHousehold",
        }
        selections: Dict[str, List[str]] = {}
        for label, col in filter_map.items():
            if col not in df.columns:
                continue
            options: List[str] = sorted(df[col].dropna().astype(str).unique())
            default_sel: List[str] = state.get(f"filter_{col}", ["ALL"])
            pick: List[str] = st.multiselect(
                label,
                ["ALL"] + options,
                default=default_sel,
                help=f"Filter by {label}. Select 'ALL' for no filter.",
            )
            selections[col] = [] if "ALL" in pick else pick

        # The submit button
        submitted: bool = st.form_submit_button(
            "Apply Filters",
            type="primary",
            help="Save selections and refresh analyses",
            use_container_width=True
        )

    # If not submitted, do nothing
    if not submitted:
        return False

    # On submit, commit everything
    try:
        # Persist form values into our section state
        state["win_start"] = win_start
        state["win_end"] = win_end
        
        # Save the actual previous period dates being used
        state["prev_start"] = prev_start
        state["prev_end"] = prev_end
        
        # If custom previous is enabled, also save these as the custom dates
        if custom_prev:
            state["custom_prev_start"] = prev_start
            state["custom_prev_end"] = prev_end
        else:
            # Clear custom dates when not using custom
            state.pop("custom_prev_start", None)
            state.pop("custom_prev_end", None)

        # Persist filter selections
        for col in filter_map.values():
            if col in selections:
                state[f"filter_{col}"] = selections[col] if selections[col] else ["ALL"]

        # Persist to session_state
        st.session_state["filters"] = selections
        st.session_state["t0"] = pd.Timestamp(win_start)
        st.session_state["t1"] = pd.Timestamp(win_end)
        st.session_state["prev_start"] = pd.Timestamp(prev_start)
        st.session_state["prev_end"] = pd.Timestamp(prev_end)
        st.session_state["last_filter_change"] = datetime.now().isoformat()
        
        # Success message
        st.sidebar.success("✅ Filters applied!", icon="✅")
        
    except Exception as err:
        st.error(f"Failed to apply filters: {err}")

    return True

def apply_filters(df: DataFrame) -> DataFrame:
    """
    Apply current filters from session state to a dataframe.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe to filter
        
    Returns:
    --------
    DataFrame
        The filtered dataframe
    """
    selections = st.session_state.get("filters", {})
    if not selections:
        return df

    df_filt = df.copy()
    for col, vals in selections.items():
        if vals and col in df_filt.columns:
            df_filt = df_filt[df_filt[col].astype(str).isin(vals)]

    return df_filt

def show_date_range_warning(df: DataFrame) -> None:
    """
    Display warnings if selected date ranges are outside data range.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe to check
    """
    try:
        data_min = df["ProjectStart"].min()
        data_max = df["ProjectExit"].max()

        for boundary, label in [
            (st.session_state.get("t0"), "Current window start"),
            (st.session_state.get("t1"), "Current window end"),
            (st.session_state.get("prev_start"), "Previous window start"),
            (st.session_state.get("prev_end"), "Previous window end"),
        ]:
            if boundary and (boundary < data_min or boundary > data_max):
                st.warning(
                    f"⚠️ {label} ({boundary.date()}) is outside the data range "
                    f"({data_min.date()} – {data_max.date()})."
                )
    except (KeyError, AttributeError):
        st.error("Data is missing required date columns.")
