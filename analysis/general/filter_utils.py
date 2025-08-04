"""
Filter utilities for HMIS dashboard.
Updated with theme support for perfect dark/light mode compatibility.
"""

import hashlib
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
from pandas import DataFrame, Timestamp

# Import theme components from the app module
from analysis.general.theme import Colors
from ui.styling import create_info_box
from core.ph_destinations import apply_custom_ph_destinations

# ==================== CONSTANTS ====================

# Default date ranges
DEFAULT_CURRENT_START = date(2023, 10, 1)
DEFAULT_CURRENT_END = date(2024, 9, 30)

# Filter form configuration
FILTER_FORM_KEY = "filter_form"
CUSTOM_PREV_CHECKBOX_KEY = "custom_prev_checkbox"

# Filter categories with display names
FILTER_CATEGORIES = {
    "Program Filters": {
        "Program CoC": "ProgramSetupCoC",
        "Local CoC": "LocalCoCCode",
        "Project Type": "ProjectTypeCode",
        "Agency Name": "AgencyName",
        "Program Name": "ProgramName",
        "SSVF RRH": "SSVF_RRH",
        "Continuum Project": "ProgramsContinuumProject",
    },
    "Client Demographics": {
        "Head of Household": "IsHeadOfHousehold",
        "Household Type": "HouseholdType",
        "Race / Ethnicity": "RaceEthnicity",
        "Gender": "Gender",
        "Entry Age Tier": "AgeTieratEntry",
        "Has Income": "HasIncome",
        "Has Disability": "HasDisability",
    },
    "Housing Status": {
        "Prior Living Situation": "PriorLivingCat",
        "Chronic Homelessness Household": "CHStartHousehold",
        "Exit Destination Category": "ExitDestinationCat",
        "Exit Destination": "ExitDestination",
    },
    "Special Populations": {
        "Veteran Status": "VeteranStatus",
        "Currently Fleeing DV": "CurrentlyFleeingDV",
    }
}

# ==================== STATE MANAGEMENT ====================

def hash_data(data: Any) -> str:
    """
    Return a short MD5 hash of data for widget keys / cache keys.
    
    Parameters:
        data: Any data to hash
        
    Returns:
        8-character hash string
    """
    return hashlib.md5(str(data).encode()).hexdigest()[:8]

def init_section_state(key: str, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialize or retrieve a per-section state dictionary.
    
    Parameters:
        key: Unique key for the section state
        defaults: Default values to initialize state with
        
    Returns:
        Section state dictionary
    """
    state_key = f"state_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = defaults or {}
    
    # Get the section state
    section_state = st.session_state[state_key]
    
    # Special handling for filter form state - restore PH destinations
    if key == "filter_form" and "selected_ph_destinations" in section_state:
        if "selected_ph_destinations" not in st.session_state:
            st.session_state["selected_ph_destinations"] = section_state["selected_ph_destinations"]
    
    return section_state

def get_filter_timestamp() -> str:
    """Get the current filter change timestamp from session state."""
    return st.session_state.get("last_filter_change", "")

def is_cache_valid(state: Dict[str, Any], timestamp: Optional[str] = None) -> bool:
    """
    Check if cached values in a section state are still valid.
    
    Parameters:
        state: Section state dictionary
        timestamp: Filter timestamp to check against (default: use session state)
        
    Returns:
        True if cache is valid, False otherwise
    """
    filter_ts = timestamp or get_filter_timestamp()
    return state.get("last_updated") == filter_ts

def invalidate_cache(state: Dict[str, Any], timestamp: Optional[str] = None) -> None:
    """
    Mark a section's cache as invalid and update timestamp.
    
    Parameters:
        state: Section state dictionary
        timestamp: New timestamp value (default: use session state)
    """
    filter_ts = timestamp or get_filter_timestamp()
    state["last_updated"] = filter_ts

# ==================== UI STYLING HELPERS ====================

def _apply_filter_form_styling() -> None:
    """Apply custom styling for filter form components with theme support."""
    st.markdown(f"""
    <style>
    /* Filter form container styling - adapts to theme */
    div[data-testid="stForm"] {{
        background: var(--bg-card, rgba(128, 128, 128, 0.05));
        border: 1px solid var(--border-color, {Colors.BORDER_COLOR});
        border-radius: var(--border-radius, 8px);
        padding: var(--spacing-md, 1rem);
        margin-bottom: var(--spacing-md, 1rem);
        transition: all 0.2s ease;
    }}
    
    /* Date input styling with theme support */
    div[data-testid="stDateInput"] > div {{
        background: var(--background-color, white);
        border-radius: var(--border-radius-sm, 4px);
        color: var(--text-color);
    }}
    
    /* Multiselect styling with theme adaptation */
    div[data-baseweb="select"] {{
        background: var(--background-color, white);
        border-color: var(--border-color, {Colors.BORDER_COLOR});
    }}
    
    /* Override for dark mode */
    @media (prefers-color-scheme: dark) {{
        div[data-baseweb="select"] {{
            background: var(--secondary-background-color, #1a1a1a);
        }}
        
        div[data-testid="stDateInput"] > div {{
            background: var(--secondary-background-color, #1a1a1a);
        }}
    }}
    
    /* Submit button styling with theme colors */
    div[data-testid="stForm"] button[type="submit"] {{
        background: {Colors.PRIMARY};
        color: white;
        border: none;
        padding: var(--spacing-sm, 0.5rem) var(--spacing-md, 1rem);
        border-radius: var(--border-radius-sm, 4px);
        font-weight: 500;
        transition: all 0.2s ease;
        cursor: pointer;
    }}
    
    div[data-testid="stForm"] button[type="submit"]:hover {{
        background: {Colors.PRIMARY_HOVER};
        transform: translateY(-1px);
        box-shadow: var(--shadow-md, 0 4px 6px rgba(0, 0, 0, 0.1));
    }}
    
    /* Checkbox styling with theme colors */
    div[data-testid="stCheckbox"] label {{
        font-weight: 500;
        color: var(--text-color, {Colors.NEUTRAL_700});
    }}
    
    /* Caption styling */
    .stCaption {{
        color: var(--text-secondary, {Colors.NEUTRAL_600});
        font-size: 0.875rem;
    }}
    
    /* Success message styling with theme support */
    .filter-success {{
        background: color-mix(in srgb, {Colors.SUCCESS} 10%, transparent);
        color: {Colors.SUCCESS};
        border: 1px solid {Colors.SUCCESS};
        border-radius: var(--border-radius-sm, 4px);
        padding: 0.75rem;
    }}
    
    /* Expander styling for better theme integration */
    .streamlit-expanderHeader {{
        background: var(--bg-hover, rgba(128, 128, 128, 0.05));
        border-radius: var(--border-radius-sm, 4px);
        transition: background 0.2s ease;
    }}
    
    .streamlit-expanderHeader:hover {{
        background: var(--bg-active, rgba(128, 128, 128, 0.1));
    }}
    
    /* Date range info box styling */
    .date-range-info {{
        background: var(--bg-card, rgba(128, 128, 128, 0.02));
        border-radius: 0 var(--border-radius-sm, 4px) var(--border-radius-sm, 4px) 0;
        padding: var(--spacing-sm, 0.5rem);
        margin: var(--spacing-sm, 0.5rem) 0;
        transition: all 0.2s ease;
    }}
    
    /* Header gradient that adapts to theme */
    .filter-header {{
        background: linear-gradient(
            135deg, 
            color-mix(in srgb, {Colors.PRIMARY} 10%, transparent) 0%, 
            transparent 100%
        );
        padding: var(--spacing-md, 1rem);
        border-radius: var(--border-radius, 8px);
        margin-bottom: var(--spacing-md, 1rem);
    }}
    
    .filter-header h2 {{
        margin: 0;
        color: var(--text-primary, {Colors.NEUTRAL_900});
    }}
    
    .filter-header p {{
        margin: 0.5rem 0 0 0;
        color: var(--text-secondary, {Colors.NEUTRAL_600});
    }}
    </style>
    """, unsafe_allow_html=True)

def _render_date_period_info(
    start: date, 
    end: date, 
    label: str,
    period_days: int,
    is_custom: bool = False
) -> None:
    """
    Render formatted date period information with theme support.
    
    Parameters:
        start: Start date
        end: End date
        label: Period label
        period_days: Number of days in period
        is_custom: Whether this is a custom period
    """
    color = Colors.PRIMARY if not is_custom else Colors.WARNING
    icon = "📅" if not is_custom else "✏️"
    
    st.markdown(f"""
    <div class="date-range-info" style="border-left: 3px solid {color};">
        <strong>{icon} {label}:</strong><br>
        {start.strftime('%b %d, %Y')} to {end.strftime('%b %d, %Y')}<br>
        <span style="color: var(--text-muted, {Colors.NEUTRAL_600}); font-size: 0.875rem;">
            ({period_days} days)
        </span>
    </div>
    """, unsafe_allow_html=True)

def render_ph_destination_selector(df: DataFrame) -> None:
    """Render a selector for customizing PH destinations."""
    if "ExitDestination" not in df.columns or "ExitDestinationCat" not in df.columns:
        return
    
    # Get PH destinations from original data
    ph_mask = df["ExitDestinationCat"] == "Permanent Housing Situations"
    original_ph_destinations = sorted(df.loc[ph_mask, "ExitDestination"].dropna().unique())
    
    if not original_ph_destinations:
        return
    
    all_destinations = sorted(df["ExitDestination"].dropna().unique())
    
    st.markdown("### 🏠 Permanent Housing Selection")
    
    # Initialize session state ONLY if it doesn't exist
    if "selected_ph_destinations" not in st.session_state:
        st.session_state["selected_ph_destinations"] = set(original_ph_destinations)
    
    # Get current selections for default, ensuring they exist in options
    current_selections = st.session_state.get("selected_ph_destinations", set())
    valid_selections = [dest for dest in current_selections if dest in all_destinations]
    
    # Destination selector WITHOUT callback (forms don't allow callbacks)
    selected = st.multiselect(
        "Select destinations to count as Permanent Housing:",
        options=all_destinations,
        default=valid_selections,
        key="ph_destination_selector",
        help="Choose which exit destinations should be considered permanent housing"
    )
    
    # Show count and status
    num_selected = len(selected)
    total_destinations = len(all_destinations)
    num_original = len(original_ph_destinations)
    
    if num_selected == num_original and set(selected) == set(original_ph_destinations):
        st.info(f"✅ Using default PH destinations ({num_selected} of {total_destinations} total)")
    else:
        st.info(f"🎯 Custom PH destinations active ({num_selected} of {total_destinations} total)")
    
# ==================== FILTER VALIDATION ====================

def _validate_date_range(
    start: date, 
    end: date, 
    label: str = "Date range"
) -> Tuple[bool, Optional[str]]:
    """
    Validate a date range.
    
    Parameters:
        start: Start date
        end: End date
        label: Label for error messages
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if start >= end:
        return False, f"{label}: Start date must be before end date."
    
    # Check for unreasonable date ranges
    days_diff = (end - start).days
    if days_diff > 3650:  # 10 years
        return False, f"{label}: Date range exceeds 10 years."
    
    return True, None

def _validate_filter_selections(
    selections: Dict[str, List[str]]
) -> Tuple[bool, List[str]]:
    """
    Validate filter selections.
    
    Parameters:
        selections: Dictionary of filter selections
        
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check for conflicting selections
    if "ProjectTypeCode" in selections:
        project_types = selections["ProjectTypeCode"]
        if len(project_types) == 0:
            warnings.append("No project types selected - this may result in empty data.")
    
    return True, warnings

# ==================== MAIN FILTER FORM ====================

def render_filter_form(df: DataFrame) -> bool:
    """
    Render a sidebar filter form with date ranges and other filters.
    Enhanced with theme support for perfect dark/light mode compatibility.
    
    Parameters:
        df: The full dataset
        
    Returns:
        True if filters were applied, False otherwise
    """
    # Apply custom styling
    _apply_filter_form_styling()
    
    # Retrieve or initialize per-section state
    state = init_section_state(FILTER_FORM_KEY)

    # Defaults for the current reporting window
    win_start_default = state.get("win_start", DEFAULT_CURRENT_START)
    win_end_default = state.get("win_end", DEFAULT_CURRENT_END)
    
    # Header with improved styling
    st.sidebar.markdown("""
    <div class="filter-header">
        <h2>🗓️ Reporting Window & Filters</h2>
        <p>Set the analysis window and apply optional filters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom previous checkbox with enhanced styling
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        custom_prev = st.checkbox(
            "✏️ Custom comparison",
            value=state.get("custom_prev", False),
            key=CUSTOM_PREV_CHECKBOX_KEY,
            help="Enable custom comparison window instead of auto-generated",
        )
    
    # Store checkbox state immediately
    state["custom_prev"] = custom_prev
    
    # Render the form with sections
    with st.sidebar.form("filters_form", clear_on_submit=False):
        
        # ===== PH Destinations Customization (Inside Form) =====
        if "ExitDestination" in df.columns:
            with st.expander("⚙️ Customize Permanent Housing Destinations", expanded=False):
                render_ph_destination_selector(df)
            st.markdown("---")
        # ===== Date Range Section =====
        st.markdown("### 📅 Time Period")
        
        # Main window inputs
        win_start, win_end = st.date_input(
            "Current Reporting Period",
            value=(win_start_default, win_end_default),
            help="Primary analysis reporting period (inclusive)",
        )
        
        # Validate current period
        is_valid, error_msg = _validate_date_range(win_start, win_end, "Current period")
        if not is_valid:
            st.error(error_msg)
        
        # Calculate auto-generated previous window
        current_period_days = (pd.Timestamp(win_end) - pd.Timestamp(win_start)).days + 1
        auto_prev_start = win_start - pd.Timedelta(days=current_period_days)
        auto_prev_end = win_start - pd.Timedelta(days=1)
        
        # Show period information
        if not custom_prev:
            _render_date_period_info(
                auto_prev_start, 
                auto_prev_end, 
                "Previous period (auto)",
                current_period_days
            )
        
        # Custom previous window inputs
        if custom_prev:
            st.markdown("#### Custom Previous Period")
            
            # Determine defaults
            if "custom_prev_start" in state and "custom_prev_end" in state:
                prev_start_default = state["custom_prev_start"]
                prev_end_default = state["custom_prev_end"]
            else:
                prev_start_default = auto_prev_start
                prev_end_default = auto_prev_end
                
            prev_start, prev_end = st.date_input(
                "Previous window dates",
                value=(prev_start_default, prev_end_default),
                help="Comparison period for change metrics",
            )
            
            # Validate previous period
            is_valid, error_msg = _validate_date_range(prev_start, prev_end, "Previous period")
            if not is_valid:
                st.error(error_msg)
            
            # Show comparison info
            custom_days = (pd.Timestamp(prev_end) - pd.Timestamp(prev_start)).days + 1
            if custom_days != current_period_days:
                st.warning(
                    f"⚠️ Period length mismatch: {custom_days} days vs "
                    f"{current_period_days} days for current period"
                )
        else:
            prev_start, prev_end = auto_prev_start, auto_prev_end

        st.markdown("---")

        # ===== Filters Section =====
        st.markdown("### 🔍 Data Filters")
        
        # Organize filters by category
        selections = {}
        
        for category_name, category_filters in FILTER_CATEGORIES.items():
            with st.expander(f"**{category_name}**", expanded=False):
                for label, col in category_filters.items():
                    if col not in df.columns:
                        continue
                    
                    options = sorted(df[col].dropna().astype(str).unique())
                    default_sel = state.get(f"filter_{col}", ["ALL"])
                    
                    # Create multiselect with custom styling
                    pick = st.multiselect(
                        label,
                        ["ALL"] + options,
                        default=default_sel,
                        help=f"Filter by {label}. Select 'ALL' for no filter.",
                    )
                    selections[col] = [] if "ALL" in pick else pick

        # Submit button with enhanced styling
        submitted = st.form_submit_button(
            "🚀 Apply Filters",
            type="primary",
            help="Save selections and refresh analyses",
            use_container_width=True
        )

    # Process form submission
    if not submitted:
        return False

    # Commit changes
    try:
        # Persist form values
        state["win_start"] = win_start
        state["win_end"] = win_end
        state["prev_start"] = prev_start
        state["prev_end"] = prev_end
        
        # Handle PH destination selections from the form
        if "ph_destination_selector" in st.session_state:
            selected_ph_destinations = set(st.session_state["ph_destination_selector"])
            st.session_state["selected_ph_destinations"] = selected_ph_destinations
            state["selected_ph_destinations"] = selected_ph_destinations
        
        # Save custom dates if applicable
        if custom_prev:
            state["custom_prev_start"] = prev_start
            state["custom_prev_end"] = prev_end
        else:
            state.pop("custom_prev_start", None)
            state.pop("custom_prev_end", None)

        # Persist filter selections
        for col, values in selections.items():
            state[f"filter_{col}"] = values if values else ["ALL"]

        # Update session state
        st.session_state["filters"] = selections
        st.session_state["t0"] = pd.Timestamp(win_start)
        st.session_state["t1"] = pd.Timestamp(win_end)
        st.session_state["prev_start"] = pd.Timestamp(prev_start)
        st.session_state["prev_end"] = pd.Timestamp(prev_end)
        st.session_state["last_filter_change"] = datetime.now().isoformat()
        
        # Success feedback with animation
        st.sidebar.markdown("""
        <div class="filter-success" style="
            text-align: center;
            animation: slideIn 0.3s ease-out;
        ">
            ✅ <strong>Filters applied successfully!</strong>
        </div>
        <style>
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """, unsafe_allow_html=True)
        
    except Exception as err:
        st.error(f"Failed to apply filters: {err}")

    return True

# ==================== FILTER APPLICATION ====================

def apply_filters(df: DataFrame) -> DataFrame:
    """
    Apply current filters from session state to a dataframe.
    
    Parameters:
        df: The dataframe to filter
        
    Returns:
        The filtered dataframe
    """
    # First apply custom PH destinations if configured
    df = apply_custom_ph_destinations(df, force=True)
    
    selections = st.session_state.get("filters", {})
    if not selections:
        return df

    df_filt = df.copy()
    
    # Apply each filter
    for col, vals in selections.items():
        if vals and col in df_filt.columns:
            df_filt = df_filt[df_filt[col].astype(str).isin(vals)]
    
    # Log filtering results
    original_rows = len(df)
    filtered_rows = len(df_filt)
    reduction_pct = (1 - filtered_rows / original_rows) * 100 if original_rows > 0 else 0
    
    if reduction_pct > 90:
        st.warning(
            f"⚠️ Filters removed {reduction_pct:.1f}% of data. "
            f"Consider relaxing filters if results seem limited."
        )

    return df_filt

# ==================== DATE RANGE VALIDATION ====================

def show_date_range_warning(df: DataFrame) -> None:
    """
    Display warnings if selected date ranges are outside data range.
    Enhanced with theme-aware styling.
    
    Parameters:
        df: The dataframe to check
    """
    try:
        data_min = df["ProjectStart"].min()
        data_max = df["ProjectExit"].max()
        
        warnings = []
        
        # Check each boundary
        boundaries = [
            (st.session_state.get("t0"), "Current window start"),
            (st.session_state.get("t1"), "Current window end"),
            (st.session_state.get("prev_start"), "Previous window start"),
            (st.session_state.get("prev_end"), "Previous window end"),
        ]
        
        for boundary, label in boundaries:
            if boundary:
                if boundary < data_min:
                    days_before = (data_min - boundary).days
                    warnings.append(
                        f"{label} is {days_before} days before available data"
                    )
                elif boundary > data_max:
                    days_after = (boundary - data_max).days
                    warnings.append(
                        f"{label} is {days_after} days after available data"
                    )
        
        if warnings:
            warning_content = (
                f"<strong>Date Range Issues:</strong><br>"
                f"• " + "<br>• ".join(warnings) + "<br><br>"
                f"<strong>Available data range:</strong> "
                f"{data_min.date()} to {data_max.date()}"
            )
            
            st.html(create_info_box(
                content=warning_content,
                type="warning",
                title="Date Range Warning",
                icon="📅"
            ))
            
    except (KeyError, AttributeError) as e:
        st.error(f"Data is missing required date columns: {e}")

# ==================== UTILITY FUNCTIONS ====================

def get_active_filters() -> Dict[str, Any]:
    """
    Get a summary of currently active filters.
    
    Returns:
        Dictionary with filter information
    """
    filters = st.session_state.get("filters", {})
    active_filters = {k: v for k, v in filters.items() if v}
    
    return {
        "date_range": {
            "current": (
                st.session_state.get("t0"),
                st.session_state.get("t1")
            ),
            "previous": (
                st.session_state.get("prev_start"),
                st.session_state.get("prev_end")
            )
        },
        "filters": active_filters,
        "filter_count": len(active_filters),
        "last_updated": st.session_state.get("last_filter_change", "Never")
    }

def reset_filters() -> None:
    """Reset all filters to defaults."""
    state = init_section_state(FILTER_FORM_KEY)
    
    # Clear all filter states
    state.clear()
    
    # Clear session state filters
    if "filters" in st.session_state:
        del st.session_state["filters"]
    
    # Update timestamp
    st.session_state["last_filter_change"] = datetime.now().isoformat()

# ==================== EXPORT PUBLIC API ====================

__all__ = [
    # Main functions
    'render_filter_form',
    'apply_filters',
    'show_date_range_warning',
    
    # State management
    'hash_data',
    'init_section_state',
    'get_filter_timestamp',
    'is_cache_valid',
    'invalidate_cache',
    
    # Utility functions
    'get_active_filters',
    'reset_filters',
    
    # Constants
    'DEFAULT_CURRENT_START',
    'DEFAULT_CURRENT_END',
    'FILTER_CATEGORIES',
]