"""Filter utilities for HMIS dashboard."""

import hashlib
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from pandas import DataFrame

from src.core.data.destinations import apply_custom_ph_destinations
from src.core.session import (
    ModuleType,
    SessionKeys,
    get_dashboard_state,
    get_session_manager,
)
from src.core.utils.helpers import (
    check_date_range_validity,
    create_multiselect_filter,
)
from src.ui.factories.components import Colors, ui
from src.ui.factories.html import html_factory

# ==================== CONSTANTS ====================

# Module identifier for state management
DASHBOARD_MODULE = ModuleType.DASHBOARD
dashboard_state = get_dashboard_state()

# Default date ranges
DEFAULT_CURRENT_START = date(2024, 10, 1)
DEFAULT_CURRENT_END = date(2025, 9, 30)

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
    },
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


def init_section_state(
    key: str, defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
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
            st.session_state[
                SessionKeys.SELECTED_PH_DESTINATIONS
            ] = section_state["selected_ph_destinations"]

    return section_state


def get_filter_timestamp() -> str:
    """Get the current filter change timestamp from session state."""
    return st.session_state.get(SessionKeys.LAST_FILTER_CHANGE, "")


def is_cache_valid(
    state: Dict[str, Any], timestamp: Optional[str] = None
) -> bool:
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


def invalidate_cache(
    state: Dict[str, Any], timestamp: Optional[str] = None
) -> None:
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
    st.markdown(
        f"""
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
    """,
        unsafe_allow_html=True,
    )


def _render_date_period_info(
    start: date,
    end: date,
    label: str,
    period_days: int,
    is_custom: bool = False,
    sidebar: bool = False,
) -> None:
    """
    Render formatted date period information using UI factory.

    Parameters:
        start: Start date
        end: End date
        label: Period label
        period_days: Number of days in period
        is_custom: Whether this is a custom period
        sidebar: Whether to render in sidebar or main area
    """
    icon = "📅" if not is_custom else "✏️"
    info_type = "info" if not is_custom else "warning"

    content = f"""
    <strong>{
        start.strftime('%b %d, %Y')} to {
        end.strftime('%b %d, %Y')}</strong><br>
    <span style='font-size: 0.875rem; opacity: 0.8;'>({period_days} days)</span>
    """

    info_box_html = html_factory.info_box(
        content=content, type=info_type, title=label, icon=icon
    )

    if sidebar:
        st.sidebar.html(info_box_html)
    else:
        st.html(info_box_html)


def render_ph_destination_selector(df: DataFrame) -> None:
    """Render a selector for customizing PH destinations."""
    if (
        "ExitDestination" not in df.columns
        or "ExitDestinationCat" not in df.columns
    ):
        return

    # Get PH destinations from original data
    ph_mask = df["ExitDestinationCat"] == "Permanent Housing Situations"
    original_ph_destinations = sorted(
        df.loc[ph_mask, "ExitDestination"].dropna().unique()
    )

    if not original_ph_destinations:
        return

    all_destinations = sorted(df["ExitDestination"].dropna().unique())

    st.html(
        html_factory.title(
            text="Permanent Housing Selection", level=3, icon="🏠"
        )
    )

    # Initialize session state ONLY if it doesn't exist
    if "selected_ph_destinations" not in st.session_state:
        st.session_state[SessionKeys.SELECTED_PH_DESTINATIONS] = set(
            original_ph_destinations
        )

    # Get current selections for default, ensuring they exist in options
    current_selections = st.session_state.get(
        "selected_ph_destinations", set()
    )
    valid_selections = [
        dest for dest in current_selections if dest in all_destinations
    ]

    # Destination selector WITHOUT callback (forms don't allow callbacks)
    # The widget's value becomes the source of truth after user interaction
    selected = st.multiselect(
        "Select destinations to count as Permanent Housing:",
        options=all_destinations,
        default=valid_selections,
        key="ph_destination_selector",
        help="Choose which exit destinations should be considered permanent housing",
    )

    # Update stored value from widget (after user interaction or on first render)
    st.session_state[SessionKeys.SELECTED_PH_DESTINATIONS] = set(selected)

    # Show count and status
    num_selected = len(selected)
    total_destinations = len(all_destinations)
    num_original = len(original_ph_destinations)

    # Show status (no button since we're inside a form)
    if num_selected == num_original and set(selected) == set(
        original_ph_destinations
    ):
        st.info(
            f"✅ Using default PH destinations "
            f"({num_selected} of {total_destinations} total)"
        )
    else:
        st.info(
            f"🎯 Custom PH destinations active "
            f"({num_selected} of {total_destinations} total)\n\n"
            f"💡 To reset: Clear all selections and re-select the "
            f"original {num_original} destinations"
        )


def render_ph_destination_selector_immediate(df: DataFrame) -> None:
    """Render a selector for customizing PH destinations with immediate callbacks."""
    if (
        "ExitDestination" not in df.columns
        or "ExitDestinationCat" not in df.columns
    ):
        return

    # Get PH destinations from original data
    ph_mask = df["ExitDestinationCat"] == "Permanent Housing Situations"
    original_ph_destinations = sorted(
        df.loc[ph_mask, "ExitDestination"].dropna().unique()
    )

    if not original_ph_destinations:
        return

    all_destinations = sorted(df["ExitDestination"].dropna().unique())

    st.html(
        html_factory.title(
            text="Permanent Housing Selection", level=3, icon="🏠"
        )
    )

    # Initialize session state ONLY if it doesn't exist
    if "selected_ph_destinations" not in st.session_state:
        st.session_state[SessionKeys.SELECTED_PH_DESTINATIONS] = set(
            original_ph_destinations
        )

    # Get current selections for default, ensuring they exist in options
    current_selections = st.session_state.get(
        "selected_ph_destinations", set()
    )
    valid_selections = [
        dest for dest in current_selections if dest in all_destinations
    ]

    # Destination selector WITH callback for immediate application
    selected = st.multiselect(
        "Select destinations to count as Permanent Housing:",
        options=all_destinations,
        default=valid_selections,
        key="ph_destination_selector_widget",
        help="Choose which exit destinations should be considered permanent housing",
        on_change=lambda: _update_filter_state("ph_destinations"),
    )

    # Show count and status
    num_selected = len(selected)
    total_destinations = len(all_destinations)
    num_original = len(original_ph_destinations)

    # Show status with immediate feedback
    if num_selected == num_original and set(selected) == set(
        original_ph_destinations
    ):
        st.info(
            f"✅ Using default PH destinations ({num_selected} of {total_destinations} total)"
        )
    else:
        st.info(
            f"🎯 Custom PH destinations active ({num_selected} of {total_destinations} total)\n\n"
            f"💡 To reset: Clear all selections and re-select the original {num_original} destinations"
        )


# ==================== FILTER VALIDATION ====================


def _validate_date_range(
    start: date,
    end: date,
    label: str = "Date range",
    df: Optional[DataFrame] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a date range with optional ReportingPeriod bounds checking.

    Parameters:
        start: Start date
        end: End date
        label: Label for error messages
        df: Optional DataFrame to check ReportingPeriod bounds against

    Returns:
        Tuple of (is_valid, error_message)
    """
    if start >= end:
        return False, f"{label}: Start date must be before end date."

    # Check for unreasonable date ranges
    days_diff = (end - start).days
    if days_diff > 3650:  # 10 years
        return False, f"{label}: Date range exceeds 10 years."

    # Check against ReportingPeriod bounds if data is available
    if df is not None and not df.empty:
        return _validate_against_reporting_period(start, end, df, label)

    return True, None


def _validate_against_reporting_period(
    start: date, end: date, df: DataFrame, label: str = "Date range"
) -> Tuple[bool, Optional[str]]:
    """
    Validate dates against ReportingPeriod bounds.

    Parameters:
        start: Start date
        end: End date
        df: DataFrame containing ReportingPeriod columns
        label: Label for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if ReportingPeriod columns exist
        if (
            "ReportingPeriodStartDate" not in df.columns
            or "ReportingPeriodEndDate" not in df.columns
        ):
            return True, None  # Skip validation if columns don't exist

        # Get reporting period bounds
        reporting_start = pd.to_datetime(
            df["ReportingPeriodStartDate"].iloc[0]
        ).date()
        reporting_end = pd.to_datetime(
            df["ReportingPeriodEndDate"].iloc[0]
        ).date()

        # Use the existing validation function for consistency
        is_valid, message = check_date_range_validity(
            analysis_start=start,
            analysis_end=end,
            data_start=reporting_start,
            data_end=reporting_end,
            warn=False,  # Don't show UI warnings here, we'll handle them
            error_on_invalid=False,
        )

        # If there are issues, use the universal warning instead of the old message
        if not is_valid:
            from src.core.utils.helpers import show_universal_date_warning

            issues = []
            if start < reporting_start:
                days_before = (reporting_start - start).days
                issues.append(
                    f"Analysis starts {days_before} days before available data"
                )
            if end > reporting_end:
                days_after = (end - reporting_end).days
                issues.append(
                    f"Analysis ends {days_after} days after available data"
                )

            # Show the universal warning with DataFrame context
            show_universal_date_warning(
                issues,
                pd.to_datetime(start),
                pd.to_datetime(end),
                pd.to_datetime(reporting_start),
                pd.to_datetime(reporting_end),
                df,
            )

            # Return simple message for the filter context
            return (
                False,
                f"{label}: Date range outside available data period ({reporting_start} to {reporting_end})",
            )

        return True, None

    except Exception as e:
        st.warning(f"Could not validate against reporting period: {e}")
        return True, None  # Don't block usage if validation fails


def _validate_filter_selections(
    selections: Dict[str, List[str]],
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
            warnings.append(
                "No project types selected - this may result in empty data."
            )

    return True, warnings


# ==================== MAIN FILTER FORM ====================


def _get_current_module_prefix() -> str:
    """Get the module prefix for the currently selected module."""
    current_module = st.session_state.get(
        "selected_module", "General Analysis"
    )

    # Map module names to module prefixes
    module_mapping = {
        "General Dashboard": "dashboard",
        "System Performance Measure 2": "spm2",
        "Inbound Recidivism": "inbound",
        "Outbound Recidivism": "outbound",
    }

    return module_mapping.get(current_module, "dashboard")


def _update_filter_state(filter_type: str = "general") -> None:
    """Streamlined single filter update function using enhanced session system."""
    try:
        # Use single filters dictionary for all filter state
        if "filters" not in st.session_state:
            st.session_state[SessionKeys.FILTERS] = {}

        filters = st.session_state[SessionKeys.FILTERS]

        # Update based on filter type
        if filter_type == "ph_destinations":
            if "ph_destination_selector_widget" in st.session_state:
                ph_dest = set(
                    st.session_state["ph_destination_selector_widget"]
                )
                filters["selected_ph_destinations"] = ph_dest
                # Also update the main session state key for consistency
                st.session_state[
                    SessionKeys.SELECTED_PH_DESTINATIONS
                ] = ph_dest
        else:
            # Update general filters
            for category_name, category_filters in FILTER_CATEGORIES.items():
                for label, col in category_filters.items():
                    widget_key = f"filter_{col}_widget"
                    if widget_key in st.session_state:
                        pick = st.session_state[widget_key]
                        filters[col] = [] if "ALL" in pick else pick

        # Set timestamp using enhanced session system
        st.session_state[
            SessionKeys.LAST_FILTER_CHANGE
        ] = datetime.now().isoformat()

        # Generate a hash of current filter state to compare with last known state
        import json

        filter_hash = hashlib.md5(
            json.dumps(filters, sort_keys=True, default=str).encode()
        ).hexdigest()

        # Only set dirty flag if filters actually changed
        current_module_prefix = _get_current_module_prefix()
        if current_module_prefix == DASHBOARD_MODULE:
            dashboard_state.check_and_mark_dirty("filter_state", filter_hash)
        else:
            # For other modules, get their state and check with value-aware dirty checking
            session_manager = get_session_manager()
            if current_module_prefix == "spm2":
                spm2_state = session_manager.get_module_state(ModuleType.SPM2)
                if spm2_state:
                    spm2_state.check_and_mark_dirty(
                        "filter_state", filter_hash
                    )
            elif current_module_prefix == "inbound":
                inbound_state = session_manager.get_module_state(
                    ModuleType.INBOUND
                )
                if inbound_state:
                    inbound_state.check_and_mark_dirty(
                        "filter_state", filter_hash
                    )
            elif current_module_prefix == "outbound":
                outbound_state = session_manager.get_module_state(
                    ModuleType.OUTBOUND
                )
                if outbound_state:
                    outbound_state.check_and_mark_dirty(
                        "filter_state", filter_hash
                    )

    except Exception as err:
        st.error(f"Failed to update filter state: {err}")


# Deprecated function removed - was causing false "Parameters Changed" warnings
# Dirty checking is now handled only when Apply button is clicked


def _apply_filters_immediate(df: DataFrame) -> None:
    """Apply filters immediately when changed (optimized single-click pattern)."""
    try:
        # Batch all filter updates together
        filter_updates = {}

        # Get filter selections from session state widgets
        selections = {}
        for category_name, category_filters in FILTER_CATEGORIES.items():
            for label, col in category_filters.items():
                if col not in df.columns:
                    continue

                widget_key = f"filter_{col}_widget"
                if widget_key in st.session_state:
                    pick = st.session_state[widget_key]
                    selections[col] = [] if "ALL" in pick else pick

        # Handle PH destination selections
        selected_ph_destinations = None
        if "ph_destination_selector_widget" in st.session_state:
            selected_ph_destinations = set(
                st.session_state["ph_destination_selector_widget"]
            )

        # Batch update session state to avoid multiple triggers
        filter_updates.update(
            {
                "filters": selections,
                "last_filter_change": datetime.now().isoformat(),
            }
        )

        if selected_ph_destinations is not None:
            filter_updates[
                "selected_ph_destinations"
            ] = selected_ph_destinations

        # Apply all updates at once to minimize session state changes
        st.session_state.update(filter_updates)

        # Use enhanced session system for consistent dirty flag handling with debouncing
        dashboard_state.mark_dirty(debounce=True)

    except Exception as err:
        st.error(f"Failed to apply filters: {err}")


def render_filter_form(df: DataFrame) -> bool:
    """
    Render a sidebar filter form with immediate filter application.
    Uses single-click pattern for consistency with other modules.

    Parameters:
        df: The full dataset

    Returns:
        True if filters were applied, False otherwise
    """
    # Apply custom styling
    _apply_filter_form_styling()

    # Retrieve or initialize per-section state
    state = init_section_state(FILTER_FORM_KEY)

    # Load saved date ranges from enhanced session system, fallback to section state
    saved_dates = dashboard_state.get_widget_state("date_range", None)
    if saved_dates:
        win_start_default = saved_dates[0]
        win_end_default = saved_dates[1]
    else:
        win_start_default = state.get("win_start", DEFAULT_CURRENT_START)
        win_end_default = state.get("win_end", DEFAULT_CURRENT_END)

    # Header with improved styling using UI factory
    st.sidebar.html(
        html_factory.card(
            content="<p style='margin: 0.5rem 0 0 0;'>Set the analysis window and apply filters immediately.</p>",
            title="Reporting Window & Filters",
            icon="🗓️",
            border_color=Colors.PRIMARY,
        )
    )

    # Custom previous checkbox with enhanced styling and callback
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

    # ===== PH Destinations Customization =====
    if "ExitDestination" in df.columns:
        with st.sidebar.expander(
            "⚙️ Customize Permanent Housing Destinations", expanded=False
        ):
            render_ph_destination_selector_immediate(df)
        st.sidebar.markdown("---")

    # ===== Date Range Section =====
    st.sidebar.html(html_factory.title(text="Time Period", level=3, icon="📅"))

    # Main window inputs with callback and error handling
    date_range_result = st.sidebar.date_input(
        "Current Reporting Period",
        value=(win_start_default, win_end_default),
        help="Primary analysis reporting period (inclusive)",
        key="date_range_widget",
    )

    # Handle incomplete date selection
    if (
        isinstance(date_range_result, (list, tuple))
        and len(date_range_result) == 2
    ):
        win_start, win_end = date_range_result
    elif isinstance(date_range_result, date):
        # Only one date selected, use defaults for the other
        win_start = date_range_result
        win_end = win_end_default
        st.sidebar.info("Please select both start and end dates")
    else:
        # Fallback to defaults
        win_start, win_end = win_start_default, win_end_default
        st.sidebar.warning("Invalid date selection, using defaults")

    # Validate current period only if we have both dates
    if win_start and win_end:
        is_valid, error_msg = _validate_date_range(
            win_start, win_end, "Current period", df
        )
        if not is_valid:
            st.sidebar.error(error_msg)

    # Calculate auto-generated previous window only if we have valid dates
    if win_start and win_end:
        current_period_days = (
            pd.Timestamp(win_end) - pd.Timestamp(win_start)
        ).days + 1
        auto_prev_start = win_start - pd.Timedelta(days=current_period_days)
        auto_prev_end = win_start - pd.Timedelta(days=1)

        # Only show auto-generated previous period info when NOT using custom comparison
        if not custom_prev:
            _render_date_period_info(
                auto_prev_start,
                auto_prev_end,
                "Previous period (auto)",
                current_period_days,
                sidebar=True,
            )
    else:
        # Use defaults if dates are invalid
        current_period_days = 365
        auto_prev_start = win_start_default - pd.Timedelta(
            days=current_period_days
        )
        auto_prev_end = win_start_default - pd.Timedelta(days=1)

    # Custom previous window inputs
    if custom_prev:
        st.sidebar.html(
            html_factory.title(
                text="Custom Previous Period",
                level=4,
                icon="📅",
            )
        )

        # Determine defaults
        if "custom_prev_start" in state and "custom_prev_end" in state:
            prev_start_default = state["custom_prev_start"]
            prev_end_default = state["custom_prev_end"]
        else:
            prev_start_default = auto_prev_start
            prev_end_default = auto_prev_end

        prev_date_range_result = st.sidebar.date_input(
            "Previous window dates",
            value=(prev_start_default, prev_end_default),
            help="Comparison period for change metrics",
            key="prev_date_range_widget",
        )

        # Handle incomplete date selection for previous period
        if (
            isinstance(prev_date_range_result, (list, tuple))
            and len(prev_date_range_result) == 2
        ):
            prev_start, prev_end = prev_date_range_result
        elif isinstance(prev_date_range_result, date):
            # Only one date selected, use defaults for the other
            prev_start = prev_date_range_result
            prev_end = prev_end_default
            st.sidebar.info(
                "Please select both start and end dates for previous period"
            )
        else:
            # Fallback to defaults
            prev_start, prev_end = prev_start_default, prev_end_default
            st.sidebar.warning(
                "Invalid previous period date selection, using defaults"
            )

        # Validate previous period only if we have both dates
        if prev_start and prev_end:
            is_valid, error_msg = _validate_date_range(
                prev_start, prev_end, "Previous period", df
            )
            if not is_valid:
                st.sidebar.error(error_msg)

        # Show comparison info only if both dates are valid
        if prev_start and prev_end and win_start and win_end:
            custom_days = (
                pd.Timestamp(prev_end) - pd.Timestamp(prev_start)
            ).days + 1
            if custom_days != current_period_days:
                st.sidebar.warning(
                    f"⚠️ Period length mismatch: {custom_days} days vs "
                    f"{current_period_days} days for current period"
                )
    else:
        prev_start, prev_end = auto_prev_start, auto_prev_end

    st.sidebar.markdown("---")

    # ===== Return Window Section =====
    st.sidebar.html(
        html_factory.title(text="Return Tracking", level=3, icon="🔄")
    )

    return_window = st.sidebar.number_input(
        "Days to track returns to homelessness",
        min_value=7,
        max_value=1095,
        value=state.get("return_window", 180),
        step=30,
        key="return_window_widget",
        help="Number of days after PH exit to track returns to homelessness",
    )

    # Store return window in state
    state["return_window"] = return_window

    st.sidebar.markdown("---")

    # ===== Filters Section =====
    st.sidebar.html(html_factory.title(text="Data Filters", level=3, icon="🔍"))

    # Organize filters by category with immediate callbacks
    for category_name, category_filters in FILTER_CATEGORIES.items():
        with st.sidebar.expander(f"**{category_name}**", expanded=False):
            for label, col in category_filters.items():
                if col not in df.columns:
                    continue

                options = sorted(df[col].dropna().astype(str).unique())
                default_sel = state.get(f"filter_{col}", ["ALL"])
                widget_key = f"filter_{col}_widget"

                # Create multiselect with persistent state and callback
                create_multiselect_filter(
                    label=label,
                    options=options,
                    default=default_sel,
                    help_text=f"Filter by {label}. Select 'ALL' for no filter.",
                    key=widget_key,
                    on_change=lambda: _update_filter_state("general"),
                    module=DASHBOARD_MODULE,
                )

    # Commit date changes to state when they change (only if dates are valid)
    if (
        win_start
        and win_end
        and prev_start
        and prev_end
        and (
            win_start != state.get("win_start")
            or win_end != state.get("win_end")
        )
    ):
        state["win_start"] = win_start
        state["win_end"] = win_end
        state["prev_start"] = prev_start
        state["prev_end"] = prev_end

        # Save custom dates if applicable
        if custom_prev:
            state["custom_prev_start"] = prev_start
            state["custom_prev_end"] = prev_end
        else:
            state.pop("custom_prev_start", None)
            state.pop("custom_prev_end", None)

        # Update session state with date changes
        st.session_state[SessionKeys.DATE_START] = pd.Timestamp(win_start)
        st.session_state[SessionKeys.DATE_END] = pd.Timestamp(win_end)
        st.session_state[SessionKeys.PREV_START] = pd.Timestamp(prev_start)
        st.session_state[SessionKeys.PREV_END] = pd.Timestamp(prev_end)
        st.session_state[
            SessionKeys.LAST_FILTER_CHANGE
        ] = datetime.now().isoformat()

    # Show analysis button for explicit triggering (consistent with other
    # modules)
    if st.sidebar.button(
        "▶️ Run Dashboard Analysis",
        type="primary",
        width="stretch",
        help="Run analysis with current filters and date ranges",
    ):
        dashboard_state.request_analysis()
        st.sidebar.success("✅ Analysis ready to run!")
        return True

    # Check if dirty flag indicates filters changed
    return st.session_state.get(SessionKeys.DASHBOARD_DIRTY, False)


# ==================== FILTER APPLICATION ====================


@st.cache_data(show_spinner=False)
def _apply_filters_cached(df: DataFrame, filter_tuple: tuple) -> DataFrame:
    """
    Cached filter application for performance.
    OPTIMIZED: Uses combined mask to avoid repeated DataFrame copies.

    Parameters:
        df: The dataframe to filter
        filter_tuple: Tuple of (column, values) tuples for filtering

    Returns:
        The filtered dataframe
    """
    if not filter_tuple:
        return df

    # OPTIMIZED: Build a combined mask instead of copying df repeatedly
    import pandas as pd

    combined_mask = pd.Series(True, index=df.index)

    # Apply each filter to the mask
    for col, vals in filter_tuple:
        if vals and col in df.columns:
            # Convert to list if single value
            val_list = (
                [vals] if not isinstance(vals, (list, tuple)) else list(vals)
            )

            # Use pre-converted string column if available for performance
            str_col = col + "_str"
            if str_col in df.columns:
                combined_mask &= df[str_col].isin(val_list)
            else:
                # Fallback to on-the-fly conversion
                combined_mask &= df[col].astype(str).isin(val_list)

    return df[combined_mask]


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

    # Get current filter selections from session
    selections = st.session_state.get(SessionKeys.FILTERS, {})

    if not selections:
        return df

    # Convert selections dict to hashable tuple for caching
    filter_tuple = tuple(
        sorted(
            (k, tuple(v) if isinstance(v, (list, set)) else v)
            for k, v in selections.items()
        )
    )

    # Use cached version for actual filtering
    df_filt = _apply_filters_cached(df, filter_tuple)

    # Log filtering results (outside cache for UI feedback)
    original_rows = len(df)
    filtered_rows = len(df_filt)
    reduction_pct = (
        (1 - filtered_rows / original_rows) * 100 if original_rows > 0 else 0
    )

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
        # Use ReportingPeriod dates as the primary data range if available
        if (
            "ReportingPeriodStartDate" in df.columns
            and "ReportingPeriodEndDate" in df.columns
        ):
            data_min = pd.to_datetime(df["ReportingPeriodStartDate"].iloc[0])
            data_max = pd.to_datetime(df["ReportingPeriodEndDate"].iloc[0])
        else:
            # Fallback to enrollment date ranges
            data_min = df["ProjectStart"].min()
            data_max = df["ProjectExit"].max()

        warnings = []

        # Check each boundary
        boundaries = [
            (
                st.session_state.get(SessionKeys.DATE_START),
                "Current window start",
            ),
            (st.session_state.get(SessionKeys.DATE_END), "Current window end"),
            (
                st.session_state.get(SessionKeys.PREV_START),
                "Previous window start",
            ),
            (
                st.session_state.get(SessionKeys.PREV_END),
                "Previous window end",
            ),
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

        # Check against ReportingPeriod dates if available
        reporting_warnings = []
        if (
            "ReportingPeriodStartDate" in df.columns
            and "ReportingPeriodEndDate" in df.columns
        ):
            try:
                reporting_start = pd.to_datetime(
                    df["ReportingPeriodStartDate"].iloc[0]
                )
                reporting_end = pd.to_datetime(
                    df["ReportingPeriodEndDate"].iloc[0]
                )

                for boundary, label in boundaries:
                    if boundary:
                        if boundary < reporting_start:
                            days_before = (reporting_start - boundary).days
                            reporting_warnings.append(
                                f"{label} is {days_before} days before reporting period"
                            )
                        elif boundary > reporting_end:
                            days_after = (boundary - reporting_end).days
                            reporting_warnings.append(
                                f"{label} is {days_after} days after reporting period"
                            )
            except Exception:
                pass  # Skip ReportingPeriod validation if it fails

        if warnings or reporting_warnings:
            warning_content_parts = []

            if warnings:
                warning_content_parts.append(
                    "<strong>Data Availability Issues:</strong><br>"
                    "• " + "<br>• ".join(warnings)
                )

            if reporting_warnings:
                warning_content_parts.append(
                    "<strong>Reporting Period Issues:</strong><br>"
                    "• " + "<br>• ".join(reporting_warnings)
                )

            warning_content = "<br><br>".join(warning_content_parts)

            # Determine the correct label for data_min/data_max
            if (
                "ReportingPeriodStartDate" in df.columns
                and "ReportingPeriodEndDate" in df.columns
            ):
                try:
                    reporting_start = pd.to_datetime(
                        df["ReportingPeriodStartDate"].iloc[0]
                    )
                    reporting_end = pd.to_datetime(
                        df["ReportingPeriodEndDate"].iloc[0]
                    )

                    # Check if data_min/data_max are ReportingPeriod dates
                    if (
                        data_min.date() == reporting_start.date()
                        and data_max.date() == reporting_end.date()
                    ):
                        warning_content += f"<br><br><strong>Official reporting period:</strong> {data_min.date()} to {data_max.date()}"

                        # Also show enrollment data range
                        if (
                            "ProjectStart" in df.columns
                            and "ProjectExit" in df.columns
                        ):
                            enrollment_min = df["ProjectStart"].min()
                            enrollment_max = df["ProjectExit"].max()
                            warning_content += f"<br><strong>Enrollment data spans:</strong> {enrollment_min.date()} to {enrollment_max.date()}"
                    else:
                        warning_content += f"<br><br><strong>Enrollment data range:</strong> {data_min.date()} to {data_max.date()}"
                        warning_content += f"<br><strong>Official reporting period:</strong> {reporting_start.date()} to {reporting_end.date()}"
                except Exception:
                    warning_content += f"<br><br><strong>Available data range:</strong> {data_min.date()} to {data_max.date()}"
            else:
                warning_content += f"<br><br><strong>Available data range:</strong> {data_min.date()} to {data_max.date()}"

            ui.info_section(
                content=warning_content,
                type="warning",
                title="Date Range Warning",
                icon="📅",
                expanded=True,
            )

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
                st.session_state.get(SessionKeys.DATE_START),
                st.session_state.get(SessionKeys.DATE_END),
            ),
            "previous": (
                st.session_state.get(SessionKeys.PREV_START),
                st.session_state.get(SessionKeys.PREV_END),
            ),
        },
        "filters": active_filters,
        "filter_count": len(active_filters),
        "last_updated": st.session_state.get("last_filter_change", "Never"),
    }


def reset_filters() -> None:
    """Reset all filters to defaults."""
    state = init_section_state(FILTER_FORM_KEY)

    # Clear all filter states
    state.clear()

    # Clear session state filters
    if "filters" in st.session_state:
        del st.session_state[SessionKeys.FILTERS]

    # Update timestamp
    st.session_state[
        SessionKeys.LAST_FILTER_CHANGE
    ] = datetime.now().isoformat()


# ==================== EXPORT PUBLIC API ====================

__all__ = [
    # Main functions
    "render_filter_form",
    "apply_filters",
    "show_date_range_warning",
    # State management
    "hash_data",
    "init_section_state",
    "get_filter_timestamp",
    "is_cache_valid",
    "invalidate_cache",
    # Utility functions
    "get_active_filters",
    "reset_filters",
    # Constants
    "DEFAULT_CURRENT_START",
    "DEFAULT_CURRENT_END",
    "FILTER_CATEGORIES",
]
