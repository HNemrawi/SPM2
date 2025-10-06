"""
General analysis dashboard for HMIS data.
"""

from typing import Optional

import pandas as pd
import streamlit as st

from src.core.data.destinations import apply_custom_ph_destinations
from src.core.session import (
    ModuleType,
    SessionKeys,
    get_analysis_result,
    get_dashboard_state,
    get_session_manager,
    set_analysis_result,
)

# Import existing analysis modules
from src.modules.dashboard.data_utils import cached_load
from src.modules.dashboard.demographics import render_breakdown_section
from src.modules.dashboard.equity import render_equity_analysis
from src.modules.dashboard.filters import (
    apply_filters,
    render_filter_form,
    show_date_range_warning,
)
from src.modules.dashboard.length_of_stay import render_length_of_stay
from src.modules.dashboard.summary import render_summary_metrics
from src.modules.dashboard.trends import render_trend_explorer
from src.ui.factories.components import Colors
from src.ui.factories.components import (
    render_about_section as render_about_component,
)
from src.ui.factories.components import render_download_button, ui
from src.ui.factories.html import html_factory
from src.ui.layouts.templates import ABOUT_GENERAL_ANALYSIS_CONTENT
from src.ui.themes.styles import apply_custom_css

# Enhanced session management instances
DASHBOARD_MODULE = ModuleType.DASHBOARD
session_manager = get_session_manager()
dashboard_state = get_dashboard_state()


def apply_neutral_tab_style():
    """Apply neutral CSS for tab styling that adapts to light/dark themes."""
    st.html(
        f"""
        <style>
        /* Main tabs container with full width adaptive styling */
        .stTabs {{
        width: 100% !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
        gap: 0.75rem;
            background-color: transparent;
            padding: 1rem;
            border-radius: 0;
            border: none;
            margin: 0 -5rem;
            padding: 1rem 5rem;
            margin-bottom: 2rem;
            background: var(--background-secondary, rgba(0, 0, 0, 0.02));
            border-top: 1px solid var(--border-color, rgba(0, 0, 0, 0.1));
            border-bottom: 1px solid var(--border-color, rgba(0, 0, 0, 0.1));
            display: flex;
            justify-content: center;
        }}

        /* Individual tab styling as buttons */
        .stTabs [data-baseweb="tab"] {{
        height: 52px;
            padding: 0 40px;
            background-color: var(--tab-bg, rgba(255, 255, 255, 0.8));
            border: 1px solid var(--tab-border, rgba(0, 0, 0, 0.12));
            border-radius: 8px;
            color: var(--text-secondary, rgba(0, 0, 0, 0.7));
            font-weight: 500;
            font-size: 1rem;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            transition: all 0.2s ease;
            white-space: nowrap;
            position: relative;
            flex: 1 1 auto;
            min-width: 140px;
            max-width: 220px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            margin: 0 6px;
        }}


        /* Default tab styling for custom light theme */
        .stTabs [data-baseweb="tab"] {{
            background-color: rgba(255, 255, 255, 0.9);
            border-color: rgba(0, 0, 0, 0.08);
            color: rgba(0, 0, 0, 0.85);
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            background-color: rgba(255, 255, 255, 1);
            border-color: rgba(0, 0, 0, 0.15);
            color: rgba(0, 0, 0, 0.95);
        }}

        /* Hover state with enhanced feedback */
        .stTabs [data-baseweb="tab"]:hover {{
        transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }}

        /* Active/selected tab with primary accent */
        .stTabs [aria-selected="true"] {{
        background-color: {Colors.PRIMARY} !important;
            color: white !important;
            font-weight: 600;
            border-color: {Colors.PRIMARY} !important;
            box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3) !important;
            transform: translateY(-1px);
        }}

        /* Active tab hover state */
        .stTabs [aria-selected="true"]:hover {{
        background-color: {Colors.PRIMARY} !important;
            filter: brightness(0.9);
            border-color: {Colors.PRIMARY} !important;
            box-shadow: 0 3px 10px rgba(33, 150, 243, 0.4) !important;
        }}

        /* Tab panel content area */
        .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 0;
            background-color: transparent;
        }}

        /* Focus state for accessibility */
        .stTabs [data-baseweb="tab"]:focus {{
        outline: 2px solid {Colors.PRIMARY};
            outline-offset: 2px;
            box-shadow: 0 0 0 4px rgba(33, 150, 243, 0.1);
        }}

        /* Remove focus outline when not using keyboard */
        .stTabs [data-baseweb="tab"]:focus:not(:focus-visible) {{
        outline: none;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }}

        /* Tab content spacing */
        .tab-content {{
        padding: 1.5rem 0;
        }}

        /* Ensure consistent font throughout */
        .stTabs {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}

        /* Full width container adjustments */
        .main > .block-container {{
        max-width: 100%;
            padding-left: 5rem;
            padding-right: 5rem;
        }}

        /* Button press effect */
        .stTabs [data-baseweb="tab"]:active {{
        transform: translateY(0);
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }}

        /* Disabled state (if needed) */
        .stTabs [data-baseweb="tab"][disabled] {{
        opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }}

        /* Icon spacing in tabs */
        .stTabs [data-baseweb="tab"] > span {{
        display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        /* Responsive adjustments */
        @media (max-width: 768px) {{
            .stTabs [data-baseweb="tab-list"] {{
                margin: 0 -1rem;
                padding: 0.75rem 1rem;
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: thin;
                gap: 0.5rem;
                justify-content: flex-start;
            }}

            .stTabs [data-baseweb="tab"] {{
                padding: 0 20px;
                font-size: 0.9rem;
                height: 44px;
                margin: 0 3px;
                flex: 0 0 auto;
                min-width: 120px;
            }}

            .main > .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}

            /* Scrollbar styling for mobile */
            .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{
                height: 4px;
            }}

            .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {{
                background: rgba(0, 0, 0, 0.05);
            }}

            .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {{
                background: rgba(0, 0, 0, 0.2);
                border-radius: 2px;
            }}
        }}

        /* Smooth transitions for theme changes */
        .stTabs [data-baseweb="tab"],
        .stTabs [data-baseweb="tab-list"] {{
        transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
        }}

        /* Additional polish for button appearance */
        .stTabs [data-baseweb="tab"]::before {{
        content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0) 100%);
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s ease;
        }}

        .stTabs [data-baseweb="tab"]:hover::before {{
        opacity: 1;
        }}

        .stTabs [aria-selected="true"]::before {{
        display: none;
        }}
        </style>
    """
    )


def _setup_page_styling() -> None:
    """Apply global styling for the dashboard page."""
    apply_custom_css()
    apply_neutral_tab_style()


def _render_page_header() -> None:
    """Render the main page header with title and about section."""
    st.html(html_factory.divider("gradient"))
    st.html(
        html_factory.title("General Analysis Dashboard", level=1, icon="🏠")
    )

    render_about_component(
        title="About General Analysis Methodology",
        content=ABOUT_GENERAL_ANALYSIS_CONTENT,
        expanded=False,
        icon="📊",
    )
    st.html(html_factory.divider("gradient"))


def _load_and_validate_data() -> Optional[pd.DataFrame]:
    """Load and validate the data using enhanced session management.

    Returns:
        DataFrame if successful, None if validation fails
    """
    # Check both session manager and direct session state for compatibility
    if not session_manager.has_data() and "df" not in st.session_state:
        ui.info_section(
            content="Please upload your HMIS file in the sidebar to begin analysis.",
            type="warning",
            title="No Data Available",
            icon="📁",
            expanded=True,
        )
        return None

    # Try to get data from either source
    df_state = session_manager.get_data()
    if df_state is None:
        df_state = st.session_state.get("df")
    if df_state is None or (
        isinstance(df_state, pd.DataFrame) and df_state.empty
    ):
        ui.info_section(
            content="Data is not available. Please check your upload.",
            type="error",
            title="Data Error",
            icon="❌",
            expanded=True,
        )
        return None

    try:
        with st.spinner("Loading and preprocessing data..."):
            df = cached_load(df_state)
            df = apply_custom_ph_destinations(df, force=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        df = df_state.copy()
        df = apply_custom_ph_destinations(df, force=True)

    if df.empty:
        ui.info_section(
            content="No rows found after preprocessing. Please check your file format.",
            type="error",
            title="Processing Error",
            expanded=True,
        )
        return None

    return df


def _check_analysis_readiness(df: pd.DataFrame) -> bool:
    """Check if analysis is ready to run and guide user if not.

    Returns:
        True if analysis should proceed, False otherwise
    """
    st.session_state.setdefault(SessionKeys.FILTERS, {})
    render_filter_form(df)

    analysis_requested = dashboard_state.is_analysis_requested()
    has_date_range = SessionKeys.DATE_START in st.session_state

    if not has_date_range or not analysis_requested:
        if dashboard_state.is_dirty():
            ui.info_section(
                content="Parameters have changed. Click 'Run Dashboard Analysis' to update.",
                type="warning",
                title="Parameters Changed",
                icon="⚠️",
                expanded=True,
            )
        elif not has_date_range:
            ui.info_section(
                content="Configure date ranges and filters in the sidebar, then click 'Run Dashboard Analysis' to begin.",
                type="info",
                title="Getting Started",
                icon="🔍",
                expanded=True,
            )
        else:
            ui.info_section(
                content="Filters configured! Click 'Run Dashboard Analysis' in the sidebar to start the analysis.",
                type="success",
                title="Ready to Analyze",
                icon="▶️",
                expanded=True,
            )
        return False

    return True


def _prepare_filtered_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply filters and prepare data for analysis using enhanced session management.

    Returns:
        Filtered DataFrame
    """
    dashboard_state.clear_analysis_request()
    df_filt = apply_filters(df)

    # Store filtered data in both places for compatibility
    st.session_state[SessionKeys.DF_FILTERED] = df_filt
    set_analysis_result(DASHBOARD_MODULE, df_filt)

    # Clear dirty flag since data has been processed
    dashboard_state.clear_dirty()

    show_date_range_warning(df)
    return df_filt


def _render_analysis_tabs(df_filt: pd.DataFrame, df: pd.DataFrame) -> None:
    """Render analysis tabs with lazy loading - only active tab renders."""
    st.html("<div style='margin-top: 2rem;'></div>")

    tabs = ui.main_dashboard_tabs()

    # Lazy rendering: only render content within active tab context
    # This prevents ALL tabs from executing simultaneously
    with tabs[0]:  # Summary Metrics
        with st.container():
            st.html('<div class="tab-content">')
            render_summary_metrics(df_filt, df)
            st.html("</div>")

    with tabs[1]:  # Demographic Breakdown
        with st.container():
            st.html('<div class="tab-content">')
            render_breakdown_section(df_filt, df)
            st.html("</div>")

    with tabs[2]:  # Trends
        with st.container():
            st.html('<div class="tab-content">')
            render_trend_explorer(df_filt, df)
            st.html("</div>")

    with tabs[3]:  # Length of Stay
        with st.container():
            st.html('<div class="tab-content">')
            render_length_of_stay(df_filt)
            st.html("</div>")

    with tabs[4]:  # Equity Analysis
        with st.container():
            st.html('<div class="tab-content">')
            render_equity_analysis(df_filt, df)
            st.html("</div>")

    with tabs[5]:  # Data Export
        with st.container():
            st.html('<div class="tab-content">')
            _render_export_section()
            st.html("</div>")


def general_analysis_page() -> None:
    """Main entry point for the general analysis dashboard."""
    # Initialize enhanced session management
    dashboard_state.initialize()

    _setup_page_styling()
    _render_page_header()

    df = _load_and_validate_data()
    if df is None:
        return

    if not _check_analysis_readiness(df):
        return

    # Show important note about waiting for processing
    st.warning(
        "⏳ **Please wait until all processing is completed before "
        "interacting with the dashboard and filters.**",
        icon="⚠️",
    )

    # Check if we have cached results and filters haven't changed
    cached_df_filt = get_analysis_result(DASHBOARD_MODULE)
    filters_changed = dashboard_state.is_dirty()

    # Validate that cached_df_filt is actually a DataFrame
    if (
        cached_df_filt is not None
        and isinstance(cached_df_filt, pd.DataFrame)
        and not filters_changed
    ):
        # Use cached filtered data
        df_filt = cached_df_filt
        st.session_state[
            SessionKeys.DF_FILTERED
        ] = df_filt  # Also update session state for compatibility
    else:
        # Apply filters and prepare new data
        df_filt = _prepare_filtered_data(df)

    _render_analysis_tabs(df_filt, df)


def _render_export_section():
    """Render the data export section - minimal processing."""
    st.html(html_factory.title("Data Export", level=2, icon="📥"))

    # Retrieve the filtered data
    df_filt_cached = st.session_state.get(SessionKeys.DF_FILTERED)

    if df_filt_cached is not None and not df_filt_cached.empty:
        st.info("📊 Download your filtered dataset as CSV")

        render_download_button(
            df=df_filt_cached,
            filename="hmis_filtered_data",
            label="Download CSV",
            file_format="csv",
            key="download_csv",
        )

    else:
        st.warning(
            "⚠️ No data available. "
            "Please apply filters and run analysis first."
        )


# Run the app when script is executed directly
if __name__ == "__main__":
    general_analysis_page()
