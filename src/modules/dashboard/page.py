"""
General analysis dashboard for HMIS data.
Updated to use neutral theme system that works with both light and dark modes.
"""

from typing import Optional

import pandas as pd
import streamlit as st

from src.core.data.destinations import apply_custom_ph_destinations

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
from src.ui.factories.components import (
    Colors,
    create_insight_container,
    styled_metric,
    ui,
)
from src.ui.factories.html import html_factory
from src.ui.layouts.templates import ABOUT_GENERAL_ANALYSIS_CONTENT
from src.ui.layouts.widgets import (
    render_about_section as render_about_component,
)
from src.ui.layouts.widgets import (
    render_dataframe_with_style,
    render_download_button,
)
from src.ui.themes.styles import apply_custom_css


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

        /* Dark mode adjustments */
        @media (prefers-color-scheme: dark) {{
            .stTabs [data-baseweb="tab-list"] {{
                background: rgba(255, 255, 255, 0.02);
                border-top-color: rgba(255, 255, 255, 0.1);
                border-bottom-color: rgba(255, 255, 255, 0.1);
            }}

            .stTabs [data-baseweb="tab"] {{
                background-color: rgba(255, 255, 255, 0.05);
                border-color: rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.7);
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
            }}

            .stTabs [data-baseweb="tab"]:hover {{
                background-color: rgba(255, 255, 255, 0.1);
                border-color: rgba(255, 255, 255, 0.2);
                color: rgba(255, 255, 255, 0.9);
            }}
        }}

        /* Light mode specific */
        @media (prefers-color-scheme: light) {{
            .stTabs [data-baseweb="tab"] {{
                background-color: rgba(255, 255, 255, 0.9);
                border-color: rgba(0, 0, 0, 0.08);
                color: rgba(0, 0, 0, 0.7);
            }}

            .stTabs [data-baseweb="tab"]:hover {{
                background-color: rgba(255, 255, 255, 1);
                border-color: rgba(0, 0, 0, 0.15);
                color: rgba(0, 0, 0, 0.87);
            }}
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
    """Load and validate the data from session state.

    Returns:
        DataFrame if successful, None if validation fails
    """
    df_state = st.session_state.get("df")
    if df_state is None or df_state.empty:
        ui.info_section(
            content="Please upload your HMIS file in the sidebar to begin analysis.",
            type="warning",
            title="No Data Available",
            icon="📁",
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
    st.session_state.setdefault("filters", {})
    render_filter_form(df)

    analysis_requested = st.session_state.get(
        "dashboard_analysis_requested", False
    )

    if "t0" not in st.session_state or not analysis_requested:
        if "t0" not in st.session_state:
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
    """Apply filters and prepare data for analysis.

    Returns:
        Filtered DataFrame
    """
    st.session_state["dashboard_analysis_requested"] = False
    df_filt = apply_filters(df)
    st.session_state["df_filt"] = df_filt
    show_date_range_warning(df)
    return df_filt


def _render_analysis_tabs(df_filt: pd.DataFrame, df: pd.DataFrame) -> None:
    """Render all analysis tabs with their content."""
    st.html("<div style='margin-top: 2rem;'></div>")

    tabs = ui.main_dashboard_tabs()

    # Define tab content renderers
    tab_renderers = [
        lambda: render_summary_metrics(df_filt, df),
        lambda: render_breakdown_section(df_filt, df),
        lambda: render_trend_explorer(df_filt, df),
        lambda: render_length_of_stay(df_filt),
        lambda: render_equity_analysis(df_filt, df),
        lambda: _render_export_section(),
    ]

    for tab, renderer in zip(tabs, tab_renderers):
        with tab:
            with st.container():
                st.html('<div class="tab-content">')
                renderer()
                st.html("</div>")


def general_analysis_page() -> None:
    """Main entry point for the general analysis dashboard."""
    _setup_page_styling()
    _render_page_header()

    df = _load_and_validate_data()
    if df is None:
        return

    if not _check_analysis_readiness(df):
        return

    df_filt = _prepare_filtered_data(df)
    _render_analysis_tabs(df_filt, df)


def _render_export_section():
    """Render the data export section with consistent styling."""
    st.html(html_factory.title("Data Export", level=2, icon="📥"))

    # Create insight container for export info
    st.html(
        create_insight_container(
            title="Export Your Analysis",
            content="Download your filtered dataset for further analysis in Excel, R, Python, or other tools.",
            type="info",
        )
    )

    # Retrieve the filtered data
    df_filt_cached = st.session_state.get("df_filt")

    if df_filt_cached is not None and not df_filt_cached.empty:
        # Stats about the filtered data
        col1, col2, col3 = st.columns(3)

        with col1:
            styled_metric(
                label="Total Rows",
                value=len(df_filt_cached),
                help="Number of records in filtered dataset",
            )

        with col2:
            styled_metric(
                label="Total Columns",
                value=len(df_filt_cached.columns),
                help="Number of data fields",
            )

        with col3:
            # Calculate data size estimate
            size_mb = (
                df_filt_cached.memory_usage(deep=True).sum() / 1024 / 1024
            )
            styled_metric(
                label="Estimated Size",
                value=f"{size_mb:.1f} MB",
                help="Approximate file size",
            )

        st.html(html_factory.divider("gradient"))

        # Download options in columns
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            render_download_button(
                df=df_filt_cached,
                filename="hmis_filtered_data",
                label="Download as CSV",
                file_format="csv",
                key="download_csv",
            )

        with col2:
            render_download_button(
                df=df_filt_cached,
                filename="hmis_filtered_data",
                label="Download as Excel",
                file_format="xlsx",
                key="download_xlsx",
            )

        with col3:
            render_download_button(
                df=df_filt_cached,
                filename="hmis_filtered_data",
                label="Download as JSON",
                file_format="json",
                key="download_json",
            )

        # Data preview section
        st.html(html_factory.divider("gradient"))
        st.html(html_factory.title("Data Preview", level=3, icon="👁️"))

        # Use the styled dataframe renderer
        render_dataframe_with_style(
            df=df_filt_cached.head(100),
            caption=f"Showing first 100 rows of {len(df_filt_cached):,} total rows",
            height=400,
            show_index=False,
        )

        # Column information expander
        with st.expander("📋 View Column Information", expanded=False):
            col_info = pd.DataFrame(
                {
                    "Column": df_filt_cached.columns,
                    "Type": df_filt_cached.dtypes.astype(str),
                    "Non-Null Count": df_filt_cached.count(),
                    "Null Count": df_filt_cached.isna().sum(),
                    "Unique Values": df_filt_cached.nunique(),
                }
            )

            render_dataframe_with_style(
                df=col_info, highlight_cols=["Null Count"], show_index=False
            )

    else:
        ui.info_section(
            content="No filtered data available. Please apply filters first to export data.",
            type="warning",
            title="No Data to Export",
            expanded=True,
        )


# Run the app when script is executed directly
if __name__ == "__main__":
    general_analysis_page()
