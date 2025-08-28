"""
HMIS Data Analysis Suite.

A professional HMIS data analysis platform with theme-adaptive UI for analyzing
homelessness data with multiple specialized analysis modules.
"""

from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from config.app_config import config
from src.core.data.loader import (
    DuplicateAnalyzer,
    load_and_preprocess_data,
    show_duplicate_info,
)
from src.core.session.manager import reset_session
from src.modules.dashboard.page import general_analysis_page
from src.modules.recidivism.inbound_page import inbound_recidivism_page
from src.modules.recidivism.outbound_page import outbound_recidivism_page
from src.modules.spm2.page import spm2_page
from src.ui.factories.html import html_factory
from src.ui.layouts.templates import get_header_logo_html, render_footer
from src.ui.themes.theme import theme

# Module configuration
MODULES = {
    "System Performance Measure 2": {
        "key": "SPM2",
        "icon": "📈",
        "color": theme.colors.primary,
        "description": (
            "Looks for returns to homelessness based on client's first exit "
            "(to perm destination by default) within the specified lookback "
            "period. Returns must be within the specified return period."
        ),
        "page_func": spm2_page,
    },
    "Inbound Recidivism": {
        "key": "Inbound",
        "icon": "➡️",
        "color": theme.colors.success,
        "description": (
            "Of all clients entering a set of programs during the specified "
            "time period, how many are returners?"
        ),
        "page_func": inbound_recidivism_page,
    },
    "Outbound Recidivism": {
        "key": "Outbound",
        "icon": "⬅️",
        "color": theme.colors.danger,
        "description": (
            "Looks for returns to homelessness based on a client's last exit "
            "during the reporting period. Any return found in the source "
            "report is included, regardless of time to return."
        ),
        "page_func": outbound_recidivism_page,
    },
    "General Dashboard": {
        "key": "General",
        "icon": "📊",
        "color": theme.colors.warning,
        "description": (
            "Comprehensive HMIS data analysis with metrics, trends, "
            "demographics, and equity analysis across your entire dataset."
        ),
        "page_func": general_analysis_page,
    },
}


def setup_page() -> None:
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title=config.page.title,
        page_icon=config.page.icon,
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            'Get Help': 'https://docs.streamlit.io',
            'Report a bug': "https://github.com/streamlit/streamlit/issues",
            'About': """
            # HMIS Data Analysis Suite
            
            A professional HMIS data analysis platform with theme-adaptive UI 
            for analyzing homelessness data with multiple specialized analysis modules.
            
            **Features:**
            - System Performance Measure 2 (SPM2) Analysis
            - Inbound/Outbound Recidivism Analysis  
            - Comprehensive Dashboard Analytics
            - Professional Theme Support
            - Advanced Data Filtering
            
            Built with Streamlit and designed for HMIS data professionals.
            """
        }
    )


def render_enhanced_header() -> None:
    """Render the application header with instructions and logo."""
    with st.container():
        # Use HTML factory for getting started section
        st.html(html_factory.title("Getting Started", level=3, icon="📋"))
        content = f"""
        <ol style="color: {theme.colors.text_secondary}; line-height: 1.8; 
           margin: 1rem 0;">
            <li><strong>Upload your HMIS data file</strong> using the 
                sidebar</li>
            <li><strong>Select an analysis module</strong> from the options 
                below</li>
            <li><strong>Configure filters and settings</strong> within each 
                module</li>
        </ol>
        """

        st.html(
            html_factory.info_box(
                content=content, type="info", title="Getting Started"
            )
        )


def render_module_card(
    title: str,
    icon: str,
    color: str,
    description: str,
    features: Optional[list] = None,
) -> None:
    """Render a styled module information card with consistent sizing.

    Args:
        title: Module title
        icon: Emoji icon for the module
        color: Border color for the card
        description: Module description text
        features: Optional list of key features
    """
    # Use HTML factory module card
    st.html(
        html_factory.module_card(
            title=title,
            description=description,
            icon=icon,
            color=color,
            features=features,
        )
    )


def render_welcome_modules() -> None:
    """Render the available modules section for new users."""
    st.html(
        html_factory.title("Available Analysis Modules", level=3, icon="🎯")
    )
    st.html(
        f"<p style='color: {theme.colors.text_muted}; "
        f"margin-bottom: 1.5rem;'>Choose the analysis type that "
        f"best fits your needs:</p>"
    )

    col1, col2 = st.columns(2)

    with col1:
        render_module_card(
            title="System Performance Measure 2",
            icon="📈",
            color=theme.colors.primary,
            description=(
                "Tracks housing stability by analyzing if and when clients "
                "return to homeless services after exiting to permanent "
                "housing. Examines exits within lookback period before the "
                "reporting period and categorizes returns by timing "
                "(<6 months, 6-12 months, 12-24 months, >24 months), "
                "same as SPM2."
            ),
        )

        render_module_card(
            title="Outbound Recidivism",
            icon="⬅️",
            color=theme.colors.danger,
            description=(
                "Similar to SPM2 but analyzes returns based on each "
                "client's last exit during the reporting period. Tracks "
                "both general returns and returns to homelessness (with "
                "special PH re-entry rules)."
            ),
        )

    with col2:
        render_module_card(
            title="Inbound Recidivism",
            icon="➡️",
            color=theme.colors.success,
            description=(
                "Tracks each client's first entry during the reporting "
                "period. Categorizes clients as: New (no prior exit), "
                "Returning (from non-housing), or Returning from Housing "
                "(from permanent destinations). Configurable lookback "
                "window for finding prior exits."
            ),
        )

        render_module_card(
            title="General Dashboard",
            icon="📊",
            color=theme.colors.warning,
            description=(
                "Comprehensive HMIS analysis across 5 modules: Overview "
                "(flow metrics, PH exits, returns), Demographics "
                "(breakdowns by race/gender/age), Trends (time series "
                "analysis), Length of Stay (enrollment-level), and Equity "
                "(statistical disparity testing with p-values and "
                "disparity indices)."
            ),
        )


def handle_duplicate_check(df: pd.DataFrame) -> None:
    """Handle manual duplicate checking for the dataframe.

    Args:
        df: The dataframe to check for duplicates
    """
    if st.button("🔍 Check Duplicates", width='stretch'):
        if "EnrollmentID" in df.columns:
            duplicates_df, analysis = DuplicateAnalyzer.analyze_duplicates(df)
            if analysis.get("has_duplicates", False):
                st.session_state["duplicate_analysis"] = analysis
                if "dedup_action" in st.session_state:
                    del st.session_state["dedup_action"]
                st.warning(
                    f"⚠️ {
                        analysis['total_duplicate_records']} duplicates found"
                )
                st.rerun()
            else:
                st.success("✅ No duplicates")
        else:
            st.warning("No EnrollmentID column")


def render_sidebar() -> Tuple[str, Dict[str, Dict]]:
    """Render the sidebar navigation and file upload.

    Returns:
        Tuple of selected module name and module configuration
    """
    with st.sidebar:
        st.html(get_header_logo_html())

        # Compact reset button
        if st.button(
            "↻ Reset",
            width='stretch',
            type="secondary",
            help="Clear all data and start over",
        ):
            reset_session()
            st.rerun()

        st.html(
            f"<div style='margin: 1rem 0; border-top: 1px solid "
            f"{theme.colors.border};'></div>"
        )

        # Data upload section - Professional and Clean
        st.html(html_factory.title("Data Source", level=4, icon="📁"))

        # Check if data is already loaded
        has_data = (
            "df" in st.session_state
            and not st.session_state.get("df", pd.DataFrame()).empty
        )

        if has_data:
            df = st.session_state["df"]
            filename = st.session_state.get("current_file", "Unknown")

            # Display data status using HTML factory
            st.html(
                html_factory.data_status_card(
                    filename=filename,
                    record_count=len(df),
                    status="active",
                )
            )

            # Streamlined action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "Check Data",
                    width='stretch',
                    type="secondary",
                ):
                    if "EnrollmentID" in df.columns:
                        (
                            duplicates_df,
                            analysis,
                        ) = DuplicateAnalyzer.analyze_duplicates(df)
                        if analysis.get("has_duplicates", False):
                            st.session_state["duplicate_analysis"] = analysis
                            if "dedup_action" in st.session_state:
                                del st.session_state["dedup_action"]
                            st.warning(
                                f"Found {analysis['total_duplicate_records']} "
                                f"duplicates"
                            )
                            st.rerun()
                        else:
                            st.success("No duplicates found")
                    else:
                        st.warning("EnrollmentID column not found")
            with col2:
                if st.button(
                    "Change File",
                    width='stretch',
                    type="secondary",
                ):
                    del st.session_state["df"]
                    del st.session_state["current_file"]
                    st.rerun()
        else:
            # Clean upload interface
            st.html(html_factory.upload_area())

            uploaded_file = st.file_uploader(
                "Choose file",
                type=["csv", "xlsx", "xls"],
                label_visibility="collapsed",
                key="file_uploader",
            )

            if uploaded_file is not None:
                with st.spinner("Loading data..."):
                    df = load_and_preprocess_data(uploaded_file)
                    if df is not None and not df.empty:
                        st.session_state["df"] = df
                        st.session_state["current_file"] = uploaded_file.name
                        st.rerun()
                    else:
                        st.error(
                            "Unable to process file. Please check the format."
                        )

        st.html(
            f"<div style='margin: 1.5rem 0; border-top: 1px solid "
            f"{theme.colors.border};'></div>"
        )

        # Module selection - Clean and Professional
        st.html(html_factory.title("Analysis Type", level=4, icon="📊"))

        selected_module = st.selectbox(
            "Select Analysis Type",
            options=list(MODULES.keys()),
            format_func=lambda x: f"{MODULES[x]['icon']} {x}",
            label_visibility="collapsed",
            key="module_selector",
        )

        st.session_state["selected_module"] = selected_module

        # Display module info in a clean, subtle card
        if selected_module in MODULES:
            module_info = MODULES[selected_module]
            st.html(
                f"""
                <div style='
                    background: {theme.colors.surface};
                    border: 1px solid {theme.colors.border};
                    border-left: 3px solid {module_info['color']};
                    border-radius: 6px;
                    padding: 0.75rem;
                    margin-top: 0.5rem;
                '>
                    <div style='color: {theme.colors.text_secondary}
                    ; font-size: 0.85rem; line-height: 1.4;'>
                        {module_info.get(
                            'description', 'Analysis module ready'
                        )}
                    </div>
                </div>
            """
            )

    return selected_module, MODULES


def render_welcome_screen() -> None:
    """Render the welcome screen for users without data loaded."""

    # Convert RGB hex to rgba for transparency
    def hex_to_rgba(hex_color, alpha=1.0):
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"

    primary_rgba_10 = hex_to_rgba(theme.colors.primary, 0.1)
    primary_rgba_05 = hex_to_rgba(theme.colors.primary, 0.05)
    primary_rgba_40 = hex_to_rgba(theme.colors.primary, 0.4)
    success_rgba_10 = hex_to_rgba(theme.colors.success, 0.1)

    st.html(
        f"""
    <div style='
        text-align: center;
        padding: 4rem 3rem;
        background: linear-gradient(135deg, {theme.colors.primary_bg}
            0%, {theme.colors.background} 50%, {theme.colors.background_secondary} 100%);
        border: 1px solid {theme.colors.border};
        border-radius: 20px;
        margin: 3rem auto;
        max-width: 700px;
        box-shadow: 0 10px 25px {primary_rgba_10}, 0 6px 10px {primary_rgba_05};
        position: relative;
        overflow: hidden;
    '>
        <div style='
            position: absolute;
            top: -50px;
            right: -50px;
            width: 200px;
            height: 200px;
            background: radial-gradient(circle, {primary_rgba_10} 0%, transparent 70%);
            border-radius: 50%;
        '></div>
        <div style='
            position: absolute;
            bottom: -30px;
            left: -30px;
            width: 150px;
            height: 150px;
            background: radial-gradient(circle, {success_rgba_10} 0%, transparent 70%);
            border-radius: 50%;
        '></div>

        <div style='font-size: 4.5rem; margin-bottom: 1.5rem; filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));'>📊</div>
        <h2 style='
            color: {theme.colors.text_primary};
            font-size: 2rem;
            margin-bottom: 1rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        '>
            Welcome to HMIS Data Analysis
        </h2>
        <p style='
            color: {theme.colors.text_secondary};
            font-size: 1.125rem;
            margin-bottom: 2.5rem;
            line-height: 1.75;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
        '>
            Analyze your HMIS data with powerful insights and visualizations
        </p>
        <div style='
            background: linear-gradient(135deg, {theme.colors.primary} 0%, {theme.colors.primary_hover} 100%);
            color: white;
            display: inline-block;
            padding: 1rem 2.5rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1.125rem;
            box-shadow: 0 4px 14px {primary_rgba_40};
            transition: transform 0.2s ease;
            cursor: pointer;
        '>
            📤 Upload Data in Sidebar →
        </div>

        <div style='
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid {theme.colors.border};
        '>
            <p style='
                color: {theme.colors.text_muted};
                font-size: 0.875rem;
                margin: 0;
            '>
                Supports CSV, Excel (XLSX, XLS) formats • Secure processing • No data stored
            </p>
        </div>
    </div>
    """
    )


def render_analysis_page(
    selected_module: str, modules_config: Dict[str, Dict]
) -> None:
    """Render the selected analysis page.

    Args:
        selected_module: The name of the selected module
        modules_config: Module configuration dictionary
    """
    df = st.session_state.get("df")
    if df is None:
        st.error("No data loaded. Please upload a file first.")
        return

    # Handle duplicate analysis if present
    if "duplicate_analysis" in st.session_state and not st.session_state.get(
        "dedup_action"
    ):
        show_duplicate_info(df, st.session_state["duplicate_analysis"])
        st.divider()

    if st.session_state.get("dedup_action") == "keep_all":
        for key in ["duplicate_analysis", "dedup_action"]:
            if key in st.session_state:
                del st.session_state[key]

    # Get module configuration
    module_config = modules_config.get(selected_module)
    if not module_config:
        st.error(f"Module '{selected_module}' not found")
        return

    # Render the module page
    try:
        module_config["page_func"]()
    except Exception as e:
        st.error(f"Error loading {selected_module}:")
        st.exception(e)
        if config.page.show_footer:
            with st.expander("Debug Information"):
                st.write(f"- Module: {selected_module}")
                st.write(f"- Data shape: {df.shape}")
                st.write(f"- Columns: {list(df.columns)[:10]}...")


def main() -> None:
    """Run the HMIS Data Analysis Suite application."""
    # Initialize page configuration
    setup_page()

    # Professional header with enhanced styling using theme colors
    st.html(
        f"""
    <div style="
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(180deg, {theme.colors.background_secondary}
            0%, {theme.colors.background} 100%);
        border-bottom: 1px solid {theme.colors.border};
        margin: -1rem -1rem 2rem -1rem;
    ">
        <h1 style="
            color: {theme.colors.primary};
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        ">
            📊 HMIS Data Analysis Suite
        </h1>
        <p style="
            color: {theme.colors.text_secondary};
            font-size: 1.125rem;
            margin: 0;
            font-weight: 400;
        ">
            Homeless Management Information Systems (HMIS) data analysis and reporting platform
        </p>
    </div>
    """
    )

    # Sidebar and module selection
    selected_module, modules_config = render_sidebar()

    # Main content area
    has_data = False
    if "df" in st.session_state:
        df = st.session_state.get("df")
        if df is not None and hasattr(df, "empty") and not df.empty:
            has_data = True

    if not has_data:
        # Welcome screen for new users
        render_enhanced_header()
        render_welcome_modules()
        render_welcome_screen()
    else:
        # Analysis page for users with data
        render_analysis_page(selected_module, modules_config)

    # Footer
    if config.page.show_footer:
        st.divider()
        render_footer()


if __name__ == "__main__":
    main()
