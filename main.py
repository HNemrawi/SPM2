"""
HMIS Data Analysis Suite.

A professional HMIS data analysis platform with theme-adaptive UI for analyzing
homelessness data with multiple specialized analysis modules.
"""

# Standard library imports
import json
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

# Third-party imports
import pandas as pd
import streamlit as st

# Local imports
from config.app_config import config
from src.core.data.loader import (
    DuplicateAnalyzer,
    load_and_preprocess_data,
    show_duplicate_info,
)
from src.core.session import (
    SessionKeys,
    get_session_manager,
    reset_session_manager,
)
from src.core.session.serializer import SessionSerializer
from src.modules.dashboard.page import general_analysis_page
from src.modules.recidivism.inbound_page import inbound_recidivism_page
from src.modules.recidivism.outbound_page import outbound_recidivism_page
from src.modules.spm2.page import spm2_page
from src.ui.factories.html import html_factory
from src.ui.layouts.templates import get_header_logo_html, render_footer
from src.ui.themes.theme import theme

# Enable pandas performance optimizations
pd.options.mode.copy_on_write = True
pd.options.compute.use_numexpr = True
pd.options.compute.use_bottleneck = True

# Initialize session management
session_manager = get_session_manager()

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
            "Get Help": "https://docs.streamlit.io",
            "Report a bug": "https://github.com/streamlit/streamlit/issues",
            "About": """
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
            """,
        },
    )


def inject_custom_css() -> None:
    """Inject custom CSS for improved native component styling."""
    custom_css = f"""
    <style>
    /* Soften Streamlit multiselect/filter pills */
    .stMultiSelect [data-baseweb="tag"] {{
        background-color: {theme.colors.primary_bg_subtle} !important;
        border: 1px solid {theme.colors.border} !important;
        color: {theme.colors.text_primary} !important;
    }}

    /* Improve table styling */
    .stDataFrame {{
        border: 1px solid {theme.colors.border} !important;
        border-radius: {theme.borders.radius_md} !important;
    }}

    .stDataFrame th {{
        background-color: {theme.colors.background_secondary} !important;
        color: {theme.colors.text_primary} !important;
        font-weight: 600 !important;
        padding: 12px !important;
        border-bottom: 2px solid {theme.colors.border} !important;
    }}

    .stDataFrame td {{
        padding: 10px 12px !important;
        border-bottom: 1px solid {theme.colors.border_light} !important;
    }}

    .stDataFrame tr:hover {{
        background-color: {theme.colors.surface_hover} !important;
    }}

    /* Soften alert boxes */
    .stAlert {{
        border-radius: {theme.borders.radius_md} !important;
        padding: 1rem !important;
    }}

    /* Improve button styling */
    .stButton > button {{
        border-radius: {theme.borders.radius_md} !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: {theme.shadows.md} !important;
    }}

    /* Soften expander styling */
    .streamlit-expanderHeader {{
        background-color: {theme.colors.background_secondary} !important;
        border-radius: {theme.borders.radius_sm} !important;
        font-weight: 600 !important;
    }}

    /* Improve select box styling */
    .stSelectbox [data-baseweb="select"] {{
        border-radius: {theme.borders.radius_md} !important;
    }}

    /* Tab styling improvements */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: {theme.borders.radius_sm} !important;
        padding: 8px 16px !important;
    }}

    /* Reduce visual weight of info messages */
    [data-testid="stNotification"] {{
        background-color: {theme.colors.info_bg_subtle} !important;
        border-left: 3px solid {theme.colors.info} !important;
    }}
    </style>
    """
    st.html(custom_css)


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
    if st.button(
        "🔍 Check Duplicates",
        width="stretch",
        key="manual_check_duplicates_button",
    ):
        if "EnrollmentID" in df.columns:
            duplicates_df, analysis = DuplicateAnalyzer.analyze_duplicates(df)
            if analysis.get("has_duplicates", False):
                st.session_state[SessionKeys.DUPLICATE_ANALYSIS] = analysis
                if SessionKeys.DEDUP_ACTION in st.session_state:
                    del st.session_state[SessionKeys.DEDUP_ACTION]
                st.warning(
                    f"⚠️ {analysis['total_duplicate_records']} "
                    "duplicates found"
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
            width="stretch",
            type="secondary",
            help="Clear all data and start over",
            key="reset_button",
        ):
            reset_session_manager()
            st.rerun()

        st.html(
            f"<div style='margin: 1rem 0; border-top: 1px solid "
            f"{theme.colors.border};'></div>"
        )

        # Data upload section - Professional and Clean
        st.html(html_factory.title("Data Source", level=4, icon="📁"))

        # Check if data is already loaded
        has_data = (
            SessionKeys.DF in st.session_state
            and st.session_state.get(SessionKeys.DF) is not None
            and isinstance(st.session_state.get(SessionKeys.DF), pd.DataFrame)
            and not st.session_state.get(SessionKeys.DF).empty
        )

        if has_data:
            df = st.session_state[SessionKeys.DF]
            filename = st.session_state.get(
                SessionKeys.CURRENT_FILE, "Unknown"
            )

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
                    width="stretch",
                    type="secondary",
                    key="check_data_button",
                ):
                    if "EnrollmentID" in df.columns:
                        (
                            duplicates_df,
                            analysis,
                        ) = DuplicateAnalyzer.analyze_duplicates(df)
                        if analysis.get("has_duplicates", False):
                            st.session_state[
                                SessionKeys.DUPLICATE_ANALYSIS
                            ] = analysis
                            if SessionKeys.DEDUP_ACTION in st.session_state:
                                del st.session_state[SessionKeys.DEDUP_ACTION]
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
                    width="stretch",
                    type="secondary",
                    key="change_file_button",
                ):
                    # Batch cleanup to avoid multiple operations
                    keys_to_remove = [
                        SessionKeys.DF,
                        SessionKeys.DATA,
                        SessionKeys.CURRENT_FILE,
                        SessionKeys.DATA_LOADED,
                        SessionKeys.DUPLICATE_ANALYSIS,
                        SessionKeys.DEDUP_ACTION,
                    ]
                    for key in keys_to_remove:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

            # Session Export/Import section
            st.html(html_factory.divider())
            st.html(
                html_factory.title("Session Management", level=5, icon="💾")
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "📤 Export",
                    width="stretch",
                    type="secondary",
                    help="Export current session settings",
                    key="export_session_button",
                ):
                    st.session_state[SessionKeys.SHOW_EXPORT_DIALOG] = True

            with col2:
                if st.button(
                    "📥 Import",
                    width="stretch",
                    type="secondary",
                    help="Import session settings",
                    key="import_session_button",
                ):
                    st.session_state[SessionKeys.SHOW_IMPORT_DIALOG] = True
        else:
            # Clean upload interface
            st.html(html_factory.upload_area())

            uploaded_file = st.file_uploader(
                "Choose file",
                type=["csv"],
                label_visibility="collapsed",
                key="file_uploader",
            )

            if uploaded_file is not None:
                with st.spinner("Loading data..."):
                    try:
                        df = load_and_preprocess_data(uploaded_file)
                        if (
                            df is not None
                            and isinstance(df, pd.DataFrame)
                            and not df.empty
                        ):
                            # Batch session state updates to avoid multiple reruns
                            st.session_state.update(
                                {
                                    SessionKeys.DF: df,
                                    SessionKeys.DATA: df,  # Add this for SessionManager compatibility
                                    SessionKeys.CURRENT_FILE: uploaded_file.name,
                                    SessionKeys.DATA_LOADED: True,
                                }
                            )
                            st.rerun()
                        else:
                            st.error(
                                "Unable to process file. Please check the format."
                            )
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
                        st.info("Please check your file format and try again.")

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

        # Track module changes and store current module
        if (
            st.session_state.get(SessionKeys.SELECTED_MODULE)
            != selected_module
        ):
            st.session_state[SessionKeys.SELECTED_MODULE] = selected_module
            st.session_state[SessionKeys.CURRENT_MODULE] = selected_module

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


@st.dialog("💾 Export Analysis Configuration", width="large")
def show_export_dialog():
    """Display modal export dialog."""
    st.markdown("### Save your current analysis settings")

    # Get current session info
    session_manager.get_session_summary()
    current_module = session_manager.get(
        SessionKeys.SELECTED_MODULE, "Unknown"
    )
    module_short = (
        current_module.replace("System Performance Measure 2", "SPM2")
        .replace(" ", "_")
        .lower()
    )

    # Session metadata inputs
    session_name = st.text_input(
        "Session Name",
        value=f"{module_short}_session_{datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Give this configuration a memorable name",
        key="export_session_name_input",
    )

    session_description = st.text_area(
        "Description (Optional)",
        placeholder="e.g., Annual report filters for 2024...",
        help="Add notes about this configuration",
        key="export_session_description_input",
        height=80,
    )

    # Export options
    st.markdown("---")
    st.markdown("### Export Options")

    current_module_only = st.checkbox(
        "Export current module only",
        value=True,
        help="Uncheck to export settings from all modules",
        key="export_current_module_checkbox",
    )

    # Preview what will be exported
    export_data = session_manager.export_state(
        include_all=False,
        current_module_only=current_module_only,
        session_name=session_name,
        session_description=session_description,
    )

    config_summary = export_data.get("configuration_summary", {})

    st.markdown("---")
    st.markdown("### What Will Be Saved")

    # Show clean summary
    st.html(
        html_factory.info_box(
            f"""
            <strong>Module:</strong> {export_data.get('module', 'Unknown')}<br/>
            <strong>Data File:</strong> {export_data['session_info'].get('data_file', 'Unknown')}<br/>
            <strong>Total Settings:</strong> {len(export_data.get('state', {}))} items
            """,
            type="info",
            title="Session Information",
        )
    )

    # Show configuration summary
    if config_summary:
        summary_text = ""
        if "date_ranges" in config_summary:
            summary_text += "<strong>📅 Date Ranges:</strong><br/>"
            for date_range in config_summary["date_ranges"]:
                summary_text += f"&nbsp;&nbsp;• {date_range}<br/>"

        if "filters" in config_summary:
            summary_text += "<br/><strong>🔍 Filters Configured:</strong><br/>"
            for filter_name, filter_info in config_summary["filters"].items():
                summary_text += (
                    f"&nbsp;&nbsp;• {filter_name}: {filter_info}<br/>"
                )

        if "lookback_period" in config_summary:
            summary_text += f"<br/><strong>⏮ Lookback:</strong> {config_summary['lookback_period']}<br/>"
        if "return_period" in config_summary:
            summary_text += f"<strong>⏭ Return Period:</strong> {config_summary['return_period']}"

        if summary_text:
            st.html(
                html_factory.info_box(
                    summary_text, type="success", title="Configuration Details"
                )
            )

    # Action buttons
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button(
            "❌ Cancel", use_container_width=True, key="export_cancel_btn"
        ):
            st.session_state[SessionKeys.SHOW_EXPORT_DIALOG] = False
            st.rerun()

    with col2:
        # Generate JSON for download
        json_str = session_manager.export_to_json(
            include_all=False,
            current_module_only=current_module_only,
            session_name=session_name,
            session_description=session_description,
        )

        filename = f"{session_name}.json"

        st.download_button(
            label="💾 Download Session File",
            data=json_str,
            file_name=filename,
            mime="application/json",
            use_container_width=True,
            type="primary",
            key="export_download_btn",
        )


@st.dialog("📥 Import Analysis Configuration", width="large")
def show_import_dialog():
    """Display modal import dialog."""
    st.markdown("### Load a saved analysis configuration")

    uploaded_session = st.file_uploader(
        "Choose session file",
        type=["json"],
        help="Select a previously exported .json session file",
        key="import_session_file_uploader",
        label_visibility="collapsed",
    )

    if uploaded_session is not None:
        try:
            # Read and parse the JSON file
            json_str = uploaded_session.read().decode("utf-8")
            session_data = json.loads(json_str)

            # Show preview using SessionSerializer
            summary = SessionSerializer.create_session_summary(session_data)

            st.markdown("---")
            st.markdown("### Session Preview")

            # Show session info (v2.1+ format)
            if "name" in summary:
                try:
                    st.html(
                        html_factory.info_box(
                            f"""
                            <strong>Name:</strong> {summary.get('name', 'Unnamed')}<br/>
                            <strong>Description:</strong> {summary.get('description', 'No description')}<br/>
                            <strong>Created:</strong> {pd.to_datetime(summary.get('created_at', '')).strftime('%b %d, %Y at %I:%M %p') if summary.get('created_at') else 'Unknown'}<br/>
                            <strong>Module:</strong> {summary.get('module', 'Unknown')}<br/>
                            <strong>Version:</strong> {summary.get('version', 'Unknown')}
                            """,
                            type="info",
                            title="Session Information",
                        )
                    )
                except Exception as e:
                    st.warning(f"⚠️ Could not display session info: {str(e)}")

                # Show configuration summary
                try:
                    config = summary.get("configuration", {})
                    if config:
                        config_text = ""
                        if "date_ranges" in config:
                            config_text += (
                                "<strong>📅 Date Ranges:</strong><br/>"
                            )
                            for date_range in config["date_ranges"]:
                                # Convert to string safely (handle lists/tuples)
                                date_str = (
                                    str(date_range)
                                    if not isinstance(date_range, str)
                                    else date_range
                                )
                                config_text += f"&nbsp;&nbsp;• {date_str}<br/>"

                        if "filters" in config:
                            config_text += (
                                "<br/><strong>🔍 Filters:</strong><br/>"
                            )
                            for filter_name, filter_info in config[
                                "filters"
                            ].items():
                                # Convert to string safely (handle lists/nested structures)
                                info_str = (
                                    str(filter_info)
                                    if not isinstance(filter_info, str)
                                    else filter_info
                                )
                                config_text += f"&nbsp;&nbsp;• {filter_name}: {info_str}<br/>"

                        if "lookback_period" in config:
                            config_text += f"<br/><strong>⏮ Lookback:</strong> {config['lookback_period']}<br/>"
                        if "return_period" in config:
                            config_text += f"<strong>⏭ Return Period:</strong> {config['return_period']}"

                        if config_text:
                            st.html(
                                html_factory.info_box(
                                    config_text,
                                    type="success",
                                    title="Configuration",
                                )
                            )
                except Exception as e:
                    st.warning(f"⚠️ Could not display configuration: {str(e)}")

            # Validate compatibility
            try:
                current_hash = None
                if session_manager.has_data():
                    df = session_manager.get_data()
                    current_hash = SessionSerializer.compute_df_hash(df)

                issues = SessionSerializer.validate_import(
                    session_data, current_hash
                )

                if issues:
                    # Ensure all issues are strings (defensive)
                    issues_text = "<br/>".join(
                        [str(issue) for issue in issues]
                    )
                    st.html(
                        html_factory.info_box(
                            f"<strong>Notes:</strong><br/>{issues_text}",
                            type="warning",
                            title="Import Compatibility",
                        )
                    )
            except Exception as e:
                st.warning(f"⚠️ Could not validate compatibility: {str(e)}")

            # Action buttons
            st.markdown("---")
            col1, col2 = st.columns([1, 1])

            with col1:
                if st.button(
                    "❌ Cancel",
                    use_container_width=True,
                    key="import_cancel_btn",
                ):
                    st.session_state[SessionKeys.SHOW_IMPORT_DIALOG] = False
                    st.rerun()

            with col2:
                if st.button(
                    "✅ Import & Apply",
                    use_container_width=True,
                    type="primary",
                    key="import_confirm_btn",
                ):
                    import_issues = session_manager.import_state(
                        session_data, validate=True
                    )
                    if not import_issues:
                        st.success("✅ Session imported successfully!")
                        st.session_state[
                            SessionKeys.SHOW_IMPORT_DIALOG
                        ] = False
                        # Small delay to show success message
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        # Ensure all issues are strings (defensive)
                        issues_str = "\n".join(
                            [str(issue) for issue in import_issues]
                        )
                        st.error(f"❌ Import failed:\n{issues_str}")

        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON file: {str(e)}")
        except Exception as e:
            st.error(f"❌ Failed to read session file: {str(e)}")
    else:
        st.info("👆 Upload a session file to preview and import")

        # Show example of what to expect
        with st.expander("ℹ️ About Session Files"):
            st.markdown(
                """
                Session files are JSON files that contain your analysis configuration:
                - Module selection (SPM2, Dashboard, etc.)
                - Date ranges and analysis periods
                - Filter selections (programs, agencies, etc.)
                - Module-specific settings

                **Note:** Session files do NOT contain your actual data, only the settings.
                """
            )


def handle_session_dialogs() -> None:
    """Handle session export/import dialogs using st.dialog()."""
    # Export dialog - use st.dialog() decorator
    if st.session_state.get(SessionKeys.SHOW_EXPORT_DIALOG, False):
        show_export_dialog()

    # Import dialog - use st.dialog() decorator
    if st.session_state.get(SessionKeys.SHOW_IMPORT_DIALOG, False):
        show_import_dialog()


def render_analysis_page(
    selected_module: str, modules_config: Dict[str, Dict]
) -> None:
    """Render the selected analysis page.

    Args:
        selected_module: The name of the selected module
        modules_config: Module configuration dictionary
    """
    df = st.session_state.get(SessionKeys.DF)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.error("No data loaded. Please upload a file first.")
        return

    # Handle duplicate analysis if present
    if (
        SessionKeys.DUPLICATE_ANALYSIS in st.session_state
        and not st.session_state.get(SessionKeys.DEDUP_ACTION)
    ):
        show_duplicate_info(
            df, st.session_state[SessionKeys.DUPLICATE_ANALYSIS]
        )
        st.divider()

    if st.session_state.get(SessionKeys.DEDUP_ACTION) == "keep_all":
        for key in [SessionKeys.DUPLICATE_ANALYSIS, SessionKeys.DEDUP_ACTION]:
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


def main() -> None:
    """Run the HMIS Data Analysis Suite application."""
    # Initialize page configuration
    setup_page()

    # Inject custom CSS for improved component styling
    inject_custom_css()

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
    if SessionKeys.DF in st.session_state:
        df = st.session_state.get(SessionKeys.DF)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            has_data = True

    if not has_data:
        # Welcome screen for new users
        render_enhanced_header()
        render_welcome_modules()
        render_welcome_screen()
    else:
        # Analysis page for users with data
        render_analysis_page(selected_module, modules_config)

    # Handle session export/import dialogs
    handle_session_dialogs()

    # Footer
    if config.page.show_footer:
        st.divider()
        render_footer()


if __name__ == "__main__":
    main()
