"""HMIS Data Analysis Suite.

A professional HMIS data analysis platform with theme-adaptive UI for analyzing
homelessness data with multiple specialized analysis modules.
"""

from typing import Dict, Optional

import streamlit as st
import pandas as pd

from config.settings import PAGE_TITLE, PAGE_ICON, DEFAULT_LAYOUT, SIDEBAR_STATE
from config.themes import setup_plotly_theme, setup_pandas_options
from ui.styling import (
    apply_custom_css,
    NeutralColors,
    create_info_box,
    create_styled_divider,
    style_metric_cards,
    get_chart_colors
)
from ui.templates import render_footer, get_header_logo_html
from core.data_loader import load_and_preprocess_data, show_duplicate_info
from core.session import reset_session, check_data_available
from analysis.spm2.page import spm2_page
from analysis.inbound.page import inbound_recidivism_page
from analysis.outbound.page import outbound_recidivism_page
from analysis.general.general_analysis_page import general_analysis_page


def setup_page() -> None:
    """Configure the Streamlit page and set up themes."""
    st.set_page_config(
        page_title="HMIS Data Analysis",
        page_icon="📊",
        layout=DEFAULT_LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE
    )
    setup_plotly_theme()
    setup_pandas_options()
    apply_custom_css()


def render_enhanced_header() -> None:
    """Render the application header with instructions and logo."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 📋 Getting Started
        - Upload your data file to get started
        - Use the **🔄 Reset Session** button if you need to start over
        - Select from our four specialized analysis modules below
        """)

    with col2:
        st.html(get_header_logo_html())


def render_module_card(
    title: str,
    icon: str,
    color: str,
    description: str,
    keyword: str
) -> None:
    """Render a styled module information card.
    
    Args:
        title: Module title
        icon: Emoji icon for the module
        color: Border color for the card
        description: Module description text
        keyword: Highlighted keyword in the description
    """
    st.markdown(f"""
        <div style='padding: 1.5rem; background-color: rgba(0, 0, 0, 0.05); border-radius: 8px; border-left: 4px solid {color}; margin-bottom: 1rem;'>
            <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                <span style='font-size: 1.5rem; margin-right: 0.5rem;'>{icon}</span>
                <h4 style='color: inherit; margin: 0;'>{title}</h4>
            </div>
            <p style='color: inherit; opacity: 0.8; font-size: 0.875rem; margin: 0;'>
                <span style="color: #00629b; font-weight: bold;">{keyword}</span>: {description}
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_welcome_modules() -> None:
    """Render the available modules section for new users."""
    st.markdown("### 🎯 Available Analysis Modules")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_module_card(
            title="System Performance Measure 2",
            icon="📈",
            color="#0066CC",
            description="Looks for returns to homelessness based on client's first exit "
                       "(to perm destination by default) within the specified lookback period. "
                       "Returns must be within the specified return period.",
            keyword="SPM 2 Analysis"
        )
        
        render_module_card(
            title="Outbound Recidivism Analysis",
            icon="⬅️",
            color="#DC2626",
            description="Looks for returns to homelessness based on a client's last exit "
                       "during the reporting period. Any return found in the source report "
                       "is included, regardless of time to return.",
            keyword="Outbound Recidivism"
        )
    
    with col2:
        render_module_card(
            title="Inbound Recidivism Analysis",
            icon="➡️",
            color="#059862",
            description="Of all clients entering a set of programs during the specified "
                       "time period, how many are returners?",
            keyword="Inbound Recidivism"
        )
        
        render_module_card(
            title="General Comprehensive Dashboard",
            icon="📊",
            color="#D97706",
            description="Comprehensive HMIS data analysis with metrics, trends, demographics, "
                       "and equity analysis across your entire dataset.",
            keyword="General Analysis"
        )


def handle_duplicate_check(df: pd.DataFrame) -> None:
    """Handle manual duplicate checking for the dataframe.
    
    Args:
        df: The dataframe to check for duplicates
    """
    if st.button("🔍 Check for Duplicates", use_container_width=True):
        if "EnrollmentID" in df.columns:
            from core.data_loader import DuplicateAnalyzer
            duplicates_df, analysis = DuplicateAnalyzer.analyze_duplicates(df)
            if analysis.get("has_duplicates", False):
                st.session_state['duplicate_analysis'] = analysis
                if 'dedup_action' in st.session_state:
                    del st.session_state['dedup_action']
                st.info(f"Found {analysis['total_duplicate_records']} duplicate records!")
                st.rerun()
            else:
                st.success("No duplicate EnrollmentIDs found!")
        else:
            st.warning("EnrollmentID column not found in dataset")


def render_sidebar() -> Optional[str]:
    """Render the sidebar navigation and file upload.
    
    Returns:
        The selected analysis module name
    """
    modules: Dict[str, str] = {
        "System Performance Measure 2": "SPM2",
        "Inbound Recidivism Analysis": "Inbound Recidivism",
        "Outbound Recidivism Analysis": "Outbound Recidivism",
        "General Comprehensive Dashboard": "General Analysis"
    }
    
    module_descriptions: Dict[str, str] = {
        "System Performance Measure 2": 
            "Looks for returns to homelessness based on client's first exit "
            "(to perm destination by default) within the specified lookback period. "
            "Returns must be within the specified return period.",
        "Inbound Recidivism Analysis": 
            "Of all clients entering a set of programs during the specified "
            "time period, how many are returners?",
        "Outbound Recidivism Analysis": 
            "Looks for returns to homelessness based on a client's last exit "
            "during the reporting period. Any return found in the source report "
            "is included, regardless of time to return.",
        "General Comprehensive Dashboard": 
            "Comprehensive HMIS data analysis with metrics, trends, demographics, "
            "and equity analysis across your entire dataset."
    }
    
    module_icons: Dict[str, str] = {
        "System Performance Measure 2": "📈",
        "Inbound Recidivism Analysis": "➡️",
        "Outbound Recidivism Analysis": "⬅️",
        "General Comprehensive Dashboard": "📊"
    }
    
    with st.sidebar:
        st.header("🏠 HMIS Data Analysis")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Reset Session", use_container_width=True):
                reset_session()
                if 'duplicate_analysis' in st.session_state:
                    del st.session_state['duplicate_analysis']
                if 'dedup_action' in st.session_state:
                    del st.session_state['dedup_action']
                st.rerun()
        
        st.divider()
        st.subheader("📤 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls"],
            help="Upload your HMIS dataset in CSV or Excel format"
        )
        
        if uploaded_file is not None:
            if "df" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
                with st.spinner("📊 Processing your data..."):
                    df = load_and_preprocess_data(uploaded_file)
                    if df is not None and not df.empty:
                        st.session_state["df"] = df
                        st.session_state["current_file"] = uploaded_file.name
                        st.success(f"✅ Loaded {len(df):,} records successfully!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Invalid file format or processing error")
        else:
            if "df" in st.session_state and not st.session_state["df"].empty:
                df = st.session_state["df"]
                st.success(f"✅ Active Dataset: {len(df):,} records loaded")
                handle_duplicate_check(df)
            else:
                st.info("📁 Please upload a data file to begin")
        
        st.markdown("---")
        
        st.markdown("""
            <h3 style='font-size: 1rem; font-weight: 600; color: #1F2937; display: flex; align-items: center; margin-bottom: 1rem;'>
                <span style='margin-right: 0.5rem;'>🔍</span>
                Analysis Modules
            </h3>
        """, unsafe_allow_html=True)
        
        selected_module = st.selectbox(
            "Choose Analysis Type",
            options=list(modules.keys()),
            format_func=lambda x: f"{module_icons[x]} {x}",
            help="Select the type of analysis to perform"
        )
        
        st.session_state["selected_module"] = selected_module
        
        if selected_module and selected_module in module_descriptions:
            st.info(f"{module_icons[selected_module]} {module_descriptions[selected_module]}")
    
    return selected_module, modules


def render_welcome_screen() -> None:
    """Render the welcome screen for users without data loaded."""
    st.html("""
        <div style='
            text-align: center; 
            padding: 3rem 1rem; 
            background-color: rgba(128, 128, 128, 0.05); 
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 10px; 
            margin: 2rem 0;
        '>
            <h2 style='
                color: currentColor; 
                opacity: 0.9;
                font-size: 2rem; 
                margin-bottom: 1rem;
                font-weight: 600;
            '>
                Upload Your HMIS Data to Begin
            </h2>
            <div style='
                background: rgba(128, 128, 128, 0.1); 
                display: inline-block; 
                padding: 1rem 2rem; 
                border-radius: 8px; 
                border: 1px solid rgba(128, 128, 128, 0.2);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            '>
                <p style='
                    color: currentColor; 
                    opacity: 0.8;
                    margin: 0; 
                    font-weight: 500;
                '>
                    📤 Use the sidebar to upload your dataset
                </p>
            </div>
        </div>
    """)


def render_analysis_page(
    selected_module: str,
    modules: Dict[str, str],
    module_icons: Dict[str, str]
) -> None:
    """Render the selected analysis page.
    
    Args:
        selected_module: The name of the selected module
        modules: Mapping of module names to module keys
        module_icons: Mapping of module names to icons
    """
    df = st.session_state["df"]
    
    if 'duplicate_analysis' in st.session_state and not st.session_state.get('dedup_action'):
        show_duplicate_info(df, st.session_state['duplicate_analysis'])
        st.markdown("---")
    
    if st.session_state.get('dedup_action') == 'keep_all':
        if 'duplicate_analysis' in st.session_state:
            del st.session_state['duplicate_analysis']
        if 'dedup_action' in st.session_state:
            del st.session_state['dedup_action']
    
    module_key = modules.get(selected_module)
    
    if module_key:
        st.info(f"{module_icons[selected_module]} Now viewing: {selected_module}")
    
    try:
        if module_key == "SPM2":
            spm2_page()
        elif module_key == "Inbound Recidivism":
            inbound_recidivism_page()
        elif module_key == "Outbound Recidivism":
            outbound_recidivism_page()
        elif module_key == "General Analysis":
            general_analysis_page()
        else:
            st.error(f"Unknown analysis module: {module_key}")
            st.write(f"Selected: {selected_module}")
            st.write(f"Available modules: {list(modules.keys())}")
    
    except Exception as e:
        st.error(f"Error loading {selected_module} page:")
        st.exception(e)
        st.write("Debug information:")
        st.write(f"- Module: {selected_module}")
        st.write(f"- Module Key: {module_key}")
        st.write(f"- Data shape: {df.shape if 'df' in st.session_state else 'No data'}")


def main() -> None:
    """Run the HMIS Data Analysis Suite application."""
    setup_page()
    
    st.title("📊 HMIS Data Analysis Suite")
    
    if "df" not in st.session_state or st.session_state["df"].empty:
        render_enhanced_header()
        render_welcome_modules()
    
    selected_module, modules = render_sidebar()
    
    module_icons: Dict[str, str] = {
        "System Performance Measure 2": "📈",
        "Inbound Recidivism Analysis": "➡️",
        "Outbound Recidivism Analysis": "⬅️",
        "General Comprehensive Dashboard": "📊"
    }
    
    if "df" not in st.session_state or st.session_state["df"].empty:
        render_welcome_screen()
    else:
        render_analysis_page(selected_module, modules, module_icons)
    
    st.markdown("---")
    render_footer()
    
    st.markdown("""
        <div style='text-align: center; padding: 1rem; color: #9CA3AF; font-size: 0.75rem;'>
            <p style='margin: 0;'>HMIS Data Analysis Suite v2.0 | Built with Streamlit</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()