"""
Homelessness Analysis Suite
--------------------------
Main application entry point that configures the app and renders the appropriate pages.
"""

import streamlit as st
from datetime import datetime

# Application setup
from config.settings import PAGE_TITLE, PAGE_ICON, DEFAULT_LAYOUT, SIDEBAR_STATE
from config.themes import setup_plotly_theme, setup_pandas_options

# UI components
from ui.styling import apply_custom_css
from ui.templates import render_header, render_footer

# Core functionality
from core.data_loader import load_and_preprocess_data
from core.session import reset_session, check_data_available

# Analysis pages
from analysis.spm2.page import spm2_page
from analysis.inbound.page import inbound_recidivism_page  
from analysis.outbound.page import outbound_recidivism_page
from analysis.general.general_analysis_page import general_analysis_page


def setup_page():
    """Configure the Streamlit page and set up themes."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=DEFAULT_LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE
    )
    setup_plotly_theme()
    setup_pandas_options()
    

def main():
    """
    Main function to run the Homelessness Analysis Suite.
    
    It sets up the page, applies custom styling, handles data upload, and navigates between analysis pages.
    """
    setup_page()
    apply_custom_css()
    st.title("Return to Homelessness Analysis")
    
    # Render header with logo
    render_header()

    # Sidebar: Data Upload & Reset Section
    st.sidebar.header("📂 Data Upload")
    if st.sidebar.button("Reset Session"):
        reset_session()
        st.rerun()

    if "df" not in st.session_state:
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV/Excel File",
            type=["csv", "xlsx", "xls"],
            help="Upload an HMIS CSV or Excel file for analysis."
        )
        if uploaded_file:
            with st.spinner("Loading data..."):
                df_loaded = load_and_preprocess_data(uploaded_file)
                if not df_loaded.empty:
                    st.session_state["df"] = df_loaded
            if "df" in st.session_state and not st.session_state["df"].empty:
                st.sidebar.success("Data loaded successfully!")
        else:
            st.sidebar.info("Please upload a data file to proceed.")

    # Sidebar: Navigation Section
    st.sidebar.header("⚙️ Navigation")
    pages = ["SPM2", "Inbound Recidivism", "Outbound Recidivism", "General Analysis"]
    choice = st.sidebar.radio("Select a Page", pages)
    
    # Check data availability before rendering pages
    if "df" not in st.session_state or st.session_state["df"].empty:
        st.warning("Please upload a valid dataset to proceed.")
        return

    # Render the selected page
    if choice == "SPM2":
        spm2_page()
    elif choice == "Inbound Recidivism":
        inbound_recidivism_page()
    elif choice == "Outbound Recidivism":
        outbound_recidivism_page()
    elif choice == "General Analysis":
        general_analysis_page()

    # Render footer
    render_footer()


if __name__ == "__main__":
    main()