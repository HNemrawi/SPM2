"""
Homelessness Analysis Suite
===========================
This Streamlit application analyzes HMIS data with two main modules:
  1. SPM2 Analysis – for exit and return analysis with flexible filtering.
  2. Inbound Recidivism Analysis – for classifying new versus returning clients.

The code is modularized into sections:
    • Setup & Configuration
    • Custom CSS & Styling
    • Data Loading & Preprocessing
    • Helper Functions (SPM2 Logic, Metrics, Visualizations)
    • SPM2 Analysis Page
    • Inbound Recidivism Analysis Page
    • Main Application

Please upload your HMIS CSV/Excel file in the sidebar. Use the reset button if you wish to clear the session.
"""

import streamlit as st
from config import setup_page
from styling import apply_custom_css
from data_preprocessing import load_and_preprocess_data
from spm2_page import spm2_page
from inbound_page import inbound_recidivism_page
from outbound_recidivism_page import outbound_recidivism_page

def reset_session():
    """
    Reset the Streamlit session state.
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def main():
    """
    Main function to run the Homelessness Analysis Suite.
    
    It sets up the page, applies custom styling, handles data upload, and navigates between analysis pages.
    """
    setup_page()
    apply_custom_css()
    st.title("Return to Homelessness Analysis")
    st.markdown("""
    - Upload your data file to get started.
    - Use the **Reset Session** button if you need to start over.
    - Navigate between the available analyses:
        1. **SPM 2 Analysis**
        2. **Inbound Recidivism Analysis**
        2. **Outbound Recidivism Analysis**
    """)

    # Sidebar: Data Upload & Reset Section.
    st.sidebar.header("📂 Data Upload")
    if st.sidebar.button("Reset Data"):
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

    # Sidebar: Navigation Section.
    st.sidebar.header("⚙️ Navigation")
    pages = ["SPM2", "Inbound Recidivism","Outbound Recidivism"]
    choice = st.sidebar.radio("Select a Page", pages)
    if "df" not in st.session_state or st.session_state["df"].empty:
        st.warning("Please upload a valid dataset to proceed.")
        return

    if choice == "SPM2":
        spm2_page()
    elif choice == "Inbound Recidivism":
        inbound_recidivism_page()
    elif choice == "Outbound Recidivism":
        outbound_recidivism_page()

if __name__ == "__main__":
    main()
