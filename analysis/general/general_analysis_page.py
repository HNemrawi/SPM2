"""
General analysis dashboard for HMIS data.
"""

import pandas as pd
import streamlit as st
from pandas import DataFrame, Timestamp

from analysis.general.data_utils import cached_load
from analysis.general.demographic_breakdown import render_breakdown_section
from analysis.general.equity_analysis import render_equity_analysis
from analysis.general.filter_utils import apply_filters, render_filter_form, show_date_range_warning
from analysis.general.length_of_stay import render_length_of_stay
from analysis.general.summary_metrics import render_summary_metrics
from analysis.general.trend_explorer import render_trend_explorer
from analysis.general.theme import blue_divider
#from analysis.general.client_journey import render_client_journey_analysis

def general_analysis_page() -> None:
    """
    Main entry point for the general analysis dashboard.
    
    This function orchestrates the overall flow of the dashboard,
    loading data, applying filters, and delegating to section-specific
    rendering functions.
    """    
    # Main title
    blue_divider()
    st.title("🏠 General Analysis Dashboard")
    blue_divider()

    # Check if the dataframe is available in session state
    df_state = st.session_state.get("df")
    if df_state is None or df_state.empty:
        st.warning("Upload your HMIS file in the sidebar first.")
        return

    # Load and preprocess data
    try:
        with st.spinner("Loading and preprocessing data..."):
            df = cached_load(df_state)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        df = df_state.copy()

    if df.empty:
        st.error("No rows after preprocessing – check your file format.")
        return

    # Initialize filters in session state if not already present
    st.session_state.setdefault("filters", {})
    
    # Render filter form in sidebar and check if filters were applied
    filters_applied = render_filter_form(df)

    # Guide user to apply filters if not done yet
    if not filters_applied and "t0" not in st.session_state:
        st.info("🔍 Set filters in the sidebar and click **Apply Filters** to begin analysis.")
        return

    # Apply filters and save filtered dataframe in session state
    df_filt = apply_filters(df)
    st.session_state["df_filt"] = df_filt  # Save for later use (download)

    # Check if date ranges are valid for the data
    show_date_range_warning(df)

    # Render each section of the dashboard
    render_summary_metrics(df_filt, df)
    blue_divider()

    render_breakdown_section(df_filt, df)
    blue_divider()

    render_trend_explorer(df_filt, df)
    blue_divider()

    render_length_of_stay(df_filt)
    blue_divider()

    render_equity_analysis(df_filt, df)
    blue_divider()

    # render_client_journey_analysis(df_filt, df)
    # blue_divider()

    # Data export option
    st.subheader("Data Export")
    st.markdown("Download your filtered dataset for further analysis.")

    # Retrieve the filtered data
    df_filt_cached = st.session_state.get("df_filt")

    if df_filt_cached is not None and not df_filt_cached.empty:
        st.download_button(
            "📥 Download filtered data (CSV)",
            data=df_filt_cached.to_csv(index=False).encode(),
            file_name="hmis_filtered.csv",
            mime="text/csv",
            use_container_width=True,
            help="Export the filtered dataset for use in other tools"
        )
    else:
        st.warning("No data available to download. Please apply filters first.")

# Run the app when script is executed directly
if __name__ == "__main__":
    general_analysis_page()
