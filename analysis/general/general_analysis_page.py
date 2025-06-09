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
from ui.templates import ABOUT_GENERAL_ANALYSIS_CONTENT,render_about_section
def apply_custom_tab_style():
    """Apply custom CSS for tab styling that works with dark mode."""
    st.markdown("""
        <style>
        /* Main tabs container */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: rgba(0, 98, 155, 0.1);
            padding: 0;
            border-radius: 10px 10px 0 0;
            overflow: hidden;
        }
        
        /* Individual tab styling */
        .stTabs [data-baseweb="tab"] {
            height: 55px;
            padding: 0 30px;
            background-color: rgba(255, 255, 255, 0.05);
            border: none;
            border-radius: 0;
            color: rgba(255, 255, 255, 0.7);
            font-weight: 600;
            font-size: 16px;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            white-space: pre-wrap;
            flex-grow: 1;
            justify-content: center;
            position: relative;
        }
        
        /* Hover state */
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(0, 98, 155, 0.2);
            color: #00a3ff;
        }
        
        /* Active/selected tab */
        .stTabs [aria-selected="true"] {
            background-color: #00629b !important;
            color: white !important;
            box-shadow: inset 0 -3px 0 #00a3ff;
        }
        
        /* Tab panel content area */
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 20px;
            background-color: transparent;
        }
        
        /* Focus state for accessibility */
        .stTabs [data-baseweb="tab"]:focus {
            outline: 2px solid #00a3ff;
            outline-offset: -2px;
        }
        
        /* Tab divider effect */
        .stTabs [data-baseweb="tab"]:not(:last-child)::after {
            content: "";
            position: absolute;
            right: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 1px;
            height: 60%;
            background-color: rgba(255, 255, 255, 0.1);
        }
        
        /* Remove divider for active tab and its neighbor */
        .stTabs [aria-selected="true"]::after,
        .stTabs [aria-selected="true"] + [data-baseweb="tab"]::after {
            display: none;
        }
        
        /* Additional spacing for content */
        .main-content-area {
            padding-top: 10px;
        }
        
        /* Make tabs more prominent */
        .stTabs {
            margin-top: 20px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

def general_analysis_page() -> None:
    """
    Main entry point for the general analysis dashboard.
    
    This function orchestrates the overall flow of the dashboard,
    loading data, applying filters, and delegating to section-specific
    rendering functions.
    """    
    # Apply custom tab styling
    apply_custom_tab_style()
    
    # Main title
    blue_divider()
    st.title("🏠 General Analysis Dashboard")
    render_about_section("About General analysis Methodology", ABOUT_GENERAL_ANALYSIS_CONTENT)

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

    # Add spacing before tabs
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab_names = [
        "📊 Overview",
        "👥 Demographics", 
        "📈 Trends",
        "⏱️ Length of Stay",
        "⚖️ Equity Analysis",
        "💾 Export Data"
    ]
    
    tabs = st.tabs(tab_names)
    
    # Overview Tab
    with tabs[0]:
        st.markdown('<div class="main-content-area">', unsafe_allow_html=True)
        render_summary_metrics(df_filt, df)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Demographics Tab
    with tabs[1]:
        st.markdown('<div class="main-content-area">', unsafe_allow_html=True)
        render_breakdown_section(df_filt, df)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Trends Tab
    with tabs[2]:
        st.markdown('<div class="main-content-area">', unsafe_allow_html=True)
        render_trend_explorer(df_filt, df)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Length of Stay Tab
    with tabs[3]:
        st.markdown('<div class="main-content-area">', unsafe_allow_html=True)
        render_length_of_stay(df_filt)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Equity Analysis Tab
    with tabs[4]:
        st.markdown('<div class="main-content-area">', unsafe_allow_html=True)
        render_equity_analysis(df_filt, df)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Export Data Tab
    with tabs[5]:
        st.markdown('<div class="main-content-area">', unsafe_allow_html=True)
        st.subheader("Data Export")
        st.markdown("Download your filtered dataset for further analysis.")

        # Retrieve the filtered data
        df_filt_cached = st.session_state.get("df_filt")

        if df_filt_cached is not None and not df_filt_cached.empty:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    "📥 Download filtered data (CSV)",
                    data=df_filt_cached.to_csv(index=False).encode(),
                    file_name="hmis_filtered.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Export the filtered dataset for use in other tools"
                )
                
                # Show data preview
                st.markdown("### Data Preview")
                st.info(f"Filtered dataset contains {len(df_filt_cached):,} rows and {len(df_filt_cached.columns)} columns")
                
                # Show first few rows
                st.dataframe(
                    df_filt_cached.head(10),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("No data available to download. Please apply filters first.")
        st.markdown('</div>', unsafe_allow_html=True)

# Run the app when script is executed directly
if __name__ == "__main__":
    general_analysis_page()