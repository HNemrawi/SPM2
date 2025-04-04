"""
Inbound Recidivism Analysis Page (Enhanced UI/UX)
------------------------------------------------
This module renders a Streamlit page for analyzing inbound recidivism,
focusing on clients returning to homelessness programs after previous exits.
It offers dynamic date filtering, multi-dimensional filters, visual analytics,
and data export options.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Import helper functions for analysis and visualization
from inbound_helpers import (
    run_return_analysis,
    display_return_metrics_cards,
    plot_time_to_entry_box,
    return_breakdown_analysis,
    create_flow_pivot_ra,
    plot_flow_sankey_ra
)
from spm2_helpers import get_top_flows_from_pivot


def inbound_recidivism_page():
    """
    Render the Inbound Recidivism Analysis page.
    
    Users can configure filters, run the analysis, view visualizations (including
    a detailed flow matrix), and export results.
    """
    st.header("📈 Inbound Recidivism Analysis")
    
    # About section with overview and methodology
    with st.expander("📘 About Inbound Recidivism Analysis", expanded=False):
        st.markdown("""
            **Inbound Recidivism Analysis Overview**  
            *Analyze client returns to homelessness programs after previous exits.*
            
            ### Key Features:
            - **Dynamic Date Filtering**: Select analysis window and lookback period
            - **Multi-dimensional Filtering**: Filter by CoC codes, programs, and agencies
            - **Return Classification**:
              - 🆕 **New Clients**: First-time entries
              - 🔄 **Returning Clients**: Previous exits within lookback period
              - 🏠 **Returning from Housing**: Stable exits to permanent housing
            - **Visual Analytics**: Interactive charts, detailed flow matrix, and flow network diagrams
            """)
        st.divider()
        st.markdown("""
            ### Methodology Highlights:
            1. **Entry Identification**: New entries within selected date range
            2. **Exit Lookup**: Full client history scan for prior exits
            3. **Classification Logic**: Calculation based on days since exit
            """)

    # Verify that data has been uploaded
    try:
        df = st.session_state.get("df")
        if df is None or df.empty:
            st.info("📭 Please upload data in the sidebar first.")
            return
    except Exception as e:
        st.error(f"🚨 Data Error: {str(e)}")
        return

    # Sidebar: Analysis parameters and filters
    st.sidebar.header("⚙️ Analysis Parameters", divider="gray")
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        try:
            date_range = st.date_input(
                "🗓️ Entry Date Range",
                [datetime(2025, 1, 1), datetime(2025, 1, 31)],
                help="Analysis period for new entries"
            )
            if len(date_range) != 2:
                st.error("⚠️ Please select both dates")
                return
            report_start, report_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        except Exception as e:
            st.error(f"📅 Date Error: {str(e)}")
            return

    with col2:
        months_lookback = st.number_input(
            "🔍 Months Lookback",
            min_value=1,
            value=24,
            help="Months prior to entry to consider exits"
        )

    # Entry Filters: Updated to match naming style from SPM2 code
    with st.sidebar.expander("📍 Entry Filters", expanded=True):
        entry_cols = st.columns(2)
        allowed_cocs, allowed_localcocs, allowed_agencies, allowed_programs = None, None, None, None
        
        if "ProgramSetupCoC" in df.columns:
            with entry_cols[0]:
                all_cocs = sorted(df["ProgramSetupCoC"].dropna().unique().tolist())
                co_selection = st.multiselect(
                    "CoC Codes - Entry",
                    ["ALL"] + all_cocs,
                    default=["ALL"]
                )
                if co_selection and "ALL" not in co_selection:
                    allowed_cocs = co_selection

        if "LocalCoCCode" in df.columns:
            with entry_cols[1]:
                all_localcocs = sorted(df["LocalCoCCode"].dropna().unique().tolist())
                lc_selection = st.multiselect(
                    "Local CoC Codes - Entry",
                    ["ALL"] + all_localcocs,
                    default=["ALL"]
                )
                if lc_selection and "ALL" not in lc_selection:
                    allowed_localcocs = lc_selection

        agency_cols = st.columns(2)
        if "AgencyName" in df.columns:
            with agency_cols[0]:
                all_agencies = sorted(df["AgencyName"].dropna().unique().tolist())
                ag_selection = st.multiselect(
                    "Agencies - Entry",
                    ["ALL"] + all_agencies,
                    default=["ALL"]
                )
                if ag_selection and "ALL" not in ag_selection:
                    allowed_agencies = ag_selection

        if "ProgramName" in df.columns:
            with agency_cols[1]:
                all_programs = sorted(df["ProgramName"].dropna().unique().tolist())
                pr_selection = st.multiselect(
                    "Programs - Entry",
                    ["ALL"] + all_programs,
                    default=["ALL"]
                )
                if pr_selection and "ALL" not in pr_selection:
                    allowed_programs = pr_selection

    # Project Type Filters: Renamed for clarity
    with st.sidebar.expander("🚪 Project Type Filters", expanded=True):
        proj_type_cols = st.columns(2)
        entry_project_types, exit_project_types = None, None
        
        if "ProjectTypeCode" in df.columns:
            with proj_type_cols[0]:
                all_proj_types = sorted(df["ProjectTypeCode"].dropna().unique().tolist())
                entry_sel = st.multiselect(
                    "Entry Project Types",
                    ["ALL"] + all_proj_types,
                    default=["ALL"]
                )
                if entry_sel and "ALL" not in entry_sel:
                    entry_project_types = entry_sel

            with proj_type_cols[1]:
                exit_sel = st.multiselect(
                    "Exit Project Types",
                    ["ALL"] + all_proj_types,
                    default=["ALL"]
                )
                if exit_sel and "ALL" not in exit_sel:
                    exit_project_types = exit_sel

    # Run the analysis when the button is clicked
    st.divider()
    st.markdown("### 🚀 Analysis Execution")
    if st.button("▶️ Run Analysis", type="primary", use_container_width=True):
        try:
            with st.status("🔍 Processing...", expanded=True) as status:
                merged_df = run_return_analysis(
                    df,
                    report_start=report_start,
                    report_end=report_end,
                    months_lookback=months_lookback,
                    allowed_cocs=allowed_cocs,
                    allowed_localcocs=allowed_localcocs,
                    allowed_programs=allowed_programs,
                    allowed_agencies=allowed_agencies,
                    entry_project_types=entry_project_types,
                    exit_project_types=exit_project_types
                )
                st.session_state["return_df"] = merged_df
                status.update(label="✅ Analysis Complete!", state="complete")
            st.toast("Analysis completed successfully!", icon="🎉")
        except Exception as e:
            st.error(f"🚨 Analysis Error: {str(e)}")

    # Display analysis results if available
    if "return_df" in st.session_state and not st.session_state["return_df"].empty:
        final_ret_df = st.session_state["return_df"]
        
        st.divider()
        st.markdown("### 📊 Key Metrics")
        display_return_metrics_cards(final_ret_df)
        
        st.divider()
        st.markdown("### ⏳ Time to Entry Distribution")
        try:
            st.plotly_chart(plot_time_to_entry_box(final_ret_df), use_container_width=True)
        except Exception as e:
            st.error(f"📉 Visualization Error: {str(e)}")

        st.divider()
        st.markdown("### 📈 Demographic Breakdown")
        possible_cols = final_ret_df.columns.tolist()
        default_breakdown = ["Exit_ProjectTypeCode"] if "Exit_ProjectTypeCode" in possible_cols else []
        chosen = st.multiselect(
            "Group By",
            possible_cols,
            default=default_breakdown,
            help="Select grouping columns"
        )
        if chosen:
            try:
                breakdown = return_breakdown_analysis(final_ret_df, chosen)
                st.dataframe(
                    breakdown.style.format(thousands=",")
                        .background_gradient(cmap="Blues", subset=["Total Entries"])
                        .set_properties(**{'text-align': 'left'}),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"📊 Breakdown Error: {str(e)}")

        st.divider()
        st.markdown("### 🌊 Client Flow Analysis")
        try:
            ra_flows_df = final_ret_df[final_ret_df["ReturnCategory"].str.contains("Returning")] \
                if "ReturnCategory" in final_ret_df.columns else pd.DataFrame()

            exit_cols_for_flow = [c for c in ra_flows_df.columns if c.startswith("Exit_")]
            entry_cols_for_flow = [c for c in ra_flows_df.columns if c.startswith("Enter_")]
            
            if exit_cols_for_flow and entry_cols_for_flow:
                flow_cols = st.columns(2)
                with flow_cols[0]:
                    exit_flow_col = st.selectbox(
                        "Exit Dimension",
                        exit_cols_for_flow,
                        index=exit_cols_for_flow.index("Exit_ProjectTypeCode")
                        if "Exit_ProjectTypeCode" in exit_cols_for_flow else 0
                    )
                with flow_cols[1]:
                    entry_flow_col = st.selectbox(
                        "Entry Dimension",
                        entry_cols_for_flow,
                        index=entry_cols_for_flow.index("Enter_ProjectTypeCode")
                        if "Enter_ProjectTypeCode" in entry_cols_for_flow else 0
                    )

                flow_pivot_ra = create_flow_pivot_ra(ra_flows_df, exit_flow_col, entry_flow_col)
                
                # Flow Matrix Details expander
                with st.expander("🔍 Flow Matrix Details", expanded=True):
                    st.dataframe(
                        flow_pivot_ra.style.background_gradient(cmap="Blues")
                            .format(precision=0),
                        use_container_width=True
                    )
                
                st.markdown("#### 🏆 Top Client Pathways")
                top_n = st.slider("Number of Flows", 5, 25, 10)
                top_flows_df = get_top_flows_from_pivot(flow_pivot_ra, top_n=top_n)
                
                if not top_flows_df.empty:
                    if "Percent" in top_flows_df.columns:
                        top_flows_df["Percent"] = top_flows_df["Percent"].astype(float)
                        styled_top_flows = top_flows_df.style.format({'Percent': '{:.1f}%'})\
                            .background_gradient(cmap="Blues", subset=["Count", "Percent"])
                        st.dataframe(styled_top_flows, use_container_width=True)
                    else:
                        st.dataframe(top_flows_df, use_container_width=True)
                else:
                    st.info("No significant flows detected")

                st.markdown("#### 🌐 Flow Network")
                sankey_ra = plot_flow_sankey_ra(flow_pivot_ra, f"{exit_flow_col} → {entry_flow_col}")
                st.plotly_chart(sankey_ra, use_container_width=True)
            else:
                st.info("📭 Insufficient data for flow analysis")
        except Exception as e:
            st.error(f"🌊 Flow Error: {str(e)}")

        st.divider()
        st.markdown("### 📤 Data Export")
        st.download_button(
            label="📥 Download Results",
            data=final_ret_df.to_csv(index=False),
            file_name="recidivism_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    inbound_recidivism_page()
