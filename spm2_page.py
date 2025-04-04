"""
SPM2 Analysis Page (Enhanced UI/UX)
----------------------------------
Modern interface with improved organization and visual hierarchy while preserving all original functionality.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from spm2_helpers import (
    run_spm2,
    compute_summary_metrics,
    display_spm_metrics,
    plot_days_to_return_box,
    breakdown_by_columns,
    create_flow_pivot,
    get_top_flows_from_pivot,
    plot_flow_sankey
)


def create_sidebar_multiselect(label: str, options: list, default: list, help_text: str):
    """
    Render a styled multiselect widget for the sidebar.
    
    Args:
        label (str): The label for the multiselect.
        options (list): List of options to choose from.
        default (list): Default selected options.
        help_text (str): Help message for the widget.
        
    Returns:
        list or None: The selected options or None if 'ALL' is selected.
    """
    try:
        with st.container():
            selection = st.multiselect(
                label=label,
                options=options,
                default=default,
                help=help_text
            )
            if not selection:
                st.error(f"Please select at least one option for {label}, or leave 'ALL'.")
                return None
            if "ALL" in selection:
                return None
            return selection
    except Exception as e:
        st.error(f"Filter error ({label}): {str(e)}")
        return None


def spm2_page():
    """
    Render the SPM2 Analysis page.
    
    This page allows users to configure filters, run the analysis, view performance metrics,
    visualize client flow, compare PH versus Non-PH exits, and export results.
    """
    st.header("📊 SPM2 Performance Analysis")
    
    # About section with methodology overview and interpretation guide
    with st.expander("📘 About SPM2 Methodology", expanded=False):
        st.markdown("""
            **SPM2 Analysis Process Overview**  
            *Measure housing stability through return-to-homelessness patterns*
            
            ### Key Features:
            - **Dual Filter Systems**: Separate filters for Exit and Return enrollments
            - **Temporal Analysis**: Customizable lookback and return periods
            - **Comparative Metrics**: PH vs Non-PH exit comparisons
            - **Flow Visualization**: Client movement analysis through Sankey diagrams
            
            ### Core Methodology:
            1. **Exit Identification**: Valid exits within lookback window
            2. **Return Detection**: First return within specified period
            3. **Classification**:
               - <6 Months (≤180d)
               - 6–12 Months (≤365d)
               - 12–24 Months (≤730d)
               - >24 Months (>730d)
            """)
        st.divider()
        st.markdown("""
            ### Interpretation Guide:
            - **Return Rates**: Percentage of exits with subsequent returns
            - **Time Distributions**: Box plots show days-to-return distribution
            - **Flow Analysis**: Paths between exit and return characteristics
            """)

    # Check that data is available
    df = st.session_state.get("df")
    if df is None or df.empty:
        st.info("📭 No data uploaded! Please upload data in the **Sidebar > Upload Data** section.")
        return

    # Sidebar: Analysis parameters
    st.sidebar.header("⚙️ Analysis Parameters", divider="gray")
    
    # Date configuration
    with st.sidebar.expander("📅 Date Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input(
                "Reporting Period",
                [datetime(2023, 10, 1), datetime(2024, 9, 30)],
                help="Primary analysis window for SPM2 metrics"
            )
            report_start, report_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        with col2:
            months_lookback = st.number_input(
                "Lookback Months",
                min_value=1,
                value=24,
                help="Months prior to report start for exit identification"
            )
            exit_window_start = report_start - pd.DateOffset(months=months_lookback)
            exit_window_end = report_end - pd.DateOffset(months=months_lookback)
            st.caption(f"Exit Window: {exit_window_start:%Y-%m-%d} to {exit_window_end:%Y-%m-%d}")

    # Global settings
    with st.sidebar.expander("⚡ Global Settings", expanded=True):
        return_period = st.number_input(
            "Return Period (Days)",
            min_value=1,
            value=730,
            help="Max days post-exit to count as return"
        )
        compare_ph_others = st.checkbox(
            "Compare PH/Non-PH Exits",
            value=False,
            help="Enable side-by-side PH comparison"
        )
        if "Programs Continuum Project" in df.columns:
            unique_continuum = sorted(df["Programs Continuum Project"].dropna().unique().tolist())
            allowed_continuum = create_sidebar_multiselect(
                "Continuum Projects",
                ["ALL"] + unique_continuum,
                default=["Yes"],
                help_text="Filter by Continuum Project participation"
            )

    # Exit filters
    with st.sidebar.expander("🚪 Exit Filters", expanded=True):
        st.markdown("#### Exit Enrollment Criteria")
        exit_cols = st.columns(2)
        with exit_cols[0]:
            exit_allowed_cocs = create_sidebar_multiselect(
                "CoC Codes - Exit",
                ["ALL"] + sorted(df["ProgramSetupCoC"].dropna().unique().tolist()) if "ProgramSetupCoC" in df.columns else [],
                default=["ALL"],
                help_text="CoC codes for exit identification"
            )
            exit_allowed_local_cocs = create_sidebar_multiselect(
                "Local CoC - Exit",
                ["ALL"] + sorted(df["LocalCoCCode"].dropna().unique().tolist()) if "LocalCoCCode" in df.columns else [],
                default=["ALL"],
                help_text="Local CoC codes for exits"
            )
        with exit_cols[1]:
            exit_allowed_agencies = create_sidebar_multiselect(
                "Agencies - Exit",
                ["ALL"] + sorted(df["AgencyName"].dropna().unique().tolist()) if "AgencyName" in df.columns else [],
                default=["ALL"],
                help_text="Agencies for exit identification"
            )
            exit_allowed_programs = create_sidebar_multiselect(
                "Programs - Exit",
                ["ALL"] + sorted(df["ProgramName"].dropna().unique().tolist()) if "ProgramName" in df.columns else [],
                default=["ALL"],
                help_text="Programs for exit identification"
            )
        if "ProjectTypeCode" in df.columns:
            exiting_projects = create_sidebar_multiselect(
                "Exit Project Types",
                sorted(df["ProjectTypeCode"].dropna().unique().tolist()) if "ProjectTypeCode" in df.columns else [],
                default=[
                    "Street Outreach", "Emergency Shelter – Entry Exit",
                    "Emergency Shelter – Night-by-Night", "Transitional Housing",
                    "Safe Haven", "PH – Housing Only",
                    "PH – Housing with Services (no disability required for entry)",
                    "PH – Permanent Supportive Housing (disability required for entry)",
                    "PH – Rapid Re-Housing"
                ],
                help_text="Project types considered valid exits"
            )
        if "ExitDestinationCat" in df.columns:
            allowed_exit_dest_cats = create_sidebar_multiselect(
                "Exit Destinations",
                ["ALL"] + sorted(df["ExitDestinationCat"].dropna().unique().tolist()),
                default=["Permanent Housing Situations"],
                help_text="Destination categories for exits"
            )

    # Return filters
    with st.sidebar.expander("↩️ Return Filters", expanded=True):
        st.markdown("#### Return Enrollment Criteria")
        return_cols = st.columns(2)
        with return_cols[0]:
            return_allowed_cocs = create_sidebar_multiselect(
                "CoC Codes - Return",
                ["ALL"] + sorted(df["ProgramSetupCoC"].dropna().unique().tolist()) if "ProgramSetupCoC" in df.columns else [],
                default=["ALL"],
                help_text="CoC codes for return identification"
            )
            return_allowed_local_cocs = create_sidebar_multiselect(
                "Local CoC - Return",
                ["ALL"] + sorted(df["LocalCoCCode"].dropna().unique().tolist()) if "LocalCoCCode" in df.columns else [],
                default=["ALL"],
                help_text="Local CoC codes for returns"
            )
        with return_cols[1]:
            return_allowed_agencies = create_sidebar_multiselect(
                "Agencies - Return",
                ["ALL"] + sorted(df["AgencyName"].dropna().unique().tolist()) if "AgencyName" in df.columns else [],
                default=["ALL"],
                help_text="Agencies for return identification"
            )
            return_allowed_programs = create_sidebar_multiselect(
                "Programs - Return",
                ["ALL"] + sorted(df["ProgramName"].dropna().unique().tolist()) if "ProgramName" in df.columns else [],
                default=["ALL"],
                help_text="Programs for return identification"
            )
        return_projects = create_sidebar_multiselect(
            "Return Project Types",
            sorted(df["ProjectTypeCode"].dropna().unique().tolist()) if "ProjectTypeCode" in df.columns else [],
            default=[
                "Street Outreach", "Emergency Shelter – Entry Exit",
                "Emergency Shelter – Night-by-Night", "Transitional Housing",
                "Safe Haven", "PH – Housing Only",
                "PH – Housing with Services (no disability required for entry)",
                "PH – Permanent Supportive Housing (disability required for entry)",
                "PH – Rapid Re-Housing"
            ],
            help_text="Project types considered valid returns"
        )

    # Main analysis execution
    st.divider()
    st.markdown("### 🚀 Execute Analysis")
    if st.button("▶️ Run SPM2 Analysis", type="primary", use_container_width=True):
        with st.status("🔍 Processing client patterns...", expanded=True) as status:
            try:
                final_df_custom = run_spm2(
                    df,
                    report_start=report_start,
                    report_end=report_end,
                    months_lookback=months_lookback,
                    exit_cocs=exit_allowed_cocs,
                    exit_localcocs=exit_allowed_local_cocs,
                    exit_agencies=exit_allowed_agencies,
                    exit_programs=exit_allowed_programs,
                    return_cocs=return_allowed_cocs,
                    return_localcocs=return_allowed_local_cocs,
                    return_agencies=return_allowed_agencies,
                    return_programs=return_allowed_programs,
                    allowed_continuum=allowed_continuum,
                    allowed_exit_dest_cats=allowed_exit_dest_cats,
                    exiting_projects=exiting_projects,
                    return_projects=return_projects,
                    return_period=return_period
                )
                st.session_state["final_df_custom"] = final_df_custom
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                st.toast("SPM2 analysis successful!", icon="🎉")
            except Exception as e:
                st.error(f"🚨 Critical Analysis Error: {str(e)}")

    # Display analysis results if available
    if "final_df_custom" in st.session_state and not st.session_state["final_df_custom"].empty:
        final_df_c = st.session_state["final_df_custom"]
        metrics = compute_summary_metrics(final_df_c, return_period)

        # Display core performance metrics
        st.divider()
        st.markdown("### 📈 Core Performance Metrics")
        display_spm_metrics(metrics, show_total_exits=True)

        # Days-to-return distribution
        st.divider()
        with st.container():
            st.markdown("### ⏳ Days to Return Distribution")
            try:
                st.plotly_chart(plot_days_to_return_box(final_df_c, return_period), use_container_width=True)
            except Exception as e:
                st.error(f"📉 Visualization Error: {str(e)}")

        # Cohort breakdown analysis
        st.divider()
        with st.container():
            st.markdown("### 📊 Cohort Breakdown")
            breakdown_options = list(final_df_c.columns)
            default_breakdown = ["Exit_ProjectTypeCode"] if "Exit_ProjectTypeCode" in breakdown_options else []
            
            analysis_cols = st.columns([3, 1])
            with analysis_cols[0]:
                chosen_cols = st.multiselect(
                    "Group By Dimensions",
                    breakdown_options,
                    default=default_breakdown,
                    help="Select up to 3 dimensions for cohort analysis"
                )
            if chosen_cols:
                try:
                    bdf = breakdown_by_columns(final_df_c, chosen_cols, return_period)
                    with analysis_cols[1]:
                        st.metric("Total Groups", len(bdf))
                    st.dataframe(
                        bdf.style.format(thousands=",")
                          .background_gradient(cmap="Blues", subset=["Total Exits", "Total Return"])
                          .set_properties(**{'text-align': 'left'}),
                        use_container_width=True,
                        height=400
                    )
                except Exception as e:
                    st.error(f"📈 Breakdown Error: {str(e)}")

        # Client journey analysis (flow visualization)
        st.divider()
        with st.container():
            st.markdown("### 🌊 Client Journey Analysis")
            try:
                exit_cols = [c for c in final_df_c.columns if c.startswith("Exit_")]
                return_cols = [c for c in final_df_c.columns if c.startswith("Return_")]
                
                if exit_cols and return_cols:
                    flow_cols = st.columns(2)
                    with flow_cols[0]:
                        ex_choice = st.selectbox(
                            "Source Dimension (Exit)",
                            exit_cols,
                            index=exit_cols.index("Exit_ProjectTypeCode") if "Exit_ProjectTypeCode" in exit_cols else 0,
                            help="Characteristic at exit point"
                        )
                    with flow_cols[1]:
                        ret_choice = st.selectbox(
                            "Target Dimension (Return)",
                            return_cols,
                            index=return_cols.index("Return_ProjectTypeCode") if "Return_ProjectTypeCode" in return_cols else 0,
                            help="Characteristic at return point"
                        )

                    # Create pivot table for flow visualization
                    pivot_c = create_flow_pivot(final_df_c, ex_choice, ret_choice)
                    
                    with st.expander("🔍 Flow Matrix Details", expanded=True):
                        st.dataframe(
                            pivot_c.style.background_gradient(cmap="Blues")
                                .format(precision=0),
                            use_container_width=True
                        )

                    st.markdown("#### 🏆 Top Client Pathways")
                    top_n = st.slider("Number of Pathways", 5, 25, 10)
                    top_flows_df = get_top_flows_from_pivot(pivot_c, top_n=top_n)
                    
                    if not top_flows_df.empty:
                        if "Percent" in top_flows_df.columns:
                            top_flows_df["Percent"] = top_flows_df["Percent"].astype(float)
                            styled_top_flows = top_flows_df.style.format({'Percent': '{:.1f}%'})\
                                .background_gradient(cmap="Blues", subset=["Count", "Percent"])
                            st.dataframe(styled_top_flows, use_container_width=True)
                        else:
                            st.dataframe(top_flows_df.style.background_gradient(cmap="Blues"), 
                                         use_container_width=True)
                    else:
                        st.info("No significant pathways detected")

                    st.markdown("#### 🌐 Client Flow Network")
                    sankey_fig = plot_flow_sankey(pivot_c, f"{ex_choice} → {ret_choice}")
                    st.plotly_chart(sankey_fig, use_container_width=True)
                else:
                    st.info("📭 Insufficient data for flow analysis")
            except Exception as e:
                st.error(f"🌊 Flow Analysis Error: {str(e)}")

        # PH vs. Non-PH exit comparison
        if compare_ph_others:
            st.divider()
            with st.container():
                st.markdown("## 🔄 PH vs. Non-PH Exit Comparison")
                ph_df = final_df_c[final_df_c["PH_Exit"] == True]
                nonph_df = final_df_c[final_df_c["PH_Exit"] == False]

                comp_cols = st.columns(2)
                with comp_cols[0]:
                    st.markdown("### 🏠 Permanent Housing Exits")
                    if not ph_df.empty:
                        ph_metrics = compute_summary_metrics(ph_df, return_period)
                        st.metric("Total Exits", ph_metrics.get("Total Exits", 0))
                        st.metric("<6 Month Returns", 
                                  f"{ph_metrics.get('Return < 6 Months', 0)} ({ph_metrics.get('% Return < 6M', 0):.1f}%)")
                        st.metric("6–12 Month Returns", 
                                  f"{ph_metrics.get('Return 6–12 Months', 0)} ({ph_metrics.get('% Return 6–12M', 0):.1f}%)")
                        st.metric("Median Return Days", ph_metrics.get('Median Days (<=period)', 0))
                        st.markdown(f"**Percentiles**: 25th: {ph_metrics.get('DaysToReturn 25th Pctl', 0):.0f} | "
                                    f"75th: {ph_metrics.get('DaysToReturn 75th Pctl', 0):.0f}")
                    else:
                        st.info("No PH exits in current filters")
                with comp_cols[1]:
                    st.markdown("### 🏕️ Non-Permanent Housing Exits")
                    if not nonph_df.empty:
                        nonph_metrics = compute_summary_metrics(nonph_df, return_period)
                        st.metric("Total Exits", nonph_metrics.get("Total Exits", 0))
                        st.metric("<6 Month Returns", 
                                  f"{nonph_metrics.get('Return < 6 Months', 0)} ({nonph_metrics.get('% Return < 6M', 0):.1f}%)")
                        st.metric("6–12 Month Returns", 
                                  f"{nonph_metrics.get('Return 6–12 Months', 0)} ({nonph_metrics.get('% Return 6–12M', 0):.1f}%)")
                        st.metric("Median Return Days", nonph_metrics.get('Median Days (<=period)', 0))
                        st.markdown(f"**Percentiles**: 25th: {nonph_metrics.get('DaysToReturn 25th Pctl', 0):.0f} | "
                                    f"75th: {nonph_metrics.get('DaysToReturn 75th Pctl', 0):.0f}")
                    else:
                        st.info("No Non-PH exits in current filters")

        # Data export section
        st.divider()
        with st.container():
            st.markdown("### 📤 Export Results")
            st.download_button(
                label="📥 Download Analysis Data",
                data=final_df_c.to_csv(index=False),
                file_name="spm2_analysis_results.csv",
                mime="text/csv",
                use_container_width=True
            )


if __name__ == "__main__":
    spm2_page()
