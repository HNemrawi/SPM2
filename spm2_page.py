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
    st.header("📊 SPM2 Analysis")
    
    # About section with methodology overview and interpretation guide
    with st.expander("📘 About SPM2 Methodology", expanded=False):
        st.markdown("""
    ## 📘 About SPM2 Methodology

    **SPM2 Analysis Overview**  
    *Assessing housing stability by tracking if and when clients return to homeless services after exiting to permanent housing.*

    ---

    ### 1. Client Universe  
    - **Included Project Types**  
    - Street Outreach (SO)  
    - Emergency Shelter (ES)  
    - Transitional Housing (TH)  
    - Safe Haven (SH)  
    - Permanent Housing (PH)  
    - **Date Window**  
    - Consider exits that occurred **within 730 days before** the reporting period  
        - Exit Date ≥ (Report Start Date – 730 days)  
        - Exit Date ≤ (Report End Date – 730 days)

    ---

    ### 2. Identifying the “Permanent Housing” Exit  
    - **Definition:** the client’s **earliest** exit into any PH project  
    - **Tie‑Breaker:** if multiple exits share that date, pick the one with **lowest Enrollment ID**

    ---

    ### 3. Scanning for a Return to Homeless Services  
    1. **Search Window:** from the PH exit date up to the end of the reporting period  
    2. **Eligible Return Enrollments:**  
    - Any SO, ES, TH, or SH project  
    - PH project re‑entries that meet both:  
        - Start date **> 14 days** after the original PH exit  
        - No overlap with **another PH transition window**  
        - Transition window = Day 1 after PH start through `min(PH exit + 14 days, report end)`  
    3. **First Qualifying Return:**  
    - Project Start ≥ PH Exit Date  
    - Project Start ≤ Report End Date  
    - Stop at the very first enrollment that fulfills the above

    ---

    ### 4. Classifying Return Timing  
    | Category         | Days from Exit       |  
    |------------------|----------------------|  
    | **< 6 months**   | 0–180 days           |  
    | **6–12 months**  | 181–365 days         |  
    | **12–24 months** | 366–730 days         |  
    | **> 24 months**  | 731+ days            |

    ---

    ### 5. Interpretation Guide  
    - **Return Rate**  
    - Percentage of clients who re‑enroll in any homeless service after exiting to PH  
    - **Timing Distribution**  
    - How long (in days) it takes for clients to return, broken down by the categories above  
    - **Trajectory Flows**  
    - Sankey or flow diagrams showing client pathways from exit through return (if any)
        """, unsafe_allow_html=True)


    df = st.session_state.get("df")
    if df is None or df.empty:
        st.info("📭 No data uploaded! Please upload data in the **Sidebar > Upload Data** section.")
        return

    # Sidebar: Analysis Parameters
    st.sidebar.header("⚙️ Analysis Parameters", divider="gray")
    
    with st.sidebar.expander("📅 Date Configuration", expanded=True):
        date_range = st.date_input(
            "SPM Reporting Period",
            [datetime(2023, 10, 1), datetime(2024, 9, 30)],
            help="Primary analysis window for SPM2 metrics"
        )
        report_start, report_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        
        # New radio button for selecting lookback unit
        unit_choice = st.radio(
            "Select Lookback Unit",
            options=["Days", "Months"],
            index=0,
            help="Choose whether to specify the lookback period in days or months."
        )
        
        # Present number input based on the selection
        if unit_choice == "Days":
            lookback_value = st.number_input(
                "Lookback Days",
                min_value=1,
                value=730,
                help="Days prior to report start for exit identification"
            )
            exit_window_start = report_start - pd.Timedelta(days=lookback_value)
            exit_window_end = report_end - pd.Timedelta(days=lookback_value)
        else:
            lookback_value = st.number_input(
                "Lookback Months",
                min_value=1,
                value=24,
                help="Months prior to report start for exit identification"
            )
            exit_window_start = report_start - pd.DateOffset(months=lookback_value)
            exit_window_end = report_end - pd.DateOffset(months=lookback_value)
        
        return_period = st.number_input(
            "Return Period (Days)",
            min_value=1,
            value=730,
            help="Max days post-exit to count as return"
        )

        st.caption(f"Exit Window: {exit_window_start:%Y-%m-%d} to {exit_window_end:%Y-%m-%d}")
        
        # Assume these new columns exist and are constant across all rows.
        data_reporting_start = pd.to_datetime(df["ReportingPeriodStartDate"].iloc[0])
        data_reporting_end = pd.to_datetime(df["ReportingPeriodEndDate"].iloc[0])
        
        # Check if the analysis range [exit_window_start, report_end] is within the available range.
        if exit_window_start < data_reporting_start or report_end > data_reporting_end:
            st.warning(
                f"WARNING: Your selected analysis range ({exit_window_start:%Y-%m-%d} to {report_end:%Y-%m-%d}) "
                f"is outside the data's available range ({data_reporting_start:%Y-%m-%d} to {data_reporting_end:%Y-%m-%d}). "
                "This may result in missing data. Please adjust your Reporting Period or Lookback period accordingly."
            )

    # Global Settings
    with st.sidebar.expander("⚡ Global Filters", expanded=True):

        if "ProgramsContinuumProject" in df.columns:
            unique_continuum = sorted(df["ProgramsContinuumProject"].dropna().unique().tolist())
            allowed_continuum = create_sidebar_multiselect(
                "Continuum Projects",
                ["ALL"] + unique_continuum,
                default=["Yes"],
                help_text="Filter by Continuum Project participation"
            )

    # Exit Filters
    with st.sidebar.expander("🚪 Exit Filters", expanded=True):
        st.markdown("#### Exit Enrollment Criteria")
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

    # Return Filters
    with st.sidebar.expander("↩️ Return Filters", expanded=True):
        st.markdown("#### Return Enrollment Criteria")
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

        # Global Settings
    with st.sidebar.expander("PH vs. Non-PH", expanded=True):

        compare_ph_others = st.checkbox(
            "Compare PH/Non-PH Exits",
            value=False,
            help="Enable side-by-side PH comparison"
        )


    # Main Analysis Execution
    st.divider()
    #st.markdown("### 🚀 Execute Analysis")
    if st.button("▶️ Run SPM2 Analysis", type="primary", use_container_width=True):
        with st.status("🔍 Processing...", expanded=True) as status:
            try:
                final_df_custom = run_spm2(
                    df,
                    report_start=report_start,
                    report_end=report_end,
                    lookback_value=lookback_value,
                    lookback_unit=unit_choice,
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
                # Drop unwanted Return columns and rename only specific Exit columns
                cols_to_remove = [
                    "Return_UniqueIdentifier",
                    "Return_ClientID",
                    "Return_RaceEthnicity",
                    "Return_Gender",
                    "Return_DOB",
                    "Return_VeteranStatus",
                    "Exit_ReportingPeriodStartDate",
                    "Exit_ReportingPeriodEndDate",
                    "Return_ReportingPeriodStartDate",
                    "Return_ReportingPeriodEndDate"
                ]
                final_df_custom.drop(columns=cols_to_remove, inplace=True, errors='ignore')
                cols_to_rename = [
                    "Exit_UniqueIdentifier", 
                    "Exit_ClientID",
                    "Exit_RaceEthnicity",
                    "Exit_Gender",
                    "Exit_DOB",
                    "Exit_VeteranStatus"
                ]
                mapping = {col: col[len("Exit_"):] for col in cols_to_rename if col in final_df_custom.columns}
                final_df_custom.rename(columns=mapping, inplace=True)
                
                st.session_state["final_df_custom"] = final_df_custom
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                st.toast("SPM2 analysis successful!", icon="🎉")
            except Exception as e:
                st.error(f"🚨 Analysis Error: {str(e)}")

    # Display Analysis Results if Available
    if "final_df_custom" in st.session_state and not st.session_state["final_df_custom"].empty:
        final_df_c = st.session_state["final_df_custom"]
        metrics = compute_summary_metrics(final_df_c, return_period)

        # Display Core Performance Metrics
        st.divider()
        st.markdown("### 📊 Returns to Homelessness Summary")
        display_spm_metrics(metrics, return_period, show_total_exits=True)

        # Days-to-Return Distribution
        st.divider()
        with st.container():
            st.markdown("### ⏳ Days to Return Distribution")
            try:
                st.plotly_chart(plot_days_to_return_box(final_df_c, return_period), use_container_width=True)
            except Exception as e:
                st.error(f"📉 Visualization Error: {str(e)}")

        # Cohort Breakdown Analysis
        st.divider()
        with st.container():
            st.markdown("### 📊 Breakdown")
            breakdown_columns = [
                "RaceEthnicity",
                "Gender",
                "VeteranStatus"
                "Exit_HasIncome",
                "Exit_HasDisability",
                "Exit_HouseholdType",
                "Exit_CHStartHousehold",
                "Exit_LocalCoCCode",
                "Exit_PriorLivingCat",
                "Exit_ProgramSetupCoC",
                "Exit_ProjectTypeCode",
                "Exit_AgencyName",
                "Exit_ProgramName",
                "Exit_ExitDestinationCat",
                "Exit_ExitDestination",
                "Exit_CustomProgramType",
                "Exit_AgeTieratEntry",
                "Return_HasIncome",
                "Return_HasDisability",
                "Return_HouseholdType",
                "Return_CHStartHousehold",
                "Return_LocalCoCCode",
                "Return_PriorLivingCat",
                "Return_ProgramSetupCoC",
                "Return_ProjectTypeCode",
                "Return_AgencyName",
                "Return_ProgramName",
                "Return_ExitDestinationCat",
                "Return_ExitDestination",
                "DaysToReturn",
                "ReturnCategory",
                "Return_AgeTieratEntry"
                "AgeAtExitRange",
            ]

            # Build the breakdown options by including only columns that exist.
            breakdown_options = [col for col in breakdown_columns if col in final_df_c.columns]
            default_breakdown = ["Exit_CustomProgramType"] if "Exit_CustomProgramType" in breakdown_options else []
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
                          .background_gradient(cmap="Blues", subset=["Number of Relevant Exits", "Total Return"])
                          .set_properties(**{'text-align': 'left'}),
                        use_container_width=True,
                        height=400
                    )
                except Exception as e:
                    st.error(f"📈 Breakdown Error: {str(e)}")

        # Client Journey Analysis (Flow Visualization)
        # Client Journey Analysis (Flow Visualization)
        st.divider()
        with st.container():
            st.markdown("### 🌊 Client Flow Analysis")
            try:
                exit_columns = [
                    "Exit_HasIncome",
                    "Exit_HasDisability",
                    "Exit_HouseholdType",
                    "Exit_CHStartHousehold",
                    "Exit_LocalCoCCode",
                    "Exit_PriorLivingCat",
                    "Exit_ProgramSetupCoC",
                    "Exit_ProjectTypeCode",
                    "Exit_AgencyName",
                    "Exit_ProgramName",
                    "Exit_ExitDestinationCat",
                    "Exit_ExitDestination",
                    "Exit_CustomProgramType",
                    "Exit_AgeTieratEntry"
                ]
                return_columns = [
                    "Return_HasIncome",
                    "Return_HasDisability",
                    "Return_HouseholdType",
                    "Return_CHStartHousehold",
                    "Return_LocalCoCCode",
                    "Return_PriorLivingCat",
                    "Return_ProgramSetupCoC",
                    "Return_ProjectTypeCode",
                    "Return_AgencyName",
                    "Return_ProgramName",
                    "Return_ExitDestinationCat",
                    "Return_ExitDestination",
                    "ReturnCategory",
                    "Return_AgeTieratEntry"
                ]
                exit_cols = [col for col in exit_columns if col in final_df_c.columns]
                return_cols = [col for col in return_columns if col in final_df_c.columns]
                if exit_cols and return_cols:
                    # original dimension selectors
                    flow_cols = st.columns(2)
                    with flow_cols[0]:
                        ex_choice = st.selectbox(
                            "Exit Dimension: Rows",
                            exit_cols,
                            index=exit_cols.index("Exit_ProjectTypeCode") if "Exit_ProjectTypeCode" in exit_cols else 0,
                            help="Characteristic at exit point"
                        )
                    with flow_cols[1]:
                        ret_choice = st.selectbox(
                            "Entry Dimension: Columns",
                            return_cols,
                            index=return_cols.index("Return_ProjectTypeCode") if "Return_ProjectTypeCode" in return_cols else 0,
                            help="Characteristic at return point"
                        )

                    # build full pivot
                    pivot_c = create_flow_pivot(final_df_c, ex_choice, ret_choice)

                    # Drill-in controls (focus on one exit or return node)
                    colL, colR = st.columns(2)
                    focus_exit = colL.selectbox(
                        "🔍 Focus Exit Dimension",
                        ["All"] + pivot_c.index.tolist(),
                        help="Show only this exit in the flow"
                    )
                    focus_return = colR.selectbox(
                        "🔍 Focus Return Dimension",
                        ["All"] + pivot_c.columns.tolist(),
                        help="Show only this return in the flow"
                    )

                    # Subset pivot_c in place
                    if focus_exit != "All":
                        pivot_c = pivot_c.loc[[focus_exit]]
                    if focus_return != "All":
                        pivot_c = pivot_c[[focus_return]]

                    # Ensure “No Return” column is last
                    if "No Return" in pivot_c.columns:
                        cols_order = [c for c in pivot_c.columns if c != "No Return"] + ["No Return"]
                        pivot_c = pivot_c[cols_order]

                    columns_to_color = [col for col in pivot_c.columns if col != "No Return"]
                    with st.expander("🔍 Flow Matrix Details", expanded=True):
                        st.dataframe(
                            pivot_c.style.background_gradient(cmap="Blues", subset=columns_to_color, axis=1)
                                .format(precision=0),
                            use_container_width=True
                        )

                    st.markdown("#### 🔝 Top Client Pathways")
                    top_n = st.slider(
                        "Number of Pathways",
                        min_value=5,
                        max_value=25,
                        value=5,
                        step=1,
                        help="Top N pathways to display"
                    )
                    top_flows_df = get_top_flows_from_pivot(pivot_c, top_n=top_n)
                    if not top_flows_df.empty:
                        if "Percent" in top_flows_df.columns:
                            top_flows_df["Percent"] = top_flows_df["Percent"].astype(float)
                            styled_top_flows = top_flows_df.style.format({'Percent': '{:.1f}%'}).background_gradient(cmap="Blues", subset=["Count", "Percent"])
                            st.dataframe(styled_top_flows, use_container_width=True)
                        else:
                            st.dataframe(top_flows_df.style.background_gradient(cmap="Blues"), use_container_width=True)
                    else:
                        st.info("No significant pathways detected")

                    st.markdown("#### 🌐 Client Flow Network")
                    sankey_fig = plot_flow_sankey(pivot_c, f"{ex_choice} → {ret_choice}")
                    st.plotly_chart(sankey_fig, use_container_width=True)
                else:
                    st.info("📭 Insufficient data for flow analysis")
            except Exception as e:
                st.error(f"🌊 Flow Analysis Error: {str(e)}")


        # PH vs. Non-PH Exit Comparison
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
                        st.metric("Number of Relevant Exits", ph_metrics.get("Number of Relevant Exits", 0))
                        st.metric("<6 Month Returns", f"{ph_metrics.get('Return < 6 Months', 0)} ({ph_metrics.get('% Return < 6M', 0):.1f}%)")
                        st.metric("6–12 Month Returns", f"{ph_metrics.get('Return 6–12 Months', 0)} ({ph_metrics.get('% Return 6–12M', 0):.1f}%)")
                        st.metric("Median Return Days", ph_metrics.get('Median Days (<=period)', 0))
                        st.markdown(f"**Percentiles**: 25th: {ph_metrics.get('DaysToReturn 25th Pctl', 0):.0f} | 75th: {ph_metrics.get('DaysToReturn 75th Pctl', 0):.0f}")
                    else:
                        st.info("No PH exits in current filters")
                with comp_cols[1]:
                    st.markdown("### 🏕️ Non-Permanent Housing Exits")
                    if not nonph_df.empty:
                        nonph_metrics = compute_summary_metrics(nonph_df, return_period)
                        st.metric("Number of Relevant Exits", nonph_metrics.get("Number of Relevant Exits", 0))
                        st.metric("<6 Month Returns", f"{nonph_metrics.get('Return < 6 Months', 0)} ({nonph_metrics.get('% Return < 6M', 0):.1f}%)")
                        st.metric("6–12 Month Returns", f"{nonph_metrics.get('Return 6–12 Months', 0)} ({nonph_metrics.get('% Return 6–12M', 0):.1f}%)")
                        st.metric("Median Return Days", nonph_metrics.get('Median Days (<=period)', 0))
                        st.markdown(f"**Percentiles**: 25th: {nonph_metrics.get('DaysToReturn 25th Pctl', 0):.0f} | 75th: {nonph_metrics.get('DaysToReturn 75th Pctl', 0):.0f}")
                    else:
                        st.info("No Non-PH exits in current filters")

        # Data Export Section
        st.divider()
        with st.container():
            st.markdown("### 📤 Data Export")
            st.download_button(
                label="📥 Download SPM2 Data",
                data=final_df_c.to_csv(index=False),
                file_name="spm2_analysis_results.csv",
                mime="text/csv",
                use_container_width=True
            )


if __name__ == "__main__":
    spm2_page()
