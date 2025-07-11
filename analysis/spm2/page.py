"""
SPM2 Analysis Page
-----------------
Renders the SPM2 analysis interface and orchestrates the workflow.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date

from typing import Dict, List, Optional, Tuple, Any

from config.settings import DEFAULT_START_DATE, DEFAULT_END_DATE
from config.constants import EXIT_COLUMNS, RETURN_COLUMNS, DEFAULT_PROJECT_TYPES
from ui.templates import ABOUT_SPM2_CONTENT, render_about_section
from ui.components import render_dataframe_with_style, render_download_button
from core.session import check_data_available, set_analysis_result, get_analysis_result
from core.utils import create_multiselect_filter, check_date_range_validity

from analysis.spm2.analysis import (
    run_spm2,
    compute_summary_metrics,
    breakdown_by_columns
)
from analysis.spm2.visualizations import (
    display_spm_metrics,
    plot_days_to_return_box,
    create_flow_pivot,
    get_top_flows_from_pivot,
    plot_flow_sankey
)


def setup_date_config(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp, int, str, int]:
    """Configure date parameters for analysis."""
    with st.sidebar.expander("📅 Date Configuration", expanded=True):
        # Set default values
        default_start = datetime(2023, 10, 1)
        default_end = datetime(2024, 9, 30)
        
        date_range = st.date_input(
            "SPM Reporting Period",
            [default_start, default_end],
            help="Primary analysis window for SPM2 metrics"
        )
        st.caption("📌 **Note:** Primary analysis window for SPM2 metrics. The selected end date will be included in the analysis period.")
        
        # Handle different return types from date_input
        if date_range is None:
            # No dates selected
            st.warning("Please select both start and end dates for the reporting period.")
            report_start = pd.to_datetime(default_start)
            report_end = pd.to_datetime(default_end)
        elif isinstance(date_range, (list, tuple)):
            if len(date_range) == 2:
                # Both dates selected
                report_start = pd.to_datetime(date_range[0])
                report_end = pd.to_datetime(date_range[1])
            elif len(date_range) == 1:
                # Only one date selected
                st.warning("Please select an end date for the reporting period.")
                report_start = pd.to_datetime(date_range[0])
                report_end = pd.to_datetime(date_range[0])
            else:
                # Empty list/tuple
                st.warning("Please select both start and end dates for the reporting period.")
                report_start = pd.to_datetime(default_start)
                report_end = pd.to_datetime(default_end)
        elif isinstance(date_range, (datetime, date)):
            # Single date object (shouldn't happen with range input, but just in case)
            st.warning("Please select an end date for the reporting period.")
            report_start = report_end = pd.to_datetime(date_range)
        else:
            # Unexpected type
            st.warning("Please select both start and end dates for the reporting period.")
            report_start = pd.to_datetime(default_start)
            report_end = pd.to_datetime(default_end)
        
        # Ensure start is before end
        if report_start > report_end:
            st.error("Start date must be before end date. Dates have been swapped.")
            report_start, report_end = report_end, report_start
        
        st.divider()
        unit_choice = st.radio(
            "Select Lookback Unit",
            options=["Days", "Months"],
            index=0,
            help="Choose whether to specify the lookback period in days or months."
        )
        
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
        st.caption(f"Exit Window: {exit_window_start:%Y-%m-%d} to {exit_window_end:%Y-%m-%d}")
        st.divider()
        return_period = st.number_input(
            "Return Period (Days)",
            min_value=1,
            value=730,
            help="Max days post-exit to count as return"
        )
        
        # Check if analysis range is within available data range
        if df is not None and not df.empty:
            data_reporting_start = pd.to_datetime(df["ReportingPeriodStartDate"].iloc[0])
            data_reporting_end = pd.to_datetime(df["ReportingPeriodEndDate"].iloc[0])
            
            check_date_range_validity(
                exit_window_start, 
                report_end, 
                data_reporting_start, 
                data_reporting_end
            )
    
    return report_start, report_end, lookback_value, unit_choice, return_period


def setup_global_filters(df: pd.DataFrame) -> Optional[List[str]]:
    """Configure global filters for analysis."""
    with st.sidebar.expander("⚡ Global Filters", expanded=True):
        allowed_continuum = None
        if "ProgramsContinuumProject" in df.columns:
            unique_continuum = sorted(df["ProgramsContinuumProject"].dropna().unique().tolist())
            allowed_continuum = create_multiselect_filter(
                "Continuum Projects",
                unique_continuum,
                default=["Yes"],
                help_text="Filter by Continuum Project participation"
            )
    
    return allowed_continuum


def setup_exit_filters(df: pd.DataFrame) -> Tuple[Optional[List[str]], ...]:
    """Configure exit-specific filters."""
    with st.sidebar.expander("🚪 Exit Filters", expanded=True):
        st.markdown("#### Exit Enrollment Criteria")
        
        exit_allowed_cocs = create_multiselect_filter(
            "CoC Codes - Exit",
            sorted(df["ProgramSetupCoC"].dropna().unique().tolist()) if "ProgramSetupCoC" in df.columns else [],
            default=["ALL"],
            help_text="CoC codes for exit identification"
        )
        
        exit_allowed_local_cocs = create_multiselect_filter(
            "Local CoC - Exit",
            sorted(df["LocalCoCCode"].dropna().unique().tolist()) if "LocalCoCCode" in df.columns else [],
            default=["ALL"],
            help_text="Local CoC codes for exits"
        )
        
        exit_allowed_agencies = create_multiselect_filter(
            "Agencies - Exit",
            sorted(df["AgencyName"].dropna().unique().tolist()) if "AgencyName" in df.columns else [],
            default=["ALL"],
            help_text="Agencies for exit identification"
        )
        
        exit_allowed_programs = create_multiselect_filter(
            "Programs - Exit",
            sorted(df["ProgramName"].dropna().unique().tolist()) if "ProgramName" in df.columns else [],
            default=["ALL"],
            help_text="Programs for exit identification"
        )
        
        all_project_types = sorted(df["ProjectTypeCode"].dropna().unique().tolist())
        default_projects = [p for p in DEFAULT_PROJECT_TYPES if p in all_project_types]
        exiting_projects = st.multiselect(
            "Project Types (Exit)",
            all_project_types,
            default=default_projects,
            help="Project types treated as exits",
        )
        
        allowed_exit_dest_cats = None
        if "ExitDestinationCat" in df.columns:
            allowed_exit_dest_cats = st.multiselect(
                "Exit Destination Categories",
                sorted(df["ExitDestinationCat"].dropna().unique()),
                default=["Permanent Housing Situations"],
                help="Limit exits to these destination categories",
            )
    
    return exit_allowed_cocs, exit_allowed_local_cocs, exit_allowed_agencies, exit_allowed_programs, exiting_projects, allowed_exit_dest_cats


def setup_return_filters(df: pd.DataFrame) -> Tuple[Optional[List[str]], ...]:
    """Configure return-specific filters."""
    with st.sidebar.expander("↩️ Return Filters", expanded=True):
        st.markdown("#### Return Enrollment Criteria")
        
        return_allowed_cocs = create_multiselect_filter(
            "CoC Codes - Return",
            sorted(df["ProgramSetupCoC"].dropna().unique().tolist()) if "ProgramSetupCoC" in df.columns else [],
            default=["ALL"],
            help_text="CoC codes for return identification"
        )
        
        return_allowed_local_cocs = create_multiselect_filter(
            "Local CoC - Return",
            sorted(df["LocalCoCCode"].dropna().unique().tolist()) if "LocalCoCCode" in df.columns else [],
            default=["ALL"],
            help_text="Local CoC codes for returns"
        )
        
        return_allowed_agencies = create_multiselect_filter(
            "Agencies - Return",
            sorted(df["AgencyName"].dropna().unique().tolist()) if "AgencyName" in df.columns else [],
            default=["ALL"],
            help_text="Agencies for return identification"
        )
        
        return_allowed_programs = create_multiselect_filter(
            "Programs - Return",
            sorted(df["ProgramName"].dropna().unique().tolist()) if "ProgramName" in df.columns else [],
            default=["ALL"],
            help_text="Programs for return identification"
        )
        
        all_project_types = sorted(df["ProjectTypeCode"].dropna().unique().tolist())
        default_projects = [p for p in DEFAULT_PROJECT_TYPES if p in all_project_types]
        return_projects = st.multiselect(
            "Project Types (Return)",
            all_project_types,
            default=default_projects,
            help="Project types treated as candidate returns",
        )
    
    return return_allowed_cocs, return_allowed_local_cocs, return_allowed_agencies, return_allowed_programs, return_projects


def setup_comparison_filters() -> bool:
    """Configure settings for PH vs. Non-PH comparisons."""
    with st.sidebar.expander("PH vs. Non-PH", expanded=True):
        compare_ph_others = st.checkbox(
            "Compare PH/Non-PH Exits",
            value=False,
            help="Enable side-by-side PH comparison"
        )
    
    return compare_ph_others


def run_analysis(df: pd.DataFrame, analysis_params: Dict[str, Any]) -> bool:
    """Execute the SPM2 analysis with specified parameters."""
    if st.button("▶️ Run SPM2 Analysis", type="primary", use_container_width=True):
        with st.status("🔍 Processing...", expanded=True) as status:
            try:
                final_df = run_spm2(
                    df,
                    report_start=analysis_params["report_start"],
                    report_end=analysis_params["report_end"],
                    lookback_value=analysis_params["lookback_value"],
                    lookback_unit=analysis_params["unit_choice"],
                    exit_cocs=analysis_params["exit_allowed_cocs"],
                    exit_localcocs=analysis_params["exit_allowed_local_cocs"],
                    exit_agencies=analysis_params["exit_allowed_agencies"],
                    exit_programs=analysis_params["exit_allowed_programs"],
                    return_cocs=analysis_params["return_allowed_cocs"],
                    return_localcocs=analysis_params["return_allowed_local_cocs"],
                    return_agencies=analysis_params["return_allowed_agencies"],
                    return_programs=analysis_params["return_allowed_programs"],
                    allowed_continuum=analysis_params["allowed_continuum"],
                    allowed_exit_dest_cats=analysis_params["allowed_exit_dest_cats"],
                    exiting_projects=analysis_params["exiting_projects"],
                    return_projects=analysis_params["return_projects"],
                    return_period=analysis_params["return_period"]
                )
                # Clean up columns
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
                final_df.drop(columns=[c for c in cols_to_remove if c in final_df.columns], inplace=True)

                # Rename flattened exit columns to simpler names
                cols_to_rename = {
                    "Exit_UniqueIdentifier": "UniqueIdentifier",
                    "Exit_ClientID": "ClientID",
                    "Exit_RaceEthnicity": "RaceEthnicity",
                    "Exit_Gender": "Gender",
                    "Exit_DOB": "DOB",
                    "Exit_VeteranStatus": "VeteranStatus"
                }
                final_df.rename(columns={k: v for k, v in cols_to_rename.items() if k in final_df.columns}, inplace=True)

                
                # Store results in session state
                set_analysis_result("spm2", final_df)
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                st.toast("SPM2 analysis successful!", icon="🎉")
                
                return True
            except Exception as e:
                st.error(f"🚨 Analysis Error: {str(e)}")
                return False


def display_summary_metrics(
    final_df: pd.DataFrame,
    return_period: int,
    allowed_exit_dest_cats: Optional[List[str]] = None
) -> Dict[str, Any]:
    st.divider()
    st.markdown("### 📊 Returns to Homelessness Summary")

    # If only Permanent Housing Situations is selected, add the note
    if allowed_exit_dest_cats == ["Permanent Housing Situations"]:
        st.caption("📌 **Note:** only Permanent Housing Situations is selected in the Exit Destination Categories filter.")

    metrics = compute_summary_metrics(final_df, return_period)
    display_spm_metrics(metrics, return_period, show_total_exits=True)
    return metrics


def display_days_to_return(final_df: pd.DataFrame, return_period: int) -> None:
    """Display the days-to-return distribution visualization."""
    st.divider()
    with st.container():
        st.markdown("### ⏳ Days to Return Distribution")
        try:
            st.plotly_chart(plot_days_to_return_box(final_df, return_period), use_container_width=True)
        except Exception as e:
            st.error(f"📉 Visualization Error: {str(e)}")

@st.fragment
def display_breakdowns(final_df: pd.DataFrame, return_period: int) -> None:
    """Display cohort breakdown analysis."""
    st.divider()
    with st.container():
        st.markdown("### 📊 Breakdown")
        breakdown_columns = [
            "RaceEthnicity",
            "Gender",
            "VeteranStatus",
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
            "ReturnCategory",
            "AgeAtExitRange",
        ]

        # Build options from available columns
        breakdown_options = [col for col in breakdown_columns if col in final_df.columns]
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
                bdf = breakdown_by_columns(final_df, chosen_cols, return_period)
                with analysis_cols[1]:
                    st.metric("Total Groups", len(bdf))
                
                render_dataframe_with_style(
                    bdf,
                    highlight_cols=["Number of Relevant Exits", "Total Return"],
                    height=400
                )
            except Exception as e:
                st.error(f"📈 Breakdown Error: {str(e)}")

@st.fragment
def display_client_flow(final_df: pd.DataFrame) -> None:
    """Display client journey flow visualization."""
    st.divider()
    with st.container():
        st.markdown("### 🌊 Client Flow Analysis")
        try:
            # Filter columns for exit and return dimensions
            exit_cols = [col for col in EXIT_COLUMNS if col in final_df.columns]
            return_cols = [col for col in RETURN_COLUMNS if col in final_df.columns]
            
            if exit_cols and return_cols:
                # Dimension selectors
                st.caption("📌 **Note:** Both the Exit and Entry Dimension filters apply to the entire flow section, including Client Flow Analysis, Top Client Pathways, and Client Flow Network.")
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

                # Build pivot table (unfiltered for matrix and top pathways)
                pivot_c = create_flow_pivot(final_df, ex_choice, ret_choice)

                # Reorder columns to keep "No Return" last
                if "No Return" in pivot_c.columns:
                    cols_order = [c for c in pivot_c.columns if c != "No Return"] + ["No Return"]
                    pivot_c = pivot_c[cols_order]

                columns_to_color = [col for col in pivot_c.columns if col != "No Return"]
                with st.expander("🔍 Flow Matrix Details", expanded=True):
                    render_dataframe_with_style(
                        pivot_c, 
                        highlight_cols=columns_to_color,
                        axis=1
                    )

                # Top pathways (using unfiltered data)
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
                    render_dataframe_with_style(
                        top_flows_df,
                        highlight_cols=["Count", "Percent"] if "Percent" in top_flows_df.columns else ["Count"]
                    )
                else:
                    st.info("No significant pathways detected")

                # Sankey diagram section with focus controls
                st.markdown("#### 🌐 Client Flow Network")
                
                # Focus controls (only for network graph)
                st.caption("🎯 **Focus filters below apply only to the network visualization**")
                colL, colR = st.columns(2)
                focus_exit = colL.selectbox(
                    "🔍 Focus Exit Dimension",
                    ["All"] + pivot_c.index.tolist(),
                    help="Show only this exit in the network"
                )
                focus_return = colR.selectbox(
                    "🔍 Focus Return Dimension",
                    ["All"] + pivot_c.columns.tolist(),
                    help="Show only this return in the network"
                )

                # Create filtered pivot for Sankey only
                pivot_sankey = pivot_c.copy()
                
                # Apply focus filters only to the Sankey data
                if focus_exit != "All":
                    pivot_sankey = pivot_sankey.loc[[focus_exit]]
                if focus_return != "All":
                    pivot_sankey = pivot_sankey[[focus_return]]

                # Generate Sankey with filtered data
                sankey_fig = plot_flow_sankey(pivot_sankey, f"{ex_choice} → {ret_choice}")
                st.plotly_chart(sankey_fig, use_container_width=True)
            else:
                st.info("📭 Insufficient data for flow analysis")
        except Exception as e:
            st.error(f"🌊 Flow Analysis Error: {str(e)}")


def display_ph_comparison(final_df: pd.DataFrame, return_period: int) -> None:
    """Display PH vs. Non-PH exit comparison."""
    st.divider()
    with st.container():
        st.markdown("## 🔄 PH vs. Non-PH Exit Comparison")
        ph_df = final_df[final_df["PH_Exit"] == True]
        nonph_df = final_df[final_df["PH_Exit"] == False]
        
        comp_cols = st.columns(2)
        with comp_cols[0]:
            st.markdown("### 🏠 Permanent Housing Exits")
            if not ph_df.empty:
                ph_metrics = compute_summary_metrics(ph_df, return_period)
                st.metric("Number of Relevant Exits", ph_metrics.get("Number of Relevant Exits", 0))
                st.metric("<6 Month Returns", f"{ph_metrics.get('Return < 6 Months', 0)} ({ph_metrics.get('% Return < 6M', 0):.1f}%)")
                st.metric("6–12 Month Returns", f"{ph_metrics.get('Return 6–12 Months', 0)} ({ph_metrics.get('% Return 6–12M', 0):.1f}%)")
                st.metric("12–24 Month Returns", f"{ph_metrics.get('Return 12–24 Months', 0)} ({ph_metrics.get('% Return 12–24M', 0):.1f}%)")
                
                # Only show >24 months if return period is greater than 730 days
                if return_period > 730:
                    st.metric(">24 Month Returns", f"{ph_metrics.get('Return > 24 Months', 0)} ({ph_metrics.get('% Return > 24M', 0):.1f}%)")
                
                st.metric("Total Returns", f"{ph_metrics.get('Total Return', 0)} ({ph_metrics.get('% Return', 0):.1f}%)")
                st.metric("Median Return Days", f"{ph_metrics.get('Median Days (<=period)', 0):.0f}")
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
                st.metric("12–24 Month Returns", f"{nonph_metrics.get('Return 12–24 Months', 0)} ({nonph_metrics.get('% Return 12–24M', 0):.1f}%)")
                
                # Only show >24 months if return period is greater than 730 days
                if return_period > 730:
                    st.metric(">24 Month Returns", f"{nonph_metrics.get('Return > 24 Months', 0)} ({nonph_metrics.get('% Return > 24M', 0):.1f}%)")
                
                st.metric("Total Returns", f"{nonph_metrics.get('Total Return', 0)} ({nonph_metrics.get('% Return', 0):.1f}%)")
                st.metric("Median Return Days", f"{nonph_metrics.get('Median Days (<=period)', 0):.0f}")
                st.markdown(f"**Percentiles**: 25th: {nonph_metrics.get('DaysToReturn 25th Pctl', 0):.0f} | 75th: {nonph_metrics.get('DaysToReturn 75th Pctl', 0):.0f}")
            else:
                st.info("No Non-PH exits in current filters")


def display_data_export(final_df: pd.DataFrame) -> None:
    """Display data export options."""
    st.divider()
    with st.container():
        st.markdown("### 📤 Data Export")
        render_download_button(
            final_df,
            filename="spm2_analysis_results.csv",
            label="📥 Download SPM2 Data"
        )


def spm2_page() -> None:
    """Render the SPM2 Analysis page with all components."""
    st.header("📊 SPM2 Analysis")
    
    # About section
    render_about_section("About SPM2 Methodology", ABOUT_SPM2_CONTENT)

    # Check for data
    df = check_data_available()
    if df is None:
        return

    # Sidebar configuration
    st.sidebar.header("⚙️ Analysis Parameters", divider="gray")
    
    # Setup all filter parameters
    report_start, report_end, lookback_value, unit_choice, return_period = setup_date_config(df)
    allowed_continuum = setup_global_filters(df)
    exit_allowed_cocs, exit_allowed_local_cocs, exit_allowed_agencies, exit_allowed_programs, exiting_projects, allowed_exit_dest_cats = setup_exit_filters(df)
    return_allowed_cocs, return_allowed_local_cocs, return_allowed_agencies, return_allowed_programs, return_projects = setup_return_filters(df)
    compare_ph_others = setup_comparison_filters()
    
    # Prepare analysis parameters
    analysis_params = {
        "report_start": report_start,
        "report_end": report_end,
        "lookback_value": lookback_value,
        "unit_choice": unit_choice,
        "return_period": return_period,
        "allowed_continuum": allowed_continuum,
        "exit_allowed_cocs": exit_allowed_cocs,
        "exit_allowed_local_cocs": exit_allowed_local_cocs,
        "exit_allowed_agencies": exit_allowed_agencies,
        "exit_allowed_programs": exit_allowed_programs,
        "exiting_projects": exiting_projects,
        "allowed_exit_dest_cats": allowed_exit_dest_cats,
        "return_allowed_cocs": return_allowed_cocs,
        "return_allowed_local_cocs": return_allowed_local_cocs,
        "return_allowed_agencies": return_allowed_agencies,
        "return_allowed_programs": return_allowed_programs,
        "return_projects": return_projects
    }

    # Run analysis when button clicked
    st.divider()
    run_analysis(df, analysis_params)
    
    # Display results if analysis was successful
    final_df = get_analysis_result("spm2")
    if final_df is not None and not final_df.empty:
        # Display core metrics
        
        display_summary_metrics(final_df, return_period, allowed_exit_dest_cats)
        
        # Display days to return distribution
        display_days_to_return(final_df, return_period)
        
        # Display cohort breakdowns
        display_breakdowns(final_df, return_period)
        
        # Display client flow visualization
        display_client_flow(final_df)
        
        # Display PH vs Non-PH comparison if enabled
        if compare_ph_others:
            display_ph_comparison(final_df, return_period)
        
        # Display data export options
        display_data_export(final_df)


if __name__ == "__main__":
    spm2_page()