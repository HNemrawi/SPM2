"""
Inbound Recidivism Analysis Page
--------------------------------
Renders the inbound recidivism analysis interface and orchestrates the workflow.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from config.settings import DEFAULT_START_DATE, DEFAULT_END_DATE
from config.constants import EXIT_COLUMNS, RETURN_COLUMNS, DEFAULT_PROJECT_TYPES, PH_CATEGORY
from ui.templates import ABOUT_INBOUND_CONTENT, render_about_section
from ui.components import render_dataframe_with_style, render_download_button
from core.session import check_data_available, set_analysis_result, get_analysis_result
from core.utils import create_multiselect_filter, check_date_range_validity

from analysis.inbound.analysis import (
    run_return_analysis,
    compute_return_metrics,
    return_breakdown_analysis
)
from analysis.inbound.visualizations import (
    display_return_metrics_cards,
    plot_time_to_entry_box,
    create_flow_pivot_ra,
    plot_flow_sankey_ra,
    get_top_flows_from_pivot
)


def setup_date_config(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
    """Configure date parameters and lookback period."""
    with st.sidebar.expander("🗓️ Entry Date & Lookback", expanded=True):
        try:
            date_range = st.date_input(
                "Entry Date Range",
                [datetime(2025, 1, 1), datetime(2025, 1, 31)],
                help="Analysis period for new entries"
            )
            st.caption("📌 **Note:** The selected end date will be included in the analysis period.")

            if len(date_range) != 2:
                st.error("⚠️ Please select both dates")
                return None, None, None
            report_start, report_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        except Exception as e:
            st.error(f"📅 Date Error: {str(e)}")
            return None, None, None

        days_lookback = st.number_input(
            "🔍 Days Lookback",
            min_value=1,
            value=730,
            help="Number of days prior to entry to consider exits"
        )
        analysis_start = report_start - pd.Timedelta(days=days_lookback)
        analysis_end = report_end
        
        if df is not None and not df.empty:
            data_reporting_start = pd.to_datetime(df["ReportingPeriodStartDate"].iloc[0])
            data_reporting_end = pd.to_datetime(df["ReportingPeriodEndDate"].iloc[0])
            
            check_date_range_validity(
                analysis_start, 
                analysis_end, 
                data_reporting_start, 
                data_reporting_end
            )
        
        return report_start, report_end, days_lookback


def setup_entry_filters(df: pd.DataFrame) -> Tuple[Optional[List[str]], ...]:
    """Configure entry-specific filters."""
    with st.sidebar.expander("📍 Entry Filters", expanded=True):
        allowed_cocs = create_multiselect_filter(
            "CoC Codes - Entry",
            df["ProgramSetupCoC"].dropna().unique().tolist() if "ProgramSetupCoC" in df.columns else [],
            default=["ALL"],
            help_text="Filter entries by CoC code"
        )

        allowed_localcocs = create_multiselect_filter(
            "Local CoC Codes - Entry",
            df["LocalCoCCode"].dropna().unique().tolist() if "LocalCoCCode" in df.columns else [],
            default=["ALL"],
            help_text="Filter entries by local CoC code"
        )

        allowed_agencies = create_multiselect_filter(
            "Agencies - Entry",
            df["AgencyName"].dropna().unique().tolist() if "AgencyName" in df.columns else [],
            default=["ALL"],
            help_text="Filter entries by agency"
        )

        allowed_programs = create_multiselect_filter(
            "Programs - Entry",
            df["ProgramName"].dropna().unique().tolist() if "ProgramName" in df.columns else [],
            default=["ALL"],
            help_text="Filter entries by program"
        )

        # Entry Project Types filter
        entry_project_types = None
        if "ProjectTypeCode" in df.columns:
            all_project_types = sorted(df["ProjectTypeCode"].dropna().unique().tolist())
            default_projects = [p for p in DEFAULT_PROJECT_TYPES if p in all_project_types]
            entry_project_types = create_multiselect_filter(
                "Project Types - Entry",
                all_project_types,
                default=default_projects,
                help_text="Filter by project types for entries"
            )

        return allowed_cocs, allowed_localcocs, allowed_agencies, allowed_programs, entry_project_types


def setup_exit_filters(df: pd.DataFrame) -> Tuple[Optional[List[str]], ...]:
    """Configure exit-specific filters."""
    with st.sidebar.expander("🚪 Exit Filters", expanded=True):
        allowed_cocs_exit = create_multiselect_filter(
            "CoC Codes - Exit",
            df["ProgramSetupCoC"].dropna().unique().tolist() if "ProgramSetupCoC" in df.columns else [],
            default=["ALL"],
            help_text="Filter exits by CoC code"
        )

        allowed_localcocs_exit = create_multiselect_filter(
            "Local CoC Codes - Exit",
            df["LocalCoCCode"].dropna().unique().tolist() if "LocalCoCCode" in df.columns else [],
            default=["ALL"],
            help_text="Filter exits by local CoC code"
        )

        allowed_agencies_exit = create_multiselect_filter(
            "Agencies - Exit",
            df["AgencyName"].dropna().unique().tolist() if "AgencyName" in df.columns else [],
            default=["ALL"],
            help_text="Filter exits by agency"
        )

        allowed_programs_exit = create_multiselect_filter(
            "Programs - Exit",
            df["ProgramName"].dropna().unique().tolist() if "ProgramName" in df.columns else [],
            default=["ALL"],
            help_text="Filter exits by program"
        )

        # Project Types
        exit_project_types = None
        if "ProjectTypeCode" in df.columns:
            all_project_types = sorted(df["ProjectTypeCode"].dropna().unique().tolist())
            default_projects = [p for p in DEFAULT_PROJECT_TYPES if p in all_project_types]
            exit_project_types = create_multiselect_filter(
                "Project Types - Exit",
                all_project_types,
                default=default_projects,
                help_text="Filter by project types for exits"
            )

        # Exit Destination Category filter
        allowed_exit_dest_cats = None
        if "ExitDestinationCat" in df.columns:
            allowed_exit_dest_cats = create_multiselect_filter(
                "Exit Destination Categories",
                sorted(df["ExitDestinationCat"].dropna().unique().tolist()),
                default=["ALL"],
                help_text="Filter exits by destination category (e.g., Permanent Housing Situations)"
            )

        return allowed_cocs_exit, allowed_localcocs_exit, allowed_agencies_exit, allowed_programs_exit, exit_project_types, allowed_exit_dest_cats


def run_analysis(df: pd.DataFrame, analysis_params: Dict[str, Any]) -> bool:
    """Execute the inbound recidivism analysis with specified parameters."""
    try:
        with st.status("🔍 Processing...", expanded=True) as status:
            merged_df = run_return_analysis(
                df,
                report_start=analysis_params["report_start"],
                report_end=analysis_params["report_end"],
                days_lookback=analysis_params["days_lookback"],
                allowed_cocs=analysis_params["allowed_cocs"],
                allowed_localcocs=analysis_params["allowed_localcocs"],
                allowed_programs=analysis_params["allowed_programs"],
                allowed_agencies=analysis_params["allowed_agencies"],
                entry_project_types=analysis_params["entry_project_types"],
                allowed_cocs_exit=analysis_params["allowed_cocs_exit"],
                allowed_localcocs_exit=analysis_params["allowed_localcocs_exit"],
                allowed_programs_exit=analysis_params["allowed_programs_exit"],
                allowed_agencies_exit=analysis_params["allowed_agencies_exit"],
                exit_project_types=analysis_params["exit_project_types"],
                allowed_exit_dest_cats=analysis_params["allowed_exit_dest_cats"]  # ADD THIS
            )

            cols_to_remove = [
                "Exit_UniqueIdentifier",    
                "Exit_ClientID",
                "Exit_RaceEthnicity",    
                "Exit_Gender",    
                "Exit_DOB",    
                "Exit_VeteranStatus",
                "Enter_ReportingPeriodStartDate",
                "Enter_ReportingPeriodEndDate",
                "Exit_ReportingPeriodStartDate",
                "Exit_ReportingPeriodEndDate"
            ]
            merged_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')

            cols_to_rename = [
                "Enter_UniqueIdentifier", 
                "Enter_ClientID",
                "Enter_RaceEthnicity",
                "Enter_Gender",
                "Enter_DOB",
                "Enter_VeteranStatus"
            ]
            mapping = {col: col[len("Enter_"):] for col in cols_to_rename if col in merged_df.columns}
            merged_df.rename(columns=mapping, inplace=True)
            
            set_analysis_result("inbound", merged_df)
            status.update(label="✅ Analysis Complete!", state="complete")
        st.toast("Analysis completed successfully!", icon="🎉")
        return True
    except Exception as e:
        st.error(f"🚨 Analysis Error: {str(e)}")
        return False


def display_summary_metrics(final_df: pd.DataFrame) -> None:
    """Display core performance metrics."""
    st.divider()
    st.markdown("### 📊 Inbound Analysis Summary")
    metrics = compute_return_metrics(final_df)
    display_return_metrics_cards(metrics)


def display_time_to_entry(final_df: pd.DataFrame) -> None:
    """Display time-to-entry distribution visualization."""
    st.divider()
    st.markdown("### ⏳ Days to Return Distribution")
    try:
        st.plotly_chart(plot_time_to_entry_box(final_df), use_container_width=True)
    except Exception as e:
        st.error(f"📉 Visualization Error: {str(e)}")

@st.fragment
def display_breakdowns(final_df: pd.DataFrame) -> None:
    """Display cohort breakdown analysis."""
    st.divider()
    st.markdown("### 📈 Breakdown")
    
    # Add the collapsible description
    with st.expander("Understanding Breakdown Categories", expanded=False):
        st.markdown("""
        This table can be grouped by:
        
        - **Client-level values** (race, gender, etc.): these are demographic factors that are stable for clients across their enrollments.
        
        - **Entry-based values**: clients are categorized based on information relating to their Inbound Entry.
        
        - **Exit-based values**: clients are categorized based on information relating to most recent exit prior to their Entry during the reporting period. Because we are grouping by values related to their previous exit, clients who did NOT have a previous exit (new clients) will be excluded from the table.
        """)
    
    breakdown_columns = [
        "RaceEthnicity",
        "Gender",
        "VeteranStatus",
        "Enter_HasIncome",
        "Enter_HasDisability",
        "Enter_HouseholdType",
        "Enter_CHStartHousehold",
        "Enter_LocalCoCCode",
        "Enter_PriorLivingCat",
        "Enter_ProgramSetupCoC",
        "Enter_ProjectTypeCode",
        "Enter_AgencyName",
        "Enter_ProgramName",
        "Enter_ExitDestinationCat",
        "Enter_ExitDestination",
        "Enter_ProgramsContinuumProject",
        "Enter_AgeTieratEntry",
        "ReturnCategory",
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
        "Exit_ProgramsContinuumProject",
        "Exit_AgeTieratEntry",
    ]

    possible_cols = [col for col in breakdown_columns if col in final_df.columns]
    
    # Change default to Enter_ProjectTypeCode if it exists
    default_breakdown = ["Enter_ProjectTypeCode"] if "Enter_ProjectTypeCode" in possible_cols else []
    
    chosen = st.multiselect(
        "Group By",
        possible_cols,
        default=default_breakdown,
        help="Select grouping columns"
    )
    
    if chosen:
        try:
            breakdown = return_breakdown_analysis(final_df, chosen)

            # If any chosen column starts with "exit_", drop the "New (%)" column
            if any(col.lower().startswith("exit_") for col in chosen):
                breakdown = breakdown.drop(columns=["New (%)"], errors="ignore")

            render_dataframe_with_style(
                breakdown,
                highlight_cols=["Total Entries"]
            )
        except Exception as e:
            st.error(f"📊 Breakdown Error: {str(e)}")

@st.fragment
def display_client_flow(final_df: pd.DataFrame) -> None:
    """Display client flow analysis visualization."""
    st.divider()
    st.markdown("### 🌊 Client Flow Analysis")
    try:
        ra_flows_df = (
            final_df[final_df["ReturnCategory"].str.contains("Returning")]
            if "ReturnCategory" in final_df.columns
            else pd.DataFrame()
        )

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
            "Exit_ProgramsContinuumProject",
            "Exit_AgeTieratEntry",
        ]
        entry_columns = [
            "Enter_HasIncome",
            "Enter_HasDisability",
            "Enter_HouseholdType",
            "Enter_CHStartHousehold",
            "Enter_LocalCoCCode",
            "Enter_PriorLivingCat",
            "Enter_ProgramSetupCoC",
            "Enter_ProjectTypeCode",
            "Enter_AgencyName",
            "Enter_ProgramName",
            "Enter_ExitDestinationCat",
            "Enter_ExitDestination",
            "Enter_ProgramsContinuumProject",
            "Enter_AgeTieratEntry",
        ]

        exit_cols_for_flow = [c for c in exit_columns if c in ra_flows_df.columns]
        entry_cols_for_flow = [c for c in entry_columns if c in ra_flows_df.columns]

        if exit_cols_for_flow and entry_cols_for_flow:
            # Pick your two dimensions
            st.caption("📌 **Note:** Both the Exit and Entry Dimension filters apply to the entire flow section, including Client Flow Analysis, Top Client Pathways, and Client Flow Network.")
            flow_cols = st.columns(2)
            with flow_cols[0]:
                exit_flow_col = st.selectbox(
                    "Exit Dimension: Rows",
                    exit_cols_for_flow,
                    index=exit_cols_for_flow.index("Exit_ProjectTypeCode") if "Exit_ProjectTypeCode" in exit_cols_for_flow else 0
                )
            with flow_cols[1]:
                entry_flow_col = st.selectbox(
                    "Entry Dimension: Columns",
                    entry_cols_for_flow,
                    index=entry_cols_for_flow.index("Enter_ProjectTypeCode") if "Enter_ProjectTypeCode" in entry_cols_for_flow else 0
                )

            # Build full pivot
            flow_pivot_ra = create_flow_pivot_ra(ra_flows_df, exit_flow_col, entry_flow_col)

            # Drill-in controls
            drill_cols = st.columns(2)
            with drill_cols[0]:
                focus_exit = st.selectbox(
                    "🔍 Focus Exit Dimension",
                    ["All"] + flow_pivot_ra.index.tolist(),
                    help="Show only this exit in the flow"
                )
            with drill_cols[1]:
                focus_return = st.selectbox(
                    "🔍 Focus Return Dimension",
                    ["All"] + flow_pivot_ra.columns.tolist(),
                    help="Show only this return in the flow"
                )

            # Subset the pivot in place
            if focus_exit != "All":
                flow_pivot_ra = flow_pivot_ra.loc[[focus_exit]]
            if focus_return != "All":
                flow_pivot_ra = flow_pivot_ra[[focus_return]]

            # If there's a "No Data" or "No Return" column, push it to the end
            if "No Data" in flow_pivot_ra.columns:
                cols = [c for c in flow_pivot_ra.columns if c != "No Data"] + ["No Data"]
                flow_pivot_ra = flow_pivot_ra[cols]

            # Show the matrix
            with st.expander("🔍 Flow Matrix Details", expanded=True):
                render_dataframe_with_style(
                    flow_pivot_ra,
                    highlight_cols=[c for c in flow_pivot_ra.columns if c != "No Data"]
                )

            # Top pathways
            st.markdown("#### 🔝 Top Client Pathways")
            top_n = st.slider("Number of Flows", 5, 25, 10)
            top_flows_df = get_top_flows_from_pivot(flow_pivot_ra, top_n=top_n)
            if not top_flows_df.empty:
                render_dataframe_with_style(
                    top_flows_df,
                    highlight_cols=["Count", "Percent"] if "Percent" in top_flows_df.columns else ["Count"]
                )
            else:
                st.info("No significant flows detected")

            # Sankey diagram
            st.markdown("#### 🌐 Client Flow Network")
            sankey_ra = plot_flow_sankey_ra(flow_pivot_ra, f"{exit_flow_col} → {entry_flow_col}")
            st.plotly_chart(sankey_ra, use_container_width=True)
        else:
            st.info("📭 Insufficient data for flow analysis")
    except Exception as e:
        st.error(f"🌊 Flow Error: {str(e)}")


def display_data_export(final_df: pd.DataFrame) -> None:
    """Display data export options."""
    st.divider()
    st.markdown("### 📤 Data Export")
    render_download_button(
        final_df,
        filename="recidivism_analysis.csv",
        label="📥 Download Inbound Data"
    )


def inbound_recidivism_page() -> None:
    """Render the Inbound Recidivism Analysis page with all components."""
    st.header("📈 Inbound Recidivism Analysis")
    
    # Display the about section
    render_about_section("About Inbound Recidivism Analysis", ABOUT_INBOUND_CONTENT)
    
    # Check data availability
    df = check_data_available()
    if df is None:
        return
    
    # Setup sidebar configuration
    st.sidebar.header("⚙️ Analysis Parameters", divider="gray")
    
    # Date configuration
    report_start, report_end, days_lookback = setup_date_config(df)
    if report_start is None:
        return
    
    # Setup filters
    allowed_cocs, allowed_localcocs, allowed_agencies, allowed_programs, entry_project_types = setup_entry_filters(df)
    allowed_cocs_exit, allowed_localcocs_exit, allowed_agencies_exit, allowed_programs_exit, exit_project_types, allowed_exit_dest_cats = setup_exit_filters(df)
    
    # Prepare analysis parameters
    analysis_params = {
        "report_start": report_start,
        "report_end": report_end,
        "days_lookback": days_lookback,
        "allowed_cocs": allowed_cocs,
        "allowed_localcocs": allowed_localcocs,
        "allowed_agencies": allowed_agencies,
        "allowed_programs": allowed_programs,
        "entry_project_types": entry_project_types,
        "allowed_cocs_exit": allowed_cocs_exit,
        "allowed_localcocs_exit": allowed_localcocs_exit,
        "allowed_programs_exit": allowed_programs_exit,
        "allowed_agencies_exit": allowed_agencies_exit,
        "exit_project_types": exit_project_types,
        "allowed_exit_dest_cats": allowed_exit_dest_cats 
    }
    
    # Run analysis when button is clicked
    st.divider()
    if st.button("▶️ Run Inbound Analysis", type="primary", use_container_width=True):
        run_analysis(df, analysis_params)
    
    # Display results if analysis was successful
    final_df = get_analysis_result("inbound")
    if final_df is not None and not final_df.empty:
        # Display core metrics
        display_summary_metrics(final_df)
        
        # Display time to entry distribution
        display_time_to_entry(final_df)
        
        # Display cohort breakdowns
        display_breakdowns(final_df)
        
        # Display client flow visualization
        display_client_flow(final_df)
        
        # Display data export options
        display_data_export(final_df)


if __name__ == "__main__":
    inbound_recidivism_page()