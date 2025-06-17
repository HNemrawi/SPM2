"""
Outbound Recidivism Analysis Page
---------------------------------
Renders the outbound recidivism analysis interface and orchestrates the workflow.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any

from config.settings import DEFAULT_START_DATE, DEFAULT_END_DATE
from config.constants import EXIT_COLUMNS, RETURN_COLUMNS, DEFAULT_PROJECT_TYPES
from ui.templates import ABOUT_OUTBOUND_CONTENT, render_about_section
from ui.components import render_dataframe_with_style, render_download_button
from core.session import check_data_available, set_analysis_result, get_analysis_result
from core.utils import create_multiselect_filter, check_date_range_validity

from analysis.outbound.analysis import (
    run_outbound_recidivism,
    compute_summary_metrics,
    breakdown_by_columns
)
from analysis.outbound.visualizations import (
    display_spm_metrics,
    display_spm_metrics_non_ph,
    display_spm_metrics_ph,
    plot_days_to_return_box,
    create_flow_pivot,
    get_top_flows_from_pivot,
    plot_flow_sankey
)


def setup_reporting_period(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Configure the reporting period dates."""
    with st.sidebar.expander("📅 Reporting Period", expanded=True):
        # Set default values
        default_start = datetime(2025, 1, 1)
        default_end = datetime(2025, 1, 31)
        
        date_range = st.date_input(
            "Exit date range",
            value=[default_start, default_end],
            help="Clients must have an exit date inside this window.",
        )
        st.caption("📌 **Note:** The selected end date will be included in the analysis period.")

        # Handle different return types from date_input
        if date_range is None:
            # No dates selected
            st.warning("Please select both start and end dates for the exit date range.")
            report_start = pd.to_datetime(default_start)
            report_end = pd.to_datetime(default_end)
        elif isinstance(date_range, (list, tuple)):
            if len(date_range) == 2:
                # Both dates selected
                report_start = pd.to_datetime(date_range[0])
                report_end = pd.to_datetime(date_range[1])
            elif len(date_range) == 1:
                # Only one date selected
                st.warning("Please select an end date for the exit date range.")
                report_start = pd.to_datetime(date_range[0])
                report_end = pd.to_datetime(date_range[0])
            else:
                # Empty list/tuple
                st.warning("Please select both start and end dates for the exit date range.")
                report_start = pd.to_datetime(default_start)
                report_end = pd.to_datetime(default_end)
        elif isinstance(date_range, (datetime, date)):
            # Single date object (shouldn't happen with range input, but just in case)
            st.warning("Please select an end date for the exit date range.")
            report_start = report_end = pd.to_datetime(date_range)
        else:
            # Unexpected type
            st.warning("Please select both start and end dates for the exit date range.")
            report_start = pd.to_datetime(default_start)
            report_end = pd.to_datetime(default_end)
        
        # Ensure start is before end
        if report_start > report_end:
            st.error("Start date must be before end date. Dates have been swapped.")
            report_start, report_end = report_end, report_start
        
        if df is not None and not df.empty:
            data_reporting_start = pd.to_datetime(df["ReportingPeriodStartDate"].iloc[0])
            data_reporting_end = pd.to_datetime(df["ReportingPeriodEndDate"].iloc[0])
            
            check_date_range_validity(
                report_start, 
                report_end, 
                data_reporting_start, 
                data_reporting_end
            )
            
        return report_start, report_end


def setup_exit_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Configure exit-specific filters."""
    with st.sidebar.expander("🚪 Exit Filters", expanded=True):
        exit_cocs = create_multiselect_filter(
            "CoC Codes",
            df["ProgramSetupCoC"].dropna().unique() if "ProgramSetupCoC" in df.columns else [],
            default=None,
            help_text="Filter exits by CoC code",
        )
        exit_localcocs = create_multiselect_filter(
            "Local CoC",
            df["LocalCoCCode"].dropna().unique() if "LocalCoCCode" in df.columns else [],
            default=None,
            help_text="Filter exits by local CoC",
        )
        exit_agencies = create_multiselect_filter(
            "Agencies",
            df["AgencyName"].dropna().unique() if "AgencyName" in df.columns else [],
            default=None,
            help_text="Filter exits by agency",
        )
        exit_programs = create_multiselect_filter(
            "Programs",
            df["ProgramName"].dropna().unique() if "ProgramName" in df.columns else [],
            default=None,
            help_text="Filter exits by program",
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
        
        return {
            "exit_cocs": exit_cocs,
            "exit_localcocs": exit_localcocs,
            "exit_agencies": exit_agencies,
            "exit_programs": exit_programs,
            "exiting_projects": exiting_projects,
            "allowed_exit_dest_cats": allowed_exit_dest_cats
        }


def setup_return_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Configure return-specific filters."""
    with st.sidebar.expander("↩️ Return Filters", expanded=True):
        return_cocs = create_multiselect_filter(
            "CoC Codes",
            df["ProgramSetupCoC"].dropna().unique() if "ProgramSetupCoC" in df.columns else [],
            default=None,
            help_text="Filter next enrollments by CoC code",
        )
        return_localcocs = create_multiselect_filter(
            "Local CoC",
            df["LocalCoCCode"].dropna().unique() if "LocalCoCCode" in df.columns else [],
            default=None,
            help_text="Filter next enrollments by local CoC",
        )
        return_agencies = create_multiselect_filter(
            "Agencies",
            df["AgencyName"].dropna().unique() if "AgencyName" in df.columns else [],
            default=None,
            help_text="Filter next enrollments by agency",
        )
        return_programs = create_multiselect_filter(
            "Programs",
            df["ProgramName"].dropna().unique() if "ProgramName" in df.columns else [],
            default=None,
            help_text="Filter next enrollments by program",
        )
        
        all_project_types = sorted(df["ProjectTypeCode"].dropna().unique().tolist())
        default_projects = [p for p in DEFAULT_PROJECT_TYPES if p in all_project_types]
        return_projects = st.multiselect(
            "Project Types (Return)",
            all_project_types,
            default=default_projects,
            help="Project types treated as candidate returns",
        )
        
        return {
            "return_cocs": return_cocs,
            "return_localcocs": return_localcocs,
            "return_agencies": return_agencies,
            "return_programs": return_programs,
            "return_projects": return_projects
        }


def setup_continuum_filters(df: pd.DataFrame) -> Optional[List[str]]:
    """Configure continuum project filters."""
    with st.sidebar.expander("⚡ Continuum", expanded=False):
        cont_opts = (
            sorted(df["ProgramsContinuumProject"].dropna().unique())
            if "ProgramsContinuumProject" in df.columns
            else []
        )
        return create_multiselect_filter(
            "Programs Continuum Project",
            cont_opts,
            default=None,
            help_text="Optional continuum filter",
        )


def run_analysis(df: pd.DataFrame, analysis_params: Dict[str, Any]) -> bool:
    """Run the outbound recidivism analysis with the provided parameters."""
    try:
        with st.status("🔄 Running Outbound Recidivism…", expanded=True) as status:
            outbound_df = run_outbound_recidivism(
                df,
                analysis_params["report_start"],
                analysis_params["report_end"],
                exit_cocs=analysis_params["exit_cocs"],
                exit_localcocs=analysis_params["exit_localcocs"],
                exit_agencies=analysis_params["exit_agencies"],
                exit_programs=analysis_params["exit_programs"],
                return_cocs=analysis_params["return_cocs"],
                return_localcocs=analysis_params["return_localcocs"],
                return_agencies=analysis_params["return_agencies"],
                return_programs=analysis_params["return_programs"],
                allowed_continuum=analysis_params["chosen_continuum"],
                allowed_exit_dest_cats=analysis_params["allowed_exit_dest_cats"],
                exiting_projects=analysis_params["exiting_projects"],
                return_projects=analysis_params["return_projects"],
            )
            
            # Clean up the results dataframe
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
            outbound_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')
            
            cols_to_rename = [
                "Exit_UniqueIdentifier", 
                "Exit_ClientID",
                "Exit_RaceEthnicity",
                "Exit_Gender",
                "Exit_DOB",
                "Exit_VeteranStatus"
            ]
            mapping = {col: col[len("Exit_"):] for col in cols_to_rename if col in outbound_df.columns}
            outbound_df.rename(columns=mapping, inplace=True)

            set_analysis_result("outbound", outbound_df)
            status.update(label="✅ Done!", state="complete")
            st.toast("Analysis complete 🎉", icon="🎉")
            return True
    except Exception as exc:
        st.status(label=f"🚨 Error: {exc}", state="error")
        return False


def display_summary_metrics(out_df: pd.DataFrame, allowed_exit_dest_cats: Optional[List[str]] = None) -> None:
    """Display the core performance metrics summary."""
    st.divider()
    st.markdown("### 📊 Outbound Analysis Summary")

    if allowed_exit_dest_cats == ["Permanent Housing Situations"]:
        st.caption("📌 **Note:** only Permanent Housing Situations is selected in the Exit Destination Categories filter.")

    # Compute metrics
    metrics = compute_summary_metrics(out_df)
    
    # Row 1: Exit Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Relevant Exits",
            value=f"{metrics['Number of Relevant Exits']:,}"
        )
    
    with col2:
        st.metric(
            label="Exits to Permanent Housing",
            value=f"{metrics['Total Exits to PH']:,}",
            help="Number of exits to permanent housing destinations"
        )
    
    with col3:
        ph_exit_rate = (metrics['Total Exits to PH'] / metrics['Number of Relevant Exits'] * 100) if metrics['Number of Relevant Exits'] > 0 else 0
        st.metric(
            label="PH Exit Rate",
            value=f"{ph_exit_rate:.1f}%",
            help="Percentage of exits going to permanent housing"
        )
    
    # Row 2: Return Analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Returns",
            value=f"{metrics['Return']:,}",
        )
    
    with col2:
        st.metric(
            label="Returns to Homelessness (From PH)",
            value=f"{metrics['Return to Homelessness']:,}",
            help="Returns to homelessness from permanent housing exits only"
        )
    
    with col3:
        st.metric(
            label="Return to Homelessness Rate (From PH)",
            value=f"{metrics['% Return to Homelessness']:.1f}%",
            help="Percentage of PH exits that return to homelessness",
        )
    
    # Row 3: Timing Analysis
    if metrics['Return'] > 0:  # Only show timing if there are returns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Median Days to Return",
                value=f"{metrics['Median Days']:.0f}",
                help="Middle value of days between exit and return"
            )
        
        with col2:
            st.metric(
                label="Average Days to Return",
                value=f"{metrics['Average Days']:.0f}",
                help="Mean number of days between exit and return"
            )
        
        with col3:
            st.metric(
                label="Maximum Days to Return",
                value=f"{metrics['Max Days']:.0f}",
                help="Longest time between exit and return"
            )


def display_days_to_return(out_df: pd.DataFrame) -> None:
    """Display the days-to-return distribution visualization."""
    st.divider()
    st.markdown("### ⏳ Days to Return Distribution")
    st.plotly_chart(plot_days_to_return_box(out_df), use_container_width=True)

@st.fragment
def display_breakdown_analysis(out_df: pd.DataFrame) -> None:
    """Display the breakdown analysis by selected dimensions."""
    st.divider()
    st.markdown("### 📈 Breakdown")
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
        "AgeAtExitRange",
    ]
    cols_to_group = st.multiselect(
        "Group by columns",
        options=[col for col in breakdown_columns if col in out_df.columns],
        default=["Exit_ProjectTypeCode"] if "Exit_ProjectTypeCode" in out_df.columns else [],
        help="Select up to 3 columns",
    )
    if cols_to_group:
        bdf = breakdown_by_columns(out_df, cols_to_group[:3])
        render_dataframe_with_style(
            bdf,
            highlight_cols=["Relevant Exits"]
        )

@st.fragment
def display_client_flow(out_df: pd.DataFrame) -> None:
    """Display client flow analysis visualization."""
    st.divider()
    st.markdown("### 🌊 Client Flow Analysis")

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
    ]
    exit_dims = [c for c in exit_columns if c in out_df.columns]
    ret_dims = [c for c in return_columns if c in out_df.columns]

    if exit_dims and ret_dims:
        # Dimension selectors
        st.caption("📌 **Note:** Both the Exit and Entry Dimension filters apply to the entire flow section, including Client Flow Analysis, Top Client Pathways, and Client Flow Network.")
        flow_cols = st.columns(2)
        with flow_cols[0]:
            ex_choice = st.selectbox(
                "Exit Dimension: Rows",
                exit_dims,
                index=exit_dims.index("Exit_ProjectTypeCode") if "Exit_ProjectTypeCode" in exit_dims else 0,
            )
        with flow_cols[1]:
            ret_choice = st.selectbox(
                "Return Dimension: Columns",
                ret_dims,
                index=ret_dims.index("Return_ProjectTypeCode") if "Return_ProjectTypeCode" in ret_dims else 0,
            )

        # Build the full pivot (unfiltered for matrix and top pathways)
        pivot_c = create_flow_pivot(out_df, ex_choice, ret_choice)
        if "No Return" in pivot_c.columns:
            cols_order = [c for c in pivot_c.columns if c != "No Return"] + ["No Return"]
            pivot_c = pivot_c[cols_order]

        # 1) Flow Matrix Details (using unfiltered data)
        with st.expander("🔍 Flow Matrix Details", expanded=True):
            if pivot_c.empty:
                st.info("📭 No return enrollments to build flow.")
            else:
                cols_to_color = [c for c in pivot_c.columns if c != "No Return"]
                render_dataframe_with_style(
                    pivot_c,
                    highlight_cols=cols_to_color
                )

        # 2) Top Client Pathways (using unfiltered data)
        st.markdown("#### 🔝 Top Client Pathways")
        
        # Check if pivot table has enough data for top flows
        if pivot_c.empty or pivot_c.sum().sum() == 0:
            st.info("No significant pathways detected")
        else:
            # Calculate minimum value for the slider based on available data
            # Get the actual number of non-zero flows in the pivot table
            non_zero_cells = (pivot_c > 0).sum().sum()
            
            # If there are very few pathways, just display them without a slider
            if non_zero_cells <= 5:
                st.info(f"Only {non_zero_cells} pathway{'s' if non_zero_cells != 1 else ''} detected")
                try:
                    top_flows_df = get_top_flows_from_pivot(pivot_c, top_n=int(non_zero_cells))
                    if not top_flows_df.empty:
                        render_dataframe_with_style(
                            top_flows_df,
                            highlight_cols=["Count", "Percent"] if "Percent" in top_flows_df.columns else ["Count"]
                        )
                except Exception as e:
                    st.error(f"Error generating flows: {str(e)}")
            else:
                # We have enough pathways for a slider
                min_flows = 1
                max_flows = min(25, non_zero_cells)
                default_flows = min(3, max_flows)
                
                top_n = st.slider(
                    "Number of Flows",
                    min_value=min_flows,
                    max_value=max_flows,
                    value=default_flows,
                    step=1,
                    help="Top N pathways to display",
                )
                
                try:
                    top_flows_df = get_top_flows_from_pivot(pivot_c, top_n=top_n)
                    
                    if top_flows_df.empty:
                        st.info("No significant pathways detected")
                    elif "Count" not in top_flows_df.columns:
                        st.info("Insufficient data to create meaningful flow paths")
                        render_dataframe_with_style(top_flows_df)
                    else:
                        render_dataframe_with_style(
                            top_flows_df,
                            highlight_cols=["Count", "Percent"] if "Percent" in top_flows_df.columns else ["Count"]
                        )
                except Exception as e:
                    st.error(f"Error generating top flows: {str(e)}")
                    st.info("Unable to display top client pathways due to insufficient data")

        # 3) Client Flow Network with focus controls
        st.markdown("#### 🌐 Client Flow Network")
        
        # Focus controls (only for network graph)
        st.caption("🎯 **Focus filters below apply only to the network visualization**")
        colL, colR = st.columns(2)
        with colL:
            focus_exit = st.selectbox(
                "🔍 Focus Exit Dimension",
                ["All"] + pivot_c.index.tolist(),
                help="Show only this exit in the network",
            )
        with colR:
            focus_return = st.selectbox(
                "🔍 Focus Return Dimension",
                ["All"] + pivot_c.columns.tolist(),
                help="Show only this return in the network",
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


def display_ph_comparison(out_df: pd.DataFrame) -> None:
    """Display PH vs. Non-PH exit comparison."""
    st.divider()
    st.markdown("### 🏠 PH vs. Non‑PH Exit Comparison")
    if st.checkbox("Show comparison", value=False):
        ph_df = out_df[out_df["PH_Exit"]]
        nonph_df = out_df[~out_df["PH_Exit"]]
        
        # Create columns with a gap in between for visual separation
        c1, spacer, c2 = st.columns([5, 0.2, 5])
        
        with c1:
            st.subheader("🏠 Permanent Housing Exits")
            if not ph_df.empty:
                display_spm_metrics_ph(compute_summary_metrics(ph_df))
            else:
                st.info("No PH exits found.")
        
        with spacer:
            # This creates a narrow column that acts as a visual separator
            st.empty()
        
        with c2:
            st.subheader("🏕️ Non‑Permanent Housing Exits")
            if not nonph_df.empty:
                display_spm_metrics_non_ph(compute_summary_metrics(nonph_df))
            else:
                st.info("No Non‑PH exits found.")


def display_data_export(out_df: pd.DataFrame) -> None:
    """Display data export options."""
    st.divider()
    render_download_button(
        out_df,
        filename="OutboundRecidivismResults.csv",
        label="⬇️ Download results (CSV)"
    )


def outbound_recidivism_page() -> None:
    """Render the Outbound Recidivism page with all components."""
    st.header("📈 Outbound Recidivism Analysis")
    
    # Display the about section
    render_about_section("About Outbound Recidivism Analysis", ABOUT_OUTBOUND_CONTENT)
    
    # Check data availability
    df = check_data_available()
    if df is None:
        return
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Outbound Parameters")
    
    # Setup filters and parameters
    report_start, report_end = setup_reporting_period(df)
    exit_filters = setup_exit_filters(df)
    return_filters = setup_return_filters(df)
    chosen_continuum = setup_continuum_filters(df)
    
    # Prepare analysis parameters
    analysis_params = {
        "report_start": report_start,
        "report_end": report_end,
        "exit_cocs": exit_filters["exit_cocs"],
        "exit_localcocs": exit_filters["exit_localcocs"],
        "exit_agencies": exit_filters["exit_agencies"],
        "exit_programs": exit_filters["exit_programs"],
        "exiting_projects": exit_filters["exiting_projects"],
        "allowed_exit_dest_cats": exit_filters["allowed_exit_dest_cats"],
        "return_cocs": return_filters["return_cocs"],
        "return_localcocs": return_filters["return_localcocs"],
        "return_agencies": return_filters["return_agencies"],
        "return_programs": return_filters["return_programs"],
        "return_projects": return_filters["return_projects"],
        "chosen_continuum": chosen_continuum
    }
    
    # Run analysis button
    st.divider()
    if st.button("▶️ Run Analysis", type="primary", use_container_width=True):
        run_analysis(df, analysis_params)
    
    # Display results if analysis was successful
    out_df = get_analysis_result("outbound")
    if out_df is not None and not out_df.empty:
        # Display core metrics and visualizations
        display_summary_metrics(out_df, exit_filters["allowed_exit_dest_cats"])
        display_days_to_return(out_df)
        display_breakdown_analysis(out_df)
        display_client_flow(out_df)
        display_ph_comparison(out_df)
        display_data_export(out_df)


if __name__ == "__main__":
    outbound_recidivism_page()