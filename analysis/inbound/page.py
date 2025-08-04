"""
Inbound Recidivism Analysis Page
--------------------------------
Renders the inbound recidivism analysis interface and orchestrates the workflow.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any

from config.constants import DEFAULT_PROJECT_TYPES
from ui.templates import ABOUT_INBOUND_CONTENT, render_about_section
from ui.components import render_dataframe_with_style, render_download_button
from core.session import check_data_available, set_analysis_result, get_analysis_result
from core.utils import create_multiselect_filter, check_date_range_validity

# Import styling utilities
from ui.styling import (
    apply_custom_css,
    style_metric_cards,
    create_info_box,
    create_styled_divider,
    apply_chart_theme,
    NeutralColors
)

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
    get_top_flows_from_pivot,
    display_time_statistics
)

# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================

def setup_date_config(df: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[int]]:
    """Configure date parameters and lookback period."""
    with st.sidebar.expander("🗓️ **Entry Date & Lookback**", expanded=True):
        # Set default values
        default_start = datetime(2025, 1, 1)
        default_end = datetime(2025, 1, 31)
        
        try:
            date_range = st.date_input(
                "Entry Date Range",
                [default_start, default_end],
                help="Analysis period for new entries"
            )
            
            # Info box for date selection
            st.html(create_info_box(
                "The selected end date will be included in the analysis period.",
                type="info",
                icon="📌"
            ))

            # Handle different return types from date_input
            if date_range is None:
                st.error("⚠️ Please select both dates")
                return None, None, None
            elif isinstance(date_range, (list, tuple)):
                if len(date_range) == 2:
                    report_start = pd.to_datetime(date_range[0])
                    report_end = pd.to_datetime(date_range[1])
                elif len(date_range) == 1:
                    st.error("⚠️ Please select an end date")
                    return None, None, None
                else:
                    st.error("⚠️ Please select both dates")
                    return None, None, None
            elif isinstance(date_range, (datetime, date)):
                st.error("⚠️ Please select an end date")
                return None, None, None
            else:
                st.error("⚠️ Invalid date selection. Please select both dates")
                return None, None, None
            
            # Ensure start is before end
            if report_start > report_end:
                st.error("⚠️ Start date must be before end date")
                return None, None, None
                
        except Exception as e:
            st.error(f"📅 Date Error: {str(e)}")
            return None, None, None

        st.html(create_styled_divider())

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
    with st.sidebar.expander("📍 **Entry Filters**", expanded=False):
        st.markdown("#### Entry Enrollment Criteria")
        
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
        
        # Add SSVF RRH filter for entries
        entry_ssvf_rrh = create_multiselect_filter(
            "SSVF RRH - Entry",
            sorted(df["SSVF_RRH"].dropna().unique().tolist()) if "SSVF_RRH" in df.columns else [],
            default=["ALL"],
            help_text="SSVF RRH filter for entries"
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

        return allowed_cocs, allowed_localcocs, allowed_agencies, allowed_programs, entry_project_types, entry_ssvf_rrh

def setup_exit_filters(df: pd.DataFrame) -> Tuple[Optional[List[str]], ...]:
    """Configure exit-specific filters."""
    with st.sidebar.expander("🚪 **Exit Filters**", expanded=False):
        st.markdown("#### Prior Exit Criteria")
        
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
        
        # Add SSVF RRH filter for exits
        exit_ssvf_rrh = create_multiselect_filter(
            "SSVF RRH - Exit",
            sorted(df["SSVF_RRH"].dropna().unique().tolist()) if "SSVF_RRH" in df.columns else [],
            default=["ALL"],
            help_text="SSVF RRH filter for exits"
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
        
        # Add Exit Destinations filter
        allowed_exit_destinations = None
        if "ExitDestination" in df.columns:
            allowed_exit_destinations = create_multiselect_filter(
                "Exit Destinations",
                sorted(df["ExitDestination"].dropna().unique().tolist()),
                default=["ALL"],
                help_text="Limit exits to these specific destinations"
            )

        return allowed_cocs_exit, allowed_localcocs_exit, allowed_agencies_exit, allowed_programs_exit, exit_project_types, exit_ssvf_rrh, allowed_exit_dest_cats, allowed_exit_destinations

# ============================================================================
# ANALYSIS EXECUTION
# ============================================================================

def run_analysis(df: pd.DataFrame, analysis_params: Dict[str, Any]) -> bool:
    """Execute the inbound recidivism analysis with specified parameters."""
    try:
        with st.status("🔍 Processing Inbound Analysis...", expanded=True) as status:
            status.write("📊 Identifying entries in the reporting period...")
            
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
                entry_ssvf_rrh=analysis_params["entry_ssvf_rrh"],
                allowed_cocs_exit=analysis_params["allowed_cocs_exit"],
                allowed_localcocs_exit=analysis_params["allowed_localcocs_exit"],
                allowed_programs_exit=analysis_params["allowed_programs_exit"],
                allowed_agencies_exit=analysis_params["allowed_agencies_exit"],
                exit_project_types=analysis_params["exit_project_types"],
                exit_ssvf_rrh=analysis_params["exit_ssvf_rrh"],
                allowed_exit_dest_cats=analysis_params["allowed_exit_dest_cats"],
                allowed_exit_destinations=analysis_params["allowed_exit_destinations"]
            )

            status.write("🔄 Matching prior exits...")

            # Clean up columns
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

            # Rename columns
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
            
            status.write("💾 Finalizing results...")
            
            set_analysis_result("inbound", merged_df)
            status.update(label="✅ Inbound Analysis Complete!", state="complete", expanded=False)
            
        st.toast("🎉 Analysis completed successfully!", icon="✅")
        
        # Rerun to display results
        st.rerun()
        
        return True
        
    except Exception as e:
        st.error(f"🚨 Analysis Error: {str(e)}")
        return False

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_summary_metrics(final_df: pd.DataFrame, allowed_exit_dest_cats: Optional[List[str]] = None) -> None:
    """Display core performance metrics with natural styling."""
    st.html(create_styled_divider("gradient"))
    st.markdown("### 📊 Inbound Analysis Summary")

    if allowed_exit_dest_cats == ["Permanent Housing Situations"]:
        st.html(create_info_box(
            "Only Permanent Housing Situations is selected in the Exit Destination Categories filter.",
            type="info",
            icon="📌"
        ))

    # Apply metric card styling
    style_metric_cards(
        border_left_color=NeutralColors.PRIMARY,
        box_shadow=True
    )

    metrics = compute_return_metrics(final_df)
    display_return_metrics_cards(metrics)

def display_time_to_entry(final_df: pd.DataFrame) -> None:
    """Display time-to-entry distribution visualization."""
    st.html(create_styled_divider("gradient"))
    st.markdown("### ⏳ Days to Return Distribution")
    
    try:
        fig = plot_time_to_entry_box(final_df)
        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display time statistics if available
        display_time_statistics(final_df)
        
    except Exception as e:
        st.error(f"📉 Visualization Error: {str(e)}")

@st.fragment
def display_breakdowns(final_df: pd.DataFrame) -> None:
    """Display cohort breakdown analysis."""
    st.html(create_styled_divider("gradient"))
    st.markdown("### 📈 Breakdown Analysis")
    
    # Add the collapsible description with styled info box
    with st.expander("Understanding Breakdown Categories", expanded=False):
        st.html(create_info_box("""
        <h4>This table can be grouped by:</h4>
        
        <p><strong>• Client-level values</strong> (race, gender, etc.): These are demographic factors that are stable for clients across their enrollments.</p>
        
        <p><strong>• Entry-based values</strong>: Clients are categorized based on information relating to their Inbound Entry.</p>
        
        <p><strong>• Exit-based values</strong>: Clients are categorized based on information relating to most recent exit prior to their Entry during the reporting period. Because we are grouping by values related to their previous exit, clients who did NOT have a previous exit (new clients) will be excluded from the table.</p>
        """, type="info"))
    
    breakdown_columns = [
        # Client Demographics
        "RaceEthnicity",
        "Gender",
        "VeteranStatus",
        
        # Entry Characteristics
        "Enter_HasIncome",
        "Enter_HasDisability",
        "Enter_HouseholdType",
        "Enter_IsHeadOfHousehold",
        "Enter_CHStartHousehold",
        "Enter_CurrentlyFleeingDV",
        "Enter_LocalCoCCode",
        "Enter_PriorLivingCat",
        "Enter_AgeTieratEntry",
        
        # Entry Program Information
        "Enter_ProgramSetupCoC",
        "Enter_ProjectTypeCode",
        "Enter_AgencyName",
        "Enter_ProgramName",
        "Enter_SSVF_RRH",
        "Enter_ProgramsContinuumProject",
        
        # Entry Exit Information (if they exit again)
        "Enter_ExitDestinationCat",
        "Enter_ExitDestination",
        
        # Return Status
        "ReturnCategory",
        
        # Prior Exit Characteristics (only for returning clients)
        "Exit_HasIncome",
        "Exit_HasDisability",
        "Exit_HouseholdType",
        "Exit_IsHeadOfHousehold",
        "Exit_CHStartHousehold",
        "Exit_CurrentlyFleeingDV",
        "Exit_LocalCoCCode",
        "Exit_PriorLivingCat",
        "Exit_AgeTieratEntry",
        
        # Prior Exit Program Information
        "Exit_ProgramSetupCoC",
        "Exit_ProjectTypeCode",
        "Exit_AgencyName",
        "Exit_ProgramName",
        "Exit_SSVF_RRH",
        "Exit_ProgramsContinuumProject",
        
        # Prior Exit Destination
        "Exit_ExitDestinationCat",
        "Exit_ExitDestination",
    ]

    possible_cols = [col for col in breakdown_columns if col in final_df.columns]
    default_breakdown = ["Enter_ProjectTypeCode"] if "Enter_ProjectTypeCode" in possible_cols else []
    
    analysis_cols = st.columns([3, 1])
    
    with analysis_cols[0]:
        chosen = st.multiselect(
            "Group By Dimensions",
            possible_cols,
            default=default_breakdown,
            help="Select grouping columns for analysis"
        )
    
    if chosen:
        try:
            breakdown = return_breakdown_analysis(final_df, chosen)

            # If any chosen column starts with "exit_", drop the "New (%)" column
            if any(col.lower().startswith("exit_") for col in chosen):
                breakdown = breakdown.drop(columns=["New (%)"], errors="ignore")

            with analysis_cols[1]:
                st.metric("Total Groups", len(breakdown))

            render_dataframe_with_style(
                breakdown,
                highlight_cols=["Total Entries", "Returning (%)", "Returning From Housing (%)"],
                height=400
            )
        except Exception as e:
            st.error(f"📊 Breakdown Error: {str(e)}")
    
    st.html('</div>')

@st.fragment
def display_client_flow(final_df: pd.DataFrame) -> None:
    """Display client flow analysis visualization."""
    st.html(create_styled_divider("gradient"))
    st.markdown("### 🌊 Client Flow Analysis")
    
    try:
        ra_flows_df = (
            final_df[final_df["ReturnCategory"].str.contains("Returning")]
            if "ReturnCategory" in final_df.columns
            else pd.DataFrame()
        )

        if ra_flows_df.empty:
            st.info("No returning clients found for flow analysis")
            return

        exit_columns = [
            "Exit_HasIncome", 
            "Exit_HasDisability", 
            "Exit_HouseholdType",
            "Exit_IsHeadOfHousehold",
            "Exit_CHStartHousehold", 
            "Exit_CurrentlyFleeingDV",
            "Exit_LocalCoCCode", 
            "Exit_PriorLivingCat",
            "Exit_ProgramSetupCoC", 
            "Exit_ProjectTypeCode", 
            "Exit_AgencyName",
            "Exit_ProgramName", 
            "Exit_SSVF_RRH",
            "Exit_ProgramsContinuumProject",
            "Exit_ExitDestinationCat", 
            "Exit_ExitDestination",
            "Exit_AgeTieratEntry",
        ]
        entry_columns = [
            "Enter_HasIncome", 
            "Enter_HasDisability", 
            "Enter_HouseholdType",
            "Enter_IsHeadOfHousehold",
            "Enter_CHStartHousehold", 
            "Enter_CurrentlyFleeingDV",
            "Enter_LocalCoCCode", 
            "Enter_PriorLivingCat",
            "Enter_ProgramSetupCoC", 
            "Enter_ProjectTypeCode", 
            "Enter_AgencyName",
            "Enter_ProgramName", 
            "Enter_SSVF_RRH",
            "Enter_ProgramsContinuumProject",
            "Enter_ExitDestinationCat", 
            "Enter_ExitDestination",
            "Enter_AgeTieratEntry",
        ]

        exit_cols_for_flow = [c for c in exit_columns if c in ra_flows_df.columns]
        entry_cols_for_flow = [c for c in entry_columns if c in ra_flows_df.columns]

        if exit_cols_for_flow and entry_cols_for_flow:
            # Dimension selectors with info box
            st.html(create_info_box(
                "Both the Exit and Entry Dimension filters apply to the entire flow section.",
                type="info",
                icon="📌"
            ))
            
            flow_cols = st.columns(2)
            with flow_cols[0]:
                exit_flow_col = st.selectbox(
                    "Exit Dimension: Rows",
                    exit_cols_for_flow,
                    index=exit_cols_for_flow.index("Exit_ProjectTypeCode") if "Exit_ProjectTypeCode" in exit_cols_for_flow else 0,
                    help="Characteristic at prior exit point"
                )
            with flow_cols[1]:
                entry_flow_col = st.selectbox(
                    "Entry Dimension: Columns",
                    entry_cols_for_flow,
                    index=entry_cols_for_flow.index("Enter_ProjectTypeCode") if "Enter_ProjectTypeCode" in entry_cols_for_flow else 0,
                    help="Characteristic at current entry point"
                )

            # Build full pivot
            flow_pivot_ra = create_flow_pivot_ra(ra_flows_df, exit_flow_col, entry_flow_col)

            # Reorder columns if needed
            if "No Data" in flow_pivot_ra.columns:
                cols = [c for c in flow_pivot_ra.columns if c != "No Data"] + ["No Data"]
                flow_pivot_ra = flow_pivot_ra[cols]

            # Flow Matrix with neutral styling
            with st.expander("🔍 **Flow Matrix Details**", expanded=True):
                render_dataframe_with_style(
                    flow_pivot_ra,
                    highlight_cols=[c for c in flow_pivot_ra.columns if c != "No Data"],
                    axis=1
                )

            # Top pathways section
            st.html(create_styled_divider())
            st.markdown("#### 🔝 Top Client Pathways")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                top_n = st.slider(
                    "Number of Pathways", 
                    min_value=5,
                    max_value=25,
                    value=10,
                    help="Top N pathways to display"
                )
            
            top_flows_df = get_top_flows_from_pivot(flow_pivot_ra, top_n=top_n)
            if not top_flows_df.empty:
                render_dataframe_with_style(
                    top_flows_df,
                    highlight_cols=["Count", "Percent"] if "Percent" in top_flows_df.columns else ["Count"]
                )
            else:
                st.info("No significant flows detected")

            # Network visualization
            st.html(create_styled_divider())
            st.markdown("#### 🌐 Client Flow Network")
            
            st.html(create_info_box(
                "Focus filters below apply only to the network visualization",
                type="warning",
                icon="🎯"
            ))
            
            drill_cols = st.columns(2)
            with drill_cols[0]:
                focus_exit = st.selectbox(
                    "🔍 Focus Exit Dimension",
                    ["All"] + flow_pivot_ra.index.tolist(),
                    help="Show only this exit in the network"
                )
            with drill_cols[1]:
                focus_return = st.selectbox(
                    "🔍 Focus Entry Dimension",
                    ["All"] + flow_pivot_ra.columns.tolist(),
                    help="Show only this entry in the network"
                )

            # Create filtered pivot for Sankey
            flow_pivot_sankey = flow_pivot_ra.copy()
            
            if focus_exit != "All":
                flow_pivot_sankey = flow_pivot_sankey.loc[[focus_exit]]
            if focus_return != "All":
                flow_pivot_sankey = flow_pivot_sankey[[focus_return]]

            # Generate Sankey with themed styling
            sankey_ra = plot_flow_sankey_ra(flow_pivot_sankey, f"{exit_flow_col} → {entry_flow_col}")
            sankey_ra = apply_chart_theme(sankey_ra)
            st.plotly_chart(sankey_ra, use_container_width=True)
        else:
            st.info("📭 Insufficient data for flow analysis")
            
    except Exception as e:
        st.error(f"🌊 Flow Error: {str(e)}")

def display_data_export(final_df: pd.DataFrame) -> None:
    """Display data export options."""
    st.html(create_styled_divider("gradient"))
    st.markdown("### 📤 Data Export")
    
    # Export section with styled card
    st.html('<div class="neutral-card">')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_download_button(
            final_df,
            filename="inbound_recidivism_analysis.csv",
            label="📥 Download Inbound Data"
        )
    st.html('</div>')

# ============================================================================
# MAIN PAGE FUNCTION
# ============================================================================

def inbound_recidivism_page() -> None:
    """Render the Inbound Recidivism Analysis page with all components."""
    # Apply custom CSS theme
    apply_custom_css()
    
    st.header("📈 Inbound Recidivism Analysis")
    
    # Display the about section
    render_about_section("About Inbound Recidivism Analysis", ABOUT_INBOUND_CONTENT)
    
    # Check data availability
    df = check_data_available()
    if df is None:
        return
    
    # Setup sidebar configuration with themed styling
    st.sidebar.html('<div class="sidebar-content">')
    st.sidebar.header("⚙️ Analysis Parameters")
    st.sidebar.html(create_styled_divider())
    
    # Date configuration
    report_start, report_end, days_lookback = setup_date_config(df)
    if report_start is None:
        return
    
    # Setup filters
    entry_filters = setup_entry_filters(df)
    allowed_cocs, allowed_localcocs, allowed_agencies, allowed_programs, entry_project_types, entry_ssvf_rrh = entry_filters
    
    exit_filters = setup_exit_filters(df)
    allowed_cocs_exit, allowed_localcocs_exit, allowed_agencies_exit, allowed_programs_exit, exit_project_types, exit_ssvf_rrh, allowed_exit_dest_cats, allowed_exit_destinations = exit_filters
    
    st.sidebar.html('</div>')
    
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
        "entry_ssvf_rrh": entry_ssvf_rrh,
        "allowed_cocs_exit": allowed_cocs_exit,
        "allowed_localcocs_exit": allowed_localcocs_exit,
        "allowed_programs_exit": allowed_programs_exit,
        "allowed_agencies_exit": allowed_agencies_exit,
        "exit_project_types": exit_project_types,
        "exit_ssvf_rrh": exit_ssvf_rrh,
        "allowed_exit_dest_cats": allowed_exit_dest_cats,
        "allowed_exit_destinations": allowed_exit_destinations
    }
    
    # Run analysis section
    st.html(create_styled_divider("gradient"))
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶️ Run Inbound Analysis", type="primary", use_container_width=True):
            run_analysis(df, analysis_params)
    
    # Display results if analysis was successful
    final_df = get_analysis_result("inbound")
    if final_df is not None and not final_df.empty:
        # Display all analysis sections with themed styling
        display_summary_metrics(final_df, allowed_exit_dest_cats)
        display_time_to_entry(final_df)
        display_breakdowns(final_df)
        display_client_flow(final_df)
        display_data_export(final_df)

if __name__ == "__main__":
    inbound_recidivism_page()