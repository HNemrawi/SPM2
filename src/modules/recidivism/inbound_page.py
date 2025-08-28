"""
Inbound Recidivism Analysis Page
--------------------------------
Renders the inbound recidivism analysis interface and orchestrates the workflow.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from config.app_config import config
from src.core.session.manager import (
    StateManager,
    check_data_available,
    get_analysis_result,
    set_analysis_result,
)
from src.core.utils.helpers import (
    check_date_range_validity,
    create_multiselect_filter,
)
from src.modules.recidivism.inbound_calculator import (
    compute_return_metrics,
    return_breakdown_analysis,
    run_return_analysis,
)
from src.modules.recidivism.inbound_viz import (
    create_flow_pivot_ra,
    display_return_metrics_cards,
    get_top_flows_from_pivot,
    plot_flow_sankey_ra,
    plot_time_to_entry_box,
)
from src.ui.factories.components import ui
from src.ui.factories.html import html_factory
from src.ui.layouts.templates import ABOUT_INBOUND_CONTENT
from src.ui.layouts.widgets import (
    render_about_section,
    render_dataframe_with_style,
    render_download_button,
)
from src.ui.themes.styles import (
    NeutralColors,
    apply_chart_theme,
    apply_custom_css,
    create_styled_divider,
)

# ============================================================================
# CONSTANTS
# ============================================================================

# Module identifier for state management
INBOUND_MODULE = StateManager.RECIDIVISM_INBOUND_PREFIX

# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================


def _set_dirty_flag():
    st.session_state.inbound_dirty = True


def setup_date_config(
    df: pd.DataFrame,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[int]]:
    """Configure date parameters and lookback period."""
    with st.sidebar.expander("🗓️ **Entry Date & Lookback**", expanded=True):
        # Use reusable date range component
        default_start = datetime(2025, 1, 1)
        default_end = datetime(2025, 1, 31)

        try:
            report_start, report_end = ui.date_range_input(
                label="Entry Date Range",
                default_start=default_start,
                default_end=default_end,
                help_text="Analysis period for new entries",
                info_message="The selected end date will be included in the analysis period.",
                on_change_callback=_set_dirty_flag,
            )

            # Return None values if date input failed
            if report_start is None or report_end is None:
                return None, None, None

        except Exception as e:
            st.error(f"📅 Date Error: {str(e)}")
            return None, None, None

        st.html(html_factory.divider("gradient"))

        days_lookback = st.number_input(
            "🔍 Days Lookback",
            min_value=1,
            value=730,
            help="Number of days prior to entry to consider exits",
            on_change=_set_dirty_flag,
        )

        analysis_start = report_start - pd.Timedelta(days=days_lookback)
        analysis_end = report_end

        if df is not None and not df.empty:
            data_reporting_start = pd.to_datetime(
                df["ReportingPeriodStartDate"].iloc[0]
            )
            data_reporting_end = pd.to_datetime(
                df["ReportingPeriodEndDate"].iloc[0]
            )

            check_date_range_validity(
                analysis_start,
                analysis_end,
                data_reporting_start,
                data_reporting_end,
                df=df,
            )

        return report_start, report_end, days_lookback


def setup_entry_filters(df: pd.DataFrame) -> Tuple[Optional[List[str]], ...]:
    """Configure entry-specific filters."""
    with st.sidebar.expander("📍 **Entry Filters**", expanded=False):
        st.html(
            html_factory.title("Entry Enrollment Criteria", level=4, icon="📝")
        )

        allowed_cocs = create_multiselect_filter(
            "CoC Codes - Entry",
            (
                df["ProgramSetupCoC"].dropna().unique().tolist()
                if "ProgramSetupCoC" in df.columns
                else []
            ),
            default=["ALL"],
            help_text="Filter entries by CoC code",
            key="inbound_entry_cocs_filter",
            on_change=_set_dirty_flag,
            module=INBOUND_MODULE,
        )

        allowed_localcocs = create_multiselect_filter(
            "Local CoC Codes - Entry",
            (
                df["LocalCoCCode"].dropna().unique().tolist()
                if "LocalCoCCode" in df.columns
                else []
            ),
            default=["ALL"],
            help_text="Filter entries by local CoC code",
            on_change=_set_dirty_flag,
        )

        allowed_agencies = create_multiselect_filter(
            "Agencies - Entry",
            (
                df["AgencyName"].dropna().unique().tolist()
                if "AgencyName" in df.columns
                else []
            ),
            default=["ALL"],
            help_text="Filter entries by agency",
            on_change=_set_dirty_flag,
        )

        allowed_programs = create_multiselect_filter(
            "Programs - Entry",
            (
                df["ProgramName"].dropna().unique().tolist()
                if "ProgramName" in df.columns
                else []
            ),
            default=["ALL"],
            help_text="Filter entries by program",
            on_change=_set_dirty_flag,
        )

        # Add SSVF RRH filter for entries
        entry_ssvf_rrh = create_multiselect_filter(
            "SSVF RRH - Entry",
            (
                sorted(df["SSVF_RRH"].dropna().unique().tolist())
                if "SSVF_RRH" in df.columns
                else []
            ),
            default=["ALL"],
            help_text="SSVF RRH filter for entries",
            on_change=_set_dirty_flag,
        )

        # Entry Project Types filter
        entry_project_types = None
        if "ProjectTypeCode" in df.columns:
            all_project_types = sorted(
                df["ProjectTypeCode"].dropna().unique().tolist()
            )
            default_projects = [
                p
                for p in config.analysis.default_project_types
                if p in all_project_types
            ]
            entry_project_types = create_multiselect_filter(
                "Project Types - Entry",
                all_project_types,
                default=default_projects,
                help_text="Filter by project types for entries",
                on_change=_set_dirty_flag,
            )

        return (
            allowed_cocs,
            allowed_localcocs,
            allowed_agencies,
            allowed_programs,
            entry_project_types,
            entry_ssvf_rrh,
        )


def setup_exit_filters(df: pd.DataFrame) -> Tuple[Optional[List[str]], ...]:
    """Configure exit-specific filters."""
    with st.sidebar.expander("🚪 **Exit Filters**", expanded=False):
        st.html(html_factory.title("Prior Exit Criteria", level=4, icon="🚪"))

        allowed_cocs_exit = create_multiselect_filter(
            "CoC Codes - Exit",
            (
                df["ProgramSetupCoC"].dropna().unique().tolist()
                if "ProgramSetupCoC" in df.columns
                else []
            ),
            default=["ALL"],
            help_text="Filter exits by CoC code",
            on_change=_set_dirty_flag,
        )

        allowed_localcocs_exit = create_multiselect_filter(
            "Local CoC Codes - Exit",
            (
                df["LocalCoCCode"].dropna().unique().tolist()
                if "LocalCoCCode" in df.columns
                else []
            ),
            default=["ALL"],
            help_text="Filter exits by local CoC code",
            on_change=_set_dirty_flag,
        )

        allowed_agencies_exit = create_multiselect_filter(
            "Agencies - Exit",
            (
                df["AgencyName"].dropna().unique().tolist()
                if "AgencyName" in df.columns
                else []
            ),
            default=["ALL"],
            help_text="Filter exits by agency",
            on_change=_set_dirty_flag,
        )

        allowed_programs_exit = create_multiselect_filter(
            "Programs - Exit",
            (
                df["ProgramName"].dropna().unique().tolist()
                if "ProgramName" in df.columns
                else []
            ),
            default=["ALL"],
            help_text="Filter exits by program",
            on_change=_set_dirty_flag,
        )

        # Add SSVF RRH filter for exits
        exit_ssvf_rrh = create_multiselect_filter(
            "SSVF RRH - Exit",
            (
                sorted(df["SSVF_RRH"].dropna().unique().tolist())
                if "SSVF_RRH" in df.columns
                else []
            ),
            default=["ALL"],
            help_text="SSVF RRH filter for exits",
            on_change=_set_dirty_flag,
        )

        # Project Types
        exit_project_types = None
        if "ProjectTypeCode" in df.columns:
            all_project_types = sorted(
                df["ProjectTypeCode"].dropna().unique().tolist()
            )
            default_projects = [
                p
                for p in config.analysis.default_project_types
                if p in all_project_types
            ]
            exit_project_types = create_multiselect_filter(
                "Project Types - Exit",
                all_project_types,
                default=default_projects,
                help_text="Filter by project types for exits",
                on_change=_set_dirty_flag,
            )

        # Exit Destination Category filter
        allowed_exit_dest_cats = None
        if "ExitDestinationCat" in df.columns:
            allowed_exit_dest_cats = create_multiselect_filter(
                "Exit Destination Categories",
                sorted(df["ExitDestinationCat"].dropna().unique().tolist()),
                default=["ALL"],
                help_text="Filter exits by destination category (e.g., Permanent Housing Situations)",
                on_change=_set_dirty_flag,
            )

        # Add Exit Destinations filter
        allowed_exit_destinations = None
        if "ExitDestination" in df.columns:
            allowed_exit_destinations = create_multiselect_filter(
                "Exit Destinations",
                sorted(df["ExitDestination"].dropna().unique().tolist()),
                default=["ALL"],
                help_text="Limit exits to these specific destinations",
                on_change=_set_dirty_flag,
            )

        return (
            allowed_cocs_exit,
            allowed_localcocs_exit,
            allowed_agencies_exit,
            allowed_programs_exit,
            exit_project_types,
            exit_ssvf_rrh,
            allowed_exit_dest_cats,
            allowed_exit_destinations,
        )


# ============================================================================
# ANALYSIS EXECUTION
# ============================================================================


def run_analysis(df: pd.DataFrame, analysis_params: Dict[str, Any]) -> bool:
    """Execute the inbound recidivism analysis with specified parameters."""
    try:
        st.session_state.inbound_dirty = False
        with st.status(
            "🔍 Processing Inbound Analysis...", expanded=True
        ) as status:
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
                allowed_localcocs_exit=analysis_params[
                    "allowed_localcocs_exit"
                ],
                allowed_programs_exit=analysis_params["allowed_programs_exit"],
                allowed_agencies_exit=analysis_params["allowed_agencies_exit"],
                exit_project_types=analysis_params["exit_project_types"],
                exit_ssvf_rrh=analysis_params["exit_ssvf_rrh"],
                allowed_exit_dest_cats=analysis_params[
                    "allowed_exit_dest_cats"
                ],
                allowed_exit_destinations=analysis_params[
                    "allowed_exit_destinations"
                ],
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
                "Exit_ReportingPeriodEndDate",
            ]
            merged_df.drop(
                columns=cols_to_remove, inplace=True, errors="ignore"
            )

            # Rename columns
            cols_to_rename = [
                "Enter_UniqueIdentifier",
                "Enter_ClientID",
                "Enter_RaceEthnicity",
                "Enter_Gender",
                "Enter_DOB",
                "Enter_VeteranStatus",
            ]
            mapping = {
                col: col[len("Enter_") :]
                for col in cols_to_rename
                if col in merged_df.columns
            }
            merged_df.rename(columns=mapping, inplace=True)

            status.write("💾 Finalizing results...")

            set_analysis_result("inbound", merged_df)
            status.update(
                label="✅ Inbound Analysis Complete!",
                state="complete",
                expanded=False,
            )

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


def display_summary_metrics(
    final_df: pd.DataFrame, allowed_exit_dest_cats: Optional[List[str]] = None
) -> None:
    """Display core performance metrics with natural styling."""
    st.html(html_factory.divider("gradient"))
    st.html(html_factory.title("Inbound Analysis Summary", level=3, icon="📊"))

    if allowed_exit_dest_cats == ["Permanent Housing Situations"]:
        ui.info_section(
            "Only Permanent Housing Situations is selected in the Exit Destination Categories filter.",
            type="info",
            icon="📌",
            expanded=True,
        )

    # Apply metric card styling
    ui.apply_metric_card_style(
        border_color=NeutralColors.PRIMARY, box_shadow=True
    )

    metrics = compute_return_metrics(final_df)
    display_return_metrics_cards(metrics)


def display_time_to_entry(final_df: pd.DataFrame) -> None:
    """Display time-to-entry distribution visualization."""
    st.html(html_factory.divider("gradient"))
    st.html(
        html_factory.title("Days to Return Distribution", level=3, icon="⏳")
    )

    try:
        fig = plot_time_to_entry_box(final_df)
        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, width='stretch')

        # Display time statistics if available
        # display_time_statistics(final_df)

    except Exception as e:
        st.error(f"📉 Visualization Error: {str(e)}")


@st.fragment
def display_breakdowns(final_df: pd.DataFrame) -> None:
    """Display cohort breakdown analysis."""
    st.html(html_factory.divider("gradient"))
    st.html(html_factory.title("Breakdown Analysis", level=3, icon="📈"))

    # Add the collapsible description with styled info box
    with st.expander("Understanding Breakdown Categories", expanded=False):
        ui.info_section(
            """
        <h4>This table can be grouped by:</h4>

        <p><strong>• Client-level values</strong> (race, gender, etc.): These are demographic factors that are stable for clients across their enrollments.</p>

        <p><strong>• Entry-based values</strong>: Clients are categorized based on information relating to their Inbound Entry.</p>

        <p><strong>• Exit-based values</strong>: Clients are categorized based on information relating to most recent exit prior to their Entry during the reporting period. Because we are grouping by values related to their previous exit, clients who did NOT have a previous exit (new clients) will be excluded from the table.</p>
        """,
            type="info",
            expanded=True,
        )

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

    possible_cols = [
        col for col in breakdown_columns if col in final_df.columns
    ]
    default_breakdown = (
        ["Enter_ProjectTypeCode"]
        if "Enter_ProjectTypeCode" in possible_cols
        else []
    )

    analysis_cols = st.columns([3, 1])

    with analysis_cols[0]:
        chosen = st.multiselect(
            "Group By Dimensions",
            possible_cols,
            default=default_breakdown,
            help="Select grouping columns for analysis",
        )

    if chosen:
        try:
            breakdown = return_breakdown_analysis(final_df, chosen)

            # If any chosen column starts with "exit_", drop the "New (%)"
            # column
            if any(col.lower().startswith("exit_") for col in chosen):
                breakdown = breakdown.drop(
                    columns=["New (%)"], errors="ignore"
                )

            with analysis_cols[1]:
                st.metric("Total Groups", len(breakdown))

            render_dataframe_with_style(
                breakdown,
                highlight_cols=[
                    "Total Entries",
                    "Returning (%)",
                    "Returning From Housing (%)",
                ],
                height=400,
            )
        except Exception as e:
            st.error(f"📊 Breakdown Error: {str(e)}")

    st.html("</div>")


@st.fragment
def display_client_flow(final_df: pd.DataFrame) -> None:
    """Display client flow analysis visualization."""
    st.html(html_factory.divider("gradient"))
    st.html(html_factory.title("Client Flow Analysis", level=3, icon="🌊"))

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

        exit_cols_for_flow = [
            c for c in exit_columns if c in ra_flows_df.columns
        ]
        entry_cols_for_flow = [
            c for c in entry_columns if c in ra_flows_df.columns
        ]

        if exit_cols_for_flow and entry_cols_for_flow:
            # Dimension selectors with info box
            ui.info_section(
                "Both the Exit and Entry Dimension filters apply to the entire flow section.",
                type="info",
                icon="📌",
                expanded=True,
            )

            flow_cols = st.columns(2)
            with flow_cols[0]:
                exit_flow_col = st.selectbox(
                    "Exit Dimension: Rows",
                    exit_cols_for_flow,
                    index=(
                        exit_cols_for_flow.index("Exit_ProjectTypeCode")
                        if "Exit_ProjectTypeCode" in exit_cols_for_flow
                        else 0
                    ),
                    help="Characteristic at prior exit point",
                )
            with flow_cols[1]:
                entry_flow_col = st.selectbox(
                    "Entry Dimension: Columns",
                    entry_cols_for_flow,
                    index=(
                        entry_cols_for_flow.index("Enter_ProjectTypeCode")
                        if "Enter_ProjectTypeCode" in entry_cols_for_flow
                        else 0
                    ),
                    help="Characteristic at current entry point",
                )

            # Build full pivot
            flow_pivot_ra = create_flow_pivot_ra(
                ra_flows_df, exit_flow_col, entry_flow_col
            )

            # Reorder columns if needed
            if "No Data" in flow_pivot_ra.columns:
                cols = [c for c in flow_pivot_ra.columns if c != "No Data"] + [
                    "No Data"
                ]
                flow_pivot_ra = flow_pivot_ra[cols]

            # Flow Matrix with neutral styling
            with st.expander("🔍 **Flow Matrix Details**", expanded=True):
                render_dataframe_with_style(
                    flow_pivot_ra,
                    highlight_cols=[
                        c for c in flow_pivot_ra.columns if c != "No Data"
                    ],
                    axis=1,
                )

            # Top pathways section
            st.html(html_factory.divider("gradient"))
            st.html(
                html_factory.title("Top Client Pathways", level=4, icon="🔝")
            )

            col1, col2 = st.columns([3, 1])
            with col1:
                top_n = st.slider(
                    "Number of Pathways",
                    min_value=5,
                    max_value=25,
                    value=10,
                    help="Top N pathways to display",
                )

            top_flows_df = get_top_flows_from_pivot(flow_pivot_ra, top_n=top_n)
            if not top_flows_df.empty:
                render_dataframe_with_style(
                    top_flows_df,
                    highlight_cols=(
                        ["Count", "Percent"]
                        if "Percent" in top_flows_df.columns
                        else ["Count"]
                    ),
                )
            else:
                st.info("No significant flows detected")

            # Network visualization
            st.html(html_factory.divider("gradient"))
            st.html(
                html_factory.title("Client Flow Network", level=4, icon="🌐")
            )

            ui.info_section(
                "Focus filters below apply only to the network visualization",
                type="warning",
                icon="🎯",
                expanded=True,
            )

            drill_cols = st.columns(2)
            with drill_cols[0]:
                focus_exit = st.selectbox(
                    "🔍 Focus Exit Dimension",
                    ["All"] + flow_pivot_ra.index.tolist(),
                    help="Show only this exit in the network",
                )
            with drill_cols[1]:
                focus_return = st.selectbox(
                    "🔍 Focus Entry Dimension",
                    ["All"] + flow_pivot_ra.columns.tolist(),
                    help="Show only this entry in the network",
                )

            # Create filtered pivot for Sankey
            flow_pivot_sankey = flow_pivot_ra.copy()

            if focus_exit != "All":
                flow_pivot_sankey = flow_pivot_sankey.loc[[focus_exit]]
            if focus_return != "All":
                flow_pivot_sankey = flow_pivot_sankey[[focus_return]]

            # Generate Sankey with themed styling
            sankey_ra = plot_flow_sankey_ra(
                flow_pivot_sankey, f"{exit_flow_col} → {entry_flow_col}"
            )
            sankey_ra = apply_chart_theme(sankey_ra)
            st.plotly_chart(sankey_ra, width='stretch')
        else:
            st.info("📭 Insufficient data for flow analysis")

    except Exception as e:
        st.error(f"🌊 Flow Error: {str(e)}")


def display_data_export(final_df: pd.DataFrame) -> None:
    """Display data export options."""
    st.html(html_factory.divider("gradient"))
    st.html(html_factory.title("Data Export", level=3, icon="📤"))

    # Export section with styled card
    st.html('<div class="neutral-card">')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_download_button(
            final_df,
            filename="inbound_recidivism_analysis.csv",
            label="📥 Download Inbound Data",
        )
    st.html("</div>")


# ============================================================================
# MAIN PAGE FUNCTION
# ============================================================================


def inbound_recidivism_page() -> None:
    """Render the Inbound Recidivism Analysis page with all components."""
    # Apply custom CSS theme
    apply_custom_css()
    st.session_state.setdefault("inbound_dirty", False)

    st.html(
        html_factory.title("Inbound Recidivism Analysis", level=1, icon="📈")
    )

    # Display the about section
    render_about_section(
        title="About Inbound Recidivism Analysis",
        content=ABOUT_INBOUND_CONTENT,
        expanded=False,
        icon="📥",
    )

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
    (
        allowed_cocs,
        allowed_localcocs,
        allowed_agencies,
        allowed_programs,
        entry_project_types,
        entry_ssvf_rrh,
    ) = entry_filters

    exit_filters = setup_exit_filters(df)
    (
        allowed_cocs_exit,
        allowed_localcocs_exit,
        allowed_agencies_exit,
        allowed_programs_exit,
        exit_project_types,
        exit_ssvf_rrh,
        allowed_exit_dest_cats,
        allowed_exit_destinations,
    ) = exit_filters

    st.sidebar.html("</div>")

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
        "allowed_exit_destinations": allowed_exit_destinations,
    }

    # Run analysis section
    st.html(html_factory.divider("gradient"))
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.get("inbound_dirty"):
            st.info("Parameters have changed. Click 'Run Analysis' to update.")
        if st.button(
            "▶️ Run Inbound Analysis", type="primary", width='stretch'
        ):
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
