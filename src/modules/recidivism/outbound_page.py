"""
Outbound Recidivism Analysis Page
---------------------------------
Renders the outbound recidivism analysis interface and orchestrates the workflow.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.core.constants import DEFAULT_PROJECT_TYPES
from src.core.session import (
    ModuleType,
    get_analysis_result,
    get_outbound_state,
    get_session_manager,
    set_analysis_result,
)

# Widget persistence now handled by enhanced session system
from src.core.utils.helpers import (
    check_date_range_validity,
    create_multiselect_filter,
)
from src.modules.recidivism.outbound_calculator import (
    breakdown_by_columns,
    compute_summary_metrics,
    run_outbound_recidivism,
)
from src.modules.recidivism.outbound_viz import (
    create_flow_pivot,
    display_spm_metrics_non_ph,
    display_spm_metrics_ph,
    get_top_flows_from_pivot,
    plot_days_to_return_box,
    plot_flow_sankey,
)
from src.ui.factories.components import (
    render_about_section,
    render_dataframe_with_style,
    render_download_button,
    ui,
)
from src.ui.factories.html import html_factory
from src.ui.layouts.templates import ABOUT_OUTBOUND_CONTENT
from src.ui.themes.styles import (
    NeutralColors,
    apply_chart_theme,
    apply_custom_css,
    create_styled_divider,
)

# ============================================================================
# CONSTANTS
# ============================================================================

# Enhanced session management
session_manager = get_session_manager()
outbound_state = get_outbound_state()
OUTBOUND_MODULE = ModuleType.OUTBOUND

# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================


def _set_dirty_flag():
    # Deprecated - use check_and_mark_dirty for value-aware dirty checking
    outbound_state.mark_dirty()


def setup_reporting_period(
    df: pd.DataFrame,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Configure the reporting period dates."""
    with st.sidebar.expander("📅 **Reporting Period**", expanded=True):
        # Set default values
        # Use reusable date range component
        default_start = datetime(2025, 1, 1)
        default_end = datetime(2025, 1, 31)

        # Get saved date range from enhanced session system
        saved_dates = outbound_state.get_widget_state("date_range", None)
        if saved_dates and len(saved_dates) == 2:
            default_start = saved_dates[0]
            default_end = saved_dates[1]

        def _save_dates():
            # The date_range_input function will trigger this callback when dates change
            # We need to save the current dates to enhanced session system
            _set_dirty_flag()

        report_start, report_end = ui.date_range_input(
            label="Exit Date Range",
            default_start=default_start,
            default_end=default_end,
            help_text="Clients must have an exit date inside this window.",
            info_message="The selected end date will be included in the analysis period.",
            on_change_callback=_save_dates,
        )

        # Save the selected dates to enhanced session system after selection
        if report_start and report_end:
            outbound_state.set_widget_state(
                "date_range", [report_start, report_end]
            )

        # Use defaults if date input failed
        if report_start is None or report_end is None:
            report_start = pd.to_datetime(default_start)
            report_end = pd.to_datetime(default_end)

        # Ensure start is before end
        if report_start > report_end:
            st.error(
                "❌ Start date must be before end date. Dates have been swapped."
            )
            report_start, report_end = report_end, report_start

        if df is not None and not df.empty:
            data_reporting_start = pd.to_datetime(
                df["ReportingPeriodStartDate"].iloc[0]
            )
            data_reporting_end = pd.to_datetime(
                df["ReportingPeriodEndDate"].iloc[0]
            )

            check_date_range_validity(
                report_start,
                report_end,
                data_reporting_start,
                data_reporting_end,
                df=df,
            )

        return report_start, report_end


def setup_exit_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Configure exit-specific filters."""
    with st.sidebar.expander("🚪 **Exit Filters**", expanded=False):
        st.html(
            html_factory.title("Exit Enrollment Criteria", level=4, icon="🚪")
        )

        exit_cocs = create_multiselect_filter(
            "CoC Codes",
            (
                df["ProgramSetupCoC"].dropna().unique()
                if "ProgramSetupCoC" in df.columns
                else []
            ),
            default=None,
            help_text="Filter exits by CoC code",
            key="outbound_exit_cocs",
            module=OUTBOUND_MODULE,
            on_change=lambda: outbound_state.mark_dirty(),
        )

        exit_localcocs = create_multiselect_filter(
            "Local CoC",
            (
                df["LocalCoCCode"].dropna().unique()
                if "LocalCoCCode" in df.columns
                else []
            ),
            default=None,
            help_text="Filter exits by local CoC",
            key="outbound_exit_localcocs",
            module=OUTBOUND_MODULE,
            on_change=lambda: outbound_state.mark_dirty(),
        )

        exit_agencies = create_multiselect_filter(
            "Agencies",
            (
                df["AgencyName"].dropna().unique()
                if "AgencyName" in df.columns
                else []
            ),
            default=None,
            help_text="Filter exits by agency",
            key="outbound_exit_agencies",
            module=OUTBOUND_MODULE,
            on_change=lambda: outbound_state.mark_dirty(),
        )

        exit_programs = create_multiselect_filter(
            "Programs",
            (
                df["ProgramName"].dropna().unique()
                if "ProgramName" in df.columns
                else []
            ),
            default=None,
            help_text="Filter exits by program",
            key="outbound_exit_programs",
            module=OUTBOUND_MODULE,
            on_change=lambda: outbound_state.mark_dirty(),
        )

        exit_ssvf_rrh = create_multiselect_filter(
            "SSVF RRH",
            (
                sorted(df["SSVF_RRH"].dropna().unique().tolist())
                if "SSVF_RRH" in df.columns
                else []
            ),
            default=None,
            help_text="SSVF RRH filter for exits",
            key="outbound_exit_ssvf_rrh",
            module=OUTBOUND_MODULE,
            on_change=lambda: outbound_state.mark_dirty(),
        )

        all_project_types = sorted(
            df["ProjectTypeCode"].dropna().unique().tolist()
        )
        default_projects = [
            p for p in DEFAULT_PROJECT_TYPES if p in all_project_types
        ]
        # Get saved project types from enhanced session system
        saved_projects = outbound_state.get_widget_state(
            "exiting_projects", default_projects
        )

        def _save_projects():
            value = st.session_state.get(
                "outbound_exiting_projects", default_projects
            )
            outbound_state.set_widget_state("exiting_projects", value)
            outbound_state.check_and_mark_dirty("exiting_projects", value)

        exiting_projects = st.multiselect(
            "Project Types (Exit)",
            all_project_types,
            default=saved_projects,
            help="Project types treated as exits",
            key="outbound_exiting_projects",
            on_change=_save_projects,
        )

        allowed_exit_dest_cats = None
        if "ExitDestinationCat" in df.columns:
            # Get saved exit destination categories from enhanced session system
            saved_exit_dest_cats = outbound_state.get_widget_state(
                "exit_dest_cats", ["Permanent Housing Situations"]
            )

            def _save_exit_dest_cats():
                value = st.session_state.get(
                    "outbound_exit_dest_cats",
                    ["Permanent Housing Situations"],
                )
                outbound_state.set_widget_state("exit_dest_cats", value)
                outbound_state.check_and_mark_dirty("exit_dest_cats", value)

            allowed_exit_dest_cats = st.multiselect(
                "Exit Destination Categories",
                sorted(df["ExitDestinationCat"].dropna().unique()),
                default=saved_exit_dest_cats,
                help="Limit exits to these destination categories",
                key="outbound_exit_dest_cats",
                on_change=_save_exit_dest_cats,
            )

        allowed_exit_destinations = None
        if "ExitDestination" in df.columns:
            allowed_exit_destinations = create_multiselect_filter(
                "Exit Destinations",
                sorted(df["ExitDestination"].dropna().unique().tolist()),
                default=["ALL"],
                help_text="Limit exits to these specific destinations",
                key="outbound_exit_destinations",
                module=OUTBOUND_MODULE,
                on_change=lambda: outbound_state.mark_dirty(),
            )

        return {
            "exit_cocs": exit_cocs,
            "exit_localcocs": exit_localcocs,
            "exit_agencies": exit_agencies,
            "exit_programs": exit_programs,
            "exit_ssvf_rrh": exit_ssvf_rrh,
            "exiting_projects": exiting_projects,
            "allowed_exit_dest_cats": allowed_exit_dest_cats,
            "allowed_exit_destinations": allowed_exit_destinations,
        }


def setup_return_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Configure return-specific filters."""
    with st.sidebar.expander("↩️ **Return Filters**", expanded=False):
        st.html(
            html_factory.title("Return Enrollment Criteria", level=4, icon="🔄")
        )

        return_cocs = create_multiselect_filter(
            "CoC Codes",
            (
                df["ProgramSetupCoC"].dropna().unique()
                if "ProgramSetupCoC" in df.columns
                else []
            ),
            default=None,
            help_text="Filter next enrollments by CoC code",
            key="outbound_return_cocs",
            module=OUTBOUND_MODULE,
            on_change=lambda: outbound_state.mark_dirty(),
        )

        return_localcocs = create_multiselect_filter(
            "Local CoC",
            (
                df["LocalCoCCode"].dropna().unique()
                if "LocalCoCCode" in df.columns
                else []
            ),
            default=None,
            help_text="Filter next enrollments by local CoC",
            key="outbound_return_localcocs",
            module=OUTBOUND_MODULE,
            on_change=lambda: outbound_state.mark_dirty(),
        )

        return_agencies = create_multiselect_filter(
            "Agencies",
            (
                df["AgencyName"].dropna().unique()
                if "AgencyName" in df.columns
                else []
            ),
            default=None,
            help_text="Filter next enrollments by agency",
            key="outbound_return_agencies",
            module=OUTBOUND_MODULE,
            on_change=lambda: outbound_state.mark_dirty(),
        )

        return_programs = create_multiselect_filter(
            "Programs",
            (
                df["ProgramName"].dropna().unique()
                if "ProgramName" in df.columns
                else []
            ),
            default=None,
            help_text="Filter next enrollments by program",
            key="outbound_return_programs",
            module=OUTBOUND_MODULE,
            on_change=lambda: outbound_state.mark_dirty(),
        )

        return_ssvf_rrh = create_multiselect_filter(
            "SSVF RRH",
            (
                sorted(df["SSVF_RRH"].dropna().unique().tolist())
                if "SSVF_RRH" in df.columns
                else []
            ),
            default=None,
            help_text="SSVF RRH filter for returns",
            key="outbound_return_ssvf_rrh",
            module=OUTBOUND_MODULE,
            on_change=lambda: outbound_state.mark_dirty(),
        )

        all_project_types = sorted(
            df["ProjectTypeCode"].dropna().unique().tolist()
        )
        default_projects = [
            p for p in DEFAULT_PROJECT_TYPES if p in all_project_types
        ]
        # Get saved return project types from enhanced session system
        saved_return_projects = outbound_state.get_widget_state(
            "return_projects", default_projects
        )

        def _save_return_projects():
            value = st.session_state.get(
                "outbound_return_projects", default_projects
            )
            outbound_state.set_widget_state("return_projects", value)
            outbound_state.check_and_mark_dirty("return_projects", value)

        return_projects = st.multiselect(
            "Project Types (Return)",
            all_project_types,
            default=saved_return_projects,
            help="Project types treated as candidate returns",
            key="outbound_return_projects",
            on_change=_save_return_projects,
        )

        return {
            "return_cocs": return_cocs,
            "return_localcocs": return_localcocs,
            "return_agencies": return_agencies,
            "return_programs": return_programs,
            "return_ssvf_rrh": return_ssvf_rrh,
            "return_projects": return_projects,
        }


def setup_continuum_filters(df: pd.DataFrame) -> Optional[List[str]]:
    """Configure continuum project filters."""
    with st.sidebar.expander("⚡ **Continuum Projects**", expanded=False):
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
            key="outbound_continuum_projects",
            module=OUTBOUND_MODULE,
            on_change=lambda: outbound_state.mark_dirty(),
        )


# ============================================================================
# ANALYSIS EXECUTION
# ============================================================================


def run_analysis(df: pd.DataFrame, analysis_params: Dict[str, Any]) -> bool:
    """Run the outbound recidivism analysis with the provided parameters."""
    try:
        outbound_state.request_analysis()
        with st.status(
            "🔄 Running Outbound Recidivism Analysis…", expanded=True
        ) as status:
            status.write("📊 Identifying exits in the reporting period...")

            outbound_df = run_outbound_recidivism(
                df,
                analysis_params["report_start"],
                analysis_params["report_end"],
                exit_cocs=analysis_params["exit_cocs"],
                exit_localcocs=analysis_params["exit_localcocs"],
                exit_agencies=analysis_params["exit_agencies"],
                exit_programs=analysis_params["exit_programs"],
                exit_ssvf_rrh=analysis_params["exit_ssvf_rrh"],
                return_cocs=analysis_params["return_cocs"],
                return_localcocs=analysis_params["return_localcocs"],
                return_agencies=analysis_params["return_agencies"],
                return_programs=analysis_params["return_programs"],
                return_ssvf_rrh=analysis_params["return_ssvf_rrh"],
                allowed_continuum=analysis_params["chosen_continuum"],
                allowed_exit_dest_cats=analysis_params[
                    "allowed_exit_dest_cats"
                ],
                allowed_exit_destinations=analysis_params[
                    "allowed_exit_destinations"
                ],
                exiting_projects=analysis_params["exiting_projects"],
                return_projects=analysis_params["return_projects"],
            )

            status.write("🔄 Matching subsequent enrollments...")

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
                "Return_ReportingPeriodEndDate",
            ]
            outbound_df.drop(
                columns=cols_to_remove, inplace=True, errors="ignore"
            )

            cols_to_rename = [
                "Exit_UniqueIdentifier",
                "Exit_ClientID",
                "Exit_RaceEthnicity",
                "Exit_Gender",
                "Exit_DOB",
                "Exit_VeteranStatus",
            ]
            mapping = {
                col: col[len("Exit_") :]
                for col in cols_to_rename
                if col in outbound_df.columns
            }
            outbound_df.rename(columns=mapping, inplace=True)

            status.write("💾 Finalizing results...")

            set_analysis_result(OUTBOUND_MODULE, outbound_df)

            # Clear analysis request and save current params as new baseline
            outbound_state.clear_analysis_request()

            # Explicitly save all current parameter values as baseline
            outbound_state.save_params_snapshot()

            status.update(
                label="✅ Outbound Analysis Complete!",
                state="complete",
                expanded=False,
            )

        st.toast("🎉 Analysis complete!", icon="✅")
        return True

    except Exception as exc:
        st.error(f"🚨 Error: {exc}")
        return False


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================


def display_summary_metrics(
    out_df: pd.DataFrame, allowed_exit_dest_cats: Optional[List[str]] = None
) -> None:
    """Display the core performance metrics summary with natural styling."""
    st.html(html_factory.divider("gradient"))
    st.html(html_factory.title("Outbound Analysis Summary", level=3, icon="📊"))

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

    # Compute metrics
    metrics = compute_summary_metrics(out_df)

    # Row 1: Exit Overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Relevant Exits",
            value=f"{metrics['Number of Relevant Exits']:,}",
        )

    with col2:
        st.metric(
            label="Exits to Permanent Housing",
            value=f"{metrics['Total Exits to PH']:,}",
            help="Number of exits to permanent housing destinations",
        )

    with col3:
        ph_exit_rate = (
            (
                metrics["Total Exits to PH"]
                / metrics["Number of Relevant Exits"]
                * 100
            )
            if metrics["Number of Relevant Exits"] > 0
            else 0
        )
        st.metric(
            label="PH Exit Rate",
            value=f"{ph_exit_rate:.1f}%",
            help="Percentage of exits going to permanent housing",
        )

    # Add spacing
    st.html(html_factory.divider("gradient"))

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
            help="Returns to homelessness from permanent housing exits only",
        )

    with col3:
        st.metric(
            label="Return to Homelessness Rate (From PH)",
            value=f"{metrics['% Return to Homelessness']:.1f}%",
            help="Percentage of PH exits that return to homelessness",
        )

    # Row 3: Timing Analysis
    if metrics["Return"] > 0:  # Only show timing if there are returns
        st.html(html_factory.divider("gradient"))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Median Days to Return",
                value=f"{metrics['Median Days']:.0f}",
                help="Middle value of days between exit and return",
            )

        with col2:
            st.metric(
                label="Average Days to Return",
                value=f"{metrics['Average Days']:.0f}",
                help="Mean number of days between exit and return",
            )

        with col3:
            st.metric(
                label="Maximum Days to Return",
                value=f"{metrics['Max Days']:.0f}",
                help="Longest time between exit and return",
            )


def display_days_to_return(out_df: pd.DataFrame) -> None:
    """Display the days-to-return distribution visualization."""
    st.html(html_factory.divider("gradient"))
    st.html(
        html_factory.title("Days to Return Distribution", level=3, icon="⏳")
    )

    fig = plot_days_to_return_box(out_df)
    fig = apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


@st.fragment
def display_breakdown_analysis(out_df: pd.DataFrame) -> None:
    """Display the breakdown analysis by selected dimensions."""
    st.html(html_factory.divider("gradient"))
    st.html(html_factory.title("Breakdown Analysis", level=3, icon="📈"))

    breakdown_columns = [
        # Client Demographics
        "RaceEthnicity",
        "Gender",
        "VeteranStatus",
        "AgeAtExitRange",
        # Exit Client Characteristics
        "Exit_HasIncome",
        "Exit_HasDisability",
        "Exit_HouseholdType",
        "Exit_IsHeadOfHousehold",
        "Exit_CHStartHousehold",
        "Exit_CurrentlyFleeingDV",
        "Exit_PriorLivingCat",
        "Exit_AgeTieratEntry",
        # Exit Program Information
        "Exit_LocalCoCCode",
        "Exit_ProgramSetupCoC",
        "Exit_ProjectTypeCode",
        "Exit_AgencyName",
        "Exit_ProgramName",
        "Exit_SSVF_RRH",
        "Exit_ProgramsContinuumProject",
        "Exit_CustomProgramType",
        # Exit Destination
        "Exit_ExitDestinationCat",
        "Exit_ExitDestination",
        # Return Characteristics (if they returned)
        "Return_HasIncome",
        "Return_HasDisability",
        "Return_HouseholdType",
        "Return_IsHeadOfHousehold",
        "Return_CHStartHousehold",
        "Return_CurrentlyFleeingDV",
        "Return_PriorLivingCat",
        # Return Program Information
        "Return_LocalCoCCode",
        "Return_ProgramSetupCoC",
        "Return_ProjectTypeCode",
        "Return_AgencyName",
        "Return_ProgramName",
        "Return_SSVF_RRH",
        "Return_ProgramsContinuumProject",
        # Return Status
        "Return_ExitDestinationCat",
        "Return_ExitDestination",
        # Analysis Fields
        "PH_Exit",  # Whether exit was to permanent housing
        "HasReturn",  # Whether client had any return
        "ReturnToHomelessness",  # Whether return qualified as return to homelessness
    ]

    available_cols = [
        col for col in breakdown_columns if col in out_df.columns
    ]
    default_breakdown = (
        ["Exit_ProjectTypeCode"]
        if "Exit_ProjectTypeCode" in available_cols
        else []
    )

    analysis_cols = st.columns([3, 1])

    with analysis_cols[0]:
        cols_to_group = st.multiselect(
            "Group By Dimensions",
            options=available_cols,
            default=default_breakdown,
            help="Select up to 3 columns for breakdown analysis",
        )

    if cols_to_group:
        bdf = breakdown_by_columns(out_df, cols_to_group[:3])

        with analysis_cols[1]:
            st.metric("Total Groups", len(bdf))

        render_dataframe_with_style(
            bdf,
            highlight_cols=["Relevant Exits", "Return", "% Return"],
            height=400,
        )

    st.html("</div>")


@st.fragment
def display_client_flow(out_df: pd.DataFrame) -> None:
    """Display client flow analysis visualization."""
    st.html(html_factory.divider("gradient"))
    st.html(html_factory.title("Client Flow Analysis", level=3, icon="🌊"))

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
    ]
    return_columns = [
        "Return_HasIncome",
        "Return_HasDisability",
        "Return_HouseholdType",
        "Return_IsHeadOfHousehold",
        "Return_CHStartHousehold",
        "Return_CurrentlyFleeingDV",
        "Return_LocalCoCCode",
        "Return_PriorLivingCat",
        "Return_ProgramSetupCoC",
        "Return_ProjectTypeCode",
        "Return_AgencyName",
        "Return_ProgramName",
        "Return_SSVF_RRH",
        "Return_ProgramsContinuumProject",
        "Return_ExitDestinationCat",
        "Return_ExitDestination",
    ]

    exit_dims = [c for c in exit_columns if c in out_df.columns]
    ret_dims = [c for c in return_columns if c in out_df.columns]

    if exit_dims and ret_dims:
        # Dimension selectors with info box
        ui.info_section(
            "Both the Exit and Return Dimension filters apply to the entire flow section.",
            type="info",
            icon="📌",
            expanded=True,
        )

        flow_cols = st.columns(2)
        with flow_cols[0]:
            ex_choice = st.selectbox(
                "Exit Dimension: Rows",
                exit_dims,
                index=(
                    exit_dims.index("Exit_ProjectTypeCode")
                    if "Exit_ProjectTypeCode" in exit_dims
                    else 0
                ),
                help="Characteristic at exit point",
            )
        with flow_cols[1]:
            ret_choice = st.selectbox(
                "Return Dimension: Columns",
                ret_dims,
                index=(
                    ret_dims.index("Return_ProjectTypeCode")
                    if "Return_ProjectTypeCode" in ret_dims
                    else 0
                ),
                help="Characteristic at return point",
            )

        # Build the full pivot
        pivot_c = create_flow_pivot(out_df, ex_choice, ret_choice)

        if "No Return" in pivot_c.columns:
            cols_order = [c for c in pivot_c.columns if c != "No Return"] + [
                "No Return"
            ]
            pivot_c = pivot_c[cols_order]

        # Flow Matrix Details with neutral styling
        with st.expander("🔍 **Flow Matrix Details**", expanded=True):
            if pivot_c.empty:
                st.info("📭 No return enrollments to build flow.")
            else:
                cols_to_color = [
                    c for c in pivot_c.columns if c != "No Return"
                ]
                render_dataframe_with_style(
                    pivot_c, highlight_cols=cols_to_color, axis=1
                )

        # Top Client Pathways section
        st.html(html_factory.divider("gradient"))
        st.html(html_factory.title("Top Client Pathways", level=4, icon="🔝"))

        # Check if pivot table has enough data
        if pivot_c.empty or pivot_c.sum().sum() == 0:
            st.info("No significant pathways detected")
        else:
            non_zero_cells = (pivot_c > 0).sum().sum()

            if non_zero_cells <= 5:
                plural = "s" if non_zero_cells != 1 else ""
                st.info(f"Only {non_zero_cells} pathway{plural} detected")
                try:
                    top_flows_df = get_top_flows_from_pivot(
                        pivot_c, top_n=int(non_zero_cells)
                    )
                    if not top_flows_df.empty:
                        render_dataframe_with_style(
                            top_flows_df,
                            highlight_cols=(
                                ["Count", "Percent"]
                                if "Percent" in top_flows_df.columns
                                else ["Count"]
                            ),
                        )
                except Exception as e:
                    st.error(f"Error generating flows: {str(e)}")
            else:
                # We have enough pathways for a slider
                col1, col2 = st.columns([3, 1])
                with col1:
                    min_flows = 1
                    max_flows = min(25, non_zero_cells)
                    default_flows = min(3, max_flows)

                    top_n = st.slider(
                        "Number of Pathways",
                        min_value=min_flows,
                        max_value=max_flows,
                        value=default_flows,
                        step=1,
                        help="Top N pathways to display",
                    )

                try:
                    top_flows_df = get_top_flows_from_pivot(
                        pivot_c, top_n=top_n
                    )

                    if top_flows_df.empty:
                        st.info("No significant pathways detected")
                    elif "Count" not in top_flows_df.columns:
                        st.info(
                            "Insufficient data to create meaningful flow paths"
                        )
                        render_dataframe_with_style(top_flows_df)
                    else:
                        render_dataframe_with_style(
                            top_flows_df,
                            highlight_cols=(
                                ["Count", "Percent"]
                                if "Percent" in top_flows_df.columns
                                else ["Count"]
                            ),
                        )
                except Exception as e:
                    st.error(f"Error generating top flows: {str(e)}")
                    st.info(
                        "Unable to display top client pathways due to insufficient data"
                    )

        # Client Flow Network with focus controls
        st.html(html_factory.divider("gradient"))
        st.html(html_factory.title("Client Flow Network", level=4, icon="🌐"))

        ui.info_section(
            "Focus filters below apply only to the network visualization",
            type="warning",
            icon="🎯",
            expanded=True,
        )

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

        # Create filtered pivot for Sankey
        pivot_sankey = pivot_c.copy()

        if focus_exit != "All":
            pivot_sankey = pivot_sankey.loc[[focus_exit]]
        if focus_return != "All":
            pivot_sankey = pivot_sankey[[focus_return]]

        # Generate Sankey with themed styling
        sankey_fig = plot_flow_sankey(
            pivot_sankey, f"{ex_choice} → {ret_choice}"
        )
        sankey_fig = apply_chart_theme(sankey_fig)
        st.plotly_chart(sankey_fig, use_container_width=True)
    else:
        st.info("📭 Insufficient data for flow analysis")


def display_ph_comparison(out_df: pd.DataFrame) -> None:
    """Display PH vs. Non-PH exit comparison."""
    st.html(html_factory.divider("gradient"))
    st.html(
        html_factory.title("PH vs. Non-PH Exit Comparison", level=3, icon="🏠")
    )

    if st.checkbox("Show comparison", value=False):
        ph_df = out_df[out_df["PH_Exit"]]
        nonph_df = out_df[~out_df["PH_Exit"]]

        # Create columns with proper spacing
        c1, spacer, c2 = st.columns([5, 0.2, 5])

        with c1:
            st.html(
                html_factory.title(
                    "Permanent Housing Exits", level=2, icon="🏠"
                )
            )
            if not ph_df.empty:
                display_spm_metrics_ph(compute_summary_metrics(ph_df))
            else:
                st.info("No PH exits found.")

        with spacer:
            st.empty()

        with c2:
            st.html(
                html_factory.title(
                    "Non-Permanent Housing Exits", level=2, icon="🏕️"
                )
            )
            if not nonph_df.empty:
                display_spm_metrics_non_ph(compute_summary_metrics(nonph_df))
            else:
                st.info("No Non‑PH exits found.")


def display_data_export(out_df: pd.DataFrame) -> None:
    """Display data export options."""
    st.html(html_factory.divider("gradient"))
    st.html(html_factory.title("Data Export", level=3, icon="📤"))

    # Export section with styled card
    st.html('<div class="neutral-card">')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_download_button(
            out_df,
            filename="outbound_recidivism_results.csv",
            label="📥 Download Outbound Data",
        )
    st.html("</div>")


# ============================================================================
# MAIN PAGE FUNCTION
# ============================================================================


def outbound_recidivism_page() -> None:
    """Render the Outbound Recidivism page with all components."""
    # Initialize enhanced session management
    outbound_state.initialize()

    # Initialize parameter baseline on first load to prevent false dirty flags
    # This ensures widgets created for the first time don't trigger "changed" state
    if not st.session_state.get(f"{outbound_state.key_prefix}last_params"):
        outbound_state.save_params_snapshot()

    # Apply custom CSS theme
    apply_custom_css()

    st.html(
        html_factory.title("Outbound Recidivism Analysis", level=1, icon="📈")
    )

    # Display the about section
    render_about_section(
        title="About Outbound Recidivism Analysis",
        content=ABOUT_OUTBOUND_CONTENT,
        expanded=False,
        icon="📤",
    )

    # Check data availability using enhanced session manager
    if not session_manager.has_data():
        st.warning("Please upload data using the sidebar to begin analysis.")
        return

    df = session_manager.get_data()
    if df is None:
        st.error("Data is not available. Please check your upload.")
        return

    # Sidebar configuration with themed styling
    st.sidebar.html('<div class="sidebar-content">')
    st.sidebar.header("⚙️ Analysis Parameters")
    st.sidebar.html(create_styled_divider())

    # Setup filters and parameters
    report_start, report_end = setup_reporting_period(df)
    exit_filters = setup_exit_filters(df)
    return_filters = setup_return_filters(df)
    chosen_continuum = setup_continuum_filters(df)

    st.sidebar.html("</div>")

    # Prepare analysis parameters
    analysis_params = {
        "report_start": report_start,
        "report_end": report_end,
        "exit_cocs": exit_filters["exit_cocs"],
        "exit_localcocs": exit_filters["exit_localcocs"],
        "exit_agencies": exit_filters["exit_agencies"],
        "exit_programs": exit_filters["exit_programs"],
        "exit_ssvf_rrh": exit_filters["exit_ssvf_rrh"],
        "exiting_projects": exit_filters["exiting_projects"],
        "allowed_exit_dest_cats": exit_filters["allowed_exit_dest_cats"],
        "allowed_exit_destinations": exit_filters["allowed_exit_destinations"],
        "return_cocs": return_filters["return_cocs"],
        "return_localcocs": return_filters["return_localcocs"],
        "return_agencies": return_filters["return_agencies"],
        "return_programs": return_filters["return_programs"],
        "return_ssvf_rrh": return_filters["return_ssvf_rrh"],
        "return_projects": return_filters["return_projects"],
        "chosen_continuum": chosen_continuum,
    }

    # Run analysis section
    st.html(html_factory.divider("gradient"))
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if outbound_state.is_dirty():
            st.info("Parameters have changed. Click 'Run Analysis' to update.")
        if st.button(
            "▶️ Run Outbound Analysis",
            type="primary",
            width="stretch",
        ):
            run_analysis(df, analysis_params)

    # Display results if analysis was successful
    out_df = get_analysis_result(OUTBOUND_MODULE)
    if out_df is not None and not out_df.empty:
        # Display all analysis sections with themed styling
        display_summary_metrics(out_df, exit_filters["allowed_exit_dest_cats"])
        display_days_to_return(out_df)
        display_breakdown_analysis(out_df)
        display_client_flow(out_df)
        display_ph_comparison(out_df)
        display_data_export(out_df)


if __name__ == "__main__":
    outbound_recidivism_page()
