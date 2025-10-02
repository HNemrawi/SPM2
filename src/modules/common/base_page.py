"""
Base classes for common module patterns.

This module provides shared functionality that was previously duplicated across
module pages, including date configuration, filter setup, and common UI patterns.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.core.utils.helpers import (
    check_date_range_validity,
    create_multiselect_filter,
)
from src.ui.factories.components import ui
from src.ui.factories.html import html_factory


class BaseDateConfig:
    """
    Common date configuration patterns used across modules.

    This centralizes the date input logic that was duplicated across
    SPM2, Inbound, and Outbound modules.
    """

    @staticmethod
    def render_date_range_input(
        title: str,
        icon: str,
        default_start: datetime,
        default_end: datetime,
        help_text: str,
        info_message: str,
        state_manager: Any,
        expanded: bool = True,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Render a standardized date range input with session management.

        Args:
            title: Title for the expander section
            icon: Emoji icon for the section
            default_start: Default start date
            default_end: Default end date
            help_text: Help text for the date input
            info_message: Info message to display
            state_manager: State manager instance for persistence
            expanded: Whether section is expanded by default
            df: Optional dataframe for validation

        Returns:
            Tuple of (start_date, end_date) or (None, None) if invalid
        """
        with st.sidebar.expander(f"{icon} **{title}**", expanded=expanded):
            try:
                # Get saved dates from state if available
                saved_dates = getattr(
                    state_manager, "get_date_range", lambda: (None, None)
                )()
                if saved_dates and all(saved_dates):
                    default_start, default_end = saved_dates

                def _save_dates():
                    if hasattr(state_manager, "check_and_mark_dirty"):
                        state_manager.check_and_mark_dirty(
                            "date_range", (default_start, default_end)
                        )

                report_start, report_end = ui.date_range_input(
                    label=title,
                    default_start=default_start,
                    default_end=default_end,
                    help_text=help_text,
                    info_message=info_message,
                    on_change_callback=_save_dates,
                )

                # Save to state manager
                if (
                    report_start
                    and report_end
                    and hasattr(state_manager, "set_date_range")
                ):
                    state_manager.set_date_range(report_start, report_end)

                # Validate against data if provided
                if (
                    df is not None
                    and not df.empty
                    and report_start
                    and report_end
                ):
                    if (
                        "ReportingPeriodStartDate" in df.columns
                        and "ReportingPeriodEndDate" in df.columns
                    ):
                        data_start = pd.to_datetime(
                            df["ReportingPeriodStartDate"].iloc[0]
                        )
                        data_end = pd.to_datetime(
                            df["ReportingPeriodEndDate"].iloc[0]
                        )
                        check_date_range_validity(
                            report_start, report_end, data_start, data_end, df
                        )

                return report_start, report_end

            except Exception as e:
                st.error(f"📅 Date Error: {str(e)}")
                return None, None

    @staticmethod
    def render_lookback_config(
        state_manager: Any,
        units: List[str] = None,
        default_days: int = 730,
        default_months: int = 24,
    ) -> Tuple[int, str]:
        """
        Render standardized lookback period configuration.

        Args:
            state_manager: State manager instance
            units: Available units (default: ["Days", "Months"])
            default_days: Default days value
            default_months: Default months value

        Returns:
            Tuple of (lookback_value, unit_choice)
        """
        if units is None:
            units = ["Days", "Months"]

        # Get saved configuration
        saved_lookback = getattr(
            state_manager,
            "get_lookback_period",
            lambda: {"period": default_days, "unit": "Days"},
        )()

        unit_choice = st.radio(
            "Select Lookback Unit",
            options=units,
            index=0 if saved_lookback["unit"] == units[0] else 1,
            help=f"Choose whether to specify the lookback period in {units[0].lower()} or {units[1].lower()}.",
            key=f"{state_manager.__class__.__name__.lower()}_lookback_unit",
        )

        if unit_choice == "Days":
            lookback_value = st.number_input(
                "Lookback Days",
                min_value=1,
                value=(
                    saved_lookback["period"]
                    if saved_lookback["unit"] == "Days"
                    else default_days
                ),
                help="Days prior to analysis period",
                key=f"{state_manager.__class__.__name__.lower()}_lookback_days",
            )
        else:
            lookback_value = st.number_input(
                "Lookback Months",
                min_value=1,
                value=(
                    saved_lookback["period"]
                    if saved_lookback["unit"] == "Months"
                    else default_months
                ),
                help="Months prior to analysis period",
                key=f"{state_manager.__class__.__name__.lower()}_lookback_months",
            )

        # Save to state manager
        if hasattr(state_manager, "set_lookback_period"):
            state_manager.set_lookback_period(lookback_value, unit_choice)

        return lookback_value, unit_choice


class BaseFilterManager:
    """
    Common filter management patterns used across modules.

    This centralizes the filter creation logic that was duplicated across modules.
    """

    @staticmethod
    def render_multiselect_filter_section(
        title: str,
        icon: str,
        filters: List[Dict[str, Any]],
        df: pd.DataFrame,
        state_manager: Any,
        expanded: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Render a section of multiselect filters.

        Args:
            title: Section title
            icon: Section icon
            filters: List of filter configurations
            df: DataFrame to filter
            state_manager: State manager instance
            expanded: Whether section is expanded by default

        Returns:
            Dictionary mapping filter names to selected values
        """
        results = {}

        with st.sidebar.expander(f"{icon} **{title}**", expanded=expanded):
            for filter_config in filters:
                filter_key = filter_config["key"]
                column_name = filter_config["column"]
                label = filter_config.get("label", filter_key)
                help_text = filter_config.get("help")

                # Get saved state
                saved_values = getattr(
                    state_manager, "get_widget_state", lambda k, d: []
                )(f"{filter_key}_filter", [])

                if column_name in df.columns:
                    selected_values = create_multiselect_filter(
                        df=df,
                        column_name=column_name,
                        label=label,
                        key=f"{state_manager.__class__.__name__.lower()}_{filter_key}",
                        help=help_text,
                        default=saved_values,
                    )

                    results[filter_key] = selected_values

                    # Save to state manager
                    if hasattr(state_manager, "set_widget_state"):
                        state_manager.set_widget_state(
                            f"{filter_key}_filter", selected_values
                        )

        return results

    @staticmethod
    def create_standard_filter_configs() -> Dict[str, List[Dict[str, Any]]]:
        """
        Create standard filter configurations used across modules.

        Returns:
            Dictionary of filter categories with their configurations
        """
        return {
            "program_filters": [
                {
                    "key": "coc",
                    "column": "ProgramSetupCoC",
                    "label": "Program CoC",
                    "help": "Continuum of Care where program operates",
                },
                {
                    "key": "local_coc",
                    "column": "LocalCoCCode",
                    "label": "Local CoC",
                    "help": "Local Continuum of Care code",
                },
                {
                    "key": "agency",
                    "column": "AgencyName",
                    "label": "Agency Name",
                    "help": "Operating agency",
                },
                {
                    "key": "program",
                    "column": "ProgramName",
                    "label": "Program Name",
                    "help": "Specific program",
                },
                {
                    "key": "project_type",
                    "column": "ProjectTypeCode",
                    "label": "Project Type",
                    "help": "HUD project type code",
                },
                {
                    "key": "ssvf_rrh",
                    "column": "SSVF_RRH",
                    "label": "SSVF RRH",
                    "help": "SSVF Rapid Rehousing designation",
                },
                {
                    "key": "continuum",
                    "column": "ProgramsContinuumProject",
                    "label": "Continuum Project",
                    "help": "Continuum project designation",
                },
            ],
            "demographics_filters": [
                {
                    "key": "head_of_household",
                    "column": "IsHeadOfHousehold",
                    "label": "Head of Household",
                    "help": "Head of household status",
                },
                {
                    "key": "household_type",
                    "column": "HouseholdType",
                    "label": "Household Type",
                    "help": "Type of household composition",
                },
                {
                    "key": "race_ethnicity",
                    "column": "RaceEthnicity",
                    "label": "Race / Ethnicity",
                    "help": "Race and ethnicity information",
                },
                {
                    "key": "gender",
                    "column": "Gender",
                    "label": "Gender",
                    "help": "Gender identity",
                },
                {
                    "key": "age_tier",
                    "column": "AgeTieratEntry",
                    "label": "Entry Age Tier",
                    "help": "Age category at program entry",
                },
                {
                    "key": "has_income",
                    "column": "HasIncome",
                    "label": "Has Income",
                    "help": "Income status at entry",
                },
                {
                    "key": "has_disability",
                    "column": "HasDisability",
                    "label": "Has Disability",
                    "help": "Disability status",
                },
            ],
            "housing_filters": [
                {
                    "key": "prior_living",
                    "column": "PriorLivingCat",
                    "label": "Prior Living Situation",
                    "help": "Living situation before program entry",
                },
                {
                    "key": "chronic_homeless",
                    "column": "CHStartHousehold",
                    "label": "Chronic Homelessness Household",
                    "help": "Chronic homelessness status",
                },
                {
                    "key": "exit_destination_cat",
                    "column": "ExitDestinationCat",
                    "label": "Exit Destination Category",
                    "help": "Category of exit destination",
                },
                {
                    "key": "exit_destination",
                    "column": "ExitDestination",
                    "label": "Exit Destination",
                    "help": "Specific exit destination",
                },
            ],
            "special_populations": [
                {
                    "key": "veteran",
                    "column": "VeteranStatus",
                    "label": "Veteran Status",
                    "help": "Military veteran status",
                },
                {
                    "key": "fleeing_dv",
                    "column": "CurrentlyFleeingDV",
                    "label": "Currently Fleeing DV",
                    "help": "Domestic violence survivor status",
                },
            ],
        }


class BaseModulePage(ABC):
    """
    Abstract base class for analysis module pages.

    This provides common structure and methods that were duplicated across
    SPM2, Inbound, and Outbound module pages.
    """

    def __init__(self, module_name: str, state_manager: Any):
        self.module_name = module_name
        self.state_manager = state_manager
        self.date_config = BaseDateConfig()
        self.filter_manager = BaseFilterManager()

    @abstractmethod
    def render_page(self) -> None:
        """Render the main page content. Must be implemented by subclasses."""

    def render_common_header(
        self, title: str, icon: str, description: str
    ) -> None:
        """
        Render a common header for the module.

        Args:
            title: Module title
            icon: Module icon
            description: Module description
        """
        st.html(html_factory.title(f"{icon} {title}", level=1))
        if description:
            st.caption(description)
        st.html(html_factory.divider())

    def render_sidebar_config(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Render common sidebar configuration elements.

        This method should be overridden by subclasses to add their specific
        configuration elements.

        Args:
            df: DataFrame for validation and options

        Returns:
            Dictionary of configuration values
        """
        return {}

    def show_data_info(self, df: pd.DataFrame) -> None:
        """
        Display standard data information section.

        Args:
            df: DataFrame to display information about
        """
        if df is not None and not df.empty:
            st.html(
                html_factory.info_box(
                    f"**Dataset**: {len(df):,} records loaded<br/>"
                    f"**Clients**: {df['ClientID'].nunique():,} unique clients<br/>"
                    f"**Date Range**: {df['ProjectStart'].min():%Y-%m-%d} to {df['ProjectStart'].max():%Y-%m-%d}",
                    type="info",
                    title="Data Summary",
                    icon="📊",
                )
            )

    def render_about_section(
        self, content: str, expanded: bool = False
    ) -> None:
        """
        Render standardized about section.

        Args:
            content: HTML or markdown content
            expanded: Whether section is expanded by default
        """
        from src.ui.factories.components import render_about_section

        render_about_section(
            title=f"About {self.module_name}",
            content=content,
            expanded=expanded,
        )

    def handle_analysis_error(
        self, error: Exception, context: str = "analysis"
    ) -> None:
        """
        Handle analysis errors in a consistent way.

        Args:
            error: The exception that occurred
            context: Context where the error occurred
        """
        st.error(f"❌ Error during {context}: {str(error)}")
        if hasattr(self.state_manager, "mark_dirty"):
            self.state_manager.mark_dirty()

        # Show debug info in expandable section
        with st.expander("🔍 Debug Information", expanded=False):
            st.exception(error)
