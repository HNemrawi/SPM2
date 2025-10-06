"""
Common date configuration utilities.

This module provides specialized date configuration functions that were
previously duplicated across different analysis modules.
"""

from datetime import datetime
from typing import Any, Optional, Tuple

import pandas as pd
import streamlit as st

from src.ui.factories.html import html_factory

from .base_page import BaseDateConfig


class SPM2DateConfig(BaseDateConfig):
    """Date configuration specialized for SPM2 analysis."""

    @staticmethod
    def render_spm2_date_config(
        df: pd.DataFrame,
        state_manager: Any,
    ) -> Tuple[pd.Timestamp, pd.Timestamp, int, str, int]:
        """
        Render SPM2-specific date configuration.

        Returns:
            Tuple of (report_start, report_end, lookback_value, unit_choice, return_period)
        """
        # Date range input
        report_start, report_end = BaseDateConfig.render_date_range_input(
            title="SPM Reporting Period",
            icon="📅",
            default_start=datetime(2024, 10, 1),
            default_end=datetime(2025, 9, 30),
            help_text="Primary analysis window for SPM2 metrics",
            info_message="Primary analysis window for SPM2 metrics. The selected end date will be included in the analysis period.",
            state_manager=state_manager,
            df=df,
        )

        if report_start is None or report_end is None:
            report_start = pd.to_datetime(datetime(2024, 10, 1))
            report_end = pd.to_datetime(datetime(2025, 9, 30))

        st.html(html_factory.divider())

        # Lookback configuration
        lookback_value, unit_choice = BaseDateConfig.render_lookback_config(
            state_manager=state_manager,
            default_days=730,
            default_months=24,
        )

        # Calculate exit window
        if unit_choice == "Days":
            exit_window_start = report_start - pd.Timedelta(
                days=lookback_value
            )
            exit_window_end = report_end - pd.Timedelta(days=lookback_value)
        else:
            exit_window_start = report_start - pd.DateOffset(
                months=lookback_value
            )
            exit_window_end = report_end - pd.DateOffset(months=lookback_value)

        # Display exit window
        st.html(
            html_factory.info_box(
                f"Exit Window: {exit_window_start:%Y-%m-%d} to {exit_window_end:%Y-%m-%d}",
                type="info",
            )
        )

        st.html(html_factory.divider())

        # Return period
        return_period = st.number_input(
            "Return Period (Days)",
            min_value=1,
            value=getattr(state_manager, "get_return_period", lambda: 730)(),
            help="Max days post-exit to count as return",
            key=f"{state_manager.__class__.__name__.lower()}_return_period",
        )

        # Save return period
        if hasattr(state_manager, "set_return_period"):
            state_manager.set_return_period(return_period)

        return (
            report_start,
            report_end,
            lookback_value,
            unit_choice,
            return_period,
        )


class InboundDateConfig(BaseDateConfig):
    """Date configuration specialized for Inbound Recidivism analysis."""

    @staticmethod
    def render_inbound_date_config(
        df: pd.DataFrame,
        state_manager: Any,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[int]]:
        """
        Render Inbound-specific date configuration.

        Returns:
            Tuple of (report_start, report_end, days_lookback)
        """
        # Date range input
        report_start, report_end = BaseDateConfig.render_date_range_input(
            title="Entry Date & Lookback",
            icon="🗓️",
            default_start=datetime(2025, 1, 1),
            default_end=datetime(2025, 1, 31),
            help_text="Analysis period for new entries",
            info_message="The selected end date will be included in the analysis period.",
            state_manager=state_manager,
            df=df,
        )

        if report_start is None or report_end is None:
            return None, None, None

        st.html(html_factory.divider("gradient"))

        # Simple days lookback (no unit choice for inbound)
        saved_lookback = getattr(
            state_manager, "get_widget_state", lambda k, d: d
        )("days_lookback", 730)

        days_lookback = st.number_input(
            "🔍 Days Lookback",
            min_value=1,
            value=saved_lookback,
            help="Number of days prior to entry to consider exits",
            key=f"{state_manager.__class__.__name__.lower()}_days_lookback",
        )

        # Save lookback
        if hasattr(state_manager, "set_widget_state"):
            state_manager.set_widget_state("days_lookback", days_lookback)

        return report_start, report_end, days_lookback


class OutboundDateConfig(BaseDateConfig):
    """Date configuration specialized for Outbound Recidivism analysis."""

    @staticmethod
    def render_outbound_date_config(
        df: pd.DataFrame,
        state_manager: Any,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], int, str]:
        """
        Render Outbound-specific date configuration.

        Returns:
            Tuple of (report_start, report_end, lookback_value, unit_choice)
        """
        # Date range input
        report_start, report_end = BaseDateConfig.render_date_range_input(
            title="Exit Date Configuration",
            icon="📅",
            default_start=datetime(2024, 10, 1),
            default_end=datetime(2025, 9, 30),
            help_text="Primary analysis window for exit tracking",
            info_message="Analysis period for tracking client exits. The selected end date will be included in the analysis period.",
            state_manager=state_manager,
            df=df,
        )

        if report_start is None or report_end is None:
            report_start = pd.to_datetime(datetime(2024, 10, 1))
            report_end = pd.to_datetime(datetime(2025, 9, 30))

        st.html(html_factory.divider())

        # Lookback configuration (for return tracking)
        lookback_value, unit_choice = BaseDateConfig.render_lookback_config(
            state_manager=state_manager,
            default_days=730,
            default_months=24,
        )

        return report_start, report_end, lookback_value, unit_choice
