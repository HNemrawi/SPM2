"""
Trend explorer section for HMIS dashboard with enhanced UI and auto-adjusting charts.
Optimized for both dark and light themes with improved organization.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas import DataFrame, Timestamp

from src.core.data.destinations import apply_custom_ph_destinations
from src.modules.dashboard.data_utils import (
    DEMOGRAPHIC_DIMENSIONS,
    FREQUENCY_MAP,
    calculate_demographic_growth,
    inflow,
    outflow,
    ph_exit_clients,
    recalculated_metric_time_series,
    recalculated_metric_time_series_by_group,
    return_after_exit,
    served_clients,
)
from src.modules.dashboard.filters import (
    get_filter_timestamp,
    hash_data,
    init_section_state,
    invalidate_cache,
    is_cache_valid,
)
from src.ui.factories.components import (
    create_insight_container,
    fmt_int,
    fmt_pct,
    ui,
)
from src.ui.factories.html import html_factory
from src.ui.themes.theme import (
    MAIN_COLOR,
    NEUTRAL_COLOR,
    blue_divider,
    professional_colors,
)
from src.ui.themes.theme import theme as theme_config

# ================================================================================
# CONSTANTS AND CONFIGURATION
# ================================================================================

TREND_SECTION_KEY = "trend_explorer"

# Enhanced color palette for better visibility in both themes
CHART_COLORS = {
    "primary": {
        "Active Clients": "#60A5FA",  # Bright sky blue
        "Inflow": "#34D399",  # Bright emerald
        "Outflow": "#FBBF24",  # Bright amber
        "PH Exits": "#A78BFA",  # Bright purple
        "Returns": "#F87171",  # Bright red (for Returns to Homelessness)
    },
    "secondary": {
        "Active Clients": "#3B82F6",  # Blue
        "Inflow": "#10B981",  # Emerald
        "Outflow": "#F59E0B",  # Amber
        "PH Exits": "#8B5CF6",  # Purple
        "Returns": "#EF4444",  # Red
    },
    "muted": {
        "Active Clients": "#93C5FD",  # Light blue
        "Inflow": "#86EFAC",  # Light emerald
        "Outflow": "#FDE68A",  # Light amber
        "PH Exits": "#C4B5FD",  # Light purple
        "Returns": "#FCA5A5",  # Light red
    },
}

# Custom color sequence for breakdowns
CUSTOM_COLOR_SEQUENCE = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#17becf",  # blue-teal
    "#bcbd22",  # curry yellow-green
    "#393b79",  # dark slate blue
    "#637939",  # olive green
    "#8c6d31",  # brownish yellow
    "#843c39",  # dark sienna
    "#7b4173",  # deep magenta
]

# Neutral metrics
NEUTRAL_METRICS = {"Active Clients", "Inflow", "Outflow"}

# Rolling window sizes by frequency
SUGGESTED_ROLLING_WINDOWS = {
    "Days": 7,  # Weekly rolling for daily data
    "Weeks": 4,  # Monthly rolling for weekly data
    "Months": 3,  # Quarterly rolling for monthly data
    "Quarters": 2,  # Biannual rolling for quarterly data
    "Years": 2,  # Biannual rolling for yearly data
}

# ================================================================================
# THEME AND STYLING UTILITIES
# ================================================================================


def get_theme_colors():
    """Get colors optimized for current theme (dark/light)."""
    # Use actual color values that work in CSS
    return {
        "background": "rgba(0, 0, 0, 0.05)",
        "surface": "rgba(0, 0, 0, 0.02)",
        "border": "rgba(128, 128, 128, 0.2)",
        "text_primary": "#333333",  # Dark gray for text
        "text_secondary": "rgba(128, 128, 128, 0.8)",
        "success": "#10B981",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "info": "#3B82F6",
        "neutral": "#6B7280",
    }


def get_plotly_theme():
    """Get Plotly theme settings that work well in both dark and light modes."""
    return {
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            "size": 12,
            "color": "#666666",  # Medium gray that works in both themes
        },
        "xaxis": {
            "gridcolor": "rgba(128, 128, 128, 0.2)",
            "linecolor": "rgba(128, 128, 128, 0.3)",
            "tickcolor": "rgba(128, 128, 128, 0.3)",
        },
        "yaxis": {
            "gridcolor": "rgba(128, 128, 128, 0.2)",
            "linecolor": "rgba(128, 128, 128, 0.3)",
            "tickcolor": "rgba(128, 128, 128, 0.3)",
        },
        "hoverlabel": {
            "bgcolor": "rgba(0, 0, 0, 0.8)",
            "bordercolor": "rgba(255, 255, 255, 0.2)",
            "font": {"size": 13, "color": "white", "family": "monospace"},
        },
    }


# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================


def _get_metric_color(metric_name: str, style: str = "primary") -> str:
    """Get appropriate color for a metric based on its type and style."""
    color_set = CHART_COLORS.get(style, CHART_COLORS["primary"])

    if "Returns to Homelessness" in metric_name:
        return color_set.get("Returns", "#F87171")

    for key, color in color_set.items():
        if key in metric_name:
            return color

    return MAIN_COLOR


def _get_hover_template(freq: str) -> str:
    """Get appropriate hover template based on frequency."""
    templates = {
        "D": "<b>%{x|%b %d, %Y}</b><br>Value: %{y:,.0f}<extra></extra>",
        "W": "<b>Week of %{x|%b %d, %Y}</b><br>Value: %{y:,.0f}<extra></extra>",
        "M": "<b>%{x|%b %Y}</b><br>Value: %{y:,.0f}<extra></extra>",
        "Q": "<b>%{x|%b %Y}</b><br>Value: %{y:,.0f}<extra></extra>",
        "Y": "<b>%{x|%Y}</b><br>Value: %{y:,.0f}<extra></extra>",
    }
    return templates.get(freq, templates["M"])


def _get_x_axis_angle(freq: str, num_points: int) -> int:
    """Determine optimal x-axis label rotation to prevent overlap."""
    if freq == "D" and num_points > 30:
        return -90
    elif freq == "W" and num_points > 20:
        return -45
    elif freq == "M" and num_points > 24:
        return -45
    return 0


def _calculate_dynamic_height(num_points: int, base_height: int = 400) -> int:
    """Calculate dynamic chart height based on number of data points."""
    if num_points <= 12:
        return base_height
    elif num_points <= 24:
        return base_height + 50
    elif num_points <= 52:
        return base_height + 100
    return min(base_height + 200, 700)


def _calculate_dynamic_margins(
    num_points: int, has_text_labels: bool = False
) -> dict:
    """Calculate dynamic margins to prevent cutoff."""
    base_margins = {"l": 80, "r": 80, "t": 100, "b": 100}

    # Increase bottom margin for rotated labels
    if num_points > 20:
        base_margins["b"] = 140
    elif num_points > 10:
        base_margins["b"] = 120

    # Increase margins for text labels
    if has_text_labels:
        base_margins["t"] = 160
        base_margins["b"] = max(base_margins["b"], 140)

    return base_margins


def _get_trend_icon(
    total_change: float, increase_is_negative: bool = False
) -> str:
    """Get appropriate emoji icon for trend direction."""
    if abs(total_change) < 0.001:
        return "➡️"
    elif total_change > 0:
        return "📉" if increase_is_negative else "📈"
    else:
        return "📈" if increase_is_negative else "📉"


def _get_trend_color(change: float, increase_is_negative: bool = False) -> str:
    """Get appropriate color for a change value."""
    if abs(change) < 0.001:
        return "gray"
    elif (change > 0 and not increase_is_negative) or (
        change < 0 and increase_is_negative
    ):
        return "green"
    else:
        return "red"


def _format_period_label(freq_label: str) -> str:
    """Convert frequency label to user-friendly period name."""
    period_map = {
        "Days": "day",
        "Weeks": "week",
        "Months": "month",
        "Quarters": "quarter",
        "Years": "year",
    }
    return period_map.get(freq_label, "period")


# ================================================================================
# UI COMPONENTS - CARDS AND EXPLANATIONS
# ================================================================================


def _render_metric_explanation_card():
    """Render metric explanation using UI factory."""

    # Create metric explanations as structured content
    content = f"""
    <h4 style="margin-top: 0;">📊 Understanding the Metrics</h4>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px;">
        <div>
            <strong>Active Clients</strong>
            <p style="margin: 4px 0; font-size: 14px;">All clients enrolled at any point during the time period</p>
        </div>
        <div>
            <strong>Inflow</strong>
            <p style="margin: 4px 0; font-size: 14px;">Clients entering during the period who weren't enrolled the day before it started</p>
        </div>
        <div>
            <strong>Outflow</strong>
            <p style="margin: 4px 0; font-size: 14px;">Clients who exited during the period and aren't enrolled on the last day</p>
        </div>
        <div style="background-color: rgba(16, 185, 129, 0.1); padding: 12px; border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.3);">
            <strong>PH Exits</strong> <span style="color: {professional_colors.SUCCESS}; font-size: 12px; font-weight: 600;">✓ Positive Outcome</span>
            <p style="margin: 4px 0; font-size: 14px;">Clients exiting to permanent housing during the period</p>
        </div>
        <div style="background-color: rgba(239, 68, 68, 0.1); padding: 12px; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);">
            <strong>Returns to Homelessness</strong> <span style="color: {professional_colors.ERROR}; font-size: 12px; font-weight: 600;">✗ Negative Outcome</span>
            <p style="margin: 4px 0; font-size: 14px;">Clients who exited to PH and returned within the specified window</p>
        </div>
    </div>
    """

    ui.info_section(content=content, type="info", title=None, expanded=True)


def _render_insight_card(
    metric_name: str,
    df: DataFrame,
    period_label: str,
    increase_is_negative: bool,
    is_neutral_metric: bool,
):
    """Render insight card using UI factory."""
    # Calculate insights
    avg_change = df["delta"].mean()
    total_change = df["delta"].sum()

    max_increase = df["delta"].max()
    max_increase_date = df.loc[df["delta"].idxmax(), "bucket"]
    max_decrease = df["delta"].min()
    max_decrease_date = df.loc[df["delta"].idxmin(), "bucket"]

    # Get trend info
    trend_icon = _get_trend_icon(total_change, increase_is_negative)

    if total_change > 0:
        if increase_is_negative:
            trend_direction = "Increasing (Negative Trend)"
        else:
            trend_direction = "Increasing (Positive Trend)"
    elif total_change < 0:
        if increase_is_negative:
            trend_direction = "Decreasing (Positive Trend)"
        else:
            trend_direction = "Decreasing (Negative Trend)"
    else:
        trend_direction = "Stable"

    if is_neutral_metric:
        trend_direction = trend_direction.split(" (")[0]

    # Create metrics summary
    metrics = {
        "Current Trend": {
            "value": trend_direction,
            "help": f"Overall direction: {trend_icon}",
        },
        "Average Change": {
            "value": f"{avg_change:+,.0f}",
            "delta": f"per {period_label}",
        },
        "Total Change": {"value": f"{total_change:+,.0f}", "delta": "clients"},
    }

    # Use UI factory to create metric row
    ui.metric_row(metrics, columns=3)

    # Create notable changes as insight containers
    st.html(html_factory.title("Notable Changes", level=4, icon="📈"))

    col1, col2 = st.columns(2)
    with col1:
        create_insight_container(
            title="Largest Increase",
            content=f"<strong>{max_increase:+,.0f}</strong> in {max_increase_date:%b %Y}",
            icon="📈",
            type="success",
        )

    with col2:
        create_insight_container(
            title="Largest Decrease",
            content=f"<strong>{max_decrease:+,.0f}</strong> in {max_decrease_date:%b %Y}",
            icon="📉",
            type="warning",
        )


def _render_interpretation_note(
    metric_name: str, is_neutral_metric: bool, increase_is_negative: bool
):
    """Render interpretation note using UI factory."""

    if is_neutral_metric:
        type_val = "info"
        icon = "ℹ️"
        message = f"{
            metric_name} is a <strong>volume metric</strong>. Changes show fluctuations in client numbers but are neither positive nor negative outcomes."
    elif increase_is_negative:
        type_val = "warning"
        icon = "⚠️"
        message = f"For {
            metric_name}, <strong>increases are negative</strong> - we want to see this number go down."
    else:
        type_val = "success"
        icon = "✅"
        message = f"For {
            metric_name}, <strong>increases are positive</strong> - we want to see this number go up."

    create_insight_container(
        title="Interpretation Note", content=message, icon=icon, type=type_val
    )


def _render_section_header():
    """Render the section header with help information using UI factory."""
    st.html(html_factory.title("Trend Analysis", level=2, icon="📈"))

    col_help = st.columns([11, 1])[1]
    with col_help:
        with st.popover("ℹ️ Help", width='stretch'):
            st.markdown(
                """
            ### Understanding Trend Analysis

            This section tracks how metrics change over time, revealing patterns, seasonality, and the impact of interventions.

            **Available Metrics:**

            **Volume Metrics (Neutral - neither good nor bad):**
            - **Active Clients**: All unique clients enrolled at any point during each time period
            - **Inflow**: Clients entering who weren't in programs the day before
            - **Outflow**: Clients exiting who aren't in programs at period end

            **Outcome Metrics:**
            - **PH Exits** ✅: Clients exiting to permanent housing (increases are positive)
            - **Returns to Homelessness** ❌: PH exits who returned within tracking window (increases are negative)

            **Time Aggregation Options:**
            - **Days**: Daily values (best for short periods)
            - **Weeks**: Weekly aggregates (good for 1-3 month views)
            - **Months**: Monthly totals (recommended for most analyses)
            - **Quarters**: Quarterly summaries (good for annual views)
            - **Years**: Annual totals (for long-term trends)

            **Analysis Features:**

            **1. Overall Trends (No breakdown):**
            - Line charts showing metric values over time
            - Rolling averages (3-period default, auto-adjusts by frequency)
            - Period-over-period change analysis
            - Combined view for comparing multiple metrics

            **2. Demographic Breakdowns:**
            - Compare trends across demographic groups
            - Automatic limiting to top groups for clarity
            - Growth rate analysis (first vs last period)
            - Disparity tracking over time

            **Key Calculations:**
            - **Each time period is calculated independently** using full metric logic
            - **Not simple counts** - proper deduplication and business rules applied
            - **Period changes**: Show both absolute and percentage changes
            - **Growth rates**: (Last Value - First Value) ÷ First Value × 100

            **Visual Components:**

            **1. Trend Line Charts:**
            - Actual values with markers at each data point
            - Rolling average lines (dashed) for smoothing
            - Auto-scaling axes and dynamic height
            - Smart date formatting based on frequency

            **2. Change Analysis Charts:**
            - Bar charts showing period-over-period changes
            - Color-coded: Green = improvement, Red = decline
            - Special handling for metrics where decrease is good
            - Text labels showing exact change values

            **3. Insights Panel:**
            - Current trend direction and total change
            - Average change per period
            - Largest increases and decreases with dates
            - Volatility: Measures how much the numbers swing up and down over time

            **4. Growth Analysis (for breakdowns):**
            - Side-by-side growth comparisons
            - Fastest growing and declining groups
            - Current disparity ratios
            - Resource utilization patterns
            """
            )


# ================================================================================
# DATA PROCESSING FUNCTIONS
# ================================================================================


def _get_trend_data(
    df_filt: DataFrame,
    full_df: DataFrame,
    metric_funcs: Dict[str, Any],
    sel_metrics: List[str],
    sel_freq: str,
    group_col: Optional[str],
    t0: Timestamp,
    t1: Timestamp,
    return_window: int,
) -> Dict[str, DataFrame]:
    """Calculate trend data for selected metrics and groups."""

    # Apply custom PH destinations
    df_filt = apply_custom_ph_destinations(df_filt, force=True)
    full_df = apply_custom_ph_destinations(full_df, force=True)

    multi_data = {}

    for metric_name in sel_metrics:
        try:
            metric_func = metric_funcs[metric_name]

            # Special handling for returns function
            if metric_name.startswith("Returns to Homelessness"):

                def metric_func_wrapper(df, s, e, mf=metric_func):
                    return mf(df, full_df, s, e, return_window)

                metric_func = metric_func_wrapper

            if group_col is None:
                # Overall trend (no breakdown)
                ts = recalculated_metric_time_series(
                    df_filt, metric_func, t0, t1, sel_freq
                )

                if ts is None or ts.empty:
                    continue

                ts = ts.sort_values("bucket")
                ts["delta"] = ts["count"].diff().fillna(0)

                # Calculate rolling average
                min_periods = 1
                ts["rolling"] = (
                    ts["count"]
                    .rolling(window=3, min_periods=min_periods)
                    .mean()
                )
                ts["metric"] = metric_name
                ts["pct_change"] = ts["count"].pct_change().fillna(0) * 100

            else:
                # Breakdown by group
                ts = recalculated_metric_time_series_by_group(
                    df_filt, metric_func, group_col, t0, t1, sel_freq
                )

                if ts is None or ts.empty:
                    continue

                ts = ts.sort_values(["group", "bucket"])
                ts["metric"] = metric_name

                # Process each group separately
                ts_groups = []
                for group, group_data in ts.groupby("group"):
                    group_data = group_data.sort_values("bucket")
                    group_data["delta"] = group_data["count"].diff().fillna(0)
                    group_data["pct_change"] = (
                        group_data["count"].pct_change().fillna(0) * 100
                    )
                    group_data["rolling"] = (
                        group_data["count"]
                        .rolling(window=3, min_periods=1)
                        .mean()
                    )
                    ts_groups.append(group_data)

                if ts_groups:
                    ts = pd.concat(ts_groups)

            multi_data[metric_name] = ts

        except Exception as e:
            st.error(f"Error calculating {metric_name}: {str(e)}")

    return multi_data


# ================================================================================
# VISUALIZATION FUNCTIONS - ENHANCED WITH BETTER STYLING
# ================================================================================


def _create_combined_trend_chart(
    combined_df: DataFrame,
    sel_metrics: List[str],
    sel_freq_label: str,
    do_roll: bool,
    roll_window: int,
    multi_data: Dict[str, DataFrame],
) -> go.Figure:
    """Create a professional combined line chart for multiple metrics."""

    # Get frequency code
    sel_freq = FREQUENCY_MAP[sel_freq_label]

    # Build color palette
    color_palette = {}
    for metric in sel_metrics:
        color_palette[metric] = _get_metric_color(metric, "primary")

    # Calculate dimensions
    num_points = len(combined_df["bucket"].unique())
    chart_height = max(500, min(800, 400 + num_points * 5))

    # Create figure with professional styling
    fig = go.Figure()

    # Add main trend lines
    for metric_name in sel_metrics:
        metric_data = combined_df[combined_df["metric"] == metric_name]
        if not metric_data.empty:
            # Main line with enhanced styling
            fig.add_trace(
                go.Scatter(
                    x=metric_data["bucket"],
                    y=metric_data["count"],
                    mode="lines+markers",
                    name=metric_name,
                    line=dict(
                        color=color_palette[metric_name],
                        width=3,
                        shape="spline",
                        smoothing=0.3,
                    ),
                    marker=dict(
                        size=8,
                        color=color_palette[metric_name],
                        line=dict(width=2, color="rgba(255, 255, 255, 0.8)"),
                    ),
                    hovertemplate=(
                        f"<b>{metric_name}</b><br>"
                        "<b>Date:</b> %{x|%b %d, %Y}<br>"
                        "<b>Value:</b> %{y:,.0f} clients<br>"
                        "<extra></extra>"
                    ),
                )
            )

            # Add rolling average if enabled
            if do_roll and metric_name in multi_data:
                df = multi_data[metric_name]
                if not df.empty and "rolling" in df.columns:
                    period_label = _format_period_label(sel_freq_label)
                    fig.add_trace(
                        go.Scatter(
                            x=df["bucket"],
                            y=df["rolling"],
                            mode="lines",
                            name=f"{metric_name} ({roll_window}-{period_label} avg)",
                            line=dict(
                                color=color_palette[metric_name],
                                width=2,
                                dash="dot",
                            ),
                            opacity=0.6,
                            hovertemplate=(
                                f"<b>{metric_name} (Rolling Avg)</b><br>"
                                "<b>Date:</b> %{x|%b %d, %Y}<br>"
                                "<b>Average:</b> %{y:,.1f} clients<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

    # Get theme settings
    theme = get_plotly_theme()

    # Build xaxis config by merging theme settings with specific settings
    xaxis_config = {
        "title": dict(text="Time Period", font=dict(size=14, weight="bold")),
        "showgrid": True,
        "gridwidth": 1,
        "showline": True,
        "linewidth": 2,
        "tickfont": dict(size=11),
        "tickformat": (
            "%b %Y"
            if sel_freq in ["M", "Q"]
            else ("%Y" if sel_freq == "Y" else "%b %d")
        ),
        "tickangle": _get_x_axis_angle(sel_freq, num_points),
        "automargin": True,
    }
    # Merge with theme settings
    xaxis_config.update(theme["xaxis"])

    # Build yaxis config similarly
    yaxis_config = {
        "title": dict(
            text="Number of Clients", font=dict(size=14, weight="bold")
        ),
        "showgrid": True,
        "gridwidth": 1,
        "showline": True,
        "linewidth": 2,
        "tickfont": dict(size=11),
        "tickformat": ",d",
        "automargin": True,
    }
    # Merge with theme settings
    yaxis_config.update(theme["yaxis"])

    # Professional layout configuration
    fig.update_layout(
        title=dict(
            text=f"Metrics Trend Analysis - {sel_freq_label}",
            font=dict(size=24, weight="bold"),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        height=chart_height,
        margin=dict(l=80, r=40, t=100, b=100),
        plot_bgcolor=theme["plot_bgcolor"],
        paper_bgcolor=theme["paper_bgcolor"],
        font=theme["font"],
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        legend=dict(
            orientation="h" if len(sel_metrics) <= 3 else "v",
            yanchor="bottom" if len(sel_metrics) <= 3 else "top",
            y=1.02 if len(sel_metrics) <= 3 else 0.98,
            xanchor="center" if len(sel_metrics) <= 3 else "left",
            x=0.5 if len(sel_metrics) <= 3 else 1.02,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            font=dict(size=12),
        ),
        hoverlabel=theme["hoverlabel"],
        dragmode="pan",
        hovermode="x unified",
    )

    # Add range slider for long time series
    if num_points > 30:
        fig.update_xaxes(
            rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor="rgba(0, 0, 0, 0.05)",
                borderwidth=1,
            )
        )
        fig.update_layout(height=chart_height + 50)

    # Configure modebar
    fig.update_layout(
        modebar=dict(
            orientation="v",
            bgcolor="rgba(0, 0, 0, 0)",
            color="rgba(0, 0, 0, 0.5)",
            activecolor="rgba(0, 0, 0, 0.9)",
        )
    )

    return fig


def _create_delta_chart(
    df: DataFrame,
    metric_name: str,
    period_label: str,
    is_neutral_metric: bool,
    increase_is_negative: bool,
    sel_freq: str,
) -> go.Figure:
    """Create a professional bar chart showing period-over-period changes."""

    # Professional color scheme from theme
    positive_color = theme_config.colors.success
    negative_color = theme_config.colors.danger
    neutral_color = theme_config.colors.neutral_500

    # Determine colors based on interpretation
    bar_colors = []
    for val in df["delta"]:
        if abs(val) < 0.001:
            bar_colors.append(neutral_color)
        elif is_neutral_metric:
            bar_colors.append("#3B82F6")  # Blue for neutral metrics
        elif (val > 0 and not increase_is_negative) or (
            val < 0 and increase_is_negative
        ):
            bar_colors.append(positive_color)
        else:
            bar_colors.append(negative_color)

    # Calculate dimensions
    num_bars = len(df)
    chart_height = max(400, min(700, 350 + num_bars * 15))

    # Create figure
    fig = go.Figure()

    # Add bars with enhanced styling
    fig.add_trace(
        go.Bar(
            x=df["bucket"],
            y=df["delta"],
            marker=dict(
                color=bar_colors,
                line=dict(color="rgba(0, 0, 0, 0.2)", width=1),
            ),
            text=[f"{val:+,.0f}" for val in df["delta"]],
            textposition="outside",
            textfont=dict(size=12, weight="bold"),
            hovertemplate=(
                "<b>Period:</b> %{x|%b %Y}<br>"
                "<b>Change:</b> %{y:+,.0f} clients<br>"
                "<b>Percentage:</b> %{customdata:.1f}%<br>"
                "<extra></extra>"
            ),
            customdata=(
                df["pct_change"]
                if "pct_change" in df.columns
                else [0] * len(df)
            ),
        )
    )

    # Get theme settings
    theme = get_plotly_theme()

    # Build xaxis config
    xaxis_config = {
        "title": dict(text="Time Period", font=dict(size=14, weight="bold")),
        "showgrid": False,
        "showline": True,
        "linewidth": 2,
        "tickfont": dict(size=11),
        "tickformat": (
            "%b %Y"
            if sel_freq in ["M", "Q"]
            else ("%Y" if sel_freq == "Y" else "%b %d")
        ),
        "tickangle": _get_x_axis_angle(sel_freq, num_bars),
        "automargin": True,
    }
    xaxis_config.update(theme["xaxis"])

    # Build yaxis config
    yaxis_config = {
        "title": dict(
            text=f"Change in Clients per {period_label.capitalize()}",
            font=dict(size=14, weight="bold"),
        ),
        "showgrid": True,
        "gridwidth": 1,
        "showline": True,
        "linewidth": 2,
        "tickfont": dict(size=11),
        "tickformat": ",d",
        "automargin": True,
        "zeroline": True,
        "zerolinewidth": 2,
        "zerolinecolor": "rgba(0, 0, 0, 0.3)",
    }
    yaxis_config.update(theme["yaxis"])

    # Professional layout configuration
    fig.update_layout(
        title=dict(
            text=f"{period_label.capitalize()}-over-{period_label.capitalize()} Changes: {metric_name}",
            font=dict(size=20, weight="bold"),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        height=chart_height,
        margin=dict(l=80, r=40, t=120, b=100),
        plot_bgcolor=theme["plot_bgcolor"],
        paper_bgcolor=theme["paper_bgcolor"],
        font=theme["font"],
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        hoverlabel=theme["hoverlabel"],
        showlegend=False,
    )

    # Adjust y-axis range for better visibility
    y_max = df["delta"].max()
    y_min = df["delta"].min()
    y_range_padding = max(abs(y_max), abs(y_min)) * 0.2

    fig.update_yaxes(range=[y_min - y_range_padding, y_max + y_range_padding])

    # Add reference line at zero
    fig.add_hline(
        y=0,
        line=dict(color="rgba(0, 0, 0, 0.3)", width=2, dash="dash"),
        annotation_text="No Change",
        annotation_position="right",
        annotation_font=dict(size=11, color="rgba(0, 0, 0, 0.5)"),
    )

    return fig


def _render_insights_panel(
    df: DataFrame,
    metric_name: str,
    period_label: str,
    increase_is_negative: bool,
    is_neutral_metric: bool,
):
    """Render a professional insights panel with key metrics."""
    colors = get_theme_colors()

    # Calculate metrics
    first_value = df["count"].iloc[0]
    last_value = df["count"].iloc[-1]
    total_change = last_value - first_value
    pct_change = (total_change / first_value * 100) if first_value > 0 else 0

    recent_periods = min(3, len(df) - 1)
    recent_values = df["count"].iloc[-recent_periods - 1 :]
    recent_change = recent_values.iloc[-1] - recent_values.iloc[0]
    recent_pct = (
        (recent_change / recent_values.iloc[0] * 100)
        if recent_values.iloc[0] > 0
        else 0
    )

    # Get trend info
    trend_icon = _get_trend_icon(total_change, increase_is_negative)
    trend_direction = (
        "increasing"
        if total_change > 0
        else "decreasing" if total_change < 0 else "stable"
    )

    # Determine colors
    if is_neutral_metric:
        change_color = colors["neutral"]
        recent_color = colors["neutral"]
    else:
        change_color_name = _get_trend_color(
            total_change, increase_is_negative
        )
        recent_color_name = _get_trend_color(
            recent_change, increase_is_negative
        )

        # Map color names to hex values
        color_map = {
            "green": colors["success"],
            "red": colors["danger"],
            "gray": colors["neutral"],
        }
        change_color = color_map.get(change_color_name, colors["neutral"])
        recent_color = color_map.get(recent_color_name, colors["neutral"])

    # Calculate volatility
    volatility = df["pct_change"].std()
    if volatility < 5:
        volatility_desc = "Very stable"
        volatility_color = colors["success"]
    elif volatility < 10:
        volatility_desc = "Moderately stable"
        volatility_color = colors["info"]
    elif volatility < 20:
        volatility_desc = "Somewhat volatile"
        volatility_color = colors["warning"]
    else:
        volatility_desc = "Highly volatile"
        volatility_color = colors["danger"]

    # Enhanced HTML with better styling
    insights_html = f"""
    <div style="
        background: linear-gradient(135deg, {colors['background']} 0%, rgba(59, 130, 246, 0.05) 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid {colors['border']};
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%;
    ">
        <h3 style="
            color: {colors['text_primary']};
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 22px;
            font-weight: 600;
        ">Key Insights</h3>

        <div style="display: grid; gap: 16px;">
            <div style="
                background-color: {colors['surface']};
                border-radius: 12px;
                padding: 16px;
                border: 1px solid {colors['border']};
            ">
                <p style="margin: 0; color: {colors['text_secondary']}; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">
                    Current Direction
                </p>
                <p style="margin: 6px 0; color: {colors['text_primary']}; font-size: 20px; font-weight: 600;">
                    {trend_direction.capitalize()} {trend_icon}
                </p>
            </div>

            <div style="
                background-color: {colors['surface']};
                border-radius: 12px;
                padding: 16px;
                border: 1px solid {colors['border']};
            ">
                <p style="margin: 0; color: {colors['text_secondary']}; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">
                    Current Value
                </p>
                <p style="margin: 6px 0; color: {colors['text_primary']}; font-size: 20px; font-weight: 600;">
                    {fmt_int(last_value)} <span style="font-size: 14px; font-weight: 400;">clients</span>
                </p>
            </div>

            <div style="
                background: linear-gradient(135deg, {colors['surface']} 0%, {change_color}10 100%);
                border-radius: 12px;
                padding: 16px;
                border: 1px solid {change_color}40;
            ">
                <p style="margin: 0; color: {colors['text_secondary']}; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">
                    Overall Change
                </p>
                <p style="margin: 6px 0; color: {change_color}; font-size: 20px; font-weight: 600;">
                    {total_change:+,.0f} <span style="font-size: 16px;">({pct_change:+.1f}%)</span>
                </p>
            </div>

            <div style="
                background-color: {colors['surface']};
                border-radius: 12px;
                padding: 16px;
                border: 1px solid {colors['border']};
            ">
                <p style="margin: 0; color: {colors['text_secondary']}; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">
                    Recent Change (Last {recent_periods} {period_label}s)
                </p>
                <p style="margin: 6px 0; color: {recent_color}; font-size: 20px; font-weight: 600;">
                    {recent_change:+,.0f} <span style="font-size: 16px;">({recent_pct:+.1f}%)</span>
                </p>
            </div>

            <div style="
                background-color: {colors['surface']};
                border-radius: 12px;
                padding: 16px;
                border: 1px solid {volatility_color}40;
            ">
                <p style="margin: 0; color: {colors['text_secondary']}; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">
                    Volatility
                </p>
                <p style="margin: 6px 0; color: {volatility_color}; font-size: 20px; font-weight: 600;">
                    {volatility_desc} <span style="font-size: 16px;">({volatility:.1f}%)</span>
                </p>
            </div>
        </div>
    </div>
    """
    st.html(insights_html)


def _render_growth_insights(
    growth_df: DataFrame, filtered_df: DataFrame, sel_break: str
):
    """Render professional growth analysis insights."""
    insights_content = []
    insights_content.append(
        """
    <style>
        .growth-insights {
            background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .growth-title {
            color: var(--text-color);
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 22px;
            font-weight: 600;
        }
        .growth-card {
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 16px;
        }
        .growth-card.positive {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .growth-card.negative {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        .growth-card.warning {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        .growth-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }
        .growth-label {
            margin: 0;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .growth-group {
            margin: 8px 0;
            color: var(--text-color);
            font-size: 18px;
            font-weight: bold;
        }
        .growth-value {
            margin: 8px 0;
            font-size: 28px;
            font-weight: 700;
        }
        .growth-details {
            margin: 0;
            color: var(--text-color-secondary);
            font-size: 14px;
        }
        .disparity-section {
            margin-top: 24px;
            padding-top: 24px;
            border-top: 1px solid var(--border-color);
        }
        .disparity-card {
            background-color: var(--background-color);
            border-radius: 12px;
            padding: 18px;
            border: 1px solid var(--border-color);
        }
        .disparity-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .disparity-ratio {
            background: linear-gradient(135deg, rgba(96, 165, 250, 0.1) 0%, rgba(96, 165, 250, 0.05) 100%);
            border-radius: 8px;
            padding: 12px;
            margin-top: 16px;
            border: 1px solid rgba(96, 165, 250, 0.3);
            text-align: center;
        }
    </style>
    <div class="growth-insights">
        <h3 class="growth-title">Key Findings</h3>
    """
    )

    if len(growth_df) >= 2:
        try:
            # Find max and min growth groups
            max_growth_idx = growth_df["growth_pct"].idxmax()
            min_growth_idx = growth_df["growth_pct"].idxmin()

            if max_growth_idx is not None and min_growth_idx is not None:
                max_growth = growth_df.loc[max_growth_idx]
                min_growth = growth_df.loc[min_growth_idx]

                # Fastest Growing Group Card
                insights_content.append(
                    f"""
                <div class="growth-card positive">
                    <div class="growth-header">
                        <span style="font-size: 20px;">🚀</span>
                        <p class="growth-label" style="color: #10B981;">Fastest Growing Group</p>
                    </div>
                    <p class="growth-group">{max_growth['group']}</p>
                    <p class="growth-value" style="color: #10B981;">{fmt_pct(max_growth['growth_pct'])}</p>
                    <p class="growth-details">
                        <span style="opacity: 0.7;">From</span> {fmt_int(max_growth['first_count'])}
                        <span style="opacity: 0.7;">to</span> {fmt_int(max_growth['last_count'])}
                        <span style="opacity: 0.7;">clients</span>
                    </p>
                </div>
                """
                )

                # Determine styling for min growth
                if min_growth["growth_pct"] < 0:
                    card_class = "negative"
                    label = "Fastest Declining Group"
                    emoji = "📉"
                    color = theme_config.colors.danger
                else:
                    card_class = "warning"
                    label = "Slowest Growing Group"
                    emoji = "🐌"
                    color = theme_config.colors.warning

                insights_content.append(
                    f"""
                <div class="growth-card {card_class}">
                    <div class="growth-header">
                        <span style="font-size: 20px;">{emoji}</span>
                        <p class="growth-label" style="color: {color};">{label}</p>
                    </div>
                    <p class="growth-group">{min_growth['group']}</p>
                    <p class="growth-value" style="color: {color};">{fmt_pct(min_growth['growth_pct'])}</p>
                    <p class="growth-details">
                        <span style="opacity: 0.7;">From</span> {fmt_int(min_growth['first_count'])}
                        <span style="opacity: 0.7;">to</span> {fmt_int(min_growth['last_count'])}
                        <span style="opacity: 0.7;">clients</span>
                    </p>
                </div>
                """
                )

            # Add disparity analysis
            latest_date = filtered_df["bucket"].max()
            latest_values = filtered_df[filtered_df["bucket"] == latest_date]

            if not latest_values.empty and len(latest_values) >= 2:
                max_idx = latest_values["count"].idxmax()
                min_idx = latest_values["count"].idxmin()

                max_group = latest_values.loc[max_idx]
                min_group = latest_values.loc[min_idx]

                if min_group["count"] > 0:
                    disparity_ratio = max_group["count"] / min_group["count"]

                    insights_content.append(
                        f"""
                    <div class="disparity-section">
                        <div class="disparity-card">
                            <p style="margin: 0 0 16px 0; color: var(--text-color-secondary); font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                                Current Disparity Analysis
                            </p>

                            <div style="display: grid; gap: 12px;">
                                <div class="disparity-row">
                                    <span style="color: #3B82F6; font-weight: 500;">📊 Highest:</span>
                                    <span>
                                        <span style="color: var(--text-color); font-weight: bold;">{max_group['group']}</span>
                                        <span style="color: var(--text-color-secondary); font-size: 14px;"> ({fmt_int(max_group['count'])} clients)</span>
                                    </span>
                                </div>

                                <div class="disparity-row">
                                    <span style="color: #F59E0B; font-weight: 500;">📉 Lowest:</span>
                                    <span>
                                        <span style="color: var(--text-color); font-weight: bold;">{min_group['group']}</span>
                                        <span style="color: var(--text-color-secondary); font-size: 14px;"> ({fmt_int(min_group['count'])} clients)</span>
                                    </span>
                                </div>
                            </div>

                            <div class="disparity-ratio">
                                <p style="margin: 0; color: #3B82F6; font-size: 16px;">
                                    Disparity Ratio: <strong style="font-size: 20px;">{disparity_ratio:.1f}x</strong>
                                </p>
                            </div>
                        </div>
                    </div>
                    """
                    )
        except Exception as e:
            insights_content.append(
                f"""
            <div style="background-color: var(--background-color); border-radius: 8px; padding: 12px; border: 1px solid var(--border-color);">
                <p style="color: var(--text-color-secondary); font-size: 14px; margin: 0;">
                    ⚠️ Could not calculate group insights: {str(e)}
                </p>
            </div>
            """
            )
    else:
        insights_content.append(
            """
        <div style="background-color: rgba(59, 130, 246, 0.1); border-radius: 8px; padding: 16px; border: 1px solid rgba(59, 130, 246, 0.3); text-align: center;">
            <p style="color: #3B82F6; font-size: 15px; margin: 0;">
                ℹ️ Not enough groups for comparison. Select more groups to see detailed insights.
            </p>
        </div>
        """
        )

    insights_content.append("</div>")
    st.html("".join(insights_content))


# ================================================================================
# MAIN RENDER FUNCTION
# ================================================================================


@st.fragment
def render_trend_explorer(
    df_filt: DataFrame, full_df: Optional[DataFrame] = None
) -> None:
    """Render trend explorer section with enhanced time series analysis."""
    if full_df is None:
        full_df = df_filt.copy()

    # Initialize section state
    state = init_section_state(TREND_SECTION_KEY)
    st.session_state.get("state_summary_metrics", {})

    # Get return window from centralized filter state
    filter_state = st.session_state.get("state_filter_form", {})
    return_window = filter_state.get("return_window", 180)

    # Check if cache is valid
    filter_timestamp = get_filter_timestamp()
    cache_valid = is_cache_valid(state, filter_timestamp)

    if not cache_valid:
        invalidate_cache(state, filter_timestamp)
        # Clear cached data
        for k in list(state.keys()):
            if k not in ["last_updated"]:
                state.pop(k, None)

    # Render header
    _render_section_header()

    # Add metric explanations
    _render_metric_explanation_card()

    # Define available metrics
    metric_opts = {
        "Active Clients": served_clients,
        "Inflow": inflow,
        "Outflow": outflow,
        "PH Exits": ph_exit_clients,
        f"Returns to Homelessness ({return_window}d)": return_after_exit,
    }

    # Create a unique key suffix based on filter timestamp
    key_suffix = hash_data(filter_timestamp)

    # Filter controls
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        # Metrics selection
        metrics_key = f"trend_metrics_{key_suffix}"

        def on_metrics_change():
            state["needs_recalc"] = True
            st.session_state["prev_trend_metrics"] = st.session_state[
                metrics_key
            ]

        # Initialize default metrics
        if "prev_trend_metrics" in st.session_state:
            valid_prev_metrics = [
                m
                for m in st.session_state["prev_trend_metrics"]
                if m in metric_opts
            ]
            default_metrics = (
                valid_prev_metrics
                if valid_prev_metrics
                else ["Active Clients"]
            )
        else:
            default_metrics = ["Active Clients"]

        sel_metrics = st.multiselect(
            "Metrics to display",
            list(metric_opts.keys()),
            default=default_metrics,
            key=metrics_key,
            on_change=on_metrics_change,
            help="Select multiple metrics to compare trends over time",
        )

    with filter_col2:
        # Frequency selection
        freq_key = f"trend_freq_{key_suffix}"

        def on_freq_change():
            new_freq = st.session_state[freq_key]
            state["needs_recalc"] = True
            st.session_state["prev_trend_freq"] = new_freq

            # Adjust rolling window size
            roll_win_key = f"trend_roll_win_{key_suffix}"
            st.session_state[roll_win_key] = SUGGESTED_ROLLING_WINDOWS.get(
                new_freq, 3
            )

        # Initialize frequency
        freq_default = "Months"
        if "prev_trend_freq" in st.session_state:
            freq_default = st.session_state["prev_trend_freq"]

        sel_freq_label = st.selectbox(
            "View by",
            list(FREQUENCY_MAP.keys()),
            index=list(FREQUENCY_MAP.keys()).index(freq_default),
            key=freq_key,
            on_change=on_freq_change,
            help="Choose how to group data points over time",
        )

        sel_freq = FREQUENCY_MAP[sel_freq_label]

    with filter_col3:
        # Breakdown selection
        breakdown_opts = [("None (overall)", None)] + DEMOGRAPHIC_DIMENSIONS

        break_key = f"trend_break_{key_suffix}"

        def on_break_change():
            state["needs_recalc"] = True
            st.session_state["prev_trend_break"] = st.session_state[break_key]
            if "trend_groups_selection" in state:
                state.pop("trend_groups_selection")

        # Initialize breakdown
        break_default = "None (overall)"
        if "prev_trend_break" in st.session_state:
            valid_breaks = [lbl for lbl, _ in breakdown_opts]
            if st.session_state["prev_trend_break"] in valid_breaks:
                break_default = st.session_state["prev_trend_break"]

        break_options = [lbl for lbl, _ in breakdown_opts]

        sel_break = st.selectbox(
            "Compare across",
            break_options,
            index=break_options.index(break_default),
            key=break_key,
            on_change=on_break_change,
            help="Choose a demographic dimension to compare trends across different groups",
        )

        group_col = dict(breakdown_opts)[sel_break]

    # Initialize settings
    roll_win_key = f"trend_roll_win_{key_suffix}"
    if roll_win_key not in st.session_state:
        st.session_state[roll_win_key] = SUGGESTED_ROLLING_WINDOWS.get(
            sel_freq_label, 3
        )

    do_roll = True
    roll_window = st.session_state[roll_win_key]

    display_key = f"trend_display_{key_suffix}"
    if display_key not in st.session_state:
        if "prev_trend_display" in st.session_state:
            st.session_state[display_key] = st.session_state[
                "prev_trend_display"
            ]
        else:
            st.session_state[display_key] = "Combined view"

    display_mode = st.session_state[display_key]
    do_delta = True

    # Define metric interpretations
    metric_interpretation = {
        m: m.startswith("Returns to Homelessness") for m in sel_metrics
    }

    # Get time boundaries
    t0 = st.session_state.get("t0")
    t1 = st.session_state.get("t1")

    if not all([t0, t1]):
        st.warning("Please set date ranges in the filter panel.")
        return

    if not sel_metrics:
        st.info("Please select at least one metric to display.")
        return

    # Generate cache key
    metrics_str = "-".join(sorted(sel_metrics))
    cache_key = (
        f"{metrics_str}_{sel_freq}_{sel_break}_{t0.strftime('%Y-%m-%d')}_"
        f"{t1.strftime('%Y-%m-%d')}_roll{roll_window}_rollon{int(do_roll)}"
    )

    # Check if we need to recalculate
    recalc = state.get("cache_key") != cache_key or state.get(
        "needs_recalc", True
    )

    # Calculate data
    if recalc or "multi_trend_data" not in state:
        state["needs_recalc"] = False
        state["cache_key"] = cache_key

        with st.spinner("Calculating trends..."):
            multi_data = _get_trend_data(
                df_filt,
                full_df,
                metric_opts,
                sel_metrics,
                sel_freq,
                group_col,
                t0,
                t1,
                return_window,
            )

            state["multi_trend_data"] = multi_data
    else:
        multi_data = state.get("multi_trend_data", {})

    # Check if we have data
    if not multi_data or all(
        df.empty for df in multi_data.values() if df is not None
    ):
        st.info("No data available for the selected metrics and time period.")
        return

    # Show divider
    blue_divider()

    # Render visualizations
    if group_col is None:
        # Overall trends
        if display_mode == "Combined view" and len(sel_metrics) > 1:
            # Combined view
            valid_dfs = [
                multi_data[m]
                for m in sel_metrics
                if m in multi_data
                and multi_data[m] is not None
                and not multi_data[m].empty
            ]

            if not valid_dfs:
                st.info("No valid data available for the selected metrics.")
                return

            combined_df = pd.concat(valid_dfs)

            if combined_df.empty:
                st.info("No data available for the selected metrics.")
                return

            # Create combined chart
            fig = _create_combined_trend_chart(
                combined_df,
                sel_metrics,
                sel_freq_label,
                do_roll,
                roll_window,
                multi_data,
            )

            st.plotly_chart(fig, width='stretch')

            # Period changes
            if do_delta:
                period_label = _format_period_label(sel_freq_label)
                with st.expander(
                    f"📊 {period_label.capitalize()}-to-{period_label} Changes",
                    expanded=True,
                ):
                    metric_tabs = st.tabs(sel_metrics)

                    for i, metric_name in enumerate(sel_metrics):
                        if (
                            metric_name in multi_data
                            and multi_data[metric_name] is not None
                            and not multi_data[metric_name].empty
                        ):
                            df = multi_data[metric_name]

                            with metric_tabs[i]:
                                chart_col, insight_col = st.columns([3, 2])

                                increase_is_negative = (
                                    metric_interpretation.get(
                                        metric_name, False
                                    )
                                )
                                is_neutral_metric = (
                                    metric_name in NEUTRAL_METRICS
                                )

                                with chart_col:
                                    fig_delta = _create_delta_chart(
                                        df,
                                        metric_name,
                                        period_label,
                                        is_neutral_metric,
                                        increase_is_negative,
                                        sel_freq,
                                    )
                                    st.plotly_chart(
                                        fig_delta, width='stretch'
                                    )
                                    _render_interpretation_note(
                                        metric_name,
                                        is_neutral_metric,
                                        increase_is_negative,
                                    )

                                with insight_col:
                                    _render_insight_card(
                                        metric_name,
                                        df,
                                        period_label,
                                        increase_is_negative,
                                        is_neutral_metric,
                                    )
        else:
            # Separate charts for each metric
            for metric_name in sel_metrics:
                if (
                    metric_name in multi_data
                    and multi_data[metric_name] is not None
                    and not multi_data[metric_name].empty
                ):
                    df = multi_data[metric_name]

                    with st.expander(
                        f"**{metric_name}** Analysis", expanded=True
                    ):
                        # Calculate dynamic height
                        num_points = len(df)
                        chart_height = _calculate_dynamic_height(
                            num_points, base_height=350
                        )

                        # Create line chart
                        fig = px.line(
                            df,
                            x="bucket",
                            y="count",
                            markers=True,
                            template="plotly_white",
                            title=f"{metric_name} Trend",
                            labels={
                                "bucket": "",
                                "count": "Number of Clients",
                            },
                            color_discrete_sequence=[
                                _get_metric_color(metric_name)
                            ],
                        )

                        # Add rolling average
                        if do_roll:
                            period_label = _format_period_label(sel_freq_label)
                            fig.add_scatter(
                                x=df["bucket"],
                                y=df["rolling"],
                                mode="lines",
                                name=f"{roll_window}-{period_label} average",
                                line=dict(
                                    dash="dash", color=NEUTRAL_COLOR, width=1.5
                                ),
                                opacity=0.7,
                                hovertemplate="<b>%{x|%b %Y}</b><br>Average: %{y:,.1f}<extra></extra>",
                            )

                        # Apply theme-aware styling
                        theme = get_plotly_theme()

                        # Build axis configurations
                        xaxis_config = {
                            "title": "Time Period",
                            "tickformat": (
                                "%b %Y"
                                if sel_freq in ["M", "Q"]
                                else ("%Y" if sel_freq == "Y" else "%b %d")
                            ),
                            "tickangle": _get_x_axis_angle(
                                sel_freq, num_points
                            ),
                        }
                        xaxis_config.update(theme.get("xaxis", {}))

                        yaxis_config = {"title": "Number of Clients"}
                        yaxis_config.update(theme.get("yaxis", {}))

                        fig.update_layout(
                            height=chart_height,
                            margin=_calculate_dynamic_margins(num_points),
                            plot_bgcolor=theme["plot_bgcolor"],
                            paper_bgcolor=theme["paper_bgcolor"],
                            font=theme["font"],
                            xaxis=xaxis_config,
                            yaxis=yaxis_config,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="center",
                                x=0.5,
                            ),
                            hoverlabel=theme.get("hoverlabel", {}),
                        )

                        # Update hover
                        fig.update_traces(
                            hovertemplate=_get_hover_template(sel_freq)
                        )

                        st.plotly_chart(fig, width='stretch')

                        # Additional analysis
                        if len(df) >= 2:
                            col1, col2 = st.columns(2)

                            increase_is_negative = metric_interpretation.get(
                                metric_name, False
                            )
                            period_label = _format_period_label(sel_freq_label)
                            is_neutral_metric = metric_name in NEUTRAL_METRICS

                            with col1:
                                if do_delta:
                                    fig_delta = _create_delta_chart(
                                        df,
                                        metric_name,
                                        period_label,
                                        is_neutral_metric,
                                        increase_is_negative,
                                        sel_freq,
                                    )
                                    st.plotly_chart(
                                        fig_delta, width='stretch'
                                    )
                                    _render_interpretation_note(
                                        metric_name,
                                        is_neutral_metric,
                                        increase_is_negative,
                                    )

                            with col2:
                                _render_insights_panel(
                                    df,
                                    metric_name,
                                    period_label,
                                    increase_is_negative,
                                    is_neutral_metric,
                                )
    else:
        # Demographic breakdown
        for metric_name in sel_metrics:
            if (
                metric_name in multi_data
                and multi_data[metric_name] is not None
                and not multi_data[metric_name].empty
            ):
                df = multi_data[metric_name]

                with st.expander(
                    f"**{metric_name}** by {sel_break}", expanded=True
                ):
                    # Group selection
                    groups_key = f"group_select_{metric_name}_{key_suffix}"

                    def on_groups_change():
                        state["trend_groups_selection"] = {
                            metric_name: st.session_state[groups_key]
                        }

                    # Get all unique groups
                    all_groups = sorted(df["group"].unique())

                    # Determine default groups
                    if len(all_groups) > 8:
                        if (
                            "trend_groups_selection" in state
                            and metric_name in state["trend_groups_selection"]
                        ):
                            default_groups = state["trend_groups_selection"][
                                metric_name
                            ]
                            default_groups = [
                                g for g in default_groups if g in all_groups
                            ]
                            if not default_groups:
                                default_groups = all_groups[:8]
                        else:
                            default_groups = all_groups[:8]

                        st.markdown(f"**Filter {sel_break} groups:**")
                        filtered_groups = st.multiselect(
                            f"Select {sel_break} groups to display",
                            options=all_groups,
                            default=default_groups,
                            key=groups_key,
                            on_change=on_groups_change,
                            help=f"Choose which {sel_break} groups to include in the chart",
                        )

                        if filtered_groups:
                            filtered_df = df[df["group"].isin(filtered_groups)]
                            st.caption(
                                f"Showing {
                                    len(filtered_groups)} of {
                                    len(all_groups)} total {sel_break} groups"
                            )
                        else:
                            st.warning(
                                f"Please select at least one {
                                sel_break} group to display."
                            )
                            continue
                    else:
                        filtered_df = df
                        filtered_groups = all_groups

                    # Display visualization
                    if not filtered_df.empty:
                        # Calculate dynamic height
                        num_points = len(filtered_df["bucket"].unique())
                        num_groups = filtered_df["group"].nunique()
                        chart_height = _calculate_dynamic_height(
                            num_points, base_height=400
                        )

                        if num_groups > 5:
                            chart_height += 50

                        # Build palette
                        palette = CUSTOM_COLOR_SEQUENCE[:num_groups]

                        # Create chart
                        fig = px.line(
                            filtered_df,
                            x="bucket",
                            y="count",
                            color="group",
                            markers=True,
                            template="plotly_white",
                            title=f"{metric_name} by {sel_break}",
                            labels={
                                "bucket": "",
                                "count": "Number of Clients",
                                "group": sel_break,
                            },
                            color_discrete_sequence=palette,
                        )

                        # Apply theme-aware styling
                        theme = get_plotly_theme()

                        # Build axis configurations
                        xaxis_config = {
                            "title": "Time Period",
                            "tickformat": (
                                "%b %Y"
                                if sel_freq in ["M", "Q"]
                                else ("%Y" if sel_freq == "Y" else "%b %d")
                            ),
                            "tickangle": _get_x_axis_angle(
                                sel_freq, num_points
                            ),
                        }
                        xaxis_config.update(theme.get("xaxis", {}))

                        yaxis_config = {"title": "Number of Clients"}
                        yaxis_config.update(theme.get("yaxis", {}))

                        fig.update_layout(
                            height=chart_height,
                            margin=_calculate_dynamic_margins(num_points),
                            plot_bgcolor=theme["plot_bgcolor"],
                            paper_bgcolor=theme["paper_bgcolor"],
                            font=theme["font"],
                            xaxis=xaxis_config,
                            yaxis=yaxis_config,
                            legend=dict(
                                orientation="h" if num_groups <= 5 else "v",
                                yanchor="bottom" if num_groups <= 5 else "top",
                                y=1.02 if num_groups <= 5 else 0.98,
                                xanchor=(
                                    "center" if num_groups <= 5 else "left"
                                ),
                                x=0.5 if num_groups <= 5 else 1.02,
                            ),
                            hoverlabel=theme.get("hoverlabel", {}),
                        )

                        # Update hover
                        fig.update_traces(
                            hovertemplate=(
                                "<b>%{fullData.name}</b><br>"
                                "<b>%{x|%b %Y}</b><br>"
                                "Clients: %{y:,.0f}<extra></extra>"
                            )
                        )

                        st.plotly_chart(fig, width='stretch')

                        # Growth analysis
                        if len(filtered_df["bucket"].unique()) >= 2:
                            try:
                                growth_df = calculate_demographic_growth(
                                    filtered_df
                                )

                                if not growth_df.empty:
                                    tab1, tab2 = st.tabs(
                                        ["📈 Growth Analysis", "📊 Data Table"]
                                    )

                                    with tab1:
                                        insight_col, viz_col = st.columns(
                                            [1, 2]
                                        )

                                        with insight_col:
                                            _render_growth_insights(
                                                growth_df,
                                                filtered_df,
                                                sel_break,
                                            )

                                        with viz_col:
                                            if len(growth_df) >= 2:
                                                # Sort for visualization
                                                growth_vis_df = (
                                                    growth_df.sort_values(
                                                        "growth_pct",
                                                        ascending=False,
                                                    )
                                                )

                                                # Mark significant groups
                                                growth_vis_df[
                                                    "significant"
                                                ] = (
                                                    growth_vis_df[
                                                        "first_count"
                                                    ]
                                                    >= 5
                                                )

                                                # Limit groups for clarity
                                                MAX_GROUPS_TO_SHOW = 12
                                                if (
                                                    len(growth_vis_df)
                                                    > MAX_GROUPS_TO_SHOW
                                                ):
                                                    growth_vis_df = pd.concat(
                                                        [
                                                            growth_vis_df.head(
                                                                6
                                                            ),
                                                            growth_vis_df.tail(
                                                                6
                                                            ),
                                                        ]
                                                    )
                                                    showing_subset = True
                                                else:
                                                    showing_subset = False

                                                # Calculate dynamic height
                                                growth_height = (
                                                    _calculate_dynamic_height(
                                                        len(growth_vis_df),
                                                        base_height=500,
                                                    )
                                                )

                                                # Create growth chart
                                                fig_growth = go.Figure()

                                                # Pre-calculate colors and
                                                # opacities for better
                                                # performance
                                                def get_growth_color(
                                                    growth_pct,
                                                ):
                                                    if growth_pct > 50:
                                                        return (
                                                            theme_config.colors.success
                                                        )
                                                    elif growth_pct > 0:
                                                        return (
                                                            theme_config.colors.success_light
                                                        )
                                                    elif growth_pct > -50:
                                                        return (
                                                            theme_config.colors.warning
                                                        )
                                                    else:
                                                        return (
                                                            theme_config.colors.danger
                                                        )

                                                colors = growth_vis_df[
                                                    "growth_pct"
                                                ].apply(get_growth_color)
                                                opacities = growth_vis_df[
                                                    "significant"
                                                ].apply(
                                                    lambda x: 1.0 if x else 0.6
                                                )

                                                # Add bars with pre-calculated
                                                # properties
                                                for (
                                                    (idx, row),
                                                    color,
                                                    opacity,
                                                ) in zip(
                                                    growth_vis_df.iterrows(),
                                                    colors,
                                                    opacities,
                                                ):
                                                    fig_growth.add_trace(
                                                        go.Bar(
                                                            x=[row["group"]],
                                                            y=[
                                                                row[
                                                                    "growth_pct"
                                                                ]
                                                            ],
                                                            marker_color=color,
                                                            opacity=opacity,
                                                            hovertemplate=(
                                                                f"<b>{row['group']}</b><br>"
                                                                f"<br>"
                                                                f"<b>Growth Rate:</b> {row['growth_pct']:+.1f}%<br>"
                                                                f"<br>"
                                                                f"<b>Start Period:</b> {int(row['first_count'])} clients<br>"
                                                                f"<b>End Period:</b> {int(row['last_count'])} clients<br>"
                                                                f"<b>Net Change:</b> {int(row['growth'])} clients<br>"
                                                                f"<extra></extra>"
                                                            ),
                                                            showlegend=False,
                                                        )
                                                    )

                                                # Update layout with theme
                                                # support
                                                theme = get_plotly_theme()

                                                # Build axis configurations
                                                xaxis_config = {
                                                    "title": dict(
                                                        text=sel_break,
                                                        font=dict(size=14),
                                                    ),
                                                    "tickangle": -45,
                                                    "automargin": True,
                                                    "tickfont": dict(size=11),
                                                }
                                                xaxis_config.update(
                                                    theme.get("xaxis", {})
                                                )

                                                yaxis_config = {
                                                    "title": dict(
                                                        text="Growth Rate (%)",
                                                        font=dict(size=14),
                                                    ),
                                                    "automargin": True,
                                                    "zeroline": True,
                                                    "zerolinewidth": 2,
                                                    "zerolinecolor": "rgba(0, 0, 0, 0.3)",
                                                    "range": [
                                                        min(
                                                            growth_vis_df[
                                                                "growth_pct"
                                                            ].min()
                                                            * 1.2,
                                                            -10,
                                                        ),
                                                        max(
                                                            growth_vis_df[
                                                                "growth_pct"
                                                            ].max()
                                                            * 1.2,
                                                            10,
                                                        ),
                                                    ],
                                                }
                                                yaxis_config.update(
                                                    theme.get("yaxis", {})
                                                )

                                                fig_growth.update_layout(
                                                    title=dict(
                                                        text=f"Growth Rates by {sel_break}",
                                                        font=dict(size=18),
                                                        y=0.98,
                                                    ),
                                                    template="plotly_white",
                                                    height=growth_height,
                                                    margin=_calculate_dynamic_margins(
                                                        len(growth_vis_df),
                                                        has_text_labels=True,
                                                    ),
                                                    plot_bgcolor=theme[
                                                        "plot_bgcolor"
                                                    ],
                                                    paper_bgcolor=theme[
                                                        "paper_bgcolor"
                                                    ],
                                                    font=theme["font"],
                                                    xaxis=xaxis_config,
                                                    yaxis=yaxis_config,
                                                    hoverlabel=theme.get(
                                                        "hoverlabel", {}
                                                    ),
                                                    bargap=0.3,
                                                )

                                                # Add value labels
                                                for i, trace in enumerate(
                                                    fig_growth.data
                                                ):
                                                    y_val = trace.y[0]
                                                    if abs(y_val) > 30:
                                                        fig_growth.data[
                                                            i
                                                        ].update(
                                                            text=[
                                                                f"{y_val:+.0f}%"
                                                            ],
                                                            textposition="inside",
                                                            textfont=dict(
                                                                color="white",
                                                                size=12,
                                                                weight="bold",
                                                            ),
                                                        )
                                                    else:
                                                        fig_growth.data[
                                                            i
                                                        ].update(
                                                            text=[
                                                                f"{y_val:+.0f}%"
                                                            ],
                                                            textposition="outside",
                                                            textfont=dict(
                                                                size=10
                                                            ),
                                                        )

                                                # Add annotations
                                                if showing_subset:
                                                    fig_growth.add_annotation(
                                                        text=f"📊 Showing top 6 and bottom 6 of {
                                                            len(growth_df)} total groups for clarity",
                                                        xref="paper",
                                                        yref="paper",
                                                        x=0.5,
                                                        y=-0.32,
                                                        showarrow=False,
                                                        font=dict(
                                                            size=13,
                                                            weight="bold",
                                                        ),
                                                        bgcolor="rgba(255, 255, 255, 0.8)",
                                                        bordercolor="rgba(0, 0, 0, 0.2)",
                                                        borderwidth=1,
                                                        borderpad=8,
                                                    )

                                                if not all(
                                                    growth_vis_df[
                                                        "significant"
                                                    ]
                                                ):
                                                    fig_growth.add_annotation(
                                                        text="ℹ️ Groups with fewer than 5 clients shown with reduced opacity",
                                                        xref="paper",
                                                        yref="paper",
                                                        x=0.5,
                                                        y=1.08,
                                                        showarrow=False,
                                                        font=dict(
                                                            size=11,
                                                            color="gray",
                                                        ),
                                                        bgcolor="rgba(255, 255, 255, 0.8)",
                                                        borderpad=4,
                                                    )

                                                st.plotly_chart(
                                                    fig_growth,
                                                    width='stretch',
                                                )
                                            else:
                                                st.info(
                                                    "Select at least two groups to see growth comparisons."
                                                )

                                    with tab2:
                                        if not growth_df.empty:
                                            # Format for display
                                            display_df = growth_df.copy()
                                            display_df["Growth"] = display_df[
                                                "growth"
                                            ].map(lambda x: fmt_int(x))
                                            display_df["Growth %"] = (
                                                display_df["growth_pct"].map(
                                                    lambda x: fmt_pct(x)
                                                )
                                            )
                                            display_df = display_df.rename(
                                                columns={
                                                    "group": sel_break,
                                                    "first_count": "Initial Value",
                                                    "last_count": "Latest Value",
                                                }
                                            )

                                            # Format count columns
                                            display_df["Initial Value"] = (
                                                display_df[
                                                    "Initial Value"
                                                ].map(fmt_int)
                                            )
                                            display_df["Latest Value"] = (
                                                display_df["Latest Value"].map(
                                                    fmt_int
                                                )
                                            )

                                            # Show table
                                            st.dataframe(
                                                display_df[
                                                    [
                                                        sel_break,
                                                        "Initial Value",
                                                        "Latest Value",
                                                        "Growth",
                                                        "Growth %",
                                                    ]
                                                ],
                                                width='stretch',
                                                hide_index=True,
                                            )
                                        else:
                                            st.info(
                                                "No growth data available."
                                            )
                            except Exception as e:
                                st.warning(
                                    f"Could not calculate growth trends: {
                                        str(e)}"
                                )
                    else:
                        st.info(
                            "No data available for the selected groups. Please adjust your selection."
                        )
