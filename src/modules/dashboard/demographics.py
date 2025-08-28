"""
Demographic breakdown section
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas import DataFrame

from src.core.data.destinations import apply_custom_ph_destinations
from src.modules.dashboard.data_utils import (
    DEMOGRAPHIC_DIMENSIONS,
    category_counts,
    inflow,
    outflow,
    ph_exit_clients,
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
from src.ui.factories.components import fmt_int, ui
from src.ui.factories.html import html_factory
from src.ui.themes.theme import (
    CUSTOM_COLOR_SEQUENCE,
    PLOT_TEMPLATE,
    WARNING_COLOR,
    blue_divider,
    theme,
)

# ==============================================================================
# CONSTANTS
# ==============================================================================
BREAKDOWN_SECTION_KEY = "demographic_breakdown"

# ==============================================================================
# DATA CALCULATION FUNCTIONS
# ==============================================================================


def _calculate_breakdown_data(
    df_filt: DataFrame,
    full_df: DataFrame,
    dim_col: str,
    t0,
    t1,
    return_window: int = 180,
) -> DataFrame:
    """
    Calculate demographic breakdown data with improved metrics.

    Parameters:
    -----------
    df_filt : DataFrame
        Filtered DataFrame
    full_df : DataFrame
        Full DataFrame for returns analysis
    dim_col : str
        Demographic dimension column name
    t0 : Timestamp
        Start date
    t1 : Timestamp
        End date
    return_window : int, optional
        Days to check for returns

    Returns:
    --------
    DataFrame
        DataFrame with breakdown metrics
    """
    # Apply custom PH destinations to both dataframes
    df_filt = apply_custom_ph_destinations(df_filt, force=True)
    full_df = apply_custom_ph_destinations(full_df, force=True)

    # Get client sets for different metrics
    served_ids = served_clients(df_filt, t0, t1)
    inflow_ids = inflow(df_filt, t0, t1)
    outflow_ids = outflow(df_filt, t0, t1)
    ph_ids = ph_exit_clients(df_filt, t0, t1)

    # Get ALL unique clients who exited for PH exit rate calculation
    all_exits_mask = df_filt["ProjectExit"].between(t0, t1)
    all_exit_ids = set(df_filt.loc[all_exits_mask, "ClientID"])

    # Calculate returns using HUD-compliant logic with FULL dataset
    # First, get the PH exits subset for returns calculation
    ph_exits_mask = (df_filt["ProjectExit"].between(t0, t1)) & (
        df_filt["ExitDestinationCat"] == "Permanent Housing Situations"
    )
    ph_exits_df = df_filt[ph_exits_mask]
    ph_exits_ids = set(ph_exits_df["ClientID"].unique())

    # Calculate returns only for those who exited to PH
    return_ids = return_after_exit(ph_exits_df, full_df, t0, t1, return_window)

    # Validate returns are subset of PH exits
    if not return_ids.issubset(ph_exits_ids):
        # WARNING: Some returns may not be in PH exits set
        return_ids = return_ids.intersection(ph_exits_ids)

    # Calculate counts by demographic dimension - ensuring unique clients
    bdf = (
        pd.concat(
            [
                category_counts(df_filt, served_ids, dim_col, "Served"),
                category_counts(df_filt, inflow_ids, dim_col, "Inflow"),
                category_counts(df_filt, outflow_ids, dim_col, "Outflow"),
                category_counts(df_filt, all_exit_ids, dim_col, "Total Exits"),
                category_counts(df_filt, ph_ids, dim_col, "PH Exits"),
            ],
            axis=1,
        )
        .fillna(0)
        .reset_index()
        .rename(columns={"index": dim_col})
    )

    # Skip if no data
    if bdf.empty:
        return pd.DataFrame()

    # FIXED: Calculate PH Exit Rate using vectorized operations
    ph_exit_rates = (
        (bdf["PH Exits"] / bdf["Total Exits"].replace(0, 1) * 100)
        .where(bdf["Total Exits"] > 0, 0.0)
        .round(1)
        .tolist()
    )

    bdf["PH Exit Rate"] = ph_exit_rates

    # FIXED: Prepare returns by demographic using validated return IDs
    # Get demographic info for clients
    clients_demo = df_filt[["ClientID", dim_col]].drop_duplicates()

    # Filter to PH exits and returns
    ph_demo = clients_demo[clients_demo["ClientID"].isin(ph_exits_ids)]
    ret_demo = clients_demo[clients_demo["ClientID"].isin(return_ids)]

    # Get counts by group
    ph_counts = (
        ph_demo.groupby(dim_col, observed=True)["ClientID"]
        .nunique()
        .reset_index(name="PH Exit Count")
    )
    ret_counts = (
        ret_demo.groupby(dim_col, observed=True)["ClientID"]
        .nunique()
        .reset_index(name="Returns Count")
    )

    # Merge return counts
    returns_df = pd.merge(ph_counts, ret_counts, on=dim_col, how="left")
    returns_df["PH Exit Count"] = (
        returns_df["PH Exit Count"].fillna(0).astype(int)
    )
    returns_df["Returns Count"] = (
        returns_df["Returns Count"].fillna(0).astype(int)
    )

    # Calculate Returns Rate with proper validation
    returns_df["Returns to Homelessness Rate"] = returns_df.apply(
        lambda row: (
            round((row["Returns Count"] / row["PH Exit Count"]) * 100, 1)
            if row["PH Exit Count"] > 0
            else np.nan
        ),
        axis=1,
    )

    # Merge with main breakdown dataframe
    bdf = pd.merge(bdf, returns_df, on=dim_col, how="left")

    # Fill any missing values appropriately
    bdf["PH Exit Count"] = bdf["PH Exit Count"].fillna(0).astype(int)
    bdf["Returns Count"] = bdf["Returns Count"].fillna(0).astype(int)
    bdf["Returns to Homelessness Rate"] = bdf[
        "Returns to Homelessness Rate"
    ].fillna(np.nan)

    # Add net flow column
    bdf["Net Flow"] = bdf["Inflow"] - bdf["Outflow"]

    # Ensure all numeric columns are properly typed
    numeric_cols = [
        "Served",
        "Inflow",
        "Outflow",
        "Total Exits",
        "PH Exits",
        "PH Exit Count",
        "Returns Count",
        "Net Flow",
    ]
    for col in numeric_cols:
        if col in bdf.columns:
            bdf[col] = bdf[col].astype(int)

    return bdf


# ==============================================================================
# VISUALIZATION FUNCTIONS - Count and Flow Charts
# ==============================================================================


def _create_counts_chart(df: DataFrame, dim_col: str) -> go.Figure:
    """
    Create counts chart for demographic breakdown with improved layout.
    """
    # Determine optimal chart height based on number of groups
    num_groups = len(df[dim_col].unique())
    chart_height = max(500, min(900, 450 + (num_groups * 25)))

    # Reshape for plotting - exclude Total Exits from visual to avoid clutter
    counts_df = df.melt(
        id_vars=dim_col,
        value_vars=[
            "Served",
            "Inflow",
            "Outflow",
            "PH Exits",
            "Returns Count",
        ],
        var_name="Metric",
        value_name="Count",
    )

    # Calculate label characteristics
    labels = df[dim_col].astype(str).tolist()
    max_label_length = max(len(label) for label in labels)

    # For very long labels, use horizontal bar chart instead
    if max_label_length > 50 or num_groups > 15:
        # Create horizontal bar chart
        fig = px.bar(
            counts_df,
            y=dim_col,  # Note: x and y are swapped
            x="Count",
            color="Metric",
            orientation="h",
            barmode="group",
            template=PLOT_TEMPLATE,
            text_auto=".0f",
            title=f"Client Count by {dim_col}",
            color_discrete_sequence=CUSTOM_COLOR_SEQUENCE,
        )

        # Update text positioning
        fig.update_traces(textposition="outside", textfont=dict(size=9))

        # Calculate left margin based on label length
        left_margin = max(150, min(400, 100 + (max_label_length * 5)))

        # Update layout
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.1)",
                bordercolor="rgba(128, 128, 128, 0.3)",
                borderwidth=1,
                font=dict(size=11),
            ),
            margin=dict(l=left_margin, r=80, t=120, b=80),
            height=chart_height,
            bargap=0.2,
            bargroupgap=0.1,
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            title=dict(font=dict(size=16)),
        )

        fig.update_xaxes(
            title="Count",
            automargin=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            title_font=dict(size=14),
            tickfont=dict(size=12),
        )

        fig.update_yaxes(
            title="",
            automargin=True,
            tickmode="linear",
            dtick=1,
            tickfont=dict(size=11),
        )

    else:
        # Create vertical bar chart with aggressive label handling
        fig = px.bar(
            counts_df,
            x=dim_col,
            y="Count",
            color="Metric",
            barmode="group",
            template=PLOT_TEMPLATE,
            text_auto=".0f",
            title=f"Client Count by {dim_col}",
            color_discrete_sequence=CUSTOM_COLOR_SEQUENCE,
        )

        # Update text positioning
        fig.update_traces(textposition="outside", textfont=dict(size=10))

        # Smart label wrapping for vertical charts
        if max_label_length > 25:
            wrapped_labels = []
            for label in labels:
                if len(label) > 25:
                    # Prioritize natural break points
                    if "," in label:
                        parts = label.split(",", 1)
                        wrapped = parts[0].strip() + ",<br>" + parts[1].strip()
                    elif " or " in label:
                        wrapped = label.replace(" or ", "<br>or ")
                    elif "/" in label:
                        parts = label.split("/", 1)
                        wrapped = parts[0].strip() + "/<br>" + parts[1].strip()
                    elif " - " in label:
                        parts = label.split(" - ", 1)
                        wrapped = (
                            parts[0].strip() + " -<br>" + parts[1].strip()
                        )
                    else:
                        # Find best space to break at
                        words = label.split()
                        if len(words) > 1:
                            mid_point = len(label) // 2
                            best_break = 0
                            min_diff = float("inf")

                            current_pos = 0
                            for i, word in enumerate(words[:-1]):
                                current_pos += len(word) + 1
                                diff = abs(current_pos - mid_point)
                                if diff < min_diff:
                                    min_diff = diff
                                    best_break = i

                            wrapped = (
                                " ".join(words[: best_break + 1])
                                + "<br>"
                                + " ".join(words[best_break + 1 :])
                            )
                        else:
                            wrapped = label
                else:
                    wrapped = label
                wrapped_labels.append(wrapped)

            # Create mapping for wrapped labels
            label_mapping = dict(zip(df[dim_col], wrapped_labels))
            counts_df = counts_df.copy()
            counts_df[dim_col] = counts_df[dim_col].map(label_mapping)

        # Calculate rotation and margins
        if num_groups <= 4 and max_label_length <= 20:
            rotation_angle = 0
            bottom_margin = 120
        elif num_groups <= 6 and max_label_length <= 30:
            rotation_angle = -30
            bottom_margin = 150
        elif num_groups <= 10:
            rotation_angle = -45
            bottom_margin = 200
        else:
            rotation_angle = -60
            bottom_margin = 250

        # Add extra margin for wrapped labels
        if max_label_length > 25:
            bottom_margin += 50

        # Update layout
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.1)",
                bordercolor="rgba(128, 128, 128, 0.3)",
                borderwidth=1,
                font=dict(size=11),
            ),
            margin=dict(l=80, r=80, t=120, b=bottom_margin),
            height=chart_height,
            bargap=0.2,
            bargroupgap=0.1,
            xaxis=dict(
                tickangle=rotation_angle,
                automargin=True,
                title_standoff=30,
                tickmode="linear",
                dtick=1,
                tickfont=dict(size=11 if max_label_length > 40 else 12),
            ),
            yaxis=dict(
                title_standoff=20,
                automargin=True,
                rangemode="tozero",
                gridcolor="rgba(128, 128, 128, 0.2)",
                title_font=dict(size=14),
                tickfont=dict(size=12),
            ),
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            title=dict(font=dict(size=16)),
        )

    return fig


def _create_flow_balance_chart(df: DataFrame, dim_col: str) -> go.Figure:
    """
    Create a chart showing net flow (inflow - outflow) by demographic.
    """
    # Sort by net flow for visual impact
    flow_df = df.sort_values("Net Flow", ascending=True)

    # Color bars based on positive/negative with gradient
    colors = []
    max_abs_flow = (
        flow_df["Net Flow"].abs().max()
        if flow_df["Net Flow"].abs().max() > 0
        else 1
    )

    for x in flow_df["Net Flow"]:
        if x >= 0:
            # Green gradient for positive
            intensity = min(abs(x) / max_abs_flow, 1)
            colors.append(f"rgba(34, 139, 34, {0.4 + intensity * 0.6})")
        else:
            # Red gradient for negative
            intensity = min(abs(x) / max_abs_flow, 1)
            colors.append(f"rgba(220, 38, 38, {0.4 + intensity * 0.6})")

    # Create horizontal bar chart for better label visibility
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=flow_df["Net Flow"],
            y=flow_df[dim_col],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255, 255, 255, 0.8)", width=1),
            ),
            text=[f"{x:+,.0f}" for x in flow_df["Net Flow"]],
            textposition="outside",
            textfont=dict(size=11),
            hovertemplate="<b>%{y}</b><br>Net Flow: %{x:+,}<br>Inflow: %{customdata[0]:,}<br>Outflow: %{customdata[1]:,}<extra></extra>",
            customdata=np.column_stack(
                (flow_df["Inflow"], flow_df["Outflow"])
            ),
        )
    )

    # Calculate height
    num_groups = len(flow_df)
    chart_height = max(400, min(800, 350 + (num_groups * 30)))

    # Update layout with better styling
    fig.update_layout(
        title={
            "text": "System Flow Balance by Group",
            "x": 0.5,
            "xanchor": "center",
            "font": dict(size=16),
        },
        template=PLOT_TEMPLATE,
        height=chart_height,
        showlegend=False,
        xaxis=dict(
            title="Net Flow (Inflow - Outflow)",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="rgba(128, 128, 128, 0.3)",
            gridcolor="rgba(128, 128, 128, 0.1)",
            automargin=True,
            title_font=dict(size=14),
            tickfont=dict(size=12),
        ),
        yaxis=dict(automargin=True, title="", tickfont=dict(size=11)),
        margin=dict(
            l=max(
                150, 50 + (flow_df[dim_col].astype(str).str.len().max() * 8)
            ),
            r=80,
            t=80,
            b=80,
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0.02)",
    )

    # Add annotation for context with improved styling
    total_net = flow_df["Net Flow"].sum()
    annotation_color = (
        "rgba(34, 139, 34, 0.8)"
        if total_net >= 0
        else "rgba(220, 38, 38, 0.8)"
    )

    fig.add_annotation(
        text=f"<b>Total System Net Flow: {total_net:+,}</b>",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.05,
        showarrow=False,
        font=dict(size=14, color=annotation_color),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor=annotation_color,
        borderwidth=2,
        borderpad=8,
    )

    # Add subtle reference line at zero
    fig.add_vline(
        x=0,
        line_color="rgba(128, 128, 128, 0.5)",
        line_width=1,
        line_dash="dot",
    )

    return fig


# ==============================================================================
# VISUALIZATION FUNCTIONS - Rate Charts
# ==============================================================================


def _create_rates_charts(
    df: pd.DataFrame, dim_col: str, return_window: int
) -> Tuple[go.Figure, go.Figure]:
    """
    Create rate charts for PH exits and returns with improved layout,
    excluding groups with 0% PH Exit Rate, and ensuring the returns
    chart shows the same groups (even if their return rate is zero).

    Parameters:
    -----------
    df : pd.DataFrame
        Breakdown data containing these columns:
        - dim_col
        - "PH Exit Rate", "PH Exits", "Total Exits"
        - "Returns to Homelessness Rate", "Returns Count", "PH Exit Count"
    dim_col : str
        Demographic dimension column name.
    return_window : int
        Days to check for returns.

    Returns:
    --------
    Tuple[go.Figure, go.Figure]
        (PH exit rate figure, returns rate figure)

    Raises:
    -------
    ValueError
        If any required column is missing.
    """
    # 1) Validate required columns
    required = {
        dim_col,
        "PH Exit Rate",
        "PH Exits",
        "Total Exits",
        "Returns to Homelessness Rate",
        "Returns Count",
        "PH Exit Count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for rate charts: {missing}")

    # 2) Filter out groups with 0% PH Exit Rate
    ph_df = df[df["PH Exit Rate"] > 0].copy()
    if ph_df.empty:
        # No groups to show
        empty_ph = go.Figure()
        empty_ph.update_layout(
            title="Permanent Housing Exit Rate (%)",
            template=PLOT_TEMPLATE,
            height=400,
            annotations=[
                {
                    "text": "No PH exit data available",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
        )
        # Also produce a matching empty returns chart
        empty_ret = go.Figure()
        empty_ret.update_layout(
            title=f"Returns to Homelessness within {return_window} days (%)",
            template=PLOT_TEMPLATE,
            height=400,
            annotations=[
                {
                    "text": "No returns data available",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
        )
        return empty_ph, empty_ret

    # 3) Prepare labels and layout parameters
    def process_labels_for_display(labels: List[str]) -> List[str]:
        """Abbreviate very long labels intelligently."""
        proc: List[str] = []
        max_len = max(len(str(label)) for label in labels)
        for lbl in labels:
            s = str(lbl)
            if max_len > 40 and len(s) > 40:
                if "American Indian, Alaska Native, or Indigenous" in s:
                    proc.append("AI/AN/Indigenous")
                elif "Black, African American, or African" in s:
                    proc.append("Black/African American")
                elif "Hispanic/Latina/e/o" in s:
                    proc.append("Hispanic/Latino")
                elif "," in s:
                    proc.append(s.split(",", 1)[0] + "...")
                else:
                    proc.append(s[:35] + "...")
            else:
                proc.append(s)
        return proc

    groups = sorted(ph_df[dim_col].dropna().unique())
    display_labels = process_labels_for_display(groups)
    display_map = dict(zip(groups, display_labels))
    n = len(groups)
    max_lbl = max((len(lbl) for lbl in display_labels), default=0)
    horizontal = n > 8 or max_lbl > 30

    if horizontal:
        height = max(400, min(800, 300 + n * 35))
        left_margin = min(400, max(150, max_lbl * 7))
        layout = dict(
            height=height,
            margin=dict(l=left_margin, r=80, t=100, b=80),
            showlegend=False,
        )
    else:
        height = max(450, min(700, 400 + n * 30))
        if n <= 3 and max_lbl <= 15:
            angle, bottom = 0, 100
        elif n <= 5 and max_lbl <= 20:
            angle, bottom = -30, 150
        else:
            angle = -45
            vs = max_lbl * 7 * abs(math.sin(math.radians(45)))
            bottom = max(200, int(vs + 80))
        layout = {
            "height": height,
            "margin": dict(l=60, r=60, t=100, b=bottom),
            "showlegend": False,
            "xaxis": dict(
                tickangle=angle,
                automargin=False,
                tickmode="linear",
                dtick=1,
                tickfont=dict(size=10 if max_lbl > 25 else 11),
            ),
        }

    # 4) Build PH Exit Rate chart
    ph_plot = ph_df.sort_values(dim_col).copy()
    ph_plot["display_label"] = ph_plot[dim_col].map(display_map)

    if horizontal:
        fig_ph = px.bar(
            ph_plot,
            y="display_label",
            x="PH Exit Rate",
            orientation="h",
            template=PLOT_TEMPLATE,
            title="Permanent Housing Exit Rate (%)",
            color="PH Exit Rate",
            color_continuous_scale="Blues",
        )
        fig_ph.update_layout(
            **layout,
            xaxis=dict(
                range=[0, max(100, ph_plot["PH Exit Rate"].max() * 1.2)]
            ),
        )
        fig_ph.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    else:
        fig_ph = px.bar(
            ph_plot,
            x="display_label",
            y="PH Exit Rate",
            template=PLOT_TEMPLATE,
            title="Permanent Housing Exit Rate (%)",
            color="PH Exit Rate",
            color_continuous_scale="Blues",
        )
        fig_ph.update_layout(
            **layout,
            yaxis=dict(
                range=[0, max(100, ph_plot["PH Exit Rate"].max() * 1.3)],
                automargin=True,
            ),
        )
        fig_ph.update_traces(texttemplate="%{y:.1f}%", textposition="outside")

    fig_ph.update_traces(
        customdata=ph_plot[
            [dim_col, "PH Exit Rate", "PH Exits", "Total Exits"]
        ].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "PH Exit Rate: %{customdata[1]:.1f}%<br>"
            "PH Exits: %{customdata[2]}<br>"
            "Total Exits: %{customdata[3]}<extra></extra>"
        ),
    )

    # 5) Build Returns Rate chart with same groups, filling zeros
    ret_df = df.copy()
    ret_df["Returns to Homelessness Rate"] = ret_df[
        "Returns to Homelessness Rate"
    ].fillna(0)
    ret_plot = ret_df[ret_df[dim_col].isin(groups)].sort_values(dim_col).copy()
    ret_plot["display_label"] = ret_plot[dim_col].map(display_map)

    if horizontal:
        fig_ret = px.bar(
            ret_plot,
            y="display_label",
            x="Returns to Homelessness Rate",
            orientation="h",
            template=PLOT_TEMPLATE,
            title=f"Returns to Homelessness within {return_window} days (%)",
            color="Returns to Homelessness Rate",
            color_continuous_scale="Reds",
        )
        fig_ret.update_layout(
            **layout,
            xaxis=dict(
                range=[
                    0,
                    max(
                        50,
                        ret_plot["Returns to Homelessness Rate"].max() * 1.3,
                    ),
                ]
            ),
        )
        fig_ret.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    else:
        fig_ret = px.bar(
            ret_plot,
            x="display_label",
            y="Returns to Homelessness Rate",
            template=PLOT_TEMPLATE,
            title=f"Returns to Homelessness within {return_window} days (%)",
            color="Returns to Homelessness Rate",
            color_continuous_scale="Reds",
        )
        fig_ret.update_layout(
            **layout,
            yaxis=dict(
                range=[
                    0,
                    max(
                        50,
                        ret_plot["Returns to Homelessness Rate"].max() * 1.3,
                    ),
                ],
                automargin=True,
            ),
        )
        fig_ret.update_traces(texttemplate="%{y:.1f}%", textposition="outside")

    fig_ret.update_traces(
        customdata=ret_plot[
            [
                dim_col,
                "Returns to Homelessness Rate",
                "Returns Count",
                "PH Exit Count",
            ]
        ].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Returns (" + str(return_window) + "d): %{customdata[1]:.1f}%<br>"
            "Returns Count: %{customdata[2]}<br>"
            "PH Exits: %{customdata[3]}<extra></extra>"
        ),
    )

    return fig_ph, fig_ret


# ==============================================================================
# VISUALIZATION FUNCTIONS - Quadrant Charts
# ==============================================================================


def _create_outcome_quadrant_chart(df: DataFrame, dim_col: str) -> go.Figure:
    """
    Create outcome quadrant chart comparing PH exits and returns with improved layout.

    Parameters:
    -----------
    df : DataFrame
        Breakdown data
    dim_col : str
        Demographic dimension column name

    Returns:
    --------
    Figure
        Plotly figure
    """
    # Filter to groups with both metrics
    comparison_df = df.dropna(
        subset=["PH Exit Rate", "Returns to Homelessness Rate"]
    )

    if comparison_df.empty:
        # Create empty figure if no comparison data
        fig = go.Figure()
        fig.update_layout(
            title="PH Exit vs Return Rate: Not enough data",
            template=PLOT_TEMPLATE,
            annotations=[
                {
                    "text": "Insufficient data for comparison",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": dict(size=14, color="rgba(128, 128, 128, 0.8)"),
                }
            ],
            height=600,
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
        )
        return fig

    # Averages for quadrant lines
    avg_exit_rate = comparison_df["PH Exit Rate"].mean()
    avg_return_rate = comparison_df["Returns to Homelessness Rate"].mean()

    # Normalize bubble sizes for better visibility
    min_served = comparison_df["Served"].min()
    max_served = comparison_df["Served"].max()
    size_range = max_served - min_served if max_served > min_served else 1

    # Create normalized sizes (15-70 range for better visibility)
    comparison_df["normalized_size"] = (
        15 + ((comparison_df["Served"] - min_served) / size_range) * 55
    )

    # Create more distinct color palette for groups
    n_groups = len(comparison_df[dim_col].unique())

    # Define a more distinct color palette with good contrast in both themes
    distinct_colors = [
        "#FF6B6B",  # Bright red
        "#4ECDC4",  # Turquoise
        "#45B7D1",  # Sky blue
        "#96CEB4",  # Mint green
        "#DDA0DD",  # Plum
        "#FFA07A",  # Light salmon
        "#98D8C8",  # Seafoam
        "#F7DC6F",  # Light yellow
        "#BB8FCE",  # Light purple
        "#85C1E2",  # Light blue
        "#F8B739",  # Golden yellow
        "#52BE80",  # Medium green
        "#EC7063",  # Soft red
        "#5DADE2",  # Bright blue
        "#45B39D",  # Teal
        "#F5B041",  # Orange
        "#AF7AC5",  # Purple
        "#48C9B0",  # Turquoise green
        "#F1948A",  # Coral
        "#85929E",  # Blue gray
    ]

    # Use custom colors if available, otherwise generate from color scale
    if n_groups <= len(distinct_colors):
        colors = distinct_colors[:n_groups]
    else:
        # Generate additional colors using HSL color space for maximum
        # distinction
        import colorsys

        colors = []
        for i in range(n_groups):
            hue = i / n_groups
            # Use varying saturation and lightness for more distinction
            saturation = 0.7 + (0.3 * (i % 3) / 2)
            lightness = 0.5 + (0.2 * (i % 5) / 4)
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            hex_color = f"#{
                int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"
            colors.append(hex_color)

    # Create scatter plot with improved styling
    fig = go.Figure()

    # Sort groups for consistent color assignment
    sorted_groups = sorted(comparison_df[dim_col].unique())
    color_map = dict(zip(sorted_groups, colors))

    # Add scatter points
    for group_name in sorted_groups:
        group_data = comparison_df[comparison_df[dim_col] == group_name]

        fig.add_trace(
            go.Scatter(
                x=group_data["PH Exit Rate"],
                y=group_data["Returns to Homelessness Rate"],
                mode="markers",
                name=str(group_name),
                marker=dict(
                    size=group_data["normalized_size"],
                    color=color_map[group_name],
                    line=dict(width=2, color="rgba(255, 255, 255, 0.5)"),
                    opacity=0.9,
                ),
                text=[
                    f"<b>{group_name}</b><br>"
                    f"PH Exit Rate: {getattr(row, 'PH_Exit_Rate', 0):.1f}%<br>"
                    f"Return Rate: {getattr(row, 'Returns_to_Homelessness_Rate', 0):.1f}%<br>"
                    f"Served: {getattr(row, 'Served', 0):,}<br>"
                    f"PH Exits: {getattr(row, 'PH_Exits', 0):,}<br>"
                    f"Returns: {getattr(row, 'Returns_Count', 0):,}"
                    for row in group_data.itertuples()
                ],
                hovertemplate="%{text}<extra></extra>",
                hoverlabel=dict(
                    bgcolor="white",
                    bordercolor=color_map[group_name],
                    font=dict(color="black", size=12),
                ),
            )
        )

    # Add shaded quadrant backgrounds
    x_min = max(0, comparison_df["PH Exit Rate"].min() - 5)
    x_max = min(100, comparison_df["PH Exit Rate"].max() + 5)
    y_min = max(0, comparison_df["Returns to Homelessness Rate"].min() - 2)
    y_max = min(100, comparison_df["Returns to Homelessness Rate"].max() + 5)

    # Add subtle background shading for quadrants
    # Bottom-right (ideal) - green tint
    fig.add_shape(
        type="rect",
        x0=avg_exit_rate,
        x1=x_max,
        y0=y_min,
        y1=avg_return_rate,
        fillcolor="rgba(76, 175, 80, 0.08)",
        line=dict(width=0),
        layer="below",
    )

    # Top-left (concern) - red tint
    fig.add_shape(
        type="rect",
        x0=x_min,
        x1=avg_exit_rate,
        y0=avg_return_rate,
        y1=y_max,
        fillcolor="rgba(244, 67, 54, 0.08)",
        line=dict(width=0),
        layer="below",
    )

    # Add quadrant divider lines with better styling
    fig.add_shape(
        type="line",
        x0=avg_exit_rate,
        x1=avg_exit_rate,
        y0=y_min,
        y1=y_max,
        line=dict(color="rgba(128, 128, 128, 0.4)", dash="dot", width=2),
        layer="below",
    )
    fig.add_shape(
        type="line",
        x0=x_min,
        x1=x_max,
        y0=avg_return_rate,
        y1=avg_return_rate,
        line=dict(color="rgba(128, 128, 128, 0.4)", dash="dot", width=2),
        layer="below",
    )

    # Add quadrant labels with improved styling
    quadrant_font = dict(
        size=11, family="system-ui, -apple-system, sans-serif"
    )

    # Ideal quadrant (bottom-right)
    fig.add_annotation(
        x=x_max - (x_max - avg_exit_rate) * 0.15,
        y=y_min + (avg_return_rate - y_min) * 0.15,
        text="<b>IDEAL ZONE</b><br>High PH Exits<br>Low Returns",
        showarrow=False,
        font=dict(**quadrant_font, color=theme.colors.success_dark),
        bgcolor=f"{theme.colors.success_bg}",
        bordercolor=theme.colors.success,
        borderwidth=2,
        borderpad=10,
        align="center",
    )

    # Concern quadrant (top-left)
    fig.add_annotation(
        x=x_min + (avg_exit_rate - x_min) * 0.15,
        y=y_max - (y_max - avg_return_rate) * 0.15,
        text="<b>CONCERN ZONE</b><br>Low PH Exits<br>High Returns",
        showarrow=False,
        font=dict(**quadrant_font, color=theme.colors.danger_dark),
        bgcolor=f"{theme.colors.danger_bg}",
        bordercolor=theme.colors.danger,
        borderwidth=2,
        borderpad=10,
        align="center",
    )

    # Mixed quadrants labels (smaller, more subtle)
    # Top-right
    fig.add_annotation(
        x=x_max - (x_max - avg_exit_rate) * 0.15,
        y=y_max - (y_max - avg_return_rate) * 0.15,
        text="High Exits<br>High Returns",
        showarrow=False,
        font=dict(size=10, color="rgba(128, 128, 128, 0.7)"),
        bgcolor="rgba(158, 158, 158, 0.1)",
        bordercolor="rgba(128, 128, 128, 0.5)",
        borderwidth=1,
        borderpad=6,
        align="center",
        opacity=0.8,
    )

    # Bottom-left
    fig.add_annotation(
        x=x_min + (avg_exit_rate - x_min) * 0.15,
        y=y_min + (avg_return_rate - y_min) * 0.15,
        text="Low Exits<br>Low Returns",
        showarrow=False,
        font=dict(size=10, color="rgba(128, 128, 128, 0.7)"),
        bgcolor="rgba(158, 158, 158, 0.1)",
        bordercolor="rgba(128, 128, 128, 0.5)",
        borderwidth=1,
        borderpad=6,
        align="center",
        opacity=0.8,
    )

    # Add average lines annotations
    fig.add_annotation(
        x=avg_exit_rate,
        y=y_max,
        text=f"Avg: {avg_exit_rate:.1f}%",
        showarrow=False,
        font=dict(size=10, color="rgba(128, 128, 128, 0.6)"),
        yshift=10,
    )

    fig.add_annotation(
        x=x_max,
        y=avg_return_rate,
        text=f"Avg: {avg_return_rate:.1f}%",
        showarrow=False,
        font=dict(size=10, color="rgba(128, 128, 128, 0.6)"),
        xshift=10,
        textangle=-90,
    )

    # Highlight best-performing group
    best_groups = comparison_df.loc[
        (comparison_df["PH Exit Rate"] >= avg_exit_rate)
        & (comparison_df["Returns to Homelessness Rate"] <= avg_return_rate)
    ].sort_values("PH Exit Rate", ascending=False)

    if not best_groups.empty:
        best_group = best_groups.iloc[0]

        # Add star marker for best performer
        fig.add_trace(
            go.Scatter(
                x=[best_group["PH Exit Rate"]],
                y=[best_group["Returns to Homelessness Rate"]],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=25,
                    color=theme.colors.warning,
                    line=dict(width=2, color=theme.colors.warning_dark),
                ),
                name="Best Performer",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Update layout with modern styling
    fig.update_layout(
        title={
            "text": "Performance Matrix: PH Exit Rate vs Returns to Homelessness",
            "x": 0.5,
            "xanchor": "center",
            "font": dict(size=16),
        },
        template=PLOT_TEMPLATE,
        xaxis=dict(
            title="Permanent Housing Exit Rate (%)",
            range=[x_min, x_max],
            gridcolor="rgba(128, 128, 128, 0.2)",
            gridwidth=0.5,
            griddash="dot",
            zeroline=False,
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            layer="below traces",
        ),
        yaxis=dict(
            title="Returns to Homelessness Rate (%)",
            range=[y_min, y_max],
            gridcolor="rgba(128, 128, 128, 0.2)",
            gridwidth=0.5,
            griddash="dot",
            zeroline=False,
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            layer="below traces",
        ),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.1)",
            bordercolor="rgba(128, 128, 128, 0.3)",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=80, r=200, t=100, b=80),
        height=700,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        hoverlabel=dict(
            bgcolor="white", bordercolor="black", font=dict(color="black")
        ),
    )

    # Update grid opacity based on theme
    fig.update_xaxes(gridcolor="rgba(128, 128, 128, 0.2)")
    fig.update_yaxes(gridcolor="rgba(128, 128, 128, 0.2)")

    # Add a size legend
    fig.add_annotation(
        text="<b>Bubble Size</b><br>= Clients Served",
        xref="paper",
        yref="paper",
        x=1.02,
        y=-0.05,
        showarrow=False,
        font=dict(size=10, color="rgba(128, 128, 128, 0.6)"),
        align="left",
    )

    return fig


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def _create_empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        title="",
        template=PLOT_TEMPLATE,
        height=400,
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 16, "color": "rgba(128, 128, 128, 0.8)"},
            }
        ],
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ==============================================================================
# MAIN RENDERING FUNCTION
# ==============================================================================


@st.fragment
def render_breakdown_section(
    df_filt: DataFrame, full_df: Optional[DataFrame] = None
) -> None:
    """
    Render demographic breakdown with enhanced visualizations.

    This section allows users to:
    - Analyze key metrics broken down by demographic dimensions
    - Compare PH exit rates and return rates across groups
    - Identify disparities and opportunities for intervention

    Parameters:
    -----------
    df_filt : DataFrame
        Filtered DataFrame
    full_df : DataFrame, optional
        Full DataFrame for returns analysis
    """
    # Ensure we have the full dataset for returns
    if full_df is None:
        full_df = st.session_state.get("df")
        if full_df is None:
            st.error(
                "Original dataset not found. Returns analysis requires full data."
            )
            return

    # Initialize or retrieve section state
    state: Dict[str, Any] = init_section_state(BREAKDOWN_SECTION_KEY)

    # Check if cache is valid
    filter_ts = get_filter_timestamp()
    cache_valid = is_cache_valid(state, filter_ts)

    if not cache_valid:
        invalidate_cache(state, filter_ts)
        state.pop("breakdown_df", None)
        state.pop("min_group_size", None)

    # Header with info button
    col_header, col_info = st.columns([6, 1])
    with col_header:
        st.html(
            html_factory.title(
                "Demographic Breakdown Analysis", level=2, icon="👥"
            )
        )
    with col_info:
        with st.popover("ℹ️ Help", width='stretch'):
            st.markdown(
                """
            ### Understanding Demographic Breakdowns

            This section analyzes key metrics broken down by demographic dimensions to identify disparities and opportunities for targeted interventions.

            **Available Dimensions:**
            - Race/Ethnicity, Gender, Entry Age Tier
            - Program CoC, Local CoC, Agency Name, Program Name
            - Project Type, Household Type, Head of Household
            - Veteran Status, Chronic Homelessness, Currently Fleeing DV

            **Metrics Calculated for Each Group:**

            **Population & Flow:**
            - **Served**: Unique clients active during the period (enrolled at any point)
            - **Inflow**: Clients entering who weren't in any programs the day before
            - **Outflow**: Clients exiting who aren't in any programs at period end
            - **Net Flow**: Inflow - Outflow (positive = growth, negative = reduction)

            **Housing Outcomes:**
            - **Total Exits**: Unique clients with any exit during the period
            - **PH Exits**: Unique clients exiting to permanent housing destinations
            - **PH Exit Rate**: (PH Exits ÷ Total Exits) × 100

            **Housing Stability:**
            - **Returns Count**: PH exits who returned to homelessness within tracking window
            - **Returns to Homelessness Rate**: (Returns ÷ PH Exits) × 100
            - **Return tracking**: Always system-wide, even with filters active

            **Important Notes:**
            - **Minimum group size filter**: Hide small groups for statistical reliability
            - **When filters are active**: Inflow/outflow only track movement within filtered programs
            - **Returns are different**: Always tracked across ALL programs regardless of filters
            - **Each client counted once**: Even with multiple enrollments

            **How to Interpret:**
            - **High PH Exit Rate** (>40%): Strong housing placement outcomes
            - **Low Return Rate** (<10%): Good housing stability
            - **Positive Net Flow**: Group is growing in the system
            - **Large disparities**: May indicate need for targeted interventions

            **Visual Components:**
            - **Client Movement Chart**: Compare served, inflow, outflow, PH exits, and returns
            - **System Flow Balance**: Visualize net flow (growth/reduction) by group
            - **Performance Metrics**: PH exit rates and return rates by category
            - **Outcome Comparison**: Quadrant chart plotting PH exits vs returns

            **Important Note About Client Counting:**
            - A client enrolled in multiple programs/projects is counted in each one
            - Example: A client who starts in Emergency Shelter then transitions to Rapid Re-Housing appears in both project type counts
            - This accurately reflects the number of unique clients served by each program/project/agency
            - The sum across all groups may exceed the total unique clients, which is expected and correct
            - This approach answers "How many clients did each program serve?" not "How many clients used only one program?"

            **Tips for Analysis:**
            - Compare PH exit rates across groups to find disparities
            - Look for groups with high return rates needing stability support
            - Identify groups with negative net flow (more leaving than entering)
            - Use the quadrant chart to find best performers (high PH exits, low returns)
            - Consider group size when interpreting - larger groups are more reliable
            """
            )

    # Get time boundaries
    t0 = st.session_state.get("t0")
    t1 = st.session_state.get("t1")

    if not all([t0, t1]):
        st.warning(
            "⚠️ Please set date ranges in the filter panel before viewing breakdowns."
        )
        return

    # Check for active filters warning
    active_filters = st.session_state.get("filters", {})
    if any(active_filters.values()):
        filter_warning_html = f"""
        <div style="background-color: rgba(255,165,0,0.1); border: 2px solid {WARNING_COLOR}
                                                                              ;
                    border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <strong>🔍 Filtered View Active</strong><br>
            Breakdown shows data for filtered subset only. Inflow/outflow are within this subset.
            Returns are tracked system-wide.
        </div>
        """
        st.html(filter_warning_html)

    # Settings row
    col1, col2, col3 = st.columns([3, 2, 2])

    with col1:
        # Choose breakdown dimension
        key_suffix = hash_data(filter_ts)
        dim_label = st.selectbox(
            "📊 Select dimension to analyze",
            [lbl for lbl, _ in DEMOGRAPHIC_DIMENSIONS],
            key=f"breakdown_dim_{key_suffix}",
            help="Choose how to segment your data",
        )
        dim_col = dict(DEMOGRAPHIC_DIMENSIONS)[dim_label]

    with col2:
        # Get return window from centralized filter state
        filter_state = st.session_state.get("state_filter_form", {})
        return_window = filter_state.get("return_window", 180)
        st.info(f"Return tracking: {return_window} days")

    with col3:
        # Minimum group size filter
        min_group_size = st.number_input(
            "Min group size",
            min_value=1,
            max_value=100,
            value=state.get("min_group_size", 10),
            step=5,
            key=f"min_group_{key_suffix}",
            help="Hide groups smaller than this",
        )

    # Check if any key parameters changed that require recalculation
    params_changed = (
        state.get("selected_dimension") != dim_label
        or state.get("cached_return_window") != return_window
        or state.get("min_group_size") != min_group_size
    )

    # Update state
    state["selected_dimension"] = dim_label
    state["min_group_size"] = min_group_size

    # If parameters changed, clear cache
    if params_changed:
        state.pop("breakdown_df", None)

    # Check if column exists
    if dim_col not in df_filt.columns:
        st.error(f"❌ Column '{dim_col}' not found in the dataset.")
        return

    # Group selection
    try:
        unique_groups = sorted(df_filt[dim_col].dropna().unique())

        if not unique_groups:
            st.warning(f"⚠️ No data available for {dim_label}.")
            return

        selected_groups = st.multiselect(
            f"Filter {dim_label} groups (showing all by default)",
            options=unique_groups,
            default=unique_groups,
            key=f"group_filter_{key_suffix}",
            help="Select specific groups to analyze",
        )

    except Exception as e:
        st.error(f"❌ Error loading categories: {e}")
        return

    if not selected_groups:
        st.info(f"ℹ️ Please select at least one {dim_label} category.")
        return

    # Compute breakdown if needed
    if "breakdown_df" not in state:
        with st.spinner(f"📊 Calculating breakdown by {dim_label}..."):
            try:
                # Calculate breakdown data
                bdf = _calculate_breakdown_data(
                    df_filt, full_df, dim_col, t0, t1, return_window
                )

                if bdf.empty:
                    st.info(
                        f"ℹ️ No data available for breakdown by {dim_label}."
                    )
                    return

                # Cache the FULL result before any filtering
                state["breakdown_df"] = bdf
                state["cached_return_window"] = return_window

            except Exception as e:
                st.error(f"❌ Error calculating breakdown: {e}")
                return

    # Get the cached data
    bdf = state[
        "breakdown_df"
    ].copy()  # Make a copy to avoid modifying cached data

    # Apply minimum group size filter
    bdf_filtered = bdf[bdf["Served"] >= min_group_size]

    if bdf_filtered.empty:
        st.info(
            f"ℹ️ No groups meet the minimum size threshold of {
            min_group_size}."
        )
        st.caption(f"The largest group has {bdf['Served'].max()} clients.")
        return

    # Apply group filter
    bdf_filtered = bdf_filtered[bdf_filtered[dim_col].isin(selected_groups)]

    if bdf_filtered.empty:
        st.info("ℹ️ No data available for the selected criteria.")
        return

    # Sort by served count for consistent display
    bdf_filtered = bdf_filtered.sort_values("Served", ascending=False)

    blue_divider()

    # Create organized layout with tabs
    tab_overview, tab_flow, tab_outcomes, tab_data = ui.dashboard_tabs()

    with tab_overview:
        # Summary metrics at top
        st.html(html_factory.title("Key Metrics Summary", level=3, icon="📊"))

        # Calculate summary stats
        total_served = bdf_filtered["Served"].sum()
        total_inflow = bdf_filtered["Inflow"].sum()
        total_outflow = bdf_filtered["Outflow"].sum()
        total_ph_exits = bdf_filtered["PH Exits"].sum()
        total_returns = bdf_filtered["Returns Count"].sum()

        # Display summary metrics
        summary_cols = st.columns(5)
        summary_cols[0].metric("Total Served", fmt_int(total_served))
        summary_cols[1].metric("Total Inflow", fmt_int(total_inflow))
        summary_cols[2].metric("Total Outflow", fmt_int(total_outflow))
        summary_cols[3].metric("Total PH Exits", fmt_int(total_ph_exits))
        summary_cols[4].metric("Total Returns", fmt_int(total_returns))

        # Volume comparison chart
        st.html(
            html_factory.title("Client Count by Group", level=3, icon="👥")
        )
        fig_counts = _create_counts_chart(bdf_filtered, dim_col)
        st.plotly_chart(fig_counts, width='stretch')

    with tab_flow:
        st.html(html_factory.title("System Flow Analysis", level=3, icon="🔄"))

        # Net flow visualization
        fig_flow = _create_flow_balance_chart(bdf_filtered, dim_col)
        st.plotly_chart(fig_flow, width='stretch')

        # Flow insights
        with st.expander("📊 Flow Insights", expanded=True):
            # Groups with highest growth/reduction
            growth_groups = bdf_filtered[
                bdf_filtered["Net Flow"] > 0
            ].sort_values("Net Flow", ascending=False)
            reduction_groups = bdf_filtered[
                bdf_filtered["Net Flow"] < 0
            ].sort_values("Net Flow", ascending=True)

            col1, col2 = st.columns(2)

            with col1:
                st.html(
                    html_factory.title("Growing Groups", level=4, icon="📈")
                )
                if not growth_groups.empty:
                    top_growth = growth_groups.head(3)
                    # Vectorized calculation of net percentage
                    net_pcts = (
                        top_growth["Net Flow"] / top_growth["Served"] * 100
                    ).where(top_growth["Served"] > 0, 0)

                    for (_, row), net_pct in zip(
                        top_growth.iterrows(), net_pcts
                    ):
                        st.markdown(
                            f"**{row[dim_col]}**: +{row['Net Flow']} ({net_pct:.1f}% of served)"
                        )
                else:
                    st.info("No groups showing growth")

            with col2:
                st.html(
                    html_factory.title("Reducing Groups", level=4, icon="📉")
                )
                if not reduction_groups.empty:
                    top_reduction = reduction_groups.head(3)
                    # Vectorized calculation of net percentage
                    net_pcts = (
                        top_reduction["Net Flow"]
                        / top_reduction["Served"]
                        * 100
                    ).where(top_reduction["Served"] > 0, 0)

                    for (_, row), net_pct in zip(
                        top_reduction.iterrows(), net_pcts
                    ):
                        st.markdown(
                            f"**{row[dim_col]}**: {row['Net Flow']} ({net_pct:.1f}% of served)"
                        )
                else:
                    st.info("No groups showing reduction")

    with tab_outcomes:
        st.html(
            html_factory.title(
                "Performance Metrics by Category", level=3, icon="🎯"
            )
        )
        st.caption("Analyze success rates and identify high-performing groups")

        # Create two-column layout for rate charts
        col1, col2 = st.columns(2)

        with col1:
            fig_ph, _ = _create_rates_charts(
                bdf_filtered, dim_col, return_window
            )
            st.plotly_chart(fig_ph, width='stretch')
            st.caption(
                "📈 Higher rates indicate better housing placement outcomes"
            )

        with col2:
            _, fig_ret = _create_rates_charts(
                bdf_filtered, dim_col, return_window
            )
            st.plotly_chart(fig_ret, width='stretch')
            st.caption("📉 Lower rates indicate better housing stability")

        # Outcome Comparison (quadrant chart)
        st.html(
            html_factory.title(
                "Outcome Comparison: Finding Success Patterns",
                level=3,
                icon="🎆",
            )
        )
        st.caption(
            "Groups in the bottom-right quadrant have the best outcomes (high PH exits, low returns)"
        )

        fig_outcome = _create_outcome_quadrant_chart(bdf_filtered, dim_col)
        st.plotly_chart(fig_outcome, width='stretch')

        # Add interpretation help
        with st.expander("📖 Understanding the charts", expanded=False):
            st.markdown(
                """
            **Rate Charts:**
            - **PH Exit Rate**: Percentage of ALL exits that went to permanent housing
            - **Return Rate**: Percentage of PH exits who returned within the tracking window

            **Quadrant Chart:**
            - **X-axis (PH Exit Rate)**: Higher is better - more exits to permanent housing
            - **Y-axis (Return Rate)**: Lower is better - fewer returns to homelessness
            - **Bubble size**: Represents the number of clients served

            The chart is divided into four quadrants by the average rates:
            - **Bottom-Right (🏆)**: High PH exits, low returns - best outcomes
            - **Top-Left (⚠️)**: Low PH exits, high returns - needs intervention
            - **Top-Right**: High PH exits but also high returns - housing stability issues
            - **Bottom-Left**: Low PH exits but also low returns - stable but limited housing placements
            """
            )

    with tab_data:
        st.html(html_factory.title("Detailed Data Export", level=3, icon="📎"))

        # Prepare export dataframe
        export_df = bdf_filtered.copy()

        # Reorder columns for clarity
        column_order = [
            dim_col,
            "Served",
            "Inflow",
            "Outflow",
            "Net Flow",
            "Total Exits",
            "PH Exits",
            "PH Exit Rate",
            "PH Exit Count",
            "Returns Count",
            "Returns to Homelessness Rate",
        ]

        # Only include columns that exist
        column_order = [
            col for col in column_order if col in export_df.columns
        ]
        export_df = export_df[column_order]

        # Format for display
        format_dict = {
            "Served": "{:,.0f}",
            "Inflow": "{:,.0f}",
            "Outflow": "{:,.0f}",
            "Net Flow": "{:+,.0f}",
            "Total Exits": "{:,.0f}",
            "PH Exits": "{:,.0f}",
            "PH Exit Rate": "{:.1f}%",
            "PH Exit Count": "{:,.0f}",
            "Returns Count": "{:,.0f}",
            "Returns to Homelessness Rate": "{:.1f}%",
        }

        # Apply formatting
        styled_export = export_df.style.format(
            {k: v for k, v in format_dict.items() if k in export_df.columns}
        )

        st.dataframe(styled_export, width='stretch', height=500)

        # Download options
        col1, col2 = st.columns(2)

        with col1:
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"breakdown_{dim_col}_{
                    t0.strftime('%Y%m%d')}_{
                    t1.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                width='stretch',
            )

        with col2:
            # Create a simple text download of key insights
            insights_text = f"""Demographic Breakdown Analysis
Date Range: {t0.strftime('%Y-%m-%d')} to {t1.strftime('%Y-%m-%d')}
Dimension: {dim_col}
Return Window: {return_window} days
Minimum Group Size: {min_group_size}

Key Metrics:
- Total Served: {export_df['Served'].sum():,}
- Total PH Exits: {export_df['PH Exits'].sum():,}
- Average PH Exit Rate: {export_df['PH Exit Rate'].mean():.1f}%
- Average Return Rate: {export_df['Returns to Homelessness Rate'].mean():.1f}%

Groups Shown: {len(export_df)} of {len(bdf)} total
Filtered Out: {len(bdf) - len(export_df)} groups with < {min_group_size} clients
"""

            st.download_button(
                label="📝 Download Summary",
                data=insights_text,
                file_name=f"breakdown_summary_{dim_col}_{
                    t0.strftime('%Y%m%d')}.txt",
                mime="text/plain",
                width='stretch',
            )
