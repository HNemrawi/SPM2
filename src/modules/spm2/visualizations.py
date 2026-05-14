"""
SPM2 Visualization Functions
----------------------------
Specialized visualizations for SPM2 analysis with natural styling.
"""

from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui.factories.charts import build_flow_sankey, default_chart
from src.ui.factories.components import ui
from src.ui.themes.styles import (
    NeutralColors,
    create_info_box,
    create_styled_divider,
)

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================


@st.cache_data(show_spinner=False, ttl=1800, max_entries=32)
def create_flow_pivot(
    final_df: pd.DataFrame, exit_col: str, return_col: str
) -> pd.DataFrame:
    """
    Create a crosstab pivot table for flow analysis (Exit → Return).

    Parameters:
        final_df (DataFrame): Final merged SPM2 DataFrame.
        exit_col (str): Column name for exit category.
        return_col (str): Column name for return category.

    Returns:
        pd.DataFrame: Crosstab pivot without margins.
    """
    df_copy = final_df.copy()
    df_copy[return_col] = df_copy[return_col].fillna("No Return").astype(str)
    pivot = pd.crosstab(df_copy[exit_col], df_copy[return_col], margins=False)
    return pivot


def get_top_flows_from_pivot(
    pivot_df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """
    Generate a table of the top flows from a crosstab pivot table,
    excluding flows involving "No Return".

    Parameters:
        pivot_df (DataFrame): Crosstab pivot table.
        top_n (int): Number of top flows to return.

    Returns:
        pd.DataFrame: Top flows with source, target, count, and percent.
    """
    df = pivot_df.copy()
    total = df.values.sum()
    rows = []
    row_labels = df.index.tolist()
    col_labels = df.columns.tolist()

    for rlab in row_labels:
        if rlab == "No Return":
            continue
        for clab in col_labels:
            if clab == "No Return":
                continue
            val = df.loc[rlab, clab]
            rows.append(
                {
                    "Exit Dimension": rlab,
                    "Entry Dimension": clab,
                    "Count": val,
                    "Percent": (val / total * 100) if total else 0,
                }
            )

    flows_df = pd.DataFrame(rows)
    flows_df = flows_df[flows_df["Count"] > 0].sort_values(
        "Count", ascending=False
    )
    return flows_df.head(top_n)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def plot_flow_sankey(
    pivot_df: pd.DataFrame, title: str = "Exit → Return Sankey Diagram"
) -> go.Figure:
    """Render the SPM2 Exit → Return flow Sankey via the shared builder.

    "No Return" is highlighted in the in-theme teal so the positive analytic
    sink stands out from the other return buckets.
    """
    return build_flow_sankey(
        pivot_df,
        title,
        source_role="Exit",
        target_role="Return",
        accent_targets=frozenset({"No Return"}),
    )


def plot_days_to_return_box(
    final_df: pd.DataFrame, return_period: int = 730
) -> go.Figure:
    """
    Create a horizontal box plot for the distribution of Days-to-Return,
    ensuring that the displayed quartiles, whiskers, and mean match your summary metrics.

    Parameters:
        final_df (DataFrame): Merged SPM2 DataFrame.
        return_period (int): Maximum days to consider.

    Returns:
        go.Figure: Plotly box plot figure.
    """
    # Filter down to clients who actually returned within the chosen period
    returned_df = final_df[
        (final_df["ReturnedToHomelessness"])
        & (final_df["DaysToReturn"].notna())
        & (final_df["DaysToReturn"] <= return_period)
    ]

    # If no returns are found, display a simple message
    if returned_df.empty:
        fig = go.Figure()
        default_chart.add_annotation(
            fig,
            text="No Returns Found Within Selected Period",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            arrow=False,
            font=dict(size=16, color=NeutralColors.NEUTRAL_500),
        )
        # Apply basic layout for empty state
        default_chart.apply_layout(
            fig,
            title="Days to Return Distribution",
            xaxis_visible=False,
            yaxis_visible=False,
        )
        return fig

    # Calculate summary statistics
    x = returned_df["DaysToReturn"].dropna()
    p25 = x.quantile(0.25)
    median_val = x.median()
    p75 = x.quantile(0.75)
    avg_val = x.mean()
    min_val = x.min()
    max_val = x.max()

    # Calculate Interquartile Range (IQR)
    IQR = p75 - p25

    # Determine the whiskers using the 1.5 * IQR rule
    lower_whisker = max(min_val, p25 - 1.5 * IQR)
    upper_whisker = min(max_val, p75 + 1.5 * IQR)

    # Create box plot using chart factory with raw data approach
    fig = go.Figure()

    # Add custom box plot trace with detailed statistics
    fig.add_trace(
        go.Box(
            x=x.values,
            orientation="h",
            name="Days to Return",
            q1=[p25],
            median=[median_val],
            q3=[p75],
            lowerfence=[lower_whisker],
            upperfence=[upper_whisker],
            mean=[avg_val],
            boxmean="sd",
            boxpoints="outliers",  # Show only outliers for cleaner visualization
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(
                color=NeutralColors.PRIMARY,
                outliercolor=NeutralColors.WARNING,
                size=6,
            ),
            line=dict(color=NeutralColors.PRIMARY),
            fillcolor="rgba(0, 102, 204, 0.2)",  # Light primary color fill
        )
    )

    # Apply consistent layout
    default_chart.apply_layout(
        fig,
        xaxis_title="Days to Return",
        showlegend=False,
        yaxis=dict(showticklabels=False, gridcolor="rgba(0,0,0,0)"),
    )

    # Add threshold lines and annotations using chart factory methods
    fig.add_shape(
        type="line",
        xref="x",
        x0=median_val,
        x1=median_val,
        yref="paper",
        y0=0,
        y1=1,
        line=dict(color=NeutralColors.SUCCESS, width=2, dash="dot"),
        opacity=0.7,
    )

    fig.add_shape(
        type="line",
        xref="x",
        x0=avg_val,
        x1=avg_val,
        yref="paper",
        y0=0,
        y1=1,
        line=dict(color=NeutralColors.WARNING, width=2, dash="dash"),
        opacity=0.7,
    )

    # Add annotations for median and mean
    default_chart.add_annotation(
        fig,
        text=f"Median: {median_val:.0f}",
        x=median_val,
        y=1.05,
        xref="x",
        yref="paper",
        arrow=False,
        font=dict(size=12, color=NeutralColors.SUCCESS),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor=NeutralColors.SUCCESS,
        borderwidth=1,
    )

    default_chart.add_annotation(
        fig,
        text=f"Mean: {avg_val:.0f}",
        x=avg_val,
        y=1.05,
        xref="x",
        yref="paper",
        arrow=False,
        font=dict(size=12, color=NeutralColors.WARNING),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor=NeutralColors.WARNING,
        borderwidth=1,
    )

    return fig


# ============================================================================
# METRIC DISPLAY FUNCTIONS
# ============================================================================


def display_spm_metrics(
    metrics: Dict[str, Any], return_period: int, show_total_exits: bool = True
):
    """
    Display key SPM2 metrics in a card-style layout with natural theming.

    Parameters:
        metrics (dict): Dictionary of computed metrics.
        return_period (int): Maximum days to consider for returns
        show_total_exits (bool): Whether to display total exit count.
    """

    # If return_period is less than 730, override the >24 Months metric
    if return_period <= 730:
        metrics["Return > 24 Months"] = None
        metrics["% Return > 24M"] = None

    # First row - Main metrics
    main_metrics = {}
    if show_total_exits:
        main_metrics[
            "Total Exits"
        ] = f"{metrics['Number of Relevant Exits']:,}"
    main_metrics[
        "PH Exits"
    ] = f"{metrics['PH Exits']:,} ({metrics['% PH Exits']:.1f}%)"
    main_metrics[
        "Total Returns"
    ] = f"{metrics['Total Return']:,} ({metrics['% Return']:.1f}%)"

    ui.metric_row(main_metrics, columns=len(main_metrics))

    # Second row - Return period breakdown
    period_metrics = {
        "Return <6M": f"{metrics['Return < 6 Months']:,} ({metrics['% Return < 6M']:.1f}%)",
        "Return 6-12M": f"{metrics['Return 6–12 Months']:,} ({metrics['% Return 6–12M']:.1f}%)",
        "Return 12-24M": f"{metrics['Return 12–24 Months']:,} ({metrics['% Return 12–24M']:.1f}%)",
    }

    # Add >24 months if applicable
    if metrics["Return > 24 Months"] is not None:
        period_metrics["Return >24M"] = (
            f"{metrics['Return > 24 Months']:,} "
            f"({metrics['% Return > 24M']:.1f}%)"
        )
    else:
        period_metrics["Return >24M"] = "N/A"

    ui.metric_row(period_metrics, columns=len(period_metrics))

    # Add spacing
    st.html(create_styled_divider())

    # Third row - Time metrics
    time_metrics = {
        "Median Days": f"{metrics['Median Days (<=period)']:.1f}",
        "Average Days": f"{metrics['Average Days (<=period)']:.1f}",
        "25th Percentile": f"{metrics['DaysToReturn 25th Pctl']:.0f} days",
        "75th Percentile": f"{metrics['DaysToReturn 75th Pctl']:.0f} days",
    }

    ui.metric_row(time_metrics, columns=4)

    # Add detailed info box with percentile information
    percentile_info = f"""
    <div style="margin-top: 20px;">
        <h4 style="margin-bottom: 10px; color: {NeutralColors.PRIMARY};">Distribution Details</h4>
        <p><strong>25th/75th Percentiles:</strong> {metrics['DaysToReturn 25th Pctl']                                                    :.0f} / {metrics['DaysToReturn 75th Pctl']:.0f} days</p>
        <p><strong>Maximum Days to Return:</strong> {metrics['DaysToReturn Max']:.0f} days</p>
        <p style="margin-top: 10px; color: var(--text-secondary);">
            25% of returns occurred within <strong>{metrics['DaysToReturn 25th Pctl']:.0f}</strong> days,
            and 25% occurred beyond <strong>{metrics['DaysToReturn 75th Pctl']:.0f}</strong> days.
            The longest time to return was <strong>{metrics['DaysToReturn Max']:.0f}</strong> days.
        </p>
    </div>
    """

    st.html(
        create_info_box(
            percentile_info,
            type="info",
            title="Statistical Summary",
            icon="📊",
        )
    )


# ============================================================================
# EXPORT ALL PUBLIC FUNCTIONS
# ============================================================================

__all__ = [
    "create_flow_pivot",
    "get_top_flows_from_pivot",
    "plot_flow_sankey",
    "plot_days_to_return_box",
    "display_spm_metrics",
]
