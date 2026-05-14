"""
Outbound Recidivism Visualizations
----------------------------------
"""

from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui.factories.charts import build_flow_sankey
from src.ui.factories.components import ui
from src.ui.themes.styles import NeutralColors


# ============================================================================
# METRIC DISPLAY FUNCTIONS
# ============================================================================
def display_spm_metrics(metrics: Dict[str, Any]) -> None:
    """
    Display outbound recidivism metrics using the component factory.

    Parameters:
        metrics (Dict[str, Any]): Dictionary of metrics
    """
    # Row 1: Core metrics
    ui.metric_row(
        {
            "Number of Relevant Exits": (
                f"{metrics['Number of Relevant Exits']:,}"
            ),
            "Exits to PH": f"{metrics['Total Exits to PH']:,}",
            "Total Returns": f"{metrics['Return']:,}",
        },
        columns=3,
    )

    # Row 2: Return rates
    ui.metric_row(
        {
            "% Return": f"{metrics['Return %']:.1f}%",
            "Return → Homeless (from PH)": f"{metrics['Return to Homelessness']:,}",
            "% Return → Homeless (from PH)": f"{metrics['% Return to Homelessness']:.1f}%",
        },
        columns=3,
    )

    # Row 3: Timing metrics
    col7, col8, col9 = st.columns(3)
    col7.metric(
        "Median Days",
        f"{metrics['Median Days']:.1f}",
        help="Median days to ANY return enrollment",
    )
    col8.metric(
        "Average Days",
        f"{metrics['Average Days']:.1f}",
        help="Average days to ANY return enrollment",
    )
    col9.metric(
        "Max Days",
        f"{metrics['Max Days']:.0f}",
        help="Maximum days to ANY return enrollment",
    )

    # Add clarifying note with info box
    ui.info_section(
        "Timing metrics (Median/Average/Max Days) include ALL returns, not just returns to homelessness. "
        "Return to Homelessness metrics are calculated only for PH exits.",
        type="info",
        icon="📌",
        expanded=True,
    )


def display_spm_metrics_ph(metrics: Dict[str, Any]) -> None:
    """
    Display metrics for PH exits only with natural theming.

    Parameters:
        metrics (Dict[str, Any]): Dictionary of metrics
    """
    # First row - Core metrics
    ui.metric_row(
        {
            "PH Exits": f"{metrics['Number of Relevant Exits']:,}",
            "Total Returns": f"{metrics['Return']:,}",
            "% Return": f"{metrics['Return %']:.1f}%",
        },
        columns=3,
    )

    # Second row - Time metrics
    ui.metric_row(
        {
            "Median Days": f"{metrics['Median Days']:.1f}",
            "Average Days": f"{metrics['Average Days']:.1f}",
            "Max Days": f"{metrics['Max Days']:.0f}",
        },
        columns=3,
    )

    # Third row - Homelessness returns
    ui.metric_row(
        {
            "Return to Homeless (from PH)": f"{metrics['Return to Homelessness']:,}",
            "% Return to Homeless (from PH)": f"{metrics['% Return to Homelessness']:.1f}%",
        },
        columns=2,
    )


def display_spm_metrics_non_ph(metrics: Dict[str, Any]) -> None:
    """
    Display metrics for non-PH exits only with natural theming.

    Parameters:
        metrics (Dict[str, Any]): Dictionary of metrics
    """
    # First row - Core metrics
    ui.metric_row(
        {
            "Non-PH Exits": f"{metrics['Number of Relevant Exits']:,}",
            "Total Returns": f"{metrics['Return']:,}",
            "% Return": f"{metrics['Return %']:.1f}%",
        },
        columns=3,
    )

    # Second row - Time metrics
    ui.metric_row(
        {
            "Median Days": f"{metrics['Median Days']:.1f}",
            "Average Days": f"{metrics['Average Days']:.1f}",
            "Max Days": f"{metrics['Max Days']:.0f}",
        },
        columns=3,
    )


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================
@st.cache_data(show_spinner=False, ttl=1800, max_entries=32)
def create_flow_pivot(
    final_df: pd.DataFrame,
    exit_col: str,
    return_col: str,
) -> pd.DataFrame:
    """
    Build a crosstab pivot table for flow analysis using the exact column names provided,
    **including** "No Return" as a category.

    Parameters:
        final_df (pd.DataFrame): Results DataFrame
        exit_col (str): Column for exit dimension
        return_col (str): Column for return dimension

    Returns:
        pd.DataFrame: Crosstab pivot table
    """
    df_copy = final_df.copy()
    df_copy[return_col] = df_copy[return_col].fillna("No Return").astype(str)
    if exit_col not in df_copy.columns:
        return pd.DataFrame()
    return pd.crosstab(df_copy[exit_col], df_copy[return_col], margins=False)


def get_top_flows_from_pivot(
    pivot_df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """
    Flatten pivot to top‑*n* flows by count, **excluding** any flows
    where either the source or target is "No Return".

    Parameters:
        pivot_df (pd.DataFrame): Crosstab pivot table
        top_n (int): Number of top flows to return

    Returns:
        pd.DataFrame: Top flows with counts and percentages
    """
    total = pivot_df.values.sum()

    # Convert pivot to long format for vectorized operations
    flows_long = pivot_df.stack().reset_index()
    flows_long.columns = ["Exit", "Return", "Count"]

    # Filter and calculate percentages vectorized
    result_df = flows_long[
        (flows_long["Count"] > 0)
        & (flows_long["Exit"] != "No Return")
        & (flows_long["Return"] != "No Return")
    ].copy()

    result_df["Count"] = result_df["Count"].astype(int)
    result_df["Percent"] = (
        (result_df["Count"] / total * 100) if total else 0
    ).round(1)
    if not result_df.empty:
        result_df = result_df.sort_values("Count", ascending=False).head(top_n)
        result_df["Percent"] = result_df["Percent"].apply(
            lambda x: f"{x:.1f}%"
        )

    return result_df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================\
def plot_flow_sankey(
    pivot_df: pd.DataFrame, title: str = "Exit → Return Sankey Diagram"
) -> go.Figure:
    """Render the Outbound Recidivism Exit → Return Sankey via the shared builder.

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


def plot_days_to_return_box(final_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal box‑plot of *DaysToReturnEnrollment* with natural styling.

    Parameters:
        final_df (pd.DataFrame): Results DataFrame

    Returns:
        go.Figure: Plotly box plot
    """
    data = final_df["DaysToReturnEnrollment"].dropna()

    if data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No Return Enrollments Found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=NeutralColors.NEUTRAL_500),
        )
        fig.update_layout(
            title="Distribution of Days to Return Enrollment",
            xaxis_visible=False,
            yaxis_visible=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        return fig

    # Calculate statistics
    median_val = float(data.median())
    avg_val = float(data.mean())
    # p25 = float(data.quantile(0.25))
    # p75 = float(data.quantile(0.75))

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=data.astype(float),
            orientation="h",
            name="Days to Return",
            boxmean="sd",
            boxpoints="outliers",
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

    fig.update_layout(
        title="Distribution of Days to Return Enrollment",
        xaxis_title="Days",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            gridcolor="rgba(128, 128, 128, 0.2)",
            zerolinecolor="rgba(128, 128, 128, 0.2)",
        ),
        yaxis=dict(showticklabels=False, gridcolor="rgba(0,0,0,0)"),
        height=400,
        margin=dict(l=20, r=20, t=60, b=60),
        shapes=[
            dict(
                type="line",
                x0=median_val,
                x1=median_val,
                yref="paper",
                y0=0,
                y1=1,
                line=dict(color=NeutralColors.SUCCESS, width=2, dash="dot"),
                opacity=0.7,
            ),
            dict(
                type="line",
                x0=avg_val,
                x1=avg_val,
                yref="paper",
                y0=0,
                y1=1,
                line=dict(color=NeutralColors.WARNING, width=2, dash="dash"),
                opacity=0.7,
            ),
        ],
        annotations=[
            dict(
                x=median_val,
                y=1.05,
                xref="x",
                yref="paper",
                text=f"Median: {median_val:.0f} days",
                showarrow=False,
                font=dict(size=12, color=NeutralColors.SUCCESS),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=NeutralColors.SUCCESS,
                borderwidth=1,
            ),
            dict(
                x=avg_val,
                y=1.05,
                xref="x",
                yref="paper",
                text=f"Mean: {avg_val:.0f} days",
                showarrow=False,
                font=dict(size=12, color=NeutralColors.WARNING),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=NeutralColors.WARNING,
                borderwidth=1,
            ),
        ],
    )

    return fig


# ============================================================================
# EXPORT ALL PUBLIC FUNCTIONS
# ============================================================================
__all__ = [
    "display_spm_metrics",
    "display_spm_metrics_ph",
    "display_spm_metrics_non_ph",
    "create_flow_pivot",
    "get_top_flows_from_pivot",
    "plot_flow_sankey",
    "plot_days_to_return_box",
]
