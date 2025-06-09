"""
Outbound Recidivism Visualizations
----------------------------------
Specialized visualizations for outbound recidivism analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Union, Any

from ui.styling import style_metric_cards


def display_spm_metrics(metrics: Dict[str, Any]) -> None:
    """
    Display outbound recidivism metrics as cards.
    
    Parameters:
        metrics (Dict[str, Any]): Dictionary of metrics
    """
    try:
        style_metric_cards()
    except Exception:
        pass
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Relevant Exits", f"{metrics['Number of Relevant Exits']:,}")
    col2.metric("Exits to PH", f"{metrics['Total Exits to PH']:,}")
    col3.metric("Return", f"{metrics['Return']:,}")
    col4, col5, col6 = st.columns(3)
    col4.metric("% Return", f"{metrics['Return %']:.1f}%")
    col5.metric("Return → Homeless (from PH)", f"{metrics['Return to Homelessness']:,}")
    col6.metric("% Return → Homeless (from PH)", f"{metrics['% Return to Homelessness']:.1f}%")
    col7, col8, col9 = st.columns(3)
    col7.metric("Median Days", f"{metrics['Median Days']:.1f}", help="Median days to ANY return enrollment (not just homeless returns)")
    col8.metric("Average Days", f"{metrics['Average Days']:.1f}", help="Average days to ANY return enrollment (not just homeless returns)")
    col9.metric("Max Days", f"{metrics['Max Days']:.0f}", help="Maximum days to ANY return enrollment (not just homeless returns)")
        
    # Add clarifying note
    st.caption("📌 **Note:** Timing metrics (Median/Average/Max Days) include ALL returns, not just returns to homelessness. Return to Homelessness metrics are calculated only for PH exits.")


def display_spm_metrics_ph(metrics: Dict[str, Any]) -> None:
    """
    Display metrics for PH exits only.
    
    Parameters:
        metrics (Dict[str, Any]): Dictionary of metrics
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("PH Exits", f"{metrics['Number of Relevant Exits']:,}")
    col2.metric("Return", f"{metrics['Return']:,}")
    col3.metric("% Return", f"{metrics['Return %']:.1f}%")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Median Days", f"{metrics['Median Days']:.1f}")
    col5.metric("Average Days", f"{metrics['Average Days']:.1f}")
    col6.metric("Max Days", f"{metrics['Max Days']:.0f}")
    
    col7, col8, _ = st.columns(3)
    col7.metric("Return → Homeless (from PH)", f"{metrics['Return to Homelessness']:,}")
    col8.metric("% Return → Homeless (from PH)", f"{metrics['% Return to Homelessness']:.1f}%")


def display_spm_metrics_non_ph(metrics: Dict[str, Any]) -> None:
    """
    Display metrics for non-PH exits only.
    
    Parameters:
        metrics (Dict[str, Any]): Dictionary of metrics
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("Non‑PH Exits", f"{metrics['Number of Relevant Exits']:,}")
    col2.metric("Return", f"{metrics['Return']:,}")
    col3.metric("% Return", f"{metrics['Return %']:.1f}%")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Median Days", f"{metrics['Median Days']:.1f}")
    col5.metric("Average Days", f"{metrics['Average Days']:.1f}")
    col6.metric("Max Days", f"{metrics['Max Days']:.0f}")


@st.cache_data(show_spinner=False)
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
    return pd.crosstab(
        df_copy[exit_col],
        df_copy[return_col],
        margins=False
    )


def get_top_flows_from_pivot(pivot_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
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
    flows = [
        {
            "Source": src,
            "Target": tgt,
            "Count": int(val),
            "Percent": (val / total * 100) if total else 0,
        }
        for src, row in pivot_df.iterrows()
        for tgt, val in row.items()
        if val > 0 and src != "No Return" and tgt != "No Return"
    ]
    return (
        pd.DataFrame(flows)
          .sort_values("Count", ascending=False)
          .head(top_n)
    )


def plot_flow_sankey(
    pivot_df: pd.DataFrame,
    title: str = "Exit → Return Sankey Diagram"
) -> go.Figure:
    """
    Build a Sankey diagram to visualize the flow from exit to return categories.
    
    Parameters:
        pivot_df (pd.DataFrame): Crosstab pivot table
        title (str): Diagram title
        
    Returns:
        go.Figure: Plotly Sankey diagram
    """
    # If no flows are present
    if pivot_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title_text="No flows available",
            template="plotly_dark"
        )
        return fig

    df = pivot_df.copy()
    exit_cats = df.index.tolist()
    return_cats = df.columns.tolist()
    nodes = exit_cats + return_cats
    n_exit = len(exit_cats)
    node_types = ["Exit"] * n_exit + ["Return"] * len(return_cats)

    sources, targets, values = [], [], []
    for i, ecat in enumerate(exit_cats):
        for j, rcat in enumerate(return_cats):
            count = df.loc[ecat, rcat]
            if count > 0:
                sources.append(i)
                targets.append(n_exit + j)
                values.append(int(count))

    sankey = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="#FFFFFF", width=0.5),
            label=nodes,
            color="#1f77b4",
            customdata=node_types,
            hovertemplate="%{label}<br>%{customdata}: %{value}<extra></extra>"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(255,127,14,0.6)",
            hovertemplate="From: %{source.label}<br>To: %{target.label}<br>= %{value}<extra></extra>"
        )
    )
    fig = go.Figure(data=[sankey])
    fig.update_layout(
        title_text=title,
        font=dict(family="Open Sans", color="#FFFFFF"),
        hovermode="x",
        template="plotly_dark"
    )
    return fig


def plot_days_to_return_box(final_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal box‑plot of *DaysToReturnEnrollment*.
    
    Parameters:
        final_df (pd.DataFrame): Results DataFrame
        
    Returns:
        go.Figure: Plotly box plot
    """
    data = final_df["DaysToReturnEnrollment"].dropna()
    if data.empty:
        return go.Figure(
            layout={
                "title": {"text": "No Return Enrollments Found"},
                "template": "plotly_dark",
            }
        )

    median_val = float(data.median())
    avg_val = float(data.mean())

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=data.astype(float),
            orientation="h",
            name="Days to Return",
            boxmean="sd",
        )
    )
    fig.update_layout(
        title="Distribution of Days to Return Enrollment",
        template="plotly_dark",
        xaxis_title="Days",
        shapes=[
            dict(type="line", x0=median_val, x1=median_val, yref="paper", y0=0, y1=1, line=dict(dash="dot", width=2)),
            dict(type="line", x0=avg_val, x1=avg_val, yref="paper", y0=0, y1=1, line=dict(dash="dash", width=2)),
        ],
        showlegend=False,
    )
    return fig