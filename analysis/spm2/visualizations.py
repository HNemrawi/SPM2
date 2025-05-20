"""
SPM2 Visualization Functions
----------------------------
Specialized visualizations for SPM2 analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Union, Any

from ui.styling import style_metric_cards


@st.cache_data(show_spinner=False)
def create_flow_pivot(final_df: pd.DataFrame, exit_col: str, return_col: str) -> pd.DataFrame:
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


def get_top_flows_from_pivot(pivot_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
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
            rows.append({
                "Source": rlab,
                "Target": clab,
                "Count": val,
                "Percent": (val / total * 100) if total else 0
            })
    flows_df = pd.DataFrame(rows)
    flows_df = flows_df[flows_df["Count"] > 0].sort_values("Count", ascending=False)
    return flows_df.head(top_n)


def plot_flow_sankey(pivot_df: pd.DataFrame, title: str = "Exit → Return Sankey Diagram") -> go.Figure:
    """
    Build a Sankey diagram to visualize the flow from exit to return categories.

    Parameters:
        pivot_df (DataFrame): Flow pivot table (rows are exit, columns are return).
        title (str): Diagram title.

    Returns:
        go.Figure: Plotly figure of the Sankey diagram.
    """
    df = pivot_df.copy()
    # Get exit categories (rows) and return categories (columns)
    exit_cats = df.index.tolist()   # left side nodes (Exit)
    return_cats = df.columns.tolist() # right side nodes (Return)

    # Overall nodes list: first exit, then return
    nodes = exit_cats + return_cats

    # Create custom node type data: "Exit" for exit nodes, "Return" for return nodes.
    n_exit = len(exit_cats)
    node_types = ["Exits"] * n_exit + ["Returns"] * len(return_cats)
    
    sources, targets, values = [], [], []
    for i, ecat in enumerate(exit_cats):
        for j, rcat in enumerate(return_cats):
            val = df.loc[ecat, rcat]
            if val > 0:
                sources.append(i)
                targets.append(n_exit + j)
                values.append(val)

    sankey = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="#FFFFFF", width=0.5),
            label=nodes,
            color="#1f77b4",
            customdata=node_types,
            hovertemplate='%{label}<br>%{customdata}: %{value}<extra></extra>'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(255,127,14,0.6)",
            hovertemplate='From: %{source.label}<br>To: %{target.label}<br>= %{value}<extra></extra>'
        ),
    )
    fig = go.Figure(data=[sankey])
    fig.update_layout(
        title_text=title,
        font=dict(family="Open Sans", color="#FFFFFF"),
        hovermode="x",
        template="plotly_dark"
    )
    return fig


def plot_days_to_return_box(final_df: pd.DataFrame, return_period: int = 730) -> go.Figure:
    """
    Create a horizontal box plot for the distribution of Days-to-Return,
    ensuring that the displayed quartiles, whiskers, and mean match your summary metrics.
    
    Parameters:
        final_df (DataFrame): Merged SPM2 DataFrame.
        return_period (int): Maximum days to consider.
    
    Returns:
        go.Figure: Plotly box plot figure.
    """
    # Filter down to clients who actually returned within the chosen period.
    returned_df = final_df[
        (final_df["ReturnedToHomelessness"]) &
        (final_df["DaysToReturn"].notna()) &
        (final_df["DaysToReturn"] <= return_period)
    ]

    # If no returns are found, display a simple message in the figure.
    if returned_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Returns Found",
            xaxis_title="Days to Return",
            template="plotly_dark"
        )
        return fig

    # Calculate summary statistics using the same approach as your metrics.
    x = returned_df["DaysToReturn"].dropna()
    p25 = x.quantile(0.25)
    median_val = x.median()          # Equivalent to quantile(0.50)
    p75 = x.quantile(0.75)
    avg_val = x.mean()
    min_val = x.min()
    max_val = x.max()
    
    # Calculate Interquartile Range (IQR)
    IQR = p75 - p25
    
    # Determine the whiskers using the 1.5 * IQR rule.
    lower_whisker = max(min_val, p25 - 1.5 * IQR)
    upper_whisker = min(max_val, p75 + 1.5 * IQR)

    fig = go.Figure()

    # Create the box plot trace by explicitly providing the quartile values and whiskers.
    fig.add_trace(go.Box(
        q1=[p25],
        median=[median_val],
        q3=[p75],
        lowerfence=[lower_whisker],
        upperfence=[upper_whisker],
        mean=[avg_val],    # Displays the mean marker with standard deviation info
        boxmean='sd',      # This shows the mean ± SD on the plot
        boxpoints='all',   # Option to show all data points; change to 'outliers' if desired
        orientation='h',   # Horizontal orientation so that the x-axis reflects Days to Return
        name="Days to Return"
    ))

    # Add vertical lines to highlight the median and mean.
    fig.update_layout(
        title="Days to Return Distribution (Box Plot)",
        template="plotly_dark",
        xaxis_title="Days to Return",
        showlegend=False,
        shapes=[
            dict(
                type="line",
                xref="x",
                x0=median_val, x1=median_val,
                yref="paper",
                y0=0, y1=1,
                line=dict(color="yellow", width=2, dash="dot")
            ),
            dict(
                type="line",
                xref="x",
                x0=avg_val, x1=avg_val,
                yref="paper",
                y0=0, y1=1,
                line=dict(color="orange", width=2, dash="dash")
            ),
        ]
    )

    return fig


def display_spm_metrics(metrics: Dict[str, Any], return_period: int, show_total_exits: bool = True):
    """
    Display key SPM2 metrics in a card-style layout.
    
    Parameters:
        metrics (dict): Dictionary of computed metrics.
        return_period (int): Maximum days to consider for returns
        show_total_exits (bool): Whether to display total exit count.
    """
    style_metric_cards()

    # If return_period is less than 730, override the >24 Months metric
    if return_period <= 730:
        metrics["Return > 24 Months"] = None
        metrics["% Return > 24M"] = None

    col1, col2, col3, col4, col5 = st.columns(5)
    if show_total_exits:
        col1.metric("Number of Relevant Exits", f"{metrics['Number of Relevant Exits']:,}")
    col1.metric("PH Exits", f"{metrics['PH Exits']:,} ({metrics['% PH Exits']:.1f}%)")
    col2.metric("Return <6M", f"{metrics['Return < 6 Months']:,} ({metrics['% Return < 6M']:.1f}%)")
    col3.metric("Return 6–12M", f"{metrics['Return 6–12 Months']:,} ({metrics['% Return 6–12M']:.1f}%)")
    col4.metric("Return 12–24M", f"{metrics['Return 12–24 Months']:,} ({metrics['% Return 12–24M']:.1f}%)")
    # Conditionally render >24 months
    if metrics["Return > 24 Months"] is None:
        col5.metric("Return >24M", "N/A")
    else:
        col5.metric("Return >24M", f"{metrics['Return > 24 Months']:,} ({metrics['% Return > 24M']:.1f}%)")
        
    colA, colB, colC = st.columns(3)
    colA.metric("Total Return", f"{metrics['Total Return']:,} ({metrics['% Return']:.1f}%)")
    colB.metric("Median Days", f"{metrics['Median Days (<=period)']:.1f}")
    colC.metric("Avg Days", f"{metrics['Average Days (<=period)']:.1f}")
    st.markdown(
        f"""
        **25th/75th:** {metrics['DaysToReturn 25th Pctl']:.1f}/{metrics['DaysToReturn 75th Pctl']:.1f}  
        **Max Days:** {metrics['DaysToReturn Max']:.0f}

        25% of returns occurred within **{metrics['DaysToReturn 25th Pctl']:.0f}** days, 
        and 25% occurred beyond **{metrics['DaysToReturn 75th Pctl']:.0f}** days. 
        The longest time to return was **{metrics['DaysToReturn Max']:.0f}** days. 
        See box plot below for additional details.
        """
    )
