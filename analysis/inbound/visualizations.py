"""
Inbound Recidivism Visualizations
---------------------------------
Specialized visualizations for inbound recidivism analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Union, Any

from ui.styling import style_metric_cards


def display_return_metrics_cards(metrics: Dict[str, Any]):
    """
    Display inbound recidivism metrics as cards.
    
    Parameters:
        metrics (Dict[str, Any]): Dictionary of metrics
    """
    style_metric_cards()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Entries", f"{metrics['Total Entries']:,}")
    col2.metric("New (%)", f"{metrics['New']:,} ({metrics['New (%)']:.1f}%)")
    col3.metric("Returning (%)", f"{metrics['Returning']:,} ({metrics['Returning (%)']:.1f}%)")
    col4.metric("Returning From Housing (%)", f"{metrics['Returning From Housing']:,} ({metrics['Returning From Housing (%)']:.1f}%)")


def plot_time_to_entry_box(final_df: pd.DataFrame) -> go.Figure:
    """
    Create a box plot for the Time-to-Entry distribution (days between exit and new entry).
    
    Parameters:
        final_df (DataFrame): Inbound recidivism DataFrame.
    
    Returns:
        go.Figure: Plotly box plot figure.
    """
    returned_df = final_df[final_df["ReturnCategory"] != "New"].dropna(subset=["days_since_last_exit"])
    if returned_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Return Entries Found",
            xaxis_title="Time to Entry (Days)",
            template="plotly_dark"
        )
        return fig

    x = returned_df["days_since_last_exit"].dropna()
    median_val = x.median()
    avg_val = x.mean()
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=x,
        name="Time to Entry",
        boxmean='sd'
    ))
    fig.update_layout(
        title="Time to Entry Distribution (Box Plot)",
        template="plotly_dark",
        xaxis_title="Days to Entry",
        showlegend=False,
        shapes=[
            dict(
                type="line",
                xref="x",
                x0=median_val, x1=median_val,
                yref="paper", y0=0, y1=1,
                line=dict(color="yellow", width=2, dash="dot"),
            ),
            dict(
                type="line",
                xref="x",
                x0=avg_val, x1=avg_val,
                yref="paper", y0=0, y1=1,
                line=dict(color="orange", width=2, dash="dash"),
            ),
        ]
    )
    return fig


def create_flow_pivot_ra(final_df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    """
    Create a pivot table for inbound recidivism flow analysis (Exit → Entry).
    
    Parameters:
        final_df (DataFrame): DataFrame with inbound recidivism data.
        source_col (str): Column representing the exit category.
        target_col (str): Column representing the entry category.
    
    Returns:
        pd.DataFrame: Crosstab pivot table.
    """
    df_copy = final_df.copy()
    df_copy[target_col] = df_copy[target_col].fillna("No Data").astype(str)
    pivot = pd.crosstab(df_copy[source_col], df_copy[target_col], margins=False, dropna=False)
    return pivot


def plot_flow_sankey_ra(pivot_df: pd.DataFrame, title: str = "Exit → Entry Sankey") -> go.Figure:
    """
    Build a Sankey diagram for inbound recidivism (Exit → Entry) with detailed labels.

    Parameters:
        pivot_df (DataFrame): Flow pivot table (rows: Exit categories, columns: Entry categories).
        title (str): Diagram title.

    Returns:
        go.Figure: Plotly Sankey figure.
    """
    df = pivot_df.copy()
    exit_cats = df.index.tolist()     # Left side nodes (Exits)
    entry_cats = df.columns.tolist()  # Right side nodes (Entries)

    nodes = exit_cats + entry_cats
    n_exit = len(exit_cats)
    
    # Node types for hover clarity
    node_types = ["Exits"] * n_exit + ["Entries"] * len(entry_cats)

    sources, targets, values = [], [], []
    for i, exit_cat in enumerate(exit_cats):
        for j, entry_cat in enumerate(entry_cats):
            val = df.loc[exit_cat, entry_cat]
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
            hovertemplate='From: %{source.label}<br>To: %{target.label}<br>Count: %{value}<extra></extra>'
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


def get_top_flows_from_pivot(pivot_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Extract the top flows from a pivot table, excluding "No Data" entries.
    
    Parameters:
        pivot_df (pd.DataFrame): Crosstab pivot table
        top_n (int): Number of top flows to include
        
    Returns:
        pd.DataFrame: Top flows with counts and percentages
    """
    total = pivot_df.values.sum()
    flows = []
    
    for source, row in pivot_df.iterrows():
        for target, value in row.items():
            if value > 0 and target != "No Data":
                flows.append({
                    "Source": source,
                    "Target": target,
                    "Count": int(value),
                    "Percent": (value / total * 100) if total else 0
                })
    
    return (pd.DataFrame(flows)
              .sort_values("Count", ascending=False)
              .head(top_n))