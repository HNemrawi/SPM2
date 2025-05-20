"""
Reusable UI components for the application
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Union, Any

def display_metric_cards(metrics: Dict[str, Any], style_func: Optional[callable] = None):
    """
    Display performance metrics as Streamlit metric cards.
    
    Parameters:
        metrics (Dict[str, Any]): Dictionary of metrics to display
        style_func (Optional[callable]): Function to apply card styling
    """
    # Apply styling if provided
    if style_func:
        style_func()
    
    # Render metrics
    total = metrics.get("Number of Relevant Exits", 0)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Relevant Exits", f"{total:,}")
    
    if "Total Return" in metrics and "% Return" in metrics:
        col2.metric("Total Return", f"{metrics['Total Return']:,}")
        col3.metric("% Return", f"{metrics['% Return']:.1f}%")
    
    if "PH Exits" in metrics and "% PH Exits" in metrics:
        col1, col2 = st.columns(2)
        col1.metric("PH Exits", f"{metrics['PH Exits']:,}")
        col2.metric("% PH Exits", f"{metrics['% PH Exits']:.1f}%")
    
    # Display timing metrics if available
    if "Median Days (<=period)" in metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Median Days", f"{metrics['Median Days (<=period)']:.1f}")
        col2.metric("Average Days", f"{metrics['Average Days (<=period)']:.1f}")
        if "DaysToReturn Max" in metrics:
            col3.metric("Max Days", f"{metrics['DaysToReturn Max']:.0f}")

def render_download_button(df: pd.DataFrame, filename: str, label: str = "Download Data"):
    """
    Render a download button for a dataframe.
    
    Parameters:
        df (pd.DataFrame): DataFrame to download
        filename (str): Name of the downloaded file
        label (str): Button label
    """
    st.download_button(
        label=label,
        data=df.to_csv(index=False),
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

def render_about_section(title: str, content: str, expanded: bool = False):
    """
    Render an about/help section with consistent styling.
    
    Parameters:
        title (str): Section title
        content (str): Markdown content
        expanded (bool): Whether the section is expanded by default
    """
    with st.expander(f"📘 {title}", expanded=expanded):
        st.markdown(content)

def render_filter_section(title: str, content: callable):
    """
    Render a filter section in the sidebar.
    
    Parameters:
        title (str): Section title
        content (callable): Function to render filter content
    """
    with st.sidebar.expander(title, expanded=True):
        return content()

def render_dataframe_with_style(
    df: pd.DataFrame,
    highlight_cols: Optional[List[str]] = None,
    precision: int = 1,
    height: Optional[int] = None,
    cmap: str = "Blues",
    axis: int = 0
) -> None:
    """
    Render a dataframe with consistent styling, integer formatting for counts,
    and optional background gradients, with centered alignment.

    Parameters:
        df (pd.DataFrame): DataFrame to display
        highlight_cols (Optional[List[str]]): Columns to highlight with background_gradient
        precision (int): Decimal precision for float columns
        height (Optional[int]): Height of the dataframe in pixels
        cmap (str): Matplotlib colormap to use for background gradient
        axis (int): Axis for background gradient: 0=column-wise, 1=row-wise
    """
    if df.empty:
        st.info("No data available to display.")
        return

    formatter: dict = {}
    for col in df.columns:
        series = df[col]
        if col.endswith('%') or '(%)' in col:
            if not series.dtype == 'object' or not series.astype(str).str.contains('%').any():
                formatter[col] = '{:.1f}%'
        elif pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                formatter[col] = '{:,d}'  # Format integers with commas
            elif series.dropna().eq(series.dropna().astype(int)).all():
                formatter[col] = '{:,d}'  # Floats that are effectively integers
            else:
                formatter[col] = f"{{:,.{precision}f}}"  # Format floats

    # Apply formatting and center alignment
    styled = df.style.format(formatter).set_properties(**{"text-align": "center"})

    # Optional background gradient
    if highlight_cols:
        valid_cols = [col for col in highlight_cols if col in df.columns]
        if valid_cols:
            styled = styled.background_gradient(cmap=cmap, subset=valid_cols, axis=axis)

    # Display
    display_kwargs = {"use_container_width": True}
    if height:
        display_kwargs["height"] = height

    st.dataframe(styled, **display_kwargs)