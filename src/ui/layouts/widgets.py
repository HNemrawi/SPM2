"""
Reusable UI Components for the Application

"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

# ==================== HELPER FUNCTIONS ====================


def safe_background_gradient(styled, **kwargs):
    """
    Apply color gradient styling without requiring matplotlib.
    Uses custom color mapping instead of background_gradient.

    Args:
        styled: Pandas Styler object
        **kwargs: Arguments (cmap, subset, axis, vmin, vmax)

    Returns:
        Styled object with color gradient applied
    """

    def color_cell(val, vmin, vmax, cmap_name):
        """Color a cell based on its value."""
        if pd.isna(val) or vmin == vmax:
            return ''

        # Normalize value between 0 and 1
        norm_val = (val - vmin) / (vmax - vmin)
        norm_val = max(0, min(1, norm_val))  # Clamp between 0 and 1

        # Define simple color maps
        if cmap_name in ['Blues', 'blues']:
            # Light blue to dark blue
            r = int(240 - norm_val * 100)
            g = int(248 - norm_val * 80)
            b = int(255 - norm_val * 50)
        elif cmap_name in ['RdYlGn', 'rdylgn']:
            # Red to Yellow to Green
            if norm_val < 0.5:
                # Red to Yellow
                r = 255
                g = int(norm_val * 2 * 255)
                b = 0
            else:
                # Yellow to Green
                r = int((2 - norm_val * 2) * 255)
                g = 255
                b = 0
        else:
            # Default: light gray to dark gray
            gray = int(255 - norm_val * 80)
            r = g = b = gray

        return f'background-color: rgb({r}, {g}, {b})'

    # Extract parameters
    cmap = kwargs.get('cmap', 'Blues')
    subset = kwargs.get('subset', None)
    kwargs.get('axis', 0)
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)

    # Get the dataframe from the styler
    df = styled.data

    # Determine columns to apply styling to
    if subset:
        cols_to_style = subset if isinstance(subset, list) else [subset]
    else:
        cols_to_style = df.select_dtypes(include=['number']).columns.tolist()

    # Apply styling to each column
    for col in cols_to_style:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            col_vmin = vmin if vmin is not None else df[col].min()
            col_vmax = vmax if vmax is not None else df[col].max()

            styled = styled.map(
                lambda val: color_cell(val, col_vmin, col_vmax, cmap),
                subset=[col],
            )

    return styled


# ==================== COMPONENT CONFIGURATION ====================


@dataclass
class ComponentTheme:
    """Configuration for component styling"""

    # Metric card configurations
    metric_positive_color: str = "#059862"  # Success green
    metric_negative_color: str = "#DC2626"  # Danger red
    metric_neutral_color: str = "#0066CC"  # Primary blue

    # DataFrame styling
    df_cmap_default: str = "Blues"
    df_cmap_diverging: str = "RdYlGn"
    df_precision_default: int = 1

    # Icons
    icon_about: str = "📘"
    icon_download: str = "📥"
    icon_filter: str = "⚙️"
    icon_info: str = "ℹ️"


# Global theme instance
theme = ComponentTheme()

# ==================== METRIC DISPLAY COMPONENTS ====================


def display_metric_cards(
    metrics: Dict[str, Any],
    style_func: Optional[Callable] = None,
    delta_reference: Optional[Dict[str, float]] = None,
) -> None:
    """
    Display performance metrics as Streamlit metric cards with enhanced styling.

    Parameters:
        metrics: Dictionary of metrics to display
        style_func: Optional function to apply card styling
        delta_reference: Optional dictionary of reference values for delta calculations

    Examples:
        >>> metrics = {"Total Exits": 100, "% Return": 25.5}
        >>> display_metric_cards(metrics)
    """
    # Apply styling if provided
    if style_func:
        style_func()

    # Determine metric layout based on available metrics
    metric_keys = list(metrics.keys())

    # Primary metric (always first)
    if metric_keys:
        total_key = next(
            (
                k
                for k in metric_keys
                if "exits" in k.lower() or "total" in k.lower()
            ),
            metric_keys[0],
        )
        total = metrics.get(total_key, 0)

        # Handle return metrics
        if "Total Return" in metrics and "% Return" in metrics:
            col1, col2, col3 = st.columns(3)

            # Format and display with delta if reference provided
            delta1 = (
                _calculate_delta(total, delta_reference.get(total_key))
                if delta_reference
                else None
            )
            col1.metric(
                label=total_key,
                value=f"{total:,}",
                delta=delta1,
                delta_color=(
                    "inverse" if "return" in total_key.lower() else "normal"
                ),
            )

            delta2 = (
                _calculate_delta(
                    metrics["Total Return"],
                    delta_reference.get("Total Return"),
                )
                if delta_reference
                else None
            )
            col2.metric(
                label="Total Return",
                value=f"{metrics['Total Return']:,}",
                delta=delta2,
                delta_color="inverse",
            )

            delta3 = (
                _calculate_delta(
                    metrics["% Return"], delta_reference.get("% Return")
                )
                if delta_reference
                else None
            )
            col3.metric(
                label="% Return",
                value=f"{metrics['% Return']:.1f}%",
                delta=f"{delta3:.1f}%" if delta3 else None,
                delta_color="inverse",
            )

        # Handle PH exit metrics
        elif "PH Exits" in metrics and "% PH Exits" in metrics:
            col1, col2, col3 = st.columns(3)

            col1.metric(label=total_key, value=f"{total:,}")
            col2.metric(label="PH Exits", value=f"{metrics['PH Exits']:,}")
            col3.metric(
                label="% PH Exits", value=f"{metrics['% PH Exits']:.1f}%"
            )

        else:
            # Default single metric display
            st.metric(label=total_key, value=f"{total:,}")

    # Display timing metrics if available
    if any("days" in k.lower() for k in metric_keys):
        _display_timing_metrics(metrics)


def _display_timing_metrics(metrics: Dict[str, Any]) -> None:
    """Helper function to display timing-related metrics."""
    timing_metrics = {}

    # Collect timing metrics
    if "Median Days (<=period)" in metrics:
        timing_metrics["Median Days"] = metrics["Median Days (<=period)"]
    if "Average Days (<=period)" in metrics:
        timing_metrics["Average Days"] = metrics["Average Days (<=period)"]
    if "DaysToReturn Max" in metrics:
        timing_metrics["Max Days"] = metrics["DaysToReturn Max"]

    if timing_metrics:
        cols = st.columns(len(timing_metrics))
        for idx, (label, value) in enumerate(timing_metrics.items()):
            cols[idx].metric(
                label=label,
                value=f"{value:.1f}" if "Max" not in label else f"{value:.0f}",
                help=f"Days to return ({label.lower()})",
            )


def _calculate_delta(
    current: float, reference: Optional[float]
) -> Optional[float]:
    """Calculate delta value for metric display."""
    if reference is None:
        return None
    return current - reference


# ==================== DATA EXPORT COMPONENTS ====================


def render_download_button(
    df: pd.DataFrame,
    filename: str,
    label: str = "Download Data",
    file_format: str = "csv",
    key: Optional[str] = None,
) -> None:
    """
    Render an enhanced download button for a dataframe with multiple format options.

    Parameters:
        df: DataFrame to download
        filename: Name of the downloaded file (without extension)
        label: Button label
        file_format: Export format ('csv', 'xlsx', 'json')
        key: Optional unique key for the button

    Examples:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> render_download_button(df, "my_data", "Download CSV")
    """
    # Prepare file data based on format
    if file_format == "csv":
        file_data = df.to_csv(index=False)
        mime_type = "text/csv"
        file_ext = "csv"
    elif file_format == "xlsx":
        from io import BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")
        file_data = buffer.getvalue()
        mime_type = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        file_ext = "xlsx"
    elif file_format == "json":
        file_data = df.to_json(orient="records", indent=2)
        mime_type = "application/json"
        file_ext = "json"
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    # Add icon to label
    icon = theme.icon_download
    full_label = f"{icon} {label}"

    # Render download button
    st.download_button(
        label=full_label,
        data=file_data,
        file_name=f"{filename}.{file_ext}",
        mime=mime_type,
        width='stretch',
        key=key,
    )


# ==================== CONTENT DISPLAY COMPONENTS ====================


def render_about_section(
    title: str,
    content: str,
    expanded: bool = False,
    icon: Optional[str] = None,
    type: str = "info",
) -> None:
    """
    Render an enhanced about/help section with consistent styling and theming.

    Parameters:
        title: Section title
        content: HTML or Markdown content
        expanded: Whether the section is expanded by default
        icon: Optional custom icon (defaults to theme icon)
        type: Section type for styling ('info', 'warning', 'help')

    Examples:
        >>> render_about_section("How it works", "This analysis...", expanded=True)
    """
    # Select appropriate icon
    if icon is None:
        icon = theme.icon_about if type == "info" else theme.icon_info

    # Apply custom styling for better appearance
    st.markdown(
        """
    <style>
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.05rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Render expander with content
    with st.expander(f"{icon} {title}", expanded=expanded):
        if content.strip().startswith("<"):
            # HTML content
            st.html(content)
        else:
            # Markdown content
            st.markdown(content)


# ==================== FILTER COMPONENTS ====================


def render_filter_section(
    title: str,
    content: Callable,
    expanded: bool = True,
    icon: Optional[str] = None,
) -> Any:
    """
    Render an enhanced filter section in the sidebar with consistent styling.

    Parameters:
        title: Section title
        content: Function to render filter content
        expanded: Whether the section is expanded by default
        icon: Optional custom icon

    Returns:
        The return value of the content function

    Examples:
        >>> def my_filters():
        ...     return st.selectbox("Choose", ["A", "B"])
        >>> result = render_filter_section("Options", my_filters)
    """
    icon = icon or theme.icon_filter

    with st.sidebar.expander(f"{icon} {title}", expanded=expanded):
        return content()


# ==================== DATAFRAME DISPLAY COMPONENTS ====================


def render_dataframe_with_style(
    df: pd.DataFrame,
    highlight_cols: Optional[List[str]] = None,
    precision: int = 1,
    height: Optional[int] = None,
    cmap: Optional[str] = None,
    axis: int = 0,
    caption: Optional[str] = None,
    index_name: Optional[str] = None,
    show_index: bool = True,
) -> None:
    """
    Render a dataframe with enhanced styling, formatting, and theme support.

    Parameters:
        df: DataFrame to display
        highlight_cols: Columns to highlight with background gradient
        precision: Decimal precision for float columns
        height: Height of the dataframe in pixels
        cmap: Colormap for background gradient (defaults to theme)
        axis: Axis for background gradient (0=column-wise, 1=row-wise)
        caption: Optional caption to display above the table
        index_name: Optional name for the index column
        show_index: Whether to show the index

    Examples:
        >>> df = pd.DataFrame({'A': [1.234, 2.345], 'B%': [10.5, 20.3]})
        >>> render_dataframe_with_style(df, highlight_cols=['B%'])
    """
    if df.empty:
        st.info(f"{theme.icon_info} No data available to display.")
        return

    # Set default colormap if not provided
    if cmap is None:
        cmap = theme.df_cmap_default

    # Display caption if provided
    if caption:
        st.caption(caption)

    # Prepare formatter dictionary
    formatter = _build_formatter(df, precision)

    # Apply base styling
    styled = df.style.format(formatter, na_rep="—")

    # Center align all cells
    styled = styled.set_properties(
        **{"text-align": "center", "vertical-align": "middle"}
    )

    # Style header
    styled = styled.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "bold"),
                    ("background-color", "rgba(0, 0, 0, 0.05)"),
                    ("border-bottom", "2px solid rgba(0, 0, 0, 0.1)"),
                ],
            }
        ]
    )

    # Apply background gradient if requested
    if highlight_cols:
        styled = _apply_gradient(styled, df, highlight_cols, cmap, axis)

    # Set index name if provided
    if index_name and show_index:
        styled = styled.set_caption(index_name)

    # Display the styled dataframe
    display_kwargs = {
        "width": 'stretch',
        "hide_index": not show_index,
    }
    if height:
        display_kwargs["height"] = height

    st.dataframe(styled, **display_kwargs)


def _build_formatter(df: pd.DataFrame, precision: int) -> Dict[str, str]:
    """Build formatting dictionary for dataframe columns."""
    formatter = {}

    for col in df.columns:
        series = df[col]

        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(series):
            continue

        # Percentage columns
        if col.endswith("%") or "(%)" in col or "rate" in col.lower():
            formatter[col] = f"{{:.{precision}f}}%"
        # Integer columns
        elif pd.api.types.is_integer_dtype(series):
            formatter[col] = "{:,d}"
        # Float columns that are effectively integers
        elif series.dropna().eq(series.dropna().astype(int)).all():
            formatter[col] = "{:,.0f}"
        # Regular float columns
        else:
            formatter[col] = f"{{:,.{precision}f}}"

    return formatter


def _apply_gradient(
    styled: Any,
    df: pd.DataFrame,
    highlight_cols: List[str],
    cmap: str,
    axis: int,
) -> Any:
    """Apply background gradient to specified columns."""
    valid_cols = [col for col in highlight_cols if col in df.columns]

    if not valid_cols:
        return styled

    # Apply gradient with low and high values for better visibility
    for col in valid_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            styled = safe_background_gradient(
                styled,
                cmap=cmap,
                subset=[col],
                axis=axis,
                vmin=df[col].quantile(0.1),
                vmax=df[col].quantile(0.9),
            )

    return styled


# ==================== UTILITY FUNCTIONS ====================


def create_metric_card_html(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
) -> str:
    """
    Create HTML for a custom metric card (for advanced layouts).

    Parameters:
        label: Metric label
        value: Metric value (formatted)
        delta: Optional delta value
        delta_color: Color scheme for delta ('normal', 'inverse')

    Returns:
        HTML string for the metric card
    """
    delta_html = ""
    if delta:
        color = (
            theme.metric_negative_color
            if delta_color == "inverse"
            else theme.metric_positive_color
        )
        delta_html = f'<div style="font-size: 0.9rem; color: {
            color}; margin-top: 4px;">{delta}</div>'

    return f"""
    <div style="
        background: rgba(0, 0, 0, 0.02);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    ">
        <div style="font-size: 0.9rem; color: rgba(0, 0, 0, 0.6); margin-bottom: 4px;">
            {label}
        </div>
        <div style="font-size: 1.5rem; font-weight: bold; color: rgba(0, 0, 0, 0.87);">
            {value}
        </div>
        {delta_html}
    </div>
    """


# ==================== EXPORT ALL PUBLIC FUNCTIONS ====================

__all__ = [
    "display_metric_cards",
    "render_download_button",
    "render_about_section",
    "render_filter_section",
    "render_dataframe_with_style",
    "create_metric_card_html",
    "ComponentTheme",
    "theme",
]
