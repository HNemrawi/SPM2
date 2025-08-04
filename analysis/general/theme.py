"""
Shared theme elements and styling for HMIS dashboard.
Modernized to match the neutral, adaptive design system.
"""
import streamlit as st
import plotly.graph_objects as go
from typing import Optional, Dict, Any, Union

# ==================== COLOR SYSTEM ====================
# Using a neutral color palette that adapts well to both light and dark themes

class Colors:
    """
    Neutral color system using modern design principles.
    These colors are chosen to have good contrast in both light and dark modes.
    """
    # Primary brand colors - work well in both modes
    PRIMARY = "#0066CC"  # Professional blue
    PRIMARY_HOVER = "#0052A3"  # Darker blue for hover states
    PRIMARY_LIGHT = "#E6F0FF"  # Very light blue for backgrounds
    
    # Semantic colors - carefully chosen for both modes
    SUCCESS = "#059862"  # Green that's not too bright
    SUCCESS_LIGHT = "#E6F7F1"
    WARNING = "#D97706"  # Amber that's readable in both modes
    WARNING_LIGHT = "#FEF3E2"
    DANGER = "#DC2626"  # Red that's not too harsh
    DANGER_LIGHT = "#FEE2E2"
    INFO = "#0066CC"  # Same as primary
    INFO_LIGHT = "#E6F0FF"
    
    # Neutral grays - work with any background
    NEUTRAL_900 = "#111827"  # Almost black
    NEUTRAL_800 = "#1F2937"  # Dark gray
    NEUTRAL_700 = "#374151"
    NEUTRAL_600 = "#4B5563"
    NEUTRAL_500 = "#6B7280"  # Mid gray
    NEUTRAL_400 = "#9CA3AF"
    NEUTRAL_300 = "#D1D5DB"
    NEUTRAL_200 = "#E5E7EB"
    NEUTRAL_100 = "#F3F4F6"  # Very light gray
    NEUTRAL_50 = "#F9FAFB"   # Almost white
    
    # Adaptive colors using currentColor and opacity
    BORDER_COLOR = "rgba(0, 0, 0, 0.1)"  # Works on any background
    SHADOW_COLOR = "rgba(0, 0, 0, 0.1)"
    OVERLAY_COLOR = "rgba(0, 0, 0, 0.05)"
    
    # Chart colors - distinct in both modes
    CHART_COLORS = [
        "#0066CC",  # Primary blue
        "#059862",  # Success green
        "#D97706",  # Warning amber
        "#DC2626",  # Danger red
        "#7C3AED",  # Purple
        "#0891B2",  # Cyan
        "#EC4899",  # Pink
        "#6366F1",  # Indigo
        "#84CC16",  # Lime
        "#F59E0B",  # Orange
    ]
    
    # Color scales for gradients
    BLUE_SCALE = [
        "#E6F0FF", "#CCE1FF", "#99C3FF", "#66A5FF", 
        "#3387FF", "#0066CC", "#0052A3", "#003D7A"
    ]
    RED_SCALE = [
        "#FEE2E2", "#FECACA", "#FCA5A5", "#F87171",
        "#EF4444", "#DC2626", "#B91C1C", "#991B1B"
    ]
    DIVERGING_SCALE = [
        "#DC2626", "#EF4444", "#FCA5A5", "#FDE68A",
        "#E6F0FF", "#93C5FD", "#3B82F6", "#0066CC"
    ]
    DISPARITY_COLOR_SCALE = ["#F4B183", "#D9D9D9", "#5B9BD5"]

# Create a global instance for backward compatibility
MAIN_COLOR = Colors.PRIMARY
SECONDARY_COLOR = Colors.WARNING
NEUTRAL_COLOR = Colors.NEUTRAL_500
SUCCESS_COLOR = Colors.SUCCESS
WARNING_COLOR = Colors.WARNING
DANGER_COLOR = Colors.DANGER

# Color sequences for charts
CUSTOM_COLOR_SEQUENCE = Colors.CHART_COLORS
BLUE_SCALE = Colors.BLUE_SCALE
RED_SCALE = Colors.RED_SCALE
DIVERGING_SCALE = Colors.DIVERGING_SCALE
DISPARITY_COLOR_SCALE = Colors.DISPARITY_COLOR_SCALE

# Plot template
PLOT_TEMPLATE = "plotly_white"  # Changed to white for better adaptability

# ==================== FORMATTING FUNCTIONS ====================

def fmt_int(n: Union[int, float, str]) -> str:
    """Format integer with thousands separator."""
    try:
        return f"{int(n):,}"
    except (ValueError, TypeError):
        return "0"

def fmt_float(n: Union[int, float, str], decimals: int = 1) -> str:
    """Format float with specified decimal places."""
    try:
        return f"{float(n):,.{decimals}f}"
    except (ValueError, TypeError):
        return f"0.{'0' * decimals}"

def fmt_pct(n: Union[int, float, str], decimals: int = 1) -> str:
    """Format number as percentage with specified decimal places."""
    try:
        return f"{float(n):,.{decimals}f}%"
    except (ValueError, TypeError):
        return f"0.{'0' * decimals}%"

def fmt_change(
    value: Union[int, float],
    previous: Optional[Union[int, float]] = None,
    as_pct: bool = False,
    with_sign: bool = True,
    decimals: int = 1
) -> str:
    """
    Format a change value with appropriate sign and formatting.
    
    Parameters:
        value: Current value
        previous: Previous value to calculate change
        as_pct: Whether to format as percentage
        with_sign: Whether to include + sign for positive values
        decimals: Number of decimal places
    
    Returns:
        Formatted change string
    """
    if previous is not None:
        change = value - previous
        if previous != 0:
            pct_change = (change / abs(previous)) * 100
        else:
            pct_change = 0 if value == 0 else 100
    else:
        change = value
        pct_change = value

    if as_pct:
        formatted = fmt_pct(pct_change, decimals)
        if with_sign and pct_change > 0:
            return f"+{formatted}"
        return formatted
    else:
        formatted = fmt_float(change, decimals)
        if with_sign and change > 0:
            return f"+{formatted}"
        return formatted

# ==================== CHART STYLING FUNCTIONS ====================

def apply_chart_style(
    fig: go.Figure,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    height: Optional[int] = None,
    showlegend: bool = True,
    legend_orientation: str = "h",
    use_container_width: bool = True
) -> go.Figure:
    """
    Apply consistent, modern styling to a Plotly figure.
    
    Parameters:
        fig: The figure to style
        title: Chart title
        xaxis_title: X-axis title
        yaxis_title: Y-axis title
        height: Chart height in pixels
        showlegend: Whether to show the legend
        legend_orientation: Legend orientation ('h' or 'v')
        use_container_width: Whether to use full container width
    
    Returns:
        The styled figure
    """
    # Modern layout with adaptive colors
    layout_updates = {
        "template": None,  # Reset template to apply custom styling
        "paper_bgcolor": "rgba(0, 0, 0, 0)",  # Transparent background
        "plot_bgcolor": "rgba(0, 0, 0, 0.02)",  # Very subtle background
        "font": dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
            size=12,
            color="rgba(0, 0, 0, 0.87)"  # High contrast text
        ),
        "margin": dict(l=60, r=30, t=60 if title else 30, b=60),
        "showlegend": showlegend,
        "colorway": Colors.CHART_COLORS,
        "hovermode": "closest",
        "autosize": True,
    }
    
    # Configure legend
    if showlegend:
        layout_updates["legend"] = dict(
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.15)",
            borderwidth=1,
            font=dict(color="rgba(0, 0, 0, 0.7)"),
            orientation=legend_orientation,
            yanchor="bottom" if legend_orientation == "h" else "top",
            y=-0.15 if legend_orientation == "h" else 0.99,
            xanchor="center" if legend_orientation == "h" else "left",
            x=0.5 if legend_orientation == "h" else 1.02
        )
    
    # Configure axes with subtle grid lines
    axis_config = dict(
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.1)",  # Subtle grid lines
        zeroline=True,
        zerolinecolor="rgba(0, 0, 0, 0.2)",
        linecolor="rgba(0, 0, 0, 0.2)",
        tickfont=dict(color="rgba(0, 0, 0, 0.7)")
    )
    
    layout_updates["xaxis"] = axis_config.copy()
    layout_updates["yaxis"] = axis_config.copy()
    
    # Add optional parameters
    if title:
        layout_updates["title"] = dict(
            text=title,
            font=dict(
                color="rgba(0, 0, 0, 0.87)",
                size=16,
                weight=600
            ),
            x=0.5,
            xanchor="center"
        )
    
    if xaxis_title:
        layout_updates["xaxis"]["title"] = dict(
            text=xaxis_title,
            font=dict(color="rgba(0, 0, 0, 0.7)")
        )
        
    if yaxis_title:
        layout_updates["yaxis"]["title"] = dict(
            text=yaxis_title,
            font=dict(color="rgba(0, 0, 0, 0.7)")
        )
        
    if height:
        layout_updates["height"] = height
    
    # Configure hover labels
    layout_updates["hoverlabel"] = dict(
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="rgba(0, 0, 0, 0.1)",
        font=dict(color="rgba(0, 0, 0, 0.87)")
    )
    
    # Apply all updates
    fig.update_layout(**layout_updates)
    
    # Update traces for better consistency
    fig.update_traces(
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            font=dict(color="rgba(0, 0, 0, 0.87)")
        )
    )
    
    return fig

# ==================== UI COMPONENTS ====================

def create_insight_container(
    title: str,
    content: str,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    type: str = "info"
) -> str:
    """
    Create a modern styled HTML container for insights.
    
    Parameters:
        title: Container title
        content: Markdown content to display
        icon: Emoji icon to display
        color: Border color (defaults based on type)
        type: Container type ('info', 'success', 'warning', 'danger')
    
    Returns:
        HTML for the styled container
    """
    # Type configuration with adaptive colors
    type_config = {
        "info": {
            "color": color or Colors.INFO,
            "bg": "rgba(0, 102, 204, 0.05)",  # Using primary color with low opacity
            "border": color or Colors.INFO,
            "icon": icon or "ℹ️"
        },
        "success": {
            "color": color or Colors.SUCCESS,
            "bg": "rgba(5, 152, 98, 0.05)",
            "border": color or Colors.SUCCESS,
            "icon": icon or "✅"
        },
        "warning": {
            "color": color or Colors.WARNING,
            "bg": "rgba(217, 119, 6, 0.05)",
            "border": color or Colors.WARNING,
            "icon": icon or "⚠️"
        },
        "danger": {
            "color": color or Colors.DANGER,
            "bg": "rgba(220, 38, 38, 0.05)",
            "border": color or Colors.DANGER,
            "icon": icon or "❌"
        }
    }
    
    config = type_config.get(type, type_config["info"])
    display_icon = icon or config["icon"]
    
    return f"""
    <div style="
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid {config['border']};
        border-left: 4px solid {config['border']};
        background-color: {config['bg']};
        transition: all 0.2s ease;
    ">
        <h4 style="
            color: {config['color']};
            margin: 0 0 0.5rem 0;
            font-size: 1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            <span style="font-size: 1.1rem;">{display_icon}</span>
            {title}
        </h4>
        <div style="
            color: rgba(0, 0, 0, 0.7);
            font-size: 0.9rem;
            line-height: 1.5;
        ">
            {content}
        </div>
    </div>
    """

def blue_divider():
    """Create a styled divider with adaptive colors."""
    st.markdown(
        f"""
        <hr style="
            border: none;
            height: 2px;
            background: linear-gradient(
                to right,
                transparent,
                {Colors.PRIMARY},
                transparent
            );
            margin: 2rem 0;
        " />
        """,
        unsafe_allow_html=True
    )

def styled_metric(
    label: str,
    value: Union[str, int, float],
    delta: Optional[Union[str, int, float]] = None,
    delta_color: str = "normal",
    help: Optional[str] = None
) -> None:
    """
    Display a styled metric with modern design.
    
    Parameters:
        label: Metric label
        value: Metric value
        delta: Optional delta value
        delta_color: Color scheme for delta ('normal', 'inverse', 'off')
        help: Optional help text
    """
    # Format value if numeric
    if isinstance(value, (int, float)):
        if isinstance(value, int) or value.is_integer():
            formatted_value = fmt_int(value)
        else:
            formatted_value = fmt_float(value)
    else:
        formatted_value = str(value)
    
    # Format delta if provided
    formatted_delta = None
    if delta is not None:
        if isinstance(delta, (int, float)):
            formatted_delta = fmt_change(delta, with_sign=True)
        else:
            formatted_delta = str(delta)
    
    # Use Streamlit's metric with enhanced styling
    st.metric(
        label=label,
        value=formatted_value,
        delta=formatted_delta,
        delta_color=delta_color,
        help=help
    )

# ==================== EXPORT ALL PUBLIC ITEMS ====================

__all__ = [
    # Color classes and constants
    'Colors',
    'MAIN_COLOR',
    'SECONDARY_COLOR',
    'NEUTRAL_COLOR',
    'SUCCESS_COLOR',
    'WARNING_COLOR',
    'DANGER_COLOR',
    'CUSTOM_COLOR_SEQUENCE',
    'BLUE_SCALE',
    'RED_SCALE',
    'DIVERGING_SCALE',
    'DISPARITY_COLOR_SCALE',
    'PLOT_TEMPLATE',
    
    # Formatting functions
    'fmt_int',
    'fmt_float',
    'fmt_pct',
    'fmt_change',
    
    # Chart styling
    'apply_chart_style',
    
    # UI components
    'create_insight_container',
    'blue_divider',
    'styled_metric'
]