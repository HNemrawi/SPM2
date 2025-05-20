"""
Shared theme elements and styling for HMIS dashboard.
"""
import streamlit as st

# ────────────────────────── Color schemes and visualization constants ────────────────────────── #

# Main color palette - based on the equity section's colors
MAIN_COLOR = "rgb(73,160,181)"  # Primary blue
SECONDARY_COLOR = "rgb(255,99,71)"  # Coral red for contrast
NEUTRAL_COLOR = "rgb(128,128,128)"  # Neutral gray
SUCCESS_COLOR = "rgb(75,181,67)"  # Green for positive metrics
WARNING_COLOR = "rgb(255,165,0)"  # Orange for warning indicators
DANGER_COLOR = "rgb(255,75,75)"  # Bright red for danger/negative indicators

# Color scales for gradients
BLUE_SCALE = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c"]
RED_SCALE = ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"]
DIVERGING_SCALE = ["#d73027", "#f46d43", "#fdae61", "#fee090", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4"]
DISPARITY_COLOR_SCALE = ["#F4B183", "#D9D9D9", "#5B9BD5"]  # From equity section

# Standard color sequence for categorical variables
CUSTOM_COLOR_SEQUENCE = [
    "#5B9BD5",  # medium blue – strong and clean
    "#F4B183",  # peachy orange – warmer contrast
    "#1F4E79",  # dark navy – strong contrast to both above
    "#A5A5A5",  # solid gray – neutral balance
    "#FFC000"   # gold – brighter pop, still fits warm tone
]

# Plot template
PLOT_TEMPLATE = "plotly_dark"  # Dark theme for better contrast and modern look

# ────────────────────────── Formatting functions ────────────────────────── #

def fmt_int(n) -> str:
    """Format integer with thousands separator."""
    try:
        return f"{int(n):,}"
    except (ValueError, TypeError):
        return "0"

def fmt_float(n, decimals=1) -> str:
    """Format float with specified decimal places."""
    try:
        return f"{float(n):,.{decimals}f}"
    except (ValueError, TypeError):
        return f"0.{'0' * decimals}"

def fmt_pct(n, decimals=1) -> str:
    """Format number as percentage with specified decimal places."""
    try:
        return f"{float(n):,.{decimals}f}%"
    except (ValueError, TypeError):
        return f"0.{'0' * decimals}%"

def fmt_change(value, previous=None, as_pct=False, with_sign=True) -> str:
    """
    Format a change value with appropriate sign and formatting.
    
    Parameters:
    -----------
    value : float
        Current value
    previous : float, optional
        Previous value to calculate change
    as_pct : bool
        Whether to format as percentage
    with_sign : bool
        Whether to include + sign for positive values
    
    Returns:
    --------
    str
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
        formatted = fmt_pct(pct_change)
        if with_sign and pct_change > 0:
            return f"+{formatted}"
        return formatted
    else:
        formatted = fmt_float(change)
        if with_sign and change > 0:
            return f"+{formatted}"
        return formatted

# ────────────────────────── Chart styling functions ────────────────────────── #

def apply_chart_style(fig, title=None, xaxis_title=None, yaxis_title=None, height=None, 
                      showlegend=True, legend_orientation="h"):
    """
    Apply consistent styling to a Plotly figure.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The figure to style
    title : str, optional
        Chart title
    xaxis_title : str, optional
        X-axis title
    yaxis_title : str, optional
        Y-axis title
    height : int, optional
        Chart height in pixels
    showlegend : bool
        Whether to show the legend
    legend_orientation : str
        Legend orientation ('h' for horizontal, 'v' for vertical)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The styled figure
    """
    # Default layout updates
    layout_updates = {
        "template": PLOT_TEMPLATE,
        "margin": dict(l=40, r=40, t=50, b=40),
        "showlegend": showlegend,
        "legend": dict(
            orientation=legend_orientation,
            yanchor="bottom" if legend_orientation == "h" else "top",
            y=-0.15 if legend_orientation == "h" else 0.99,
            xanchor="center" if legend_orientation == "h" else "left",
            x=0.5 if legend_orientation == "h" else 1.02
        ),
        "xaxis": dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.2)"
        ),
        "yaxis": dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.2)"
        )
    }
    
    # Add optional parameters if provided
    if title:
        layout_updates["title"] = dict(text=title, font=dict(size=18))
    
    if xaxis_title:
        layout_updates["xaxis"]["title"] = xaxis_title
        
    if yaxis_title:
        layout_updates["yaxis"]["title"] = yaxis_title
        
    if height:
        layout_updates["height"] = height
        
    # Apply all updates
    fig.update_layout(**layout_updates)
    
    return fig

def create_insight_container(title, content, icon=None, color=MAIN_COLOR):
    """
    Create a styled HTML container for insights.
    
    Parameters:
    -----------
    title : str
        Container title
    content : str
        Markdown content to display
    icon : str, optional
        Emoji icon to display
    color : str
        Border and title color
    
    Returns:
    --------
    str
        HTML for the styled container
    """
    icon_html = f"{icon} " if icon else ""
    return f"""
    <div style="padding:15px; border-radius:5px; margin-bottom:15px; border:1px solid {color}">
        <h3 style="color:{color}">{icon_html}{title}</h3>
        <div>{content}</div>
    </div>
    """


def blue_divider():
    st.markdown(
        """
        <hr style="border: none; height: 3px; background-color: #00629b; margin-top: 10px; margin-bottom: 25px;" />
        """,
        unsafe_allow_html=True
    )
