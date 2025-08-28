"""
UI Styling and Theme Components
"""

from typing import Optional

import streamlit as st

from src.ui.themes.theme import theme

# ==================== COLOR SYSTEM ====================
# Import colors from unified theme but keep for backward compatibility


class NeutralColors:
    """
    Neutral color system using CSS variables for theme adaptability.
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
    NEUTRAL_50 = "#F9FAFB"  # Almost white

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
    ]


# ==================== CSS STYLES ====================


def get_neutral_css() -> str:
    """
    Generate CSS that works well in both light and dark modes.
    Uses relative colors and careful contrast ratios.
    """
    return f"""
    <style>
    /* ===== CSS Variables for Theme Adaptability ===== */
    :root {{
        /* Adaptive text colors using system preferences */
        --text-primary: color-mix(in srgb, currentColor 90%, transparent);
        --text-secondary: color-mix(in srgb, currentColor 70%, transparent);
        --text-muted: color-mix(in srgb, currentColor 50%, transparent);

        /* Adaptive backgrounds using transparency */
        --bg-card: rgba(128, 128, 128, 0.05);
        --bg-hover: rgba(128, 128, 128, 0.1);
        --bg-active: rgba(128, 128, 128, 0.15);

        /* Borders that work on any background */
        --border-color: {NeutralColors.BORDER_COLOR};
        --border-radius: 8px;
        --border-radius-sm: 4px;
        --border-radius-lg: 12px;

        /* Shadows with transparency */
        --shadow-sm: 0 1px 2px 0 {NeutralColors.SHADOW_COLOR};
        --shadow-md: 0 4px 6px -1px {NeutralColors.SHADOW_COLOR};
        --shadow-lg: 0 10px 15px -3px {NeutralColors.SHADOW_COLOR};

        /* Spacing system */
        --spacing-xs: 0.25rem;
        --spacing-sm: 0.5rem;
        --spacing-md: 1rem;
        --spacing-lg: 1.5rem;
        --spacing-xl: 2rem;
    }}

    /* ===== Global Resets ===== */
    * {{
        box-sizing: border-box;
    }}

    /* ===== Typography ===== */
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                     'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    /* Headings with better hierarchy */
    h1, h2, h3, h4, h5, h6 {{
        font-weight: 600;
        line-height: 1.25;
        margin-bottom: var(--spacing-md);
        color: var(--text-primary);
    }}

    h1 {{font-size: 2rem; }}
    h2 {{font-size: 1.5rem; }}
    h3 {{font-size: 1.25rem; }}
    h4 {{font-size: 1.125rem; }}

    /* ===== Container Styling ===== */
    .block-container {{
        padding: var(--spacing-lg) var(--spacing-xl) !important;
        max-width: 100%;
    }}

    /* ===== Card Components ===== */
    .neutral-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: var(--spacing-lg);
        margin-bottom: var(--spacing-md);
        transition: all 0.2s ease;
    }}

    .neutral-card:hover {{
        background: var(--bg-hover);
        box-shadow: var(--shadow-md);
    }}

    /* ===== Metric Cards ===== */
    div[data-testid="stMetric"],
    div[data-testid="metric-container"] {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: var(--spacing-lg);
        transition: all 0.2s ease;
        position: relative;
        overflow: visible;
        min-height: 80px;
    }}

    /* Metric label styling */
    div[data-testid="stMetric"] label {{
        font-size: 0.875rem !important;
        color: var(--text-secondary) !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        margin-bottom: 0.25rem !important;
    }}

    /* Metric value styling */
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.2 !important;
    }}

    /* Metric delta styling */
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {{
        font-size: 0.875rem !important;
        margin-top: 0.25rem !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }}

    /* Metric card accent border */
    div[data-testid="stMetric"]::before,
    div[data-testid="metric-container"]::before {{
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: {NeutralColors.PRIMARY};
    }}

    /* ===== Buttons ===== */
    .stButton > button {{
        background: {NeutralColors.PRIMARY};
        color: white;
        border: none;
        border-radius: var(--border-radius-sm);
        padding: var(--spacing-sm) var(--spacing-lg);
        font-weight: 500;
        transition: all 0.2s ease;
        cursor: pointer;
    }}

    .stButton > button:hover {{
        background: {NeutralColors.PRIMARY_HOVER};
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* Secondary button style */
    .stButton.secondary > button {{
        background: transparent;
        color: {NeutralColors.PRIMARY};
        border: 1px solid {NeutralColors.PRIMARY};
    }}

    .stButton.secondary > button:hover {{
        background: {NeutralColors.PRIMARY_LIGHT};
    }}

    /* ===== Expanders ===== */
    .streamlit-expanderHeader {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-sm);
        font-weight: 500;
        color: var(--text-primary);
        transition: all 0.2s ease;
    }}

    .streamlit-expanderHeader:hover {{
        background: var(--bg-hover);
    }}

    /* ===== DataFrames ===== */
    .dataframe {{
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        overflow: hidden;
    }}

    .dataframe thead th {{
        background: var(--bg-hover);
        color: var(--text-primary);
        font-weight: 600;
        padding: var(--spacing-sm) var(--spacing-md);
        border-bottom: 2px solid var(--border-color);
    }}

    .dataframe tbody tr {{
        transition: background 0.2s ease;
    }}

    .dataframe tbody tr:hover {{
        background: var(--bg-hover);
    }}

    .dataframe tbody td {{
        padding: var(--spacing-sm) var(--spacing-md);
        border-bottom: 1px solid var(--border-color);
    }}

    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab-list"] {{
        gap: var(--spacing-xs);
        background: var(--bg-card);
        padding: var(--spacing-xs);
        border-radius: var(--border-radius);
        border: 1px solid var(--border-color);
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border: none;
        color: var(--text-secondary);
        font-weight: 500;
        padding: var(--spacing-sm) var(--spacing-lg);
        border-radius: var(--border-radius-sm);
        transition: all 0.2s ease;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background: var(--bg-hover);
        color: var(--text-primary);
    }}

    .stTabs [aria-selected="true"] {{
        background: {NeutralColors.PRIMARY} !important;
        color: white !important;
    }}

    /* ===== Alerts & Info boxes ===== */
    .stAlert {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: var(--spacing-md);
    }}

    /* ===== Dividers ===== */
    hr {{
        border: none;
        border-top: 1px solid var(--border-color);
        margin: var(--spacing-xl) 0;
    }}

    /* Custom styled divider */
    .styled-divider {{
        height: 2px;
        background: linear-gradient(
            to right,
            transparent,
            {NeutralColors.PRIMARY},
            transparent
        );
        border: none;
        margin: var(--spacing-xl) 0;
    }}

    /* ===== Sidebar ===== */
    .css-1d391kg {{
        background: var(--bg-card);
        border-right: 1px solid var(--border-color);
    }}

    /* ===== Tooltips ===== */
    .tooltip {{
        background: {NeutralColors.NEUTRAL_900};
        color: white;
        padding: var(--spacing-sm);
        border-radius: var(--border-radius-sm);
        font-size: 0.875rem;
        box-shadow: var(--shadow-lg);
    }}

    /* ===== Loading states ===== */
    .stSpinner > div {{
        border-color: {NeutralColors.PRIMARY} transparent transparent transparent;
    }}

    /* ===== Accessibility improvements ===== */
    :focus {{
        outline: 2px solid {NeutralColors.PRIMARY};
        outline-offset: 2px;
    }}

    /* Remove default focus for better custom styling */
    *:focus:not(:focus-visible) {{
        outline: none;
    }}

    /* ===== Utility classes ===== */
    .text-muted {{color: var(--text-muted); }}
    .text-small {{font-size: 0.875rem; }}
    .text-large {{font-size: 1.125rem; }}
    .font-mono {{font-family: monospace; }}

    .mt-1 {{margin-top: var(--spacing-sm); }}
    .mt-2 {{margin-top: var(--spacing-md); }}
    .mt-3 {{margin-top: var(--spacing-lg); }}

    .mb-1 {{margin-bottom: var(--spacing-sm); }}
    .mb-2 {{margin-bottom: var(--spacing-md); }}
    .mb-3 {{margin-bottom: var(--spacing-lg); }}

    /* ===== Responsive adjustments ===== */
    @media (max-width: 768px) {{
        .block-container {{
            padding: var(--spacing-md) !important
        }}

        h1 {{font-size: 1.5rem
            }}
        h2 {{font-size: 1.25rem
            }}
        h3 {{font-size: 1.125rem
            }}
    }}

    /* ===== Fix for metric cards in columns ===== */
    .stColumn > div > div > div[data-testid="stVerticalBlock"] > div[data-testid="stMetric"] {{
        width: 100% !important;
        box-sizing: border-box !important;
    }}

    /* Ensure columns don't overflow */
    .stColumn {{
        min-width: 0 !important;
        flex: 1 1 0 !important;
    }}

    /* Fix metric container spacing */
    div[data-testid="stMetric"] {{
        margin-bottom: 0.5rem !important;
    }}

    /* Prevent text overflow in metric cards */
    div[data-testid="stMetric"] * {{
        max-width: 100% !important;
        overflow-wrap: break-word !important;
    }}
    </style>
    """


# ==================== COMPONENT STYLING FUNCTIONS ====================


def apply_custom_css():
    """Apply the unified theme CSS to the Streamlit app."""
    # Theme is now handled by Streamlit's config.toml
    st.markdown(
        get_neutral_css(), unsafe_allow_html=True
    )  # Keep legacy CSS for backward compatibility


def style_metric_cards(
    background_color: str = None,
    border_size_px: int = 1,
    border_color: str = None,
    border_radius_px: int = 8,
    border_left_color: str = None,
    box_shadow: bool = True,
) -> None:
    """
    Legacy function for backward compatibility.
    Applies styling for metric cards with customizable options.

    Parameters:
        background_color (str): Background color of the cards.
        border_size_px (int): Border width in pixels.
        border_color (str): Border color.
        border_radius_px (int): Border radius in pixels.
        border_left_color (str): Left border accent color.
        box_shadow (bool): Whether to apply a shadow effect.
    """
    # Use neutral theme colors if not specified
    if background_color is None:
        background_color = "var(--bg-card)"
    if border_color is None:
        border_color = "var(--border-color)"
    if border_left_color is None:
        border_left_color = NeutralColors.PRIMARY

    box_shadow_str = (
        "box-shadow: var(--shadow-md);" if box_shadow else "box-shadow: none;"
    )

    st.markdown(
        f"""
    <style>
        div[data-testid="stMetric"],
        div[data-testid="metric-container"] {{
            background: {background_color};
            border: {border_size_px}px solid {border_color};
            padding: 1.25rem 1rem;
            border-radius: {border_radius_px}px;
            border-left: 0.5rem solid {border_left_color} !important;
            {box_shadow_str}
            transition: all 0.2s ease;
            overflow: visible !important;
            min-height: 90px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        /* Ensure metric content doesn't get cut off */
        div[data-testid="stMetric"] label {{
            white-space: normal !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
            hyphens: auto !important;
            max-width: 100% !important;
        }}

        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
            white-space: normal !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
            max-width: 100% !important;
        }}

        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {{
            white-space: normal !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
            font-size: 0.875rem !important;
            max-width: 100% !important;
        }}

        /* Adjust column spacing for metrics */
        .stMetric {{
            min-width: 0 !important;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def create_info_box(
    content: str,
    type: str = "info",
    title: Optional[str] = None,
    icon: Optional[str] = None,
) -> str:
    """
    Create a themed info box styled for Streamlit with light/dark-friendly colors.

    Args:
        content (str): Main box content.
        type (str): One of 'info', 'success', 'warning', 'danger'.
        title (Optional[str]): Optional heading.
        icon (Optional[str]): Optional emoji or symbol.

    Returns:
        str: HTML snippet to render in Streamlit.
    """
    type_config = {
        "info": {
            "bg": "rgba(128, 128, 128, 0.08)",
            "border": "rgba(128, 128, 128, 0.4)",
            "icon": "",
        },
        "success": {
            "bg": "rgba(128, 128, 128, 0.08)",
            "border": "rgba(128, 128, 128, 0.4)",
            "icon": "",
        },
        "warning": {
            "bg": "rgba(128, 128, 128, 0.08)",
            "border": "rgba(128, 128, 128, 0.4)",
            "icon": "⚠️",
        },
        "danger": {
            "bg": "rgba(128, 128, 128, 0.08)",
            "border": "rgba(128, 128, 128, 0.4)",
            "icon": "❌",
        },
    }

    config = type_config.get(type, type_config["info"])
    icon = icon or config["icon"]

    # If there's a title, include icon with title
    if title:
        return f"""
        <div style='
            padding: 0.75rem 1rem;
            background-color: {config["bg"]};
            border-radius: 6px;
            border-left: 3px solid {config["border"]};
            margin-bottom: 0.75rem;
        '>
            <h4 style='
                color: currentColor;
                opacity: 0.9;
                margin: 0 0 0.25rem 0;
                font-size: 0.95rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            '>
                <span style='font-size: 1rem;'>{icon}</span>
                {title}
            </h4>
            <p style='
                color: currentColor;
                opacity: 0.75;
                font-size: 0.85rem;
                margin: 0;
                line-height: 1.5;
            '>
                {content}
            </p>
        </div>
        """
    else:
        # No title, just icon with content
        return f"""
        <div style='
            padding: 0.75rem 1rem;
            background-color: {config["bg"]};
            border-radius: 6px;
            border-left: 3px solid {config["border"]};
            margin-bottom: 0.75rem;
        '>
            <p style='
                color: currentColor;
                opacity: 0.75;
                font-size: 0.85rem;
                margin: 0;
                line-height: 1.5;
                display: flex;
                align-items: flex-start;
                gap: 0.5rem;
            '>
                <span style='font-size: 1rem; line-height: 1.2;'>{icon}</span>
                <span>{content}</span>
            </p>
        </div>
        """


def style_dataframe(
    df, highlight_columns: Optional[list] = None, precision: int = 2
) -> str:
    """
    Apply styling to a pandas DataFrame for better display.

    Args:
        df: DataFrame to style
        highlight_columns: Columns to highlight (optional)
        precision: Decimal precision for floats

    Returns:
        Styled DataFrame
    """
    # Create styler
    styler = df.style

    # Set precision
    styler = styler.format(precision=precision)

    # Highlight specific columns if requested
    if highlight_columns:

        def highlight_cols(s):
            return [
                (
                    "background-color: var(--bg-hover)"
                    if s.name in highlight_columns
                    else ""
                )
                for _ in s
            ]

        styler = styler.apply(highlight_cols, axis=0)

    # Add hover effect
    styler = styler.set_table_styles(
        [
            {
                "selector": "tr:hover",
                "props": [("background-color", "var(--bg-hover)")],
            },
            {
                "selector": "th",
                "props": [
                    ("background-color", "var(--bg-hover)"),
                    ("color", "var(--text-primary)"),
                    ("font-weight", "600"),
                ],
            },
        ]
    )

    return styler


def create_styled_divider(style: str = "solid") -> str:
    """
    Create a styled divider.

    Args:
        style: Divider style (solid, gradient, dots)

    Returns:
        HTML string for the divider
    """
    if style == "gradient":
        return '<hr class="styled-divider">'
    elif style == "dots":
        return f"""
        <div style="
            text-align: center;
            margin: 32px 0;
            color: {NeutralColors.NEUTRAL_400};
            letter-spacing: 8px;
        ">•••</div>
        """
    else:
        return "<hr>"


# ==================== LAYOUT HELPERS ====================


def create_columns_with_gap(ratios: list, gap: str = "20px"):
    """
    Create columns with custom gap spacing.

    Args:
        ratios: List of column width ratios
        gap: Gap between columns

    Returns:
        List of column objects
    """
    # Apply custom CSS for gap
    st.markdown(
        f"""
    <style>
    .row-widget.stHorizontalBlock {{
        gap: {gap};
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    return st.columns(ratios)


# ==================== THEME UTILITIES ====================


def get_chart_colors() -> list:
    """Get the chart color sequence for consistency."""
    return theme.colors.chart_colors  # Use unified theme colors


def apply_chart_theme(fig):
    """
    Apply neutral theme to Plotly charts.

    Args:
        fig: Plotly figure object

    Returns:
        Modified figure
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color="var(--text-primary)",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        colorway=NeutralColors.CHART_COLORS,
        xaxis=dict(
            gridcolor="var(--border-color)",
            zerolinecolor="var(--border-color)",
        ),
        yaxis=dict(
            gridcolor="var(--border-color)",
            zerolinecolor="var(--border-color)",
        ),
    )
    return fig


# ==================== EXPORT ALL PUBLIC FUNCTIONS ====================

__all__ = [
    "NeutralColors",
    "apply_custom_css",
    "style_metric_cards",
    "create_info_box",
    "style_dataframe",
    "create_styled_divider",
    "create_columns_with_gap",
    "get_chart_colors",
    "apply_chart_theme",
]
