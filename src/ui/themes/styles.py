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

    /* ====================================================================
       Centralized rules (Phase 1, see UI_LAYOUT_AUDIT.md)
       Three CSS sources were merged here so the whole app loads one block:
         - main.inject_custom_css        (was global)
         - components.apply_metric_card_style  (was per-call on dashboard)
         - dashboard.apply_neutral_tab_style   (was dashboard-only)
       Cascade order is intentional: rules later in the file override
       earlier rules of equal specificity.
       ==================================================================== */

    /* ===== From main.inject_custom_css ===== */
    /* Soften Streamlit multiselect/filter pills */
    .stMultiSelect [data-baseweb="tag"] {{
        background-color: {theme.colors.primary_bg_subtle} !important;
        border: 1px solid {theme.colors.border} !important;
        color: {theme.colors.text_primary} !important;
    }}

    /* Native dataframe wrapper styling (applied alongside the .dataframe
       rules above; .stDataFrame is the Streamlit wrapper, .dataframe is
       the inner pandas-emitted table). */
    .stDataFrame {{
        border: 1px solid {theme.colors.border} !important;
        border-radius: {theme.borders.radius_md} !important;
    }}

    .stDataFrame th {{
        background-color: {theme.colors.background_secondary} !important;
        color: {theme.colors.text_primary} !important;
        font-weight: 600 !important;
        padding: 12px !important;
        border-bottom: 2px solid {theme.colors.border} !important;
    }}

    .stDataFrame td {{
        padding: 10px 12px !important;
        border-bottom: 1px solid {theme.colors.border_light} !important;
    }}

    .stDataFrame tr:hover {{
        background-color: {theme.colors.surface_hover} !important;
    }}

    /* Soften alert boxes (overrides the .stAlert block above). */
    .stAlert {{
        border-radius: {theme.borders.radius_md} !important;
        padding: 1rem !important;
    }}

    /* Button overrides (theme-token versions; the .stButton block above
       is the structural baseline). */
    .stButton > button {{
        border-radius: {theme.borders.radius_md} !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: {theme.shadows.md} !important;
    }}

    /* Expander header (theme-token override of .streamlit-expanderHeader
       above). */
    .streamlit-expanderHeader {{
        background-color: {theme.colors.background_secondary} !important;
        border-radius: {theme.borders.radius_sm} !important;
        font-weight: 600 !important;
    }}

    /* Selectbox border radius. */
    .stSelectbox [data-baseweb="select"] {{
        border-radius: {theme.borders.radius_md} !important;
    }}

    /* Notification visual weight. */
    [data-testid="stNotification"] {{
        background-color: {theme.colors.info_bg_subtle} !important;
        border-left: 3px solid {theme.colors.info} !important;
    }}

    /* ===== Metric cards (was components.apply_metric_card_style) =====
       These rules previously injected only on dashboard sections that
       called ui.apply_metric_card_style(). Now applied globally so every
       st.metric in the app gets the same enhanced treatment. */
    div[data-testid="metric-container"] {{
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,250,252,0.9) 100%);
        border: 1px solid {theme.colors.border};
        border-left: 4px solid {theme.colors.primary};
        border-radius: {theme.borders.radius_lg};
        padding: 1.25rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        min-height: auto !important;
        height: auto !important;
        overflow: visible !important;
    }}

    div[data-testid="metric-container"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }}

    div[data-testid="metric-container"] label {{
        color: {theme.colors.text_muted};
        font-size: {theme.typography.size_sm};
        font-weight: {theme.typography.weight_medium};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.4;
    }}

    div[data-testid="metric-container"] [data-testid="metric-value"] {{
        color: {theme.colors.text_primary};
        font-weight: {theme.typography.weight_bold};
        font-size: clamp(1.25rem, 2vw, {theme.typography.size_2xl});
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        line-height: 1.2;
    }}

    div[data-testid="metric-container"] [data-testid="metric-delta"] {{
        font-size: {theme.typography.size_sm};
        font-weight: {theme.typography.weight_medium};
        white-space: normal !important;
        word-wrap: break-word !important;
    }}

    /* Equal-height metric containers in column layouts. */
    div[data-testid="column"] > div > div[data-testid="metric-container"] {{
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}

    /* ===== Tab strip (was dashboard.apply_neutral_tab_style) =====
       NOTE: full-bleed hack ahead. The .stTabs [data-baseweb="tab-list"]
       block uses `margin: 0 -5rem; padding: 1rem 5rem;` to bleed the tab
       strip past the page padding, and `.main > .block-container` sets
       that page padding to 5rem so the negative margin lines up. Both
       rules MUST stay in sync, and BOTH are fragile against Streamlit
       DOM-structure changes — revalidate visually on every Streamlit
       upgrade. */
    .stTabs {{
        width: 100% !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.75rem;
        background-color: transparent;
        padding: 1rem;
        border-radius: 0;
        border: none;
        margin: 0 -5rem;
        padding: 1rem 5rem;
        margin-bottom: 2rem;
        background: var(--background-secondary, rgba(0, 0, 0, 0.02));
        border-top: 1px solid var(--border-color, rgba(0, 0, 0, 0.1));
        border-bottom: 1px solid var(--border-color, rgba(0, 0, 0, 0.1));
        display: flex;
        justify-content: center;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 52px;
        padding: 0 40px;
        background-color: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 8px;
        color: rgba(0, 0, 0, 0.85);
        font-weight: 500;
        font-size: 1rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        transition: all 0.2s ease;
        white-space: nowrap;
        position: relative;
        flex: 1 1 auto;
        min-width: 140px;
        max-width: 220px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin: 0 6px;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background-color: rgba(255, 255, 255, 1);
        border-color: rgba(0, 0, 0, 0.15);
        color: rgba(0, 0, 0, 0.95);
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {NeutralColors.PRIMARY} !important;
        color: white !important;
        font-weight: 600;
        border-color: {NeutralColors.PRIMARY} !important;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3) !important;
        transform: translateY(-1px);
    }}

    .stTabs [aria-selected="true"]:hover {{
        background-color: {NeutralColors.PRIMARY} !important;
        filter: brightness(0.9);
        border-color: {NeutralColors.PRIMARY} !important;
        box-shadow: 0 3px 10px rgba(33, 150, 243, 0.4) !important;
    }}

    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 0;
        background-color: transparent;
    }}

    .stTabs [data-baseweb="tab"]:focus {{
        outline: 2px solid {NeutralColors.PRIMARY};
        outline-offset: 2px;
        box-shadow: 0 0 0 4px rgba(33, 150, 243, 0.1);
    }}

    .stTabs [data-baseweb="tab"]:focus:not(:focus-visible) {{
        outline: none;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }}

    .tab-content {{
        padding: 1.5rem 0;
    }}

    /* Page padding paired with the negative-margin tab bleed above. */
    .main > .block-container {{
        max-width: 100%;
        padding-left: 5rem;
        padding-right: 5rem;
    }}

    .stTabs [data-baseweb="tab"]:active {{
        transform: translateY(0);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }}

    .stTabs [data-baseweb="tab"][disabled] {{
        opacity: 0.5;
        cursor: not-allowed;
        transform: none;
    }}

    .stTabs [data-baseweb="tab"] > span {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .stTabs [data-baseweb="tab"]::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 8px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0) 100%);
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.2s ease;
    }}

    .stTabs [data-baseweb="tab"]:hover::before {{
        opacity: 1;
    }}

    .stTabs [aria-selected="true"]::before {{
        display: none;
    }}

    .stTabs [data-baseweb="tab"],
    .stTabs [data-baseweb="tab-list"] {{
        transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
    }}

    @media (max-width: 768px) {{
        .stTabs [data-baseweb="tab-list"] {{
            margin: 0 -1rem;
            padding: 0.75rem 1rem;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: thin;
            gap: 0.5rem;
            justify-content: flex-start;
        }}

        .stTabs [data-baseweb="tab"] {{
            padding: 0 20px;
            font-size: 0.9rem;
            height: 44px;
            margin: 0 3px;
            flex: 0 0 auto;
            min-width: 120px;
        }}

        .main > .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{
            height: 4px;
        }}

        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {{
            background: rgba(0, 0, 0, 0.05);
        }}

        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 2px;
        }}
    }}
    </style>
    """


# ==================== COMPONENT STYLING FUNCTIONS ====================


def apply_custom_css():
    """Apply the unified theme CSS to the Streamlit app.

    Single global CSS injection — see ``get_neutral_css`` for the
    bundled rules. Called once per session from
    ``main.inject_custom_css``.
    """
    st.markdown(get_neutral_css(), unsafe_allow_html=True)


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
    "create_info_box",
    "style_dataframe",
    "create_styled_divider",
    "create_columns_with_gap",
    "get_chart_colors",
    "apply_chart_theme",
]
