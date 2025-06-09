"""
Equity analysis section for HMIS dashboard - Improved Version
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas import DataFrame, Timestamp
from scipy import stats

from analysis.general.data_utils import DEMOGRAPHIC_DIMENSIONS
from analysis.general.filter_utils import (
    get_filter_timestamp, hash_data, init_section_state, is_cache_valid, invalidate_cache
)
from analysis.general.theme import (
    CUSTOM_COLOR_SEQUENCE, MAIN_COLOR, NEUTRAL_COLOR, PLOT_TEMPLATE, SECONDARY_COLOR,
    SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR, apply_chart_style, fmt_int, fmt_pct,
    blue_divider
)

# Constants
EQUITY_SECTION_KEY = "equity_analysis"

# Color scales for equity visualization
EQUITY_COLOR_SCALE = {
    "severe": DANGER_COLOR,      # Red for severe disparities
    "significant": WARNING_COLOR, # Orange for significant disparities  
    "moderate": NEUTRAL_COLOR,   # Gray for moderate disparities
    "none": SUCCESS_COLOR       # Green for no/minimal disparities
}

def _safe_div(a: float, b: float, default: float = 0.0, multiplier: float = 1.0) -> float:
    """Safe division with optional multiplier."""
    return round((a / b) * multiplier if b else default, 1)

def _calculate_chart_height(num_groups: int, base_height: int = 450) -> int:
    """Calculate optimal chart height based on number of groups."""
    if num_groups <= 6:
        return base_height
    elif num_groups <= 12:
        return base_height + 100
    elif num_groups <= 20:
        return base_height + 200
    else:
        return min(base_height + 350, 800)

def _get_disparity_color(di_value: float) -> str:
    """Get color based on disparity index value."""
    if di_value >= 0.95:
        return SUCCESS_COLOR  # Green - minimal disparity
    elif di_value >= 0.8:
        return MAIN_COLOR     # Blue - moderate disparity
    elif di_value >= 0.5:
        return WARNING_COLOR  # Orange - significant disparity
    else:
        return DANGER_COLOR   # Red - severe disparity

def _get_disparity_level(di_value: float) -> str:
    """Get disparity level description."""
    if di_value >= 0.95:
        return "Minimal"
    elif di_value >= 0.8:
        return "Moderate"
    elif di_value >= 0.5:
        return "Significant"
    else:
        return "Severe"

def _create_disparity_summary_html(
    best_group: pd.Series,
    worst_group: pd.Series,
    gap: float,
    total_improvement: int,
    is_returns: bool,
    outcome_name: str
) -> str:
    """Create HTML summary for disparity findings."""
    
    # Determine colors based on metric type
    best_color = SUCCESS_COLOR
    worst_color = DANGER_COLOR
    impact_color = MAIN_COLOR
    
    # Create impact description
    if abs(total_improvement) > 0:
        impact_type = "fewer returns" if is_returns else "more successful exits"
        impact_text = f"""
        <div style="background-color: rgba(33,102,172,0.1); border: 1px solid {impact_color}; 
                    border-radius: 10px; padding: 20px; margin: 20px 0;">
            <h4 style="color: {impact_color}; margin: 0 0 10px 0;">💡 Potential System Impact</h4>
            <p style="margin: 0; font-size: 16px;">
                If all groups performed at the level of <strong>{best_group['group']}</strong>, 
                the system could achieve approximately 
                <strong style="font-size: 20px; color: {SUCCESS_COLOR};">{abs(total_improvement):,}</strong> 
                {impact_type}.
            </p>
        </div>
        """
    else:
        impact_text = ""
    
    return f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0;">
        
        <div style="background-color: rgba(73,160,181,0.15); border: 2px solid {best_color}; 
                    border-radius: 10px; padding: 20px; text-align: center;">
            <h4 style="color: {best_color}; margin: 0;">Best Performer</h4>
            <h2 style="margin: 10px 0;">{best_group['group']}</h2>
            <h3 style="color: {best_color}; margin: 0;">{fmt_pct(best_group['outcome_rate'])}</h3>
            <p style="margin: 5px 0 0 0; font-size: 14px;">
                {"Lowest return rate" if is_returns else "Highest exit rate"}
            </p>
        </div>
        
        <div style="background-color: rgba(255,99,71,0.15); border: 2px solid {worst_color}; 
                    border-radius: 10px; padding: 20px; text-align: center;">
            <h4 style="color: {worst_color}; margin: 0;">Needs Improvement</h4>
            <h2 style="margin: 10px 0;">{worst_group['group']}</h2>
            <h3 style="color: {worst_color}; margin: 0;">{fmt_pct(worst_group['outcome_rate'])}</h3>
            <p style="margin: 5px 0 0 0; font-size: 14px;">
                {"Highest return rate" if is_returns else "Lowest exit rate"}
            </p>
        </div>
        
        <div style="background-color: rgba(128,128,128,0.15); border: 2px solid {NEUTRAL_COLOR}; 
                    border-radius: 10px; padding: 20px; text-align: center;">
            <h4 style="color: {NEUTRAL_COLOR}; margin: 0;">Performance Gap</h4>
            <h2 style="margin: 10px 0;">{fmt_pct(gap)}</h2>
            <h3 style="color: {NEUTRAL_COLOR}; margin: 0;">percentage points</h3>
            <p style="margin: 5px 0 0 0; font-size: 14px;">
                Difference between best and worst
            </p>
        </div>
        
    </div>
    {impact_text}
    """

def _create_methodology_html(
    equity_label: str,
    outcome_name: str,
    t0: Timestamp,
    t1: Timestamp,
    min_pop: int,
    return_window: Optional[int] = None
) -> str:
    """Create methodology HTML section."""
    
    outcome_definitions = ""
    if "Permanent housing exits" in outcome_name:
        outcome_definitions = f"""
        <div style="background-color: rgba(75,181,67,0.1); border: 1px solid {SUCCESS_COLOR}; 
                    border-radius: 5px; padding: 15px; margin-bottom: 20px;">
            <h4 style="margin-top: 0;">Permanent Housing Exits</h4>
            <ul style="margin-bottom: 0;">
                <li>Clients who exited to a permanent housing destination during the reporting period</li>
                <li>Base population: All clients who exited programs during the reporting period</li>
                <li><strong>Higher rates are better</strong> (more successful housing placements)</li>
            </ul>
        </div>
        """
    else:  # Returns to homelessness
        outcome_definitions = f"""
        <div style="background-color: rgba(255,99,71,0.1); border: 1px solid {SECONDARY_COLOR}; 
                    border-radius: 5px; padding: 15px; margin-bottom: 20px;">
            <h4 style="margin-top: 0;">Returns to Homelessness</h4>
            <ul style="margin-bottom: 0;">
                <li>Clients who exited to PH and returned to homelessness within {return_window} days</li>
                <li>Base population: All clients who exited to permanent housing during the period</li>
                <li>Excludes: Prevention, coordinated entry, and services-only project enrollments</li>
                <li>Direct PH placements (move-in date = project start) not counted as returns</li>
                <li><strong>Lower rates are better</strong> (fewer people returning to homelessness)</li>
            </ul>
        </div>
        """
    
    return f"""
    <div style="background-color: rgba(0,0,0,0.2); border-radius: 10px; padding: 20px;">
        
        <h3 style="color: {MAIN_COLOR}; margin-top: 0;">Analysis Parameters</h3>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                <td style="padding: 8px; width: 50%;"><strong>Demographic dimension:</strong></td>
                <td style="padding: 8px;">{equity_label}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                <td style="padding: 8px;"><strong>Outcome measure:</strong></td>
                <td style="padding: 8px;">{outcome_name}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                <td style="padding: 8px;"><strong>Date range:</strong></td>
                <td style="padding: 8px;">{t0.date()} to {t1.date()}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                <td style="padding: 8px;"><strong>Minimum group size:</strong></td>
                <td style="padding: 8px;">{min_pop}</td>
            </tr>
        </table>
        
        <h3 style="color: {MAIN_COLOR};">Statistical Methods</h3>
        <ul style="margin-bottom: 20px;">
            <li>Chi-square test with Yates' correction for groups with n ≥ 5 in all cells</li>
            <li>Fisher's exact test for smaller samples</li>
            <li>Statistical significance threshold: p < 0.05</li>
            <li>Groups below minimum size threshold excluded from analysis</li>
        </ul>
        
        <h3 style="color: {MAIN_COLOR};">Outcome Definitions</h3>
        {outcome_definitions}
        
        <h3 style="color: {WARNING_COLOR};">Important Limitations</h3>
        <ul style="margin-bottom: 0;">
            <li>Disparities show correlation, not causation</li>
            <li>Multiple unmeasured factors may influence outcomes</li>
            <li>Single outcome measure - comprehensive assessment requires multiple metrics</li>
            {"<li>Returns analysis depends on complete re-entry data capture</li>" if return_window else ""}
        </ul>
        
        <h3 style="color: {MAIN_COLOR};">Disparity Index Calculation</h3>
        <div style="background-color: rgba(255,255,255,0.05); border-radius: 5px; padding: 15px;">
            <p style="margin: 0;">
                {"For returns: DI = (Lowest Rate ÷ Group Rate)<br>Groups with lowest return rates get DI = 1.0 (best)" if "Returns" in outcome_name else "For exits: DI = (Group Rate ÷ Highest Rate)<br>Groups with highest exit rates get DI = 1.0 (best)"}
            </p>
        </div>
    </div>
    """

def equity_analysis(
    df: DataFrame,
    demographic_col: str,
    _outcome_metric,
    start: Timestamp,
    end: Timestamp,
    min_population: int = 30,
    _population_filter=None,
    restrict_groups: Optional[List[Any]] = None,
    return_window: int = 730,
    full_df: Optional[DataFrame] = None
) -> DataFrame:
    """
    Compare outcome rates across demographic groups and identify disparities.

    Parameters
    ----------
    df : DataFrame
        The input dataset (filtered based on user selections).
    demographic_col : str
        The column name to group by (e.g., 'Race', 'Gender').
    _outcome_metric : callable
        A function that receives (df, start, end) and returns a set of ClientIDs who meet the outcome.
    start : Timestamp
        Start date of the analysis window.
    end : Timestamp
        End date of the analysis window.
    min_population : int, optional
        Minimum group size to include in results (default is 30).
    _population_filter : callable, optional
        A function that filters the base population and returns ClientIDs (default is None).
    restrict_groups : list, optional
        A list of specific group values to restrict the analysis to (default is None).
    return_window : int, optional
        Number of days after exit to check for returns (default is 730).
    full_df : DataFrame, optional
        The complete dataset, needed for return-to-homelessness calculations (default is None).

    Returns
    -------
    DataFrame
        A DataFrame with equity analysis results
    """
    # Check required columns
    required_cols = ["ClientID", "ProjectStart", "ProjectExit"]
    if not all(col in df.columns for col in required_cols + [demographic_col]):
        raise KeyError(f"Missing column(s): one or more required columns not found")

    # Step 1: Build the base population based on the type of analysis
    if _population_filter is None:
        # For PH exits, base population is all clients who exited during the period
        pop_mask = (df["ProjectExit"] >= start) & (df["ProjectExit"] <= end)
        population = df.loc[pop_mask].copy()
    else:
        # For custom populations like short-stay or returns, use the provided filter
        pop_ids = _population_filter(df, start, end)
        population = df[df["ClientID"].isin(pop_ids)].copy()

    # Convert any non-string values to strings in the demographic column
    if demographic_col in population.columns:
        population.loc[:, demographic_col] = population[demographic_col].astype(str)

    # Step 2: Apply subdimension filtering if restrict_groups provided
    if restrict_groups is not None:
        # Convert restrict_groups to strings for comparison
        string_restrict_groups = [str(group) for group in restrict_groups]
        population = population[population[demographic_col].isin(string_restrict_groups)]

    # Handle missing/invalid values
    missing_values = ["", "nan", "NaN", "None", "none", "null", "Null", "NA", "na", "N/A", "n/a"]
    population = population[~population[demographic_col].isin(missing_values)]

    # Handle empty population after filtering
    if population.empty:
        return pd.DataFrame(
            columns=[
                demographic_col,
                "population",
                "population_pct",
                "outcome_count",
                "outcome_rate",
                "disparity_index",
                "p_value",
                "potential_improvement",
                "sig_marker"
            ]
        )

    # Step 3: Get unique client counts by demographic group
    client_demos = population[["ClientID", demographic_col]].drop_duplicates()
    demo_pop = client_demos.groupby(demographic_col)["ClientID"].count()
    universe = int(demo_pop.sum())

    # Step 4: Get the overall outcome IDs based on the metric type
    if "return" in str(_outcome_metric).lower() and full_df is not None:
        # For returns to homelessness, we need the full dataset to scan for re-entries
        from analysis.general.data_utils import return_after_exit
        outcome_ids_full = return_after_exit(population, full_df, start, end, return_window)
    else:
        # Standard case for other metrics (PH exits, short stays)
        outcome_ids_full = _outcome_metric(population, start, end)

    rows: List[Dict[str, Any]] = []

    # Step 5: Loop through each group and compute metrics
    for demo_value, demo_count in demo_pop.items():
        # Skip invalid values
        if demo_value in missing_values:
            continue

        if restrict_groups is not None and demo_value not in [str(g) for g in restrict_groups]:
            continue

        # Skip groups that don't meet the minimum population threshold
        if demo_count < min_population:
            continue

        # Get data for this demographic group
        demo_df = population[population[demographic_col] == demo_value]
        
        # Get outcome metrics for this demographic group
        if "return" in str(_outcome_metric).lower() and full_df is not None:
            # For returns, use the specialized function with the full dataset
            from analysis.general.data_utils import return_after_exit
            outcome_ids_demo = return_after_exit(demo_df, full_df, start, end, return_window)
        else:
            # Standard outcome calculation for other metrics
            outcome_ids_demo = _outcome_metric(demo_df, start, end)
            
        outcome_demo = len(outcome_ids_demo)
        outcome_other = len(outcome_ids_full) - outcome_demo
        pop_other = universe - demo_count

        # Build contingency table for chi-square test
        # [outcome_demo, non_outcome_demo]
        # [outcome_other, non_outcome_other]
        contingency = np.array(
            [
                [outcome_demo, demo_count - outcome_demo],
                [outcome_other, pop_other - outcome_other],
            ]
        )

        # Chi-square test with Yates' correction for small samples
        try:
            # Only run the test if we have sufficient data in all cells
            if np.all(contingency >= 5):  # Standard minimum for chi-square
                _, p_val, _, _ = stats.chi2_contingency(
                    contingency,
                    correction=True,
                )
            else:
                # For small samples, use Fisher's exact test
                _, p_val = stats.fisher_exact(contingency)
        except Exception:
            p_val = np.nan

        rows.append(
            {
                demographic_col: demo_value,
                "population": int(demo_count),
                "population_pct": _safe_div(demo_count, universe, multiplier=100),
                "outcome_count": outcome_demo,
                "outcome_rate": _safe_div(outcome_demo, demo_count, multiplier=100),
                "p_value": float(p_val),
            }
        )

    # Step 6: Build result DataFrame
    result = pd.DataFrame(rows)

    if result.empty:
        return pd.DataFrame(
            columns=[
                demographic_col,
                "population",
                "population_pct",
                "outcome_count",
                "outcome_rate",
                "disparity_index",
                "p_value",
                "potential_improvement",
                "sig_marker"
            ]
        )

    # Step 7: Add disparity index relative to best performer
    if "return" in str(_outcome_metric).lower():
        # For returns, lowest rate is best (0% is perfect)
        best_rate = result["outcome_rate"].min()
        worst_rate = result["outcome_rate"].max()
        
        # Calculate disparity index
        if worst_rate == 0:
            # SPECIAL CASE: All groups have 0% returns - perfect performance!
            result["disparity_index"] = 1.0
            # Add a flag to indicate this special case
            result["all_groups_perfect"] = True
        elif best_rate == 0:
            # Some groups have 0% (perfect), others don't
            # Groups with 0% get DI = 1.0, others get scaled down
            result["disparity_index"] = result["outcome_rate"].apply(
                lambda x: 1.0 if x == 0 else max(0, 1.0 - (x / worst_rate))
            )
            result["all_groups_perfect"] = False
        elif worst_rate == best_rate:
            # All groups have the same non-zero rate
            result["disparity_index"] = 1.0
            result["all_groups_perfect"] = False
        else:
            # Normal case: best_rate > 0, variation exists
            # DI = best_rate / group_rate (lower rates get higher DI)
            result["disparity_index"] = result["outcome_rate"].apply(
                lambda x: min(1.0, best_rate / x) if x > 0 else 1.0
            )
            result["all_groups_perfect"] = False
    else:
        # For other metrics (like PH exits), highest rate is best
        best_rate = result["outcome_rate"].max()
        worst_rate = result["outcome_rate"].min()
        
        if best_rate == 100:
            # SPECIAL CASE: All groups have 100% success rate
            result["disparity_index"] = 1.0
            result["all_groups_perfect"] = (worst_rate == 100)
        elif best_rate == worst_rate:
            # All groups have the same rate
            result["disparity_index"] = 1.0
            result["all_groups_perfect"] = False
        elif best_rate == 0:
            # No one succeeded
            result["disparity_index"] = 0.0
            result["all_groups_perfect"] = False
        else:
            # DI = group_rate / best_rate
            result["disparity_index"] = result["outcome_rate"].apply(
                lambda x: min(1.0, x / best_rate)
            )
            result["all_groups_perfect"] = False
    
    # Step 8: Calculate potential impact
    if "return" in str(_outcome_metric).lower():
        # For returns, calculate reduction in returns if all groups had lowest rate
        min_group = result.loc[result["outcome_rate"].idxmin()]
        min_rate = min_group["outcome_rate"]
        
        # If best rate is 0, only groups with non-zero rates have potential improvement
        result["potential_improvement"] = result.apply(
            lambda row: -int(np.floor((row["outcome_rate"]/100) * row["population"])) 
            if row["outcome_rate"] > min_rate else 0,
            axis=1
        )
    else:
        # For PH exits, calculate increase if all groups had highest rate
        max_group = result.loc[result["outcome_rate"].idxmax()]
        max_rate = max_group["outcome_rate"]
        result["potential_improvement"] = np.floor(
            (max_rate/100 - result["outcome_rate"]/100) * result["population"]
        ).clip(lower=0).astype(int)
    
    # Add significance markers
    result["sig_marker"] = ""
    for level, marker in zip([0.001, 0.01, 0.05], ["***", "**", "*"]):
        result.loc[result["p_value"] <= level, "sig_marker"] = marker

    # Add renamed group column for consistency
    result["group"] = result[demographic_col]

    return result

def ph_exit_pop_filter(df: DataFrame, s: Timestamp, e: Timestamp) -> Set[int]:
    """Filter function to get clients who exited during the period."""
    mask = (df["ProjectExit"] >= s) & (df["ProjectExit"] <= e)
    return set(df.loc[mask, "ClientID"])
    
def returns_pop_filter(df: DataFrame, s: Timestamp, e: Timestamp) -> Set[int]:
    """Filter function to get clients who exited to PH during the period."""
    mask = (
        (df["ProjectExit"] >= s) 
        & (df["ProjectExit"] <= e)
        & (df["ExitDestinationCat"] == "Permanent Housing Situations")
    )
    return set(df.loc[mask, "ClientID"])

@st.fragment
def render_equity_analysis(df_filt: DataFrame, full_df: Optional[DataFrame] = None) -> None:
    """
    Render the equity analysis section with enhanced visualizations.
    
    Parameters:
    -----------
    df_filt : DataFrame
        Filtered DataFrame
    full_df : DataFrame, optional
        Full DataFrame for returns analysis
    """
    # Initialize section state
    state = init_section_state(EQUITY_SECTION_KEY)
    
    # Fallback for full_df
    if full_df is None:
        full_df = st.session_state.get("df")
        if full_df is None:
            st.error("Original dataset not found. Equity analysis requires full data.")
            return

    # Check if cache is valid
    filter_timestamp = get_filter_timestamp()
    cache_valid = is_cache_valid(state, filter_timestamp)
    
    if not cache_valid:
        invalidate_cache(state, filter_timestamp)
        # Clear cached data
        for k in list(state.keys()):
            if k not in ["last_updated"]:
                state.pop(k, None)

    # Header with help button
    col_header, col_info = st.columns([6, 1])
    with col_header:
        st.subheader("⚖️ Equity Analysis")
    with col_info:
        with st.popover("ℹ️ Help", use_container_width=True):
            st.markdown("""
            ### Understanding Equity Analysis
            
            This section uses statistical methods to identify significant outcome disparities across demographic groups, helping ensure equitable service delivery.
            
            **Available Analyses:**
            
            **1. Permanent Housing Exits**
            - **Population**: All unique clients who exited during the period
            - **Outcome**: Exited to permanent housing destination
            - **Rate**: (PH exits ÷ Total exits) × 100
            - **Goal**: Higher rates are better (more successful placements)
            
            **2. Returns to Homelessness**
            - **Population**: All unique clients who exited to PH during the period
            - **Outcome**: Returned to homelessness within tracking window (7-1095 days)
            - **Rate**: (Returns ÷ PH exits) × 100
            - **Goal**: Lower rates are better (fewer returns)
            - **Note**: Uses HUD-compliant logic with exclusions for short PH stays
            
            **Statistical Methods:**
            - **Chi-square test**: Used when all cells have n ≥ 5 (with Yates' correction)
            - **Fisher's exact test**: Used for smaller samples
            - **Significance levels**: * p<0.05, ** p<0.01, *** p<0.001
            - **Minimum group size**: Filters out groups below threshold (default 30)
            
            **Disparity Index (DI) Calculation:**
            
            For PH Exits (higher is better):
            - DI = Group Rate ÷ Highest Rate
            - Best group gets DI = 1.0
            
            For Returns (lower is better):
            - DI = Lowest Rate ÷ Group Rate (when lowest > 0)
            - Special handling when best group has 0% returns
            - Groups with 0% returns get DI = 1.0
            
            **Disparity Categories:**
            - **Minimal** (DI: 0.95-1.0): Within 5% of best performer ✅
            - **Moderate** (DI: 0.80-0.94): 5-20% gap from best 🔵
            - **Significant** (DI: 0.50-0.79): 20-50% gap from best 🟠
            - **Severe** (DI: <0.50): Over 50% gap from best 🔴
            
            **Visual Components:**
            
            **1. Outcome Rates Chart**
            - Bar chart showing rates for each group
            - Sorted from best to worst performance
            - Color-coded by disparity level
            - Shows statistical significance markers
            - Includes population percentage as secondary metric
            
            **2. Disparity Index Chart**
            - Horizontal bars showing DI values
            - Reference line at 1.0 (parity)
            - Color-coded by severity level
            - Shows gap from best performer
            
            **3. Key Findings Summary**
            - Best and worst performing groups
            - Performance gap in percentage points
            - Potential system impact if all groups matched best
            - Groups with statistically significant disparities
            
            **Interpretation Guide:**
            - **Focus on significant disparities**: Look for * markers and DI < 0.8
            - **Consider group size**: Larger groups provide more reliable estimates
            - **Check both metrics**: High PH exits + low returns = best outcomes
            - **Potential improvement**: Shows additional successes if all matched best group
            
            **Filter Options:**
            - **Demographic dimension**: Choose which characteristic to analyze
            - **Outcome measure**: PH exits or returns to homelessness
            - **Minimum group size**: Ensure statistical reliability (10-5000)
            - **Return tracking window**: Days to monitor returns (7-1095)
            - **Project types**: Include/exclude specific program types
            - **Demographic groups**: Select specific groups to compare
            
            **Important Considerations:**
            - **Correlation vs Causation**: Disparities show associations, not causes
            - **Unmeasured factors**: Many variables may influence outcomes
            - **Intersectionality**: Single dimensions don't capture full complexity
            - **Sample size matters**: Small groups may show extreme results
            - **Returns are system-wide**: Tracked across all programs regardless of filters
            
            **Using Results for Action:**
            1. Identify groups with severe disparities (DI < 0.5)
            2. Review if disparities are statistically significant
            3. Consider group size and representation
            4. Investigate potential causes (barriers, service gaps)
            5. Design targeted interventions
            6. Monitor progress over time
            """)
    
    # Introduction box
    intro_html = f"""
    <div style="background-color: rgba(33,102,172,0.1); border: 1px solid {MAIN_COLOR}; 
                border-radius: 10px; padding: 15px; margin-bottom: 20px;">
        <p style="margin: 0;">
            <strong>Purpose:</strong> This analysis helps ensure all populations receive equitable services and outcomes. 
            It identifies groups that may face additional barriers or need targeted interventions.
        </p>
    </div>
    """
    st.html(intro_html)

    # Control panel - first row
    c1, c2, c3 = st.columns(3)
    
    # Dimension selection
    equity_label = c1.selectbox(
        "Compare by",
        [lbl for lbl, _ in DEMOGRAPHIC_DIMENSIONS],
        key=f"equity_dim_{filter_timestamp}",
        help="Select which demographic characteristic to analyze"
    )
    dim_col = dict(DEMOGRAPHIC_DIMENSIONS)[equity_label]

    # Outcome selection
    outcome_label = c2.selectbox(
        "Outcome to measure",
        ["Permanent housing exits", "Returns to homelessness"],
        key=f"equity_outcome_{filter_timestamp}",
        help="Select which outcome to analyze"
    )
    
    # Minimum group size
    min_pop = c3.number_input(
        "Minimum group size",
        min_value=10,
        max_value=5000,
        value=30,
        step=10,
        key=f"equity_min_pop_{filter_timestamp}",
        help="Groups smaller than this won't be shown (ensures statistical reliability)"
    )

    # Group filter
    try:
        unique_groups = df_filt[dim_col].dropna().unique().tolist()
        unique_groups = [g for g in unique_groups if str(g) not in ["", "nan", "NaN", "None", "none", "null", "NA"]]
        unique_groups.sort()
        
        subdimension_selected = st.multiselect(
            f"Select {equity_label} groups to include",
            options=unique_groups,
            default=unique_groups,
            key=f"equity_subdim_{filter_timestamp}",
            help=f"Choose which {equity_label} groups to analyze"
        )
    except Exception as e:
        st.error(f"Error loading groups: {e}")
        return

    # Additional options based on outcome
    if outcome_label == "Returns to homelessness":
        st.markdown("##### Return Analysis Options")
        d1, d2 = st.columns(2)
        
        return_window = d1.number_input(
            "Days to track returns",
            min_value=7,
            max_value=1095,
            value=180,
            step=30,
            key=f"equity_return_window_{filter_timestamp}",
            help="How many days after exit to check for returns"
        )
        
        all_types = sorted(df_filt["ProjectTypeCode"].dropna().unique())
        proj_selected = d2.multiselect(
            "Project types to analyze",
            options=all_types,
            default=all_types,
            key=f"equity_proj_types_return_{filter_timestamp}",
            help="Which project types to include"
        )
    else:  # PH exits
        all_types = sorted(df_filt["ProjectTypeCode"].dropna().unique())
        proj_selected = st.multiselect(
            "Project types to analyze",
            options=all_types,
            default=all_types,
            key=f"equity_proj_types_ph_{filter_timestamp}",
            help="Which project types to include"
        )
        return_window = 730  # Not used for PH exits

    # Check date range
    t0 = st.session_state.get("t0")
    t1 = st.session_state.get("t1")
    
    if not (t0 and t1):
        st.warning("Please set your date range in the filter panel.")
        return

    # Apply filters
    if not subdimension_selected:
        st.warning(f"Please select at least one {equity_label} group.")
        return
    
    df_subset = df_filt[df_filt[dim_col].astype(str).isin([str(s) for s in subdimension_selected])]
    
    if df_subset.empty:
        st.warning(f"No data found for the selected {equity_label} groups.")
        return

    # Apply project type filter
    if not proj_selected:
        st.warning("Please select at least one project type.")
        return
    
    df_subset = df_subset[df_subset["ProjectTypeCode"].isin(proj_selected)]
    
    if df_subset.empty:
        st.warning("No data available for the selected project types.")
        return

    # Import necessary functions based on outcome type
    if outcome_label == "Permanent housing exits":
        from analysis.general.data_utils import ph_exit_clients
        outcome_func = ph_exit_clients
        pop_filter_fn = ph_exit_pop_filter
        outcome_name = "Permanent Housing Exits"
    else:  # Returns to homelessness
        from analysis.general.data_utils import return_after_exit
        # Create a wrapper function that includes the return_window parameter
        outcome_func = lambda df_sub, s, e: return_after_exit(df_sub, full_df, s, e, return_window)
        pop_filter_fn = returns_pop_filter
        outcome_name = f"Returns Within {return_window} Days"

    # Create cache key
    key = (
        f"{equity_label}|{outcome_name}|min{min_pop}|"
        f"{t0.date()}–{t1.date()}|"
        + (f"rw{return_window}|" if "Returns" in outcome_label else "")
        + f"subdim:{','.join(sorted(map(str, subdimension_selected)))}|"
        + ",".join(sorted(map(str, proj_selected)))
    )

    # Check if we need to run the analysis
    if state.get("cache_key") != key or "equity_data" not in state:
        with st.spinner("Analyzing equity..."):
            try:
                # Run equity analysis
                df_disp = equity_analysis(
                    df_subset,
                    dim_col,
                    outcome_func,
                    t0,
                    t1,
                    min_population=min_pop,
                    _population_filter=pop_filter_fn,
                    restrict_groups=subdimension_selected,
                    return_window=return_window,
                    full_df=full_df
                )
                
                if df_disp.empty:
                    st.info("No groups meet the minimum size threshold.")
                    return

                # Don't re-sort here - keep the disparity index as calculated
                # The equity_analysis function already handles the calculations correctly
                
                # Categorize disparity levels
                df_disp["disparity_magnitude"] = pd.cut(
                    df_disp["disparity_index"],
                    bins=[0, 0.5, 0.8, 0.95, 1.01],
                    labels=["Severe", "Significant", "Moderate", "None"]
                )
                
                # Cache results
                state["equity_data"] = df_disp
                state["outcome_name"] = outcome_name
                state["cache_key"] = key

            except Exception as e:
                st.error(f"Failed to run equity analysis: {e}")
                return
    else:
        # Use cached results
        df_disp = state["equity_data"]
        outcome_name = state.get("outcome_name", outcome_name)

    # Check if we have results
    if df_disp.empty:
        st.info("No groups meet the minimum size threshold.")
        return

    blue_divider()

    # Display clear interpretation guide
    is_returns = "Returns" in outcome_name
    if is_returns:
        interpretation_html = f"""
        <div style="background-color: rgba(33,102,172,0.1); border: 1px solid {MAIN_COLOR}; 
                    border-radius: 8px; padding: 12px; margin-bottom: 20px;">
            <p style="margin: 0;"><strong>📊 For returns to homelessness: Lower rates are BETTER (fewer people returning)</strong></p>
        </div>
        """
    else:
        interpretation_html = f"""
        <div style="background-color: rgba(33,102,172,0.1); border: 1px solid {MAIN_COLOR}; 
                    border-radius: 8px; padding: 12px; margin-bottom: 20px;">
            <p style="margin: 0;"><strong>📊 For housing exits: Higher rates are BETTER (more people housed)</strong></p>
        </div>
        """
    st.html(interpretation_html)

    # Create tabs for different views
    tab_overview, tab_disparity, tab_details = st.tabs([
        "📊 Outcome Rates", "📈 Disparity Analysis", "📋 Data & Methodology"
    ])

    with tab_overview:
        # Create outcome rate chart
        st.markdown(f"### {outcome_name} by {equity_label}")
        
        # Add sorting explanation
        if is_returns:
            sort_explanation_html = f"""
            <div style="background-color: rgba(75,181,67,0.1); border: 1px solid {SUCCESS_COLOR}; 
                        border-radius: 8px; padding: 10px; margin-bottom: 15px;">
                <p style="margin: 0; font-size: 14px;">
                    📊 <strong>Sorted from best to worst:</strong> Lowest return rates (best) appear first
                </p>
            </div>
            """
        else:
            sort_explanation_html = f"""
            <div style="background-color: rgba(75,181,67,0.1); border: 1px solid {SUCCESS_COLOR}; 
                        border-radius: 8px; padding: 10px; margin-bottom: 15px;">
                <p style="margin: 0; font-size: 14px;">
                    📊 <strong>Sorted from best to worst:</strong> Highest exit rates (best) appear first
                </p>
            </div>
            """
        st.html(sort_explanation_html)
        
        # Prepare chart data - ensure proper sorting and DI values
        if is_returns:
            # For returns, sort ascending (0% first, then higher rates)
            chart_df = df_disp.sort_values("outcome_rate", ascending=True).copy()
            
            # Recalculate DI to ensure correctness after any data manipulation
            min_rate = chart_df["outcome_rate"].min()
            max_rate = chart_df["outcome_rate"].max()
            
            if max_rate == min_rate:
                chart_df["disparity_index"] = 1.0
            elif min_rate == 0:
                # Groups with 0% get DI = 1.0, others scaled down
                chart_df["disparity_index"] = chart_df["outcome_rate"].apply(
                    lambda x: 1.0 - (x / max_rate) if max_rate > 0 else 1.0
                )
            else:
                # Normal case
                chart_df["disparity_index"] = chart_df["outcome_rate"].apply(
                    lambda x: min(1.0, min_rate / x) if x > 0 else 1.0
                )
        else:
            # For PH exits, sort descending (higher is better, best at top)
            chart_df = df_disp.sort_values("outcome_rate", ascending=False).copy()
        
        # Calculate dynamic height
        num_groups = len(chart_df)
        chart_height = _calculate_chart_height(num_groups)
        
        # Create the outcome rate bar chart
        fig = go.Figure()
        
        # Determine best and worst rates based on outcome type
        if is_returns:
            # For returns: lower is better
            best_rate = chart_df["outcome_rate"].min()
            worst_rate = chart_df["outcome_rate"].max()
        else:
            # For PH exits: higher is better
            best_rate = chart_df["outcome_rate"].max()
            worst_rate = chart_df["outcome_rate"].min()
        
        # Color bars based on disparity index
        bar_colors = []
        for idx, row in chart_df.iterrows():
            di_val = row["disparity_index"]
            color = _get_disparity_color(di_val)
            bar_colors.append(color)
        
        # Add bars
        fig.add_bar(
            x=chart_df['group'],
            y=chart_df["outcome_rate"],
            name="Outcome rate (%)",
            text=[f"{x:.1f}%" + (f" {s}" if s else "") for x, s in zip(chart_df["outcome_rate"], chart_df["sig_marker"])],
            textposition="outside",
            textfont=dict(color="white", size=12),
            marker_color=bar_colors,
            hoverinfo="text",
            hovertext=[f"{group}: {rate:.1f}%<br>Population: {pop:,} ({pop_pct:.1f}%)<br>Significance: {sig if sig else 'None'}" 
                      for group, rate, pop, pop_pct, sig in zip(
                          chart_df['group'], 
                          chart_df["outcome_rate"], 
                          chart_df["population"],
                          chart_df["population_pct"],
                          chart_df["sig_marker"]
                      )]
        )
        
        # Add population as a secondary element
        fig.add_scatter(
            x=chart_df['group'],
            y=chart_df["population_pct"],
            name="% of Population",
            yaxis="y2",
            mode="markers",
            marker=dict(
                size=10,
                symbol="circle",
                color=NEUTRAL_COLOR,
                line=dict(color="white", width=1)
            ),
            opacity=0.7
        )
        
        # Add reference line for system average
        avg_rate = chart_df["outcome_rate"].mean()
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(chart_df) - 0.5,
            y0=avg_rate,
            y1=avg_rate,
            line=dict(color="white", width=2, dash="dash"),
        )
        
        # Apply consistent styling
        fig.update_layout(
            yaxis=dict(
                title="Outcome Rate (%)",
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=False,
                range=[0, max(chart_df["outcome_rate"].max() * 1.15, 10)]  # Add 15% padding
            ),
            yaxis2=dict(
                title="% of Population",
                overlaying="y",
                side="right",
                range=[0, min(100, chart_df["population_pct"].max() * 1.5)],
                gridcolor='rgba(255,255,255,0.05)',
                zeroline=False
            ),
            template=PLOT_TEMPLATE,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1
            ),
            margin=dict(t=80, b=100, l=80, r=80),  # Generous margins
            xaxis=dict(
                tickangle=-45 if len(chart_df) > 6 else 0,
                title="",
                gridcolor='rgba(255,255,255,0.05)',
                automargin=True
            ),
            height=chart_height,
            bargap=0.2
        )
        
        # Add annotations outside the plot area
        fig.add_annotation(
            text=f"System Average: {avg_rate:.1f}%",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=12, color="white"),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white",
            borderwidth=1,
            borderpad=4
        )
        
        # Add direction indicator for clarity
        if is_returns:
            fig.add_annotation(
                text="← Better (Lower rates)",
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                showarrow=False,
                font=dict(size=12, color=SUCCESS_COLOR, weight="bold"),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor=SUCCESS_COLOR,
                borderwidth=1,
                borderpad=4,
                xanchor="right"
            )
        else:
            fig.add_annotation(
                text="Better (Higher rates) →",
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                showarrow=False,
                font=dict(size=12, color=SUCCESS_COLOR, weight="bold"),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor=SUCCESS_COLOR,
                borderwidth=1,
                borderpad=4,
                xanchor="right"
            )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Add color explanation with clear legend
        color_guide_html = f"""
        <div style="background-color: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px; margin: 10px 0;">
            <p style="margin: 0 0 10px 0; font-weight: bold;">Bar Color = Disparity Level:</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 30px; height: 20px; background-color: {SUCCESS_COLOR}; border-radius: 3px; border: 1px solid white;"></div>
                    <span><strong>Green</strong> = Minimal (DI ≥ 0.95)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 30px; height: 20px; background-color: {MAIN_COLOR}; border-radius: 3px; border: 1px solid white;"></div>
                    <span><strong>Blue</strong> = Moderate (DI 0.80-0.94)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 30px; height: 20px; background-color: {WARNING_COLOR}; border-radius: 3px; border: 1px solid white;"></div>
                    <span><strong>Orange</strong> = Significant (DI 0.50-0.79)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 30px; height: 20px; background-color: {DANGER_COLOR}; border-radius: 3px; border: 1px solid white;"></div>
                    <span><strong>Red</strong> = Severe (DI < 0.50)</span>
                </div>
            </div>
            <p style="margin: 10px 0 0 0; font-size: 13px; font-style: italic;">
                DI = Disparity Index (1.0 means equal to best group, lower values mean larger gaps)
            </p>
        </div>
        """
        st.html(color_guide_html)

    with tab_disparity:
        st.markdown("### Disparity Index Analysis")
        
        # Explanation of disparity index
        di_explanation_html = f"""
        <div style="background-color: rgba(0,0,0,0.2); border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 14px;">
                <strong>What is the Disparity Index?</strong> It shows how far each group is from the best performer.
                <br>• <strong>1.0</strong> = Equal to best group
                <br>• <strong>0.8</strong> = 20% gap from best group  
                <br>• <strong>0.5</strong> = 50% gap from best group
                <br>• <strong>Lower values</strong> = Larger disparities
            </p>
        </div>
        """
        st.html(di_explanation_html)
        
        # Sort by disparity index for this view (worst to best)
        gap_df = chart_df.sort_values("disparity_index", ascending=True).copy()
        
        # Calculate height
        num_groups_di = len(gap_df)
        di_chart_height = max(450, min(700, 400 + num_groups_di * 40))
        
        # Create horizontal bar chart for disparity index
        di_fig = go.Figure()
        
        # Add each bar individually with its specific color based on DI value
        for idx, row in gap_df.iterrows():
            di_val = row["disparity_index"]
            color = _get_disparity_color(di_val)
            
            # Add individual bar
            di_fig.add_trace(go.Bar(
                x=[di_val],
                y=[row['group']],
                orientation='h',
                marker=dict(
                    color=color,
                    line=dict(width=2, color="white")
                ),
                text=f"{di_val:.2f}" + (f" {row['sig_marker']}" if row['sig_marker'] else ""),
                textposition="outside",
                textfont=dict(color="white", size=12, weight="bold"),
                hoverinfo="text",
                hovertext=f"{row['group']}: {di_val:.2f} disparity index<br>" +
                         f"Rate: {row['outcome_rate']:.1f}%<br>" +
                         f"Gap from best: {((1-di_val)*100):.0f}%<br>" +
                         (f"p-value: {row['p_value']:.3f}" if row['sig_marker'] else "Not significant"),
                showlegend=False,
                name=row['group']
            ))
        
        # Add reference line for parity
        di_fig.add_shape(
            type="line",
            x0=1,
            x1=1,
            y0=-0.5,
            y1=len(gap_df) - 0.5,
            line=dict(color="white", width=3, dash="dash"),
        )
        
        # Add annotation for parity line
        di_fig.add_annotation(
            x=1,
            y=len(gap_df),
            text="Parity Line",
            showarrow=False,
            yshift=15,
            font=dict(size=12, color="white", weight="bold"),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white",
            borderwidth=1,
            borderpad=4
        )
        
        # Apply styling with better spacing
        di_fig.update_layout(
            xaxis=dict(
                title=dict(
                    text="Disparity Index (1.0 = Equal to Best Group)",
                    font=dict(size=14)
                ),
                range=[0, 1.15],  # Fixed range for clarity
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=False,
                tickformat=".1f",
                tickmode="array",
                tickvals=[0, 0.25, 0.5, 0.75, 0.8, 0.95, 1.0],
                ticktext=["0", "0.25", "0.5<br><span style='font-size:10px'>Severe</span>", 
                         "0.75", "0.8<br><span style='font-size:10px'>Moderate</span>", 
                         "0.95", "1.0<br><span style='font-size:10px'>Parity</span>"],
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title="",
                autorange="reversed",
                automargin=True,
                tickfont=dict(size=12)
            ),
            margin=dict(l=20, r=100, t=80, b=120),  # More bottom margin for labels
            template=PLOT_TEMPLATE,
            height=di_chart_height,
            bargap=0.35,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Display the chart
        st.plotly_chart(di_fig, use_container_width=True)
        
        # Color legend for disparity levels
        disparity_legend_html = f"""
        <div style="background-color: rgba(0,0,0,0.2); border-radius: 8px; padding: 15px; margin-top: 20px;">
            <p style="margin: 0 0 10px 0; font-weight: bold;">Disparity Level Categories:</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 20px; height: 20px; background-color: {SUCCESS_COLOR}; border-radius: 3px;"></div>
                    <span><strong>Minimal</strong> (0.95-1.0): Within 5% of best</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 20px; height: 20px; background-color: {MAIN_COLOR}; border-radius: 3px;"></div>
                    <span><strong>Moderate</strong> (0.8-0.95): 5-20% gap</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 20px; height: 20px; background-color: {WARNING_COLOR}; border-radius: 3px;"></div>
                    <span><strong>Significant</strong> (0.5-0.8): 20-50% gap</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 20px; height: 20px; background-color: {DANGER_COLOR}; border-radius: 3px;"></div>
                    <span><strong>Severe</strong> (<0.5): Over 50% gap</span>
                </div>
            </div>
        </div>
        """
        st.html(disparity_legend_html)

        blue_divider()

        # Key findings section
        st.markdown("### Key Findings")
        
        # For returns: Find groups with min and max rates
        if is_returns:
            # Best = lowest return rate
            best_idx = chart_df["outcome_rate"].idxmin()
            best = chart_df.loc[best_idx]
            
            # Worst = highest return rate
            worst_idx = chart_df["outcome_rate"].idxmax()
            worst = chart_df.loc[worst_idx]
            
            # Gap is always positive (worst - best)
            gap = worst["outcome_rate"] - best["outcome_rate"]
        else:
            # For PH exits: best = highest rate
            best_idx = chart_df["outcome_rate"].idxmax()
            best = chart_df.loc[best_idx]
            
            # Worst = lowest rate
            worst_idx = chart_df["outcome_rate"].idxmin()
            worst = chart_df.loc[worst_idx]
            
            # Gap is always positive (best - worst)
            gap = best["outcome_rate"] - worst["outcome_rate"]
        
        # Calculate system-wide impact
        total_improvement = chart_df["potential_improvement"].sum()
        
        # Create findings HTML
        findings_html = _create_disparity_summary_html(
            best, worst, gap, total_improvement, is_returns, outcome_name
        )
        st.html(findings_html)
        
        # Significant disparities section
        sig_disparities = chart_df[(chart_df["p_value"] < 0.05) & (chart_df["disparity_index"] < 0.8)]
        if not sig_disparities.empty:
            sig_html = f"""
            <div style="background-color: rgba(255,99,71,0.1); border: 1px solid {SECONDARY_COLOR}; 
                        border-radius: 10px; padding: 20px; margin: 20px 0;">
                <h4 style="color: {SECONDARY_COLOR}; margin: 0 0 15px 0;">
                    ⚠️ Groups with Significant Disparities
                </h4>
            """
            
            for _, row in sig_disparities.iterrows():
                gap_pct = (1 - row['disparity_index']) * 100
                
                # Color code based on disparity level
                level_text = _get_disparity_level(row['disparity_index'])
                border_color = _get_disparity_color(row['disparity_index'])
                
                sig_html += f"""
                <div style="border-left: 4px solid {border_color}; padding-left: 15px; margin-bottom: 15px; 
                            background-color: rgba(255,255,255,0.05); padding: 10px 15px; border-radius: 5px;">
                    <h5 style="margin: 0 0 8px 0; color: {border_color};">{row['group']} - {level_text.upper()} DISPARITY</h5>
                    <p style="margin: 0; font-size: 14px;">
                        • Rate: <strong>{fmt_pct(row['outcome_rate'])}</strong><br>
                        • Disparity Index: <strong>{row['disparity_index']:.2f}</strong> ({gap_pct:.0f}% gap from best)<br>
                        • Statistical significance: <strong>{row['sig_marker']}</strong> (p={row['p_value']:.3f})<br>
                        • Affects <strong>{fmt_int(row['population'])}</strong> people
                    </p>
                </div>
                """
            
            sig_html += "</div>"
            st.html(sig_html)
        else:
            no_sig_html = f"""
            <div style="background-color: rgba(75,181,67,0.1); border: 1px solid {SUCCESS_COLOR}; 
                        border-radius: 10px; padding: 20px; margin: 20px 0;">
                <h4 style="color: {SUCCESS_COLOR}; margin: 0;">
                    ✅ No Severe Disparities Detected
                </h4>
                <p style="margin: 10px 0 0 0;">
                    No groups show statistically significant disparities greater than 20% 
                    from the best performing group.
                </p>
            </div>
            """
            st.html(no_sig_html)

    with tab_details:
        # Data table
        st.markdown("### Detailed Data Export")
        
        display_df = chart_df[[
            dim_col, "population", "population_pct", "outcome_count", "outcome_rate",
            "disparity_index", "p_value", "sig_marker", "potential_improvement"
        ]].copy()
        
        display_df.columns = [
            "Group", "Population", "% of Total", "Outcome Count", "Rate (%)",
            "Disparity Index", "p-value", "Significance", "Potential Impact"
        ]

        # Display formatted table
        st.dataframe(
            display_df.style
            .format({
                "Population": "{:,}",
                "% of Total": "{:.1f}%",
                "Outcome Count": "{:,}",
                "Rate (%)": "{:.1f}%",
                "Disparity Index": "{:.2f}",
                "p-value": "{:.3f}",
                "Potential Impact": "{:+,}"
            })
            ,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"equity_analysis_{equity_label}_{outcome_name}_{t0.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        blue_divider()
        
        # Methodology section
        st.markdown("### Methodology & Technical Details")
        
        methodology_html = _create_methodology_html(
            equity_label, outcome_name, t0, t1, min_pop, 
            return_window if "Returns" in outcome_name else None
        )
        
        st.html(methodology_html)