"""
Equity analysis section for HMIS dashboard.
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
    SUCCESS_COLOR, WARNING_COLOR, apply_chart_style, fmt_int, fmt_pct
)

# Constants
EQUITY_SECTION_KEY = "equity_analysis"

def _safe_div(a: float, b: float, default: float = 0.0, multiplier: float = 1.0) -> float:
    """Safe division with optional multiplier."""
    return round((a / b) * multiplier if b else default, 1)

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
        population = df.loc[pop_mask].copy()  # Create a copy here to avoid the warning
    else:
        # For custom populations like short-stay or returns, use the provided filter
        pop_ids = _population_filter(df, start, end)
        population = df[df["ClientID"].isin(pop_ids)].copy()  # Create a copy here to avoid the warning

    # Convert any non-string values to strings in the demographic column
    if demographic_col in population.columns:
        # Fix: Use .loc to modify the DataFrame
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

    # Step 7: Add disparity index relative to top group
    if "return" in str(_outcome_metric).lower():
        # For returns, lowest rate is best, so use minimum as reference
        min_rate = result["outcome_rate"].min()
        if min_rate > 0:  # Avoid division by zero
            result["disparity_index"] = result["outcome_rate"].apply(
                lambda x: _safe_div(min_rate, x, default=0.0) if x > 0 else 0.0
            )
        else:
            result["disparity_index"] = 1.0
    else:
        # For other metrics (like PH exits), highest rate is best
        max_rate = result["outcome_rate"].max()
        result["disparity_index"] = result["outcome_rate"].apply(
            lambda x: _safe_div(x, max_rate, default=0.0) if max_rate > 0 else 0.0
        )
    
    # Step 8: Calculate potential impact
    if "return" in str(_outcome_metric).lower():
        # For returns, calculate reduction in returns if all groups had lowest rate
        min_group = result.loc[result["outcome_rate"].idxmin()]
        min_rate = min_group["outcome_rate"]
        result["potential_improvement"] = np.floor(
            (result["outcome_rate"]/100 - min_rate/100) * result["population"]
        ).clip(lower=0).astype(int) * -1  # Negative = reduction in returns
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
        full_df = df_filt.copy()

    # Check if cache is valid
    filter_timestamp = get_filter_timestamp()
    cache_valid = is_cache_valid(state, filter_timestamp)
    
    if not cache_valid:
        invalidate_cache(state, filter_timestamp)
        # Clear cached data
        for k in list(state.keys()):
            if k not in ["last_updated"]:
                state.pop(k, None)

    # Header and description
    st.subheader("⚖️ Equity Analysis")
    st.markdown("""
    Identify and address disparities in outcomes across demographic groups. This analysis helps:
    
    - **Detect significant inequities** that may require targeted interventions
    - **Measure the equity impact** of program design and resource allocation 
    - **Track progress** toward reducing disparities over time
    """)

    # Control panel - first row
    c1, c2, c3 = st.columns(3)
    
    # Dimension selection
    equity_label = c1.selectbox(
        "Dimension",
        [lbl for lbl, _ in DEMOGRAPHIC_DIMENSIONS],
        key=f"equity_dim_{filter_timestamp}",
        help="Select the demographic characteristic to analyze for disparities"
    )
    dim_col = dict(DEMOGRAPHIC_DIMENSIONS)[equity_label]

    # Outcome selection
    outcome_label = c2.selectbox(
        "Outcome",
        ["Permanent-housing exits", "Return to homelessness after PH exit"],
        key=f"equity_outcome_{filter_timestamp}",
        help="Select which outcome to analyze for disparities across groups"
    )
    
    # Minimum group size
    min_pop = c3.number_input(
        "Min group size",
        min_value=10,
        max_value=5000,
        value=30,
        step=10,
        key=f"equity_min_pop_{filter_timestamp}",
        help="Minimum sample size required for each group to be included in analysis"
    )

    # Group filter
    try:
        unique_groups = df_filt[dim_col].dropna().unique().tolist()
        unique_groups = [g for g in unique_groups if str(g) not in ["", "nan", "NaN", "None", "none", "null", "NA"]]
        unique_groups.sort()
        
        subdimension_selected = st.multiselect(
            f"Filter {equity_label} groups to include",
            options=unique_groups,
            default=unique_groups,
            key=f"equity_subdim_{filter_timestamp}",
            help=f"Only include these {equity_label} groups. Default is all."
        )
    except Exception as e:
        st.error(f"Error loading groups: {e}")
        return

    # Additional options based on outcome
    if outcome_label == "Return to homelessness after PH exit":
        st.markdown("##### Additional Options for Returns Analysis")
        d1, d2 = st.columns(2)
        
        return_window = d1.number_input(
            "Return window (days)",
            min_value=7,
            max_value=1095,
            value=180,
            step=30,
            key=f"equity_return_window_{filter_timestamp}",
            help="Number of days after PH exit to check for returns to homelessness"
        )
        
        all_types = sorted(df_filt["ProjectTypeCode"].dropna().unique())
        proj_selected = d2.multiselect(
            "Project Types (to INCLUDE)",
            options=all_types,
            default=all_types,
            key=f"equity_proj_types_return_{filter_timestamp}",
            help="Which ProjectTypeCodes to check for PH exits (returns will be tracked in ALL programs)"
        )
        
        short_thresh = 90  # Not used for returns analysis
    else:  # PH exits
        all_types = sorted(df_filt["ProjectTypeCode"].dropna().unique())
        proj_selected = st.multiselect(
            "Project Types (to INCLUDE)",
            options=all_types,
            default=all_types,
            key=f"equity_proj_types_ph_{filter_timestamp}",
            help="Which ProjectTypeCodes to include in PH exit analysis."
        )
        short_thresh = 90  # Not used for PH exits
        return_window = 730  # Not used for PH exits

    # Check date range
    t0 = st.session_state.get("t0")
    t1 = st.session_state.get("t1")
    
    if not (t0 and t1):
        st.warning("Please set your date range in the sidebar first.")
        return

    # Apply group filter
    if not subdimension_selected:
        st.warning(f"No {equity_label} groups selected. Please select at least one group.")
        return
    
    df_subset = df_filt[df_filt[dim_col].astype(str).isin([str(s) for s in subdimension_selected])]
    
    if df_subset.empty:
        st.warning(f"No data found for the selected {equity_label} group(s). Please check your selections.")
        return

    # Apply project type filter
    if not proj_selected:
        st.warning("No project types selected. Please select at least one project type.")
        return
    
    df_subset = df_subset[df_subset["ProjectTypeCode"].isin(proj_selected)]
    
    if df_subset.empty:
        st.warning("No data available for the selected project types. Please adjust your selection.")
        return

    # Import necessary functions based on outcome type
    if outcome_label == "Permanent-housing exits":
        from analysis.general.data_utils import ph_exit_clients
        outcome_func = ph_exit_clients
        pop_filter_fn = ph_exit_pop_filter
        outcome_name = "Permanent-housing exits"
    else:  # Return to homelessness after PH exit
        from analysis.general.data_utils import return_after_exit
        # Create a wrapper function that includes the return_window parameter
        outcome_func = lambda df_sub, s, e: return_after_exit(df_sub, full_df, s, e, return_window)
        pop_filter_fn = returns_pop_filter
        outcome_name = f"Returns within {return_window} days of PH exit"

    # Create cache key
    key = (
        f"{equity_label}|{outcome_name}|min{min_pop}|"
        f"{t0.date()}–{t1.date()}|"
        + (f"rw{return_window}|" if "Return to homelessness" in outcome_label else "")
        + f"subdim:{','.join(sorted(map(str, subdimension_selected)))}|"
        + ",".join(sorted(map(str, proj_selected)))
    )

    # Check if we need to run the analysis
    if state.get("cache_key") != key or "equity_data" not in state:
        with st.spinner("Computing disparities…"):
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

                # Sort based on outcome type
                if outcome_label == "Return to homelessness after PH exit":
                    # For returns, sort by rate ascending (lower is better)
                    df_disp = df_disp.sort_values("outcome_rate", ascending=True)
                else:
                    # For PH exits, sort by rate descending (higher is better)
                    df_disp = df_disp.sort_values("outcome_rate", ascending=False)
                
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

    # Display interpretation guide
    if outcome_label == "Permanent-housing exits":
        st.info("**Interpretation guide:** Higher rates indicate better outcomes (more exits to permanent housing).")
        better_direction = "higher"
        use_inverse_color = False
    else:  # Return to homelessness after PH exit
        st.info("**Interpretation guide:** Lower rates indicate better outcomes (fewer returns to homelessness).")
        better_direction = "lower"
        use_inverse_color = True

    # Create outcome rate chart
    st.subheader(f"{outcome_name} by {equity_label}")
    
    # Special handling for returns - sort and color differently
    if outcome_label == "Return to homelessness after PH exit":
        # For returns, sort ascending (lower is better)
        chart_df = df_disp.sort_values("outcome_rate", ascending=True).copy()
        
        # Recalculate disparity index for visualization if needed
        if "disparity_index" in chart_df.columns:
            # Make sure we're showing disparity correctly for returns
            min_rate = chart_df["outcome_rate"].min()
            if min_rate > 0:  # Avoid division by zero
                chart_df["disparity_index_returns"] = min_rate / chart_df["outcome_rate"]
                # This will be 1.0 for the best group (lowest return rate) and lower for worse groups
                chart_df["disparity_index"] = chart_df["disparity_index_returns"]
    else:
        # For PH exits, sort descending (higher is better)
        chart_df = df_disp.sort_values("outcome_rate", ascending=False).copy()
    
    # Create the outcome rate bar chart
    fig = go.Figure()
    
    # Determine bar colors based on outcome type
    if outcome_label == "Return to homelessness after PH exit":
        # For returns, lower values are better (blue) and higher values are worse (red)
        bar_colors = [
            MAIN_COLOR if i == 0 or r == chart_df["outcome_rate"].min() else SECONDARY_COLOR
            for i, r in enumerate(chart_df["outcome_rate"])
        ]
    else:
        # For PH exits, higher values are better (blue) and lower values are worse (red)
        bar_colors = [
            MAIN_COLOR if i == 0 or r == chart_df["outcome_rate"].max() else SECONDARY_COLOR
            for i, r in enumerate(chart_df["outcome_rate"])
        ]
    
    # Add bar chart with improved formatting
    fig.add_bar(
        x=chart_df[dim_col],
        y=chart_df["outcome_rate"],
        name="Outcome rate (%)",
        text=[f"{x:.1f}%" + (f" {s}" if s else "") for x, s in zip(chart_df["outcome_rate"], chart_df["sig_marker"])],
        textposition="outside",
        marker_color=bar_colors,
        hoverinfo="text",
        hovertext=[f"{group}: {rate:.1f}%<br>Population: {pop:,} ({pop_pct:.1f}%)<br>Significance: {sig}" 
                  for group, rate, pop, pop_pct, sig in zip(
                      chart_df[dim_col], 
                      chart_df["outcome_rate"], 
                      chart_df["population"],
                      chart_df["population_pct"],
                      chart_df["sig_marker"] if chart_df["sig_marker"].astype(bool).any() else ["None"] * len(chart_df)
                  )]
    )
    
    # Add population as a secondary element
    fig.add_scatter(
        x=chart_df[dim_col],
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
        line=dict(color="white", width=1.5, dash="dash"),
    )
    
    # Add annotation for system average
    fig.add_annotation(
        x=len(chart_df) / 2,
        y=avg_rate,
        text=f"System average: {avg_rate:.1f}%",
        showarrow=False,
        yshift=10,
        font=dict(size=12, color="white"),
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="white",
        borderwidth=1,
        borderpad=4,
        opacity=0.8
    )
    
    # Add significance explanation if needed
    significance_note = "* p<0.05   ** p<0.01   *** p<0.001" if any(chart_df["sig_marker"].astype(bool)) else ""
    
    # Add directional note
    better_worse_note = "Higher values (blue) are better" if better_direction == "higher" else "Lower values (blue) are better"
    
    # Create title with notes
    title_with_note = f"{outcome_name} by {equity_label}"
    if significance_note:
        title_with_note += f"<br><span style='font-size:12px'>{significance_note}</span>"
    if better_worse_note:
        title_with_note += f"<br><span style='font-size:12px'>{better_worse_note}</span>"
    
    # Apply consistent styling
    fig.update_layout(
        title=dict(
            text=title_with_note,
            font=dict(size=16)
        ),
        yaxis=dict(
            title="Outcome rate (%)",
            gridcolor='rgba(233,233,233,0.3)',
            zeroline=False
        ),
        yaxis2=dict(
            title="Group % of population",
            overlaying="y",
            side="right",
            range=[0, min(100, chart_df["population_pct"].max() * 1.5)],
            gridcolor='rgba(233,233,233,0.1)',
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
        margin=dict(t=100, b=80, l=50, r=50),
        xaxis=dict(
            tickangle=-45,
            title="",
            gridcolor='rgba(233,233,233,0.1)'
        ),
        height=500 if len(chart_df) > 6 else 400
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Create Equity Gap Analysis visualization
    st.subheader("Equity Gap Analysis")
    
    # Special handling for returns vs PH exits
    if outcome_label == "Return to homelessness after PH exit":
        # For returns, the lowest rate is the best performer
        gap_df = chart_df.copy()
    else:
        # For PH exits, sort by disparity_index (already calculated correctly)
        gap_df = chart_df.sort_values("disparity_index").copy()
    
    # Create custom hover text
    hover_text = [
        f"{group}: {di:.2f} disparity index<br>" +
        f"Outcome rate: {rate:.1f}% ({(di*100):.1f}% of {('lowest' if outcome_label == 'Return to homelessness after PH exit' else 'top')} group)<br>" +
        f"Population: {pop:,} ({pop_pct:.1f}%)<br>" +
        (f"Significance: p={pval:.3f} {sig}" if sig else "Not statistically significant")
        for group, di, rate, pop, pop_pct, pval, sig in zip(
            gap_df[dim_col],
            gap_df["disparity_index"],
            gap_df["outcome_rate"],
            gap_df["population"],
            gap_df["population_pct"],
            gap_df["p_value"],
            gap_df["sig_marker"]
        )
    ]
    
    # Define color scale based on outcome type
    if use_inverse_color:
        # For returns (Lower is better)
        color_scale = [[0, "#2171b5"], [0.5, "#9ecae1"], [1, "#f4a582"]]  # Blue to light blue to salmon
    else:
        # For PH exits (Higher is better)
        color_scale = [[0, "#f4a582"], [0.5, "#9ecae1"], [1, "#2171b5"]]  # Salmon to light blue to blue
    
    # Create horizontal bar chart
    di_fig = go.Figure()
    
    # Add bars with better hover information
    di_fig.add_bar(
        x=gap_df["disparity_index"],
        y=gap_df[dim_col],
        orientation='h',
        marker=dict(
            color=gap_df["disparity_index"],
            colorscale=color_scale,
            colorbar=dict(
                title="Disparity<br>Index",
                tickvals=[0, 0.5, 0.8, 1],
                ticktext=["0<br>Severe", "0.5<br>Significant", "0.8<br>Moderate", "1.0<br>None"],
                lenmode="fraction",
                len=0.8
            ),
            line=dict(width=1, color="white")
        ),
        text=[f"{di:.2f}" + (f" {m}" if m else "") for di, m in zip(gap_df["disparity_index"], gap_df["sig_marker"])],
        textposition="auto",
        hoverinfo="text",
        hovertext=hover_text
    )
    
    # Add reference line for parity
    di_fig.add_shape(
        type="line",
        x0=1,
        x1=1,
        y0=-0.5,
        y1=len(gap_df) - 0.5,
        line=dict(color="white", width=1.5, dash="dash"),
    )
    
    # Add colored background zones
    di_fig.add_shape(
        type="rect", 
        x0=0, 
        x1=0.5, 
        y0=-0.5, 
        y1=len(gap_df) - 0.5,
        fillcolor="rgba(244,165,130,0.15)", 
        line=dict(width=0), 
        layer="below"
    )
    di_fig.add_shape(
        type="rect", 
        x0=0.5, 
        x1=0.8, 
        y0=-0.5, 
        y1=len(gap_df) - 0.5,
        fillcolor="rgba(186,186,186,0.15)", 
        line=dict(width=0), 
        layer="below"
    )
    di_fig.add_shape(
        type="rect", 
        x0=0.8, 
        x1=0.95, 
        y0=-0.5, 
        y1=len(gap_df) - 0.5,
        fillcolor="rgba(158,202,225,0.15)", 
        line=dict(width=0), 
        layer="below"
    )
    
    # Add zone labels
    for label, x_pos, y_shift in [("Severe<br>disparity", 0.25, 0), 
                                ("Significant<br>disparity", 0.65, 0),
                                ("Moderate<br>disparity", 0.88, 0),
                                ("Parity", 1.03, 0)]:
        di_fig.add_annotation(
            x=x_pos,
            y=len(gap_df) - 0.5,
            yshift=y_shift,
            text=label,
            showarrow=False,
            font=dict(size=10, color="white"),
            align="center",
            bordercolor="white",
            borderwidth=1,
            borderpad=3,
            bgcolor="rgba(0,0,0,0.5)",
            opacity=0.7
        )
    
    # Apply consistent styling
    di_fig.update_layout(
        title=dict(
            text="Disparity Index: Comparison to Highest Performing Group",
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Disparity Index (1.0 = parity with top group)",
            range=[0, max(1.1, df_disp["disparity_index"].max() * 1.05)],
            gridcolor='rgba(233,233,233,0.3)',
            zeroline=False
        ),
        yaxis=dict(
            title="",
            autorange="reversed"  # This puts the highest disparity at the bottom
        ),
        margin=dict(l=10, r=10, t=60, b=50),
        template=PLOT_TEMPLATE,
        height=500 if len(gap_df) > 6 else 400
    )
    
    # Display the chart
    st.plotly_chart(di_fig, use_container_width=True)

    # Add explanation of the disparity index
    with st.expander("Understanding the Disparity Index"):
        if outcome_label == "Return to homelessness after PH exit":
            st.markdown(f"""
            The **Disparity Index** for returns to homelessness measures how each group's return rate compares to the group with the lowest return rate:
            
            - **1.0** = Equal to the group with lowest returns (parity)
            - **0.8** = Return rate is 1.25x higher than the best group (moderate disparity)  
            - **0.5** = Return rate is 2x higher than the best group (significant disparity)
            - **< 0.5** = Return rate is more than 2x higher than the best group (severe disparity)
            
            Statistical significance is indicated by asterisks:
            - * = p < 0.05 (significant)
            - ** = p < 0.01 (highly significant)
            - *** = p < 0.001 (extremely significant)
            
            Where p-value represents the probability that the observed disparity occurred by chance.
            """)
        else:
            st.markdown(f"""
            The **Disparity Index** measures how each group's outcome rate compares to the highest performing group:
            
            - **1.0** = Equal to the top group (parity)
            - **0.8** = 80% of the top group's rate (moderate disparity)  
            - **0.5** = 50% of the top group's rate (significant disparity)
            - **< 0.5** = Less than half of the top group's rate (severe disparity)
            
            Statistical significance is indicated by asterisks:
            - * = p < 0.05 (significant)
            - ** = p < 0.01 (highly significant)
            - *** = p < 0.001 (extremely significant)
            
            Where p-value represents the probability that the observed disparity occurred by chance.
            """)

    # Data table with formatting
    st.subheader("Detailed Results")
    display_df = chart_df[[
        dim_col, "population", "population_pct", "outcome_count", "outcome_rate",
        "disparity_index", "p_value", "sig_marker", "potential_improvement"
    ]].copy()
    
    display_df.columns = [
        "Group", "Population", "% of Total", "Outcome Count", "Outcome Rate",
        "Disparity Index", "p-value", "Significance", "Potential Improvement"
    ]

    # Create a custom colormap
    cmap_name = "PuOr" if use_inverse_color else "BuGn"

    # Display the table with formatting
    st.dataframe(
        display_df.style
        .format({
            "Population": "{:,}",
            "% of Total": "{:.1f}%",
            "Outcome Count": "{:,}",
            "Outcome Rate": "{:.1f}%",
            "Disparity Index": "{:.2f}",
            "p-value": "{:.3f}",
            "Potential Improvement": "{:+,}"  # Add plus sign for better readability
        })
        .background_gradient(
            subset=["Disparity Index"],
            cmap=cmap_name,
            vmin=0,
            vmax=1
        )
        # Highlight statistically significant values
        .apply(lambda x: ['background-color: rgba(255,255,0,0.2)' if v else '' 
                         for v in x == "*"], subset=["Significance"])
        .apply(lambda x: ['background-color: rgba(255,165,0,0.2)' if v else '' 
                         for v in x == "**"], subset=["Significance"])
        .apply(lambda x: ['background-color: rgba(255,0,0,0.2)' if v else '' 
                         for v in x == "***"], subset=["Significance"]),
        use_container_width=True
    )
    
    # Key findings visualization
    # Get top and bottom groups based on outcome type
    if outcome_label == "Return to homelessness after PH exit":
        # For returns, lowest rate is best (sorted ascending above)
        top = chart_df.iloc[0]  # Lowest return rate (best performer)
        bot = chart_df.iloc[-1]  # Highest return rate (worst performer)
        gap = bot["outcome_rate"] - top["outcome_rate"]  # Gap is high minus low
    else:
        # For PH exits, highest rate is best (sorted descending above)
        top = chart_df.iloc[0]  # Highest PH exit rate (best performer)
        bot = chart_df.iloc[-1]  # Lowest PH exit rate (worst performer)
        gap = top["outcome_rate"] - bot["outcome_rate"]  # Gap is high minus low
    
    # Calculate system-wide impact
    total_improvement = chart_df["potential_improvement"].sum()
    
    # Adjust impact description based on outcome type
    if outcome_label == "Permanent-housing exits":
        impact_desc = f"If all groups achieved the same PH exit rate as the {top[dim_col]} group"
    else:  # Return to homelessness after PH exit
        impact_desc = f"If all groups had the same low return rate as the {top[dim_col]} group"
    
    # Create key findings section
    st.markdown("### Key Findings")
    
    # Create three columns for key metrics
    k1, k2, k3 = st.columns(3)
    
    # Adjust labels based on outcome type
    highest_label = "Highest Rate" if outcome_label == "Permanent-housing exits" else "Lowest Rate (Best)"
    lowest_label = "Lowest Rate" if outcome_label == "Permanent-housing exits" else "Highest Rate (Worst)"
    
    with k1:
        st.markdown(f"""
        <div style="background-color:rgba(73,160,181,0.2); padding:10px; border-radius:5px; text-align:center;">
        <h4>{highest_label}</h4>
        <h2>{top[dim_col]}</h2>
        <h3>{fmt_pct(top["outcome_rate"])}</h3>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown(f"""
        <div style="background-color:rgba(255,99,71,0.2); padding:10px; border-radius:5px; text-align:center;">
        <h4>{lowest_label}</h4>
        <h2>{bot[dim_col]}</h2>
        <h3>{fmt_pct(bot["outcome_rate"])}</h3>
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        st.markdown(f"""
        <div style="background-color:rgba(128,128,128,0.2); padding:10px; border-radius:5px; text-align:center;">
        <h4>Gap Between Groups</h4>
        <h2>{fmt_pct(gap)}</h2>
        <h3>percentage points</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Impact information in a highlighted box
    impact_sign = "" if total_improvement < 0 else "+"
    impact_color = "rgba(73,160,181,0.1)" if total_improvement >= 0 else "rgba(255,99,71,0.1)"
    
    st.markdown(f"""
    <div style="background-color:{impact_color}; padding:15px; border-radius:5px; margin-top:15px;">
    <h4>System-wide Impact Potential</h4>
    <p>{impact_desc}, approximately <strong style="font-size:1.2em;">{impact_sign}{total_improvement:,}</strong> additional clients would achieve the positive outcome.</p>
    </div>
    """, unsafe_allow_html=True)

    # Display significant disparities and recommendations
    sig = chart_df[(chart_df["p_value"] < 0.05) & (chart_df["disparity_index"] < 0.8)]
    if not sig.empty:
        # Sort by disparity index to find the most disparate groups
        sig_sorted = sig.sort_values("disparity_index")
        
        st.markdown("""
        <div style="background-color:rgba(255,99,71,0.1); padding:15px; border-radius:5px; margin-top:20px;">
        <h3>⚠️ Significant Disparities Detected</h3>
        """, unsafe_allow_html=True)
        
        # Create a list of all significant disparities with better formatting
        for idx, row in sig_sorted.iterrows():
            st.markdown(f"""
            <div style="border-left: 3px solid #ff6347; padding-left: 15px; margin-bottom: 15px;">
            <h4>{row[dim_col]}</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>📊 Population: <strong>{fmt_int(row['population'])}</strong> people</li>
                <li>📈 Outcome rate: <strong>{fmt_pct(row['outcome_rate'])}</strong> ({fmt_pct(row['disparity_index'] * 100)} of top group)</li>
                <li>🔍 Statistical significance: p = <strong>{row['p_value']:.3f}</strong> {row['sig_marker']}</li>
                <li>💡 Potential impact: <strong>{row['potential_improvement']:,}</strong> additional clients with improved outcomes</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Add recommendations section
        st.markdown("""
        <h3>🎯 Action Recommendations</h3>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div style="background-color:rgba(0,204,0,0.1); padding:20px; border-radius:5px; margin-top:20px;">
        <h3>✅ No Statistically Significant Disparities Detected</h3>
        <p>No statistically significant disparities were found in this analysis (p < 0.05). This suggests that the system is producing relatively equitable outcomes across the analyzed groups for this specific measure.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add methodology notes
    with st.expander("Methodology Notes"):
        st.markdown(f"""
        ### Analysis Parameters
        | Parameter | Value |
        | --- | --- |
        | Demographic dimension | {equity_label} |
        | Outcome measure | {outcome_name} |
        | Date range | {t0.date()} to {t1.date()} |
        | Minimum group size | {min_pop} |
        
        ### Statistical Methods
        - Chi-square test with Yates' correction used for groups with sufficient data
        - Fisher's exact test used for smaller groups
        - p < 0.05 threshold for statistical significance
        - Groups smaller than the minimum size threshold were excluded
        
        ### Outcome Definitions
        """)
        
        if outcome_label == "Permanent-housing exits":
            st.markdown("""
            **Permanent housing exits** are defined as:
            - Clients who exited to a permanent housing destination during the reporting period
            - Base population includes all clients who exited programs during the reporting period
            """)
        else:  # Return to homelessness after PH exit
            st.markdown(f"""
            **Returns to homelessness** are defined as:
            - Clients who exited to permanent housing and then returned to a homeless program within {return_window} days
            - Returns do not include enrollments in prevention, coordinated entry, or services-only projects
            - When a client moved directly into permanent housing with a move-in date matching their project start date, this is not counted as a return
            """)
        
        st.markdown(f"""
        ### Limitations
        - Disparities indicate correlation, not necessarily causation
        - Multiple factors may influence outcomes beyond the analyzed demographic dimension
        - This analysis only examines one outcome; comprehensive equity assessment requires multiple measures
        {"- Returns analysis requires both exit and re-entry data, which may be incomplete" if outcome_label == "Return to homelessness after PH exit" else ""}
        """)