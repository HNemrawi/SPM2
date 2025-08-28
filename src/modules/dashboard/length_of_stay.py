"""
Length of stay analysis section for HMIS dashboard - Enhanced Version
Provides enrollment-level analysis with proper project type context and HUD alignment
"""

from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas import DataFrame, Timestamp

from src.core.data.destinations import apply_custom_ph_destinations
from src.modules.dashboard.data_utils import DEMOGRAPHIC_DIMENSIONS
from src.modules.dashboard.filters import (
    get_filter_timestamp,
    hash_data,
    init_section_state,
    invalidate_cache,
    is_cache_valid,
)
from src.ui.factories.components import ui
from src.ui.factories.html import html_factory
from src.ui.themes.theme import (
    DANGER_COLOR,
    MAIN_COLOR,
    NEUTRAL_COLOR,
    PLOT_TEMPLATE,
    SECONDARY_COLOR,
    SUCCESS_COLOR,
    WARNING_COLOR,
    blue_divider,
    theme,
)

# Constants
LOS_SECTION_KEY = "length_of_stay"

# Project type benchmarks based on HUD standards
PROJECT_TYPE_BENCHMARKS = {
    "Emergency Shelter": {
        "typical_range": (1, 60),
        "target": 30,
        "interpretation": "Crisis stabilization",
        "long_stay_threshold": 90,
    },
    "Emergency Shelter – Entry Exit": {
        "typical_range": (1, 60),
        "target": 30,
        "interpretation": "Crisis stabilization",
        "long_stay_threshold": 90,
    },
    "Emergency Shelter – Night-by-Night": {
        "typical_range": (1, 60),
        "target": 30,
        "interpretation": "Crisis stabilization",
        "long_stay_threshold": 90,
    },
    "Transitional Housing": {
        "typical_range": (30, 730),
        "target": 180,
        "interpretation": "Transitional support",
        "long_stay_threshold": 730,
    },
    "PH – Rapid Re-Housing": {
        "typical_range": (30, 365),
        "target": 120,
        "interpretation": "Housing placement & stabilization",
        "long_stay_threshold": 365,
    },
    "PH – Permanent Supportive Housing (disability required for entry)": {
        "typical_range": (90, None),
        "target": None,
        "interpretation": "Permanent housing (no time limit)",
        "long_stay_threshold": None,
    },
    "PH – Housing Only": {
        "typical_range": (90, None),
        "target": None,
        "interpretation": "Permanent housing (no time limit)",
        "long_stay_threshold": None,
    },
    "PH – Housing with Services (no disability required for entry)": {
        "typical_range": (90, None),
        "target": None,
        "interpretation": "Permanent housing (no time limit)",
        "long_stay_threshold": None,
    },
    "Street Outreach": {
        "typical_range": (1, 180),
        "target": None,
        "interpretation": "Engagement period varies",
        "long_stay_threshold": 365,
    },
    "Services Only": {
        "typical_range": (1, 90),
        "target": None,
        "interpretation": "Service engagement",
        "long_stay_threshold": 180,
    },
    "Homelessness Prevention": {
        "typical_range": (1, 365),
        "target": 90,
        "interpretation": "Prevention services",
        "long_stay_threshold": 365,
    },
}

# Exit destination categories as they appear in the data
EXIT_CATEGORIES = {
    "Permanent Housing Situations": "positive",
    "Temporary Housing Situations": "temporary",
    "Institutional Situations": "institutional",
    "Homeless Situations": "negative",
    "Other": "other",
}

# LOS category thresholds with context
LOS_CATEGORIES = {
    "0–7 days": {"color": DANGER_COLOR, "desc": "Very short stay"},
    "8–30 days": {"color": WARNING_COLOR, "desc": "Short-term"},
    "31–90 days": {"color": NEUTRAL_COLOR, "desc": "Medium-term"},
    "91–180 days": {"color": MAIN_COLOR, "desc": "Extended"},
    "181–365 days": {"color": SECONDARY_COLOR, "desc": "Long-term"},
    "365+ days": {"color": SUCCESS_COLOR, "desc": "Very long-term"},
}


def _calculate_chart_height(num_groups: int, base_height: int = 400) -> int:
    """Calculate optimal chart height based on number of groups."""
    if num_groups <= 6:
        return base_height
    elif num_groups <= 12:
        return base_height + 100
    elif num_groups <= 20:
        return base_height + 200
    else:
        return min(base_height + 300, 800)


def _get_los_color(category: str) -> str:
    """Get color for LOS category."""
    return LOS_CATEGORIES.get(category, {}).get("color", MAIN_COLOR)


def _get_project_type_from_code(project_type_code: str) -> str:
    """Map project type code to readable name."""
    if pd.isna(project_type_code):
        return "Unknown"

    # Try to match the code directly
    for key in PROJECT_TYPE_BENCHMARKS.keys():
        if str(project_type_code) in key or key in str(project_type_code):
            return key

    return str(project_type_code)


def _get_los_interpretation(
    avg_los: float, median_los: float, project_type_mix: pd.Series
) -> Tuple[str, str, str]:
    """Get interpretation of LOS metrics based on project type context."""

    # Determine dominant project type
    if not project_type_mix.empty:
        # Map project types to readable names
        project_type_mix = project_type_mix.apply(_get_project_type_from_code)
        dominant_type = project_type_mix.value_counts().index[0]
        # benchmark = PROJECT_TYPE_BENCHMARKS.get(dominant_type, {})
    else:
        dominant_type = "Unknown"

    # Check if it's permanent housing
    is_permanent_housing = any(
        ph in dominant_type
        for ph in [
            "Permanent Supportive Housing",
            "Housing Only",
            "Housing with Services",
        ]
    )

    # For permanent housing, long stays are positive
    if is_permanent_housing:
        if avg_los < 90:
            severity = "Concerning"
            color = WARNING_COLOR
            desc = "Short stays in permanent housing may indicate housing instability"
        elif avg_los < 365:
            severity = "Developing Stability"
            color = NEUTRAL_COLOR
            desc = "Residents establishing housing stability"
        else:
            severity = "Stable Housing"
            color = SUCCESS_COLOR
            desc = "Long-term housing stability achieved"
    else:
        # For time-limited programs
        if avg_los < 7:
            severity = "Very Short Stays"
            color = DANGER_COLOR
            desc = "May indicate self-resolution or immediate exits"
        elif avg_los < 30:
            severity = "Short-term"
            color = WARNING_COLOR
            desc = "Typical for crisis response"
        elif avg_los < 90:
            severity = "Medium-term"
            color = NEUTRAL_COLOR
            desc = "Standard duration for transitional programs"
        elif avg_los < 180:
            severity = "Extended"
            color = MAIN_COLOR
            desc = "Longer stays may indicate service needs or barriers"
        else:
            severity = "Very Long Stays"
            color = SECONDARY_COLOR
            desc = "May indicate barriers to permanent housing placement"

    return severity, color, desc


def _create_los_summary_html(
    avg_los: float,
    median_los: float,
    q1: float,
    q3: float,
    total_enrollments: int,
    unique_clients: int,
    project_type_mix: pd.Series,
) -> str:
    """Create HTML summary for LOS metrics with enrollment context using UI factories."""
    severity, color, desc = _get_los_interpretation(
        avg_los, median_los, project_type_mix
    )

    # Calculate skewness indicator
    skewness = avg_los - median_los
    if abs(skewness) < 5:
        skew_text = "Balanced distribution"
        skew_color = NEUTRAL_COLOR
    elif skewness > 0:
        skew_text = "Right-skewed (some very long stays)"
        skew_color = WARNING_COLOR
    else:
        skew_text = "Left-skewed (mostly longer stays)"
        skew_color = MAIN_COLOR

    # Get dominant project type for context
    if not project_type_mix.empty:
        project_type_mix = project_type_mix.apply(_get_project_type_from_code)
        dominant_type = project_type_mix.value_counts().index[0]
    else:
        dominant_type = "Mixed"

    # Create metric cards
    metrics_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 1rem 0;'>"

    # Average Stay metric
    metrics_html += html_factory.metric_card(
        label="Average Stay",
        value=f"{avg_los:.0f} days",
        delta="Mean across all enrollments",
        color=color,
        icon="📊",
    )

    # Typical Stay metric
    metrics_html += html_factory.metric_card(
        label="Typical Stay",
        value=f"{median_los:.0f} days",
        delta="Middle value (median)",
        color=MAIN_COLOR,
        icon="📊",
    )

    # Common Range metric
    metrics_html += html_factory.metric_card(
        label="Common Range",
        value=f"{int(q1)}-{int(q3)} days",
        delta="Middle 50% of enrollments",
        color=MAIN_COLOR,
        icon="📊",
    )

    # Distribution metric
    metrics_html += html_factory.metric_card(
        label="Distribution",
        value=skew_text,
        delta=f"Based on {total_enrollments:,} enrollments",
        color=skew_color,
        icon="📊",
    )

    metrics_html += "</div>"

    # Context information
    context_info = f"""
    <p style='margin: 0 0 10px 0; font-size: 16px;'>{desc}</p>
    <p style='margin: 0 0 15px 0; font-size: 14px; color: {theme.colors.text_secondary}
        ;'>
        <strong>Dominant Project Type:</strong> {dominant_type} |
        <strong>Analysis Level:</strong> Enrollment-based ({total_enrollments:,} enrollments from {unique_clients:,} unique clients)
    </p>
    """

    # Main container with title
    title_html = html_factory.title(
        text=f"Length of Stay Profile: {severity}",
        level=2,
        color=color,
        icon="⏱️",
    )

    return title_html + html_factory.card(
        content=context_info + metrics_html, border_color=color
    )


def _create_demographic_insights_html(
    longest_stay: pd.Series,
    shortest_stay: pd.Series,
    disparity_ratio: float,
    dim_col: str,
) -> str:
    """Create HTML for demographic LOS insights using UI factories."""

    # Determine disparity severity
    if disparity_ratio < 1.5:
        disparity_level = "Low"
        disparity_color = SUCCESS_COLOR
    elif disparity_ratio < 2.0:
        disparity_level = "Moderate"
        disparity_color = NEUTRAL_COLOR
    elif disparity_ratio < 3.0:
        disparity_level = "High"
        disparity_color = WARNING_COLOR
    else:
        disparity_level = "Severe"
        disparity_color = DANGER_COLOR

    # Create metric cards for the key patterns
    metrics_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 1rem 0;'>"

    # Longest stays metric
    metrics_html += html_factory.metric_card(
        label="Longest Average Stays",
        value=longest_stay[dim_col],
        delta=f"{
            longest_stay['mean']:.0f} days • {
            longest_stay['count']:,} enrollments",
        color=SECONDARY_COLOR,
        icon="🔺",
    )

    # Shortest stays metric
    metrics_html += html_factory.metric_card(
        label="Shortest Average Stays",
        value=shortest_stay[dim_col],
        delta=f"{
            shortest_stay['mean']:.0f} days • {
            shortest_stay['count']:,} enrollments",
        color=SUCCESS_COLOR,
        icon="🔻",
    )

    # Disparity level metric
    metrics_html += html_factory.metric_card(
        label="Disparity Level",
        value=disparity_level,
        delta=f"{
            disparity_ratio:.1f}x difference between longest and shortest",
        color=disparity_color,
        icon="📊",
    )

    metrics_html += "</div>"

    # Gap analysis info box
    gap_analysis = f"""
    <strong>Gap Analysis:</strong> The difference of {(longest_stay['mean'] -
                                                       shortest_stay['mean']):.0f} days
    between groups suggests {"significant" if disparity_ratio >= 2.0 else "moderate"}
                                                                               variations in service patterns
    or client needs across {dim_col.lower()} groups.
    """

    gap_info = html_factory.info_box(
        content=gap_analysis, type="info", title="Analysis Summary", icon="🔍"
    )

    # Main title and container
    title_html = html_factory.title(
        text="Key Length of Stay Patterns",
        level=3,
        color=MAIN_COLOR,
        icon="📈",
    )

    return title_html + metrics_html + gap_info


def length_of_stay(
    df: DataFrame, start: Timestamp, end: Timestamp
) -> Dict[str, Any]:
    """
    Calculate length of stay statistics for all enrollments.

    Parameters:
    -----------
    df : DataFrame
        Filtered dataframe containing enrollment data
    start : Timestamp
        Start date of reporting period
    end : Timestamp
        End date of reporting period

    Returns:
    --------
    Dict containing:
        - los_by_enrollment: DataFrame with LOS for each enrollment
        - avg_los: Average length of stay
        - median_los: Median length of stay
        - distribution: DataFrame with counts by LOS category
        - data_quality_issues: Count of problematic records
        - quality_details: Detailed breakdown of issues
        - unique_clients: Number of unique clients
        - project_types: Series of project types in the data
        - quality_issues_df: DataFrame with the actual problematic records
    """
    # Validate required columns
    required_cols = ["EnrollmentID", "ProjectStart", "ProjectExit", "ClientID"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_cols)}"
        )

    # Filter active enrollments
    active = df[
        ((df["ProjectExit"] >= start) | df["ProjectExit"].isna())
        & (df["ProjectStart"] <= end)
    ].copy()

    # Handle empty dataframe
    if active.empty:
        empty_dist = pd.DataFrame(columns=["LOS_Category", "count"])
        empty_los = pd.DataFrame(columns=["EnrollmentID", "ClientID", "LOS"])
        return {
            "avg_los": 0.0,
            "median_los": 0.0,
            "distribution": empty_dist,
            "los_by_enrollment": empty_los,
            "data_quality_issues": 0,
            "quality_details": {},
            "unique_clients": 0,
            "project_types": pd.Series(dtype="object"),
            "quality_issues_df": pd.DataFrame(),
        }

    # Calculate LOS according to the specified rules
    # For exited enrollments: Exit Date - Entry Date + 1
    # For active enrollments: Report End Date - Entry Date + 1
    active["ExitDateUsed"] = active["ProjectExit"].fillna(end)
    active["LOS"] = (
        active["ExitDateUsed"] - active["ProjectStart"]
    ).dt.days + 1

    # Track data quality issues and create a dataframe for download
    quality_issues_list = []

    # Exit before start
    exit_before_start_mask = active["ProjectExit"].notna() & (
        active["ProjectExit"] < active["ProjectStart"]
    )
    if exit_before_start_mask.any():
        issues_df = active[exit_before_start_mask].copy()
        issues_df["Issue_Type"] = "Exit Date Before Entry Date"
        issues_df["Issue_Description"] = (
            "ProjectExit ("
            + issues_df["ProjectExit"].astype(str)
            + ") < ProjectStart ("
            + issues_df["ProjectStart"].astype(str)
            + ")"
        )
        quality_issues_list.append(issues_df)

    # Future entry dates
    future_entry_mask = active["ProjectStart"] > end
    if future_entry_mask.any():
        issues_df = active[future_entry_mask].copy()
        issues_df["Issue_Type"] = "Future Entry Date"
        issues_df["Issue_Description"] = (
            "ProjectStart ("
            + issues_df["ProjectStart"].astype(str)
            + ") > Report End Date ("
            + str(end)
            + ")"
        )
        quality_issues_list.append(issues_df)

    # Extremely long stays (5+ years) - but NOT for permanent housing
    # Get project type info if available
    if "ProjectTypeCode" in active.columns:
        # Map project types to readable names
        active["Project_Type_Name"] = active["ProjectTypeCode"].apply(
            _get_project_type_from_code
        )

        # Only flag as issue if NOT permanent housing
        is_not_permanent_housing = ~active["Project_Type_Name"].str.contains(
            "Permanent|Housing Only|Housing with Services", case=False, na=True
        )
        extremely_long_mask = (active["LOS"] > 1825) & is_not_permanent_housing
    else:
        # If no project type info, flag all 5+ year stays
        extremely_long_mask = active["LOS"] > 1825

    if extremely_long_mask.any():
        issues_df = active[extremely_long_mask].copy()
        issues_df["Issue_Type"] = (
            "Extremely Long Stay (5+ years) in Non-Permanent Housing"
        )
        issues_df["Issue_Description"] = (
            "Length of Stay = " + issues_df["LOS"].astype(str) + " days"
        )
        if "Project_Type_Name" in issues_df.columns:
            issues_df["Issue_Description"] += (
                " in " + issues_df["Project_Type_Name"]
            )
        quality_issues_list.append(issues_df)

    # Combine all quality issues
    if quality_issues_list:
        quality_issues_df = pd.concat(quality_issues_list, ignore_index=True)
        # Select relevant columns for export
        export_cols = [
            "EnrollmentID",
            "ClientID",
            "ProjectStart",
            "ProjectExit",
            "LOS",
            "Issue_Type",
            "Issue_Description",
        ]
        if "ProjectName" in quality_issues_df.columns:
            export_cols.insert(2, "ProjectName")
        if "ProjectTypeCode" in quality_issues_df.columns:
            export_cols.insert(3, "ProjectTypeCode")
        quality_issues_df = quality_issues_df[export_cols]
    else:
        quality_issues_df = pd.DataFrame()

    # Count data quality issues
    data_quality_issues = {
        "negative_los": (active["LOS"] < 0).sum(),
        "zero_los": (active["LOS"] == 0).sum(),
        "exit_before_start": exit_before_start_mask.sum(),
        "future_entry": future_entry_mask.sum(),
        "extremely_long_stays": (
            extremely_long_mask.sum()
            if "extremely_long_mask" in locals()
            else 0
        ),
    }

    # Handle same-day services (0 days -> 1 day)
    # This should not happen with our calculation method (always adding 1)
    # but we'll check anyway
    same_day_mask = active["LOS"] == 0
    if same_day_mask.any():
        active.loc[same_day_mask, "LOS"] = 1

    # Exclude problematic records (exit before start) from analysis
    if exit_before_start_mask.any():
        active = active[~exit_before_start_mask].copy()

    # Recalculate stats after cleaning
    if active.empty:
        return {
            "avg_los": 0.0,
            "median_los": 0.0,
            "distribution": pd.DataFrame(columns=["LOS_Category", "count"]),
            "los_by_enrollment": pd.DataFrame(
                columns=["EnrollmentID", "ClientID", "LOS"]
            ),
            "data_quality_issues": sum(data_quality_issues.values()),
            "quality_details": data_quality_issues,
            "unique_clients": 0,
            "project_types": pd.Series(dtype="object"),
            "quality_issues_df": quality_issues_df,
        }

    # Calculate statistics on clean data
    avg_los = round(active["LOS"].mean(), 1)
    median_los = round(active["LOS"].median(), 1)
    unique_clients = active["ClientID"].nunique()

    # Get project types if available
    project_types = pd.Series(dtype="object")
    if "ProjectTypeCode" in active.columns:
        project_types = active["ProjectTypeCode"]

    # Create LOS distribution categories
    bins = [0, 7, 30, 90, 180, 365, float("inf")]
    labels = [
        "0–7 days",
        "8–30 days",
        "31–90 days",
        "91–180 days",
        "181–365 days",
        "365+ days",
    ]

    # Count by category
    dist_df = (
        active["LOS"]
        .pipe(pd.cut, bins=bins, labels=labels, right=False)
        .value_counts(sort=False)
        .rename_axis("LOS_Category")
        .reset_index(name="count")
    )

    return {
        "los_by_enrollment": active[["EnrollmentID", "ClientID", "LOS"]],
        "avg_los": avg_los,
        "median_los": median_los,
        "distribution": dist_df,
        "data_quality_issues": sum(data_quality_issues.values()),
        "quality_details": data_quality_issues,
        "unique_clients": unique_clients,
        "project_types": project_types,
        "quality_issues_df": quality_issues_df,
    }


def analyze_los_with_destinations(df: DataFrame, los_data: Dict) -> DataFrame:
    """Combine LOS analysis with exit destinations for better insights."""
    if (
        "los_by_enrollment" not in los_data
        or los_data["los_by_enrollment"].empty
    ):
        return pd.DataFrame()

    # Apply custom PH destinations
    df = apply_custom_ph_destinations(df, force=True)

    # Get enrollments with LOS
    los_df = los_data["los_by_enrollment"].copy()

    # Merge with exit destination data
    if "ExitDestinationCat" in df.columns:
        exit_data = df[
            ["EnrollmentID", "ExitDestinationCat", "ProjectExit"]
        ].drop_duplicates()
        los_with_dest = los_df.merge(exit_data, on="EnrollmentID", how="left")

        # Only include exits (not active enrollments)
        exits_only = los_with_dest[los_with_dest["ProjectExit"].notna()]

        if not exits_only.empty:
            # Group by destination type
            dest_summary = (
                exits_only.groupby("ExitDestinationCat")["LOS"]
                .agg(mean_los="mean", median_los="median", count="count")
                .round(1)
            )

            # Sort by mean LOS
            dest_summary = dest_summary.sort_values(
                "mean_los", ascending=False
            )

            return dest_summary

    return pd.DataFrame()


def los_by_demographic(
    df: DataFrame,
    dim_col: str,
    t0: Timestamp,
    t1: Timestamp,
    min_group_size: int = 10,
) -> DataFrame:
    """
    Calculate length of stay statistics by demographic group.

    Parameters:
    -----------
    df : DataFrame
        Filtered dataframe
    dim_col : str
        Demographic dimension column
    t0 : Timestamp
        Start date
    t1 : Timestamp
        End date
    min_group_size : int
        Minimum group size to include

    Returns:
    --------
    DataFrame
        LOS statistics by demographic group
    """
    # Validate the column exists
    if dim_col not in df.columns:
        raise ValueError(f"Column '{dim_col}' not found in dataframe")

    # Get basic LOS calculation
    los_data = length_of_stay(df, t0, t1)

    # Get LOS by enrollment with demographic info
    los_df = los_data["los_by_enrollment"].copy()

    # Add demographic info
    demo_map = df[["EnrollmentID", dim_col]].copy()

    # Handle categorical columns properly
    if pd.api.types.is_categorical_dtype(demo_map[dim_col]):
        # Add "Not Reported" to categories if not present
        if "Not Reported" not in demo_map[dim_col].cat.categories:
            demo_map[dim_col] = demo_map[dim_col].cat.add_categories(
                ["Not Reported"]
            )

    # Handle missing values
    demo_map[dim_col] = demo_map[dim_col].fillna("Not Reported")

    # Merge with LOS data using EnrollmentID
    los_with_demo = los_df.merge(demo_map, on="EnrollmentID", how="left")

    # Handle categorical columns in merged data
    if pd.api.types.is_categorical_dtype(los_with_demo[dim_col]):
        if "Not Reported" not in los_with_demo[dim_col].cat.categories:
            los_with_demo[dim_col] = los_with_demo[dim_col].cat.add_categories(
                ["Not Reported"]
            )

    # Fill any remaining NAs
    los_with_demo[dim_col] = los_with_demo[dim_col].fillna("Not Reported")

    # Calculate summary statistics by group
    los_by_demo = (
        los_with_demo.groupby(dim_col, observed=True)["LOS"]
        .agg(
            mean=lambda x: x.mean() if len(x) > 0 else 0,
            median=lambda x: x.median() if len(x) > 0 else 0,
            count="count",
            q1=lambda x: x.quantile(0.25) if len(x) > 0 else 0,
            q3=lambda x: x.quantile(0.75) if len(x) > 0 else 0,
        )
        .reset_index()
    )

    # Filter to groups that meet minimum size
    los_by_demo = los_by_demo[los_by_demo["count"] >= min_group_size]

    # Calculate total enrollment days by group
    los_by_demo["total_days"] = los_by_demo["mean"] * los_by_demo["count"]

    # Calculate disparity index relative to overall average
    if los_data["avg_los"] > 0:
        los_by_demo["disparity"] = (
            los_by_demo["mean"] / los_data["avg_los"]
        ).round(2)
    else:
        los_by_demo["disparity"] = 1.0

    # Calculate IQR (interquartile range)
    los_by_demo["iqr"] = los_by_demo["q3"] - los_by_demo["q1"]

    # Sort by mean LOS (descending)
    los_by_demo = los_by_demo.sort_values("mean", ascending=False)

    return los_by_demo


def get_los_recommendations(
    los_data: Dict, project_types: pd.Series, df: DataFrame
) -> List[Dict]:
    """Generate recommendations based on LOS patterns and project types."""
    recommendations = []

    # Get distribution data
    dist_df = los_data.get("distribution", pd.DataFrame())
    if dist_df.empty:
        return recommendations

    # Calculate percentages
    total = dist_df["count"].sum()
    if total == 0:
        return recommendations

    very_short = (
        dist_df[dist_df["LOS_Category"] == "0–7 days"]["count"].sum()
        / total
        * 100
    )
    very_long = (
        dist_df[dist_df["LOS_Category"] == "365+ days"]["count"].sum()
        / total
        * 100
    )

    # Get dominant project type
    if not project_types.empty:
        project_types = project_types.apply(_get_project_type_from_code)
        type_counts = project_types.value_counts()
        if not type_counts.empty:
            dominant_type = type_counts.index[0]

            # Emergency Shelter specific recommendations
            if "Emergency Shelter" in dominant_type:
                if los_data["avg_los"] > 60:
                    recommendations.append(
                        {
                            "icon": "⏰",
                            "title": "Extended Emergency Shelter Stays",
                            "color": WARNING_COLOR,
                            "actions": [
                                "Review housing placement barriers for long-stay clients",
                                "Assess need for more transitional/permanent housing options",
                                "Implement rapid exit strategies and housing-focused case management",
                                "Consider whether some clients need higher level of care",
                            ],
                        }
                    )

                if very_short > 40:
                    recommendations.append(
                        {
                            "icon": "⚡",
                            "title": "High Rate of Very Short Stays",
                            "color": MAIN_COLOR,
                            "actions": [
                                "Analyze exit destinations for 0-7 day stays",
                                "Determine if exits are to permanent housing (positive) or returns to street",
                                "Consider implementing diversion programs at entry",
                                "Review intake procedures for appropriateness",
                            ],
                        }
                    )

            # Permanent Housing specific recommendations
            elif any(
                ph in dominant_type
                for ph in [
                    "Permanent Supportive Housing",
                    "Housing Only",
                    "Housing with Services",
                ]
            ):
                if los_data["avg_los"] < 180:
                    recommendations.append(
                        {
                            "icon": "⚠️",
                            "title": "Short Stays in Permanent Housing",
                            "color": DANGER_COLOR,
                            "actions": [
                                "Review reasons for exits from permanent housing",
                                "Assess adequacy of supportive services",
                                "Check for involuntary exits or lease violations",
                                "Evaluate housing quality and client-housing match",
                            ],
                        }
                    )

            # Rapid Re-Housing specific recommendations
            elif "Rapid Re-Housing" in dominant_type:
                if los_data["avg_los"] > 365:
                    recommendations.append(
                        {
                            "icon": "📅",
                            "title": "Extended RRH Assistance Periods",
                            "color": WARNING_COLOR,
                            "actions": [
                                "Review if clients need permanent supportive housing instead",
                                "Assess local housing market conditions affecting exits",
                                "Evaluate progressive engagement and step-down strategies",
                                "Consider time limits and extensions policy",
                            ],
                        }
                    )

    # General recommendations based on patterns
    if very_short > 30:
        recommendations.append(
            {
                "icon": "⚡",
                "title": "High Short-Stay Rate",
                "color": WARNING_COLOR,
                "actions": [
                    "Review exit destinations for 0-7 day stays",
                    "Assess if rapid exits indicate self-resolution or premature exits",
                    "Consider diversion programs if appropriate",
                    "Analyze by project type to identify specific program issues",
                ],
            }
        )

    if very_long > 20:
        recommendations.append(
            {
                "icon": "⏰",
                "title": "Significant Long-Stay Population",
                "color": SECONDARY_COLOR,
                "actions": [
                    "Review housing barriers for long-stay clients",
                    "Assess need for permanent supportive housing",
                    "Implement progressive engagement strategies",
                    "Consider move-on strategies for stable PSH residents",
                ],
            }
        )

    if los_data["avg_los"] > los_data["median_los"] * 1.5:
        recommendations.append(
            {
                "icon": "📊",
                "title": "Highly Skewed Distribution",
                "color": MAIN_COLOR,
                "actions": [
                    "Identify outlier cases pulling up the average",
                    "Consider different interventions for different stay lengths",
                    "Review if program types match client needs",
                    "Implement targeted interventions for long-stayers",
                ],
            }
        )

    # Add exit destination analysis if available
    if "ExitDestinationCat" in df.columns:
        dest_summary = analyze_los_with_destinations(df, los_data)
        if (
            not dest_summary.empty
            and "Homeless Situations" in dest_summary.index
        ):
            homeless_exits_los = dest_summary.loc[
                "Homeless Situations", "mean_los"
            ]
            if homeless_exits_los > 90:
                recommendations.append(
                    {
                        "icon": "🚨",
                        "title": "Long Stays Ending in Homelessness",
                        "color": DANGER_COLOR,
                        "actions": [
                            "Review why long-term clients are exiting to homelessness",
                            "Assess effectiveness of housing placement efforts",
                            "Consider different interventions before program exit",
                            "Implement exit planning protocols earlier in stay",
                        ],
                    }
                )

    return recommendations


@st.fragment
def render_length_of_stay(df_filt: DataFrame) -> None:
    """
    Render the length of stay analysis section with enhanced visualizations.

    Parameters:
    -----------
    df_filt : DataFrame
        Filtered DataFrame
    """
    # Initialize section state
    state = init_section_state(LOS_SECTION_KEY)

    # Check if cache is valid
    filter_timestamp = get_filter_timestamp()
    cache_valid = is_cache_valid(state, filter_timestamp)

    if not cache_valid:
        invalidate_cache(state, filter_timestamp)
        # Clear cached data
        for k in list(state.keys()):
            if k not in ["last_updated"]:
                state.pop(k, None)

    # Header with info button using UI factory
    st.html(html_factory.title("Length of Stay Analysis", level=2, icon="⏱️"))

    col_help = st.columns([11, 1])[1]
    with col_help:
        with st.popover("ℹ️ Help", width='stretch'):
            st.markdown(
                """
            ### Understanding Length of Stay

            **What this measures:**
            - How long clients remain enrolled in programs
            - Time from entry to exit (or report end date if still enrolled)
            - Patterns across different groups and project types

            **Analysis Level:**
            - This analysis is at the ENROLLMENT level
            - A client with multiple enrollments is counted separately for each
            - All enrollment days use entry/exit dates (not bed nights)

            **Calculation Method:**
            - **For Exited Enrollments:** Exit Date - Entry Date + 1
            - **For Active Enrollments:** Report End Date - Entry Date + 1
            - **Same-Day Services:** Count as 1 day (0 days → 1 day)
            - **Data Quality Checks:**
               - Exclude records where exit date < entry date
               - Flag stays over 5 years as potential data issues
               - Identify future entry dates

            **Why it matters:**
            - **Short stays** may indicate:
              - Rapid housing success (positive if exiting to PH)
              - Self-resolution
              - Program exits without housing (negative if returning to homelessness)
            - **Long stays** may indicate:
              - Housing stability (positive in permanent housing)
              - Housing barriers (concerning in emergency programs)
              - Need for different intervention level

            **Project Type Context:**
            Length of stay has different meanings by project type

            **Key Metrics:**
            - **Average (Mean)**: Total days ÷ number of enrollments
            - **Median**: Middle value when sorted
            - **Quartiles**: 25th and 75th percentile ranges
            - **Distribution**: Breakdown by stay categories
            """
            )

    # Introduction box using UI factory
    ui.info_section(
        content="<strong>Purpose:</strong> Analyze how long clients stay in programs to understand service patterns, identify potential bottlenecks, and ensure appropriate service matching based on project type.",
        type="info",
        expanded=True,
    )

    # Get time boundaries
    t0 = st.session_state.get("t0")
    t1 = st.session_state.get("t1")

    if not (t0 and t1):
        st.warning("Please set date ranges in the filter panel.")
        return

    # Calculate LOS data if needed
    if "los_data" not in state:
        with st.spinner("Calculating length of stay statistics..."):
            try:
                los_data = length_of_stay(df_filt, t0, t1)

                # Validate los_data structure
                if not isinstance(los_data, dict):
                    st.error(
                        "LOS calculation produced invalid data structure."
                    )
                    return

                required_keys = [
                    "los_by_enrollment",
                    "avg_los",
                    "median_los",
                    "distribution",
                    "unique_clients",
                    "project_types",
                ]
                if not all(key in los_data for key in required_keys):
                    st.error(
                        "LOS calculation missing required data components."
                    )
                    return

                # Ensure critical dataframes aren't empty
                if los_data["los_by_enrollment"].empty:
                    st.info(
                        "No enrollment data available for length of stay analysis."
                    )
                    return

                state["los_data"] = los_data
            except Exception as e:
                st.error(f"Error calculating LOS: {str(e)}")
                return
    else:
        los_data = state["los_data"]

    # Check for data quality issues
    if (
        "data_quality_issues" in los_data
        and los_data["data_quality_issues"] > 0
    ):
        quality_details = los_data.get("quality_details", {})
        quality_issues_df = los_data.get("quality_issues_df", pd.DataFrame())

        # Build warning content
        warning_issues = []
        if quality_details.get("exit_before_start", 0) > 0:
            warning_issues.append(
                f"{quality_details['exit_before_start']} record(s) with exit date before entry date (excluded)"
            )
        if quality_details.get("extremely_long_stays", 0) > 0:
            warning_issues.append(
                f"{quality_details['extremely_long_stays']} record(s) with stays over 5 years in non-permanent housing"
            )
        if quality_details.get("future_entry", 0) > 0:
            warning_issues.append(
                f"{quality_details['future_entry']} record(s) with entry dates after report end"
            )

        warning_content = f"""
        Found {los_data['data_quality_issues']}
            enrollment(s) with data quality issues:
        <ul style='margin: 10px 0; padding-left: 20px;'>
            <li>{'</li><li>'.join(warning_issues)}</li>
        </ul>
        <em>Please review and correct these data entry errors in the source system.</em>
        """

        st.html(
            html_factory.info_box(
                content=warning_content,
                type="warning",
                title="Data Quality Warning",
                icon="⚠️",
            )
        )

        # Add download button for quality issues
        if not quality_issues_df.empty:
            csv = quality_issues_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data Quality Issues Report",
                data=csv,
                file_name=f"los_data_quality_issues_{
                    t0.strftime('%Y%m%d')}_{
                    t1.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Download the list of enrollments with data quality issues for review",
            )

    # Create tabs for different views
    tab_overview, tab_demo = ui.los_tabs()

    with tab_overview:
        # Calculate additional metrics
        all_los = los_data["los_by_enrollment"]["LOS"]
        q1 = all_los.quantile(0.25) if not all_los.empty else 0
        q3 = all_los.quantile(0.75) if not all_los.empty else 0
        total_enrollments = len(all_los)
        unique_clients = los_data.get("unique_clients", 0)
        project_types = los_data.get("project_types", pd.Series())

        # Display enhanced summary
        summary_html = _create_los_summary_html(
            los_data["avg_los"],
            los_data["median_los"],
            q1,
            q3,
            total_enrollments,
            unique_clients,
            project_types,
        )
        st.html(summary_html)

        blue_divider()

        # Distribution chart
        st.html(
            html_factory.title(
                text="Length of Stay Distribution", level=3, icon="📊"
            )
        )

        if los_data["distribution"].empty:
            st.info(
                "No length of stay data available for the selected period."
            )
            return

        # Prepare distribution data
        dist_df = los_data["distribution"].copy()
        dist_df["color"] = dist_df["LOS_Category"].apply(_get_los_color)

        # Calculate percentage
        total_count = dist_df["count"].sum()
        if total_count > 0:
            dist_df["percentage"] = (
                dist_df["count"] / total_count * 100
            ).round(1)
        else:
            dist_df["percentage"] = 0

        # Create distribution chart
        fig = go.Figure()

        for idx, row in dist_df.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[row["LOS_Category"]],
                    y=[row["count"]],
                    name=row["LOS_Category"],
                    marker_color=row["color"],
                    text=f"{row['count']:,}<br>({row['percentage']:.0f}%)",
                    textposition="outside",
                    textfont=dict(color="white", size=14),
                    hovertemplate=(
                        f"<b>{row['LOS_Category']}</b><br>"
                        f"Count: {row['count']:,}<br>"
                        f"Percentage: {row['percentage']:.1f}%<br>"
                        f"<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

        fig.update_layout(
            template=PLOT_TEMPLATE,
            height=500,
            margin=dict(l=80, r=80, t=100, b=100),
            xaxis=dict(
                title="Length of Stay Category", tickangle=0, automargin=True
            ),
            yaxis=dict(
                title="Number of Enrollments",
                automargin=True,
                rangemode="tozero",
                range=[0, dist_df["count"].max() * 1.25],
            ),
            showlegend=False,
            bargap=0.2,
        )

        st.plotly_chart(fig, width='stretch')

        # Category explanation with improved UI
        category_content = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;'>"

        for category, info in LOS_CATEGORIES.items():
            category_content += f"""
                <div style='display: flex; align-items: center; gap: 10px;'>
                    <div style='width: 20px; height: 20px; background-color: {info['color']};
                                border-radius: 3px; border: 1px solid white;'></div>
                    <span><strong>{category}
                </strong>: {info['desc']}</span>
                </div>
            """

        category_content += "</div>"

        st.html(
            html_factory.info_box(
                content=category_content,
                type="info",
                title="Understanding Stay Categories",
                icon="📚",
            )
        )

        # Exit destination analysis if available
        if "ExitDestinationCat" in df_filt.columns:
            with st.expander(
                "📤 Length of Stay by Exit Destination", expanded=False
            ):
                dest_summary = analyze_los_with_destinations(df_filt, los_data)
                if not dest_summary.empty:
                    st.html(
                        html_factory.title(
                            text="Average Stay by Exit Destination",
                            level=3,
                            icon="📤",
                        )
                    )

                    # Create bar chart
                    fig_dest = px.bar(
                        dest_summary.reset_index(),
                        x="ExitDestinationCat",
                        y="mean_los",
                        text="mean_los",
                        color="ExitDestinationCat",
                        color_discrete_map={
                            "Permanent Housing Situations": SUCCESS_COLOR,
                            "Temporary Housing Situations": WARNING_COLOR,
                            "Homeless Situations": DANGER_COLOR,
                            "Institutional Situations": NEUTRAL_COLOR,
                            "Other": SECONDARY_COLOR,
                        },
                        title="Average Length of Stay by Exit Destination",
                    )

                    fig_dest.update_traces(
                        texttemplate="%{text:.0f} days", textposition="outside"
                    )

                    fig_dest.update_layout(
                        template=PLOT_TEMPLATE,
                        height=400,
                        showlegend=False,
                        xaxis_title="Exit Destination Category",
                        yaxis_title="Average Length of Stay (days)",
                    )

                    st.plotly_chart(fig_dest, width='stretch')

                    # Show summary table
                    st.dataframe(
                        dest_summary.style.format(
                            {
                                "mean_los": "{:.0f} days",
                                "median_los": "{:.0f} days",
                                "count": "{:,}",
                            }
                        ),
                        width='stretch',
                    )
                else:
                    st.info("No exit destination data available.")

    with tab_demo:
        st.html(
            html_factory.title(
                text="Length of Stay by Demographics", level=3, icon="👥"
            )
        )

        # Add explanation with improved UI
        demo_explanation = """
        <strong>Why this matters:</strong> Identifying disparities in length of stay can reveal:
        <ul style='margin: 10px 0; padding-left: 20px;'>
            <li>Groups facing additional housing barriers</li>
            <li>Potential service gaps or mismatches</li>
            <li>Opportunities for targeted interventions</li>
        </ul>
        """

        st.html(
            html_factory.info_box(
                content=demo_explanation,
                type="info",
                title="Analysis Purpose",
                icon="🔍",
            )
        )

        # Create unique key for widgets
        key_suffix = hash_data(filter_timestamp)

        # Filter dimensions to only those in the dataset
        available_dimensions = [
            (lbl, col)
            for lbl, col in DEMOGRAPHIC_DIMENSIONS
            if col in df_filt.columns
        ]

        if not available_dimensions:
            st.warning(
                "No demographic dimensions available in the filtered dataset."
            )
            return

        # Dimension selection
        demo_label = st.selectbox(
            "Compare by",
            [lbl for lbl, _ in available_dimensions],
            key=f"los_dim_{key_suffix}",
            help="Choose which characteristic to analyze",
        )

        # Get the actual column name
        demo_col = next(
            (col for lbl, col in available_dimensions if lbl == demo_label),
            None,
        )

        if not demo_col:
            st.error(f"Column mapping not found for {demo_label}.")
            return

        # Add filter for values within selected demographic
        try:
            demo_values = sorted(df_filt[demo_col].dropna().unique())
            selected_values = st.multiselect(
                f"Select {demo_label} groups to include",
                options=demo_values,
                default=demo_values,
                key=f"los_val_filter_{key_suffix}",
                help=f"Choose which {demo_label} groups to show",
            )
        except Exception as e:
            st.error(f"Error loading demographic values: {e}")
            return

        if not selected_values:
            st.warning(f"Please select at least one {demo_label} group.")
            return

        # Minimum group size
        min_group = st.slider(
            "Minimum group size to display",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            key=f"los_min_group_{key_suffix}",
            help="Hide groups with fewer enrollments for statistical reliability",
        )

        # Track dimension and selected values for recompute trigger
        previous_selection = state.get("selected_dimension", "")
        previous_value_filter = state.get("selected_values", [])
        previous_min_group = state.get("min_group_size", 10)

        dim_changed = previous_selection != demo_label
        values_changed = sorted(previous_value_filter) != sorted(
            selected_values
        )
        min_group_changed = previous_min_group != min_group

        if dim_changed or values_changed or min_group_changed:
            state["selected_dimension"] = demo_label
            state["selected_values"] = selected_values
            state["min_group_size"] = min_group
            if "los_by_demo" in state:
                state.pop("los_by_demo")

        # Calculate LOS by demographic if needed
        if "los_by_demo" not in state:
            with st.spinner(f"Calculating length of stay by {demo_label}..."):
                try:
                    # Filter to selected demographic values
                    df_subset = df_filt[
                        df_filt[demo_col].isin(selected_values)
                    ]

                    if df_subset.empty:
                        st.info(
                            f"No data available for the selected {demo_label} groups."
                        )
                        return

                    # Calculate LOS by demographic
                    los_by_demo = los_by_demographic(
                        df_subset, demo_col, t0, t1, min_group
                    )

                    if los_by_demo.empty:
                        st.info(
                            f"No groups meet the minimum size threshold of {
                            min_group} enrollments."
                        )
                        return

                    state["los_by_demo"] = los_by_demo

                except Exception as e:
                    st.error(f"Error calculating LOS by {demo_label}: {e}")
                    return
        else:
            los_by_demo = state["los_by_demo"]

        blue_divider()

        # Sort data for visualization
        sorted_los_by_demo = los_by_demo.sort_values("mean", ascending=False)

        # Limit number of groups to display
        MAX_GROUPS = 15
        if len(sorted_los_by_demo) > MAX_GROUPS:
            st.info(
                f"📊 Showing top {MAX_GROUPS} groups by average stay length"
            )
            sorted_los_by_demo = sorted_los_by_demo.head(MAX_GROUPS)

        # Calculate chart height
        chart_height = _calculate_chart_height(len(sorted_los_by_demo), 450)

        # Create combined chart comparing median, mean, and quartiles
        fig_combined = go.Figure()

        # Add IQR range as horizontal bars
        for idx, row in sorted_los_by_demo.iterrows():
            # Main IQR bar
            fig_combined.add_trace(
                go.Bar(
                    y=[row[demo_col]],
                    x=[row["q3"] - row["q1"]],  # Width of bar = IQR
                    base=row["q1"],  # Start position = Q1
                    orientation="h",
                    name="Typical Range",
                    marker=dict(
                        color=MAIN_COLOR,
                        opacity=0.3,
                        line=dict(color=MAIN_COLOR, width=2),
                    ),
                    showlegend=idx == 0,  # Only show in legend once
                    hoverinfo="text",
                    hovertext=f"Most enrollments ({row[demo_col]}) stay between {
                        row['q1']:.0f}-{row['q3']:.0f} days",
                )
            )

        # Add median markers
        fig_combined.add_trace(
            go.Scatter(
                y=sorted_los_by_demo[demo_col],
                x=sorted_los_by_demo["median"],
                mode="markers",
                name="Median (Middle Value)",
                marker=dict(
                    symbol="line-ns",
                    color=MAIN_COLOR,
                    size=25,
                    line=dict(width=4, color=MAIN_COLOR),
                ),
                text=sorted_los_by_demo["median"].apply(
                    lambda x: f"{x:.0f} days"
                ),
                hovertemplate="<b>%{y}</b><br>Median stay: %{text}<extra></extra>",
            )
        )

        # Add mean markers
        fig_combined.add_trace(
            go.Scatter(
                y=sorted_los_by_demo[demo_col],
                x=sorted_los_by_demo["mean"],
                mode="markers",
                name="Average",
                marker=dict(
                    symbol="diamond",
                    color=SECONDARY_COLOR,
                    size=14,
                    line=dict(width=2, color="white"),
                ),
                text=sorted_los_by_demo["mean"].apply(
                    lambda x: f"{x:.0f} days"
                ),
                hovertemplate="<b>%{y}</b><br>Average stay: %{text}<extra></extra>",
            )
        )

        # Add reference line for overall system median
        if "median_los" in los_data:
            fig_combined.add_vline(
                x=los_data["median_los"],
                line=dict(dash="dash", color=NEUTRAL_COLOR, width=2),
                annotation=dict(
                    text=f"System Median: {los_data['median_los']:.0f} days",
                    font=dict(size=13, color=NEUTRAL_COLOR, weight="bold"),
                    bordercolor=NEUTRAL_COLOR,
                    borderwidth=2,
                    bgcolor="rgba(0,0,0,0.8)",
                    borderpad=4,
                ),
                annotation_position="top right",
            )

        # Update layout
        fig_combined.update_layout(
            title=dict(
                text=f"Length of Stay by {demo_label}", font=dict(size=20)
            ),
            template=PLOT_TEMPLATE,
            height=chart_height,
            margin=dict(l=150, r=100, t=120, b=80),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.3)",
                borderwidth=1,
            ),
            yaxis=dict(automargin=True, tickfont=dict(size=12), title=""),
            xaxis=dict(
                rangemode="tozero",
                title=dict(text="Length of Stay (days)", standoff=25),
                gridcolor="rgba(255,255,255,0.1)",
            ),
            bargap=0.3,
        )

        # Display the chart
        st.plotly_chart(fig_combined, width='stretch')

        # Chart guide
        chart_guide_html = f"""
        <div style="background-color: rgba(0,0,0,0.2); border-radius: 8px; padding: 10px; margin: 10px 0;">
            <p style="margin: 0; font-size: 14px;">
                📊 <strong>Chart Guide:</strong>
                Bar shows typical range (25th-75th percentile) |
                <span style="color: {MAIN_COLOR};">Blue line</span> = median |
                <span style="color: {SECONDARY_COLOR};">Red diamond</span> = average
            </p>
        </div>
        """
        st.html(chart_guide_html)

        # Resource utilization analysis
        blue_divider()
        st.html(
            html_factory.title(
                text="Resource Utilization Analysis", level=3, icon="📊"
            )
        )

        # Calculate proportional metrics
        total_days = sorted_los_by_demo["total_days"].sum()
        total_enrollments = sorted_los_by_demo["count"].sum()

        # Add percentage columns
        sorted_los_by_demo["days_pct"] = (
            sorted_los_by_demo["total_days"] / total_days * 100
        ).round(1)
        sorted_los_by_demo["enrollment_pct"] = (
            sorted_los_by_demo["count"] / total_enrollments * 100
        ).round(1)

        # Create proportional usage comparison
        compare_df = sorted_los_by_demo.head(8).copy()

        # Create grouped bar chart for comparison
        fig_compare = go.Figure()

        # Add bars for enrollment percentage
        fig_compare.add_trace(
            go.Bar(
                y=compare_df[demo_col],
                x=compare_df["enrollment_pct"],
                orientation="h",
                name="% of Enrollments",
                marker_color=NEUTRAL_COLOR,
                text=[f"{x:.0f}%" for x in compare_df["enrollment_pct"]],
                textposition="outside",
                textfont=dict(color="white", size=12),
            )
        )

        # Add bars for days percentage
        fig_compare.add_trace(
            go.Bar(
                y=compare_df[demo_col],
                x=compare_df["days_pct"],
                orientation="h",
                name="% of Total Enrollment Days",
                marker_color=MAIN_COLOR,
                text=[f"{x:.0f}%" for x in compare_df["days_pct"]],
                textposition="outside",
                textfont=dict(color="white", size=12),
            )
        )

        # Update layout
        fig_compare.update_layout(
            title="Resource Usage: Enrollment Count vs. Total Days",
            template=PLOT_TEMPLATE,
            barmode="group",
            bargap=0.15,
            bargroupgap=0.1,
            height=450,
            margin=dict(l=150, r=150, t=120, b=100),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            yaxis=dict(automargin=True, tickfont=dict(size=12)),
            xaxis=dict(
                title=dict(text="Percentage (%)", standoff=25),
                range=[
                    0,
                    max(
                        compare_df["days_pct"].max(),
                        compare_df["enrollment_pct"].max(),
                    )
                    * 1.3,
                ],
            ),
        )

        # Display the chart
        st.plotly_chart(fig_compare, width='stretch')

        # Insights section
        with st.expander("💡 Key Findings", expanded=True):
            if len(sorted_los_by_demo) > 1:
                longest_stay = sorted_los_by_demo.iloc[0]
                shortest_stay = sorted_los_by_demo.iloc[-1]
                disparity_ratio = (
                    longest_stay["mean"] / shortest_stay["mean"]
                    if shortest_stay["mean"] > 0
                    else 0
                )

                # Create insights HTML
                insights_html = _create_demographic_insights_html(
                    longest_stay, shortest_stay, disparity_ratio, demo_col
                )
                st.html(insights_html)

                # Check for high disparity groups
                high_disparity = sorted_los_by_demo[
                    sorted_los_by_demo["disparity"] > 1.5
                ]
                if not high_disparity.empty:
                    groups_list = [
                        f"<strong>{row[demo_col]}</strong>"
                        for _, row in high_disparity.iterrows()
                    ]
                    groups_text = ", ".join(groups_list)

                    # Context-aware message based on project type
                    if demo_col == "ProjectTypeCode":
                        # Check if any high disparity groups are permanent
                        # housing
                        ph_groups = [
                            row
                            for _, row in high_disparity.iterrows()
                            if any(
                                ph in str(row[demo_col])
                                for ph in [
                                    "Permanent",
                                    "Housing Only",
                                    "Housing with Services",
                                ]
                            )
                        ]
                        if ph_groups:
                            success_content = f"""
                            These permanent housing programs show excellent housing stability with extended stays:
                            <div style='padding-left: 20px; margin: 10px 0;'>{groups_text}</div>
                            <em>Long stays in permanent housing indicate successful housing retention.</em>
                            """

                            notable_html = html_factory.info_box(
                                content=success_content,
                                type="success",
                                title="Housing Stability Success",
                                icon="🏠",
                            )
                        else:
                            extended_content = f"""
                            These groups have stays at least 50% longer than average:
                            <div style='padding-left: 20px; margin: 10px 0;'>{groups_text}
                                </div>
                            """

                            notable_html = html_factory.info_box(
                                content=extended_content,
                                type="warning",
                                title="Groups with Extended Stays",
                                icon="⏰",
                            )
                    else:
                        extended_content = f"""
                        These groups have stays at least 50% longer than average:
                        <div style='padding-left: 20px; margin: 10px 0;'>{groups_text}
                            </div>
                        """

                        notable_html = html_factory.info_box(
                            content=extended_content,
                            type="warning",
                            title="Groups with Extended Stays",
                            icon="⏰",
                        )
                    st.html(notable_html)
