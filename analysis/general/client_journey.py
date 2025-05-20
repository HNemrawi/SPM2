"""
Client Journey Analysis for HMIS Dashboard.

This module analyzes client pathways through the homeless service system,
showing how clients move between different programs within the reporting period.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas import DataFrame, Timestamp

from analysis.general.data_utils import (
    _safe_div, ph_exit_clients
)
from analysis.general.filter_utils import (
    get_filter_timestamp, hash_data, init_section_state, is_cache_valid, invalidate_cache
)
from analysis.general.theme import (
    CUSTOM_COLOR_SEQUENCE, MAIN_COLOR, PLOT_TEMPLATE, SECONDARY_COLOR, SUCCESS_COLOR,
    WARNING_COLOR, apply_chart_style, fmt_int, fmt_pct
)

# Constants
JOURNEY_SECTION_KEY = "client_journey"

# Project type mapping for consistent display
PROJECT_TYPE_MAP = {
    # Full program names to abbreviated, consistent naming
    "Coordinated Entry": "Coordinated Entry",
    "Day Shelter": "Day Shelter",
    "Emergency Shelter – Entry Exit": "Emergency Shelter",
    "Emergency Shelter – Night-by-Night": "Emergency Shelter",
    "Homelessness Prevention": "Prevention",
    "Other": "Other",
    "PH – Housing Only": "PH - Housing Only",
    "PH – Housing with Services (no disability required for entry)": "PH - Housing with Services",
    "PH – Permanent Supportive Housing (disability required for entry)": "PSH",
    "PH – Rapid Re-Housing": "RRH",
    "Safe Haven": "Safe Haven",
    "Services Only": "Services Only",
    "Street Outreach": "Street Outreach",
    "Transitional Housing": "Transitional Housing",
    # Numeric codes
    "1": "Emergency Shelter",
    "2": "Transitional Housing",
    "3": "PH",
    "4": "Street Outreach",
    "6": "Services Only",
    "7": "Prevention",
    "8": "Coordinated Entry",
    "9": "Safe Haven",
    "10": "PSH",
    "11": "RRH",
    "12": "PH - Housing Only",
    "13": "PH - Housing with Services",
    "14": "Day Shelter",
}


def deduplicate_program_sequence(sequence):
    """
    Remove consecutive duplicates from a program sequence.
    Example: ["CE", "CE", "ES", "ES", "ES", "RRH"] becomes ["CE", "ES", "RRH"]
    """
    if not sequence:
        return []
    
    deduped = [sequence[0]]
    for prog in sequence[1:]:
        if prog != deduped[-1]:  # Only add if different from the last program
            deduped.append(prog)
    
    return deduped


@st.cache_data(show_spinner="Analyzing client journeys...")
def analyze_client_pathways(
    df_filt: DataFrame,
    df_full: DataFrame,
    start: Timestamp,
    end: Timestamp,
    include_ph_exits: bool = False,
    focus_programs: Optional[List[str]] = None,
    deduplicate_sequences: bool = False
) -> Dict[str, Any]:
    """
    Analyze client pathways based on enrollments WITHIN the reporting period.
    
    Parameters:
    -----------
    df_filt : DataFrame
        Filtered DataFrame for initial client selection
    df_full : DataFrame
        Full DataFrame to analyze all enrollments for selected clients
    start : Timestamp
        Start date for the analysis period
    end : Timestamp
        End date for the analysis period
    include_ph_exits : bool, default=False
        Whether to analyze pathways specifically for PH exits
    focus_programs : Optional[List[str]], default=None
        List of program types to focus analysis on. If provided, only
        clients who touched these programs will be included.
    deduplicate_sequences : bool, default=False
        If True, remove consecutive duplicate programs in a client's pathway
        (e.g., "ES → ES → ES → RRH" becomes "ES → RRH").
        
    Returns:
    --------
    Dict[str, Any] containing pathway analysis results
    """
    # Empty results template
    empty_results = {
        "transitions": pd.DataFrame(),
        "program_counts": pd.DataFrame(),
        "concurrent_pairs": pd.DataFrame(),
        "metrics": {
            "avg_enrollments": 0.0,
            "clients_analyzed": 0,
            "multi_enrollment_count": 0,
            "multi_enrollment_pct": 0.0,
            "concurrent_enrollment_count": 0,
            "concurrent_enrollment_pct": 0.0,
            "focus_programs": focus_programs or []
        },
    }
    
    # Validate required columns
    required_cols = ["ClientID", "ProjectStart", "ProjectExit", "ProjectTypeCode"]
    if not all(col in df_filt.columns for col in required_cols) or not all(col in df_full.columns for col in required_cols):
        return empty_results
    
    # Return empty results if DataFrame is empty
    if df_filt.empty:
        return empty_results
    
    # Step 1: Get list of clients from filtered dataframe (based on dashboard filters)
    initial_clients = set(df_filt["ClientID"].unique())
    
    # Step 2: Find all active enrollments for these clients from the full dataframe
    # This ensures we have complete enrollment data even if dashboard filters excluded some
    full_mask = (
        (df_full["ClientID"].isin(initial_clients)) &
        ((df_full["ProjectExit"] >= start) | df_full["ProjectExit"].isna()) & 
        (df_full["ProjectStart"] <= end)
    )
    all_active_enrollments = df_full.loc[full_mask].copy()
    
    # Early return if no active enrollments
    if all_active_enrollments.empty:
        return empty_results
    
    # Map project types to consistent names
    all_active_enrollments["Program"] = (
        all_active_enrollments["ProjectTypeCode"]
        .astype(str)
        .map(lambda x: PROJECT_TYPE_MAP.get(x, x))
    )
    
    # Set of clients who have active enrollments in the reporting period
    active_clients = set(all_active_enrollments["ClientID"].unique())
    
    # Step 3: If we're focusing on PH exits, further filter clients
    # but still show ALL enrollments for these clients
    if include_ph_exits:
        # Get clients with PH exits in the full data during date range
        ph_clients = ph_exit_clients(df_full, start, end)
        
        # Only keep clients who had PH exits
        ph_exit_clients_set = active_clients.intersection(ph_clients)
        
        # Early return if no clients match after applying PH exit filter
        if not ph_exit_clients_set:
            return empty_results
            
        # Update active clients to only those with PH exits
        active_clients = ph_exit_clients_set
    
    # Step 4: If we're focusing on specific programs, further filter clients
    # but still show ALL enrollments for these clients
    if focus_programs:
        # Find clients who touched any of the focus programs during the reporting period
        focus_clients_mask = (
            (all_active_enrollments["Program"].isin(focus_programs)) &
            (all_active_enrollments["ClientID"].isin(active_clients))
        )
        focus_clients = set(all_active_enrollments.loc[focus_clients_mask, "ClientID"].unique())
        
        # Early return if no clients match after applying focus
        if not focus_clients:
            return empty_results
            
        # Update active clients to only those who touched focus programs
        active_clients = focus_clients
    
    # Step 5: Get ALL enrollments for the final filtered clients
    # This ensures we have the complete journey for each included client
    client_enrollments = all_active_enrollments[all_active_enrollments["ClientID"].isin(active_clients)].copy()
    
    # Calculate client metrics
    # 1. Count clients with multiple enrollments
    enrollment_counts = client_enrollments.groupby("ClientID").size().reset_index(name="EnrollmentCount")
    multi_enrollment_clients = enrollment_counts[enrollment_counts["EnrollmentCount"] > 1]
    multi_enrollment_count = len(multi_enrollment_clients)
    multi_enrollment_pct = (multi_enrollment_count / len(active_clients) * 100) if active_clients else 0.0
    
    # Calculate average enrollments per client
    avg_enrollments = enrollment_counts["EnrollmentCount"].mean() if not enrollment_counts.empty else 0.0
    
    # 2. Identify concurrent enrollments (clients enrolled in multiple programs at the same time)
    concurrent_clients = set()
    concurrent_programs = {}  # For tracking which programs commonly overlap
    
    # For each client, check if any enrollments overlap in time
    for cid, grp in client_enrollments.groupby("ClientID"):
        if len(grp) <= 1:
            continue
            
        # Sort by start date
        sorted_enrolls = grp.sort_values("ProjectStart")
        
        # Check for overlaps
        for i in range(len(sorted_enrolls)):
            for j in range(i + 1, len(sorted_enrolls)):
                curr_row = sorted_enrolls.iloc[i]
                next_row = sorted_enrolls.iloc[j]
                
                curr_start = curr_row["ProjectStart"]
                curr_exit = curr_row["ProjectExit"]
                next_start = next_row["ProjectStart"]
                next_exit = next_row["ProjectExit"]
                
                # Handle null exit dates (still active)
                if pd.isna(curr_exit):
                    curr_exit = end
                if pd.isna(next_exit):
                    next_exit = end
                
                # Check for overlap: start of one is before end of other, and end of one is after start of other
                if curr_start <= next_exit and curr_exit >= next_start:
                    concurrent_clients.add(cid)
                    
                    # Track which programs were concurrent
                    prog_pair = tuple(sorted([curr_row["Program"], next_row["Program"]]))
                    concurrent_programs[prog_pair] = concurrent_programs.get(prog_pair, 0) + 1
    
    concurrent_count = len(concurrent_clients)
    concurrent_pct = (concurrent_count / len(active_clients) * 100) if active_clients else 0.0
    
    # Create concurrent programs dataframe
    concurrent_pairs = []
    for (prog1, prog2), count in concurrent_programs.items():
        concurrent_pairs.append({
            "Program1": prog1,
            "Program2": prog2,
            "ClientCount": count,
            "Percentage": (count / concurrent_count * 100) if concurrent_count else 0.0
        })
    
    concurrent_pairs_df = pd.DataFrame(concurrent_pairs).sort_values("ClientCount", ascending=False)
    
    # Calculate transitions between programs for each client's COMPLETE pathway
    transitions = []
    program_touchpoints = {}
    
    for cid, grp in client_enrollments.groupby("ClientID"):
        # Sort by start date to get chronological program sequence
        sorted_grp = grp.sort_values("ProjectStart")
        sequence = list(sorted_grp["Program"])
        
        # Skip if no programs
        if not sequence:
            continue
            
        # Deduplicate consecutive identical programs if requested
        if deduplicate_sequences:
            deduped = deduplicate_program_sequence(sequence)
            sequence = deduped
        
        # Record transitions between programs
        for i in range(len(sequence) - 1):
            transitions.append({
                "ClientID": cid,
                "FromProgram": sequence[i],
                "ToProgram": sequence[i + 1]
            })
        
        # Count program touchpoints
        for prog in sequence:
            program_touchpoints[prog] = program_touchpoints.get(prog, 0) + 1
    
    # Convert to DataFrame
    transitions_df = pd.DataFrame(transitions)
    
    # Calculate key metrics
    metrics = {
        "avg_enrollments": avg_enrollments,
        "clients_analyzed": len(active_clients),
        "multi_enrollment_count": multi_enrollment_count,
        "multi_enrollment_pct": multi_enrollment_pct,
        "concurrent_enrollment_count": concurrent_count,
        "concurrent_enrollment_pct": concurrent_pct,
        "enrollment_counts": enrollment_counts.to_dict() if not enrollment_counts.empty else {},
        "focus_programs": focus_programs or []
    }
    
    # Aggregate transitions
    if not transitions_df.empty:
        transition_counts = (
            transitions_df.groupby(["FromProgram", "ToProgram"])
            .agg(ClientCount=("ClientID", "nunique"))
            .reset_index()
            .sort_values("ClientCount", ascending=False)
        )
        
        # Calculate percentages
        total_transitions = len(transitions_df)
        transition_counts["Percentage"] = (transition_counts["ClientCount"] / total_transitions * 100).round(1)
    else:
        transition_counts = pd.DataFrame()
    
    # Create program touchpoints dataframe
    program_counts = pd.DataFrame([
        {"Program": prog, "ClientCount": count}
        for prog, count in program_touchpoints.items()
    ]).sort_values("ClientCount", ascending=False)
    
    # Calculate percentages - this shows % of clients who used each program
    if not program_counts.empty:
        program_counts["Percentage"] = (program_counts["ClientCount"] / len(active_clients) * 100).round(1)
    
    return {
        "transitions": transition_counts,
        "program_counts": program_counts,
        "concurrent_pairs": concurrent_pairs_df,
        "metrics": metrics,
    }
    
    # Calculate client metrics
    # 1. Count clients with multiple enrollments
    enrollment_counts = client_enrollments.groupby("ClientID").size().reset_index(name="EnrollmentCount")
    multi_enrollment_clients = enrollment_counts[enrollment_counts["EnrollmentCount"] > 1]
    multi_enrollment_count = len(multi_enrollment_clients)
    multi_enrollment_pct = (multi_enrollment_count / len(active_clients) * 100) if active_clients else 0.0
    
    # Calculate average enrollments per client
    avg_enrollments = enrollment_counts["EnrollmentCount"].mean() if not enrollment_counts.empty else 0.0
    
    # 2. Identify concurrent enrollments (clients enrolled in multiple programs at the same time)
    concurrent_clients = set()
    concurrent_programs = {}  # For tracking which programs commonly overlap
    
    # For each client, check if any enrollments overlap in time
    for cid, grp in client_enrollments.groupby("ClientID"):
        if len(grp) <= 1:
            continue
            
        # Sort by start date
        sorted_enrolls = grp.sort_values("ProjectStart")
        
        # Check for overlaps
        for i in range(len(sorted_enrolls)):
            for j in range(i + 1, len(sorted_enrolls)):
                curr_row = sorted_enrolls.iloc[i]
                next_row = sorted_enrolls.iloc[j]
                
                curr_start = curr_row["ProjectStart"]
                curr_exit = curr_row["ProjectExit"]
                next_start = next_row["ProjectStart"]
                next_exit = next_row["ProjectExit"]
                
                # Handle null exit dates (still active)
                if pd.isna(curr_exit):
                    curr_exit = end
                if pd.isna(next_exit):
                    next_exit = end
                
                # Check for overlap: start of one is before end of other, and end of one is after start of other
                if curr_start <= next_exit and curr_exit >= next_start:
                    concurrent_clients.add(cid)
                    
                    # Track which programs were concurrent
                    prog_pair = tuple(sorted([curr_row["Program"], next_row["Program"]]))
                    concurrent_programs[prog_pair] = concurrent_programs.get(prog_pair, 0) + 1
    
    concurrent_count = len(concurrent_clients)
    concurrent_pct = (concurrent_count / len(active_clients) * 100) if active_clients else 0.0
    
    # Create concurrent programs dataframe
    concurrent_pairs = []
    for (prog1, prog2), count in concurrent_programs.items():
        concurrent_pairs.append({
            "Program1": prog1,
            "Program2": prog2,
            "ClientCount": count,
            "Percentage": (count / concurrent_count * 100) if concurrent_count else 0.0
        })
    
    concurrent_pairs_df = pd.DataFrame(concurrent_pairs).sort_values("ClientCount", ascending=False)
    
    # Calculate transitions between programs
    transitions = []
    program_touchpoints = {}
    
    for cid, grp in client_enrollments.groupby("ClientID"):
        # Sort by start date to get chronological program sequence
        sorted_grp = grp.sort_values("ProjectStart")
        sequence = list(sorted_grp["Program"])
        
        # Skip if no programs
        if not sequence:
            continue
            
        # Deduplicate consecutive identical programs if requested
        if deduplicate_sequences:
            deduped = deduplicate_program_sequence(sequence)
            sequence = deduped
        
        # Record transitions between programs
        for i in range(len(sequence) - 1):
            transitions.append({
                "ClientID": cid,
                "FromProgram": sequence[i],
                "ToProgram": sequence[i + 1]
            })
        
        # Count program touchpoints
        for prog in sequence:
            program_touchpoints[prog] = program_touchpoints.get(prog, 0) + 1
    
    # Convert to DataFrame
    transitions_df = pd.DataFrame(transitions)
    
    # Calculate key metrics
    metrics = {
        "avg_enrollments": avg_enrollments,
        "clients_analyzed": len(active_clients),
        "multi_enrollment_count": multi_enrollment_count,
        "multi_enrollment_pct": multi_enrollment_pct,
        "concurrent_enrollment_count": concurrent_count,
        "concurrent_enrollment_pct": concurrent_pct,
        "enrollment_counts": enrollment_counts.to_dict() if not enrollment_counts.empty else {},
        "focus_programs": focus_programs or []
    }
    
    # Aggregate transitions
    if not transitions_df.empty:
        transition_counts = (
            transitions_df.groupby(["FromProgram", "ToProgram"])
            .agg(ClientCount=("ClientID", "nunique"))
            .reset_index()
            .sort_values("ClientCount", ascending=False)
        )
        
        # Calculate percentages
        total_transitions = len(transitions_df)
        transition_counts["Percentage"] = (transition_counts["ClientCount"] / total_transitions * 100).round(1)
    else:
        transition_counts = pd.DataFrame()
    
    # Create program touchpoints dataframe
    program_counts = pd.DataFrame([
        {"Program": prog, "ClientCount": count}
        for prog, count in program_touchpoints.items()
    ]).sort_values("ClientCount", ascending=False)
    
    # Calculate percentages - this shows % of clients who used each program
    if not program_counts.empty:
        program_counts["Percentage"] = (program_counts["ClientCount"] / len(active_clients) * 100).round(1)
    
    return {
        "transitions": transition_counts,
        "program_counts": program_counts,
        "concurrent_pairs": concurrent_pairs_df,
        "metrics": metrics,
    }
    
    # Calculate client metrics
    # 1. Count clients with multiple enrollments
    enrollment_counts = client_enrollments.groupby("ClientID").size().reset_index(name="EnrollmentCount")
    multi_enrollment_clients = enrollment_counts[enrollment_counts["EnrollmentCount"] > 1]
    multi_enrollment_count = len(multi_enrollment_clients)
    multi_enrollment_pct = (multi_enrollment_count / len(filtered_clients) * 100) if filtered_clients else 0.0
    
    # Calculate average enrollments per client
    avg_enrollments = enrollment_counts["EnrollmentCount"].mean() if not enrollment_counts.empty else 0.0
    
    # 2. Identify concurrent enrollments (clients enrolled in multiple programs at the same time)
    concurrent_clients = set()
    concurrent_programs = {}  # For tracking which programs commonly overlap
    
    # For each client, check if any enrollments overlap in time
    for cid, grp in client_enrollments.groupby("ClientID"):
        if len(grp) <= 1:
            continue
            
        # Sort by start date
        sorted_enrolls = grp.sort_values("ProjectStart")
        
        # Check for overlaps
        for i in range(len(sorted_enrolls)):
            for j in range(i + 1, len(sorted_enrolls)):
                curr_row = sorted_enrolls.iloc[i]
                next_row = sorted_enrolls.iloc[j]
                
                curr_start = curr_row["ProjectStart"]
                curr_exit = curr_row["ProjectExit"]
                next_start = next_row["ProjectStart"]
                next_exit = next_row["ProjectExit"]
                
                # Handle null exit dates (still active)
                if pd.isna(curr_exit):
                    curr_exit = end
                if pd.isna(next_exit):
                    next_exit = end
                
                # Check for overlap: start of one is before end of other, and end of one is after start of other
                if curr_start <= next_exit and curr_exit >= next_start:
                    concurrent_clients.add(cid)
                    
                    # Track which programs were concurrent
                    prog_pair = tuple(sorted([curr_row["Program"], next_row["Program"]]))
                    concurrent_programs[prog_pair] = concurrent_programs.get(prog_pair, 0) + 1
    
    concurrent_count = len(concurrent_clients)
    concurrent_pct = (concurrent_count / len(filtered_clients) * 100) if filtered_clients else 0.0
    
    # Create concurrent programs dataframe
    concurrent_pairs = []
    for (prog1, prog2), count in concurrent_programs.items():
        concurrent_pairs.append({
            "Program1": prog1,
            "Program2": prog2,
            "ClientCount": count,
            "Percentage": (count / concurrent_count * 100) if concurrent_count else 0.0
        })
    
    concurrent_pairs_df = pd.DataFrame(concurrent_pairs).sort_values("ClientCount", ascending=False)
    
    # Calculate transitions between programs
    transitions = []
    program_touchpoints = {}
    
    for cid, grp in client_enrollments.groupby("ClientID"):
        # Sort by start date to get chronological program sequence
        sorted_grp = grp.sort_values("ProjectStart")
        sequence = list(sorted_grp["Program"])
        
        # Skip if no programs
        if not sequence:
            continue
            
        # Deduplicate consecutive identical programs if requested
        if deduplicate_sequences:
            deduped = deduplicate_program_sequence(sequence)
            sequence = deduped
        
        # Record transitions between programs
        for i in range(len(sequence) - 1):
            transitions.append({
                "ClientID": cid,
                "FromProgram": sequence[i],
                "ToProgram": sequence[i + 1]
            })
        
        # Count program touchpoints
        for prog in sequence:
            program_touchpoints[prog] = program_touchpoints.get(prog, 0) + 1
    
    # Convert to DataFrame
    transitions_df = pd.DataFrame(transitions)
    
    # Calculate key metrics
    metrics = {
        "avg_enrollments": avg_enrollments,
        "clients_analyzed": len(filtered_clients),
        "multi_enrollment_count": multi_enrollment_count,
        "multi_enrollment_pct": multi_enrollment_pct,
        "concurrent_enrollment_count": concurrent_count,
        "concurrent_enrollment_pct": concurrent_pct,
        "enrollment_counts": enrollment_counts.to_dict() if not enrollment_counts.empty else {},
        "focus_programs": focus_programs or []
    }
    
    # Aggregate transitions
    if not transitions_df.empty:
        transition_counts = (
            transitions_df.groupby(["FromProgram", "ToProgram"])
            .agg(ClientCount=("ClientID", "nunique"))
            .reset_index()
            .sort_values("ClientCount", ascending=False)
        )
        
        # Calculate percentages
        total_transitions = len(transitions_df)
        transition_counts["Percentage"] = (transition_counts["ClientCount"] / total_transitions * 100).round(1)
    else:
        transition_counts = pd.DataFrame()
    
    # Create program touchpoints dataframe
    program_counts = pd.DataFrame([
        {"Program": prog, "ClientCount": count}
        for prog, count in program_touchpoints.items()
    ]).sort_values("ClientCount", ascending=False)
    
    # Calculate percentages - this shows % of clients who used each program
    if not program_counts.empty:
        program_counts["Percentage"] = (program_counts["ClientCount"] / len(filtered_clients) * 100).round(1)
    
    return {
        "transitions": transition_counts,
        "program_counts": program_counts,
        "concurrent_pairs": concurrent_pairs_df,
        "metrics": metrics,
    }
    
    # Calculate client metrics
    # 1. Count clients with multiple enrollments
    enrollment_counts = client_enrollments.groupby("ClientID").size().reset_index(name="EnrollmentCount")
    multi_enrollment_clients = enrollment_counts[enrollment_counts["EnrollmentCount"] > 1]
    multi_enrollment_count = len(multi_enrollment_clients)
    multi_enrollment_pct = (multi_enrollment_count / len(filtered_clients) * 100) if filtered_clients else 0.0
    
    # Calculate average enrollments per client
    avg_enrollments = enrollment_counts["EnrollmentCount"].mean() if not enrollment_counts.empty else 0.0
    
    # 2. Identify concurrent enrollments (clients enrolled in multiple programs at the same time)
    concurrent_clients = set()
    concurrent_programs = {}  # For tracking which programs commonly overlap
    
    # For each client, check if any enrollments overlap in time
    for cid, grp in client_enrollments.groupby("ClientID"):
        if len(grp) <= 1:
            continue
            
        # Sort by start date
        sorted_enrolls = grp.sort_values("ProjectStart")
        
        # Check for overlaps
        for i in range(len(sorted_enrolls)):
            for j in range(i + 1, len(sorted_enrolls)):
                curr_row = sorted_enrolls.iloc[i]
                next_row = sorted_enrolls.iloc[j]
                
                curr_start = curr_row["ProjectStart"]
                curr_exit = curr_row["ProjectExit"]
                next_start = next_row["ProjectStart"]
                next_exit = next_row["ProjectExit"]
                
                # Handle null exit dates (still active)
                if pd.isna(curr_exit):
                    curr_exit = end
                if pd.isna(next_exit):
                    next_exit = end
                
                # Check for overlap: start of one is before end of other, and end of one is after start of other
                if curr_start <= next_exit and curr_exit >= next_start:
                    concurrent_clients.add(cid)
                    
                    # Track which programs were concurrent
                    prog_pair = tuple(sorted([curr_row["Program"], next_row["Program"]]))
                    concurrent_programs[prog_pair] = concurrent_programs.get(prog_pair, 0) + 1
    
    concurrent_count = len(concurrent_clients)
    concurrent_pct = (concurrent_count / len(filtered_clients) * 100) if filtered_clients else 0.0
    
    # Create concurrent programs dataframe
    concurrent_pairs = []
    for (prog1, prog2), count in concurrent_programs.items():
        concurrent_pairs.append({
            "Program1": prog1,
            "Program2": prog2,
            "ClientCount": count,
            "Percentage": (count / concurrent_count * 100) if concurrent_count else 0.0
        })
    
    concurrent_pairs_df = pd.DataFrame(concurrent_pairs).sort_values("ClientCount", ascending=False)
    
    # Calculate transitions between programs
    transitions = []
    program_touchpoints = {}
    
    for cid, grp in client_enrollments.groupby("ClientID"):
        # Sort by start date to get chronological program sequence
        sorted_grp = grp.sort_values("ProjectStart")
        sequence = list(sorted_grp["Program"])
        
        # Skip if no programs
        if not sequence:
            continue
            
        # Deduplicate consecutive identical programs if requested
        if deduplicate_sequences:
            deduped = deduplicate_program_sequence(sequence)
            sequence = deduped
        
        # Record transitions between programs
        for i in range(len(sequence) - 1):
            transitions.append({
                "ClientID": cid,
                "FromProgram": sequence[i],
                "ToProgram": sequence[i + 1]
            })
        
        # Count program touchpoints
        for prog in sequence:
            program_touchpoints[prog] = program_touchpoints.get(prog, 0) + 1
    
    # Convert to DataFrame
    transitions_df = pd.DataFrame(transitions)
    
    # Calculate key metrics
    metrics = {
        "avg_enrollments": avg_enrollments,
        "clients_analyzed": len(filtered_clients),
        "multi_enrollment_count": multi_enrollment_count,
        "multi_enrollment_pct": multi_enrollment_pct,
        "concurrent_enrollment_count": concurrent_count,
        "concurrent_enrollment_pct": concurrent_pct,
        "enrollment_counts": enrollment_counts.to_dict() if not enrollment_counts.empty else {},
        "focus_programs": focus_programs or []
    }
    
    # Aggregate transitions
    if not transitions_df.empty:
        transition_counts = (
            transitions_df.groupby(["FromProgram", "ToProgram"])
            .agg(ClientCount=("ClientID", "nunique"))
            .reset_index()
            .sort_values("ClientCount", ascending=False)
        )
        
        # Calculate percentages
        total_transitions = len(transitions_df)
        transition_counts["Percentage"] = (transition_counts["ClientCount"] / total_transitions * 100).round(1)
    else:
        transition_counts = pd.DataFrame()
    
    # Create program touchpoints dataframe
    program_counts = pd.DataFrame([
        {"Program": prog, "ClientCount": count}
        for prog, count in program_touchpoints.items()
    ]).sort_values("ClientCount", ascending=False)
    
    # Calculate percentages - this shows % of clients who used each program
    if not program_counts.empty:
        program_counts["Percentage"] = (program_counts["ClientCount"] / len(filtered_clients) * 100).round(1)
    
    return {
        "transitions": transition_counts,
        "program_counts": program_counts,
        "concurrent_pairs": concurrent_pairs_df,
        "metrics": metrics,
    }


@st.fragment
def render_client_journey_analysis(df_filt: DataFrame, df_full: DataFrame) -> None:
    """
    Render the client pathway analysis section with clear visualizations and explanations.
    
    Parameters:
    -----------
    df_filt : DataFrame
        Filtered DataFrame for initial client selection
    df_full : DataFrame
        Full DataFrame for analyzing all enrollments
    """
    # Initialize section state
    state = init_section_state(JOURNEY_SECTION_KEY)
    
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
    st.subheader("🛣️ Client Journey Analysis")
    st.markdown("""
    This analysis examines how clients move through the service system during the reporting period.
    Unlike point-in-time metrics, this section reveals client pathways, transitions, and usage patterns
    to help you understand system flow and service coordination.
    """)
    
    # Get time boundaries
    t0 = st.session_state.get("t0")
    t1 = st.session_state.get("t1")
    
    if not (t0 and t1):
        st.warning("Please set date ranges in the filter panel.")
        return
    
    # Analysis settings
    st.markdown("### Analysis Settings")
    col1, col2 = st.columns([3, 3])
    
    # Determine available program types for filtering
    program_options = []
    try:
        # Get unique program types from full data
        prog_col = "ProjectTypeCode"
        if prog_col in df_full.columns:
            prog_types = df_full[prog_col].astype(str).unique()
            program_options = [(pt, PROJECT_TYPE_MAP.get(pt, pt)) for pt in prog_types if pd.notna(pt)]
            # Sort by display name
            program_options.sort(key=lambda x: x[1])
    except Exception as e:
        st.error(f"Error getting program types: {e}")
    
    # Get unique project types to focus on
    focus_programs = col1.multiselect(
        "Focus on specific program types",
        options=[pt[1] for pt in program_options],
        default=[],
        key=f"focus_programs_{filter_timestamp}",
        help="Select specific programs to focus on. If none selected, all programs are included."
    )
    
    focus_on_ph = col1.checkbox(
        "Housing exits only",
        value=False,
        key=f"ph_focus_{filter_timestamp}",
        help="When checked, only analyzes pathways for clients who exited to permanent housing"
    )
    
    # Add a toggle for deduplicating sequences
    deduplicate_sequences = col2.checkbox(
        "Deduplicate consecutive program entries",
        value=False,
        key=f"deduplicate_sequences_{filter_timestamp}",
        help="Remove consecutive entries in the same program (e.g., 'ES → ES → ES → RRH' becomes 'ES → RRH')"
    )
    
    # Set date range in a notice
    date_range = f"{t0.strftime('%b %d, %Y')} to {t1.strftime('%b %d, %Y')}"
    st.info(f"📅 **Analysis Period**: {date_range}")
    
    # Explain the data logic in a notice
    with st.expander("Understanding this analysis", expanded=True):
        st.markdown(f"""
        ### What data is being analyzed
        
        This analysis examines client enrollments **within the reporting period** ({date_range}). Here's exactly how it works:
        
        1. **Start with filtered clients**: We identify clients based on the current dashboard filters.
        
        2. **Get ALL enrollments**: We examine ALL enrollments for these clients that were active during the reporting period, including those that may have been filtered out in the dashboard.
        
        3. **Apply additional filters**: 
           * If "Housing exits only" is checked, we only include clients who exited to permanent housing during the reporting period.
           * If specific program types are selected, we only include clients who touched those programs at some point during the reporting period.
        
        4. **Show COMPLETE journeys**: For each client, we show their ENTIRE journey through all programs they used, including those that don't match any focus programs.
           * All enrollments are ordered by their start dates to show the actual chronological flow
           * Transitions include ALL movements between programs, not just those involving focus programs
           * Concurrent enrollments show ALL program combinations, not just those involving focus programs
        
        5. **What this means for your analysis**:
           * When you select a focus program like Coordinated Entry, you'll see how ALL clients who touched Coordinated Entry moved through your entire system
           * You'll see transitions both before and after clients used the focus program
           * This gives you the complete picture of client pathways
        """)
        
        if deduplicate_sequences:
            st.markdown("""
            ### Enrollment handling
            
            With "Deduplicate consecutive program entries" checked:
            * Multiple consecutive enrollments in the same program are combined (e.g., "ES → ES → ES → RRH" becomes "ES → RRH")
            * This makes the visualizations clearer by focusing on transitions between DIFFERENT program types
            * It removes the "noise" of multiple entries into the same program type
            """)
        else:
            st.markdown("""
            ### Enrollment handling
            
            Without "Deduplicate consecutive program entries" checked:
            * Every program enrollment appears exactly as it is in the data
            * This includes multiple consecutive enrollments in the same program type (e.g., "ES → ES → ES → RRH")
            * This shows more detail but may create more complex pathways with many transitions within the same program
            """)
            
        # Add clarification about program focus
        if focus_programs:
            programs_str = ", ".join(f"**{p}**" for p in focus_programs)
            st.markdown(f"""
            ### Program Focus: {programs_str}
            
            You've selected to focus on clients who used {programs_str}. This means:
            
            * Only clients who touched {programs_str} at some point are included in this analysis
            * BUT we still show their COMPLETE journeys through ALL programs they used
            * You'll see transitions both before and after they used {programs_str}
            * Concurrent enrollments will show ALL program combinations these clients experienced
            * This gives you the full picture of how these clients navigated your entire system
            """)
            
    
    # Map selected display names back to program codes for analysis
    focus_program_codes = []
    if focus_programs:
        # Create reverse mapping from display names to codes
        display_to_code = {disp: code for code, disp in program_options}
        # Map each selected display name back to its code
        focus_program_codes = [display_to_code.get(disp, disp) for disp in focus_programs]
    
    # Generate cache key
    cache_key = f"{'-'.join(sorted(focus_program_codes))}_{focus_on_ph}_{deduplicate_sequences}_{t0.date()}_{t1.date()}"
    recalc = state.get("cache_key") != cache_key
    
    # Calculate pathway data if needed
    if recalc or "pathway_data" not in state:
        state["cache_key"] = cache_key
        with st.spinner("Analyzing client journeys..."):
            try:
                # Analyze client pathways
                pathway_data = analyze_client_pathways(
                    df_filt,
                    df_full,
                    t0,
                    t1,
                    include_ph_exits=focus_on_ph,
                    focus_programs=focus_program_codes if focus_program_codes else None,
                    deduplicate_sequences=deduplicate_sequences
                )
                
                # Store in state
                state["pathway_data"] = pathway_data
                
            except Exception as e:
                st.error(f"Error analyzing client pathways: {e}")
                st.exception(e)
                return
    else:
        pathway_data = state.get("pathway_data", {})
    
    # Extract data components
    transitions = pathway_data.get("transitions", pd.DataFrame())
    program_counts = pathway_data.get("program_counts", pd.DataFrame())
    concurrent_pairs = pathway_data.get("concurrent_pairs", pd.DataFrame())
    metrics = pathway_data.get("metrics", {})
    
    # Handle empty data
    if not metrics.get("clients_analyzed", 0):
        # Provide specific guidance based on selected filters
        if focus_on_ph and focus_programs:
            st.info("No clients found who both used the selected program types AND exited to permanent housing during the selected date range. Try expanding your date range, selecting different programs, or unchecking 'Housing exits only'.")
        elif focus_on_ph:
            st.info("No clients with permanent housing exits found in the selected date range. Try expanding your date range or unchecking 'Housing exits only'.")
        elif focus_programs:
            st.info("No clients found who used the selected program types during the selected date range. Try expanding your date range or selecting different programs.")
        else:
            st.info("No client journey data available for the selected filters. Try expanding your date range.")
        return
    
    # Display overview metrics
    st.markdown("### System Navigation Summary")
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    
    client_count = metrics.get("clients_analyzed", 0)
    avg_enrollments = metrics.get("avg_enrollments", 0)
    multi_enrollment_pct = metrics.get("multi_enrollment_pct", 0)
    
    col1.metric(
        "Clients Analyzed", 
        fmt_int(client_count),
        help="Number of clients included in this analysis"
    )
    
    col2.metric(
        "Average Enrollments Per Client", 
        f"{avg_enrollments:.1f}",
        help="Average number of program enrollments each client had during the reporting period"
    )
    
    col3.metric(
        "Clients with Multiple Enrollments", 
        f"{multi_enrollment_pct:.1f}%",
        help="Percentage of clients who had more than one program enrollment during the reporting period"
    )
    
    # System interpretation based on metrics
    if multi_enrollment_pct < 15:
        st.info("📋 **System Structure**: Your system operates primarily with single-program interventions. Few clients (less than 15%) have multiple enrollments.")
    elif multi_enrollment_pct < 40:
        st.info("📋 **System Structure**: Your system shows moderate service coordination. About one-third of clients have multiple enrollments during their time in the system.")
    else:
        st.info(f"📋 **System Structure**: Your system shows significant client movement between programs. {multi_enrollment_pct:.1f}% of clients have multiple enrollments, suggesting strong coordination or complex client needs.")
    
    # Display analysis tabs
    tab1, tab2, tab3 = st.tabs([
        "Program Usage", "Program Transitions", "Enrollment Patterns"
    ])
    
    # TAB 1: PROGRAM USAGE
    with tab1:
        st.markdown("### Program Usage")
        st.markdown("""
        This chart shows which programs were used by clients during the reporting period.
        The percentages indicate what portion of clients used each program at some point.
        Since clients can use multiple programs, these percentages may sum to more than 100%.
        """)
        
        if not program_counts.empty:
            # Create bar chart
            fig = px.bar(
                program_counts,
                x="Percentage",
                y="Program",
                orientation="h",
                text="Percentage",
                labels={"Program": "Program Type", "Percentage": "% of Clients Using Program"},
                title="Program Usage During Reporting Period",
                color="ClientCount",
                color_continuous_scale="Blues",
                hover_data=["ClientCount"]
            )
            
            # Update hover template
            fig.update_traces(
                hovertemplate="<b>%{y}</b><br>" +
                              "Clients: %{customdata[0]}<br>" +
                              "Percentage: %{x:.1f}%<extra></extra>"
            )
            
            # Format text
            fig.update_traces(
                texttemplate="%{x:.1f}%",
                textposition="outside"
            )
            
            # Style the chart
            fig = apply_chart_style(
                fig,
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation about percentages
            st.info("""
            📊 **Understanding these percentages**: 
            Each percentage shows what portion of clients used that program at some point during the reporting period.
            Since many clients use multiple programs, these percentages can sum to more than 100%.
            """)
            
            # Create insights based on program usage
            high_usage = [p for p, r in zip(program_counts["Program"], program_counts["Percentage"]) if r > 50]
            if high_usage:
                programs_str = ", ".join(f"**{p}**" for p in high_usage)
                st.success(f"📊 **Key Observation**: {programs_str} {'is a dominant program' if len(high_usage) == 1 else 'are dominant programs'} in your system, used by over 50% of clients.")
            
            # If focus programs are selected, highlight them in the data
            if focus_programs:
                focus_in_data = [p for p in focus_programs if p in program_counts["Program"].values]
                if focus_in_data:
                    focus_data = []
                    for prog in focus_in_data:
                        row = program_counts[program_counts["Program"] == prog].iloc[0]
                        focus_data.append(f"**{prog}**: {row['Percentage']:.1f}% of clients ({row['ClientCount']} clients)")
                    
                    focus_stats = ", ".join(focus_data)
                    st.success(f"📊 **Focus Program Usage**: {focus_stats}")
        else:
            st.info("No program usage data available.")
    
    # TAB 2: PROGRAM TRANSITIONS
    with tab2:
        st.markdown("### Program-to-Program Transitions")
        st.markdown("""
        This analysis shows how clients move between programs in your system.
        Each transition represents a client moving from one program to another based on 
        enrollment start dates. Understanding these transitions helps identify common pathways
        and potential gaps in coordination.
        """)
        
        if not transitions.empty and len(transitions) > 0:
            # Create a combined label for visualization
            top_count = min(10, len(transitions))
            top_transitions = transitions.head(top_count).copy()
            
            top_transitions["Transition"] = top_transitions.apply(
                lambda x: f"{x['FromProgram']} → {x['ToProgram']}", axis=1
            )
            
            # Create horizontal bar chart for transitions
            fig = px.bar(
                top_transitions,
                x="Percentage",
                y="Transition",
                orientation="h",
                text="Percentage",
                labels={"Transition": "Program Transition", "Percentage": "% of All Transitions"},
                title="Common Program-to-Program Transitions",
                color="ClientCount",
                color_continuous_scale="Blues",
                hover_data=["ClientCount"]
            )
            
            # Format text
            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )
            
            # Style the chart
            fig = apply_chart_style(
                fig,
                height=450,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top transition insight
            if len(top_transitions) > 0:
                top_from = top_transitions.iloc[0]["FromProgram"]
                top_to = top_transitions.iloc[0]["ToProgram"]
                top_count = top_transitions.iloc[0]["ClientCount"]
                top_pct = top_transitions.iloc[0]["Percentage"]
                
                st.success(f"📊 **Key Observation**: The most common transition is from **{top_from}** to **{top_to}** ({top_count} clients, {top_pct:.1f}% of all transitions). This suggests an established pathway between these programs.")
                
                # Add more context if applicable
                if top_from == top_to and not deduplicate_sequences:
                    st.info("""
                    ℹ️ **Note**: The top transition shows clients moving from a program to the same program type.
                    This could represent clients being re-enrolled in the same program type after a previous enrollment ended.
                    To focus on transitions between different program types, try checking "Deduplicate consecutive program entries".
                    """)
                
                # If focus programs are selected, highlight relevant transitions
                if focus_programs:
                    # Find transitions involving focus programs
                    focus_transitions = []
                    for prog in focus_programs:
                        # Transitions FROM focus program
                        from_trans = transitions[transitions["FromProgram"] == prog]
                        if not from_trans.empty:
                            top_from_dest = from_trans.iloc[0]["ToProgram"]
                            top_from_count = from_trans.iloc[0]["ClientCount"]
                            top_from_pct = from_trans.iloc[0]["Percentage"]
                            focus_transitions.append(f"From **{prog}** to **{top_from_dest}** ({top_from_count} clients, {top_from_pct:.1f}%)")
                        
                        # Transitions TO focus program
                        to_trans = transitions[transitions["ToProgram"] == prog]
                        if not to_trans.empty:
                            top_to_source = to_trans.iloc[0]["FromProgram"]
                            top_to_count = to_trans.iloc[0]["ClientCount"]
                            top_to_pct = to_trans.iloc[0]["Percentage"]
                            focus_transitions.append(f"From **{top_to_source}** to **{prog}** ({top_to_count} clients, {top_to_pct:.1f}%)")
                    
                    if focus_transitions:
                        st.success(f"📊 **Focus Program Transitions**: Most common transitions involving selected programs:\n" + "\n".join([f"- {t}" for t in focus_transitions[:3]]))
            
            # Show full table with explanations
            with st.expander("Complete Transition Data", expanded=False):
                # Create display table
                display_table = transitions.copy()
                
                # Rename columns for clarity
                display_table = display_table.rename(columns={
                    "FromProgram": "From Program",
                    "ToProgram": "To Program",
                    "ClientCount": "Number of Clients",
                    "Percentage": "% of All Transitions"
                })
                
                # Format percentages
                display_table["% of All Transitions"] = display_table["% of All Transitions"].apply(lambda x: f"{x:.1f}%")
                
                # Display datatable
                st.dataframe(
                    display_table[["From Program", "To Program", "Number of Clients", "% of All Transitions"]],
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("""
                **Understanding this data:**
                
                - Each row represents clients moving from one program type to another
                - Transitions are based on program start dates (chronological order)
                - Strong transitions (high numbers) indicate established pathways
                - Low transitions may indicate gaps in coordination or referral processes
                """)
                
            # Create a Sankey diagram for program flow
            if len(transitions) >= 3 and len(transitions) <= 20:
                st.markdown("### Program Flow Visualization")
                st.markdown("""
                This Sankey diagram shows how clients flow between programs. Wider lines indicate
                more clients following that path. The diagram is based on the chronological order 
                of program enrollments (by start date) for each client.
                """)
                
                # Prepare data for Sankey diagram
                all_programs = list(set(transitions["FromProgram"].tolist() + transitions["ToProgram"].tolist()))
                program_indices = {prog: i for i, prog in enumerate(all_programs)}
                
                # Create source, target, and value lists
                sources = [program_indices[prog] for prog in transitions["FromProgram"]]
                targets = [program_indices[prog] for prog in transitions["ToProgram"]]
                values = transitions["ClientCount"].tolist()
                
                # Create Sankey data
                sankey_data = dict(
                    type='sankey',
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_programs,
                        color=["rgba(31, 119, 180, 0.8)"] * len(all_programs)
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=["rgba(31, 119, 180, 0.4)"] * len(sources)
                    )
                )
                
                # Create figure
                fig = go.Figure(data=[sankey_data])
                
                # Update layout
                fig.update_layout(
                    title_text="Client Flow Between Programs",
                    font_size=10,
                    height=500,
                    margin=dict(l=0, r=0, t=30, b=5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a clear explanation of the diagram
                st.info("""
                **Reading the Sankey Diagram**: 
                
                This diagram shows how clients move between programs in chronological order:
                - Programs are shown as nodes (boxes)
                - Lines show clients moving from one program to another
                - Thicker lines represent more clients following that pathway
                - The diagram follows the sequence of enrollments based on start dates
                """)
            
            # Add summary of what this data means
            st.markdown("""
            **Why this matters**: Understanding program transitions reveals how clients navigate through your system.
            Strong patterns indicate established pathways, while missing or weak transitions may suggest
            opportunities to improve coordination between programs.
            """)
        else:
            st.info("No transition data available or no clients used multiple programs.")
    
    # TAB 3: ENROLLMENT PATTERNS
    with tab3:
        st.markdown("### Enrollment Pattern Analysis")
        
        # Multi-enrollment metrics
        multi_count = metrics.get("multi_enrollment_count", 0)
        multi_pct = metrics.get("multi_enrollment_pct", 0.0)
        concurrent_count = metrics.get("concurrent_enrollment_count", 0)
        concurrent_pct = metrics.get("concurrent_enrollment_pct", 0.0)
        
        # Create metrics display
        col1, col2 = st.columns(2)
        
        col1.metric(
            "Clients with Multiple Enrollments",
            f"{multi_pct:.1f}%",
            help="Percentage of clients who used more than one program during the reporting period"
        )
        
        col2.metric(
            "Clients with Concurrent Enrollments",
            f"{concurrent_pct:.1f}%",
            help="Percentage of clients who were enrolled in multiple programs at the same time"
        )
        
        # Add insight about enrollment patterns
        if multi_pct > 0:
            if multi_pct < 20:
                st.info(f"📊 **Enrollment Pattern Observation**: Only {multi_pct:.1f}% of clients had multiple enrollments. Your system primarily uses single-program interventions, with limited service coordination.")
            elif multi_pct > 60:
                st.success(f"📊 **Enrollment Pattern Observation**: {multi_pct:.1f}% of clients had multiple enrollments, indicating strong service coordination and a progressive engagement approach.")
            else:
                st.info(f"📊 **Enrollment Pattern Observation**: {multi_pct:.1f}% of clients had multiple enrollments, showing moderate service coordination across your system.")
        
        # Concurrent enrollment analysis
        if concurrent_pct > 0 and not concurrent_pairs.empty and len(concurrent_pairs) > 0:
            st.markdown("### Concurrent Program Enrollments")
            st.markdown("""
            This analysis shows which program types clients were enrolled in simultaneously.
            Concurrent enrollments may indicate complementary services or potential duplication.
            """)
            
            # Limit to top pairs for visualization
            top_pairs = min(8, len(concurrent_pairs))
            display_pairs = concurrent_pairs.head(top_pairs).copy()
            
            # Create a combined label for visualization
            display_pairs["ProgramPair"] = display_pairs.apply(
                lambda x: f"{x['Program1']} + {x['Program2']}", axis=1
            )
            
            # Create horizontal bar chart for program pairs
            fig = px.bar(
                display_pairs,
                x="Percentage",
                y="ProgramPair",
                orientation="h",
                text="Percentage",
                labels={"ProgramPair": "Program Combination", "Percentage": "% of Concurrent Clients"},
                title="Common Program Combinations Used Simultaneously",
                color="ClientCount",
                color_continuous_scale="Blues",
                hover_data=["ClientCount"]
            )
            
            # Format text
            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )
            
            # Style the chart
            fig = apply_chart_style(
                fig,
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insight about concurrent enrollments
            if len(display_pairs) > 0:
                top_pair = display_pairs.iloc[0]["ProgramPair"]
                top_pair_pct = display_pairs.iloc[0]["Percentage"]
                
                st.info(f"""
                📊 **Concurrent Enrollment Observation**: The most common combination of simultaneous programs is **{top_pair}** 
                ({top_pair_pct:.1f}% of clients with concurrent enrollments). These programs may offer complementary 
                services that work well together.
                """)
                
            # Add guidance on interpreting this data
            st.markdown("""
            **Understanding concurrent enrollments**:
            
            - **Complementary Services**: Some program combinations are designed to work together 
              (e.g., outreach + shelter, or RRH + services only)
            
            - **Potential Duplication**: Other combinations might indicate service duplication 
              (e.g., two similar housing programs)
            
            - **System Coordination**: High rates of concurrent enrollment suggest coordination 
              between programs serving the same clients
            """)
        
        # Distribution of enrollment counts
        if "enrollment_counts" in metrics and metrics["enrollment_counts"]:
            enrollment_counts_data = pd.DataFrame.from_dict(metrics["enrollment_counts"])
            if not enrollment_counts_data.empty:
                # Calculate the distribution
                count_distribution = (
                    enrollment_counts_data
                    .groupby("EnrollmentCount")
                    .size()
                    .reset_index(name="ClientCount")
                )
                
                count_distribution["Percentage"] = (count_distribution["ClientCount"] / count_distribution["ClientCount"].sum() * 100).round(1)
                
                # Sort by enrollment count
                count_distribution = count_distribution.sort_values("EnrollmentCount")
                
                st.markdown("### Number of Programs Used Per Client")
                st.markdown("""
                This chart shows the distribution of how many program enrollments clients had during the reporting period.
                """)
                
                # Create bar chart
                fig = px.bar(
                    count_distribution,
                    x="EnrollmentCount",
                    y="Percentage",
                    text="Percentage",
                    labels={"EnrollmentCount": "Number of Enrollments", "Percentage": "% of Clients"},
                    title="Distribution of Enrollments Per Client",
                    color="ClientCount",
                    color_continuous_scale="Blues",
                    hover_data=["ClientCount"]
                )
                
                # Update hover template
                fig.update_traces(
                    hovertemplate="<b>%{x} enrollments</b><br>" +
                                  "Clients: %{customdata[0]}<br>" +
                                  "Percentage: %{y:.1f}%<extra></extra>"
                )
                
                # Format text
                fig.update_traces(
                    texttemplate="%{y:.1f}%",
                    textposition="outside"
                )
                
                # Style the chart
                fig = apply_chart_style(
                    fig,
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add insight about program usage distribution
                if len(count_distribution) > 1:
                    single_enroll_pct = count_distribution[count_distribution["EnrollmentCount"] == 1]["Percentage"].values[0] if 1 in count_distribution["EnrollmentCount"].values else 0
                    max_enrollments = count_distribution["EnrollmentCount"].max()
                    
                    if single_enroll_pct > 70:
                        st.info(f"📊 **Enrollment Pattern Observation**: {single_enroll_pct:.1f}% of clients had only one enrollment during the reporting period, suggesting that most clients' needs are met by a single program.")
                    elif max_enrollments >= 4:
                        high_usage_pct = count_distribution[count_distribution["EnrollmentCount"] >= 4]["ClientCount"].sum() / count_distribution["ClientCount"].sum() * 100
                        st.info(f"📊 **Enrollment Pattern Observation**: {high_usage_pct:.1f}% of clients had 4 or more enrollments, suggesting multiple interventions are being provided to address client needs.")