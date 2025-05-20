"""
Data Loading & Preprocessing
----------------------------
Handles the user file upload, reading CSV/Excel, renaming columns, and parsing dates.
"""

import streamlit as st
import pandas as pd
from io import BytesIO
from typing import Dict, List, Optional, Union

@st.cache_data(show_spinner=False)
def load_and_preprocess_data(uploaded_file: BytesIO) -> pd.DataFrame:
    """
    Load and preprocess a user-uploaded CSV/Excel file.
    
    - Reads CSV or Excel files.
    - Cleans column names.
    - Renames key columns.
    - Parses dates for specific columns.
    
    Parameters:
        uploaded_file (BytesIO): The file uploaded by the user.
    
    Returns:
        pd.DataFrame: Preprocessed data or an empty DataFrame if an error occurs.
    """
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, low_memory=False)
        else:
            df = pd.read_excel(uploaded_file)

        # Clean and rename columns for consistency.
        df.columns = df.columns.str.strip()
        rename_map = {
            # Client-level fields
            "Clients Unique Identifier": "UniqueIdentifier",
            "Clients Client ID": "ClientID",
            "Clients Date of Birth Date": "DOB",
            "Clients Gender": "Gender",
            "Clients Race and Ethnicity": "RaceEthnicity",
            "Clients Veteran Status": "VeteranStatus",
            # Entry screen fields
            "Entry Screen Age Tier": "AgeTieratEntry",
            "Entry Screen Income from any Source": "HasIncome",
            "Entry Screen Any Disability": "HasDisability",
            "Entry Screen Head of Household (Yes / No)": "IsHeadOfHousehold",
            "Entry Screen Currently Fleeing Domestic Violence": "CurrentlyFleeingDV",
            "Entry Screen Prior Living Situation Category": "PriorLivingCat",
            "Entry Screen Chronically Homeless Project Start - Household": "CHStartHousehold",
            # Enrollment fields
            "Enrollments Enrollment ID": "EnrollmentID",
            "Enrollments Household Type": "HouseholdType",
            "Enrollments Household Move-In Date": "HouseholdMoveInDate",
            "Enrollments Project Start Date": "ProjectStart",
            "Enrollments Project Exit Date": "ProjectExit",
            "Enrollments Reporting Period Start Date": "ReportingPeriodStartDate",
            "Enrollments Reporting Period End Date": "ReportingPeriodEndDate",
            # Program-level fields
            "Programs Agency Name": "AgencyName",
            "Programs Name": "ProgramName",
            "Program Custom Local CoC Code": "LocalCoCCode",
            "Programs Program Setup CoC": "ProgramSetupCoC",
            "Programs Continuum Project": "ProgramsContinuumProject",
            "Programs Project Type Code": "ProjectTypeCode",
            # Exit screen fields
            "Update/Exit Screen Destination Category": "ExitDestinationCat",
            "Update/Exit Screen Destination": "ExitDestination",
        }

        df = df.rename(columns=rename_map)

        # Parse date columns (with error handling)
        for col in ["ProjectStart", "ProjectExit", "DOB", "HouseholdMoveInDate", "ReportingPeriodStartDate", "ReportingPeriodEndDate"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()
