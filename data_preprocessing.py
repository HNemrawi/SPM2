"""
Data Loading & Preprocessing
----------------------------
Handles the user file upload, reading CSV/Excel, renaming columns, and parsing dates.
"""

import streamlit as st
import pandas as pd
from io import BytesIO

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
            "Clients Unique Identifier": "UniqueIdentifier",
            "Clients Client ID": "ClientID",
            "Clients Race and Ethnicity": "RaceEthnicity",
            "Clients Gender": "Gender",
            "Clients Date of Birth Date": "DOB",
            "Clients Veteran Status": "VeteranStatus",
            "Enrollments Enrollment ID": "EnrollmentID",
            "Entry Screen Income from any Source": "HasIncome",
            "Entry Screen Any Disability": "HasDisability",
            "Enrollments Household Type": "HouseholdType",
            "Entry Screen Chronically Homeless Project Start - Household": "CHStartHousehold",
            "Program Custom Local CoC Code": "LocalCoCCode",
            "Entry Screen Prior Living Situation Category": "PriorLivingCat",
            "Enrollments Project Start Date": "ProjectStart",
            "Enrollments Project Exit Date": "ProjectExit",
            "Programs Program Setup CoC": "ProgramSetupCoC",
            "Programs Project Type Code": "ProjectTypeCode",
            "Programs Name": "ProgramName",
            "Programs Agency Name": "AgencyName",
            "Update/Exit Screen Destination Category": "ExitDestinationCat",
            "Update/Exit Screen Destination": "ExitDestination",
            "Enrollments Household Move-In Date": "HouseholdMoveInDate",
            "Programs Continuum Project" : "ProgramsContinuumProject",
            "Enrollments Reporting Period Start Date" : "ReportingPeriodStartDate",
            "Enrollments Reporting Period End Date" : "ReportingPeriodEndDate",
            "Entry Screen Age Tier" : "AgeTieratEntry"
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
