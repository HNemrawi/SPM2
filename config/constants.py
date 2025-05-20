 
"""
Shared constants used across the application
"""

# Project type categories
PH_PROJECTS = {
    "PH – Housing Only",
    "PH – Housing with Services (no disability required for entry)",
    "PH – Permanent Supportive Housing (disability required for entry)",
    "PH – Rapid Re-Housing",
}

NON_HOMELESS_PROJECTS = {
    "Coordinated Entry",
    "Day Shelter",
    "Homelessness Prevention",
    "Other",
    "Services Only",
}

# Exit/Return categories
PH_CATEGORY = "Permanent Housing Situations"

# Return period categories
RETURN_PERIODS = {
    "< 6 Months": 180,
    "6–12 Months": 365, 
    "12–24 Months": 730,
    "> 24 Months": float('inf')
}

# Standard column groups for filtering
EXIT_COLUMNS = [
    "Exit_HasIncome",
    "Exit_HasDisability",
    "Exit_HouseholdType",
    "Exit_CHStartHousehold",
    "Exit_LocalCoCCode",
    "Exit_PriorLivingCat",
    "Exit_ProgramSetupCoC",
    "Exit_ProjectTypeCode",
    "Exit_AgencyName",
    "Exit_ProgramName",
    "Exit_ExitDestinationCat",
    "Exit_ExitDestination",
    "Exit_CustomProgramType",
    "Exit_AgeTieratEntry",
]

RETURN_COLUMNS = [
    "Return_HasIncome",
    "Return_HasDisability",
    "Return_HouseholdType",
    "Return_CHStartHousehold",
    "Return_LocalCoCCode",
    "Return_PriorLivingCat",
    "Return_ProgramSetupCoC",
    "Return_ProjectTypeCode",
    "Return_AgencyName",
    "Return_ProgramName",
    "Return_ExitDestinationCat",
    "Return_ExitDestination",
    "ReturnCategory",
    "Return_AgeTieratEntry",
]

# Required columns for analysis
REQUIRED_COLUMNS = {
    "base": ["ClientID", "ProjectStart", "ProjectExit", "ProjectTypeCode", "EnrollmentID"],
    "exit": ["ExitDestinationCat"],
    "return": [],
    "demographics": ["DOB", "Gender", "RaceEthnicity", "VeteranStatus"],
}

# Standard columns that may need cleaning when generating summary dataframes
COLUMNS_TO_REMOVE = [
    "Return_UniqueIdentifier",
    "Return_ClientID",
    "Return_RaceEthnicity",
    "Return_Gender",
    "Return_DOB",
    "Return_VeteranStatus",
    "Exit_ReportingPeriodStartDate",
    "Exit_ReportingPeriodEndDate",
    "Return_ReportingPeriodStartDate",
    "Return_ReportingPeriodEndDate"
]

COLUMNS_TO_RENAME = [
    "Exit_UniqueIdentifier", 
    "Exit_ClientID",
    "Exit_RaceEthnicity",
    "Exit_Gender",
    "Exit_DOB",
    "Exit_VeteranStatus"
]

# Default project types
DEFAULT_PROJECT_TYPES = [
    "Street Outreach",
    "Emergency Shelter – Entry Exit",
    "Emergency Shelter – Night-by-Night",
    "Transitional Housing",
    "Safe Haven",
    "PH – Housing Only",
    "PH – Housing with Services (no disability required for entry)",
    "PH – Permanent Supportive Housing (disability required for entry)",
    "PH – Rapid Re-Housing"
]