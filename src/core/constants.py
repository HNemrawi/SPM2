"""Application-wide constants for HMIS analysis."""

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

PH_CATEGORY = "Permanent Housing Situations"

RETURN_PERIODS = {
    "< 6 Months": 180,
    "6–12 Months": 365,
    "12–24 Months": 730,
    "> 24 Months": float("inf"),
}

EXIT_COLUMNS = [
    "Exit_HasIncome",
    "Exit_HasDisability",
    "Exit_HouseholdType",
    "Exit_IsHeadOfHousehold",
    "Exit_CHStartHousehold",
    "Exit_CurrentlyFleeingDV",
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
    "Exit_SSVF_RRH",
    "Exit_ProgramsContinuumProject",
]

RETURN_COLUMNS = [
    "Return_HasIncome",
    "Return_HasDisability",
    "Return_HouseholdType",
    "Return_IsHeadOfHousehold",
    "Return_CHStartHousehold",
    "Return_CurrentlyFleeingDV",
    "Return_LocalCoCCode",
    "Return_PriorLivingCat",
    "Return_ProgramSetupCoC",
    "Return_ProjectTypeCode",
    "Return_AgencyName",
    "Return_ProgramName",
    "Return_ExitDestinationCat",
    "Return_ExitDestination",
    "Return_SSVF_RRH",
    "Return_ProgramsContinuumProject",
]

REQUIRED_COLUMNS = {
    "base": [
        "ClientID",
        "ProjectStart",
        "ProjectExit",
        "ProjectTypeCode",
        "EnrollmentID",
    ],
    "exit": ["ExitDestinationCat"],
    "return": [],
    "demographics": ["DOB", "Gender", "RaceEthnicity", "VeteranStatus"],
}

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
    "Return_ReportingPeriodEndDate",
]

COLUMNS_TO_RENAME = {
    "Exit_UniqueIdentifier": "UniqueIdentifier",
    "Exit_ClientID": "ClientID",
    "Exit_RaceEthnicity": "RaceEthnicity",
    "Exit_Gender": "Gender",
    "Exit_DOB": "DOB",
    "Exit_VeteranStatus": "VeteranStatus",
}

DEFAULT_PROJECT_TYPES = [
    "Street Outreach",
    "Emergency Shelter – Entry Exit",
    "Emergency Shelter – Night-by-Night",
    "Transitional Housing",
    "Safe Haven",
    "PH – Housing Only",
    "PH – Housing with Services (no disability required for entry)",
    "PH – Permanent Supportive Housing (disability required for entry)",
    "PH – Rapid Re-Housing",
]

DEMOGRAPHIC_DIMENSIONS = [
    ("Race / Ethnicity", "RaceEthnicity"),
    ("Gender", "Gender"),
    ("Entry Age Tier", "AgeTieratEntry"),
    ("Veteran Status", "VeteranStatus"),
    ("Has Disability", "HasDisability"),
    ("Has Income", "HasIncome"),
    ("Household Type", "HouseholdType"),
    ("Head of Household", "IsHeadOfHousehold"),
    ("Prior Living Situation", "PriorLivingCat"),
    ("Chronically Homeless", "CHStartHousehold"),
    ("Currently Fleeing DV", "CurrentlyFleeingDV"),
]

__all__ = [
    "PH_PROJECTS",
    "NON_HOMELESS_PROJECTS",
    "PH_CATEGORY",
    "RETURN_PERIODS",
    "EXIT_COLUMNS",
    "RETURN_COLUMNS",
    "REQUIRED_COLUMNS",
    "COLUMNS_TO_REMOVE",
    "COLUMNS_TO_RENAME",
    "DEFAULT_PROJECT_TYPES",
    "DEMOGRAPHIC_DIMENSIONS",
]
