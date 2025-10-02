"""
Centralized filter constants and configurations.

This module contains all filter-related constants that were previously
scattered across different modules, providing a single source of truth.
"""

from datetime import date

# Default date ranges used across modules
DEFAULT_CURRENT_START = date(2024, 10, 1)
DEFAULT_CURRENT_END = date(2025, 9, 30)
DEFAULT_PREVIOUS_START = date(2023, 10, 1)
DEFAULT_PREVIOUS_END = date(2024, 9, 30)

# Filter form configuration keys
FILTER_FORM_KEY = "filter_form"
CUSTOM_PREV_CHECKBOX_KEY = "custom_prev_checkbox"

# Filter categories with display names and column mappings
# This was moved from src/modules/dashboard/filters.py for centralization
FILTER_CATEGORIES = {
    "Program Filters": {
        "Program CoC": "ProgramSetupCoC",
        "Local CoC": "LocalCoCCode",
        "Project Type": "ProjectTypeCode",
        "Agency Name": "AgencyName",
        "Program Name": "ProgramName",
        "SSVF RRH": "SSVF_RRH",
        "Continuum Project": "ProgramsContinuumProject",
    },
    "Client Demographics": {
        "Head of Household": "IsHeadOfHousehold",
        "Household Type": "HouseholdType",
        "Race / Ethnicity": "RaceEthnicity",
        "Gender": "Gender",
        "Entry Age Tier": "AgeTieratEntry",
        "Has Income": "HasIncome",
        "Has Disability": "HasDisability",
    },
    "Housing Status": {
        "Prior Living Situation": "PriorLivingCat",
        "Chronic Homelessness Household": "CHStartHousehold",
        "Exit Destination Category": "ExitDestinationCat",
        "Exit Destination": "ExitDestination",
    },
    "Special Populations": {
        "Veteran Status": "VeteranStatus",
        "Currently Fleeing DV": "CurrentlyFleeingDV",
    },
}

# Common filter configurations used across modules
COMMON_FILTER_CONFIGS = {
    "spm2_exit_filters": {
        "title": "Exit Filters",
        "icon": "🚪",
        "filters": [
            {
                "key": "exit_coc",
                "column": "ProgramSetupCoC",
                "label": "Program CoC (Exit)",
                "help": "Continuum of Care for exit programs",
            },
            {
                "key": "exit_local_coc",
                "column": "LocalCoCCode",
                "label": "Local CoC (Exit)",
                "help": "Local CoC for exit programs",
            },
            {
                "key": "exit_agency",
                "column": "AgencyName",
                "label": "Agency (Exit)",
                "help": "Agency for exit programs",
            },
            {
                "key": "exit_program",
                "column": "ProgramName",
                "label": "Program (Exit)",
                "help": "Specific exit programs",
            },
            {
                "key": "exit_ssvf_rrh",
                "column": "SSVF_RRH",
                "label": "SSVF RRH (Exit)",
                "help": "SSVF RRH designation for exits",
            },
            {
                "key": "exit_project_type",
                "column": "ProjectTypeCode",
                "label": "Project Type (Exit)",
                "help": "HUD project type for exits",
            },
            {
                "key": "exit_destination_cat",
                "column": "ExitDestinationCat",
                "label": "Exit Destination Category",
                "help": "Category of exit destination",
            },
            {
                "key": "exit_destination",
                "column": "ExitDestination",
                "label": "Exit Destination",
                "help": "Specific exit destination",
            },
        ],
    },
    "spm2_return_filters": {
        "title": "Return Filters",
        "icon": "↩️",
        "filters": [
            {
                "key": "return_coc",
                "column": "ProgramSetupCoC",
                "label": "Program CoC (Return)",
                "help": "Continuum of Care for return programs",
            },
            {
                "key": "return_local_coc",
                "column": "LocalCoCCode",
                "label": "Local CoC (Return)",
                "help": "Local CoC for return programs",
            },
            {
                "key": "return_agency",
                "column": "AgencyName",
                "label": "Agency (Return)",
                "help": "Agency for return programs",
            },
            {
                "key": "return_program",
                "column": "ProgramName",
                "label": "Program (Return)",
                "help": "Specific return programs",
            },
            {
                "key": "return_ssvf_rrh",
                "column": "SSVF_RRH",
                "label": "SSVF RRH (Return)",
                "help": "SSVF RRH designation for returns",
            },
            {
                "key": "return_project_type",
                "column": "ProjectTypeCode",
                "label": "Project Type (Return)",
                "help": "HUD project type for returns",
            },
        ],
    },
    "global_filters": {
        "title": "Global Filters",
        "icon": "⚡",
        "filters": [
            {
                "key": "continuum_project",
                "column": "ProgramsContinuumProject",
                "label": "Continuum Project",
                "help": "Filter by continuum project designation",
            }
        ],
    },
    "entry_filters": {
        "title": "Entry Filters",
        "icon": "🚪",
        "filters": [
            {
                "key": "entry_coc",
                "column": "ProgramSetupCoC",
                "label": "Program CoC (Entry)",
                "help": "Continuum of Care for entry programs",
            },
            {
                "key": "entry_local_coc",
                "column": "LocalCoCCode",
                "label": "Local CoC (Entry)",
                "help": "Local CoC for entry programs",
            },
            {
                "key": "entry_agency",
                "column": "AgencyName",
                "label": "Agency (Entry)",
                "help": "Agency for entry programs",
            },
            {
                "key": "entry_program",
                "column": "ProgramName",
                "label": "Program (Entry)",
                "help": "Specific entry programs",
            },
            {
                "key": "entry_project_type",
                "column": "ProjectTypeCode",
                "label": "Project Type (Entry)",
                "help": "HUD project type for entries",
            },
        ],
    },
}

# Default filter states
DEFAULT_FILTER_STATE = {
    "current_period_start": DEFAULT_CURRENT_START,
    "current_period_end": DEFAULT_CURRENT_END,
    "previous_period_start": DEFAULT_PREVIOUS_START,
    "previous_period_end": DEFAULT_PREVIOUS_END,
    "selected_ph_destinations": [],
    "selected_filters": {},
}

# Filter validation rules
REQUIRED_COLUMNS = [
    "ClientID",
    "EnrollmentID",
    "ProjectStart",
    "ProjectTypeCode",
]

OPTIONAL_COLUMNS = [
    "ProjectExit",
    "ExitDestination",
    "ExitDestinationCat",
    "ProgramSetupCoC",
    "LocalCoCCode",
    "AgencyName",
    "ProgramName",
    "RaceEthnicity",
    "Gender",
    "AgeTieratEntry",
    "IsHeadOfHousehold",
    "HouseholdType",
    "HasIncome",
    "HasDisability",
    "PriorLivingCat",
    "VeteranStatus",
    "CurrentlyFleeingDV",
    "SSVF_RRH",
    "ProgramsContinuumProject",
    "CHStartHousehold",
]
