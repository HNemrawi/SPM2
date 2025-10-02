"""Data loading and preprocessing with intelligent column mapping and duplicate detection."""

import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chardet
import pandas as pd
import streamlit as st

from src.ui.factories.html import html_factory
from src.ui.themes.theme import theme


class DataLoadError(Exception):
    pass


class ColumnMapper:
    """Maps HMIS export column variations to standardized names."""

    STANDARD_MAPPINGS = {
        "UniqueIdentifier": [
            "Clients Unique Identifier",
            "Unique Identifier",
        ],
        "ClientID": [
            "Clients Client ID",
            "Client ID",
        ],
        "DOB": [
            "Clients Date of Birth Date",
            "Date of Birth",
            "Date of Birth Date",
            "DOB",
        ],
        "Gender": [
            "Clients Gender",
            "Gender",
        ],
        "RaceEthnicity": [
            "Clients Race and Ethnicity",
            "Race/Ethnicity",
            "Race",
            "Ethnicity",
            "Race and Ethnicity",
        ],
        "VeteranStatus": [
            "Clients Veteran Status",
            "Veteran Status",
            "Veteran",
        ],
        "AgeTieratEntry": [
            "Entry Screen Age Tier",
            "Age Tier",
            "Age at Entry",
        ],
        "HasIncome": [
            "Entry Screen Income from any Source",
            "Income",
            "Has Income",
            "Income from any Source",
        ],
        "HasDisability": [
            "Entry Screen Any Disability",
            "Disability",
            "Has Disability",
            "Any Disability",
        ],
        "IsHeadOfHousehold": [
            "Entry Screen Head of Household (Yes / No)",
            "Head of Household",
            "Head of Household (Yes / No)",
        ],
        "CurrentlyFleeingDV": [
            "Entry Screen Currently Fleeing Domestic Violence",
            "Fleeing DV",
            "DV",
            "Currently Fleeing Domestic Violence",
        ],
        "PriorLivingCat": [
            "Entry Screen Prior Living Situation Category",
            "Prior Living",
            "Living Situation",
            "Prior Living Situation Category",
        ],
        "CHStartHousehold": [
            "Entry Screen Chronically Homeless Project Start - Household",
            "CH Start",
            "Chronic",
            "Chronically Homeless Project Start - Household",
        ],
        "EnrollmentID": [
            "Enrollments Enrollment ID",
            "Enrollment ID",
            "EnrollmentIdentifier",
        ],
        "HouseholdType": [
            "Enrollments Household Type",
            "Household Type",
            "HH Type",
        ],
        "HouseholdMoveInDate": [
            "Enrollments Household Move-In Date",
            "Move In Date",
            "MoveInDate",
            "Household Move-In Date",
            "Housing Move-in Date",
        ],
        "ProjectStart": [
            "Enrollments Project Start Date",
            "Project Start",
            "Entry Date",
            "StartDate",
            "Project Start Date",
        ],
        "ProjectExit": [
            "Enrollments Project Exit Date",
            "Project Exit",
            "Exit Date",
            "ExitDate",
            "Project Exit Date",
        ],
        "ReportingPeriodStartDate": [
            "Enrollments Reporting Period Start Date",
            "Report Start",
            "Reporting Period Start Date",
        ],
        "ReportingPeriodEndDate": [
            "Enrollments Reporting Period End Date",
            "Report End",
            "Reporting Period End Date",
        ],
        "AgencyName": [
            "Programs Agency Name",
            "Agency",
            "Organization",
            "Agency Name",
        ],
        "ProgramName": [
            "Programs Name",
            "Program Name",
            "Program",
            "Name",
        ],
        "LocalCoCCode": [
            "Program Custom Local CoC Code",
            "Local CoC",
            "LocalCoC",
            "Local CoC Code",
        ],
        "ProgramSetupCoC": [
            "Programs Program Setup CoC",
            "CoC",
            "CoCCode",
            "Program Setup CoC",
        ],
        "ProgramsContinuumProject": [
            "Programs Continuum Project",
            "Continuum Project",
        ],
        "ProjectTypeCode": [
            "Programs Project Type Code",
            "Project Type",
            "ProjectType",
            "Project Type Code",
        ],
        "SSVF_RRH": [
            "SSVF RRH",
            "SSVF_RRH",
            "SSVF-RRH",
        ],
        "ExitDestinationCat": [
            "Update/Exit Screen Destination Category",
            "Exit Category",
            "Destination Category",
        ],
        "ExitDestination": [
            "Update/Exit Screen Destination",
            "Exit Destination",
            "Destination",
        ],
    }

    @classmethod
    @st.cache_data(show_spinner=False)
    def create_mapping(cls, _columns: List[str]) -> Dict[str, str]:
        columns = _columns
        mapping = {}
        columns_lower = {col.lower(): col for col in columns}

        for standard_name, variations in cls.STANDARD_MAPPINGS.items():
            for variation in variations:
                if variation in columns:
                    mapping[variation] = standard_name
                    break
                elif variation.lower() in columns_lower:
                    mapping[columns_lower[variation.lower()]] = standard_name
                    break

        return mapping


class DataValidator:
    """Validates HMIS data structure and content."""

    REQUIRED_COLUMNS = ["ClientID", "ProjectStart"]
    DATE_COLUMNS = [
        "ProjectStart",
        "ProjectExit",
        "DOB",
        "HouseholdMoveInDate",
        "ReportingPeriodStartDate",
        "ReportingPeriodEndDate",
    ]
    NUMERIC_COLUMNS = ["ClientID", "EnrollmentID"]
    CATEGORICAL_COLUMNS = [
        "Gender",
        "RaceEthnicity",
        "VeteranStatus",
        "ProjectTypeCode",
    ]

    @classmethod
    def validate_structure(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        issues = []

        missing_required = [
            col for col in cls.REQUIRED_COLUMNS if col not in df.columns
        ]
        if missing_required:
            issues.append(
                f"Missing required columns: {', '.join(missing_required)}"
            )

        if df.empty:
            issues.append("DataFrame is empty")

        if len(df.columns) < 5:
            issues.append(
                f"Too few columns ({len(df.columns)}), expected at least 5"
            )

        return len(issues) == 0, issues


class DuplicateAnalyzer:
    """Analyzes and reports duplicate EnrollmentID records."""

    @staticmethod
    def analyze_duplicates(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if "EnrollmentID" not in df.columns:
            return pd.DataFrame(), {
                "has_duplicates": False,
                "message": "EnrollmentID column not found",
            }

        # Vectorized duplicate detection
        duplicate_mask = df.duplicated(subset=["EnrollmentID"], keep=False)

        if not duplicate_mask.any():
            return pd.DataFrame(), {"has_duplicates": False, "count": 0}

        duplicates_df = df[duplicate_mask].copy()

        # Use sort_values with kind='stable' for consistent ordering
        duplicates_df = duplicates_df.sort_values(
            "EnrollmentID", kind="stable"
        )

        # Vectorized column variation detection
        varying_columns = []
        enrollment_groups = duplicates_df.groupby("EnrollmentID", sort=False)

        # Process only groups with multiple records
        multi_record_groups = enrollment_groups.filter(lambda x: len(x) > 1)

        if not multi_record_groups.empty:
            # Check variations for each column using vectorized operations
            for col in multi_record_groups.columns:
                if col != "EnrollmentID":
                    # Group by EnrollmentID and count unique values per group
                    unique_counts = multi_record_groups.groupby(
                        "EnrollmentID"
                    )[col].nunique(dropna=False)
                    # If any group has more than 1 unique value, column varies
                    if (unique_counts > 1).any():
                        varying_columns.append(col)

        # Count duplicate groups
        duplicate_count = df["EnrollmentID"].duplicated(keep=False).sum()
        unique_enrollment_count = df[duplicate_mask]["EnrollmentID"].nunique()

        analysis = {
            "has_duplicates": True,
            "total_duplicate_records": duplicate_count,
            "unique_enrollment_ids_with_duplicates": unique_enrollment_count,
            "varying_columns": varying_columns,
            "duplicate_df": duplicates_df,
        }

        return duplicates_df, analysis


@st.cache_data(show_spinner=False)
def detect_file_encoding(file_content: bytes, sample_size: int = 10000) -> str:
    sample = (
        file_content[:sample_size]
        if len(file_content) > sample_size
        else file_content
    )
    result = chardet.detect(sample)
    return result["encoding"] or "utf-8"


def show_duplicate_info(
    df: pd.DataFrame, analysis: Dict[str, Any]
) -> pd.DataFrame:
    metrics_html = f"""
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;'>
        <div style='
            background: {theme.colors.background};
            border: 1px solid {theme.colors.border};
            border-radius: 6px;
            padding: 1rem;
        '>
            <div style='color: {
        theme.colors.text_secondary}
        ; font-size: 0.85rem; margin-bottom: 0.25rem;'>
                Total Duplicate Records
            </div>
            <div style='color: {theme.colors.warning_dark}
        ; font-size: 1.5rem; font-weight: 600;'>
                {analysis['total_duplicate_records']:,}
            </div>
        </div>
        <div style='
            background: {theme.colors.background};
            border: 1px solid {theme.colors.border};
            border-radius: 6px;
            padding: 1rem;
        '>
            <div style='color: {theme.colors.text_secondary}
        ; font-size: 0.85rem; margin-bottom: 0.25rem;'>
                Unique EnrollmentIDs
            </div>
            <div style='color: {theme.colors.warning_dark}
        ; font-size: 1.5rem; font-weight: 600;'>
                {analysis['unique_enrollment_ids_with_duplicates']:,}
            </div>
        </div>
    </div>
    """

    st.html(
        html_factory.info_box(
            content=metrics_html,
            type="warning",
            title="Duplicate EnrollmentIDs Detected",
            icon="⚠️",
        )
    )

    if analysis["varying_columns"]:
        # Create formatted list of varying columns
        cols_list = "\n".join(
            [
                f"<li style='margin: 0.25rem 0;'><strong>{col}</strong></li>"
                for col in analysis["varying_columns"][:10]
            ]
        )
        if len(analysis["varying_columns"]) > 10:
            remaining_count = len(analysis["varying_columns"]) - 10
            cols_list += (
                f"<li style='margin: 0.25rem 0; font-style: italic;'>"
                f"... and {remaining_count} more columns</li>"
            )

        consistency_content = f"""
        <p>The following columns have variations among duplicate records:</p>
        <ul style='margin: 0.5rem 0 0 1rem; padding: 0;'>
            {cols_list}
        </ul>
        """

        st.html(
            html_factory.info_box(
                content=consistency_content,
                type="warning",
                title="Data Consistency Issue",
            )
        )

    # Deduplication options section
    st.html(html_factory.divider())

    st.html(
        html_factory.title(
            text="Choose an Action", level=3, margin="0 0 0.5rem 0"
        )
    )

    st.html(
        f"""
    <p style='color: {
            theme.colors.text_secondary}; font-size: 0.9rem; margin: 0 0 1rem 0;'>
        Select how to handle the duplicate records
    </p>
    """
    )

    col_action1, col_action2, col_action3 = st.columns(3)

    with col_action1:
        if st.button(
            "Keep All Records",
            width="stretch",
            type="primary",
            help="Continue with all records including duplicates",
            key="keep_all_records_button",
        ):
            st.session_state["dedup_action"] = "keep_all"
            st.success("Keeping all records including duplicates")
            return df

    with col_action2:
        if st.button(
            "Keep First Occurrence",
            width="stretch",
            type="secondary",
            help=(
                "Remove duplicates, keeping the first occurrence "
                "of each EnrollmentID"
            ),
            key="keep_first_occurrence_button",
        ):
            df_dedup = df.drop_duplicates(
                subset=["EnrollmentID"], keep="first"
            )
            removed = len(df) - len(df_dedup)
            # Batch state updates to avoid multiple reruns
            st.session_state.update(
                {"dedup_action": "keep_first", "df": df_dedup}
            )
            st.success(f"Removed {removed:,} duplicate records")
            st.rerun()

    with col_action3:
        if st.button(
            "Keep Last Occurrence",
            width="stretch",
            type="secondary",
            help="Remove duplicates, keeping the last occurrence of each EnrollmentID",
            key="keep_last_occurrence_button",
        ):
            df_dedup = df.drop_duplicates(subset=["EnrollmentID"], keep="last")
            removed = len(df) - len(df_dedup)
            # Batch state updates to avoid multiple reruns
            st.session_state.update(
                {"dedup_action": "keep_last", "df": df_dedup}
            )
            st.success(f"Removed {removed:,} duplicate records")
            st.rerun()

    # Show sample of duplicates - Clean presentation
    st.html(html_factory.divider())

    with st.expander("View Sample Duplicate Records", expanded=False):
        sample_size = min(50, len(analysis["duplicate_df"]))

        st.html(
            html_factory.info_box(
                content=(
                    f"Showing first {sample_size} of "
                    f"{len(analysis['duplicate_df']):,} duplicate records"
                ),
                type="info",
            )
        )

        # Display the dataframe
        st.dataframe(
            analysis["duplicate_df"].head(sample_size),
            width="stretch",
            height=300,
        )

        # Download button for all duplicates
        csv = analysis["duplicate_df"].to_csv(index=False)
        st.download_button(
            label="Download All Duplicate Records",
            data=csv,
            file_name=(
                f"duplicate_enrollments_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            ),
            mime="text/csv",
            type="secondary",
            key="download_duplicates_button",
        )

    return df


def _load_file_data(
    uploaded_file: BytesIO,
    encoding: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    file_extension = Path(uploaded_file.name).suffix.lower()

    with st.spinner(f"Loading {uploaded_file.name}..."):
        if file_extension == ".csv":
            if encoding is None:
                file_content = uploaded_file.read()
                encoding = detect_file_encoding(file_content)
                uploaded_file.seek(0)
            optimized_dtypes = {
                "Clients Client ID": "Int32",
                "Enrollments Enrollment ID": "Int32",
                "Programs Project Type Code": "category",
                "Clients Gender": "category",
                "Clients Race and Ethnicity": "category",
                "Clients Veteran Status": "category",
                "Programs Agency Name": "category",
                "Programs Name": "category",
                "Update/Exit Screen Destination Category": "category",
                "Update/Exit Screen Destination": "category",
                "Entry Screen Prior Living Situation Category": "category",
                "Entry Screen Head of Household (Yes / No)": "category",
                "Entry Screen Income from any Source": "category",
                "Entry Screen Any Disability": "category",
                "Enrollments Household Type": "category",
                "Programs Continuum Project": "category",
                "Program Custom Local CoC Code": "category",
                "Programs Program Setup CoC": "category",
            }

            read_params = {
                "low_memory": False,
                "encoding": encoding,
                "on_bad_lines": "warn",
                "dtype": optimized_dtypes,
                "engine": "c",
                "parse_dates": False,
            }

            if chunk_size:
                chunks = []
                for chunk in pd.read_csv(
                    uploaded_file, chunksize=chunk_size, **read_params
                ):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(uploaded_file, **read_params)

        else:
            raise DataLoadError(f"Unsupported file type: {file_extension}")

    return df


def _process_loaded_data(
    df: pd.DataFrame, validate: bool, parse_dates: bool
) -> pd.DataFrame:
    with st.spinner("Processing data..."):
        df = clean_columns(df)

        column_mapping = ColumnMapper.create_mapping(df.columns)
        if column_mapping:
            df = df.rename(columns=column_mapping)

        if parse_dates:
            df = parse_date_columns(df)

        df = standardize_data_types(df)
        df = add_derived_columns(df)
        if validate:
            is_valid, issues = DataValidator.validate_structure(df)
            if not is_valid:
                st.html(
                    html_factory.info_box(
                        content=f"Data validation: {'; '.join(issues)}",
                        type="warning",
                        icon="⚠️",
                    )
                )

    return df


def _handle_duplicate_detection(df: pd.DataFrame) -> None:
    if "EnrollmentID" in df.columns:
        duplicates_df, analysis = DuplicateAnalyzer.analyze_duplicates(df)
        if analysis.get("has_duplicates", False):
            st.session_state["duplicate_analysis"] = {
                "has_duplicates": analysis["has_duplicates"],
                "total_duplicate_records": analysis["total_duplicate_records"],
                "unique_enrollment_ids_with_duplicates": analysis[
                    "unique_enrollment_ids_with_duplicates"
                ],
                "varying_columns": analysis["varying_columns"],
                "duplicate_df": analysis["duplicate_df"],
            }


@st.cache_data(show_spinner=False)
def load_and_preprocess_data(
    uploaded_file: BytesIO,
    validate: bool = True,
    encoding: Optional[str] = None,
    parse_dates: bool = True,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Enhanced data loading with validation and error handling.

    Parameters:
        uploaded_file: File uploaded by user
        validate: Whether to validate data structure
        encoding: File encoding (auto-detected if None)
        parse_dates: Whether to parse date columns
        chunk_size: For processing large files in chunks

    Returns:
        Preprocessed DataFrame (including all duplicates)

    Raises:
        DataLoadError: If critical loading errors occur
    """
    try:
        # Validate input parameters
        if uploaded_file is None:
            st.error("No file provided for processing.")
            return pd.DataFrame()

        # Load raw data from file
        df = _load_file_data(uploaded_file, encoding, chunk_size)

        # Check for empty data early
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            st.error("The uploaded file is empty or invalid.")
            return pd.DataFrame()

        # Process the data
        df = _process_loaded_data(df, validate, parse_dates)

        # Final emptiness check
        if df.empty:
            st.error("No data available after processing.")
            return pd.DataFrame()

        # Handle duplicate detection
        _handle_duplicate_detection(df)

        return df

    except DataLoadError as e:
        st.error(f"Data loading error: {str(e)}")
        raise

    except pd.errors.EmptyDataError:
        st.error("The uploaded file contains no data.")
        return pd.DataFrame()

    except pd.errors.ParserError as e:
        st.error(f"Error parsing file: {str(e)}")
        st.info("Please check that your file is properly formatted.")
        return pd.DataFrame()

    except UnicodeDecodeError:
        st.error("Error decoding file. Please check the file encoding.")
        st.info("Try saving the file as UTF-8 or use a different encoding.")
        return pd.DataFrame()

    except MemoryError:
        st.error(
            "File is too large to process. Please try a smaller file or enable chunking."
        )
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Unexpected error loading file: {str(e)}")
        st.error("Please check your file format and try again.")

        with st.expander("Error details"):
            st.code(str(type(e).__name__) + ": " + str(e))
            st.code(traceback.format_exc())

        return pd.DataFrame()


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and remove empty columns"""
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

    empty_cols = df.columns[df.isna().all()]
    if not empty_cols.empty:
        df = df.drop(columns=empty_cols)
        st.info(f"Removed {len(empty_cols)} empty columns")

    return df


def parse_date_columns(
    df: pd.DataFrame, date_formats: Optional[List[str]] = None
) -> pd.DataFrame:
    """Parse date columns with optimized single-pass parsing"""
    if date_formats is None:
        date_formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%m-%d-%Y",
            "%d-%m-%Y",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ]

    for col in DataValidator.DATE_COLUMNS:
        if col in df.columns:
            # Use pandas' optimized datetime parsing with format inference
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False)

    return df


def standardize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate data types with optimization"""
    # Enable copy-on-write for memory efficiency
    with pd.option_context("mode.copy_on_write", True):
        # Convert numeric columns with optimized types
        for col in DataValidator.NUMERIC_COLUMNS:
            if col in df.columns:
                # Try Int32 first for memory efficiency
                try:
                    # Check if values fit in Int32 range
                    numeric_vals = pd.to_numeric(df[col], errors="coerce")
                    max_val = numeric_vals.max()
                    if pd.notna(max_val) and max_val < 2147483647:
                        df[col] = numeric_vals.astype("Int32")
                    else:
                        df[col] = numeric_vals.astype("Int64")
                except (ValueError, TypeError, AttributeError):
                    df[col] = pd.to_numeric(df[col], errors="coerce")

    # Extended list of categorical columns for better memory efficiency
    categorical_candidates = [
        "Gender",
        "RaceEthnicity",
        "VeteranStatus",
        "ProjectTypeCode",
        "AgencyName",
        "ProgramName",
        "HouseholdType",
        "IsHeadOfHousehold",
        "ExitDestination",
        "ExitDestinationCat",
        "PriorLivingCat",
        "CHStartHousehold",
        "HasIncome",
        "HasDisability",
        "CurrentlyFleeingDV",
        "AgeTieratEntry",
        "LocalCoCCode",
        "ProgramSetupCoC",
        "SSVF_RRH",
        "ProgramsContinuumProject",
    ]

    for col in categorical_candidates:
        if col in df.columns:
            # Convert to category if not already
            if df[col].dtype != "category":
                unique_ratio = (
                    df[col].nunique() / len(df) if len(df) > 0 else 1
                )
                # Use categorical for any column with < 50% unique values
                if unique_ratio < 0.5:
                    df[col] = df[col].astype("category")

    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add useful derived columns"""
    if "ProjectStart" in df.columns and "ProjectExit" in df.columns:
        df["IsActive"] = df["ProjectExit"].isna()
        df["LengthOfStay"] = (df["ProjectExit"] - df["ProjectStart"]).dt.days
        df["LengthOfStay"] = df["LengthOfStay"].clip(lower=0)

    if "DOB" in df.columns and "ProjectStart" in df.columns:
        df["AgeAtEntry"] = (
            (df["ProjectStart"] - df["DOB"]).dt.days / 365.25
        ).round(1)
        df["AgeAtEntry"] = df["AgeAtEntry"].clip(lower=0, upper=120)

    # Pre-convert common filter columns to strings for performance
    # These are frequently used in filter comparisons
    filter_columns = [
        "Gender",
        "RaceEthnicity",
        "VeteranStatus",
        "ProjectTypeCode",
        "AgencyName",
        "ProgramName",
        "ExitDestination",
        "ExitDestinationCat",
        "PriorLivingCat",
        "IsHeadOfHousehold",
        "HasIncome",
        "HasDisability",
        "LocalCoCCode",
        "ProgramSetupCoC",
        "ProgramsContinuumProject",
    ]

    for col in filter_columns:
        if col in df.columns:
            # Convert to string with proper handling of categorical types
            if df[col].dtype == "category":
                df[col + "_str"] = df[col].astype(str)
            else:
                df[col + "_str"] = df[col].astype(str)

    return df


def export_preprocessing_report(
    df: pd.DataFrame, filename: str = "data_quality_report.txt"
) -> str:
    """Generate detailed preprocessing report"""
    report = []
    report.append("DATA PREPROCESSING REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(
        f"\nData Shape: {df.shape[0]:,} rows × {df.shape[1]} columns"
    )
    report.append(
        f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    )

    # Add duplicate information if available
    if "EnrollmentID" in df.columns:
        duplicates_df, analysis = DuplicateAnalyzer.analyze_duplicates(df)
        if analysis.get("has_duplicates", False):
            report.append(
                f"\nDuplicate EnrollmentIDs: {analysis['total_duplicate_records']} records"
            )
            report.append(
                f"Unique EnrollmentIDs with duplicates: {analysis['unique_enrollment_ids_with_duplicates']}"
            )
            if analysis.get("varying_columns"):
                report.append(
                    f"Columns with variations: {', '.join(analysis['varying_columns'][:10])}"
                )

    return "\n".join(report)


__all__ = [
    "load_and_preprocess_data",
    "DataLoadError",
    "ColumnMapper",
    "DataValidator",
    "DuplicateAnalyzer",
    "show_duplicate_info",
    "clean_columns",
    "parse_date_columns",
    "standardize_data_types",
    "add_derived_columns",
    "detect_file_encoding",
    "export_preprocessing_report",
]
