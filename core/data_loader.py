"""
Data Loading & Preprocessing with Duplicate Detection
"""

import streamlit as st
import pandas as pd
from io import BytesIO
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import chardet
from pathlib import Path

class DataLoadError(Exception):
   """Custom exception for data loading errors"""
   pass

class ColumnMapper:
   """Centralized column mapping configuration"""
   
   STANDARD_MAPPINGS = {
       "UniqueIdentifier": ["Clients Unique Identifier", "Unique Identifier"],
       "ClientID": ["Clients Client ID", "Client ID"],
       "DOB": ["Clients Date of Birth Date", "Date of Birth", "Date of Birth Date", "DOB"],
       "Gender": ["Clients Gender", "Gender"],
       "RaceEthnicity": ["Clients Race and Ethnicity", "Race/Ethnicity", "Race", "Ethnicity"],
       "VeteranStatus": ["Clients Veteran Status", "Veteran Status", "Veteran"],
       "AgeTieratEntry": ["Entry Screen Age Tier", "Age Tier", "Age at Entry"],
       "HasIncome": ["Entry Screen Income from any Source", "Income", "Has Income"],
       "HasDisability": ["Entry Screen Any Disability", "Disability", "Has Disability"],
       "IsHeadOfHousehold": ["Entry Screen Head of Household (Yes / No)", "Head of Household", "Head of Household (Yes / No)"],
       "CurrentlyFleeingDV": ["Entry Screen Currently Fleeing Domestic Violence", "Fleeing DV", "DV"],
       "PriorLivingCat": ["Entry Screen Prior Living Situation Category", "Prior Living", "Living Situation"],
       "CHStartHousehold": ["Entry Screen Chronically Homeless Project Start - Household", "CH Start", "Chronic"],
       "EnrollmentID": ["Enrollments Enrollment ID", "Enrollment ID", "EnrollmentIdentifier"],
       "HouseholdType": ["Enrollments Household Type", "Household Type", "HH Type"],
       "HouseholdMoveInDate": ["Enrollments Household Move-In Date", "Move In Date", "MoveInDate"],
       "ProjectStart": ["Enrollments Project Start Date", "Project Start", "Entry Date", "StartDate"],
       "ProjectExit": ["Enrollments Project Exit Date", "Project Exit", "Exit Date", "ExitDate"],
       "ReportingPeriodStartDate": ["Enrollments Reporting Period Start Date", "Report Start"],
       "ReportingPeriodEndDate": ["Enrollments Reporting Period End Date", "Report End"],
       "AgencyName": ["Programs Agency Name", "Agency", "Organization"],
       "ProgramName": ["Programs Name", "Program Name", "Program"],
       "LocalCoCCode": ["Program Custom Local CoC Code", "Local CoC", "LocalCoC"],
       "ProgramSetupCoC": ["Programs Program Setup CoC", "CoC", "CoCCode"],
       "ProgramsContinuumProject": ["Programs Continuum Project", "Continuum Project"],
       "ProjectTypeCode": ["Programs Project Type Code", "Project Type", "ProjectType"],
       "SSVF_RRH": ["SSVF RRH", "SSVF_RRH", "SSVF-RRH"],
       "ExitDestinationCat": ["Update/Exit Screen Destination Category", "Exit Category", "Destination Category"],
       "ExitDestination": ["Update/Exit Screen Destination", "Exit Destination", "Destination"]
   }
   
   @classmethod
   def create_mapping(cls, columns: List[str]) -> Dict[str, str]:
       """Create column mapping based on available columns"""
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
   """Data validation utilities"""
   
   REQUIRED_COLUMNS = ["ClientID", "ProjectStart"]
   DATE_COLUMNS = ["ProjectStart", "ProjectExit", "DOB", "HouseholdMoveInDate", 
                   "ReportingPeriodStartDate", "ReportingPeriodEndDate"]
   NUMERIC_COLUMNS = ["ClientID", "EnrollmentID"]
   CATEGORICAL_COLUMNS = ["Gender", "RaceEthnicity", "VeteranStatus", "ProjectTypeCode"]
   
   @classmethod
   def validate_structure(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
       """Validate dataframe structure"""
       issues = []
       
       missing_required = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
       if missing_required:
           issues.append(f"Missing required columns: {', '.join(missing_required)}")
       
       if df.empty:
           issues.append("DataFrame is empty")
       
       if len(df.columns) < 5:
           issues.append(f"Too few columns ({len(df.columns)}), expected at least 5")
       
       return len(issues) == 0, issues

class DuplicateAnalyzer:
    """Analyze and handle duplicate EnrollmentIDs"""
    
    @staticmethod
    def analyze_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Analyze duplicates based on EnrollmentID
        
        Returns:
            Tuple of (duplicates_df, analysis_dict)
        """
        if "EnrollmentID" not in df.columns:
            return pd.DataFrame(), {"has_duplicates": False, "message": "EnrollmentID column not found"}
        
        # Find duplicate EnrollmentIDs
        duplicate_mask = df.duplicated(subset=["EnrollmentID"], keep=False)
        duplicates_df = df[duplicate_mask].copy()
        
        if duplicates_df.empty:
            return pd.DataFrame(), {"has_duplicates": False, "count": 0}
        
        # Sort by EnrollmentID for better viewing
        duplicates_df = duplicates_df.sort_values("EnrollmentID")
        
        # Analyze what columns differ among duplicates
        varying_columns = []
        enrollment_groups = duplicates_df.groupby("EnrollmentID")
        
        for enrollment_id, group in enrollment_groups:
            if len(group) > 1:
                # Check each column for variations
                for col in group.columns:
                    if col != "EnrollmentID":
                        unique_values = group[col].nunique(dropna=False)
                        if unique_values > 1:
                            if col not in varying_columns:
                                varying_columns.append(col)
        
        # Count duplicate groups
        duplicate_count = df["EnrollmentID"].duplicated(keep=False).sum()
        unique_enrollment_count = df[duplicate_mask]["EnrollmentID"].nunique()
        
        analysis = {
            "has_duplicates": True,
            "total_duplicate_records": duplicate_count,
            "unique_enrollment_ids_with_duplicates": unique_enrollment_count,
            "varying_columns": varying_columns,
            "duplicate_df": duplicates_df
        }
        
        return duplicates_df, analysis

@st.cache_data(show_spinner=False, ttl=3600)
def detect_file_encoding(file_content: bytes, sample_size: int = 10000) -> str:
   """Detect file encoding using chardet"""
   sample = file_content[:sample_size] if len(file_content) > sample_size else file_content
   result = chardet.detect(sample)
   return result['encoding'] or 'utf-8'

def show_duplicate_info(df: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
    """
    Display duplicate information in an expander with deduplication options
    
    Returns:
        Processed DataFrame (deduplicated if user chooses, original otherwise)
    """
    with st.expander(f"⚠️ Duplicate EnrollmentIDs Found ({analysis['total_duplicate_records']} records)", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Duplicate Records", f"{analysis['total_duplicate_records']:,}")
            st.metric("Unique EnrollmentIDs with Duplicates", f"{analysis['unique_enrollment_ids_with_duplicates']:,}")
        
        with col2:
            if analysis['varying_columns']:
                st.write("**Columns with variations:**")
                # Show first 10 varying columns
                for col in analysis['varying_columns'][:10]:
                    st.write(f"• {col}")
                if len(analysis['varying_columns']) > 10:
                    st.write(f"• ... and {len(analysis['varying_columns']) - 10} more columns")
        
        # Deduplication options
        st.write("---")
        st.write("**Choose an action:**")
        
        col_action1, col_action2, col_action3 = st.columns(3)
        
        with col_action1:
            if st.button("✅ Keep As Is", use_container_width=True, type="primary"):
                st.session_state['dedup_action'] = 'keep_all'
                st.success("Keeping all records including duplicates")
                return df
        
        with col_action2:
            if st.button("🔧 Deduplicate (Keep First)", use_container_width=True):
                st.session_state['dedup_action'] = 'keep_first'
                df_dedup = df.drop_duplicates(subset=["EnrollmentID"], keep="first")
                removed = len(df) - len(df_dedup)
                st.success(f"Removed {removed:,} duplicate records (kept first occurrence)")
                st.session_state["df"] = df_dedup
                st.rerun()
        
        with col_action3:
            if st.button("🔧 Deduplicate (Keep Last)", use_container_width=True):
                st.session_state['dedup_action'] = 'keep_last'
                df_dedup = df.drop_duplicates(subset=["EnrollmentID"], keep="last")
                removed = len(df) - len(df_dedup)
                st.success(f"Removed {removed:,} duplicate records (kept last occurrence)")
                st.session_state["df"] = df_dedup
                st.rerun()
        
        # Show sample of duplicates
        st.write("---")
        st.write("**Sample Duplicate Records:**")
        sample_size = min(50, len(analysis['duplicate_df']))
        st.dataframe(
            analysis['duplicate_df'].head(sample_size),
            use_container_width=True,
            height=300
        )
        
        # Download button for all duplicates
        csv = analysis['duplicate_df'].to_csv(index=False)
        st.download_button(
            label="📥 Download All Duplicate Records",
            data=csv,
            file_name=f"duplicate_enrollments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    return df

@st.cache_data(show_spinner=False)
def load_and_preprocess_data(
    uploaded_file: BytesIO,
    validate: bool = True,
    encoding: Optional[str] = None,
    parse_dates: bool = True,
    chunk_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Enhanced data loading with validation and error handling.
    Now includes duplicate detection for informational purposes only.
    
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
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        with st.spinner(f"Loading {uploaded_file.name}..."):
            if file_extension == ".csv":
                if encoding is None:
                    file_content = uploaded_file.read()
                    encoding = detect_file_encoding(file_content)
                    uploaded_file.seek(0)
                
                read_params = {
                    "low_memory": False,
                    "encoding": encoding,
                    "on_bad_lines": 'warn',
                    "dtype": str
                }
                
                if chunk_size:
                    chunks = []
                    for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size, **read_params):
                        chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(uploaded_file, **read_params)
            
            elif file_extension in [".xlsx", ".xls"]:
                df = pd.read_excel(
                    uploaded_file,
                    engine='openpyxl' if file_extension == ".xlsx" else 'xlrd',
                    dtype=str
                )
            
            else:
                raise DataLoadError(f"Unsupported file type: {file_extension}")
        
        # Check if dataframe is empty
        if df.empty:
            st.error("The uploaded file is empty.")
            return pd.DataFrame()
        
        with st.spinner("Processing data..."):
            # Clean columns
            df = clean_columns(df)
            
            # Apply column mapping
            column_mapping = ColumnMapper.create_mapping(df.columns)
            if column_mapping:
                df = df.rename(columns=column_mapping)
                st.info(f"Applied column mapping for {len(column_mapping)} columns")
            
            # Parse dates if requested
            if parse_dates:
                df = parse_date_columns(df)
            
            # Standardize data types
            df = standardize_data_types(df)
            
            # Add derived columns
            df = add_derived_columns(df)
            
            # Validate if requested
            if validate:
                is_valid, issues = DataValidator.validate_structure(df)
                if not is_valid:
                    st.warning(f"Data validation issues: {'; '.join(issues)}")
                    # Don't fail on validation issues, just warn
        
        # Final check to ensure we have data
        if df.empty:
            st.error("No data available after processing.")
            return pd.DataFrame()
        
        # Success message
        st.success(f"Successfully loaded {len(df):,} records from {uploaded_file.name}")
        
        # Check for duplicates and store info if found
        if "EnrollmentID" in df.columns:
            duplicates_df, analysis = DuplicateAnalyzer.analyze_duplicates(df)
            if analysis.get("has_duplicates", False):
                # Store duplicate info in session state for display outside spinner
                # Make sure to store the analysis dict without the DataFrame to avoid serialization issues
                st.session_state['duplicate_analysis'] = {
                    "has_duplicates": analysis["has_duplicates"],
                    "total_duplicate_records": analysis["total_duplicate_records"],
                    "unique_enrollment_ids_with_duplicates": analysis["unique_enrollment_ids_with_duplicates"],
                    "varying_columns": analysis["varying_columns"],
                    "duplicate_df": analysis["duplicate_df"]
                }
        
        return df
    
    except DataLoadError as e:
        # Re-raise custom errors
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
        st.error("File is too large to process. Please try a smaller file or enable chunking.")
        return pd.DataFrame()
    
    except Exception as e:
        # Log the full error for debugging
        st.error(f"Unexpected error loading file: {str(e)}")
        st.error("Please check your file format and try again.")
        
        # Show more details in an expander for debugging
        with st.expander("Error details"):
            st.code(str(type(e).__name__) + ": " + str(e))
            import traceback
            st.code(traceback.format_exc())
        
        return pd.DataFrame()

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
   """Clean column names and remove empty columns"""
   df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
   
   empty_cols = df.columns[df.isna().all()]
   if not empty_cols.empty:
       df = df.drop(columns=empty_cols)
       st.info(f"Removed {len(empty_cols)} empty columns")
   
   return df

def parse_date_columns(
   df: pd.DataFrame,
   date_formats: Optional[List[str]] = None
) -> pd.DataFrame:
   """Parse date columns with multiple format support"""
   if date_formats is None:
       date_formats = [
           "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
           "%Y/%m/%d", "%m-%d-%Y", "%d-%m-%Y",
           "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"
       ]
   
   for col in DataValidator.DATE_COLUMNS:
       if col in df.columns:
           for fmt in date_formats:
               try:
                   df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                   break
               except:
                   continue
           else:
               df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
   
   return df

def standardize_data_types(df: pd.DataFrame) -> pd.DataFrame:
   """Convert columns to appropriate data types"""
   for col in DataValidator.NUMERIC_COLUMNS:
       if col in df.columns:
           df[col] = pd.to_numeric(df[col], errors='coerce')
   
   for col in DataValidator.CATEGORICAL_COLUMNS:
       if col in df.columns:
           df[col] = df[col].astype('category')
   
   return df

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
   """Add useful derived columns"""
   if "ProjectStart" in df.columns and "ProjectExit" in df.columns:
       df["IsActive"] = df["ProjectExit"].isna()
       df["LengthOfStay"] = (df["ProjectExit"] - df["ProjectStart"]).dt.days
       df["LengthOfStay"] = df["LengthOfStay"].clip(lower=0)
   
   if "DOB" in df.columns and "ProjectStart" in df.columns:
       df["AgeAtEntry"] = ((df["ProjectStart"] - df["DOB"]).dt.days / 365.25).round(1)
       df["AgeAtEntry"] = df["AgeAtEntry"].clip(lower=0, upper=120)
   
   return df

def export_preprocessing_report(
   df: pd.DataFrame,
   filename: str = "data_quality_report.txt"
) -> str:
   """Generate detailed preprocessing report"""
   report = []
   report.append("DATA PREPROCESSING REPORT")
   report.append("=" * 50)
   report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
   report.append(f"\nData Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
   report.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
   
   # Add duplicate information if available
   if "EnrollmentID" in df.columns:
       duplicates_df, analysis = DuplicateAnalyzer.analyze_duplicates(df)
       if analysis.get("has_duplicates", False):
           report.append(f"\nDuplicate EnrollmentIDs: {analysis['total_duplicate_records']} records")
           report.append(f"Unique EnrollmentIDs with duplicates: {analysis['unique_enrollment_ids_with_duplicates']}")
           if analysis.get('varying_columns'):
               report.append(f"Columns with variations: {', '.join(analysis['varying_columns'][:10])}")
   
   return "\n".join(report)

__all__ = [
   'load_and_preprocess_data',
   'DataLoadError',
   'ColumnMapper',
   'DataValidator',
   'DuplicateAnalyzer',
   'show_duplicate_info',
   'clean_columns',
   'parse_date_columns',
   'standardize_data_types',
   'add_derived_columns',
   'detect_file_encoding',
   'export_preprocessing_report'
]