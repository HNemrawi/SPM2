"""
Simplified session serializer for HMIS Data Analysis Suite.
"""

import hashlib
from typing import Any, Dict, Optional

import pandas as pd


class SessionSerializer:
    """Simple session serializer for data export/import."""

    @staticmethod
    def create_session_summary(session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of session data.

        Args:
            session_data: The session data to summarize

        Returns:
            Dictionary with summary information
        """
        # Handle v2.1+ format
        if "session_info" in session_data:
            session_info = session_data["session_info"]
            return {
                "name": session_info.get("name", "Unnamed Session"),
                "description": session_info.get("description", ""),
                "created_at": session_info.get("created_at", "Unknown"),
                "data_file": session_info.get("data_file", "Unknown"),
                "module": session_data.get("module", "Unknown"),
                "version": session_data.get("version", "Unknown"),
                "has_data": "data_info" in session_data,
                "configuration": session_data.get("configuration_summary", {}),
            }

        # Handle v2.0 format
        return {
            "timestamp": session_data.get("timestamp", "Unknown"),
            "version": session_data.get("version", "Unknown"),
            "selected_module": session_data.get("selected_module", "Not set"),
            "filters_count": len(session_data.get("filters", {})),
            "has_data": "data_info" in session_data,
            "data_reference": session_data.get("data_reference", "Unknown"),
        }

    @staticmethod
    def compute_df_hash(df: pd.DataFrame) -> str:
        """Compute hash of DataFrame for comparison."""
        if df is None or df.empty:
            return ""

        # Create hash from shape and column names
        hash_str = f"{df.shape}_{','.join(df.columns)}"
        return hashlib.md5(hash_str.encode()).hexdigest()

    @staticmethod
    def validate_import(
        session_data: Dict[str, Any], current_hash: Optional[str] = None
    ) -> list:
        """
        Validate session data for import.

        Args:
            session_data: The session data to validate
            current_hash: Optional hash of current data for comparison

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        if not isinstance(session_data, dict):
            return ["❌ Invalid session data format"]

        if "version" not in session_data:
            issues.append("⚠️ Missing version information")

        # Basic version compatibility check
        version = session_data.get("version", "")
        if version:
            major_version = version.split(".")[0]
            if major_version != "2":
                issues.append(
                    f"⚠️ Incompatible version: {version} (expected 2.x)"
                )

        # Check data file reference (v2.1+ format)
        if "session_info" in session_data:
            data_file = session_data["session_info"].get("data_file")
            if data_file and data_file != "Unknown":
                issues.append(f"ℹ️ Session was created with file: {data_file}")
        # Check old format
        elif "data_reference" in session_data:
            data_ref = session_data["data_reference"]
            if data_ref and data_ref != "Unknown":
                issues.append(f"ℹ️ Session was created with file: {data_ref}")

        # Validate state exists
        if "state" not in session_data and "filters" not in session_data:
            issues.append("⚠️ No configuration data found in session file")

        return issues
