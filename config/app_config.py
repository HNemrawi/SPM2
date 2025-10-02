"""Unified application configuration with dataclass-based settings."""

# Standard library imports
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third-party imports
import yaml

# Local imports
from src.core.constants import DEFAULT_PROJECT_TYPES

CONFIG_DIR = Path(__file__).parent
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"


class LayoutMode(str, Enum):
    WIDE = "wide"
    CENTERED = "centered"


class SidebarState(str, Enum):
    EXPANDED = "expanded"
    COLLAPSED = "collapsed"
    AUTO = "auto"


class DateFormat(str, Enum):
    ISO = "%Y-%m-%d"
    US = "%m/%d/%Y"
    EU = "%d/%m/%Y"


class TimeUnit(str, Enum):
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"


@dataclass
class PageConfig:
    """Page-level configuration."""

    title: str = "HMIS Data Analysis Suite"
    icon: str = ""
    layout: LayoutMode = LayoutMode.WIDE
    sidebar_state: SidebarState = SidebarState.EXPANDED
    show_footer: bool = True
    enable_analytics: bool = False
    theme: Optional[str] = None


@dataclass
class DateConfig:
    """Date and time configuration."""

    default_start_date: str = field(
        default_factory=lambda: (datetime.now() - timedelta(days=31)).strftime(
            "%Y-%m-%d"
        )
    )
    default_end_date: str = field(
        default_factory=lambda: (datetime.now() - timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
    )
    min_date: str = "2000-01-01"
    max_date: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d")
    )
    date_format: DateFormat = DateFormat.ISO
    display_format: str = "%B %d, %Y"
    default_lookback_days: int = 730
    max_lookback_days: int = 3650
    default_return_window: int = 730
    quick_periods: Dict[str, int] = field(
        default_factory=lambda: {
            "Last 30 days": 30,
            "Last 90 days": 90,
            "Last 6 months": 180,
            "Last year": 365,
            "Last 2 years": 730,
        }
    )


@dataclass
class DataConfig:
    """Data processing and validation configuration."""

    max_file_size_mb: int = 100
    allowed_file_types: List[str] = field(default_factory=lambda: ["csv"])
    chunk_size: int = 10000
    encoding_options: List[str] = field(
        default_factory=lambda: ["utf-8", "latin-1", "cp1252"]
    )
    required_columns: List[str] = field(
        default_factory=lambda: [
            "ClientID",
            "EnrollmentID",
            "ProjectStart",
            "ProjectExit",
            "ProjectTypeCode",
        ]
    )
    max_stay_years: int = 5
    min_age: int = 0
    max_age: int = 120
    missing_value_threshold: float = 0.5
    date_parse_formats: List[str] = field(
        default_factory=lambda: [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%m-%d-%Y",
            "%d-%m-%Y",
        ]
    )


@dataclass
class CacheConfig:
    """Caching and performance configuration."""

    data_cache_ttl: int = 3600
    analysis_cache_ttl: int = 1800
    chart_cache_ttl: int = 900
    max_cache_size_mb: int = 500
    max_cached_datasets: int = 10
    enable_caching: bool = True
    enable_parallel_processing: bool = True
    num_workers: int = field(
        default_factory=lambda: min(4, os.cpu_count() or 1)
    )


@dataclass
class AnalysisConfig:
    """Analysis-specific configuration."""

    min_group_size: int = 30
    min_cell_size: int = 5
    significance_level: float = 0.05
    confidence_level: float = 0.95
    default_project_types: List[str] = field(
        default_factory=lambda: DEFAULT_PROJECT_TYPES
    )
    ph_gap_days: int = 14
    same_day_as_one: bool = True
    demographic_groups: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "Race/Ethnicity": ["Race", "Ethnicity", "RaceEthnicity"],
            "Gender": ["Gender", "GenderCode"],
            "Age": ["AgeAtEntry", "AgeTier", "AgeTieratEntry"],
            "Veteran": ["VeteranStatus"],
            "Household": ["HouseholdType", "IsHeadOfHousehold"],
        }
    )


@dataclass
class UIConfig:
    """User interface configuration."""

    table_page_size: int = 20
    max_chart_points: int = 1000
    decimal_places: int = 1
    chart_height: int = 400
    chart_colors: List[str] = field(
        default_factory=lambda: [
            "#0066CC",
            "#059862",
            "#D97706",
            "#DC2626",
            "#7C3AED",
            "#0891B2",
            "#EC4899",
            "#6366F1",
        ]
    )
    loading_messages: List[str] = field(
        default_factory=lambda: [
            "Analyzing your data...",
            "Crunching the numbers...",
            "Preparing insights...",
            "Almost there...",
        ]
    )
    export_formats: List[str] = field(default_factory=lambda: ["csv", "json"])
    report_logo_path: Optional[str] = None


class ConfigLoader:
    """Configuration loader from YAML files."""

    @staticmethod
    def load_yaml(file_path: Path) -> Dict[str, Any]:
        if not file_path.exists():
            return {}

        with open(file_path, "r") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def merge_configs(base: Dict, override: Dict) -> Dict:
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                result[key] = value

        return result


class EnvironmentConfig:
    """Environment-specific configuration handling."""

    @staticmethod
    def get_env_var(
        key: str, default: Any = None, var_type: type = str
    ) -> Any:
        value = os.environ.get(key, default)

        if value is None:
            return None

        if var_type is bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif var_type is int:
            return int(value)
        elif var_type is float:
            return float(value)
        elif var_type is list:
            import json

            return json.loads(value) if isinstance(value, str) else value

        return value

    @classmethod
    def override_from_env(cls, config: Any) -> Any:
        prefix = "HMIS_"

        for field_name in config.__dataclass_fields__:
            env_key = f"{prefix}{field_name.upper()}"
            if env_key in os.environ:
                field_type = config.__dataclass_fields__[field_name].type
                setattr(
                    config,
                    field_name,
                    cls.get_env_var(env_key, var_type=field_type),
                )

        return config


class AppConfig:
    """Main application configuration container."""

    def __init__(self):
        yaml_config = (
            ConfigLoader.load_yaml(SETTINGS_FILE)
            if SETTINGS_FILE.exists()
            else {}
        )

        self.page = EnvironmentConfig.override_from_env(PageConfig())
        self.date = EnvironmentConfig.override_from_env(DateConfig())
        self.data = EnvironmentConfig.override_from_env(DataConfig())
        self.cache = EnvironmentConfig.override_from_env(CacheConfig())
        self.analysis = EnvironmentConfig.override_from_env(AnalysisConfig())
        self.ui = EnvironmentConfig.override_from_env(UIConfig())

        self._apply_yaml_config(yaml_config)
        self._validate()

    def _apply_yaml_config(self, yaml_config: Dict[str, Any]):
        for section_name, section_config in yaml_config.items():
            if hasattr(self, section_name) and isinstance(
                section_config, dict
            ):
                config_section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(config_section, key):
                        setattr(config_section, key, value)

    def _validate(self):
        try:
            start = datetime.strptime(self.date.default_start_date, "%Y-%m-%d")
            end = datetime.strptime(self.date.default_end_date, "%Y-%m-%d")
            if start > end:
                self.date.default_start_date, self.date.default_end_date = (
                    self.date.default_end_date,
                    self.date.default_start_date,
                )
        except ValueError:
            self.date = DateConfig()

        self.data.required_columns = list(set(self.data.required_columns))
        self.analysis.significance_level = max(
            0.001, min(0.5, self.analysis.significance_level)
        )
        self.ui.decimal_places = max(0, min(6, self.ui.decimal_places))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page": self.page.__dict__,
            "date": self.date.__dict__,
            "data": self.data.__dict__,
            "cache": self.cache.__dict__,
            "analysis": self.analysis.__dict__,
            "ui": self.ui.__dict__,
        }

    def update(self, section: str, **kwargs):
        if hasattr(self, section):
            config_section = getattr(self, section)
            for key, value in kwargs.items():
                if hasattr(config_section, key):
                    setattr(config_section, key, value)
            self._validate()
        else:
            raise ValueError(f"Unknown configuration section: {section}")


def get_date_range() -> tuple:
    return (
        datetime.strptime(config.date.default_start_date, "%Y-%m-%d"),
        datetime.strptime(config.date.default_end_date, "%Y-%m-%d"),
    )


def format_date(
    date: Union[str, datetime], format_type: str = "display"
) -> str:
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    if format_type == "display":
        return date.strftime(config.date.display_format)
    elif format_type == "iso":
        return date.strftime(DateFormat.ISO.value)
    else:
        return date.strftime(config.date.date_format.value)


def is_development() -> bool:
    return (
        EnvironmentConfig.get_env_var("HMIS_ENV", "production")
        == "development"
    )


config = AppConfig()
PAGE_TITLE = config.page.title
PAGE_ICON = config.page.icon
DEFAULT_LAYOUT = config.page.layout.value
SIDEBAR_STATE = config.page.sidebar_state.value
DEFAULT_START_DATE = config.date.default_start_date
DEFAULT_END_DATE = config.date.default_end_date
DEFAULT_DAYS_LOOKBACK = config.date.default_lookback_days
CACHE_TTL = config.cache.data_cache_ttl

__all__ = [
    "config",
    "AppConfig",
    "PageConfig",
    "DateConfig",
    "DataConfig",
    "CacheConfig",
    "AnalysisConfig",
    "UIConfig",
    "LayoutMode",
    "SidebarState",
    "DateFormat",
    "TimeUnit",
    "PAGE_TITLE",
    "PAGE_ICON",
    "DEFAULT_LAYOUT",
    "SIDEBAR_STATE",
    "DEFAULT_START_DATE",
    "DEFAULT_END_DATE",
    "DEFAULT_DAYS_LOOKBACK",
    "CACHE_TTL",
    "get_date_range",
    "format_date",
    "is_development",
]
