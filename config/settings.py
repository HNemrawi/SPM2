"""
Global application settings and configuration
"""

# Page configuration
PAGE_TITLE = "Homelessness Analysis Suite"
PAGE_ICON = "📈"
DEFAULT_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# Default dates for analysis
DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = "2025-01-31"

# Default lookback days
DEFAULT_DAYS_LOOKBACK = 730

# Cache settings - storage options
CACHE_TTL = 3600  # Time to live for cached data (in seconds)