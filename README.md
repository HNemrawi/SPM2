# SPM2 - HMIS Data Analysis Suite

A professional Streamlit-based application for analyzing Homeless Management Information System (HMIS) data, providing comprehensive performance metrics, recidivism analysis, and demographic insights with an enhanced modern UI.

## 🌟 Features

### Core Analysis Modules

#### 📈 System Performance Measure 2 (SPM2)
- Analyzes returns to homelessness based on client exits to permanent housing
- Configurable lookback periods (days or months)
- Customizable return windows for tracking
- Detailed flow visualizations showing client pathways
- PH vs Non-PH exit comparisons with side-by-side metrics
- Network analysis of program transitions

#### ➡️ Inbound Recidivism Analysis
- Measures how many clients entering programs are returners
- Analyzes time between previous exit and current entry
- Tracks pathways between different program types
- Provides detailed client flow visualizations
- Breakdown analysis by demographics and program types
- Days to return distribution charts

#### ⬅️ Outbound Recidivism Analysis
- Tracks clients who exit programs and later return
- Compares permanent vs non-permanent housing outcomes
- Analyzes return patterns by demographics
- Visualizes client flow networks
- Top pathway analysis
- Statistical summaries with percentile breakdowns

#### 🏠 General Dashboard
- **Summary View**: Key metrics with period-over-period comparisons
- **Demographics**: Population breakdown with outcome analysis
- **Trends**: Time-series analysis with change detection
- **Length of Stay**: Duration patterns, disparities, and quality checks
- **Equity Analysis**: Identifies outcome disparities across demographic groups

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SPM2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📊 Data Requirements

### Required Columns
The application expects HMIS data in CSV or Excel format with these essential columns:

- **ClientID**: Unique client identifier
- **EnrollmentID**: Unique enrollment identifier (used for duplicate detection)
- **ProjectStart**: Entry/enrollment date
- **ProjectExit**: Exit date (can be null for active clients)
- **ProjectTypeCode**: HMIS project type (ES, TH, PSH, RRH, etc.)
- **ExitDestination**: Destination at exit

### Optional Columns for Enhanced Analysis
- **Gender**: Client gender
- **RaceEthnicity**: Race/ethnicity information
- **VeteranStatus**: Veteran status indicator
- **DOB**: Date of birth
- **HouseholdType**: Household composition
- **ProgramName**: Program/project name
- **AgencyName**: Agency/organization name
- **ProgramsContinuumProject**: Continuum project participation
- **LocalCoCCode**: Local Continuum of Care code

### Data Quality Features
- **Automatic duplicate detection** based on EnrollmentID
- **Data validation** with clear error messages
- **Encoding detection** for various file formats
- **Quality issue reporting** with downloadable reports

## 🏗️ Architecture

### Project Structure
```
SPM2/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── CLAUDE.md                   # AI assistant guidelines
├── config/                     # Configuration management
│   └── app_config.py          # Application settings
├── src/
│   ├── core/                  # Core functionality
│   │   ├── constants.py       # Project types and definitions
│   │   ├── data/              # Data loading components
│   │   ├── session/           # Session management components
│   │   └── utils/            
│   │       └── helpers.py    # Utility functions
│   ├── modules/               # Analysis modules
│   │   ├── spm2/             
│   │   │   ├── calculator.py # SPM2 business logic
│   │   │   ├── page.py      # SPM2 UI rendering
│   │   │   └── visualizations.py # SPM2 charts
│   │   ├── recidivism/       
│   │   │   ├── inbound_calculator.py
│   │   │   ├── inbound_page.py
│   │   │   ├── inbound_viz.py
│   │   │   ├── outbound_calculator.py
│   │   │   ├── outbound_page.py
│   │   │   └── outbound_viz.py
│   │   └── dashboard/        
│   │       ├── page.py       # Dashboard main page
│   │       ├── summary.py    # Summary metrics
│   │       ├── demographics.py # Demographic analysis
│   │       ├── trends.py     # Trend analysis
│   │       ├── length_of_stay.py # LOS analysis
│   │       ├── equity.py     # Equity/disparity analysis
│   │       ├── filters.py    # Filter components
│   │       └── data_utils.py # Dashboard utilities
│   └── ui/                   # User interface layer
│       ├── factories/        
│       │   ├── html.py      # HTML component factory
│       │   ├── charts.py    # Chart creation factory
│       │   ├── components.py # Master UI component factory
│       │   └── formatters.py # Data formatting utilities
│       ├── layouts/         
│       │   └── widgets.py   # Reusable UI widgets
│       └── themes/          
│           ├── theme.py     # Unified theme system
│           └── styles.py    # Additional styling
```

### Design Patterns

#### Factory Pattern
UI components are created through centralized factories ensuring consistency:
- `HTMLFactory`: Generates styled HTML components with enhanced titles
- `ChartFactory`: Creates Plotly charts with consistent styling
- `UIComponentFactory`: Master factory combining all UI elements

#### Enhanced UI System
- **Hierarchical Titles**: 6 levels with gradient backgrounds and icons
- **Info Boxes**: Semantic styling (info, success, warning, danger)
- **Metric Cards**: Professional displays with deltas and icons
- **Consistent Theme**: Unified color system throughout

#### Session Management
- Centralized state management via `SessionManager`
- Efficient data caching with Streamlit's `@st.cache_data`
- Automatic cleanup and reset functionality

## 🎨 UI Components

### Enhanced Title System
All titles use a hierarchical system with background highlights:
- **Level 1**: Main page headers with gradient backgrounds and 4px borders
- **Level 2**: Section headers with subtle highlights and 3px borders
- **Level 3**: Subsections with light backgrounds and accent colors
- **Level 4-6**: Minor headers with progressively lighter styling

### Info Boxes
Semantic info boxes with proper icon handling:
- **Info**: Blue-themed with light blue background
- **Success**: Green-themed with light green background
- **Warning**: Orange-themed with light orange background
- **Danger**: Red-themed with light red background
- Smart icon management prevents duplicates

### Metric Cards
Professional metric displays featuring:
- Primary value display with formatting
- Delta indicators for comparisons
- Contextual colors based on metric type
- Optional icons for visual context
- Responsive grid layouts

## 🔧 Configuration

### Settings File (config/settings.yaml)
```yaml
data:
  max_file_size_mb: 100
  cache_ttl_seconds: 3600
  date_formats: ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]
  
analysis:
  default_lookback_days: 730
  default_return_window_days: 365
  min_group_size: 10
  
ui:
  enable_animations: true
  show_borders: true
  responsive: true
  max_chart_points: 1000
```

### Environment Variables
- `STREAMLIT_SERVER_PORT`: Server port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: localhost)
- `STREAMLIT_THEME_BASE`: Theme base (light/dark)
- `STREAMLIT_SERVER_HEADLESS`: Run in headless mode (true/false)

## 📈 Performance Optimization

### Data Processing
- Efficient pandas operations with vectorization
- Smart caching using Streamlit's `@st.cache_data` decorator
- Chunked processing for large files (>50MB)
- Automatic duplicate detection and handling
- Optimized date parsing with multiple format support

### UI Rendering
- Lazy loading of heavy components
- Optimized Plotly chart rendering
- Minimal DOM manipulation
- Responsive design for various screen sizes
- Efficient HTML generation through factories

## 🔒 Data Privacy & Security

- **Local Processing**: All data processing happens locally
- **No External Calls**: No data sent to external servers
- **Temporary Files**: Automatically cleaned after processing
- **Session Isolation**: Each user session is isolated
- **Secure Defaults**: No credentials or keys in code

## 📝 Code Quality

### Standards Applied
- **PEP8 Compliant**: All code formatted with black (79-char lines)
- **Import Organization**: isort applied (standard → third-party → local)
- **No Dead Code**: Unused imports and code removed
- **Type Hints**: Used where appropriate for clarity
- **Comprehensive Docstrings**: All major functions documented

### Testing Considerations
- Module imports verified
- Core functionality tested
- Data validation in place
- Error handling for edge cases

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

3. Make your changes following existing patterns

4. Format and check code:
```bash
# Format with black
black --line-length 79 .

# Organize imports
isort .

# Remove unused imports
autoflake --remove-all-unused-imports --in-place -r .
```

5. Test your changes:
```bash
streamlit run app.py
```

6. Submit a pull request

### Code Style Guidelines
- Follow PEP 8 with 79-character line limit
- Use type hints for function parameters
- Add docstrings to all public functions
- Keep functions focused and under 50 lines
- Use the existing factory patterns for UI components

## 🐛 Known Issues & Limitations

- Large files (>100MB) may require increased memory
- Some date formats may need manual specification
- Network visualizations limited to top pathways for performance

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Provide sample data (anonymized) when reporting bugs

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) v1.28+
- Charts powered by [Plotly](https://plotly.com/)
- Data processing with [Pandas](https://pandas.pydata.org/)
- Icons from Unicode Emoji

## 📄 License

[Your License Here]

---

**Version**: 2.0.0  
**Last Updated**: August 2025
**Status**: Production Ready  
**Python**: 3.8+ Required  
**Streamlit**: 1.28+ Required