# PlanetScope-py

A professional Python library for PlanetScope satellite imagery analysis, providing comprehensive tools for scene discovery, metadata analysis, and spatial-temporal density calculations using Planet's Data API.

## Status

**Current Phase**: Phase 1 Complete (Foundation)  
**Test Coverage**: 147/147 tests passing (100%)  
**Python Support**: 3.10+  
**License**: MIT  

## Overview

PlanetScope-py is designed for remote sensing researchers, GIS analysts, and Earth observation professionals who need reliable tools for working with PlanetScope satellite imagery. The library provides a robust foundation for scene inventory management and sophisticated spatial-temporal analysis workflows.

## Features

### Phase 1: Foundation (Complete)
- **Authentication System**: Hierarchical API key detection with secure credential management
- **Configuration Management**: Multi-source configuration with environment variable support
- **Input Validation**: Comprehensive geometry, date, and parameter validation
- **Exception Handling**: Professional error hierarchy with detailed context and troubleshooting guidance
- **Security**: API key masking, secure session management, and credential protection
- **Cross-Platform**: Full compatibility with Windows, macOS, and Linux environments

### Phase 2: Planet API Integration (In Development)
- Scene discovery and search functionality
- Metadata processing and analysis
- Rate limiting and retry strategies
- API response caching and optimization

### Phase 3: Spatial Analysis (Planned)
- Spatial density calculations with multiple algorithms
- Temporal pattern analysis and gap detection
- Grid compatibility and coordinate system support
- Advanced coverage statistics

## Installation

### Standard Installation
```bash
pip install planetscope-py
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/Black-Lights/planetscope-py.git
cd planetscope-py

# Create and activate a virtual environment (recommended)
python -m venv myvenv
# On Windows:
myvenv\Scripts\activate
# On macOS/Linux:
source myvenv/bin/activate

# Install the package in editable mode and dev requirements
pip install -e .
pip install -r requirements-dev.txt
```

## Authentication

PlanetScope-py supports multiple authentication methods with automatic discovery in order of priority:

### Method 1: Environment Variable (Recommended)
```bash
# Linux/macOS
export PL_API_KEY="your_planet_api_key_here"

# Windows Command Prompt
set PL_API_KEY=your_planet_api_key_here

# Windows PowerShell
$env:PL_API_KEY="your_planet_api_key_here"
```

### Method 2: Configuration File
Create `~/.planet.json` in your home directory:
```json
{
    "api_key": "your_planet_api_key_here"
}
```

### Method 3: Direct Parameter
```python
from planetscope_py import PlanetAuth
auth = PlanetAuth(api_key="your_planet_api_key_here")
```

### Obtaining Your API Key
Get your Planet API key from [Planet Account Settings](https://www.planet.com/account/#/).

## Quick Start

```python
from planetscope_py import PlanetAuth, PlanetScopeConfig

# Initialize authentication (auto-detects API key)
auth = PlanetAuth()
print(f"Authentication successful: {auth.is_authenticated}")

# Get authenticated session for API calls
session = auth.get_session()
response = session.get("https://api.planet.com/data/v1/")
print(f"Planet API status: {response.status_code}")

# Access configuration settings
config = PlanetScopeConfig()
print(f"Base URL: {config.base_url}")
print(f"Rate limits: {config.rate_limits}")
print(f"Default CRS: {config.to_dict()['default_crs']}")
```

## Core Components

### Authentication Management
```python
from planetscope_py import PlanetAuth

# Automatic API key discovery
auth = PlanetAuth()

# Check authentication status
if auth.is_authenticated:
    print("Successfully authenticated with Planet API")
    
# Get session for API requests
session = auth.get_session()

# Get authentication tuple for manual requests
username, password = auth.get_auth_tuple()
```

### Configuration System
```python
from planetscope_py import PlanetScopeConfig

# Load default configuration
config = PlanetScopeConfig()

# Access configuration values
base_url = config.base_url
rate_limits = config.rate_limits
timeouts = config.timeouts

# Modify configuration
config.set('max_retries', 5)
config.set('max_roi_area_km2', 15000)

# Export configuration
config_dict = config.to_dict()
```

### Utility Functions
```python
from planetscope_py.utils import (
    validate_geometry,
    validate_date_range,
    validate_cloud_cover,
    create_point_geometry,
    create_bbox_geometry
)

# Validate GeoJSON geometry
geometry = {
    "type": "Polygon",
    "coordinates": [[[-122.5, 37.7], [-122.3, 37.7], 
                     [-122.3, 37.8], [-122.5, 37.8], [-122.5, 37.7]]]
}
validate_geometry(geometry)

# Validate date range
start_date, end_date = validate_date_range("2025-01-01", "2025-12-31")

# Validate cloud cover
cloud_cover = validate_cloud_cover(0.15)  # 15%

# Create geometries
point = create_point_geometry(-122.4, 37.75)
bbox = create_bbox_geometry(-122.5, 37.7, -122.3, 37.8)
```

### Exception Handling
```python
from planetscope_py.exceptions import (
    PlanetScopeError,
    AuthenticationError,
    ValidationError,
    ConfigurationError
)

try:
    auth = PlanetAuth()
except AuthenticationError as e:
    print(f"Authentication failed: {e.message}")
    print(f"Available methods: {e.details.get('methods', [])}")
    print(f"Help URL: {e.details.get('help_url', 'N/A')}")
except ConfigurationError as e:
    print(f"Configuration error: {e.message}")
    print(f"Details: {e.details}")
```

## Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=planetscope_py --cov-report=html

# Run specific test modules
python -m pytest tests/test_auth.py -v
python -m pytest tests/test_config.py -v
python -m pytest tests/test_exceptions.py -v
python -m pytest tests/test_utils.py -v
```

### Test Coverage
Current test coverage: **147/147 tests passing (100%)**

| Component | Tests | Status |
|-----------|-------|--------|
| Authentication | 24 | All passing |
| Configuration | 21 | All passing |
| Exceptions | 48 | All passing |
| Utilities | 54 | All passing |

## Development Roadmap

### Phase 1: Foundation (Complete)
- Robust authentication system with hierarchical API key detection
- Advanced configuration management with environment support
- Comprehensive exception handling with detailed error context
- Complete utility functions with geometry and date validation
- 100% test coverage with professional testing practices
- Security-first design with credential masking and protection

### Phase 2: Planet API Integration (Next)
- Scene discovery and search capabilities
- Metadata extraction and processing
- Rate limiting and request optimization
- Response caching and pagination handling
- Advanced filtering and selection tools

### Phase 3: Spatial Analysis Engine
- Multi-algorithm spatial density calculations
- Temporal pattern analysis and frequency assessment
- Coverage gap detection and reporting
- Grid system compatibility and alignment
- Statistical analysis and quality metrics

### Phase 4: Visualization and Export
- Interactive mapping and visualization
- Timeline and density plotting
- Export capabilities (GeoJSON, CSV, GeoTIFF)
- Report generation and documentation
- Integration with GIS software

## API Reference

### Configuration Defaults
```python
# Planet API Configuration
BASE_URL = "https://api.planet.com/data/v1"
DEFAULT_ITEM_TYPES = ["PSScene"]
DEFAULT_ASSET_TYPES = ["ortho_analytic_4b", "ortho_analytic_4b_xml"]

# Rate Limits (requests per second)
RATE_LIMITS = {
    "search": 10,
    "activate": 5,
    "download": 15,
    "general": 10
}

# Timeout Settings (seconds)
TIMEOUTS = {
    "connect": 10.0,
    "read": 30.0,
    "activation_poll": 300.0,
    "download": 3600.0
}

# Validation Limits
MAX_ROI_AREA_KM2 = 10000
MAX_GEOMETRY_VERTICES = 1000
DEFAULT_CRS = "EPSG:4326"
```

### Exception Hierarchy
```
PlanetScopeError (Base)
├── AuthenticationError     # API key and authentication issues
├── ValidationError        # Input validation failures
├── RateLimitError         # API rate limit exceeded
├── APIError               # Planet API communication errors
├── ConfigurationError     # Configuration file and setup issues
└── AssetError            # Asset activation and download failures
```

## Development

### Code Quality Standards
- **Testing**: Comprehensive test coverage with pytest (147/147 tests passing)
- **Type Hints**: Progressive type annotation implementation with Python 3.10+ support
- **Documentation**: Detailed docstrings and comprehensive README with examples
- **Security**: Credential protection and secure error handling with API key masking
- **Cross-Platform**: Full compatibility with Windows, macOS, and Linux environments
- **Professional Structure**: Modular design with clear separation of concerns

### Contributing
1. Fork the repository on GitHub
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make changes and add comprehensive tests
5. Ensure all tests pass: `python -m pytest tests/ -v`
6. Run code quality checks: `black . && flake8 . && mypy planetscope_py/`
7. Submit a pull request with detailed description

### Development Setup
```bash
# Clone repository
git clone https://github.com/Black-Lights/planetscope-py.git
cd planetscope-py

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Verify installation
python -m pytest tests/ -v
```

## Requirements

### System Requirements
- Python 3.10 or higher
- Active internet connection for Planet API access
- Valid Planet API key

### Core Dependencies
- requests: HTTP client with session management
- shapely: Geometric operations and validation
- pyproj: Coordinate transformations and CRS handling
- numpy: Numerical computations
- pandas: Data manipulation and analysis
- python-dateutil: Date parsing and operations
- orjson: High-performance JSON processing

## Documentation

### Available Resources
- **API Reference**: Complete function and class documentation
- **Getting Started Guide**: Installation and authentication setup
- **Configuration Guide**: Advanced configuration options
- **Testing Guide**: Running and extending the test suite
- **Development Guide**: Contributing and development workflows

### Building Documentation
```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Generate documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## Support

- **Issues**: [GitHub Issues](https://github.com/Black-Lights/planetscope-py/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Black-Lights/planetscope-py/discussions)
- **Documentation**: [Project Wiki](https://github.com/Black-Lights/planetscope-py/wiki)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Planet Labs PBC** for providing the Planet API and PlanetScope imagery
- **Dr. Daniela Stroppiana** - Project Advisor
- **Prof. Giovanna Venuti** - Project Supervisor
- **Politecnico di Milano** - Geoinformatics Engineering Program

## Authors

**Ammar & Umayr**  
Geoinformatics Engineering Students  
Politecnico di Milano

---

**Note**: This project is independently developed and is not officially affiliated with Planet Labs PBC. It is designed to work with Planet's publicly available APIs following their terms of service.
