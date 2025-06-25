#!/usr/bin/env python3
"""Setup configuration for planetscope-py v4.0.1."""

from pathlib import Path

from setuptools import find_packages, setup

# Read version from _version.py
version_file = Path(__file__).parent / "planetscope_py" / "_version.py"
version_info = {}
with open(version_file) as f:
    exec(f.read(), version_info)

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

# Core requirements (production) - Updated June 2025
install_requires = [
    # HTTP requests (latest stable)
    "requests>=2.32.4",
    
    # Geometric operations (latest with NumPy 2.x support)
    "shapely>=2.1.0",
    
    # Coordinate transformations (latest stable)
    "pyproj>=3.7.1",
    
    # Numerical computing (latest stable - NumPy 2.x)
    "numpy>=2.2.0",
    
    # Data manipulation (latest with NumPy 2.x compatibility)
    "pandas>=2.2.3",
    
    # Date/time handling
    "python-dateutil>=2.9.0",
    
    # JSON processing (faster alternative to built-in json)
    "orjson>=3.10.0",
    
    # Enhanced Dependencies for v4.0.0+
    "xarray>=2024.02.0",              # Temporal analysis data cubes
    "aiohttp>=3.8.0",                # Async asset downloads
    "geopandas>=1.0.1",             # GeoPackage vector operations
    "fiona>=1.9.0",                  # GeoPackage I/O
    "rasterio>=1.4.3",               # Raster processing and clipping
    "scipy>=1.13.0",                  # Statistical analysis for temporal patterns
    
    # Visualization Dependencies (moved to core for v4.0.1 fix)
    "matplotlib>=3.10.0",            # Core plotting functionality
    "contextily>=1.6.2",             # Basemaps in visualizations (fixes import issues)
    "folium>=0.19.1",                # Interactive mapping capabilities
    "seaborn>=0.12.2",               # Statistical data visualization
]

setup(
    name="planetscope-py",
    version=version_info["__version__"],
    author="Ammar & Umayr",
    author_email="mohammadammarmughees@gmail.com",
    description="Professional Python library for PlanetScope satellite imagery analysis with complete temporal analysis, spatial density analysis, and advanced data export capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Black-Lights/planetscope-py",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Database",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Framework :: AsyncIO",
        "Environment :: Console",
        "Natural Language :: English",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require={
        "dev": [
            # Testing
            "pytest>=8.3.4",
            "pytest-cov>=6.0.0",
            "pytest-mock>=3.14.0",
            "pytest-xdist>=3.6.0",
            "pytest-asyncio>=0.24.0",
            # Code quality
            "black>=24.12.0",
            "flake8>=7.2.0",
            "flake8-bugbear>=24.12.12",
            "flake8-docstrings>=1.7.0",
            "mypy>=1.13.0",
            "types-requests>=2.32.0.20241217",
            "pre-commit>=4.0.1",
            "ruff>=0.8.1",
            "bandit>=1.8.0",
            # Build tools
            "build>=1.2.2",
            "twine>=6.0.1",
            "wheel>=0.45.1",
        ],
        "docs": [
            "sphinx>=8.1.3",
            "sphinx-rtd-theme>=3.0.2",
            "myst-parser>=4.0.0",
        ],
        "jupyter": [
            "jupyter>=1.1.1",
            "ipywidgets>=8.1.5",            # Interactive controls (moved to core)
            "notebook>=7.3.1",
        ],
        "spatial": [
            # Core spatial dependencies now in main requirements
            "plotly>=5.24.1",                # Advanced plotting
        ],
        "temporal": [
            # Core temporal dependencies now in main requirements  
            "plotly>=5.24.1",                # Advanced plotting for temporal data
        ],
        "asset": [
            # Core asset dependencies now in main requirements
            "tqdm>=4.66.0",                  # Progress bars for downloads
        ],
        "geopackage": [
            # Core geopackage dependencies now in main requirements
            "sqlite3",                       # Database operations (built-in)
        ],
        "interactive": [
            "ipywidgets>=8.1.5",
            "jupyter>=1.1.1",
            "notebook>=7.3.1",
        ],
        "viz": [
            # Core viz dependencies now in main requirements
            "plotly>=5.24.1",                # Advanced interactive plotting
            "bokeh>=3.0.0",                  # Alternative plotting backend
        ],
        "performance": [
            "line-profiler>=4.2.0",
            "memory-profiler>=0.61.0",
        ],
        "all": [
            # Include all optional dependencies
            "pytest>=8.3.4",
            "pytest-cov>=6.0.0",
            "pytest-mock>=3.14.0",
            "pytest-asyncio>=0.24.0",
            "black>=24.12.0",
            "flake8>=7.2.0",
            "mypy>=1.13.0",
            "pre-commit>=4.0.1",
            "sphinx>=8.1.3",
            "sphinx-rtd-theme>=3.0.2",
            "jupyter>=1.1.1",
            "ipywidgets>=8.1.5",            # Interactive controls
            "notebook>=7.3.1",
            "plotly>=5.24.1",                # Advanced plotting
            "bokeh>=3.0.0",                  # Alternative plotting
            "tqdm>=4.66.0",                  # Progress bars
            # Note: Core visualization dependencies (matplotlib, contextily, folium, seaborn) 
            # are now included in the base installation
        ],
    },
    package_data={
        "planetscope_py": ["py.typed"],
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "planetscope=planetscope_py.cli:main",
        ],
    },
    keywords=[
        "satellite",
        "imagery",
        "planet",
        "planetscope",
        "geospatial",
        "remote-sensing",
        "earth-observation",
        "gis",
        "spatial-analysis",
        "density-analysis",
        "temporal-analysis",
        "rasterization",
        "vector-overlay",
        "adaptive-grid",
        "visualization",
        "asset-management",
        "geopackage",
        "async-downloads",
        "quota-monitoring",
        "grid-based-analysis",
        "coordinate-system-fixes",
        "scene-footprints",
        "metadata-processing",
        "quality-assessment",
        "bug-fix",  # Added for v4.0.1
        "import-fixes",  # Added for v4.0.1
    ],
    project_urls={
        "Bug Reports": "https://github.com/Black-Lights/planetscope-py/issues",
        "Source": "https://github.com/Black-Lights/planetscope-py",
        "Documentation": "https://github.com/Black-Lights/planetscope-py/wiki",
        "Changelog": "https://github.com/Black-Lights/planetscope-py/blob/main/CHANGELOG.md",
        "PyPI": "https://pypi.org/project/planetscope-py/",
    },
)