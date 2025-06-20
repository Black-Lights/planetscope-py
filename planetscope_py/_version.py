#!/usr/bin/env python3
"""Version information for planetscope-py."""

# ==========================================
# PHASE 3 RELEASE: Spatial Analysis Engine Complete
# ==========================================

# FOR PHASE 3 RELEASE (Current)
__version__ = "3.0.0"
__version_info__ = (3, 0, 0)
__phase__ = "Spatial Analysis Engine Complete"
__status__ = "Production Ready"

# AFTER PHASE 3 RELEASE, FOR PHASE 4 DEVELOPMENT:
# __version__ = "4.0.0-dev"
# __version_info__ = (4, 0, 0, "dev")
# __phase__ = "Phase 4: Advanced Visualization and Export"
# __status__ = "Development"

# Development metadata
__build_date__ = "2025-06-20"
__author__ = "Ammar & Umayr"
__email__ = "mohammadammarmughees@gmail.com"

# Release notes for Phase 3
__release_notes__ = """
PlanetScope-py v3.0.0 - Spatial Analysis Engine Complete

Major Features:
- Multi-algorithm spatial density calculations (rasterization, vector overlay, adaptive grid)
- High-resolution analysis support (3m to 1000m grid resolutions)
- Automatic performance optimization and method selection
- Memory-efficient processing for large areas with adaptive grid
- Basic visualization with GeoTIFF export and QGIS integration
- Statistical analysis and comprehensive quality metrics
- 280 tests with 100% coverage
- Complete Planet API integration from v2.0.0

Technical Highlights:
- Three computational methods optimized for different use cases
- Hierarchical adaptive grid for large-scale analysis
- Professional GIS integration with automatic styling
- Sub-pixel analysis capabilities
- Performance benchmarking and optimization

Next: Phase 4 will add advanced interactive visualization, timeline plotting, 
and enhanced export capabilities with dashboard integration.
"""