#!/usr/bin/env python3
"""Version information for planetscope-py."""

# ==========================================
# RECOMMENDED STRATEGY: Release Phase 2, then start Phase 3 development
# ==========================================

# FOR PHASE 2 RELEASE (Use this now)
__version__ = "2.0.0"
__version_info__ = (2, 0, 0)
__phase__ = "Phase 2: Planet API Integration"
__status__ = "Production Ready"

# AFTER PHASE 2 RELEASE, IMMEDIATELY SWITCH TO:
# __version__ = "3.0.0-dev"
# __version_info__ = (3, 0, 0, "dev")
# __phase__ = "Phase 3: Spatial Analysis Engine"
# __status__ = "Development"

# Development metadata
__build_date__ = "2025-06-19"
__author__ = "Ammar & Umayr"
__email__ = "mohammadammarmughees@gmail.com"

# Release notes for Phase 2
__release_notes__ = """
PlanetScope-py v2.0.0 - Phase 2 Complete

Major Features:
- Complete Planet API integration with scene discovery
- Comprehensive metadata processing and quality assessment
- Intelligent rate limiting with retry logic
- Bulletproof error handling and validation
- 249 tests with 99%+ coverage
- Production-ready reliability

Next: Phase 3 will add spatial analysis engine with density calculations.
"""