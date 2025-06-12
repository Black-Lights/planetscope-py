"""Test suite for planetscope-py.

This package contains comprehensive tests for all components of the planetscope-py library.

Test Structure:
    - test_auth.py: Authentication system tests
    - test_config.py: Configuration management tests  
    - test_utils.py: Core utility function tests
    - test_exceptions.py: Exception handling tests
    - conftest.py: Shared fixtures and pytest configuration

Test Categories:
    - unit: Individual function/method tests
    - integration: Multi-component interaction tests
    - auth: Authentication-related tests
    - validation: Input validation tests

Usage:
    # Run all tests
    pytest
    
    # Run specific test categories
    pytest -m "unit"
    pytest -m "auth"
    
    # Run with coverage
    pytest --cov=planetscope_py --cov-report=html
    
    # Run specific test file
    pytest tests/test_auth.py
"""

import sys
from pathlib import Path

# Add the parent directory to the path so tests can import planetscope_py
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Test version for tracking
__test_version__ = "1.0.0"
__test_phase__ = "Phase 1: Foundation"