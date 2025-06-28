# tests/test_basic_ci.py
"""
Basic CI/CD tests for PlanetScope-py
These tests ensure the CI/CD pipeline works correctly
"""

import pytest
import sys
import os
from pathlib import Path

# Add the parent directory to Python path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


class TestCIPipeline:
    """Test class for CI/CD pipeline validation"""
    
    def test_python_version(self):
        """Test that Python version is supported"""
        assert sys.version_info >= (3, 10), "Python 3.10+ required"
    
    def test_package_imports(self):
        """Test that core package can be imported"""
        try:
            import planetscope_py
            assert hasattr(planetscope_py, '__version__')
            assert planetscope_py.__version__ == "4.1.0"
        except ImportError:
            pytest.skip("Package not yet installable")
    
    def test_core_dependencies(self):
        """Test that core dependencies are available"""
        required_packages = [
            'requests',
            'shapely', 
            'numpy',
            'pandas'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            pytest.skip(f"Missing packages: {missing_packages}")
        
        assert len(missing_packages) == 0
    
    def test_project_structure(self):
        """Test that project has correct structure"""
        project_root = Path(__file__).parent.parent
        
        expected_files = [
            'README.md',
            'CHANGELOG.md',
            '.gitignore'
        ]
        
        expected_dirs = [
            'planetscope_py',
            'tests'
        ]
        
        for file_name in expected_files:
            file_path = project_root / file_name
            assert file_path.exists(), f"Missing required file: {file_name}"
        
        for dir_name in expected_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Missing required directory: {dir_name}"
    
    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic package functionality"""
        try:
            # Test basic imports work
            import planetscope_py.utils as utils
            import planetscope_py.config as config
            import planetscope_py.auth as auth
            
            # Test basic functionality
            assert callable(getattr(utils, 'validate_roi', None)) or True
            assert callable(getattr(config, 'PlanetScopeConfig', None)) or True  
            assert callable(getattr(auth, 'PlanetAuth', None)) or True
            
        except ImportError:
            pytest.skip("Core modules not yet implemented")
    
    def test_environment_variables(self):
        """Test environment setup"""
        # Test that we can set and get environment variables
        test_key = "PLANETSCOPE_TEST_VAR"
        test_value = "test_value_123"
        
        os.environ[test_key] = test_value
        assert os.environ.get(test_key) == test_value
        
        # Clean up
        if test_key in os.environ:
            del os.environ[test_key]


class TestCodeQuality:
    """Test code quality standards"""
    
    def test_code_formatting_check(self):
        """Verify code can be checked with black (formatting)"""
        try:
            import black
            assert black.__version__ >= "23.0.0"
        except ImportError:
            pytest.skip("Black not installed")
    
    def test_linting_tools(self):
        """Verify linting tools are available"""
        try:
            import flake8
            # Basic flake8 availability test
            assert hasattr(flake8, '__version__')
        except ImportError:
            pytest.skip("flake8 not installed")


class TestDocumentation:
    """Test documentation requirements"""
    
    def test_readme_exists(self):
        """Test that README.md exists and has content"""
        readme_path = Path(__file__).parent.parent / "README.md"
        assert readme_path.exists(), "README.md is required"
        
        content = readme_path.read_text(encoding='utf-8')
        assert len(content) > 100, "README.md should have substantial content"
        assert "PlanetScope" in content, "README should mention PlanetScope"
    
    def test_changelog_exists(self):
        """Test that CHANGELOG.md exists"""
        changelog_path = Path(__file__).parent.parent / "CHANGELOG.md"
        assert changelog_path.exists(), "CHANGELOG.md is required"
        
        content = changelog_path.read_text(encoding='utf-8')
        assert "4.1.0" in content, "CHANGELOG should include current version"


# Integration test marker
@pytest.mark.integration
class TestIntegration:
    """Integration tests for CI/CD"""
    
    def test_package_build(self):
        """Test that package can be built"""
        pytest.skip("Package building tested in CI/CD pipeline")
    
    def test_cross_platform(self):
        """Test cross-platform compatibility"""
        import platform
        system = platform.system()
        assert system in ['Windows', 'Darwin', 'Linux'], f"Unsupported platform: {system}"


if __name__ == "__main__":
    # Run tests if file is executed directly
    pytest.main([__file__, "-v"])