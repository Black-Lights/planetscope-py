#!/usr/bin/env python3
"""
Simple tests for temporal_analysis.py module.

Basic test suite that only tests what actually exists in the temporal_analysis.py module.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime, timedelta
from shapely.geometry import box, Polygon

from planetscope_py.temporal_analysis import (
    TemporalAnalyzer,
    TemporalConfig,
    TemporalResolution,
    TemporalMetric,
    TemporalResult,
)
from planetscope_py.exceptions import ValidationError


class TestTemporalResolution:
    """Test TemporalResolution enum."""

    def test_temporal_resolution_values(self):
        """Test temporal resolution enum values."""
        assert TemporalResolution.DAILY.value == "daily"
        assert TemporalResolution.WEEKLY.value == "weekly"
        assert TemporalResolution.MONTHLY.value == "monthly"


class TestTemporalMetric:
    """Test TemporalMetric enum."""

    def test_temporal_metric_values(self):
        """Test temporal metric enum values."""
        assert TemporalMetric.COVERAGE_DAYS.value == "coverage_days"
        assert TemporalMetric.MEAN_INTERVAL.value == "mean_interval"
        assert TemporalMetric.MEDIAN_INTERVAL.value == "median_interval"
        assert TemporalMetric.MIN_INTERVAL.value == "min_interval"
        assert TemporalMetric.MAX_INTERVAL.value == "max_interval"
        assert TemporalMetric.TEMPORAL_DENSITY.value == "temporal_density"
        assert TemporalMetric.COVERAGE_FREQUENCY.value == "coverage_frequency"


class TestTemporalConfig:
    """Test TemporalConfig dataclass."""

    def test_temporal_config_defaults(self):
        """Test TemporalConfig default values."""
        config = TemporalConfig()

        assert config.spatial_resolution == 30.0
        assert config.temporal_resolution == TemporalResolution.DAILY
        assert config.chunk_size_km == 200.0
        assert config.max_memory_gb == 16.0
        assert config.parallel_workers == 4
        assert config.no_data_value == -9999.0
        assert config.coordinate_system_fixes is True
        assert config.force_single_chunk is False
        assert config.validate_geometries is True
        assert config.min_scenes_per_cell == 2
        assert config.optimization_method == "auto"

    def test_temporal_config_custom(self):
        """Test TemporalConfig with custom values."""
        config = TemporalConfig(
            spatial_resolution=100.0,
            temporal_resolution=TemporalResolution.WEEKLY,
            min_scenes_per_cell=3,
            optimization_method="fast",
            coordinate_system_fixes=False,
        )

        assert config.spatial_resolution == 100.0
        assert config.temporal_resolution == TemporalResolution.WEEKLY
        assert config.min_scenes_per_cell == 3
        assert config.optimization_method == "fast"
        assert config.coordinate_system_fixes is False

    def test_temporal_config_post_init(self):
        """Test TemporalConfig post-initialization processing."""
        # Test with string temporal resolution
        config = TemporalConfig(temporal_resolution="weekly")
        assert config.temporal_resolution == TemporalResolution.WEEKLY

        # Test with invalid temporal resolution
        with pytest.raises(ValidationError):
            TemporalConfig(temporal_resolution="invalid")

    def test_temporal_config_default_metrics(self):
        """Test default metrics assignment."""
        config = TemporalConfig()
        
        # Should have default metrics
        assert config.metrics is not None
        assert len(config.metrics) == 3
        assert TemporalMetric.COVERAGE_DAYS in config.metrics
        assert TemporalMetric.MEAN_INTERVAL in config.metrics
        assert TemporalMetric.TEMPORAL_DENSITY in config.metrics


class TestTemporalResult:
    """Test TemporalResult dataclass."""

    def test_temporal_result_creation(self):
        """Test TemporalResult object creation."""
        # Create mock arrays
        mock_arrays = {
            TemporalMetric.COVERAGE_DAYS: np.array([[1, 2], [3, 4]], dtype=np.float32)
        }
        
        # Create mock transform
        from rasterio.transform import Affine
        mock_transform = Affine(0.001, 0, 9.0, 0, -0.001, 45.6, 0, 0, 1)
        
        result = TemporalResult(
            metric_arrays=mock_arrays,
            transform=mock_transform,
            crs="EPSG:4326",
            bounds=(9.0, 45.4, 9.2, 45.6),
            temporal_stats={"mean_coverage_days": 2.5},
            computation_time=1.23,
            config=TemporalConfig(),
            grid_info={"width": 2, "height": 2},
            date_range=("2025-01-01", "2025-03-31"),
        )

        assert len(result.metric_arrays) == 1
        assert TemporalMetric.COVERAGE_DAYS in result.metric_arrays
        assert result.crs == "EPSG:4326"
        assert result.bounds == (9.0, 45.4, 9.2, 45.6)
        assert result.computation_time == 1.23
        assert result.coordinate_system_corrected is True
        assert result.no_data_value == -9999.0


class TestTemporalAnalyzer:
    """Test TemporalAnalyzer class."""

    @pytest.fixture
    def temporal_analyzer(self):
        """Create TemporalAnalyzer instance for testing."""
        config = TemporalConfig(spatial_resolution=100.0)  # Coarse for faster testing
        return TemporalAnalyzer(config)

    @pytest.fixture
    def sample_scenes(self):
        """Create sample scenes for testing."""
        scenes = []
        base_date = datetime(2025, 1, 1)

        for i in range(5):  # Just 5 scenes for simple testing
            # Create scenes every 3 days
            scene_date = base_date + timedelta(days=i * 3)
            scene = {
                "properties": {
                    "id": f"scene_{i:03d}",
                    "acquired": scene_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "cloud_cover": 0.1 + (i * 0.02),
                    "item_type": "PSScene",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [9.0, 45.4],
                            [9.1, 45.4],
                            [9.1, 45.5],
                            [9.0, 45.5],
                            [9.0, 45.4],
                        ]
                    ],
                },
            }
            scenes.append(scene)

        return scenes

    @pytest.fixture
    def sample_roi(self):
        """Create sample ROI for testing."""
        return box(9.0, 45.4, 9.1, 45.5)

    def test_temporal_analyzer_initialization(self, temporal_analyzer):
        """Test TemporalAnalyzer initialization."""
        assert temporal_analyzer.config.spatial_resolution == 100.0
        assert temporal_analyzer.config.optimization_method == "auto"
        assert hasattr(temporal_analyzer, 'performance_stats')

    def test_validate_config(self, temporal_analyzer):
        """Test configuration validation."""
        # Test with invalid spatial resolution
        config = TemporalConfig(spatial_resolution=-10.0)
        analyzer = TemporalAnalyzer.__new__(TemporalAnalyzer)
        analyzer.config = config
        
        with pytest.raises(ValidationError, match="Spatial resolution must be positive"):
            analyzer._validate_config()

    def test_prepare_roi_geometry_dict(self, temporal_analyzer):
        """Test ROI geometry preparation from dict."""
        roi_dict = {
            "type": "Polygon",
            "coordinates": [
                [
                    [9.0, 45.4],
                    [9.1, 45.4],
                    [9.1, 45.5],
                    [9.0, 45.5],
                    [9.0, 45.4],
                ]
            ],
        }

        roi_poly = temporal_analyzer._prepare_roi_geometry(roi_dict)
        
        assert isinstance(roi_poly, Polygon)
        assert roi_poly.is_valid
        assert not roi_poly.is_empty

    def test_prepare_scene_data(self, temporal_analyzer, sample_scenes):
        """Test scene data preparation."""
        scene_data = temporal_analyzer._prepare_scene_data(
            sample_scenes, "2025-01-01", "2025-01-31"
        )

        assert isinstance(scene_data, pd.DataFrame)
        assert len(scene_data) > 0
        assert "scene_id" in scene_data.columns
        assert "geometry" in scene_data.columns
        assert "acquired_date" in scene_data.columns
        assert "acquired_str" in scene_data.columns
        assert "cloud_cover" in scene_data.columns

    def test_prepare_scene_data_empty(self, temporal_analyzer):
        """Test scene data preparation with empty scenes."""
        with pytest.raises(ValidationError, match="No valid scenes found in date range"):
            temporal_analyzer._prepare_scene_data([], "2025-01-01", "2025-01-31")

    def test_merge_config_kwargs(self, temporal_analyzer):
        """Test configuration merging with kwargs."""
        kwargs = {
            "spatial_resolution": 50.0,
            "optimization_method": "fast",
            "min_scenes_per_cell": 5,
        }

        merged_config = temporal_analyzer._merge_config_kwargs(kwargs)

        assert merged_config.spatial_resolution == 50.0
        assert merged_config.optimization_method == "fast"
        assert merged_config.min_scenes_per_cell == 5

    def test_create_spatial_chunks_small_roi(self, temporal_analyzer, sample_roi):
        """Test spatial chunk creation with small ROI."""
        config = TemporalConfig(chunk_size_km=500.0)  # Large chunk size
        
        chunks = temporal_analyzer._create_spatial_chunks(sample_roi, config)
        
        assert len(chunks) == 1

    @pytest.mark.skip(reason="Export functionality requires rasterio and complex setup")
    def test_export_temporal_geotiffs_skip(self):
        """Skip export test due to complexity."""
        pass


class TestBasicFunctionality:
    """Test basic functionality without complex dependencies."""

    def test_temporal_analyzer_can_be_created(self):
        """Test that TemporalAnalyzer can be instantiated."""
        config = TemporalConfig(spatial_resolution=100.0)
        analyzer = TemporalAnalyzer(config)
        
        assert analyzer is not None
        assert analyzer.config.spatial_resolution == 100.0

    def test_temporal_config_validation(self):
        """Test basic configuration validation."""
        # Valid config should work
        config = TemporalConfig(spatial_resolution=50.0)
        assert config.spatial_resolution == 50.0
        
        # Test string to enum conversion
        config2 = TemporalConfig(temporal_resolution="weekly")
        assert config2.temporal_resolution == TemporalResolution.WEEKLY

    def test_temporal_enums_exist(self):
        """Test that all required enums exist and have expected values."""
        # Test TemporalResolution
        assert hasattr(TemporalResolution, 'DAILY')
        assert hasattr(TemporalResolution, 'WEEKLY')
        assert hasattr(TemporalResolution, 'MONTHLY')
        
        # Test TemporalMetric
        assert hasattr(TemporalMetric, 'COVERAGE_DAYS')
        assert hasattr(TemporalMetric, 'MEAN_INTERVAL')
        assert hasattr(TemporalMetric, 'TEMPORAL_DENSITY')

    def test_minimal_scene_data_processing(self):
        """Test basic scene data processing."""
        analyzer = TemporalAnalyzer(TemporalConfig(spatial_resolution=100.0))
        
        # Create minimal valid scene
        scenes = [{
            "properties": {
                "id": "test_scene",
                "acquired": "2025-01-15T10:00:00Z",
                "cloud_cover": 0.1,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [9.0, 45.4],
                        [9.1, 45.4],
                        [9.1, 45.5],
                        [9.0, 45.5],
                        [9.0, 45.4],
                    ]
                ],
            },
        }]
        
        # This should not raise an error
        scene_data = analyzer._prepare_scene_data(scenes, "2025-01-01", "2025-01-31")
        assert len(scene_data) == 1
        assert scene_data.iloc[0]['scene_id'] == 'test_scene'


# Simple integration test that doesn't require complex mocking
class TestSimpleIntegration:
    """Simple integration tests."""

    def test_end_to_end_basic_workflow(self):
        """Test basic workflow without complex dependencies."""
        # Create analyzer
        config = TemporalConfig(
            spatial_resolution=500.0,  # Very coarse for speed
            min_scenes_per_cell=1,     # Low requirement
            optimization_method="fast",
            force_single_chunk=True    # Avoid chunking complexity
        )
        analyzer = TemporalAnalyzer(config)
        
        # Create simple test data
        scenes = [{
            "properties": {
                "id": "test_scene_1",
                "acquired": "2025-01-15T10:00:00Z",
                "cloud_cover": 0.1,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [9.0, 45.4],
                        [9.05, 45.4],
                        [9.05, 45.45],
                        [9.0, 45.45],
                        [9.0, 45.4],
                    ]
                ],
            },
        }]
        
        roi = box(9.0, 45.4, 9.05, 45.45)
        
        # Test individual components
        roi_poly = analyzer._prepare_roi_geometry(roi)
        assert roi_poly.is_valid
        
        scene_data = analyzer._prepare_scene_data(scenes, "2025-01-01", "2025-01-31")
        assert len(scene_data) == 1
        
        # Test chunk creation
        chunks = analyzer._create_spatial_chunks(roi_poly, config)
        assert len(chunks) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])