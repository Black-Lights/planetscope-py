#!/usr/bin/env python3
"""Tests for metadata processing and analysis.

Comprehensive test suite for metadata.py functionality including
metadata extraction, quality assessment, and coverage statistics.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from planetscope_py.metadata import MetadataProcessor
from planetscope_py.exceptions import ValidationError, PlanetScopeError


class TestMetadataProcessor:
    """Test suite for MetadataProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create MetadataProcessor instance for testing."""
        return MetadataProcessor()

    @pytest.fixture
    def sample_scene(self):
        """Sample Planet scene with realistic Planet API date format."""
        return {
            "type": "Feature",
            "id": "test_scene_1",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-122.4194, 37.7749],
                        [-122.4094, 37.7749],
                        [-122.4094, 37.7849],
                        [-122.4194, 37.7849],
                        [-122.4194, 37.7749],
                    ]
                ],
            },
            "properties": {
                "id": "test_scene_1",
                "item_type": "PSScene",
                "satellite_id": "test_satellite",
                "provider": "planet",
                # Realistic Planet API date format with 5-digit microseconds
                "acquired": "2024-01-15T14:30:00.12345Z",
                "published": "2024-01-15T16:00:00.67890Z",
                "updated": "2024-01-15T16:00:00.67890Z",
                "cloud_cover": 0.15,
                "sun_azimuth": 180.5,
                "sun_elevation": 45.2,
                "usable_data": 0.92,
                "quality_category": "standard",
                "pixel_resolution": 3.0,
                "ground_control": True,
            },
        }

    # 2. Update sample_scenes_collection fixture
    @pytest.fixture
    def sample_scenes_collection(self):
        """Collection of sample scenes with realistic Planet API dates."""
        return [
            {
                "type": "Feature",
                "id": "scene_1",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.42, 37.77],
                            [-122.41, 37.77],
                            [-122.41, 37.78],
                            [-122.42, 37.78],
                            [-122.42, 37.77],
                        ]
                    ],
                },
                "properties": {
                    "id": "scene_1",
                    "item_type": "PSScene",
                    "satellite_id": "sat_1",
                    # Realistic Planet API format
                    "acquired": "2024-01-15T14:30:00.12345Z",
                    "cloud_cover": 0.05,
                    "sun_elevation": 50.0,
                    "usable_data": 0.95,
                    "quality_category": "standard",
                },
            },
            {
                "type": "Feature",
                "id": "scene_2",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.41, 37.77],
                            [-122.40, 37.77],
                            [-122.40, 37.78],
                            [-122.41, 37.78],
                            [-122.41, 37.77],
                        ]
                    ],
                },
                "properties": {
                    "id": "scene_2",
                    "item_type": "PSScene",
                    "satellite_id": "sat_2",
                    "acquired": "2024-02-10T15:45:00.67890Z",
                    "cloud_cover": 0.25,
                    "sun_elevation": 35.0,
                    "usable_data": 0.80,
                    "quality_category": "standard",
                },
            },
            {
                "type": "Feature",
                "id": "scene_3",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.40, 37.77],
                            [-122.39, 37.77],
                            [-122.39, 37.78],
                            [-122.40, 37.78],
                            [-122.40, 37.77],
                        ]
                    ],
                },
                "properties": {
                    "id": "scene_3",
                    "item_type": "PSScene",
                    "satellite_id": "sat_1",
                    "acquired": "2024-03-05T13:15:00.11111Z",
                    "cloud_cover": 0.45,
                    "sun_elevation": 25.0,
                    "usable_data": 0.65,
                    "quality_category": "standard",
                },
            },
        ]

    def test_initialization(self, processor):
        """Test MetadataProcessor initialization."""
        assert processor.config is not None
        assert hasattr(processor, "QUALITY_THRESHOLDS")
        assert "cloud_cover" in processor.QUALITY_THRESHOLDS
        assert "sun_elevation" in processor.QUALITY_THRESHOLDS
        assert "usable_data" in processor.QUALITY_THRESHOLDS

    def test_extract_scene_metadata_basic(self, processor, sample_scene):
        """Test basic metadata extraction from scene."""
        metadata = processor.extract_scene_metadata(sample_scene)

        # Verify basic scene information
        assert metadata["scene_id"] == "test_scene_1"
        assert metadata["item_type"] == "PSScene"
        assert metadata["satellite_id"] == "test_satellite"
        assert metadata["provider"] == "planet"

        # Verify acquisition information
        assert metadata["acquired"] == "2024-01-15T14:30:00.12345Z"
        assert metadata["acquisition_date"] == "2024-01-15"
        assert metadata["year"] == 2024
        assert metadata["month"] == 1
        assert metadata["day_of_year"] == 15  # Jan 15th

        # Verify quality metrics
        assert metadata["cloud_cover"] == 0.15
        assert metadata["sun_elevation"] == 45.2
        assert metadata["usable_data"] == 0.92
        assert metadata["ground_control"] is True

    def test_extract_scene_metadata_derived_metrics(self, processor, sample_scene):
        """Test derived metrics calculation."""
        metadata = processor.extract_scene_metadata(sample_scene)

        # Check season calculation (January = winter)
        assert metadata["season"] == "winter"

        # Check solar conditions (45.2° elevation = optimal)
        assert metadata["solar_conditions"] == "optimal"

        # Check suitability - UPDATED FOR NEW LOGIC
        # Sample scene has: cloud_cover=0.15, usable_data=0.92
        # With combined logic: cloud_cover <= 0.2 AND usable_data >= 0.8 = "good"
        assert metadata["suitability"] == "good"  # This should still be correct

        # Check quality scores
        assert "quality_scores" in metadata
        assert "overall_quality" in metadata
        assert metadata["overall_quality"] > 0.8  # Should be high quality

    def test_extract_scene_metadata_missing_usable_data(self, processor):
        """Test metadata extraction when usable_data is missing (real Planet API scenario)."""
        # Create a scene similar to Milan example
        scene_missing_usable_data = {
            "type": "Feature",
            "id": "milan_scene_example",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [8.55, 45.91],
                        [9.83, 45.91],
                        [9.83, 45.02],
                        [8.55, 45.02],
                        [8.55, 45.91],
                    ]
                ],
            },
            "properties": {
                "id": "milan_scene_example",
                "item_type": "PSScene",
                "acquired": "2024-08-24T10:32:08.000Z",
                "cloud_cover": 0.0,  # Perfect conditions
                "sun_elevation": 53.9,
                "clear_percent": 100,  # Planet API provides this
                "visible_percent": 100,  # But not usable_data
                # Note: no usable_data field
            },
        }

        metadata = processor.extract_scene_metadata(scene_missing_usable_data)

        # Verify extracted values
        assert metadata["cloud_cover"] == 0.0
        assert metadata["usable_data"] is None  # Should be None since not in Planet API
        assert metadata["sun_elevation"] == 53.9
        assert metadata["month"] == 8  # August

        # Verify derived metrics use fallback logic
        assert metadata["season"] == "summer"
        assert metadata["solar_conditions"] == "optimal"
        assert metadata["suitability"] == "excellent"  # Should use cloud_cover fallback

        # Quality should still be calculated properly
        assert metadata["overall_quality"] > 0.9  # Should be very high

    def test_extract_scene_metadata_geometry(self, processor, sample_scene):
        """Test geometry metadata extraction."""
        metadata = processor.extract_scene_metadata(sample_scene)

        # Verify geometric properties
        assert metadata["geometry_type"] == "Polygon"
        assert "area_km2" in metadata
        assert metadata["area_km2"] > 0

        # Verify bounds
        assert "bounds" in metadata
        bounds = metadata["bounds"]
        assert bounds["west"] == -122.4194
        assert bounds["east"] == -122.4094
        assert bounds["south"] == 37.7749
        assert bounds["north"] == 37.7849

        # Verify centroid
        assert "centroid" in metadata
        centroid = metadata["centroid"]
        assert abs(centroid["longitude"] - (-122.4144)) < 0.001
        assert abs(centroid["latitude"] - 37.7799) < 0.001

    def test_extract_scene_metadata_invalid_input(self, processor):
        """Test metadata extraction with invalid input."""
        with pytest.raises(ValidationError):
            processor.extract_scene_metadata("not_a_dict")

        with pytest.raises(ValidationError):
            processor.extract_scene_metadata({})  # Empty dict

    def test_calculate_quality_scores(self, processor):
        """Test quality score calculation."""
        metadata = {
            "cloud_cover": 0.1,
            "sun_elevation": 45.0,
            "usable_data": 0.9,
            "ground_control": True,
        }

        scores = processor._calculate_quality_scores(metadata)

        assert scores["cloud_cover"] == 0.9  # 1 - 0.1
        assert scores["sun_elevation"] == 0.75  # 45/60
        assert scores["usable_data"] == 0.9
        assert scores["ground_control"] == 1.0

    def test_calculate_overall_quality(self, processor):
        """Test overall quality calculation."""
        quality_scores = {
            "cloud_cover": 0.9,
            "sun_elevation": 0.75,
            "usable_data": 0.9,
            "ground_control": 1.0,
        }

        overall = processor._calculate_overall_quality(quality_scores)

        # Weighted average: 0.4*0.9 + 0.2*0.75 + 0.3*0.9 + 0.1*1.0 = 0.88
        assert abs(overall - 0.88) < 0.01

    def test_assess_coverage_quality(self, processor, sample_scenes_collection):
        """Test coverage quality assessment."""
        target_geometry = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122.43, 37.76],
                    [-122.38, 37.76],
                    [-122.38, 37.79],
                    [-122.43, 37.79],
                    [-122.43, 37.76],
                ]
            ],
        }

        assessment = processor.assess_coverage_quality(
            scenes=sample_scenes_collection, target_geometry=target_geometry
        )

        assert assessment["total_scenes"] == 3
        assert "coverage_statistics" in assessment
        assert "temporal_analysis" in assessment
        assert "quality_analysis" in assessment
        assert "recommendations" in assessment

        # Verify temporal analysis
        temporal = assessment["temporal_analysis"]
        assert temporal["total_scenes"] == 3
        assert temporal["span_days"] > 0
        assert "monthly_distribution" in temporal
        assert "seasonal_distribution" in temporal

    def test_assess_coverage_quality_empty_scenes(self, processor):
        """Test coverage assessment with empty scene list."""
        assessment = processor.assess_coverage_quality(scenes=[])

        assert assessment["total_scenes"] == 0
        assert assessment["coverage_area_km2"] == 0.0
        assert assessment["temporal_span_days"] == 0
        assert "No scenes available" in assessment["recommendations"][0]

    def test_filter_by_metadata_criteria(self, processor, sample_scenes_collection):
        """Test metadata-based scene filtering."""
        criteria = {
            "max_cloud_cover": 0.2,
            "min_sun_elevation": 40.0,
            "min_usable_data": 0.85,
        }

        filtered_scenes, stats = processor.filter_by_metadata_criteria(
            scenes=sample_scenes_collection, criteria=criteria
        )

        # Only scene_1 should pass all criteria
        assert len(filtered_scenes) == 1
        assert filtered_scenes[0]["properties"]["id"] == "scene_1"

        # Verify filter statistics
        assert stats["original_count"] == 3
        assert stats["filtered_count"] == 1
        assert stats["retention_rate"] == 1 / 3
        assert "cloud_cover_exceeded" in stats["rejection_reasons"]
        assert "sun_elevation_too_low" in stats["rejection_reasons"]

    def test_check_metadata_criteria(self, processor):
        """Test individual criteria checking."""
        metadata = {
            "cloud_cover": 0.3,
            "sun_elevation": 25.0,
            "usable_data": 0.7,
            "quality_category": "standard",
        }

        # Test passing criteria
        passed, reason = processor._check_metadata_criteria(
            metadata, {"max_cloud_cover": 0.5}
        )
        assert passed is True
        assert reason == ""

        # Test failing criteria
        passed, reason = processor._check_metadata_criteria(
            metadata, {"max_cloud_cover": 0.2}
        )
        assert passed is False
        assert reason == "cloud_cover_exceeded"

        passed, reason = processor._check_metadata_criteria(
            metadata, {"min_sun_elevation": 30.0}
        )
        assert passed is False
        assert reason == "sun_elevation_too_low"

    def test_generate_metadata_summary(self, processor, sample_scenes_collection):
        """Test comprehensive metadata summary generation."""
        summary = processor.generate_metadata_summary(sample_scenes_collection)

        assert "collection_overview" in summary
        assert "temporal_analysis" in summary
        assert "quality_analysis" in summary
        assert "spatial_analysis" in summary
        assert "satellite_analysis" in summary

        # Verify collection overview
        overview = summary["collection_overview"]
        assert overview["total_scenes"] == 3
        assert overview["unique_satellites"] == 2  # sat_1 and sat_2
        assert overview["unique_item_types"] == 1  # PSScene

        # Verify quality summary
        quality_summary = overview["quality_summary"]
        assert (
            "excellent" in quality_summary
            or "good" in quality_summary
            or "fair" in quality_summary
        )

        # Verify satellite analysis
        satellite_analysis = summary["satellite_analysis"]
        assert "satellite_distribution" in satellite_analysis
        assert "item_type_distribution" in satellite_analysis
        assert (
            satellite_analysis["satellite_distribution"]["sat_1"] == 2
        )  # scenes 1 and 3
        assert satellite_analysis["satellite_distribution"]["sat_2"] == 1  # scene 2

    def test_analyze_temporal_distribution(self, processor, sample_scenes_collection):
        """Test temporal distribution analysis."""
        # Extract metadata first
        scene_metadata = [
            processor.extract_scene_metadata(scene)
            for scene in sample_scenes_collection
        ]

        temporal_stats = processor._analyze_temporal_distribution(scene_metadata)

        assert "date_range" in temporal_stats
        assert "span_days" in temporal_stats
        assert "total_scenes" in temporal_stats
        assert "monthly_distribution" in temporal_stats
        assert "seasonal_distribution" in temporal_stats

        # Verify date range
        date_range = temporal_stats["date_range"]
        assert "2024-01-15" in date_range["start"]
        assert "2024-03-05" in date_range["end"]

        # Verify seasonal distribution
        seasonal = temporal_stats["seasonal_distribution"]
        assert seasonal["winter"] >= 1  # January scene
        assert seasonal["spring"] >= 1  # March scene

    def test_analyze_quality_distribution(self, processor, sample_scenes_collection):
        """Test quality distribution analysis."""
        scene_metadata = [
            processor.extract_scene_metadata(scene)
            for scene in sample_scenes_collection
        ]

        quality_stats = processor._analyze_quality_distribution(scene_metadata)

        assert "suitability_distribution" in quality_stats
        assert "cloud_cover" in quality_stats
        assert "sun_elevation" in quality_stats
        assert "overall_quality" in quality_stats

        # Verify cloud cover statistics
        cc_stats = quality_stats["cloud_cover"]
        assert cc_stats["min"] == 0.05
        assert cc_stats["max"] == 0.45
        assert cc_stats["mean"] > 0.05 and cc_stats["mean"] < 0.45

    def test_analyze_spatial_distribution(self, processor, sample_scenes_collection):
        """Test spatial distribution analysis."""
        scene_metadata = [
            processor.extract_scene_metadata(scene)
            for scene in sample_scenes_collection
        ]

        spatial_stats = processor._analyze_spatial_distribution(scene_metadata)

        assert "area_km2" in spatial_stats
        assert "extent" in spatial_stats
        assert "center" in spatial_stats

        # Verify extent
        extent = spatial_stats["extent"]
        assert extent["west"] <= -122.42
        assert extent["east"] >= -122.39
        assert extent["south"] <= 37.77
        assert extent["north"] >= 37.78

    def test_calculate_coverage_statistics(self, processor, sample_scenes_collection):
        """Test coverage statistics calculation."""
        scene_metadata = [
            processor.extract_scene_metadata(scene)
            for scene in sample_scenes_collection
        ]

        target_geometry = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122.43, 37.76],
                    [-122.38, 37.76],
                    [-122.38, 37.79],
                    [-122.43, 37.79],
                    [-122.43, 37.76],
                ]
            ],
        }

        coverage_stats = processor._calculate_coverage_statistics(
            scene_metadata, target_geometry
        )

        assert "total_area_km2" in coverage_stats
        assert "unique_coverage_km2" in coverage_stats
        assert "overlap_factor" in coverage_stats
        assert "coverage_efficiency" in coverage_stats
        assert "target_area_km2" in coverage_stats

        assert coverage_stats["total_area_km2"] > 0
        assert coverage_stats["overlap_factor"] >= 1.0

    def test_generate_coverage_recommendations(self, processor):
        """Test coverage recommendation generation."""
        # Test low coverage efficiency
        coverage_stats = {"coverage_efficiency": 0.3, "overlap_factor": 1.5}
        temporal_stats = {"span_days": 90, "total_scenes": 5}
        quality_stats = {"suitability_distribution": {"poor": 8, "excellent": 2}}

        recommendations = processor._generate_coverage_recommendations(
            coverage_stats, temporal_stats, quality_stats
        )

        assert len(recommendations) > 0
        assert any("Low coverage efficiency" in rec for rec in recommendations)
        assert any("poor quality scenes" in rec for rec in recommendations)

        # Test good coverage
        coverage_stats = {"coverage_efficiency": 0.9, "overlap_factor": 1.2}
        temporal_stats = {"span_days": 30, "total_scenes": 10}
        quality_stats = {"suitability_distribution": {"excellent": 8, "poor": 2}}

        recommendations = processor._generate_coverage_recommendations(
            coverage_stats, temporal_stats, quality_stats
        )

        assert any(
            "meets quality and coverage requirements" in rec for rec in recommendations
        )

    def test_derived_metrics_calculation(self, processor):
        """Test calculation of derived metrics."""
        # Test different seasons
        winter_metadata = {"month": 1}
        spring_metadata = {"month": 4}
        summer_metadata = {"month": 7}
        autumn_metadata = {"month": 10}

        assert (
            processor._calculate_derived_metrics(winter_metadata)["season"] == "winter"
        )
        assert (
            processor._calculate_derived_metrics(spring_metadata)["season"] == "spring"
        )
        assert (
            processor._calculate_derived_metrics(summer_metadata)["season"] == "summer"
        )
        assert (
            processor._calculate_derived_metrics(autumn_metadata)["season"] == "autumn"
        )

        # Test solar conditions
        optimal_metadata = {"sun_elevation": 50.0}
        good_metadata = {"sun_elevation": 35.0}
        marginal_metadata = {"sun_elevation": 20.0}
        poor_metadata = {"sun_elevation": 10.0}

        assert (
            processor._calculate_derived_metrics(optimal_metadata)["solar_conditions"]
            == "optimal"
        )
        assert (
            processor._calculate_derived_metrics(good_metadata)["solar_conditions"]
            == "good"
        )
        assert (
            processor._calculate_derived_metrics(marginal_metadata)["solar_conditions"]
            == "marginal"
        )
        assert (
            processor._calculate_derived_metrics(poor_metadata)["solar_conditions"]
            == "poor"
        )

        # Test suitability classification
        excellent_metadata = {"cloud_cover": 0.05, "usable_data": 0.95}
        good_metadata = {"cloud_cover": 0.15, "usable_data": 0.85}
        fair_metadata = {"cloud_cover": 0.3, "usable_data": 0.75}
        poor_metadata = {"cloud_cover": 0.5, "usable_data": 0.6}

        assert (
            processor._calculate_derived_metrics(excellent_metadata)["suitability"]
            == "excellent"
        )
        assert (
            processor._calculate_derived_metrics(good_metadata)["suitability"] == "good"
        )
        assert (
            processor._calculate_derived_metrics(fair_metadata)["suitability"] == "fair"
        )
        assert (
            processor._calculate_derived_metrics(poor_metadata)["suitability"] == "poor"
        )

    def test_extract_geometry_metadata_error_handling(self, processor):
        """Test geometry metadata extraction with invalid geometry."""
        invalid_geometry = {"type": "InvalidType", "coordinates": []}

        # Should not raise exception, but return safe defaults
        result = processor._extract_geometry_metadata(invalid_geometry)

        # Updated expectation: now returns safe defaults instead of empty dict
        expected = {
            "geometry_type": "Unknown",
            "area_km2": 0.0,
            "bounds": {"west": 0.0, "south": 0.0, "east": 0.0, "north": 0.0},
            "centroid": {"longitude": 0.0, "latitude": 0.0},
            "perimeter_km": 0.0,
            "aspect_ratio": 1.0,
        }

        assert result == expected

    def test_metadata_with_missing_fields(self, processor):
        """Test metadata extraction with missing optional fields."""
        minimal_scene = {
            "type": "Feature",
            "id": "minimal_scene",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-122.42, 37.77],
                        [-122.41, 37.77],
                        [-122.41, 37.78],
                        [-122.42, 37.78],
                        [-122.42, 37.77],
                    ]
                ],
            },
            "properties": {
                "id": "minimal_scene",
                "acquired": "2024-01-15T14:30:00.000Z",
                # Many fields missing
            },
        }

        metadata = processor.extract_scene_metadata(minimal_scene)

        # Should still extract basic information
        assert metadata["scene_id"] == "minimal_scene"
        assert metadata["acquired"] == "2024-01-15T14:30:00.000Z"

        # Missing fields should be None
        assert metadata["cloud_cover"] is None
        assert metadata["sun_elevation"] is None
        assert metadata["usable_data"] is None

    def test_process_stats_response(self, processor):
        """Test processing of Planet API stats response."""
        stats_data = {
            "buckets": [
                {"start_time": "2024-01-01T00:00:00Z", "count": 15},
                {"start_time": "2024-02-01T00:00:00Z", "count": 22},
                {"start_time": "2024-03-01T00:00:00Z", "count": 8},
            ],
            "interval": "month",
        }

        processed = processor._process_stats_response(stats_data)

        assert processed["total_scenes"] == 45  # 15 + 22 + 8
        assert processed["interval"] == "month"
        assert len(processed["temporal_distribution"]) == 3
        assert processed["temporal_distribution"]["2024-01-01T00:00:00Z"] == 15

    def test_suitability_cloud_cover_fallback(self, processor):
        """Test suitability classification using cloud cover fallback logic when usable_data is missing."""

        # Test excellent classification (cloud_cover <= 0.05)
        excellent_cloud_only = {"cloud_cover": 0.00, "usable_data": None}
        result = processor._calculate_derived_metrics(excellent_cloud_only)
        assert result["suitability"] == "excellent"

        excellent_cloud_boundary = {"cloud_cover": 0.05, "usable_data": None}
        result = processor._calculate_derived_metrics(excellent_cloud_boundary)
        assert result["suitability"] == "excellent"

        # Test good classification (0.05 < cloud_cover <= 0.15)
        good_cloud_low = {"cloud_cover": 0.06, "usable_data": None}
        result = processor._calculate_derived_metrics(good_cloud_low)
        assert result["suitability"] == "good"

        good_cloud_boundary = {"cloud_cover": 0.15, "usable_data": None}
        result = processor._calculate_derived_metrics(good_cloud_boundary)
        assert result["suitability"] == "good"

        # Test fair classification (0.15 < cloud_cover <= 0.30)
        fair_cloud_low = {"cloud_cover": 0.16, "usable_data": None}
        result = processor._calculate_derived_metrics(fair_cloud_low)
        assert result["suitability"] == "fair"

        fair_cloud_boundary = {"cloud_cover": 0.30, "usable_data": None}
        result = processor._calculate_derived_metrics(fair_cloud_boundary)
        assert result["suitability"] == "fair"

        # Test poor classification (cloud_cover > 0.30)
        poor_cloud = {"cloud_cover": 0.31, "usable_data": None}
        result = processor._calculate_derived_metrics(poor_cloud)
        assert result["suitability"] == "poor"

        poor_cloud_high = {"cloud_cover": 0.80, "usable_data": None}
        result = processor._calculate_derived_metrics(poor_cloud_high)
        assert result["suitability"] == "poor"

    def test_suitability_edge_cases(self, processor):
        """Test suitability classification edge cases and error handling."""

        # Test with both cloud_cover and usable_data as None
        no_data = {"cloud_cover": None, "usable_data": None}
        result = processor._calculate_derived_metrics(no_data)
        assert result["suitability"] == "unknown"

        # Test with cloud_cover as string (should be unknown)
        invalid_cloud_cover = {"cloud_cover": "0.05", "usable_data": None}
        result = processor._calculate_derived_metrics(invalid_cloud_cover)
        assert result["suitability"] == "unknown"

        # Test with cloud_cover as negative (should still work)
        negative_cloud = {"cloud_cover": -0.01, "usable_data": None}
        result = processor._calculate_derived_metrics(negative_cloud)
        assert result["suitability"] == "excellent"  # Should still process

        # Test with cloud_cover > 1.0 (should be poor)
        extreme_cloud = {"cloud_cover": 1.5, "usable_data": None}
        result = processor._calculate_derived_metrics(extreme_cloud)
        assert result["suitability"] == "poor"

    def test_suitability_real_planet_api_scenarios(self, processor):
        """Test suitability with realistic Planet API data scenarios."""

        # Scenario 1: Milan example - excellent conditions
        milan_scene = {
            "cloud_cover": 0.0,
            "usable_data": None,  # Missing from Planet API
            "sun_elevation": 53.9,
            "month": 8,
        }
        result = processor._calculate_derived_metrics(milan_scene)
        assert result["suitability"] == "excellent"
        assert result["solar_conditions"] == "optimal"
        assert result["season"] == "summer"

        # Scenario 2: Slightly cloudy scene
        cloudy_scene = {
            "cloud_cover": 0.08,
            "usable_data": None,
            "sun_elevation": 45.0,
            "month": 6,
        }
        result = processor._calculate_derived_metrics(cloudy_scene)
        assert result["suitability"] == "good"

        # Scenario 3: Moderately cloudy scene
        moderate_scene = {
            "cloud_cover": 0.25,
            "usable_data": None,
            "sun_elevation": 40.0,
            "month": 3,
        }
        result = processor._calculate_derived_metrics(moderate_scene)
        assert result["suitability"] == "fair"

    def test_suitability_combined_vs_fallback_logic(self, processor):
        """Test that combined logic takes precedence over fallback when usable_data is available."""

        # When both are available, should use combined logic
        combined_data = {
            "cloud_cover": 0.18,
            "usable_data": 0.85,
        }  # Would be "good" in combined
        result = processor._calculate_derived_metrics(combined_data)
        assert (
            result["suitability"] == "good"
        )  # Combined: cloud_cover <= 0.2 AND usable_data >= 0.8

        # When usable_data is missing, should use fallback logic
        fallback_data = {
            "cloud_cover": 0.18,
            "usable_data": None,
        }  # Would be "fair" in fallback
        result = processor._calculate_derived_metrics(fallback_data)
        assert result["suitability"] == "fair"  # Fallback: 0.15 < cloud_cover <= 0.30

        # Verify the difference between combined and fallback thresholds
        edge_case = {
            "cloud_cover": 0.12,
            "usable_data": 0.75,
        }  # usable_data too low for "good"
        result = processor._calculate_derived_metrics(edge_case)
        assert (
            result["suitability"] == "fair"
        )  # Combined logic: doesn't meet "good" criteria

        edge_case_fallback = {"cloud_cover": 0.12, "usable_data": None}
        result = processor._calculate_derived_metrics(edge_case_fallback)
        assert result["suitability"] == "good"  # Fallback logic: cloud_cover <= 0.15

    def test_planet_api_field_substitution(self, processor):
        """Test using Planet API alternative fields when usable_data is missing."""

        # This test assumes you might implement clear_percent substitution
        # If you decide to use clear_percent as substitute for usable_data

        # Test that clear_percent could be used (if implemented)
        planet_data = {
            "cloud_cover": 0.0,
            "usable_data": None,
            "clear_percent": 100,  # Planet API field
            "visible_percent": 100,  # Planet API field
            "clear_confidence_percent": 97,  # Planet API field
        }

        # For now, this should use fallback logic
        result = processor._calculate_derived_metrics(planet_data)
        assert result["suitability"] == "excellent"  # Based on cloud_cover fallback


class TestQualityAssessment:
    """Test suite for quality assessment functionality."""

    @pytest.fixture
    def processor(self):
        return MetadataProcessor()

    def test_quality_thresholds(self, processor):
        """Test quality threshold definitions."""
        thresholds = processor.QUALITY_THRESHOLDS

        # Verify all required metrics have thresholds
        assert "cloud_cover" in thresholds
        assert "sun_elevation" in thresholds
        assert "usable_data" in thresholds

        # Verify threshold structure
        for metric, levels in thresholds.items():
            assert "excellent" in levels
            assert "good" in levels
            assert "fair" in levels
            assert "poor" in levels

    def test_quality_score_edge_cases(self, processor):
        """Test quality score calculation edge cases."""
        # Test with None values
        metadata_with_nones = {
            "cloud_cover": None,
            "sun_elevation": None,
            "usable_data": None,
            "ground_control": False,
        }

        scores = processor._calculate_quality_scores(metadata_with_nones)
        assert scores["ground_control"] == 0.8  # Only non-None value

        # Test with extreme values
        extreme_metadata = {
            "cloud_cover": 1.0,  # 100% cloud cover
            "sun_elevation": 0.0,  # Horizon
            "usable_data": 0.0,  # No usable data
            "ground_control": False,
        }

        scores = processor._calculate_quality_scores(extreme_metadata)
        assert scores["cloud_cover"] == 0.0
        assert scores["sun_elevation"] == 0.0
        assert scores["usable_data"] == 0.0

    def test_overall_quality_empty_scores(self, processor):
        """Test overall quality calculation with empty scores."""
        overall = processor._calculate_overall_quality({})
        assert overall == 0.0

        # Test with partial scores
        partial_scores = {"cloud_cover": 0.8}
        overall = processor._calculate_overall_quality(partial_scores)
        assert overall == 0.8  # Only cloud cover weight (0.4) normalized


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""

    @pytest.fixture
    def processor(self):
        return MetadataProcessor()

    def test_large_scene_collection_processing(self, processor):
        """Test processing of large scene collections."""
        # Create 1000 scenes
        large_collection = []
        for i in range(1000):
            scene = {
                "type": "Feature",
                "id": f"scene_{i}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.42 + i * 0.001, 37.77],
                            [-122.41 + i * 0.001, 37.77],
                            [-122.41 + i * 0.001, 37.78],
                            [-122.42 + i * 0.001, 37.78],
                            [-122.42 + i * 0.001, 37.77],
                        ]
                    ],
                },
                "properties": {
                    "id": f"scene_{i}",
                    "item_type": "PSScene",
                    "satellite_id": f"sat_{i % 10}",
                    "acquired": f"2024-{(i % 12) + 1:02d}-15T14:30:00.000Z",
                    "cloud_cover": (i % 100) / 100.0,
                    "sun_elevation": 30 + (i % 30),
                    "usable_data": 0.7 + (i % 30) / 100.0,
                },
            }
            large_collection.append(scene)

        # Test summary generation (should complete without errors)
        summary = processor.generate_metadata_summary(large_collection)

        assert summary["collection_overview"]["total_scenes"] == 1000
        assert summary["collection_overview"]["unique_satellites"] == 10
        assert len(summary["temporal_analysis"]["monthly_distribution"]) <= 12

    def test_memory_efficiency_with_large_collections(self, processor):
        """Test memory efficiency with large scene collections."""
        import sys

        # Test incremental processing to avoid memory issues
        scenes_batch_1 = [
            {
                "type": "Feature",
                "id": f"batch1_scene_{i}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.42, 37.77],
                            [-122.41, 37.77],
                            [-122.41, 37.78],
                            [-122.42, 37.78],
                            [-122.42, 37.77],
                        ]
                    ],
                },
                "properties": {
                    "id": f"batch1_scene_{i}",
                    "acquired": "2024-01-15T14:30:00.000Z",
                    "cloud_cover": 0.1,
                },
            }
            for i in range(100)
        ]

        # Process batch and verify memory doesn't grow excessively
        initial_size = sys.getsizeof(processor)

        for scene in scenes_batch_1:
            metadata = processor.extract_scene_metadata(scene)
            # Verify metadata is extracted properly
            assert metadata["scene_id"] is not None

        final_size = sys.getsizeof(processor)

        # Processor size shouldn't grow significantly
        assert final_size - initial_size < 1000  # Allow some growth but not excessive


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @pytest.fixture
    def processor(self):
        return MetadataProcessor()

    def test_malformed_scene_data(self, processor):
        """Test handling of malformed scene data."""
        malformed_scenes = [
            None,  # None value
            "string",  # String instead of dict
            {"invalid": "structure"},  # Missing required fields
            {"properties": None},  # None properties
        ]

        for malformed_scene in malformed_scenes:
            with pytest.raises((ValidationError, PlanetScopeError, AttributeError)):
                processor.extract_scene_metadata(malformed_scene)

    def test_invalid_date_formats(self, processor):
        """Test handling of invalid date formats."""
        scene_with_bad_date = {
            "type": "Feature",
            "id": "bad_date_scene",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-122.42, 37.77],
                        [-122.41, 37.77],
                        [-122.41, 37.78],
                        [-122.42, 37.78],
                        [-122.42, 37.77],
                    ]
                ],
            },
            "properties": {
                "id": "bad_date_scene",
                "acquired": "invalid-date-format",
                "cloud_cover": 0.1,
            },
        }

        # Should handle gracefully without crashing
        metadata = processor.extract_scene_metadata(scene_with_bad_date)
        assert metadata["scene_id"] == "bad_date_scene"
        # Date-related fields might be missing or None, but shouldn't crash

    def test_extreme_coordinate_values(self, processor):
        """Test handling of extreme coordinate values."""
        scene_with_extreme_coords = {
            "type": "Feature",
            "id": "extreme_coords_scene",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[180, 90], [179, 90], [179, 89], [180, 89], [180, 90]]
                ],  # Near pole
            },
            "properties": {
                "id": "extreme_coords_scene",
                "acquired": "2024-01-15T14:30:00.000Z",
                "cloud_cover": 0.1,
            },
        }

        # Should handle extreme coordinates without error
        metadata = processor.extract_scene_metadata(scene_with_extreme_coords)
        assert metadata["scene_id"] == "extreme_coords_scene"
        assert "bounds" in metadata
        assert "centroid" in metadata

    def test_empty_collections_handling(self, processor):
        """Test handling of empty scene collections."""
        # Empty list
        assessment = processor.assess_coverage_quality([])
        assert assessment["total_scenes"] == 0

        summary = processor.generate_metadata_summary([])
        assert "error" in summary

        # Filter empty collection
        filtered, stats = processor.filter_by_metadata_criteria([], {})
        assert len(filtered) == 0
        assert stats["original_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__])
