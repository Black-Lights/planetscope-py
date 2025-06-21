#!/usr/bin/env python3
"""Metadata processing and analysis for Planet scenes.

This module provides comprehensive metadata extraction, quality assessment,
and coverage statistics for Planet imagery following RASD specifications.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import defaultdict
import statistics

import numpy as np
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely.ops import unary_union

from .config import default_config
from .exceptions import ValidationError, PlanetScopeError
from .utils import validate_geometry, calculate_area_km2

logger = logging.getLogger(__name__)


class MetadataProcessor:
    """Advanced metadata processing and analysis for Planet scenes.

    Provides comprehensive metadata extraction, quality assessment,
    coverage analysis, and statistical reporting capabilities.

    Attributes:
        config: Configuration settings
        quality_thresholds: Quality assessment thresholds
    """

    # Quality assessment thresholds
    QUALITY_THRESHOLDS = {
        "cloud_cover": {"excellent": 0.05, "good": 0.15, "fair": 0.30, "poor": 1.0},
        "sun_elevation": {"excellent": 45.0, "good": 30.0, "fair": 15.0, "poor": 0.0},
        "usable_data": {"excellent": 0.95, "good": 0.85, "fair": 0.70, "poor": 0.0},
    }

    def __init__(self, config: Optional[Dict] = None):
        """Initialize metadata processor.

        Args:
            config: Custom configuration settings (optional)
        """
        self.config = default_config
        if config:
            for key, value in config.items():
                self.config.set(key, value)

        logger.info("MetadataProcessor initialized")

    def extract_scene_metadata(self, scene: Dict) -> Dict:
        """Extract comprehensive metadata from a Planet scene.

        Args:
            scene: Planet scene feature from search results

        Returns:
            Dictionary containing extracted and enhanced metadata

        Raises:
            ValidationError: Invalid scene structure
        """
        # Add proper validation
        if not isinstance(scene, dict):
            raise ValidationError("Scene must be a dictionary")

        if not scene:  # Empty dict
            raise ValidationError("Scene dictionary cannot be empty")

        # Check for required structure
        if "properties" not in scene and "id" not in scene:
            raise ValidationError("Scene must have properties or id field")

        try:
            properties = scene.get("properties", {})
            geometry = scene.get("geometry", {})

            # Basic scene information
            metadata = {
                "scene_id": properties.get("id"),
                "item_type": properties.get("item_type"),
                "satellite_id": properties.get("satellite_id"),
                "provider": properties.get("provider", "planet"),
                "published": properties.get("published"),
                "updated": properties.get("updated"),
            }

            # Acquisition information with bulletproof error handling
            acquired = properties.get("acquired")
            if acquired:
                try:
                    acq_datetime = datetime.fromisoformat(
                        acquired.replace("Z", "+00:00")
                    )
                    metadata.update(
                        {
                            "acquired": acquired,
                            "acquisition_date": acq_datetime.date().isoformat(),
                            "acquisition_time": acq_datetime.time().isoformat(),
                            "day_of_year": acq_datetime.timetuple().tm_yday,
                            "week_of_year": acq_datetime.isocalendar()[1],
                            "month": acq_datetime.month,
                            "year": acq_datetime.year,
                        }
                    )
                except Exception as e:
                    # Handle ANY date parsing error gracefully
                    logger.warning(f"Invalid date format '{acquired}': {e}")
                    metadata.update(
                        {
                            "acquired": acquired,
                            "acquisition_date": None,
                            "acquisition_time": None,
                            "day_of_year": None,
                            "week_of_year": None,
                            "month": None,
                            "year": None,
                        }
                    )
            else:
                # No acquired date provided
                metadata.update(
                    {
                        "acquired": None,
                        "acquisition_date": None,
                        "acquisition_time": None,
                        "day_of_year": None,
                        "week_of_year": None,
                        "month": None,
                        "year": None,
                    }
                )

            # Quality metrics with safe type conversion
            cloud_cover = properties.get("cloud_cover")
            sun_elevation = properties.get("sun_elevation")
            usable_data = properties.get("usable_data")

            # Convert and validate numeric fields safely
            try:
                if cloud_cover is not None:
                    cloud_cover = float(cloud_cover)
            except (TypeError, ValueError):
                cloud_cover = None

            try:
                if sun_elevation is not None:
                    sun_elevation = float(sun_elevation)
            except (TypeError, ValueError):
                sun_elevation = None

            try:
                if usable_data is not None:
                    usable_data = float(usable_data)
            except (TypeError, ValueError):
                usable_data = None

            metadata.update(
                {
                    "cloud_cover": cloud_cover,
                    "sun_azimuth": properties.get("sun_azimuth"),
                    "sun_elevation": sun_elevation,
                    "usable_data": usable_data,
                    "quality_category": properties.get("quality_category", "standard"),
                    "pixel_resolution": properties.get("pixel_resolution"),
                    "ground_control": properties.get("ground_control", False),
                }
            )

            # Geometric information - safely extract with comprehensive error handling
            if geometry:
                try:
                    geom_metadata = self._extract_geometry_metadata(geometry)
                    metadata.update(geom_metadata)
                except Exception as e:
                    logger.warning(
                        f"Error extracting geometry metadata for scene {metadata.get('scene_id', 'unknown')}: {e}"
                    )
                    # Add safe defaults for all geometric fields
                    metadata.update(
                        {
                            "geometry_type": "Unknown",
                            "area_km2": 0.0,
                            "bounds": {
                                "west": 0.0,
                                "south": 0.0,
                                "east": 0.0,
                                "north": 0.0,
                            },
                            "centroid": {"longitude": 0.0, "latitude": 0.0},
                            "perimeter_km": 0.0,
                            "aspect_ratio": 1.0,
                        }
                    )

            # Calculate quality scores with comprehensive error handling
            try:
                metadata["quality_scores"] = self._calculate_quality_scores(metadata)
                metadata["overall_quality"] = self._calculate_overall_quality(
                    metadata["quality_scores"]
                )
            except Exception as e:
                logger.warning(
                    f"Error calculating quality scores for scene {metadata.get('scene_id', 'unknown')}: {e}"
                )
                metadata["quality_scores"] = {}
                metadata["overall_quality"] = 0.0

            # Additional derived metrics - with bulletproof error handling
            try:
                derived_metrics = self._calculate_derived_metrics(metadata)
                metadata.update(derived_metrics)
            except Exception as e:
                logger.warning(
                    f"Error calculating derived metrics for scene {metadata.get('scene_id', 'unknown')}: {e}"
                )
                # Provide bulletproof safe defaults
                metadata.update(
                    {
                        "season": "unknown",
                        "solar_conditions": "unknown",
                        "suitability": "unknown",
                    }
                )

            return metadata

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Error extracting scene metadata: {str(e)}")

    def assess_coverage_quality(
        self, scenes: List[Dict], target_geometry: Optional[Union[Dict, Polygon]] = None
    ) -> Dict:
        """Assess coverage quality for a collection of scenes.

        Args:
            scenes: List of Planet scene features
            target_geometry: Target area geometry for coverage analysis

        Returns:
            Dictionary containing comprehensive coverage assessment
        """
        if not scenes:
            return {
                "total_scenes": 0,
                "coverage_area_km2": 0.0,
                "temporal_span_days": 0,
                "quality_distribution": {},
                "recommendations": ["No scenes available for analysis"],
            }

        # Extract metadata for all scenes
        scene_metadata = [self.extract_scene_metadata(scene) for scene in scenes]

        # Calculate coverage statistics
        coverage_stats = self._calculate_coverage_statistics(
            scene_metadata, target_geometry
        )

        # Temporal analysis
        temporal_stats = self._analyze_temporal_distribution(scene_metadata)

        # Quality assessment
        quality_stats = self._analyze_quality_distribution(scene_metadata)

        # Generate recommendations
        recommendations = self._generate_coverage_recommendations(
            coverage_stats, temporal_stats, quality_stats
        )

        return {
            "total_scenes": len(scenes),
            "coverage_statistics": coverage_stats,
            "temporal_analysis": temporal_stats,
            "quality_analysis": quality_stats,
            "recommendations": recommendations,
        }

    def filter_by_metadata_criteria(
        self, scenes: List[Dict], criteria: Dict
    ) -> Tuple[List[Dict], Dict]:
        """Filter scenes based on metadata criteria and collect comprehensive rejection statistics.

        This method evaluates each scene against all specified criteria and collects
        statistics on why scenes were rejected. Unlike methods that stop at the first
        failing criterion, this checks all criteria to provide complete rejection analytics.

        Args:
            scenes (List[Dict]): List of Planet scene features containing 'properties'
                            and 'geometry' fields. Each scene should be a valid GeoJSON
                            Feature object from Planet API search results.
            criteria (Dict): Filtering criteria dictionary containing one or more of:
                            - 'max_cloud_cover' (float): Maximum allowed cloud cover (0.0-1.0)
                            - 'min_sun_elevation' (float): Minimum sun elevation in degrees
                            - 'min_usable_data' (float): Minimum usable data fraction (0.0-1.0)
                            - 'quality_category' (str): Required quality category
                            - 'min_overall_quality' (float): Minimum overall quality score

        Returns:
            Tuple[List[Dict], Dict]: A tuple containing:
                - filtered_scenes: List of scenes that passed all criteria
                - filter_statistics: Dictionary with the following structure:
                    {
                        'original_count': int,     # Total input scenes
                        'filtered_count': int,     # Scenes that passed all criteria
                        'retention_rate': float,   # Ratio of passed/total scenes
                        'rejection_reasons': {     # Count of each rejection type
                            'cloud_cover_exceeded': int,
                            'sun_elevation_too_low': int,
                            'insufficient_usable_data': int,
                            'quality_category_mismatch': int,
                            'overall_quality_insufficient': int
                        }
                    }

        Example:
            >>> criteria = {
            ...     'max_cloud_cover': 0.2,
            ...     'min_sun_elevation': 40.0,
            ...     'min_usable_data': 0.85
            ... }
            >>> filtered, stats = processor.filter_by_metadata_criteria(scenes, criteria)
            >>> print(f"Kept {stats['filtered_count']} of {stats['original_count']} scenes")
            >>> print(f"Rejections: {dict(stats['rejection_reasons'])}")

        Note:
            - Scenes with missing metadata values are not rejected for those criteria
            - A single scene can contribute to multiple rejection reason counts
            - The rejection_reasons uses defaultdict(int) for automatic zero initialization
            - If a scene fails multiple criteria, all applicable rejection reasons are counted

        Raises:
            ValidationError: If scene metadata extraction fails due to invalid scene structure
            Exception: Re-raises any exceptions from extract_scene_metadata()
        """
        filtered_scenes = []
        filter_stats = {
            "original_count": len(scenes),
            "filtered_count": 0,
            "rejection_reasons": defaultdict(int),
        }

        for scene in scenes:
            metadata = self.extract_scene_metadata(scene)

            # Check ALL criteria for each scene to get complete rejection statistics
            scene_passed = True
            rejection_reasons = []

            # Check cloud cover
            max_cloud_cover = criteria.get("max_cloud_cover")
            if max_cloud_cover is not None:
                cloud_cover = metadata.get("cloud_cover")
                if cloud_cover is not None and cloud_cover > max_cloud_cover:
                    scene_passed = False
                    rejection_reasons.append("cloud_cover_exceeded")

            # Check sun elevation
            min_sun_elevation = criteria.get("min_sun_elevation")
            if min_sun_elevation is not None:
                sun_elevation = metadata.get("sun_elevation")
                if sun_elevation is not None and sun_elevation < min_sun_elevation:
                    scene_passed = False
                    rejection_reasons.append("sun_elevation_too_low")

            # Check usable data
            min_usable_data = criteria.get("min_usable_data")
            if min_usable_data is not None:
                usable_data = metadata.get("usable_data")
                if usable_data is not None and usable_data < min_usable_data:
                    scene_passed = False
                    rejection_reasons.append("insufficient_usable_data")

            # Check quality category
            required_quality = criteria.get("quality_category")
            if required_quality is not None:
                quality_category = metadata.get("quality_category")
                if quality_category != required_quality:
                    scene_passed = False
                    rejection_reasons.append("quality_category_mismatch")

            # Check overall quality threshold
            min_overall_quality = criteria.get("min_overall_quality")
            if min_overall_quality is not None:
                overall_quality = metadata.get("overall_quality")
                if (
                    overall_quality is not None
                    and overall_quality < min_overall_quality
                ):
                    scene_passed = False
                    rejection_reasons.append("overall_quality_insufficient")

            # Add scene to filtered list if it passed all criteria
            if scene_passed:
                filtered_scenes.append(scene)
            else:
                # Count each type of rejection reason that occurred
                for reason in rejection_reasons:
                    filter_stats["rejection_reasons"][reason] += 1

        filter_stats["filtered_count"] = len(filtered_scenes)
        filter_stats["retention_rate"] = (
            filter_stats["filtered_count"] / filter_stats["original_count"]
            if filter_stats["original_count"] > 0
            else 0.0
        )

        return filtered_scenes, filter_stats

    def generate_metadata_summary(self, scenes: List[Dict]) -> Dict:
        """Generate comprehensive metadata summary for scene collection.

        Args:
            scenes: List of Planet scene features

        Returns:
            Dictionary containing detailed metadata summary
        """
        if not scenes:
            return {"error": "No scenes provided for analysis"}

        # Extract all metadata
        all_metadata = [self.extract_scene_metadata(scene) for scene in scenes]

        summary = {
            "collection_overview": self._create_collection_overview(all_metadata),
            "temporal_analysis": self._analyze_temporal_distribution(all_metadata),
            "quality_analysis": self._analyze_quality_distribution(all_metadata),
            "spatial_analysis": self._analyze_spatial_distribution(all_metadata),
            "satellite_analysis": self._analyze_satellite_distribution(all_metadata),
        }

        return summary

    def _extract_geometry_metadata(self, geometry: Dict) -> Dict:
        """Extract metadata from scene geometry with bulletproof error handling.

        Args:
            geometry: GeoJSON geometry from scene

        Returns:
            Dictionary containing geometric metadata
        """
        try:
            geom = shape(geometry)

            # Basic geometric properties
            bounds = geom.bounds  # (minx, miny, maxx, maxy)
            centroid = geom.centroid

            # Safely calculate aspect ratio with comprehensive validation
            aspect_ratio = 1.0  # Safe default
            try:
                if len(bounds) >= 4:
                    width = bounds[2] - bounds[0]
                    height = bounds[3] - bounds[1]

                    # Check for valid numeric bounds and non-zero height
                    if (
                        isinstance(width, (int, float))
                        and isinstance(height, (int, float))
                        and height != 0
                        and width >= 0
                        and height > 0
                    ):
                        aspect_ratio = width / height
            except Exception:
                aspect_ratio = 1.0  # Fallback on any calculation error

            # Safely calculate area
            try:
                area_km2 = calculate_area_km2(geometry)
            except Exception:
                area_km2 = 0.0

            return {
                "geometry_type": geom.geom_type,
                "area_km2": area_km2,
                "bounds": {
                    "west": bounds[0] if len(bounds) > 0 else 0.0,
                    "south": bounds[1] if len(bounds) > 1 else 0.0,
                    "east": bounds[2] if len(bounds) > 2 else 0.0,
                    "north": bounds[3] if len(bounds) > 3 else 0.0,
                },
                "centroid": {
                    "longitude": centroid.x if hasattr(centroid, "x") else 0.0,
                    "latitude": centroid.y if hasattr(centroid, "y") else 0.0,
                },
                "perimeter_km": (
                    geom.length * 111.32 if hasattr(geom, "length") else 0.0
                ),
                "aspect_ratio": aspect_ratio,
            }

        except Exception as e:
            logger.warning(f"Error extracting geometry metadata: {e}")
            return {
                "geometry_type": "Unknown",
                "area_km2": 0.0,
                "bounds": {"west": 0.0, "south": 0.0, "east": 0.0, "north": 0.0},
                "centroid": {"longitude": 0.0, "latitude": 0.0},
                "perimeter_km": 0.0,
                "aspect_ratio": 1.0,
            }

    def _calculate_quality_scores(self, metadata: Dict) -> Dict:
        """Calculate quality scores for individual metrics.

        Args:
            metadata: Scene metadata dictionary

        Returns:
            Dictionary containing quality scores (0-1)
        """
        scores = {}

        # Cloud cover score
        cloud_cover = metadata.get("cloud_cover")
        if cloud_cover is not None:
            scores["cloud_cover"] = max(0, 1 - cloud_cover)

        # Sun elevation score
        sun_elevation = metadata.get("sun_elevation")
        if sun_elevation is not None:
            scores["sun_elevation"] = min(1, max(0, sun_elevation / 60.0))

        # Usable data score
        usable_data = metadata.get("usable_data")
        if usable_data is not None:
            scores["usable_data"] = usable_data

        # Ground control bonus
        if metadata.get("ground_control"):
            scores["ground_control"] = 1.0
        else:
            scores["ground_control"] = 0.8

        return scores

    def _calculate_overall_quality(self, quality_scores: Dict) -> float:
        """Calculate overall quality score from individual metrics with proper precision handling.

        Computes a weighted average of quality components, handling floating point precision
        issues by rounding the final result to avoid artifacts like 0.8000000000000002.

        Args:
            quality_scores (Dict): Dictionary of individual quality scores (0-1) containing:
                                - 'cloud_cover': Cloud cover quality score (higher = better)
                                - 'sun_elevation': Sun elevation quality score (higher = better)
                                - 'usable_data': Usable data fraction quality score
                                - 'ground_control': Ground control availability score

        Returns:
            float: Overall quality score (0-1), rounded to 10 decimal places to avoid
                floating point precision artifacts. Returns 0.0 if no valid scores provided.

        Example:
            >>> quality_scores = {"cloud_cover": 0.8}
            >>> processor._calculate_overall_quality(quality_scores)
            0.8

            >>> quality_scores = {
            ...     "cloud_cover": 0.9,
            ...     "sun_elevation": 0.75,
            ...     "usable_data": 0.9,
            ...     "ground_control": 1.0
            ... }
            >>> processor._calculate_overall_quality(quality_scores)
            0.88

        Note:
            - Uses weighted average with weights: cloud_cover=0.4, sun_elevation=0.2,
            usable_data=0.3, ground_control=0.1
            - Only includes metrics that are present in quality_scores and have valid weights
            - Returns 0.0 for empty input or when no weighted metrics are available
            - Rounds result to 10 decimal places to eliminate floating point precision issues
        """
        if not quality_scores:
            return 0.0

        # Weighted average of quality components
        weights = {
            "cloud_cover": 0.4,
            "sun_elevation": 0.2,
            "usable_data": 0.3,
            "ground_control": 0.1,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for metric, score in quality_scores.items():
            if metric in weights and score is not None:
                weighted_sum += score * weights[metric]
                total_weight += weights[metric]

        if total_weight == 0:
            return 0.0

        # Round to 10 decimal places to avoid floating point precision issues
        # This ensures that 0.8 * 0.4 / 0.4 = 0.8 exactly, not 0.8000000000000002
        result = weighted_sum / total_weight
        return round(result, 10)

    def _calculate_derived_metrics(self, metadata: Dict) -> Dict:
        """Calculate additional derived metrics from metadata with bulletproof error handling.

        Args:
            metadata: Basic scene metadata

        Returns:
            Dictionary containing derived metrics
        """
        derived = {}

        # Season calculation with bulletproof error handling
        try:
            month = metadata.get("month")
            if month is not None and isinstance(month, (int, float)):
                month = int(month)  # Ensure it's an integer
                if 1 <= month <= 12:  # Validate month range
                    if month in [12, 1, 2]:
                        derived["season"] = "winter"
                    elif month in [3, 4, 5]:
                        derived["season"] = "spring"
                    elif month in [6, 7, 8]:
                        derived["season"] = "summer"
                    else:  # [9, 10, 11]
                        derived["season"] = "autumn"
                else:
                    derived["season"] = "unknown"
            else:
                derived["season"] = "unknown"
        except Exception:
            derived["season"] = "unknown"

        # Solar conditions with bulletproof error handling
        try:
            sun_elevation = metadata.get("sun_elevation")
            if sun_elevation is not None and isinstance(sun_elevation, (int, float)):
                sun_elevation = float(sun_elevation)
                if sun_elevation >= 45:
                    derived["solar_conditions"] = "optimal"
                elif sun_elevation >= 30:
                    derived["solar_conditions"] = "good"
                elif sun_elevation >= 15:
                    derived["solar_conditions"] = "marginal"
                else:
                    derived["solar_conditions"] = "poor"
            else:
                derived["solar_conditions"] = "unknown"
        except Exception:
            derived["solar_conditions"] = "unknown"

        # Image suitability classification with bulletproof error handling
        try:
            cloud_cover = metadata.get("cloud_cover")
            usable_data = metadata.get("usable_data")

            # Check if we have both values (preferred method)
            if (
                cloud_cover is not None
                and usable_data is not None
                and isinstance(cloud_cover, (int, float))
                and isinstance(usable_data, (int, float))
            ):

                cloud_cover = float(cloud_cover)
                usable_data = float(usable_data)

                if cloud_cover <= 0.1 and usable_data >= 0.9:
                    derived["suitability"] = "excellent"
                elif cloud_cover <= 0.2 and usable_data >= 0.8:
                    derived["suitability"] = "good"
                elif cloud_cover <= 0.4 and usable_data >= 0.7:
                    derived["suitability"] = "fair"
                else:
                    derived["suitability"] = "poor"

            # Fallback: Use cloud cover alone if usable_data is missing
            elif cloud_cover is not None and isinstance(cloud_cover, (int, float)):
                cloud_cover = float(cloud_cover)

                # More lenient thresholds since we only have cloud cover
                if cloud_cover <= 0.05:
                    derived["suitability"] = "excellent"
                elif cloud_cover <= 0.15:
                    derived["suitability"] = "good"
                elif cloud_cover <= 0.30:
                    derived["suitability"] = "fair"
                else:
                    derived["suitability"] = "poor"
            else:
                derived["suitability"] = "unknown"
        except Exception:
            derived["suitability"] = "unknown"

        return derived

    def _calculate_coverage_statistics(
        self,
        scene_metadata: List[Dict],
        target_geometry: Optional[Union[Dict, Polygon]],
    ) -> Dict:
        """Calculate coverage statistics for scene collection.

        Args:
            scene_metadata: List of extracted scene metadata
            target_geometry: Target area for coverage analysis

        Returns:
            Dictionary containing coverage statistics
        """
        stats = {
            "total_area_km2": 0.0,
            "unique_coverage_km2": 0.0,
            "overlap_factor": 1.0,
            "coverage_efficiency": 0.0,
        }

        # Calculate total and unique coverage areas
        scene_geometries = []
        total_area = 0.0

        for metadata in scene_metadata:
            area = metadata.get("area_km2", 0.0)
            total_area += area

            # Collect geometries for union calculation
            bounds = metadata.get("bounds")
            if bounds:
                # Create polygon from bounds
                poly = Polygon(
                    [
                        (bounds["west"], bounds["south"]),
                        (bounds["east"], bounds["south"]),
                        (bounds["east"], bounds["north"]),
                        (bounds["west"], bounds["north"]),
                    ]
                )
                scene_geometries.append(poly)

        stats["total_area_km2"] = total_area

        # Calculate unique coverage area
        if scene_geometries:
            try:
                union_geom = unary_union(scene_geometries)
                unique_area = calculate_area_km2(union_geom.__geo_interface__)
                stats["unique_coverage_km2"] = unique_area

                if unique_area > 0:
                    stats["overlap_factor"] = total_area / unique_area

            except Exception as e:
                logger.warning(f"Error calculating coverage union: {e}")

        # Calculate coverage efficiency vs target
        if target_geometry:
            try:
                target_geom = validate_geometry(target_geometry)
                target_area = calculate_area_km2(target_geom)

                if target_area > 0:
                    stats["coverage_efficiency"] = min(
                        1.0, stats["unique_coverage_km2"] / target_area
                    )
                    stats["target_area_km2"] = target_area

            except Exception as e:
                logger.warning(f"Error calculating target coverage: {e}")

        return stats

    def _analyze_temporal_distribution(self, scene_metadata: List[Dict]) -> Dict:
        """Analyze temporal distribution of scenes.

        Args:
            scene_metadata: List of extracted scene metadata

        Returns:
            Dictionary containing temporal analysis
        """
        if not scene_metadata:
            return {}

        # Extract acquisition dates
        acquisition_dates = []
        for metadata in scene_metadata:
            acquired = metadata.get("acquired")
            if acquired:
                try:
                    date = datetime.fromisoformat(acquired.replace("Z", "+00:00"))
                    acquisition_dates.append(date)
                except:
                    continue

        if not acquisition_dates:
            return {"error": "No valid acquisition dates found"}

        # Sort dates
        acquisition_dates.sort()

        # Calculate temporal statistics
        temporal_stats = {
            "date_range": {
                "start": acquisition_dates[0].isoformat(),
                "end": acquisition_dates[-1].isoformat(),
            },
            "span_days": (acquisition_dates[-1] - acquisition_dates[0]).days,
            "total_scenes": len(acquisition_dates),
            "average_interval_days": 0.0,
        }

        # Calculate average interval between acquisitions
        if len(acquisition_dates) > 1:
            intervals = [
                (acquisition_dates[i] - acquisition_dates[i - 1]).days
                for i in range(1, len(acquisition_dates))
            ]
            temporal_stats["average_interval_days"] = statistics.mean(intervals)
            temporal_stats["median_interval_days"] = statistics.median(intervals)
            temporal_stats["min_interval_days"] = min(intervals)
            temporal_stats["max_interval_days"] = max(intervals)

        # Monthly distribution
        monthly_counts = defaultdict(int)
        seasonal_counts = defaultdict(int)

        for date in acquisition_dates:
            month_key = f"{date.year}-{date.month:02d}"
            monthly_counts[month_key] += 1

            # Season calculation
            month = date.month
            if month in [12, 1, 2]:
                seasonal_counts["winter"] += 1
            elif month in [3, 4, 5]:
                seasonal_counts["spring"] += 1
            elif month in [6, 7, 8]:
                seasonal_counts["summer"] += 1
            else:
                seasonal_counts["autumn"] += 1

        temporal_stats["monthly_distribution"] = dict(monthly_counts)
        temporal_stats["seasonal_distribution"] = dict(seasonal_counts)

        return temporal_stats

    def _analyze_quality_distribution(self, scene_metadata: List[Dict]) -> Dict:
        """Analyze quality distribution across scenes.

        Args:
            scene_metadata: List of extracted scene metadata

        Returns:
            Dictionary containing quality analysis
        """
        if not scene_metadata:
            return {}

        # Collect quality metrics
        cloud_covers = []
        sun_elevations = []
        usable_data_values = []
        overall_qualities = []
        suitability_counts = defaultdict(int)

        for metadata in scene_metadata:
            cloud_cover = metadata.get("cloud_cover")
            if cloud_cover is not None:
                cloud_covers.append(cloud_cover)

            sun_elevation = metadata.get("sun_elevation")
            if sun_elevation is not None:
                sun_elevations.append(sun_elevation)

            usable_data = metadata.get("usable_data")
            if usable_data is not None:
                usable_data_values.append(usable_data)

            overall_quality = metadata.get("overall_quality")
            if overall_quality is not None:
                overall_qualities.append(overall_quality)

            suitability = metadata.get("suitability")
            if suitability:
                suitability_counts[suitability] += 1

        # Calculate statistics
        quality_stats = {"suitability_distribution": dict(suitability_counts)}

        # Cloud cover statistics
        if cloud_covers:
            quality_stats["cloud_cover"] = {
                "mean": statistics.mean(cloud_covers),
                "median": statistics.median(cloud_covers),
                "min": min(cloud_covers),
                "max": max(cloud_covers),
                "std": statistics.stdev(cloud_covers) if len(cloud_covers) > 1 else 0.0,
            }

        # Sun elevation statistics
        if sun_elevations:
            quality_stats["sun_elevation"] = {
                "mean": statistics.mean(sun_elevations),
                "median": statistics.median(sun_elevations),
                "min": min(sun_elevations),
                "max": max(sun_elevations),
            }

        # Overall quality statistics
        if overall_qualities:
            quality_stats["overall_quality"] = {
                "mean": statistics.mean(overall_qualities),
                "median": statistics.median(overall_qualities),
                "min": min(overall_qualities),
                "max": max(overall_qualities),
            }

        return quality_stats

    def _analyze_spatial_distribution(self, scene_metadata: List[Dict]) -> Dict:
        """Analyze spatial distribution of scenes using actual geometry bounds, not centroids.

        Computes spatial statistics including extent bounds (derived from actual scene boundaries),
        centroid locations, and area distributions. Uses the actual bounds from each scene's
        geometry rather than centroid coordinates to determine the true spatial extent.

        Args:
            scene_metadata (List[Dict]): List of extracted scene metadata dictionaries.
                                    Each should contain:
                                    - 'area_km2': Scene area in square kilometers
                                    - 'bounds': Dict with 'west', 'east', 'south', 'north' coordinates
                                    - 'centroid': Dict with 'longitude' and 'latitude' (for center calc)
                                    - 'aspect_ratio': Width/height ratio (optional)

        Returns:
            Dict: Spatial statistics dictionary containing:
                - 'area_km2': Area statistics (total, mean, median, min, max)
                - 'extent': Overall bounding box using actual scene bounds (west, east, south, north)
                - 'center': Geographic center point calculated from centroids (longitude, latitude)

                Returns empty dict if no valid scene_metadata provided.

        Example:
            >>> scene_metadata = [
            ...     {
            ...         'area_km2': 10.5,
            ...         'bounds': {'west': -122.42, 'east': -122.41, 'south': 37.77, 'north': 37.78},
            ...         'centroid': {'longitude': -122.415, 'latitude': 37.775}
            ...     },
            ...     {
            ...         'area_km2': 8.3,
            ...         'bounds': {'west': -122.41, 'east': -122.40, 'south': 37.77, 'north': 37.78},
            ...         'centroid': {'longitude': -122.405, 'latitude': 37.775}
            ...     }
            ... ]
            >>> stats = processor._analyze_spatial_distribution(scene_metadata)
            >>> stats['extent']['west']  # Uses actual bounds, not centroid
            -122.42
            >>> stats['center']['longitude']  # Uses centroid average
            -122.41

        Note:
            - Extent calculation uses actual scene bounds, not centroids, for accurate coverage
            - Center calculation uses centroids for representative geographic center
            - Coordinates rounded to 8 decimal places (~1mm precision) for consistency
            - Gracefully handles missing bounds or centroid information
            - Only processes scenes with valid data for each calculation
        """
        if not scene_metadata:
            return {}

        # Collect spatial metrics
        areas = []
        centroids = []
        bounds_list = []
        aspect_ratios = []

        for metadata in scene_metadata:
            # Collect area data
            area = metadata.get("area_km2")
            if area is not None:
                areas.append(area)

            # Collect centroid data (for center calculation)
            centroid = metadata.get("centroid")
            if centroid and "longitude" in centroid and "latitude" in centroid:
                centroids.append(
                    {
                        "longitude": round(centroid["longitude"], 8),
                        "latitude": round(centroid["latitude"], 8),
                    }
                )

            # Collect bounds data (for extent calculation)
            bounds = metadata.get("bounds")
            if bounds and all(
                key in bounds for key in ["west", "east", "south", "north"]
            ):
                bounds_list.append(
                    {
                        "west": round(bounds["west"], 8),
                        "east": round(bounds["east"], 8),
                        "south": round(bounds["south"], 8),
                        "north": round(bounds["north"], 8),
                    }
                )

            # Collect aspect ratio data
            aspect_ratio = metadata.get("aspect_ratio")
            if aspect_ratio is not None:
                aspect_ratios.append(aspect_ratio)

        spatial_stats = {}

        # Area statistics
        if areas:
            spatial_stats["area_km2"] = {
                "total": sum(areas),
                "mean": statistics.mean(areas),
                "median": statistics.median(areas),
                "min": min(areas),
                "max": max(areas),
            }

        # Extent analysis using actual scene bounds (not centroids)
        if bounds_list:
            all_west = [b["west"] for b in bounds_list]
            all_east = [b["east"] for b in bounds_list]
            all_south = [b["south"] for b in bounds_list]
            all_north = [b["north"] for b in bounds_list]

            spatial_stats["extent"] = {
                "west": round(min(all_west), 8),  # Westernmost point across all scenes
                "east": round(max(all_east), 8),  # Easternmost point across all scenes
                "south": round(
                    min(all_south), 8
                ),  # Southernmost point across all scenes
                "north": round(
                    max(all_north), 8
                ),  # Northernmost point across all scenes
            }

        # Center calculation using centroids
        if centroids:
            lons = [c["longitude"] for c in centroids]
            lats = [c["latitude"] for c in centroids]

            spatial_stats["center"] = {
                "longitude": round(statistics.mean(lons), 8),
                "latitude": round(statistics.mean(lats), 8),
            }

        return spatial_stats

    def _analyze_satellite_distribution(self, scene_metadata: List[Dict]) -> Dict:
        """Analyze satellite and item type distribution.

        Args:
            scene_metadata: List of extracted scene metadata

        Returns:
            Dictionary containing satellite analysis
        """
        satellite_counts = defaultdict(int)
        item_type_counts = defaultdict(int)

        for metadata in scene_metadata:
            satellite_id = metadata.get("satellite_id")
            if satellite_id:
                satellite_counts[satellite_id] += 1

            item_type = metadata.get("item_type")
            if item_type:
                item_type_counts[item_type] += 1

        return {
            "satellite_distribution": dict(satellite_counts),
            "item_type_distribution": dict(item_type_counts),
        }

    def _check_metadata_criteria(
        self, metadata: Dict, criteria: Dict
    ) -> Tuple[bool, str]:
        """Check if metadata meets specified criteria.

        Args:
            metadata: Scene metadata
            criteria: Filtering criteria

        Returns:
            Tuple of (passed, rejection_reason)
        """
        # Cloud cover check
        max_cloud_cover = criteria.get("max_cloud_cover")
        if max_cloud_cover is not None:
            cloud_cover = metadata.get("cloud_cover")
            if cloud_cover is not None and cloud_cover > max_cloud_cover:
                return False, "cloud_cover_exceeded"

        # Sun elevation check
        min_sun_elevation = criteria.get("min_sun_elevation")
        if min_sun_elevation is not None:
            sun_elevation = metadata.get("sun_elevation")
            if sun_elevation is not None and sun_elevation < min_sun_elevation:
                return False, "sun_elevation_too_low"

        # Usable data check
        min_usable_data = criteria.get("min_usable_data")
        if min_usable_data is not None:
            usable_data = metadata.get("usable_data")
            if usable_data is not None and usable_data < min_usable_data:
                return False, "insufficient_usable_data"

        # Quality category check
        required_quality = criteria.get("quality_category")
        if required_quality is not None:
            quality_category = metadata.get("quality_category")
            if quality_category != required_quality:
                return False, "quality_category_mismatch"

        # Overall quality threshold
        min_overall_quality = criteria.get("min_overall_quality")
        if min_overall_quality is not None:
            overall_quality = metadata.get("overall_quality")
            if overall_quality is not None and overall_quality < min_overall_quality:
                return False, "overall_quality_insufficient"

        return True, ""

    def _generate_coverage_recommendations(
        self, coverage_stats: Dict, temporal_stats: Dict, quality_stats: Dict
    ) -> List[str]:
        """Generate recommendations based on coverage analysis.

        Args:
            coverage_stats: Coverage statistics
            temporal_stats: Temporal analysis
            quality_stats: Quality analysis

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Coverage efficiency recommendations
        coverage_efficiency = coverage_stats.get("coverage_efficiency", 0.0)
        if coverage_efficiency < 0.5:
            recommendations.append(
                "Low coverage efficiency detected. Consider expanding search area or date range."
            )
        elif coverage_efficiency < 0.8:
            recommendations.append(
                "Moderate coverage efficiency. Additional scenes may improve coverage."
            )

        # Overlap factor recommendations
        overlap_factor = coverage_stats.get("overlap_factor", 1.0)
        if overlap_factor > 3.0:
            recommendations.append(
                "High scene overlap detected. Consider filtering for optimal scenes to reduce redundancy."
            )

        # Temporal recommendations
        span_days = temporal_stats.get("span_days", 0)
        total_scenes = temporal_stats.get("total_scenes", 0)

        if span_days > 0 and total_scenes > 0:
            avg_interval = span_days / total_scenes
            if avg_interval > 30:
                recommendations.append(
                    "Large temporal gaps between acquisitions. Consider expanding date range for better temporal coverage."
                )

        # Quality recommendations
        suitability_dist = quality_stats.get("suitability_distribution", {})
        excellent_count = suitability_dist.get("excellent", 0)
        poor_count = suitability_dist.get("poor", 0)

        if poor_count > excellent_count:
            recommendations.append(
                "Many poor quality scenes detected. Consider stricter quality filters."
            )

        if not recommendations:
            recommendations.append(
                "Scene collection meets quality and coverage requirements."
            )

        return recommendations

    def _create_collection_overview(self, scene_metadata: List[Dict]) -> Dict:
        """Create overview summary of scene collection.

        Args:
            scene_metadata: List of extracted scene metadata

        Returns:
            Dictionary containing collection overview
        """
        if not scene_metadata:
            return {"total_scenes": 0}

        # Basic counts
        overview = {
            "total_scenes": len(scene_metadata),
            "unique_satellites": len(
                set(
                    m.get("satellite_id")
                    for m in scene_metadata
                    if m.get("satellite_id")
                )
            ),
            "unique_item_types": len(
                set(m.get("item_type") for m in scene_metadata if m.get("item_type"))
            ),
        }

        # Quality overview
        suitability_counts = defaultdict(int)
        for metadata in scene_metadata:
            suitability = metadata.get("suitability", "unknown")
            suitability_counts[suitability] += 1

        overview["quality_summary"] = dict(suitability_counts)

        # Date range
        dates = [
            datetime.fromisoformat(m.get("acquired", "").replace("Z", "+00:00"))
            for m in scene_metadata
            if m.get("acquired")
        ]

        if dates:
            dates.sort()
            overview["date_range"] = {
                "start": dates[0].date().isoformat(),
                "end": dates[-1].date().isoformat(),
                "span_days": (dates[-1] - dates[0]).days,
            }

        return overview

    def _process_stats_response(self, stats_data: Dict) -> Dict:
        """Process and enhance Planet API stats response.

        Args:
            stats_data: Raw stats response from Planet API

        Returns:
            Processed statistics dictionary
        """
        processed = {
            "buckets": stats_data.get("buckets", []),
            "interval": stats_data.get("interval", "month"),
            "total_scenes": 0,
            "temporal_distribution": {},
        }

        # Calculate totals and temporal distribution
        for bucket in processed["buckets"]:
            count = bucket.get("count", 0)
            processed["total_scenes"] += count

            start_time = bucket.get("start_time")
            if start_time:
                processed["temporal_distribution"][start_time] = count

        return processed
