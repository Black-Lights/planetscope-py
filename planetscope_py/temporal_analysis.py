#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlanetScope-py Phase 4: Enhanced Temporal Analysis
Advanced temporal pattern analysis with 3D data cube support.

FIXED: Import conflict resolution for shapely.geometry.shape function
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False
    xr = None

# FIX: Import shapely functions with explicit naming to avoid conflicts
from shapely.geometry import Polygon, Point, mapping
from shapely.geometry import (
    shape as shapely_shape,
)  # Rename to avoid conflict with numpy array shape
from shapely.ops import transform
import pyproj

try:
    from scipy import stats
    from scipy.signal import find_peaks

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None

from .exceptions import ValidationError, PlanetScopeError
from .utils import validate_geometry, calculate_area_km2
from .metadata import MetadataProcessor

logger = logging.getLogger(__name__)

# ... rest of the classes remain the same ...


class TemporalResolution(Enum):
    """Supported temporal resolutions for data cube creation."""

    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    YEARLY = "Y"


class SeasonalPeriod(Enum):
    """Seasonal period definitions."""

    SPRING = "MAM"  # March, April, May
    SUMMER = "JJA"  # June, July, August
    AUTUMN = "SON"  # September, October, November
    WINTER = "DJF"  # December, January, February


@dataclass
class TemporalConfig:
    """Configuration for temporal analysis operations."""

    temporal_resolution: TemporalResolution = TemporalResolution.WEEKLY
    spatial_resolution: float = 30.0  # meters
    min_scenes_per_period: int = 1
    max_gap_days: int = 30
    seasonal_analysis: bool = True
    quality_weighting: bool = True
    cloud_cover_threshold: float = 0.3


@dataclass
class TemporalGap:
    """Represents a temporal gap in coverage."""

    start_date: datetime
    end_date: datetime
    duration_days: int
    location: Optional[Tuple[float, float]] = None  # lat, lon
    severity: str = "medium"  # low, medium, high, critical

    @property
    def duration_weeks(self) -> float:
        """Gap duration in weeks."""
        return self.duration_days / 7.0


@dataclass
class SeasonalPattern:
    """Seasonal acquisition pattern information."""

    season: SeasonalPeriod
    avg_scenes_per_week: float
    peak_acquisition_month: str
    coverage_quality: str  # poor, fair, good, excellent
    recommended_months: List[str]


class TemporalAnalyzer:
    """
    Advanced temporal pattern analysis with 3D data cube support.

    Provides comprehensive temporal analysis capabilities including spatiotemporal
    data cube creation, seasonal pattern detection, gap analysis, and optimization
    recommendations for PlanetScope scene acquisition patterns.
    """

    def __init__(
        self,
        config: Optional[TemporalConfig] = None,
        metadata_processor: Optional[MetadataProcessor] = None,
    ):
        """Initialize temporal analyzer.

        Args:
            config: Temporal analysis configuration
            metadata_processor: Metadata processor for quality analysis
        """
        if not XARRAY_AVAILABLE:
            raise ImportError(
                "xarray is required for temporal analysis. Install with: pip install xarray>=2023.1.0"
            )

        self.config = config or TemporalConfig()
        self.metadata_processor = metadata_processor or MetadataProcessor()

        # Analysis cache
        self._datacube_cache: Dict[str, xr.Dataset] = {}
        self._analysis_cache: Dict[str, Dict] = {}

        logger.info(
            f"TemporalAnalyzer initialized with {self.config.temporal_resolution.value} resolution"
        )

    def create_spatiotemporal_datacube(
        self,
        scenes: List[Dict],
        roi: Polygon,
        spatial_resolution: Optional[float] = None,
        temporal_resolution: Optional[TemporalResolution] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> xr.Dataset:
        """
        Create 3D spatiotemporal data cube with dimensions: (lat, lon, time).

        This is the core function that transforms scene metadata into a structured
        3D data cube enabling advanced temporal-spatial analysis.

        Args:
            scenes: List of Planet scene features
            roi: Region of interest polygon
            spatial_resolution: Grid resolution in meters (default from config)
            temporal_resolution: Time aggregation resolution (default from config)
            start_date: Analysis start date (optional)
            end_date: Analysis end date (optional)

        Returns:
            xarray.Dataset with coordinates (lat, lon, time) and data variables:
            - scene_count: Number of scenes per grid cell per time period
            - cloud_cover: Average cloud cover
            - quality_score: Average quality score
            - acquisition_days: Days since last acquisition
        """
        # Setup parameters
        spatial_res = spatial_resolution or self.config.spatial_resolution
        temporal_res = temporal_resolution or self.config.temporal_resolution

        # Validate inputs
        if not scenes:
            raise ValidationError("No scenes provided for data cube creation")

        validate_geometry(mapping(roi))

        # Extract temporal bounds from scenes if not provided
        scene_dates = []
        for scene in scenes:
            acquired = scene.get("properties", {}).get("acquired")
            if acquired:
                scene_dates.append(pd.to_datetime(acquired))

        if not scene_dates:
            raise ValidationError("No valid acquisition dates found in scenes")

        data_start = start_date or min(scene_dates)
        data_end = end_date or max(scene_dates)

        # Create spatial grid
        minx, miny, maxx, maxy = roi.bounds

        # Calculate grid dimensions
        # Convert spatial resolution from meters to degrees (approximate)
        deg_per_meter = 1 / 111320  # approximate meters per degree at equator
        spatial_res_deg = spatial_res * deg_per_meter

        # Create coordinate arrays
        lats = np.arange(miny, maxy, spatial_res_deg)
        lons = np.arange(minx, maxx, spatial_res_deg)

        # Create temporal coordinate array
        time_freq = temporal_res.value
        times = pd.date_range(start=data_start, end=data_end, freq=time_freq)

        # Initialize data arrays
        array_shape = (len(times), len(lats), len(lons))
        scene_count = np.zeros(array_shape, dtype=np.float32)
        cloud_cover = np.full(array_shape, np.nan, dtype=np.float32)
        quality_score = np.full(array_shape, np.nan, dtype=np.float32)
        acquisition_days = np.full(array_shape, np.nan, dtype=np.float32)

        logger.info(
            f"Creating data cube: {len(lats)}x{len(lons)}x{len(times)} = {array_shape[0]*array_shape[1]*array_shape[2]:,} cells"
        )

        # Process scenes into data cube
        for scene in scenes:
            try:
                # Extract scene metadata safely
                if not isinstance(scene, dict):
                    continue

                properties = scene.get("properties", {})
                geometry = scene.get("geometry", {})

                if not geometry or not properties:
                    continue

                # Extract scene metadata using the metadata processor
                metadata = self.metadata_processor.extract_scene_metadata(scene)

                # Create scene geometry - Use renamed shapely_shape function to avoid import conflict
                try:
                    from shapely.geometry import shape as shapely_shape

                    scene_geom = shapely_shape(geometry)
                    if not scene_geom.is_valid:
                        scene_geom = scene_geom.buffer(0)  # Fix invalid geometries
                except Exception as geom_error:
                    logger.warning(f"Error creating scene geometry: {geom_error}")
                    continue

                # Get acquisition date safely
                acquired = metadata.get("acquired") or properties.get("acquired")
                if not acquired:
                    continue

                scene_date = pd.to_datetime(acquired)

                # Find temporal index - FIX: Proper timedelta creation
                time_idx = None
                for i, time_period in enumerate(times):
                    period_start = time_period
                    # FIX: Create proper timedelta based on frequency string
                    if time_freq == "D":
                        period_delta = pd.Timedelta(days=1)
                    elif time_freq == "W":
                        period_delta = pd.Timedelta(weeks=1)
                    elif time_freq == "M":
                        period_delta = pd.DateOffset(months=1)
                    elif time_freq == "Q":
                        period_delta = pd.DateOffset(months=3)
                    elif time_freq == "Y":
                        period_delta = pd.DateOffset(years=1)
                    else:
                        # Fallback: try to add "1" prefix for pandas compatibility
                        try:
                            period_delta = pd.Timedelta(f"1{time_freq}")
                        except:
                            period_delta = pd.Timedelta(weeks=1)  # Ultimate fallback

                    period_end = period_start + period_delta
                    if period_start <= scene_date < period_end:
                        time_idx = i
                        break

                if time_idx is None:
                    continue  # Scene outside temporal range

                # Find spatial indices that intersect with scene
                try:
                    scene_bounds = scene_geom.bounds  # This is a property, not a method
                    if len(scene_bounds) != 4:
                        logger.warning(f"Invalid scene bounds: {scene_bounds}")
                        continue

                    lat_mask = (lats >= scene_bounds[1]) & (lats <= scene_bounds[3])
                    lon_mask = (lons >= scene_bounds[0]) & (lons <= scene_bounds[2])

                    lat_indices = np.where(lat_mask)[0]
                    lon_indices = np.where(lon_mask)[0]

                except Exception as bounds_error:
                    logger.warning(f"Error processing scene bounds: {bounds_error}")
                    continue

                # Update data cube for intersecting cells
                for lat_idx in lat_indices:
                    for lon_idx in lon_indices:
                        try:
                            # Create cell geometry for intersection test
                            if lat_idx >= len(lats) or lon_idx >= len(lons):
                                continue

                            cell_minx = lons[lon_idx]
                            cell_miny = lats[lat_idx]
                            cell_maxx = min(lons[lon_idx] + spatial_res_deg, maxx)
                            cell_maxy = min(lats[lat_idx] + spatial_res_deg, maxy)

                            # Create cell polygon
                            cell_coords = [
                                (cell_minx, cell_miny),
                                (cell_maxx, cell_miny),
                                (cell_maxx, cell_maxy),
                                (cell_minx, cell_maxy),
                                (cell_minx, cell_miny),  # Close the polygon
                            ]

                            cell_geom = Polygon(cell_coords)

                            # Check for actual intersection
                            if hasattr(scene_geom, "intersects") and callable(
                                scene_geom.intersects
                            ):
                                if scene_geom.intersects(cell_geom):
                                    # Update scene count
                                    scene_count[time_idx, lat_idx, lon_idx] += 1

                                    # Update cloud cover (average)
                                    current_cc = cloud_cover[time_idx, lat_idx, lon_idx]
                                    scene_cc = metadata.get("cloud_cover", np.nan)
                                    if not np.isnan(scene_cc):
                                        if np.isnan(current_cc):
                                            cloud_cover[time_idx, lat_idx, lon_idx] = (
                                                scene_cc
                                            )
                                        else:
                                            # Running average
                                            count = scene_count[
                                                time_idx, lat_idx, lon_idx
                                            ]
                                            cloud_cover[time_idx, lat_idx, lon_idx] = (
                                                current_cc * (count - 1) + scene_cc
                                            ) / count

                                    # Update quality score
                                    current_qs = quality_score[
                                        time_idx, lat_idx, lon_idx
                                    ]
                                    scene_qs = metadata.get("overall_quality", np.nan)
                                    if not np.isnan(scene_qs):
                                        if np.isnan(current_qs):
                                            quality_score[
                                                time_idx, lat_idx, lon_idx
                                            ] = scene_qs
                                        else:
                                            # Running average
                                            count = scene_count[
                                                time_idx, lat_idx, lon_idx
                                            ]
                                            quality_score[
                                                time_idx, lat_idx, lon_idx
                                            ] = (
                                                current_qs * (count - 1) + scene_qs
                                            ) / count
                            else:
                                logger.warning(
                                    f"Scene geometry does not have intersects method: {type(scene_geom)}"
                                )

                        except Exception as cell_error:
                            logger.warning(
                                f"Error processing grid cell [{lat_idx}, {lon_idx}]: {cell_error}"
                            )
                            continue

            except Exception as e:
                logger.warning(f"Error processing scene for data cube: {e}")
                continue

        # Calculate acquisition days (days since last acquisition)
        for lat_idx in range(len(lats)):
            for lon_idx in range(len(lons)):
                last_acquisition = None
                for time_idx in range(len(times)):
                    if scene_count[time_idx, lat_idx, lon_idx] > 0:
                        last_acquisition = times[time_idx]
                        acquisition_days[time_idx, lat_idx, lon_idx] = 0
                    elif last_acquisition is not None:
                        days_since = (times[time_idx] - last_acquisition).days
                        acquisition_days[time_idx, lat_idx, lon_idx] = days_since

        # Create xarray Dataset
        datacube = xr.Dataset(
            {
                "scene_count": (["time", "lat", "lon"], scene_count),
                "cloud_cover": (["time", "lat", "lon"], cloud_cover),
                "quality_score": (["time", "lat", "lon"], quality_score),
                "acquisition_days": (["time", "lat", "lon"], acquisition_days),
            },
            coords={
                "time": times,
                "lat": lats,
                "lon": lons,
            },
            attrs={
                "title": "PlanetScope Spatiotemporal Data Cube",
                "temporal_resolution": temporal_res.value,
                "spatial_resolution_meters": spatial_res,
                "roi_bounds": roi.bounds,
                "creation_date": datetime.now().isoformat(),
                "total_scenes_processed": len(scenes),
                "grid_shape": f"{len(lats)}x{len(lons)}x{len(times)}",
            },
        )

        logger.info(f"Data cube created successfully: {datacube.attrs['grid_shape']}")
        return datacube

    def _make_json_serializable(self, obj):
        """Make object JSON serializable by converting numpy types."""
        # Handle numpy arrays first (before pd.isna check)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            # Check for NaN values - but only for scalar values
            try:
                if pd.isna(obj):
                    return None
            except (ValueError, TypeError):
                # If pd.isna fails (e.g., for arrays), just pass through
                pass
            return obj

    def analyze_acquisition_patterns(self, datacube: xr.Dataset) -> Dict:
        """
        Comprehensive temporal pattern analysis of the data cube.

        Args:
            datacube: Spatiotemporal data cube from create_spatiotemporal_datacube()

        Returns:
            Dictionary containing comprehensive temporal analysis results
        """
        logger.info("Starting comprehensive temporal pattern analysis...")

        analysis = {
            "acquisition_frequency": self._calculate_frequency_stats(datacube),
            "seasonal_patterns": self._detect_seasonal_patterns(datacube),
            "temporal_gaps": self._identify_temporal_gaps(datacube),
            "quality_trends": self._analyze_quality_trends(datacube),
            "spatial_temporal_correlation": self._analyze_spatial_temporal_correlation(
                datacube
            ),
            "optimal_windows": self._find_optimal_acquisition_windows(datacube),
            "summary_statistics": self._calculate_summary_statistics(datacube),
        }

        logger.info("Temporal pattern analysis completed")
        return analysis

    def _calculate_frequency_stats(self, datacube: xr.Dataset) -> Dict:
        """Calculate acquisition frequency statistics."""
        scene_counts = datacube["scene_count"].values

        # SIMPLE FIX: Count unique scenes properly
        # Since we're in a spatiotemporal context, we need to count unique scenes per time period
        # then sum across time periods to get total unique scenes

        # Sum scene counts across spatial dimensions for each time period
        scenes_per_period = np.nansum(
            scene_counts, axis=(1, 2)
        )  # Sum over spatial dimensions

        # For total_scenes: if each scene appears in only one time period,
        # then we can estimate unique scenes by looking at the data pattern

        # Method: Count the number of input scenes that actually contributed to the datacube
        # We'll estimate this by counting periods with data and assuming reasonable scene distribution
        non_zero_periods = np.sum(scenes_per_period > 0)

        # Better approach: Use the fact that each scene should appear in exactly one time period
        # The total unique scenes is approximately the sum of scenes across time periods,
        # but we need to account for the spatial spreading

        # Since scenes are spread across multiple spatial cells, we need to "de-duplicate"
        # For now, let's use the number of non-zero periods as a proxy for unique scenes
        # This works when scenes don't overlap temporally

        # SIMPLE SOLUTION: Use metadata tracking instead
        # For now, estimate based on non-zero time periods
        # This assumes roughly one scene per active time period
        estimated_unique_scenes = non_zero_periods

        # If we have the datacube attrs with total_scenes_processed, use that
        if hasattr(datacube, "attrs") and "total_scenes_processed" in datacube.attrs:
            total_scenes = datacube.attrs["total_scenes_processed"]
        else:
            total_scenes = estimated_unique_scenes

        total_periods = scene_counts.shape[0]

        frequency_stats = {
            "total_scenes": int(
                total_scenes
            ),  # FIXED: Now represents unique input scenes
            "total_time_periods": total_periods,
            "periods_with_data": int(non_zero_periods),
            "data_availability_percentage": (non_zero_periods / total_periods) * 100,
            "avg_scenes_per_period": float(np.nanmean(scenes_per_period)),
            "max_scenes_per_period": float(np.nanmax(scenes_per_period)),
            "min_scenes_per_period": (
                float(np.nanmin(scenes_per_period[scenes_per_period > 0]))
                if np.any(scenes_per_period > 0)
                else 0
            ),
            "std_scenes_per_period": float(np.nanstd(scenes_per_period)),
            "coefficient_of_variation": (
                float(np.nanstd(scenes_per_period) / np.nanmean(scenes_per_period))
                if np.nanmean(scenes_per_period) > 0
                else 0
            ),
            "temporal_distribution": scenes_per_period.tolist(),
            "time_coordinates": datacube.time.values.tolist(),
            # Add debug info
            "debug_info": {
                "estimated_from_periods": estimated_unique_scenes,
                "total_scene_cell_intersections": int(np.nansum(scene_counts)),
                "spatial_grid_size": f"{scene_counts.shape[1]}x{scene_counts.shape[2]}",
            },
        }

        return frequency_stats

    def _detect_seasonal_patterns(self, datacube: xr.Dataset) -> Dict:
        """Detect and analyze seasonal acquisition patterns."""
        if not self.config.seasonal_analysis:
            return {"seasonal_analysis_disabled": True}

        # Convert time to pandas for easier seasonal analysis
        times = pd.to_datetime(datacube.time.values)
        scene_counts = np.nansum(datacube["scene_count"].values, axis=(1, 2))

        # Create DataFrame for analysis
        df = pd.DataFrame(
            {
                "time": times,
                "scene_count": scene_counts,
                "month": times.month,
                "season": times.month.map(self._get_season),
                "year": times.year,
            }
        )

        seasonal_patterns = {}

        # Analyze each season
        for season in SeasonalPeriod:
            season_data = df[df["season"] == season.value]

            if len(season_data) > 0:
                seasonal_patterns[season.value] = SeasonalPattern(
                    season=season,
                    avg_scenes_per_week=float(season_data["scene_count"].mean()),
                    peak_acquisition_month=season_data.groupby("month")["scene_count"]
                    .mean()
                    .idxmax(),
                    coverage_quality=self._assess_coverage_quality(
                        season_data["scene_count"].mean()
                    ),
                    recommended_months=self._get_recommended_months(season_data),
                )

        # Overall seasonal statistics
        monthly_avg = df.groupby("month")["scene_count"].mean()
        peak_month = monthly_avg.idxmax()
        seasonal_variation = (
            monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() > 0 else 0
        )

        return {
            "seasonal_patterns": {k: v.__dict__ for k, v in seasonal_patterns.items()},
            "peak_acquisition_month": int(peak_month),
            "seasonal_variation_coefficient": float(seasonal_variation),
            "monthly_averages": monthly_avg.to_dict(),
            "recommended_seasonal_strategy": self._generate_seasonal_strategy(
                seasonal_patterns
            ),
        }

    def _identify_temporal_gaps(self, datacube: xr.Dataset) -> Dict:
        """Identify temporal gaps in scene coverage."""
        times = pd.to_datetime(datacube.time.values)
        scene_counts = np.nansum(datacube["scene_count"].values, axis=(1, 2))

        gaps = []
        gap_start = None

        for i, count in enumerate(scene_counts):
            if count == 0:  # No scenes in this period
                if gap_start is None:
                    gap_start = i
            else:  # Scenes present
                if gap_start is not None:
                    # End of gap
                    gap_duration = i - gap_start
                    gap_days = (times[i] - times[gap_start]).days

                    if gap_days >= self.config.max_gap_days:
                        gap = TemporalGap(
                            start_date=times[gap_start],
                            end_date=times[i - 1],
                            duration_days=gap_days,
                            severity=self._assess_gap_severity(gap_days),
                        )
                        gaps.append(gap)

                    gap_start = None

        # Handle gap extending to end
        if gap_start is not None:
            gap_days = (times[-1] - times[gap_start]).days
            if gap_days >= self.config.max_gap_days:
                gap = TemporalGap(
                    start_date=times[gap_start],
                    end_date=times[-1],
                    duration_days=gap_days,
                    severity=self._assess_gap_severity(gap_days),
                )
                gaps.append(gap)

        # Gap statistics
        total_gap_days = sum(gap.duration_days for gap in gaps)
        total_analysis_days = (times[-1] - times[0]).days
        gap_percentage = (
            (total_gap_days / total_analysis_days) * 100
            if total_analysis_days > 0
            else 0
        )

        return {
            "identified_gaps": [gap.__dict__ for gap in gaps],
            "total_gaps": len(gaps),
            "total_gap_days": total_gap_days,
            "gap_percentage": float(gap_percentage),
            "largest_gap_days": max((gap.duration_days for gap in gaps), default=0),
            "avg_gap_duration": (
                float(np.mean([gap.duration_days for gap in gaps])) if gaps else 0
            ),
            "gap_severity_distribution": self._calculate_gap_severity_distribution(
                gaps
            ),
        }

    def _analyze_quality_trends(self, datacube: xr.Dataset) -> Dict:
        """Analyze temporal trends in quality metrics."""
        if "quality_score" not in datacube.data_vars:
            return {"quality_analysis_unavailable": "No quality scores in datacube"}

        quality_data = datacube["quality_score"].values
        cloud_data = datacube["cloud_cover"].values
        times = datacube.time.values

        # Calculate temporal quality trends with proper handling of empty slices
        temporal_quality = []
        temporal_cloud = []

        # Process each time step individually to handle empty slices gracefully
        for t in range(quality_data.shape[0]):
            # Extract spatial slice for this time period
            quality_slice = quality_data[t, :, :]
            cloud_slice = cloud_data[t, :, :]

            # Check if slice has any valid data
            if np.any(~np.isnan(quality_slice)):
                temporal_quality.append(np.nanmean(quality_slice))
            else:
                temporal_quality.append(np.nan)

            if np.any(~np.isnan(cloud_slice)):
                temporal_cloud.append(np.nanmean(cloud_slice))
            else:
                temporal_cloud.append(np.nan)

        # Convert to numpy arrays
        temporal_quality = np.array(temporal_quality)
        temporal_cloud = np.array(temporal_cloud)

        # Alternative approach using warnings suppression (if you prefer the original method):
        # import warnings
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", RuntimeWarning)
        #     temporal_quality = np.nanmean(quality_data, axis=(1, 2))
        #     temporal_cloud = np.nanmean(cloud_data, axis=(1, 2))

        # Remove NaN values for trend analysis
        valid_quality_mask = ~np.isnan(temporal_quality)
        valid_cloud_mask = ~np.isnan(temporal_cloud)

        quality_trend = None
        cloud_trend = None

        if SCIPY_AVAILABLE and np.sum(valid_quality_mask) > 1:
            # Calculate trend using linear regression
            x = np.arange(len(temporal_quality))[valid_quality_mask]
            y = temporal_quality[valid_quality_mask]
            if len(x) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                quality_trend = {
                    "slope": float(slope),
                    "r_squared": float(r_value**2),
                    "p_value": float(p_value),
                    "trend_direction": (
                        "improving"
                        if slope > 0
                        else "declining" if slope < 0 else "stable"
                    ),
                }

        if SCIPY_AVAILABLE and np.sum(valid_cloud_mask) > 1:
            x = np.arange(len(temporal_cloud))[valid_cloud_mask]
            y = temporal_cloud[valid_cloud_mask]
            if len(x) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                cloud_trend = {
                    "slope": float(slope),
                    "r_squared": float(r_value**2),
                    "p_value": float(p_value),
                    "trend_direction": (
                        "improving"
                        if slope < 0
                        else "worsening" if slope > 0 else "stable"
                    ),
                }

        # Calculate averages with proper NaN handling
        avg_quality = (
            np.nanmean(temporal_quality)
            if np.any(~np.isnan(temporal_quality))
            else np.nan
        )
        avg_cloud = (
            np.nanmean(temporal_cloud) if np.any(~np.isnan(temporal_cloud)) else np.nan
        )
        quality_var = (
            np.nanstd(temporal_quality)
            if np.any(~np.isnan(temporal_quality))
            else np.nan
        )
        cloud_var = (
            np.nanstd(temporal_cloud) if np.any(~np.isnan(temporal_cloud)) else np.nan
        )

        return {
            "quality_trend": quality_trend,
            "cloud_cover_trend": cloud_trend,
            "avg_quality_score": (
                float(avg_quality) if not np.isnan(avg_quality) else None
            ),
            "avg_cloud_cover": float(avg_cloud) if not np.isnan(avg_cloud) else None,
            "quality_variability": (
                float(quality_var) if not np.isnan(quality_var) else None
            ),
            "cloud_variability": float(cloud_var) if not np.isnan(cloud_var) else None,
            "temporal_quality_series": temporal_quality.tolist(),
            "temporal_cloud_series": temporal_cloud.tolist(),
            "data_availability": {
                "quality_periods_with_data": int(np.sum(valid_quality_mask)),
                "cloud_periods_with_data": int(np.sum(valid_cloud_mask)),
                "total_time_periods": len(temporal_quality),
            },
        }

    def _analyze_spatial_temporal_correlation(self, datacube: xr.Dataset) -> Dict:
        """Analyze correlation between spatial and temporal patterns."""
        scene_counts = datacube["scene_count"].values

        # Calculate spatial consistency over time
        spatial_means = np.nanmean(scene_counts, axis=0)  # Mean over time
        spatial_stds = np.nanstd(scene_counts, axis=0)  # Std over time

        # Calculate temporal consistency over space
        temporal_means = np.nanmean(scene_counts, axis=(1, 2))  # Mean over space
        temporal_stds = np.nanstd(scene_counts, axis=(1, 2))  # Std over space

        # Find hotspots and cold spots
        hotspot_threshold = np.nanpercentile(spatial_means, 75)
        coldspot_threshold = np.nanpercentile(spatial_means, 25)

        hotspot_locations = np.where(spatial_means >= hotspot_threshold)
        coldspot_locations = np.where(spatial_means <= coldspot_threshold)

        return {
            "spatial_consistency": {
                "mean_scenes_per_location": float(np.nanmean(spatial_means)),
                "spatial_variability": float(np.nanmean(spatial_stds)),
                "coefficient_of_variation": (
                    float(np.nanmean(spatial_stds) / np.nanmean(spatial_means))
                    if np.nanmean(spatial_means) > 0
                    else 0
                ),
            },
            "temporal_consistency": {
                "mean_scenes_per_time": float(np.nanmean(temporal_means)),
                "temporal_variability": float(np.nanstd(temporal_means)),
                "coefficient_of_variation": (
                    float(np.nanstd(temporal_means) / np.nanmean(temporal_means))
                    if np.nanmean(temporal_means) > 0
                    else 0
                ),
            },
            "hotspot_analysis": {
                "hotspot_count": len(hotspot_locations[0]),
                "coldspot_count": len(coldspot_locations[0]),
                "coverage_ratio": (
                    len(hotspot_locations[0]) / len(coldspot_locations[0])
                    if len(coldspot_locations[0]) > 0
                    else float("inf")
                ),
            },
        }

    def _find_optimal_acquisition_windows(self, datacube: xr.Dataset) -> Dict:
        """Find optimal time windows for future acquisitions."""
        scene_counts = datacube["scene_count"].values
        times = pd.to_datetime(datacube.time.values)

        # Calculate acquisition gaps
        temporal_means = np.nanmean(scene_counts, axis=(1, 2))

        # Find periods with low coverage
        low_coverage_threshold = np.nanpercentile(temporal_means, 25)
        low_coverage_periods = temporal_means <= low_coverage_threshold

        # Identify optimal windows (consecutive low coverage periods)
        optimal_windows = []
        window_start = None

        for i, is_low_coverage in enumerate(low_coverage_periods):
            if is_low_coverage:
                if window_start is None:
                    window_start = i
            else:
                if window_start is not None:
                    # End of window
                    optimal_windows.append(
                        {
                            "start_date": times[window_start].isoformat(),
                            "end_date": times[i - 1].isoformat(),
                            "duration_periods": i - window_start,
                            "avg_current_coverage": float(
                                np.mean(temporal_means[window_start:i])
                            ),
                            "priority": "high" if i - window_start > 2 else "medium",
                        }
                    )
                    window_start = None

        # Handle window extending to end
        if window_start is not None:
            optimal_windows.append(
                {
                    "start_date": times[window_start].isoformat(),
                    "end_date": times[-1].isoformat(),
                    "duration_periods": len(times) - window_start,
                    "avg_current_coverage": float(
                        np.mean(temporal_means[window_start:])
                    ),
                    "priority": "high" if len(times) - window_start > 2 else "medium",
                }
            )

        return {
            "optimal_windows": optimal_windows,
            "total_optimization_opportunities": len(optimal_windows),
            "high_priority_windows": len(
                [w for w in optimal_windows if w["priority"] == "high"]
            ),
            "recommendations": self._generate_acquisition_recommendations(
                optimal_windows
            ),
        }

    def _calculate_summary_statistics(self, datacube: xr.Dataset) -> Dict:
        """Calculate comprehensive summary statistics for the data cube."""
        scene_counts = datacube["scene_count"].values

        return {
            "datacube_shape": list(datacube.sizes.values()),
            "total_grid_cells": int(np.prod(list(datacube.sizes.values()))),
            "data_density": float(
                np.sum(scene_counts > 0) / np.prod(list(datacube.sizes.values()))
            ),
            "cells_with_data": int(
                np.sum(scene_counts > 0)
            ),  # Count of cells with at least one scene #### check later
            "temporal_span_days": int(
                (datacube.time[-1] - datacube.time[0])
                .values.astype("timedelta64[D]")
                .astype(int)
            ),
            "spatial_extent_km2": self._calculate_spatial_extent(datacube),
            "avg_scenes_per_cell": float(np.nanmean(scene_counts)),
            "max_scenes_per_cell": float(np.nanmax(scene_counts)),
            "scene_distribution_percentiles": {
                "25th": float(np.nanpercentile(scene_counts, 25)),
                "50th": float(np.nanpercentile(scene_counts, 50)),
                "75th": float(np.nanpercentile(scene_counts, 75)),
                "90th": float(np.nanpercentile(scene_counts, 90)),
                "95th": float(np.nanpercentile(scene_counts, 95)),
            },
        }

    def generate_temporal_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable temporal analysis recommendations."""
        recommendations = []

        # Frequency recommendations
        freq_stats = analysis.get("acquisition_frequency", {})
        data_availability = freq_stats.get("data_availability_percentage", 0)

        if data_availability < 50:
            recommendations.append(
                f"Low data availability ({data_availability:.1f}%). Consider expanding temporal coverage or relaxing scene quality criteria."
            )
        elif data_availability > 90:
            recommendations.append(
                f"Excellent data availability ({data_availability:.1f}%). Current acquisition strategy is optimal."
            )

        # Gap recommendations
        gaps = analysis.get("temporal_gaps", {})
        total_gaps = gaps.get("total_gaps", 0)
        gap_percentage = gaps.get("gap_percentage", 0)

        if total_gaps > 0:
            recommendations.append(
                f"Found {total_gaps} temporal gaps covering {gap_percentage:.1f}% of analysis period. Consider targeted acquisitions during gap periods."
            )

        # Seasonal recommendations
        seasonal = analysis.get("seasonal_patterns", {})
        if "recommended_seasonal_strategy" in seasonal:
            recommendations.append(
                f"Seasonal strategy: {seasonal['recommended_seasonal_strategy']}"
            )

        # Quality recommendations
        quality = analysis.get("quality_trends", {})
        quality_trend = quality.get("quality_trend") if quality else None
        if quality_trend and quality_trend.get("trend_direction") == "declining":
            recommendations.append(
                "Quality scores are declining over time. Review acquisition parameters and scene selection criteria."
            )

        # Optimization recommendations
        optimal = analysis.get("optimal_windows", {})
        high_priority = optimal.get("high_priority_windows", 0)
        if high_priority > 0:
            recommendations.append(
                f"Identified {high_priority} high-priority time windows for optimized acquisitions."
            )

        return recommendations

    def export_temporal_analysis(
        self,
        analysis: Dict,
        datacube: xr.Dataset,
        output_dir: str,
        format: str = "comprehensive",
    ) -> Dict[str, str]:
        """Export temporal analysis results to various formats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        exported_files = {}

        # Export data cube as NetCDF
        if format in ["comprehensive", "datacube"]:
            datacube_path = output_path / "spatiotemporal_datacube.nc"

            # Convert datetime coordinates to avoid xarray/netcdf issues
            datacube_copy = datacube.copy()
            if "time" in datacube_copy.coords:
                # Convert timezone-aware datetime to timezone-naive
                import pandas as pd  # ADD THIS IMPORT

                time_values = pd.to_datetime(datacube_copy.time.values)

                # Handle timezone conversion safely
                if hasattr(time_values, "tz") and time_values.tz is not None:
                    time_values = time_values.tz_localize(None)

                datacube_copy = datacube_copy.assign_coords(time=time_values)

            datacube_copy.to_netcdf(datacube_path)
            exported_files["datacube_netcdf"] = str(datacube_path)  # CHANGED KEY

        # Export analysis as JSON
        if format in ["comprehensive", "analysis"]:
            import json

            analysis_path = output_path / "temporal_analysis.json"

            # Convert numpy arrays and datetime objects for JSON serialization
            json_analysis = self._prepare_analysis_for_json(analysis)

            with open(analysis_path, "w") as f:
                json.dump(json_analysis, f, indent=2, default=str)
            exported_files["analysis_json"] = str(analysis_path)  # CHANGED KEY

        # Export recommendations
        if format in ["comprehensive", "recommendations"]:
            recommendations = self.generate_temporal_recommendations(analysis)
            rec_path = output_path / "recommendations.txt"

            with open(rec_path, "w") as f:
                f.write("PlanetScope Temporal Analysis Recommendations\n")
                f.write("=" * 50 + "\n\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n\n")
            exported_files["recommendations_txt"] = str(rec_path)  # CHANGED KEY

        # Export summary CSV (add this if the test expects it)
        if format in ["comprehensive", "summary"]:
            summary_path = output_path / "summary.csv"

            # Create a simple summary CSV
            freq_stats = analysis.get("acquisition_frequency", {})
            gaps = analysis.get("temporal_gaps", {})

            summary_data = {
                "metric": [
                    "total_scenes",
                    "data_availability_percentage",
                    "total_gaps",
                    "gap_percentage",
                ],
                "value": [
                    freq_stats.get("total_scenes", 0),
                    freq_stats.get("data_availability_percentage", 0),
                    gaps.get("total_gaps", 0),
                    gaps.get("gap_percentage", 0),
                ],
            }

            import pandas as pd_local  # Use different name to avoid conflicts

            pd_local.DataFrame(summary_data).to_csv(summary_path, index=False)
            exported_files["summary_csv"] = str(summary_path)  # ADD THIS KEY

        # Export visualizations
        if MATPLOTLIB_AVAILABLE and format in ["comprehensive", "visualizations"]:
            vis_files = self._export_temporal_visualizations(
                analysis, datacube, output_path
            )
            exported_files.update(vis_files)

        return exported_files

    # Helper methods
    def _get_season(self, month: int) -> str:
        """Map month to season."""
        if month in [3, 4, 5]:
            return SeasonalPeriod.SPRING.value
        elif month in [6, 7, 8]:
            return SeasonalPeriod.SUMMER.value
        elif month in [9, 10, 11]:
            return SeasonalPeriod.AUTUMN.value
        else:
            return SeasonalPeriod.WINTER.value

    def _assess_coverage_quality(self, avg_scenes: float) -> str:
        """Assess coverage quality based on average scene count."""
        if avg_scenes >= 10.0:  # CHANGED: increased threshold for excellent
            return "excellent"
        elif avg_scenes >= 5.0:  # CHANGED: increased threshold for good
            return "good"
        elif avg_scenes >= 2.0:  # CHANGED: increased threshold for fair
            return "fair"
        else:
            return "poor"

    def _get_recommended_months(self, season_data: pd.DataFrame) -> List[str]:
        """Get recommended months for a season."""
        if len(season_data) == 0:
            return []

        monthly_avg = season_data.groupby("month")["scene_count"].mean()
        # Recommend months with above-average coverage
        avg_coverage = monthly_avg.mean()
        recommended = monthly_avg[monthly_avg >= avg_coverage].index.tolist()

        # Convert month numbers to names
        month_names = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December",
        }

        return [month_names[month] for month in recommended]

    def _generate_seasonal_strategy(self, seasonal_patterns: Dict) -> str:
        """Generate seasonal acquisition strategy."""
        if not seasonal_patterns:
            return "Insufficient seasonal data for strategy recommendations."

        # Find best and worst performing seasons
        season_scores = {}
        for season_name, pattern in seasonal_patterns.items():
            if isinstance(pattern, dict):
                season_scores[season_name] = pattern.get("avg_scenes_per_week", 0)
            else:
                season_scores[season_name] = pattern.avg_scenes_per_week

        if not season_scores:
            return "No seasonal performance data available."

        best_season = max(season_scores, key=season_scores.get)
        worst_season = min(season_scores, key=season_scores.get)

        return f"Focus acquisition efforts during {best_season} (highest coverage). Increase monitoring during {worst_season} (lowest coverage)."

    def _assess_gap_severity(self, gap_days: int) -> str:
        """Assess temporal gap severity."""
        if gap_days >= 90:
            return "critical"
        elif gap_days >= 60:
            return "high"
        elif gap_days >= 30:
            return "medium"
        else:
            return "low"

    def _calculate_gap_severity_distribution(
        self, gaps: List[TemporalGap]
    ) -> Dict[str, int]:
        """Calculate distribution of gap severities."""
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for gap in gaps:
            severity_counts[gap.severity] += 1

        return severity_counts

    def _generate_acquisition_recommendations(
        self, optimal_windows: List[Dict]
    ) -> List[str]:
        """Generate specific acquisition recommendations."""
        recommendations = []

        high_priority_windows = [w for w in optimal_windows if w["priority"] == "high"]
        medium_priority_windows = [
            w for w in optimal_windows if w["priority"] == "medium"
        ]

        if high_priority_windows:
            recommendations.append(
                f"High priority: Schedule acquisitions during {len(high_priority_windows)} identified time windows with poor coverage."
            )

        if medium_priority_windows:
            recommendations.append(
                f"Medium priority: Consider additional acquisitions during {len(medium_priority_windows)} time windows with moderate coverage gaps."
            )

        if not optimal_windows:
            recommendations.append(
                "Current temporal coverage appears well-distributed. Maintain existing acquisition strategy."
            )

        return recommendations

    def _calculate_spatial_extent(self, datacube: xr.Dataset) -> float:
        """Calculate spatial extent of datacube in km2."""
        # Get coordinate bounds
        lat_min, lat_max = float(datacube.lat.min()), float(datacube.lat.max())
        lon_min, lon_max = float(datacube.lon.min()), float(datacube.lon.max())

        # Create bounding box polygon
        from shapely.geometry import box

        bbox = box(lon_min, lat_min, lon_max, lat_max)

        return calculate_area_km2(bbox)

    def _prepare_analysis_for_json(self, analysis: Dict) -> Dict:
        """Prepare analysis dictionary for JSON serialization."""
        import copy

        json_analysis = copy.deepcopy(analysis)

        # Convert numpy arrays and datetime objects
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj

        return convert_for_json(json_analysis)

    def _export_temporal_visualizations(
        self, analysis: Dict, datacube: xr.Dataset, output_path: Path
    ) -> Dict[str, str]:
        """Export temporal analysis visualizations."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping visualization export.")
            return {}

        exported_files = {}

        try:
            # Temporal frequency plot
            freq_stats = analysis.get("acquisition_frequency", {})
            if "temporal_distribution" in freq_stats:
                fig, ax = plt.subplots(figsize=(12, 6))
                times = pd.to_datetime(freq_stats.get("time_coordinates", []))
                counts = freq_stats["temporal_distribution"]

                ax.plot(times, counts, marker="o", linewidth=2, markersize=4)
                ax.set_title(
                    "Temporal Acquisition Frequency", fontsize=14, fontweight="bold"
                )
                ax.set_xlabel("Time")
                ax.set_ylabel("Scene Count per Period")
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()

                freq_plot_path = output_path / "temporal_frequency.png"
                plt.savefig(freq_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                exported_files["frequency_plot"] = str(freq_plot_path)

            # Seasonal patterns plot
            seasonal = analysis.get("seasonal_patterns", {})
            if "monthly_averages" in seasonal:
                fig, ax = plt.subplots(figsize=(10, 6))
                monthly_data = seasonal["monthly_averages"]
                months = list(monthly_data.keys())
                values = list(monthly_data.values())

                ax.bar(months, values, color="skyblue", alpha=0.7)
                ax.set_title(
                    "Average Monthly Scene Acquisition", fontsize=14, fontweight="bold"
                )
                ax.set_xlabel("Month")
                ax.set_ylabel("Average Scene Count")
                ax.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()

                seasonal_plot_path = output_path / "seasonal_patterns.png"
                plt.savefig(seasonal_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                exported_files["seasonal_plot"] = str(seasonal_plot_path)

            # Gap analysis plot
            gaps = analysis.get("temporal_gaps", {})
            if "gap_severity_distribution" in gaps:
                fig, ax = plt.subplots(figsize=(8, 6))
                severity_dist = gaps["gap_severity_distribution"]

                colors = {
                    "low": "green",
                    "medium": "yellow",
                    "high": "orange",
                    "critical": "red",
                }
                severities = list(severity_dist.keys())
                counts = list(severity_dist.values())
                plot_colors = [colors[sev] for sev in severities]

                ax.bar(severities, counts, color=plot_colors, alpha=0.7)
                ax.set_title(
                    "Temporal Gap Severity Distribution", fontsize=14, fontweight="bold"
                )
                ax.set_xlabel("Gap Severity")
                ax.set_ylabel("Number of Gaps")
                plt.tight_layout()

                gaps_plot_path = output_path / "gap_analysis.png"
                plt.savefig(gaps_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                exported_files["gaps_plot"] = str(gaps_plot_path)

        except Exception as e:
            logger.warning(f"Error creating visualizations: {e}")

        return exported_files

    def create_temporal_summary_report(
        self, analysis: Dict, datacube: xr.Dataset, output_path: str
    ) -> str:
        """Create comprehensive temporal analysis summary report."""
        report_path = Path(output_path)

        # Generate recommendations
        recommendations = self.generate_temporal_recommendations(analysis)

        # Create report content
        report_content = self._generate_report_content(
            analysis, datacube, recommendations
        )

        # Write report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Temporal analysis report generated: {report_path}")
        return str(report_path)

    def _generate_report_content(
        self, analysis: Dict, datacube: xr.Dataset, recommendations: List[str]
    ) -> str:
        """Generate comprehensive report content."""

        # Extract key statistics
        freq_stats = analysis.get("acquisition_frequency", {})
        gaps = analysis.get("temporal_gaps", {})
        seasonal = analysis.get("seasonal_patterns", {})
        quality = analysis.get("quality_trends", {})
        summary = analysis.get("summary_statistics", {})

        report = f"""
PlanetScope Temporal Analysis Report
===================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
-----------------
• Total scenes analyzed: {freq_stats.get('total_scenes', 'N/A'):,}
• Temporal span: {summary.get('temporal_span_days', 'N/A')} days
• Data availability: {freq_stats.get('data_availability_percentage', 0):.1f}%
• Spatial extent: {summary.get('spatial_extent_km2', 'N/A'):.1f} km²
• Temporal gaps identified: {gaps.get('total_gaps', 0)}

ACQUISITION FREQUENCY ANALYSIS
------------------------------
• Average scenes per period: {freq_stats.get('avg_scenes_per_period', 0):.2f}
• Maximum scenes per period: {freq_stats.get('max_scenes_per_period', 0)}
• Coefficient of variation: {freq_stats.get('coefficient_of_variation', 0):.3f}
• Periods with data: {freq_stats.get('periods_with_data', 0)} / {freq_stats.get('total_time_periods', 0)}

TEMPORAL GAP ANALYSIS
--------------------
• Total gap duration: {gaps.get('total_gap_days', 0)} days ({gaps.get('gap_percentage', 0):.1f}% of period)
• Largest gap: {gaps.get('largest_gap_days', 0)} days
• Average gap duration: {gaps.get('avg_gap_duration', 0):.1f} days

SEASONAL PATTERNS
-----------------
"""

        if seasonal and "peak_acquisition_month" in seasonal:
            report += f"• Peak acquisition month: {seasonal.get('peak_acquisition_month', 'N/A')}\n"
            report += f"• Seasonal variation coefficient: {seasonal.get('seasonal_variation_coefficient', 0):.3f}\n"

        if quality and not quality.get("quality_analysis_unavailable"):
            report += f"""
QUALITY TRENDS
--------------
• Average quality score: {quality.get('avg_quality_score', 0):.3f}
• Average cloud cover: {quality.get('avg_cloud_cover', 0):.3f}
• Quality trend: {quality.get('quality_trend', {}).get('trend_direction', 'N/A')}
"""

        report += f"""
SPATIAL-TEMPORAL CHARACTERISTICS
-------------------------------
• Grid shape: {summary.get('datacube_shape', 'N/A')}
• Data density: {summary.get('data_density', 0):.3f}
• Average scenes per cell: {summary.get('avg_scenes_per_cell', 0):.2f}

RECOMMENDATIONS
--------------
"""

        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"

        report += f"""

TECHNICAL DETAILS
----------------
• Temporal resolution: {self.config.temporal_resolution.value}
• Spatial resolution: {self.config.spatial_resolution}m
• Analysis configuration: {self.config.__dict__}

---
Report generated by PlanetScope-py v{datacube.attrs.get('creation_date', 'Unknown')}
"""

        return report
