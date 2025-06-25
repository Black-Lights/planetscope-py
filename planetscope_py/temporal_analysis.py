#!/usr/bin/env python3
"""
PlanetScope-py Temporal Analysis Module - COMPLETE FIXED VERSION
Grid-based temporal pattern analysis for PlanetScope satellite imagery.

This module implements temporal analysis capabilities to understand acquisition patterns
and identify coverage gaps using the same grid-based approach as spatial density analysis.

Key Features:
- Grid-based temporal interval analysis (same as spatial density)
- Daily temporal resolution
- Scene-to-grid intersection analysis  
- Temporal statistics per grid cell
- ROI clipping support (same as spatial density)
- Coordinate system fixes integration
- Professional GeoTIFF export with QML styling
- Integration with visualization module
- FAST and ACCURATE optimization methods

FIXES INCLUDED:
- Proper optimization method selection (respects user choice)
- Missing import os
- Function signature fixes
- JSON serialization fixes
- Performance improvements
- Complete function implementations

Author: Ammar & Umayr
Version: 4.1.0 (Enhanced + Metadata Fixes + JSON Serialization)
"""

import logging
import time
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds, Affine
from rasterio.crs import CRS
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from .exceptions import ValidationError, PlanetScopeError
from .utils import validate_geometry, calculate_geometry_bounds

logger = logging.getLogger(__name__)


class TemporalMetric(Enum):
    """Types of temporal metrics to calculate."""
    
    COVERAGE_DAYS = "coverage_days"  # Number of days with scene coverage
    MEAN_INTERVAL = "mean_interval"  # Mean days between consecutive scenes
    MEDIAN_INTERVAL = "median_interval"  # Median days between consecutive scenes
    MIN_INTERVAL = "min_interval"  # Minimum days between scenes
    MAX_INTERVAL = "max_interval"  # Maximum days between scenes
    TEMPORAL_DENSITY = "temporal_density"  # Scenes per day
    COVERAGE_FREQUENCY = "coverage_frequency"  # Days with coverage / total days


class TemporalResolution(Enum):
    """Temporal resolution for analysis."""
    
    DAILY = "daily"  # 1-day resolution (default)
    WEEKLY = "weekly"  # 7-day resolution
    MONTHLY = "monthly"  # Monthly resolution


@dataclass
class TemporalConfig:
    """Configuration for temporal analysis calculations."""
    
    spatial_resolution: float = 30.0  # Spatial grid resolution in meters
    temporal_resolution: TemporalResolution = TemporalResolution.DAILY
    metrics: List[TemporalMetric] = None  # Metrics to calculate (None = all)
    chunk_size_km: float = 200.0  # Spatial chunk size for large areas
    max_memory_gb: float = 16.0  # Memory limit
    parallel_workers: int = 4  # Number of parallel workers
    no_data_value: float = -9999.0  # NoData value for output rasters
    coordinate_system_fixes: bool = True  # Enable coordinate system corrections
    force_single_chunk: bool = False  # Force single chunk processing
    validate_geometries: bool = True  # Validate input geometries
    min_scenes_per_cell: int = 2  # Minimum scenes required for temporal analysis
    optimization_method: str = "auto"  # FIXED: Add optimization method to config
    
    def __post_init__(self):
        """Post-initialization validation and defaults."""
        if self.metrics is None:
            # Default to key temporal metrics
            self.metrics = [
                TemporalMetric.COVERAGE_DAYS,
                TemporalMetric.MEAN_INTERVAL,
                TemporalMetric.TEMPORAL_DENSITY
            ]
        
        # Ensure temporal_resolution is enum
        if isinstance(self.temporal_resolution, str):
            resolution_mapping = {
                "daily": TemporalResolution.DAILY,
                "weekly": TemporalResolution.WEEKLY,
                "monthly": TemporalResolution.MONTHLY,
            }
            if self.temporal_resolution.lower() in resolution_mapping:
                self.temporal_resolution = resolution_mapping[self.temporal_resolution.lower()]
            else:
                raise ValidationError(f"Invalid temporal resolution: {self.temporal_resolution}")


@dataclass
class TemporalResult:
    """Results from temporal analysis calculation."""
    
    metric_arrays: Dict[TemporalMetric, np.ndarray]  # Arrays for each metric
    transform: rasterio.Affine  # Coordinate transform
    crs: str  # Coordinate reference system
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    temporal_stats: Dict[str, Any]  # Temporal statistics summary
    computation_time: float  # Processing time
    config: TemporalConfig  # Configuration used
    grid_info: Dict[str, Any]  # Grid information
    date_range: Tuple[str, str]  # Analysis date range
    coordinate_system_corrected: bool = True  # Coordinate fixes applied
    no_data_value: float = -9999.0  # NoData value


class TemporalAnalyzer:
    """
    Grid-based temporal analysis engine for PlanetScope scenes.
    
    This analyzer creates a spatial grid (like spatial density analysis) and calculates
    temporal patterns and statistics for each grid cell based on scene acquisition dates.
    
    Key Features:
    - Same grid creation approach as spatial density engine
    - Temporal interval calculation between consecutive scenes
    - Multiple temporal metrics per grid cell
    - ROI clipping support with coordinate system fixes
    - Professional export capabilities
    - Integration with visualization module
    - FAST and ACCURATE optimization methods
    """
    
    def __init__(self, config: Optional[TemporalConfig] = None):
        """Initialize the temporal analyzer.
        
        Args:
            config: Configuration for temporal analysis
        """
        self.config = config or TemporalConfig()
        self._validate_config()
        
        # Performance tracking
        self.performance_stats = {}
        
        logger.info("Temporal analyzer initialized")
        logger.info(f"Spatial resolution: {self.config.spatial_resolution}m")
        logger.info(f"Temporal resolution: {self.config.temporal_resolution.value}")
        logger.info(f"Metrics to calculate: {[m.value for m in self.config.metrics]}")
        logger.info(f"Optimization method: {self.config.optimization_method}")
    
    def _validate_config(self) -> None:
        """Validate temporal configuration parameters."""
        if self.config.spatial_resolution <= 0:
            raise ValidationError(
                "Spatial resolution must be positive",
                {"spatial_resolution": self.config.spatial_resolution}
            )
        
        if self.config.chunk_size_km <= 0:
            raise ValidationError(
                "Chunk size must be positive",
                {"chunk_size_km": self.config.chunk_size_km}
            )
        
        if self.config.min_scenes_per_cell < 1:
            raise ValidationError(
                "Minimum scenes per cell must be at least 1",
                {"min_scenes_per_cell": self.config.min_scenes_per_cell}
            )
        
        # Validate spatial resolution is reasonable
        if self.config.spatial_resolution < 3.0:
            logger.warning(f"Very fine resolution ({self.config.spatial_resolution}m) may cause memory issues")
        
        if self.config.spatial_resolution > 1000.0:
            logger.warning(f"Coarse resolution ({self.config.spatial_resolution}m) may lack detail")
    
    def analyze_temporal_patterns(
        self,
        scene_footprints: List[Dict],
        roi_geometry: Union[Dict, Polygon],
        start_date: str,
        end_date: str,
        clip_to_roi: bool = True,
        **kwargs
    ) -> TemporalResult:
        """
        Analyze temporal patterns using grid-based approach.
        
        Args:
            scene_footprints: List of scene features with geometry and properties
            roi_geometry: Region of interest geometry
            start_date: Analysis start date (YYYY-MM-DD)
            end_date: Analysis end date (YYYY-MM-DD)
            clip_to_roi: If True, clip output to ROI shape. If False, analyze full grid
            **kwargs: Additional parameters
            
        Returns:
            TemporalResult with temporal analysis results
        """
        start_time = time.time()
        
        try:
            logger.info("Starting temporal pattern analysis")
            logger.info(f"Date range: {start_date} to {end_date}")
            logger.info(f"Clip to ROI: {clip_to_roi}")
            
            # Validate inputs
            roi_poly = self._prepare_roi_geometry(roi_geometry)
            scene_data = self._prepare_scene_data(scene_footprints, start_date, end_date)
            
            # Update config with kwargs
            config = self._merge_config_kwargs(kwargs)
            
            # Force single chunk for consistency (like spatial density)
            if config.force_single_chunk or config.chunk_size_km >= 200.0:
                logger.info("Using single chunk processing for temporal analysis")
                return self._process_single_temporal_analysis(
                    scene_data, roi_poly, start_date, end_date, config, start_time, clip_to_roi
                )
            
            # Check if ROI needs chunking
            chunks = self._create_spatial_chunks(roi_poly, config)
            
            if len(chunks) > 1:
                logger.info(f"Large ROI detected: processing in {len(chunks)} chunks")
                return self._process_chunked_temporal_analysis(
                    scene_data, chunks, start_date, end_date, config, start_time, clip_to_roi
                )
            else:
                # Single chunk processing (preferred)
                return self._process_single_temporal_analysis(
                    scene_data, roi_poly, start_date, end_date, config, start_time, clip_to_roi
                )
                
        except Exception as e:
            computation_time = time.time() - start_time
            logger.error(f"Temporal analysis failed after {computation_time:.2f}s: {e}")
            if isinstance(e, (ValidationError, PlanetScopeError)):
                raise
            raise PlanetScopeError(f"Temporal analysis error: {e}")
    
    def _prepare_roi_geometry(self, roi_geometry: Union[Dict, Polygon]) -> Polygon:
        """Prepare and validate ROI geometry (same as spatial density)."""
        if isinstance(roi_geometry, dict):
            if roi_geometry.get("type") == "Polygon":
                coords = roi_geometry["coordinates"][0]
                roi_poly = Polygon(coords)
            else:
                raise ValidationError(
                    "Only Polygon ROI supported", {"type": roi_geometry.get("type")}
                )
        elif isinstance(roi_geometry, Polygon):
            roi_poly = roi_geometry
        else:
            raise ValidationError(
                "Invalid ROI geometry type", {"type": type(roi_geometry)}
            )
        
        # Enhanced geometry validation
        if self.config.validate_geometries:
            if not roi_poly.is_valid:
                logger.warning("Invalid ROI geometry detected, attempting to fix")
                roi_poly = roi_poly.buffer(0)
                
                if not roi_poly.is_valid:
                    raise ValidationError("Could not fix invalid ROI geometry")
            
            if roi_poly.is_empty:
                raise ValidationError("ROI geometry is empty")
        
        return roi_poly
    
    def _prepare_scene_data(self, scene_footprints: List[Dict], start_date: str, end_date: str) -> pd.DataFrame:
        """Prepare scene data with temporal information."""
        scene_data = []
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        for i, scene in enumerate(scene_footprints):
            try:
                # Extract geometry
                geom = scene.get("geometry")
                if not geom:
                    logger.warning(f"Scene {i} missing geometry, skipping")
                    continue
                
                if geom["type"] == "Polygon":
                    coords = geom["coordinates"][0]
                    poly = Polygon(coords)
                elif geom["type"] == "MultiPolygon":
                    # Take largest polygon
                    polygons = [Polygon(ring[0]) for ring in geom["coordinates"]]
                    poly = max(polygons, key=lambda p: p.area)
                else:
                    logger.warning(f"Scene {i} unsupported geometry type: {geom['type']}")
                    continue
                
                # Extract acquisition date
                properties = scene.get("properties", {})
                acquired = properties.get("acquired")
                if not acquired:
                    logger.warning(f"Scene {i} missing acquisition date, skipping")
                    continue
                
                # Parse acquisition date
                if isinstance(acquired, str):
                    # Handle different date formats
                    if "T" in acquired:
                        acq_date = datetime.strptime(acquired.split("T")[0], "%Y-%m-%d")
                    else:
                        acq_date = datetime.strptime(acquired, "%Y-%m-%d")
                else:
                    logger.warning(f"Scene {i} invalid date format: {acquired}")
                    continue
                
                # Filter by date range
                if start_dt <= acq_date <= end_dt:
                    scene_data.append({
                        'scene_id': properties.get('id', f'scene_{i}'),
                        'geometry': poly,
                        'acquired_date': acq_date,
                        'acquired_str': acq_date.strftime("%Y-%m-%d"),
                        'cloud_cover': properties.get('cloud_cover', 0),
                        'properties': properties
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to process scene {i}: {e}")
                continue
        
        if not scene_data:
            raise ValidationError("No valid scenes found in date range")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(scene_data)
        logger.info(f"Prepared {len(df)} scenes for temporal analysis")
        logger.info(f"Date range in data: {df['acquired_str'].min()} to {df['acquired_str'].max()}")
        
        return df
    
    def _merge_config_kwargs(self, kwargs: Dict) -> TemporalConfig:
        """Merge configuration with keyword arguments."""
        config = TemporalConfig(
            spatial_resolution=kwargs.get("spatial_resolution", self.config.spatial_resolution),
            temporal_resolution=kwargs.get("temporal_resolution", self.config.temporal_resolution),
            metrics=kwargs.get("metrics", self.config.metrics),
            chunk_size_km=kwargs.get("chunk_size_km", self.config.chunk_size_km),
            max_memory_gb=kwargs.get("max_memory_gb", self.config.max_memory_gb),
            parallel_workers=kwargs.get("parallel_workers", self.config.parallel_workers),
            no_data_value=kwargs.get("no_data_value", self.config.no_data_value),
            coordinate_system_fixes=kwargs.get("coordinate_system_fixes", self.config.coordinate_system_fixes),
            force_single_chunk=kwargs.get("force_single_chunk", self.config.force_single_chunk),
            validate_geometries=kwargs.get("validate_geometries", self.config.validate_geometries),
            min_scenes_per_cell=kwargs.get("min_scenes_per_cell", self.config.min_scenes_per_cell),
            optimization_method=kwargs.get("optimization_method", self.config.optimization_method),  # FIXED: Add this
        )
        
        return config
    
    def _create_spatial_chunks(self, roi_poly: Polygon, config: TemporalConfig) -> List[Polygon]:
        """Create spatial chunks for large ROI processing (same as spatial density)."""
        bounds = roi_poly.bounds
        roi_width_km = (bounds[2] - bounds[0]) * 111.0  # Rough conversion to km
        roi_height_km = (bounds[3] - bounds[1]) * 111.0
        
        # Check if chunking needed
        if max(roi_width_km, roi_height_km) <= config.chunk_size_km:
            logger.info("ROI fits within single chunk")
            return [roi_poly]
        
        # Calculate chunk grid
        n_chunks_x = int(np.ceil(roi_width_km / config.chunk_size_km))
        n_chunks_y = int(np.ceil(roi_height_km / config.chunk_size_km))
        
        logger.info(f"Creating {n_chunks_x}x{n_chunks_y} spatial chunks")
        
        chunks = []
        chunk_width = (bounds[2] - bounds[0]) / n_chunks_x
        chunk_height = (bounds[3] - bounds[1]) / n_chunks_y
        
        for i in range(n_chunks_x):
            for j in range(n_chunks_y):
                minx = bounds[0] + i * chunk_width
                maxx = bounds[0] + (i + 1) * chunk_width
                miny = bounds[1] + j * chunk_height
                maxy = bounds[1] + (j + 1) * chunk_height
                
                chunk_box = box(minx, miny, maxx, maxy)
                chunk_roi = chunk_box.intersection(roi_poly)
                
                if not chunk_roi.is_empty:
                    chunks.append(chunk_roi)
        
        return chunks
    
    def _process_single_temporal_analysis(
        self,
        scene_data: pd.DataFrame,
        roi_poly: Polygon,
        start_date: str,
        end_date: str,
        config: TemporalConfig,
        start_time: float,
        clip_to_roi: bool = True,
    ) -> TemporalResult:
        """Process temporal analysis for single ROI with coordinate fixes."""
        logger.info("Executing grid-based temporal analysis")
        
        if clip_to_roi:
            # Use ROI bounds and apply mask
            bounds = roi_poly.bounds
            logger.info(f"ROI bounds: {bounds}")
            apply_roi_mask = True
        else:
            # Use bounds covering all scene footprints for full grid analysis
            scene_bounds_list = [geom.bounds for geom in scene_data['geometry'] if geom.intersects(roi_poly)]
            if not scene_bounds_list:
                bounds = roi_poly.bounds
                apply_roi_mask = True
            else:
                # Create bounds covering all intersecting scene footprints
                all_minx = min(b[0] for b in scene_bounds_list)
                all_miny = min(b[1] for b in scene_bounds_list)
                all_maxx = max(b[2] for b in scene_bounds_list)
                all_maxy = max(b[3] for b in scene_bounds_list)
                
                # Expand bounds to include ROI as well
                roi_bounds = roi_poly.bounds
                bounds = (
                    min(all_minx, roi_bounds[0]),
                    min(all_miny, roi_bounds[1]),
                    max(all_maxx, roi_bounds[2]),
                    max(all_maxy, roi_bounds[3])
                )
                apply_roi_mask = False
                logger.info(f"Full grid bounds (all scenes + ROI): {bounds}")
        
        # Calculate raster dimensions (same as spatial density)
        resolution_deg = config.spatial_resolution / 111000  # Convert meters to degrees
        width = int((bounds[2] - bounds[0]) / resolution_deg)
        height = int((bounds[3] - bounds[1]) / resolution_deg)
        
        total_cells = width * height
        
        # FIXED: Proper optimization method selection with auto-detection
        optimization_method = config.optimization_method
        
        # Honor user's explicit choice first
        if optimization_method != "auto":
            logger.info(f"Using user-specified {optimization_method.upper()} optimization method")
        elif total_cells > 500_000:
            optimization_method = "fast"
            logger.warning(f"Large grid detected: {width}x{height} = {total_cells:,} cells")
            logger.warning(f"Automatically switching to FAST optimization for better performance")
        else:
            optimization_method = "accurate"
            logger.info(f"Using ACCURATE method for medium-sized grid ({total_cells:,} cells)")
        
        logger.info(f"Grid dimensions: {width} x {height} cells ({total_cells:,} total)")
        logger.info(f"Spatial resolution: {config.spatial_resolution}m ({resolution_deg:.6f} degrees)")
        logger.info(f"Using {optimization_method.upper()} optimization method")
        
        # Create corrected transform with proper orientation (same as spatial density)
        if config.coordinate_system_fixes:
            pixel_width = (bounds[2] - bounds[0]) / width   # Positive (west to east)
            pixel_height = -(bounds[3] - bounds[1]) / height  # Negative (north to south)
            
            transform = Affine(
                pixel_width, 0.0, bounds[0],      # X: west to east
                0.0, pixel_height, bounds[3],     # Y: north to south (negative height)
                0.0, 0.0, 1.0
            )
            
            logger.info(f"Applied coordinate system fixes:")
            logger.info(f"  Pixel width: {pixel_width:.8f} (positive)")
            logger.info(f"  Pixel height: {pixel_height:.8f} (negative, north-to-south)")
            logger.info(f"  Origin: Northwest corner ({bounds[0]:.6f}, {bounds[3]:.6f})")
        else:
            transform = from_bounds(
                bounds[0], bounds[1], bounds[2], bounds[3], width, height
            )
            logger.warning("Coordinate system fixes disabled")
        
        # FIXED: Choose optimization method based on user's choice
        if optimization_method == "fast":
            metric_arrays = self._calculate_temporal_metrics_grid_fast(
                scene_data, transform, width, height, config
            )
        else:  # "accurate"
            metric_arrays = self._calculate_temporal_metrics_grid_accurate(
                scene_data, transform, width, height, config
            )
        
        # Apply ROI mask if requested
        if apply_roi_mask:
            roi_mask = rasterize(
                [(roi_poly, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            
            # Apply mask to all metric arrays
            for metric, array in metric_arrays.items():
                metric_arrays[metric] = np.where(roi_mask == 1, array, config.no_data_value)
            
            valid_pixels = np.sum(roi_mask == 1)
            total_pixels = roi_mask.size
            logger.info(f"ROI masking applied: {valid_pixels:,}/{total_pixels:,} valid pixels")
        else:
            # Count valid pixels (non-no-data)
            sample_array = list(metric_arrays.values())[0]
            valid_pixels = np.sum(sample_array != config.no_data_value)
            total_pixels = sample_array.size
            logger.info(f"Full grid analysis: {valid_pixels:,}/{total_pixels:,} pixels with data")
        
        # Calculate comprehensive temporal statistics
        temporal_stats = self._calculate_temporal_statistics(
            metric_arrays, scene_data, start_date, end_date, config.no_data_value
        )
        
        computation_time = time.time() - start_time
        
        result = TemporalResult(
            metric_arrays=metric_arrays,
            transform=transform,
            crs="EPSG:4326",
            bounds=bounds,
            temporal_stats=temporal_stats,
            computation_time=computation_time,
            config=config,
            grid_info={
                "width": width,
                "height": height,
                "spatial_resolution": config.spatial_resolution,
                "temporal_resolution": config.temporal_resolution.value,
                "resolution_degrees": resolution_deg,
                "total_cells": width * height,
                "valid_cells": valid_pixels,
                "coverage_percent": 100 * valid_pixels / total_pixels,
                "roi_clipped": apply_roi_mask,
                "optimization_method": optimization_method,
            },
            date_range=(start_date, end_date),
            coordinate_system_corrected=config.coordinate_system_fixes,
            no_data_value=config.no_data_value,
        )
        
        analysis_type = "ROI-clipped" if apply_roi_mask else "full grid"
        logger.info(f"Temporal analysis ({analysis_type}) completed in {computation_time:.2f}s")
        logger.info(f"Mean coverage days: {temporal_stats.get('mean_coverage_days', 0):.1f}")
        
        return result
    
    def _calculate_temporal_metrics_grid_fast(
        self,
        scene_data: pd.DataFrame,
        transform: Affine,
        width: int,
        height: int,
        config: TemporalConfig
    ) -> Dict[TemporalMetric, np.ndarray]:
        """
        FAST temporal metrics calculation using vectorized operations.
        
        This approach is similar to spatial density analysis - process by dates
        instead of cells for much better performance.
        """
        logger.info("Calculating temporal metrics using FAST vectorized approach")
        
        # Initialize output arrays for each metric
        metric_arrays = {}
        for metric in config.metrics:
            metric_arrays[metric] = np.full((height, width), config.no_data_value, dtype=np.float32)
        
        # Group scenes by date for faster processing
        scene_data['date_only'] = scene_data['acquired_date'].dt.date
        scenes_by_date = scene_data.groupby('date_only')
        unique_dates = sorted(scenes_by_date.groups.keys())
        
        logger.info(f"Processing {len(unique_dates)} unique dates with vectorized operations")
        
        # STEP 1: Create coverage masks for each date (vectorized like spatial density)
        date_coverage_masks = {}
        total_scenes_mask = np.zeros((height, width), dtype=np.float32)
        
        for i, (date, scenes_group) in enumerate(scenes_by_date):
            # Rasterize all scenes for this date at once (FAST like spatial density)
            date_mask = np.zeros((height, width), dtype=np.uint8)
            
            for _, scene in scenes_group.iterrows():
                scene_mask = rasterize(
                    [(scene['geometry'], 1)],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                )
                date_mask = np.maximum(date_mask, scene_mask)  # Union of scenes for this date
            
            date_coverage_masks[date] = date_mask
            total_scenes_mask += date_mask.astype(np.float32)
            
            # Progress logging
            if (i + 1) % 10 == 0 or i == len(scenes_by_date) - 1:
                logger.info(f"Processed {i + 1}/{len(unique_dates)} dates ({(i+1)/len(unique_dates)*100:.1f}%)")
        
        # STEP 2: Calculate temporal metrics using vectorized operations
        logger.info("Calculating temporal metrics using array operations")
        
        # Coverage Days metric (fast)
        if TemporalMetric.COVERAGE_DAYS in config.metrics:
            coverage_days = np.zeros((height, width), dtype=np.float32)
            for date_mask in date_coverage_masks.values():
                coverage_days += date_mask.astype(np.float32)
            
            # Apply minimum scenes filter
            coverage_days = np.where(
                coverage_days >= config.min_scenes_per_cell,
                coverage_days,
                config.no_data_value
            )
            metric_arrays[TemporalMetric.COVERAGE_DAYS] = coverage_days
        
        # Temporal Density metric (fast)
        if TemporalMetric.TEMPORAL_DENSITY in config.metrics:
            analysis_days = (unique_dates[-1] - unique_dates[0]).days + 1
            temporal_density = total_scenes_mask / analysis_days
            
            # Apply minimum scenes filter
            temporal_density = np.where(
                total_scenes_mask >= config.min_scenes_per_cell,
                temporal_density,
                config.no_data_value
            )
            metric_arrays[TemporalMetric.TEMPORAL_DENSITY] = temporal_density
        
        # Interval-based metrics (more complex but still optimized)
        if any(m in config.metrics for m in [
            TemporalMetric.MEAN_INTERVAL, TemporalMetric.MEDIAN_INTERVAL,
            TemporalMetric.MIN_INTERVAL, TemporalMetric.MAX_INTERVAL
        ]):
            logger.info("Calculating interval-based metrics with optimized approach")
            
            # Initialize interval arrays
            mean_intervals = np.full((height, width), config.no_data_value, dtype=np.float32)
            median_intervals = np.full((height, width), config.no_data_value, dtype=np.float32)
            min_intervals = np.full((height, width), config.no_data_value, dtype=np.float32)
            max_intervals = np.full((height, width), config.no_data_value, dtype=np.float32)
            
            # Process intervals using optimized chunks
            self._calculate_intervals_fast(
                date_coverage_masks, unique_dates, config,
                mean_intervals, median_intervals, min_intervals, max_intervals,
                height, width
            )
            
            # Store results
            if TemporalMetric.MEAN_INTERVAL in config.metrics:
                metric_arrays[TemporalMetric.MEAN_INTERVAL] = mean_intervals
            if TemporalMetric.MEDIAN_INTERVAL in config.metrics:
                metric_arrays[TemporalMetric.MEDIAN_INTERVAL] = median_intervals
            if TemporalMetric.MIN_INTERVAL in config.metrics:
                metric_arrays[TemporalMetric.MIN_INTERVAL] = min_intervals
            if TemporalMetric.MAX_INTERVAL in config.metrics:
                metric_arrays[TemporalMetric.MAX_INTERVAL] = max_intervals
        
        # Count cells with valid data
        sample_array = list(metric_arrays.values())[0]
        cells_with_data = np.sum(sample_array != config.no_data_value)
        total_cells = sample_array.size
        
        logger.info(f"FAST temporal metrics calculation complete")
        logger.info(f"Cells with temporal data: {cells_with_data:,}/{total_cells:,} ({cells_with_data/total_cells*100:.1f}%)")
        
        return metric_arrays
    
    def _calculate_intervals_fast(
        self,
        date_coverage_masks: Dict,
        unique_dates: List,
        config: TemporalConfig,
        mean_intervals: np.ndarray,
        median_intervals: np.ndarray, 
        min_intervals: np.ndarray,
        max_intervals: np.ndarray,
        height: int,
        width: int
    ):
        """
        Calculate interval-based metrics using optimized chunk processing.
        
        Process only cells that meet minimum requirements for better performance.
        """
        # Create a combined mask of all cells that have enough coverage
        total_coverage = np.zeros((height, width), dtype=np.float32)
        for date_mask in date_coverage_masks.values():
            total_coverage += date_mask.astype(np.float32)
        
        # Find cells that meet minimum requirements
        valid_cells_mask = total_coverage >= config.min_scenes_per_cell
        valid_indices = np.where(valid_cells_mask)
        
        if len(valid_indices[0]) == 0:
            logger.warning("No cells meet minimum scenes requirement for interval calculation")
            return
        
        logger.info(f"Processing intervals for {len(valid_indices[0]):,} valid cells using chunks")
        
        # Process valid cells in chunks for better performance
        chunk_size = 1000  # Process 1000 cells at a time
        n_valid_cells = len(valid_indices[0])
        
        for chunk_start in range(0, n_valid_cells, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_valid_cells)
            
            # Process chunk
            for idx in range(chunk_start, chunk_end):
                i, j = valid_indices[0][idx], valid_indices[1][idx]
                
                # Find dates when this cell has coverage
                cell_dates = []
                for date, mask in date_coverage_masks.items():
                    if mask[i, j] == 1:
                        cell_dates.append(date)
                
                if len(cell_dates) >= config.min_scenes_per_cell:
                    # Calculate intervals between consecutive dates
                    cell_dates.sort()
                    intervals = []
                    for k in range(1, len(cell_dates)):
                        interval_days = (cell_dates[k] - cell_dates[k-1]).days
                        intervals.append(interval_days)
                    
                    if intervals:
                        mean_intervals[i, j] = np.mean(intervals)
                        median_intervals[i, j] = np.median(intervals)
                        min_intervals[i, j] = min(intervals)
                        max_intervals[i, j] = max(intervals)
            
            # Progress logging
            if (chunk_end) % 10000 == 0 or chunk_end == n_valid_cells:
                progress = chunk_end / n_valid_cells * 100
                logger.info(f"Interval calculation: {chunk_end:,}/{n_valid_cells:,} cells ({progress:.1f}%)")
    
    def _calculate_temporal_metrics_grid_accurate(
        self,
        scene_data: pd.DataFrame,
        transform: Affine,
        width: int,
        height: int,
        config: TemporalConfig
    ) -> Dict[TemporalMetric, np.ndarray]:
        """
        ACCURATE temporal metrics calculation using cell-by-cell processing.
        
        This is the original implementation that processes each cell individually
        for maximum accuracy but slower performance.
        """
        logger.info("Calculating temporal metrics using ACCURATE cell-by-cell approach")
        
        # Initialize output arrays for each metric
        metric_arrays = {}
        for metric in config.metrics:
            metric_arrays[metric] = np.full((height, width), config.no_data_value, dtype=np.float32)
        
        # Create spatial index for scenes
        scene_gdf = gpd.GeoDataFrame(scene_data)
        scene_sindex = scene_gdf.sindex
        
        # Process each grid cell
        cells_processed = 0
        cells_with_data = 0
        total_cells = width * height
        
        for i in range(height):
            for j in range(width):
                # Calculate cell bounds from transform
                x_min = transform.c + j * transform.a
                x_max = transform.c + (j + 1) * transform.a
                y_max = transform.f + i * transform.e  # Note: e is negative
                y_min = transform.f + (i + 1) * transform.e
                
                # Create cell geometry
                cell = box(x_min, y_min, x_max, y_max)
                
                # Find intersecting scenes using spatial index
                possible_matches_index = list(scene_sindex.intersection(cell.bounds))
                if not possible_matches_index:
                    cells_processed += 1
                    continue
                
                # Get scenes that actually intersect the cell
                intersecting_scenes = []
                for idx in possible_matches_index:
                    scene_geom = scene_data.iloc[idx]['geometry']
                    if scene_geom.intersects(cell):
                        intersecting_scenes.append(scene_data.iloc[idx])
                
                if len(intersecting_scenes) < config.min_scenes_per_cell:
                    cells_processed += 1
                    continue
                
                # Calculate temporal metrics for this cell
                cell_metrics = self._calculate_cell_temporal_metrics(
                    intersecting_scenes, config
                )
                
                # Store metrics in arrays
                for metric in config.metrics:
                    if metric in cell_metrics:
                        metric_arrays[metric][i, j] = cell_metrics[metric]
                
                cells_with_data += 1
                cells_processed += 1
                
                # Progress logging
                if cells_processed % 10000 == 0:
                    progress = cells_processed / total_cells * 100
                    logger.info(f"Processed {cells_processed:,}/{total_cells:,} cells ({progress:.1f}%)")
        
        logger.info(f"ACCURATE temporal metrics calculation complete")
        logger.info(f"Cells with temporal data: {cells_with_data:,}/{total_cells:,}")
        
        return metric_arrays
    
    def _calculate_cell_temporal_metrics(
        self,
        intersecting_scenes: List[Dict],
        config: TemporalConfig
    ) -> Dict[TemporalMetric, float]:
        """Calculate temporal metrics for a single grid cell."""
        # Extract unique acquisition dates for this cell
        dates = []
        for scene in intersecting_scenes:
            dates.append(scene['acquired_date'])
        
        # Group by date (handle multiple scenes on same day)
        unique_dates = sorted(list(set(dates)))
        
        if len(unique_dates) < config.min_scenes_per_cell:
            return {}
        
        # Calculate temporal intervals between consecutive dates
        intervals = []
        for i in range(1, len(unique_dates)):
            interval_days = (unique_dates[i] - unique_dates[i-1]).days
            intervals.append(interval_days)
        
        # Calculate metrics
        metrics = {}
        
        if TemporalMetric.COVERAGE_DAYS in config.metrics:
            metrics[TemporalMetric.COVERAGE_DAYS] = len(unique_dates)
        
        if intervals:  # Need at least 2 dates for interval calculations
            if TemporalMetric.MEAN_INTERVAL in config.metrics:
                metrics[TemporalMetric.MEAN_INTERVAL] = np.mean(intervals)
            
            if TemporalMetric.MEDIAN_INTERVAL in config.metrics:
                metrics[TemporalMetric.MEDIAN_INTERVAL] = np.median(intervals)
            
            if TemporalMetric.MIN_INTERVAL in config.metrics:
                metrics[TemporalMetric.MIN_INTERVAL] = min(intervals)
            
            if TemporalMetric.MAX_INTERVAL in config.metrics:
                metrics[TemporalMetric.MAX_INTERVAL] = max(intervals)
        
        if TemporalMetric.TEMPORAL_DENSITY in config.metrics:
            # Calculate scenes per day over the date range
            date_range_days = (unique_dates[-1] - unique_dates[0]).days + 1
            if date_range_days > 0:
                metrics[TemporalMetric.TEMPORAL_DENSITY] = len(dates) / date_range_days
        
        if TemporalMetric.COVERAGE_FREQUENCY in config.metrics:
            # Calculate coverage frequency (days with coverage / analysis period)
            analysis_start = datetime.strptime(config.date_range[0], "%Y-%m-%d") if hasattr(config, 'date_range') else unique_dates[0]
            analysis_end = datetime.strptime(config.date_range[1], "%Y-%m-%d") if hasattr(config, 'date_range') else unique_dates[-1]
            analysis_days = (analysis_end - analysis_start).days + 1
            if analysis_days > 0:
                metrics[TemporalMetric.COVERAGE_FREQUENCY] = len(unique_dates) / analysis_days
        
        return metrics
    
    def _calculate_temporal_statistics(
        self,
        metric_arrays: Dict[TemporalMetric, np.ndarray],
        scene_data: pd.DataFrame,
        start_date: str,
        end_date: str,
        no_data_value: float
    ) -> Dict[str, Any]:
        """Calculate comprehensive temporal statistics."""
        stats = {
            'analysis_period': {
                'start_date': start_date,
                'end_date': end_date,
                'total_days': (datetime.strptime(end_date, "%Y-%m-%d") - 
                              datetime.strptime(start_date, "%Y-%m-%d")).days + 1
            },
            'scene_summary': {
                'total_scenes': len(scene_data),
                'unique_dates': len(scene_data['acquired_str'].unique()),
                'date_range_in_data': {
                    'first_scene': scene_data['acquired_str'].min(),
                    'last_scene': scene_data['acquired_str'].max()
                }
            }
        }
        
        # Calculate statistics for each metric
        for metric, array in metric_arrays.items():
            valid_data = array[array != no_data_value]
            
            if len(valid_data) > 0:
                metric_stats = {
                    'count': int(len(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'median': float(np.median(valid_data)),
                    'percentiles': {
                        '25': float(np.percentile(valid_data, 25)),
                        '75': float(np.percentile(valid_data, 75)),
                        '90': float(np.percentile(valid_data, 90)),
                        '95': float(np.percentile(valid_data, 95)),
                    }
                }
            else:
                metric_stats = {'error': 'No valid data'}
            
            stats[f'{metric.value}_stats'] = metric_stats
        
        # Add summary statistics
        if TemporalMetric.COVERAGE_DAYS in metric_arrays:
            coverage_array = metric_arrays[TemporalMetric.COVERAGE_DAYS]
            valid_coverage = coverage_array[coverage_array != no_data_value]
            if len(valid_coverage) > 0:
                stats['mean_coverage_days'] = float(np.mean(valid_coverage))
                stats['total_coverage_observations'] = float(np.sum(valid_coverage))
        
        return stats
    
    def _process_chunked_temporal_analysis(
        self,
        scene_data: pd.DataFrame,
        chunks: List[Polygon],
        start_date: str,
        end_date: str,
        config: TemporalConfig,
        start_time: float,
        clip_to_roi: bool = True,
    ) -> TemporalResult:
        """Process temporal analysis for chunked ROI (simplified implementation)."""
        logger.info("Processing chunked temporal analysis")
        logger.warning("Multi-chunk processing may have coordinate alignment issues")
        
        # For now, process first chunk only (like spatial density)
        # Full mosaicking would require complex merging logic
        first_chunk = chunks[0]
        
        result = self._process_single_temporal_analysis(
            scene_data, first_chunk, start_date, end_date, config, start_time, clip_to_roi
        )
        
        # Update metadata to indicate chunking
        result.grid_info["chunks_processed"] = len(chunks)
        result.grid_info["chunking_method"] = "simplified"
        result.computation_time = time.time() - start_time
        
        logger.warning(
            f"Simplified chunk processing used - processed first chunk only. "
            f"Full mosaic implementation needed for {len(chunks)} chunks."
        )
        
        return result
    
    def export_temporal_geotiffs(
        self,
        result: TemporalResult,
        output_dir: str,
        roi_polygon: Optional[Polygon] = None,
        clip_to_roi: bool = True,
        compress: str = "lzw"
    ) -> Dict[str, str]:
        """
        Export temporal analysis results as GeoTIFF files with robust PROJ handling.
        
        Args:
            result: TemporalResult object
            output_dir: Output directory for GeoTIFF files
            roi_polygon: ROI polygon for clipping
            clip_to_roi: Whether to clip to ROI shape
            compress: Compression method
            
        Returns:
            dict: Paths to exported files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        for metric, array in result.metric_arrays.items():
            try:
                # Apply ROI clipping if requested
                export_array = array
                if clip_to_roi and roi_polygon is not None:
                    export_array = self._clip_array_to_roi(
                        array, result.transform, roi_polygon, result.no_data_value
                    )
                
                # Create output filename
                metric_name = metric.value
                if clip_to_roi and roi_polygon is not None:
                    filename = f"temporal_{metric_name}_clipped.tif"
                else:
                    filename = f"temporal_{metric_name}.tif"
                
                output_path = os.path.join(output_dir, filename)
                
                # Try multiple CRS approaches for PROJ compatibility
                crs_options = [
                    "EPSG:4326",
                    "+proj=longlat +datum=WGS84 +no_defs",
                    None
                ]
                
                success = False
                for crs_option in crs_options:
                    try:
                        logger.info(f"Exporting {metric_name} with CRS: {crs_option}")
                        
                        with rasterio.open(
                            output_path, "w", driver="GTiff",
                            height=export_array.shape[0],
                            width=export_array.shape[1],
                            count=1, dtype=export_array.dtype,
                            crs=crs_option, transform=result.transform,
                            compress=compress, nodata=result.no_data_value,
                        ) as dst:
                            dst.write(export_array, 1)
                            
                            # Add comprehensive metadata
                            metadata = {
                                'title': f'PlanetScope Temporal Analysis - {metric_name}',
                                'description': f'Temporal {metric_name} analysis with coordinate system fixes',
                                'metric': metric_name,
                                'spatial_resolution': str(result.config.spatial_resolution),
                                'temporal_resolution': result.config.temporal_resolution.value,
                                'date_range': f"{result.date_range[0]} to {result.date_range[1]}",
                                'computation_time': str(result.computation_time),
                                'coordinate_system_corrected': str(result.coordinate_system_corrected),
                                'crs_used': str(crs_option) if crs_option else 'none',
                                'roi_clipped': str(clip_to_roi and roi_polygon is not None),
                                'no_data_value': str(result.no_data_value),
                                'optimization_method': result.grid_info.get('optimization_method', 'unknown'),
                                'created_by': 'PlanetScope-py Temporal Analysis Module v1.2.0 (COMPLETE)',
                            }
                            dst.update_tags(**metadata)
                        
                        success = True
                        break
                        
                    except Exception as e:
                        logger.warning(f"Export failed with CRS {crs_option}: {e}")
                        continue
                
                if success:
                    exported_files[metric_name] = output_path
                    
                    # Create QML style file
                    qml_path = output_path.replace('.tif', '.qml')
                    self._create_temporal_qml_style(export_array, qml_path, metric, result.no_data_value)
                    exported_files[f"{metric_name}_qml"] = qml_path
                    
                    logger.info(f"Exported {metric_name} to: {output_path}")
                else:
                    logger.error(f"Failed to export {metric_name}")
                    
            except Exception as e:
                logger.error(f"Export failed for {metric_name}: {e}")
                continue
        
        # Export metadata as JSON
        metadata_path = os.path.join(output_dir, "temporal_analysis_metadata.json")
        self._export_temporal_metadata(result, metadata_path)
        exported_files['metadata'] = metadata_path
        
        logger.info(f"Temporal analysis export completed: {len(exported_files)} files")
        return exported_files
    
    def _clip_array_to_roi(
        self,
        array: np.ndarray,
        transform: Affine,
        roi_polygon: Polygon,
        no_data_value: float
    ) -> np.ndarray:
        """Clip array to ROI polygon shape (same as visualization module)."""
        try:
            height, width = array.shape
            roi_mask = rasterize(
                [(roi_polygon, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            
            clipped_array = np.where(roi_mask == 1, array, no_data_value)
            
            valid_pixels = np.sum(roi_mask == 1)
            total_pixels = roi_mask.size
            logger.info(f"ROI clipping: {valid_pixels:,}/{total_pixels:,} valid pixels")
            
            return clipped_array
            
        except Exception as e:
            logger.error(f"Failed to clip array to ROI: {e}")
            return array
    
    def _create_temporal_qml_style(
        self,
        array: np.ndarray,
        qml_path: str,
        metric: TemporalMetric,
        no_data_value: float
    ):
        """Create QGIS style file for temporal analysis raster."""
        try:
            valid_data = array[array != no_data_value]
            if len(valid_data) == 0:
                min_val, max_val = 0, 1
            else:
                min_val = float(np.min(valid_data))
                max_val = float(np.max(valid_data))
            
            # Choose colors based on metric type
            if metric in [TemporalMetric.COVERAGE_DAYS, TemporalMetric.TEMPORAL_DENSITY]:
                # Green palette for positive metrics (more = better)
                color_ramp = {
                    'color1': "255,255,229,255",  # Light yellow
                    'color2': "0,104,55,255",     # Dark green
                    'stops': "0.25;194,230,153,255:0.5;120,198,121,255:0.75;49,163,84,255"
                }
            elif metric in [TemporalMetric.MEAN_INTERVAL, TemporalMetric.MEDIAN_INTERVAL]:
                # Red-yellow palette for intervals (less = better)
                color_ramp = {
                    'color1': "255,255,178,255",  # Light yellow
                    'color2': "189,0,38,255",     # Dark red
                    'stops': "0.25;254,204,92,255:0.5;253,141,60,255:0.75;227,26,28,255"
                }
            else:
                # Blue palette for other metrics
                color_ramp = {
                    'color1': "247,251,255,255",  # Light blue
                    'color2': "8,48,107,255",     # Dark blue
                    'stops': "0.25;198,219,239,255:0.5;107,174,214,255:0.75;33,113,181,255"
                }
            
            # Create color stops
            color_stops = [
                (min_val, color_ramp['color1']),
                (min_val + (max_val-min_val)*0.25, color_ramp['stops'].split(':')[0].split(';')[1]),
                (min_val + (max_val-min_val)*0.5, color_ramp['stops'].split(':')[1].split(';')[1]),
                (min_val + (max_val-min_val)*0.75, color_ramp['stops'].split(':')[2].split(';')[1]),
                (max_val, color_ramp['color2'])
            ]
            
            # Create QML content
            qml_content = f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" hasScaleBasedVisibilityFlag="0" styleCategories="AllStyleCategories">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <pipe>
    <provider>
      <resampling enabled="false" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer alphaBand="-1" opacity="1" type="singlebandpseudocolor" band="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>MinMax</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader minimumValue="{min_val}" maximumValue="{max_val}" colorRampType="INTERPOLATED" classificationMode="1" clip="0">
          <colorramp type="gradient" name="{metric.value}">
            <Option type="Map">
              <Option type="QString" name="color1" value="{color_ramp['color1']}"/>
              <Option type="QString" name="color2" value="{color_ramp['color2']}"/>
              <Option type="QString" name="stops" value="{color_ramp['stops']}"/>
            </Option>
          </colorramp>'''
            
            # Add color stops
            for value, color in color_stops:
                qml_content += f'\n          <item alpha="255" value="{value}" label="{value:.1f}" color="{color}"/>'
            
            qml_content += f'''
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" gamma="1" contrast="0"/>
    <huesaturation colorizeOn="0" grayscaleMode="0" saturation="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
</qgis>'''
            
            with open(qml_path, 'w', encoding='utf-8') as f:
                f.write(qml_content)
            
            logger.info(f"QML style file created: {qml_path}")
            
        except Exception as e:
            logger.warning(f"QML creation failed for {metric.value}: {e}")
    
    def _export_temporal_metadata(self, result: TemporalResult, metadata_path: str):
        """Export comprehensive temporal analysis metadata with JSON serialization fixes."""
        try:
            import json
            
            # FIXED: Convert numpy types to native Python types for JSON serialization
            def convert_to_json_serializable(obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                elif hasattr(obj, 'isoformat'):  # datetime objects
                    return obj.isoformat()
                else:
                    return obj
            
            metadata = {
                "temporal_analysis_info": {
                    "analysis_type": "grid_based_temporal_patterns",
                    "timestamp": datetime.now().isoformat(),
                    "library_version": "1.2.0 (COMPLETE)",
                    "computation_time_seconds": float(result.computation_time),
                    "coordinate_system_corrected": bool(result.coordinate_system_corrected),
                    "optimization_method": result.grid_info.get("optimization_method", "unknown"),
                },
                "temporal_parameters": {
                    "date_range": {
                        "start_date": result.date_range[0],
                        "end_date": result.date_range[1],
                        "total_days": int(result.temporal_stats['analysis_period']['total_days'])
                    },
                    "temporal_resolution": result.config.temporal_resolution.value,
                    "metrics_calculated": [m.value for m in result.config.metrics],
                    "min_scenes_per_cell": int(result.config.min_scenes_per_cell),
                },
                "spatial_parameters": {
                    "spatial_resolution_meters": float(result.config.spatial_resolution),
                    "crs": result.crs,
                    "bounds": list(result.bounds),
                    "coordinate_system_fixes": bool(result.coordinate_system_corrected),
                },
                "grid_info": convert_to_json_serializable(result.grid_info),
                "temporal_statistics": convert_to_json_serializable(result.temporal_stats),
                "coordinate_system": {
                    "crs": result.crs,
                    "transform_corrected": bool(result.coordinate_system_corrected),
                    "pixel_orientation": "north_to_south",
                    "display_orientation": "corrected_for_visualization",
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Temporal metadata exported to: {metadata_path}")
            
        except Exception as e:
            logger.warning(f"Metadata export failed: {e}")


# High-level workflow functions for easy access
def analyze_temporal_patterns(
    roi_polygon: Union[Polygon, list, dict],
    time_period: Union[str, tuple],
    spatial_resolution: float = 30.0,
    cloud_cover_max: float = 0.3,
    output_dir: str = "./temporal_analysis",
    clip_to_roi: bool = True,
    metrics: Optional[List[TemporalMetric]] = None,
    create_visualizations: bool = True,
    export_geotiffs: bool = True,
    show_plots: bool = True,
    optimization_level: str = "auto",  # FIXED: "fast", "accurate", "auto"
    **kwargs
) -> Dict[str, Any]:
    """
    High-level function for temporal pattern analysis.
    
    Args:
        roi_polygon: Region of interest (Polygon, coordinate list, or GeoJSON dict)
        time_period: Time period as "YYYY-MM-DD/YYYY-MM-DD" or tuple
        spatial_resolution: Spatial grid resolution in meters (default: 30m)
        cloud_cover_max: Maximum cloud coverage threshold (0.0-1.0)
        output_dir: Output directory for results
        clip_to_roi: If True, clip analysis to ROI shape. If False, analyze full grid
        metrics: List of temporal metrics to calculate (None = default set)
        create_visualizations: Whether to generate plots
        export_geotiffs: Whether to export GeoTIFF files
        show_plots: Whether to display plots (default: True)
        optimization_level: Performance optimization ("fast", "accurate", "auto")
        **kwargs: Additional parameters
        
    Returns:
        dict: Complete temporal analysis results
        
    Example:
        >>> from planetscope_py import analyze_temporal_patterns
        >>> from shapely.geometry import Polygon
        >>> 
        >>> milan_roi = Polygon([
        ...     [8.7, 45.1], [9.8, 44.9], [10.3, 45.3], [10.1, 45.9],
        ...     [9.5, 46.2], [8.9, 46.0], [8.5, 45.6], [8.7, 45.1]
        ... ])
        >>> 
        >>> result = analyze_temporal_patterns(
        ...     milan_roi, "2025-01-01/2025-03-31",
        ...     spatial_resolution=100, clip_to_roi=True, cloud_cover_max=0.2,
        ...     optimization_level="fast"
        ... )
        >>> print(f"Mean coverage days: {result['temporal_result'].temporal_stats['mean_coverage_days']:.1f}")
    """
    from .workflows import parse_time_period, parse_roi_input, create_output_directory
    from .query import PlanetScopeQuery
    
    logger.info("Starting high-level temporal pattern analysis")
    
    try:
        # Parse inputs
        roi_poly = parse_roi_input(roi_polygon)
        if isinstance(time_period, str) and "/" in time_period:
            start_date, end_date = time_period.split("/")
            start_date, end_date = start_date.strip(), end_date.strip()
        elif isinstance(time_period, tuple):
            start_date, end_date = str(time_period[0]), str(time_period[1])
        else:
            start_date, end_date = parse_time_period(time_period)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"temporal_analysis_{timestamp}")
        os.makedirs(output_path, exist_ok=True)
        
        results = {
            'scenes_found': 0,
            'temporal_result': None,
            'visualizations': {},
            'exports': {},
            'summary': {},
            'output_directory': output_path,
        }
        
        # Scene Discovery
        logger.info("Discovering scenes via Planet API")
        query = PlanetScopeQuery()
        scenes_result = query.search_scenes(
            geometry=roi_poly,
            start_date=start_date,
            end_date=end_date,
            cloud_cover_max=cloud_cover_max
        )
        
        scenes_found = len(scenes_result['features'])
        results['scenes_found'] = scenes_found
        logger.info(f"Found {scenes_found} scenes for temporal analysis")
        
        if scenes_found == 0:
            logger.warning("No scenes found for temporal analysis")
            return results
        
        # Configure temporal analysis
        config = TemporalConfig(
            spatial_resolution=spatial_resolution,
            temporal_resolution=TemporalResolution.DAILY,
            metrics=metrics,  # Use provided metrics or defaults
            coordinate_system_fixes=True,
            force_single_chunk=kwargs.get('force_single_chunk', False),
            min_scenes_per_cell=kwargs.get('min_scenes_per_cell', 2),
            optimization_method=optimization_level,  # FIXED: Use the provided optimization level
        )
        
        # Perform temporal analysis
        logger.info("Calculating temporal patterns")
        analyzer = TemporalAnalyzer(config)
        
        temporal_result = analyzer.analyze_temporal_patterns(
            scene_footprints=scenes_result['features'],
            roi_geometry=roi_poly,
            start_date=start_date,
            end_date=end_date,
            clip_to_roi=clip_to_roi,
            **kwargs
        )
        
        results['temporal_result'] = temporal_result
        logger.info(f"Temporal analysis completed in {temporal_result.computation_time:.2f}s")
        
        # Export GeoTIFFs
        if export_geotiffs:
            logger.info("Exporting temporal analysis GeoTIFFs")
            exported_files = analyzer.export_temporal_geotiffs(
                temporal_result,
                output_path,
                roi_polygon=roi_poly,
                clip_to_roi=clip_to_roi
            )
            results['exports'] = exported_files
        
        # Create visualizations (integrate with visualization module)
        if create_visualizations:
            logger.info("Generating temporal analysis visualizations")
            vis_files = create_temporal_visualizations(
                temporal_result, roi_poly, output_path, clip_to_roi, show_plots
            )
            results['visualizations'] = vis_files
        
        # Create summary
        results['summary'] = {
            'roi_area_km2': roi_poly.area * (111.0**2),  # Rough conversion
            'analysis_mode': 'ROI-clipped' if clip_to_roi else 'full grid',
            'date_range': f"{start_date} to {end_date}",
            'scenes_found': scenes_found,
            'spatial_resolution_m': spatial_resolution,
            'computation_time_s': temporal_result.computation_time,
            'mean_coverage_days': temporal_result.temporal_stats.get('mean_coverage_days', 0),
            'unique_dates_in_data': temporal_result.temporal_stats['scene_summary']['unique_dates'],
            'coordinate_system_corrected': True,
            'optimization_method': temporal_result.grid_info.get('optimization_method', optimization_level),
            'output_directory': output_path,
        }
        
        logger.info("Temporal pattern analysis completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Temporal analysis workflow failed: {e}")
        raise PlanetScopeError(f"Temporal pattern analysis failed: {e}")


def create_temporal_visualizations(
    temporal_result: TemporalResult,
    roi_polygon: Polygon,
    output_dir: str,
    clip_to_roi: bool = True,
    show_plots: bool = True
) -> Dict[str, str]:
    """Create visualizations for temporal analysis results."""
    try:
        from .visualization import DensityVisualizer
        
        vis_files = {}
        visualizer = DensityVisualizer()
        
        # 1. Create individual plots for each temporal metric
        for metric, array in temporal_result.metric_arrays.items():
            try:
                # Create a mock density result for visualization compatibility
                mock_result = type('MockResult', (), {
                    'density_array': array,
                    'transform': temporal_result.transform,
                    'bounds': temporal_result.bounds,
                    'stats': temporal_result.temporal_stats.get(f'{metric.value}_stats', {}),
                    'grid_info': temporal_result.grid_info,
                    'no_data_value': temporal_result.no_data_value
                })()
                
                # Generate visualization
                plot_path = os.path.join(output_dir, f"temporal_{metric.value}_map.png")
                
                fig = visualizer.plot_density_map(
                    mock_result,
                    roi_polygon=roi_polygon,
                    title=f"Temporal Analysis: {metric.value.replace('_', ' ').title()}",
                    colormap=_get_metric_colormap(metric),
                    save_path=plot_path,
                    clip_to_roi=clip_to_roi,
                    show_plot=show_plots
                )
                
                vis_files[f"{metric.value}_map"] = plot_path
                logger.info(f"Created visualization for {metric.value}")
                
            except Exception as e:
                logger.warning(f"Failed to create visualization for {metric.value}: {e}")
                continue
        
        # 2. Always create comprehensive summary plot (like spatial density)
        try:
            summary_path = os.path.join(output_dir, "temporal_analysis_summary.png")
            create_temporal_summary_plot(
                temporal_result, roi_polygon, summary_path, clip_to_roi, show_plots
            )
            vis_files['summary'] = summary_path
            logger.info("Created comprehensive temporal analysis summary plot")
        except Exception as e:
            logger.warning(f"Failed to create summary plot: {e}")
        
        return vis_files
        
    except Exception as e:
        logger.error(f"Temporal visualization creation failed: {e}")
        return {}


def _get_metric_colormap(metric: TemporalMetric) -> str:
    """Get appropriate colormap for temporal metric."""
    if metric in [TemporalMetric.COVERAGE_DAYS, TemporalMetric.TEMPORAL_DENSITY]:
        return "Greens"  # More is better
    elif metric in [TemporalMetric.MEAN_INTERVAL, TemporalMetric.MEDIAN_INTERVAL]:
        return "turbo"  # Less is better (reversed)
    else:
        return "turbo"  # Default


def create_temporal_summary_plot(
    temporal_result: TemporalResult,
    roi_polygon: Polygon,
    save_path: str,
    clip_to_roi: bool = True,
    show_plots: bool = True
):
    """Create comprehensive temporal analysis summary plot with 4 panels like spatial density."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Create figure with 2x2 layout - increased size for better text spacing
    fig = plt.figure(figsize=(18, 14))
    
    # Get primary metrics for display
    coverage_days_array = temporal_result.metric_arrays.get(TemporalMetric.COVERAGE_DAYS)
    mean_interval_array = temporal_result.metric_arrays.get(TemporalMetric.MEAN_INTERVAL)
    
    # Apply ROI clipping if requested
    if clip_to_roi and roi_polygon is not None:
        try:
            from .visualization import DensityVisualizer
            visualizer = DensityVisualizer()
            if coverage_days_array is not None:
                coverage_days_display = visualizer.clip_density_to_roi(
                    coverage_days_array, temporal_result.transform, roi_polygon, temporal_result.no_data_value
                )
            if mean_interval_array is not None:
                mean_interval_display = visualizer.clip_density_to_roi(
                    mean_interval_array, temporal_result.transform, roi_polygon, temporal_result.no_data_value
                )
        except:
            coverage_days_display = coverage_days_array
            mean_interval_display = mean_interval_array
    else:
        coverage_days_display = coverage_days_array
        mean_interval_display = mean_interval_array
    
    # Define subplot layout with proper spacing
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25, 
                         left=0.08, right=0.95, top=0.88, bottom=0.08)
    
    # 1. Coverage Days Map (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    if coverage_days_display is not None:
        # Prepare for display (flip for correct orientation)
        plot_array = np.flipud(np.ma.masked_equal(coverage_days_display, temporal_result.no_data_value))
        
        bounds = temporal_result.bounds
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        
        im1 = ax1.imshow(
            plot_array,
            extent=extent,
            cmap="Greens",
            origin="lower",
            interpolation="nearest",
        )
        
        # Add colorbar with better positioning
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7, pad=0.02)
        cbar1.set_label("Coverage Days", rotation=270, labelpad=20)
        
        # Add ROI boundary
        if roi_polygon is not None:
            x, y = roi_polygon.exterior.xy
            ax1.plot(x, y, 'r-', linewidth=1.5, alpha=0.8)
    
    ax1.set_xlabel("Longitude", fontsize=10)
    ax1.set_ylabel("Latitude", fontsize=10)
    ax1.set_title("Coverage Days (ROI Clipped)" if clip_to_roi else "Coverage Days", 
                  fontsize=11, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=9)
    
    # 2. Mean Interval Map (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    if mean_interval_display is not None:
        # Prepare for display (flip for correct orientation)
        plot_array = np.flipud(np.ma.masked_equal(mean_interval_display, temporal_result.no_data_value))
        
        bounds = temporal_result.bounds
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        
        im2 = ax2.imshow(
            plot_array,
            extent=extent,
            cmap="turbo",  # Changed from "Reds_r" to "viridis"
            origin="lower",
            interpolation="nearest",
        )
                
        # Add colorbar with better positioning
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.7, pad=0.02)
        cbar2.set_label("Mean Interval (Days)", rotation=270, labelpad=20)
        
        # Add ROI boundary
        if roi_polygon is not None:
            x, y = roi_polygon.exterior.xy
            ax2.plot(x, y, 'r-', linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel("Longitude", fontsize=10)
    ax2.set_ylabel("Latitude", fontsize=10) 
    ax2.set_title("Mean Interval (ROI Clipped)" if clip_to_roi else "Mean Interval",
                  fontsize=11, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=9)
    
    # 3. Histogram of Mean Intervals (Bottom Left)
    ax3 = fig.add_subplot(gs[1, 0])
    if mean_interval_display is not None:
        # Get valid data for histogram
        valid_data = mean_interval_display[mean_interval_display != temporal_result.no_data_value]
        
        if len(valid_data) > 0:
            # Calculate histogram
            bins = np.linspace(np.min(valid_data), np.max(valid_data), 25)  # Reduced bins for cleaner look
            counts, bin_edges = np.histogram(valid_data, bins=bins)
            
            # Plot histogram
            ax3.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), 
                   color='skyblue', alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add statistics lines
            mean_val = np.mean(valid_data)
            median_val = np.median(valid_data)
            
            ax3.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
            ax3.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
            
            # Add statistics table on histogram - positioned to avoid legend overlap
            stats_text = (f"Count: {len(valid_data):,}\n"
                         f"Range: [{np.min(valid_data):.1f}, {np.max(valid_data):.1f}]\n"
                         f"Mean: {mean_val:.1f}\n"
                         f"Std: {np.std(valid_data):.1f}")
            
            ax3.text(0.98, 0.75, stats_text, transform=ax3.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))
            
            # Better legend positioning to avoid overlap
            ax3.legend(loc='upper center', fontsize=9, bbox_to_anchor=(0.5, 0.98))
    
    ax3.set_xlabel("Mean Interval (Days)", fontsize=10)
    ax3.set_ylabel("Frequency (Number of Pixels)", fontsize=10)
    title_text = f"Interval Distribution\n({len(valid_data):,} valid pixels)"
    if clip_to_roi:
        title_text += " - ROI Clipped"
    ax3.set_title(title_text, fontsize=11, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=9)
    
    # 4. Summary Statistics Table (Bottom Right) - UPDATED to match spatial density styling
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Enhanced statistics summary with time period and cloud cover (like spatial density)
    if temporal_result.temporal_stats and "error" not in temporal_result.temporal_stats:
        stats = temporal_result.temporal_stats

        # Calculate meaningful statistics from the display data
        if clip_to_roi and roi_polygon is not None:
            # Use first available metric array to determine data coverage
            sample_array = list(temporal_result.metric_arrays.values())[0]
            analysis_data = sample_array[sample_array != temporal_result.no_data_value]
            stats_title = "ROI Analysis Results"
            coverage_note = f"Analysis of {len(analysis_data):,} pixels within ROI"
        else:
            # Use first available metric array to determine data coverage
            sample_array = list(temporal_result.metric_arrays.values())[0]
            analysis_data = sample_array[sample_array != temporal_result.no_data_value]
            stats_title = "Full Grid Analysis Results" 
            coverage_note = f"Analysis of {len(analysis_data):,} total pixels"

        if len(analysis_data) > 0:
            # Build enhanced statistics data with time period (like spatial density)
            stats_data = [
                ["Pixels Analyzed", f"{len(analysis_data):,}"],
                ["Grid Resolution", f"{temporal_result.config.spatial_resolution}m"],
                ["Grid Dimensions", f"{temporal_result.grid_info['width']}x{temporal_result.grid_info['height']}"],
                ["Total Scenes", f"{temporal_result.temporal_stats['scene_summary']['total_scenes']:,}"],
                ["Unique Dates", f"{temporal_result.temporal_stats['scene_summary']['unique_dates']}"],
                ["", ""],  # Separator
            ]
            
            # Add time period information
            time_period_display = f"{temporal_result.date_range[0]} to {temporal_result.date_range[1]}"
            stats_data.append(["Time Period", time_period_display])
            
            # Add analysis days
            stats_data.append(["Analysis Days", f"{temporal_result.temporal_stats['analysis_period']['total_days']} days"])
            
            stats_data.append(["", ""])  # Separator
            
            # Add coverage statistics
            if TemporalMetric.COVERAGE_DAYS in temporal_result.metric_arrays:
                coverage_stats = temporal_result.temporal_stats.get('coverage_days_stats', {})
                if 'mean' in coverage_stats:
                    stats_data.append(["Mean Coverage Days", f"{coverage_stats['mean']:.1f}"])
                    stats_data.append(["Max Coverage Days", f"{coverage_stats['max']:.0f}"])
            
            # Add interval statistics
            if TemporalMetric.MEAN_INTERVAL in temporal_result.metric_arrays:
                interval_stats = temporal_result.temporal_stats.get('mean_interval_stats', {})
                if 'mean' in interval_stats:
                    stats_data.append(["Mean Interval", f"{interval_stats['mean']:.1f} days"])
                    stats_data.append(["Median Interval", f"{interval_stats['median']:.1f} days"])
            
            # Add temporal density
            if TemporalMetric.TEMPORAL_DENSITY in temporal_result.metric_arrays:
                density_stats = temporal_result.temporal_stats.get('temporal_density_stats', {})
                if 'mean' in density_stats:
                    stats_data.append(["Temporal Density", f"{density_stats['mean']:.3f} scenes/day"])
            
            stats_data.append(["", ""])  # Separator
            
            # Add technical information
            stats_data.extend([
                ["Method", f"{temporal_result.grid_info.get('optimization_method', 'unknown')}"],
                ["Computation Time", f"{temporal_result.computation_time:.2f}s"],
                ["Coordinate Fixes", "Applied" if temporal_result.coordinate_system_corrected else "Disabled"],
            ])
        else:
            stats_data = [["No valid data", "N/A"]]

        # Set clear title
        ax4.text(0.5, 0.95, stats_title, 
                transform=ax4.transAxes, 
                ha='center', va='top',
                fontsize=12, weight='bold')
        
        # Add coverage note
        ax4.text(0.5, 0.88, coverage_note, 
                transform=ax4.transAxes, 
                ha='center', va='top',
                fontsize=9, style='italic')

        # Remove axis ticks and labels
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis("off")

        # Create clean table with enhanced statistics (same styling as spatial density)
        table_data = []
        colors = []
        for label, value in stats_data:
            if label == "":  # Separator row
                continue
            table_data.append([label, value])
            
            # Color coding for different types of information (same as spatial density)
            if label in ["Pixels Analyzed", "Grid Resolution", "Grid Dimensions", "Total Scenes", "Unique Dates", "Mean Coverage Days", "Max Coverage Days", "Mean Interval", "Median Interval", "Temporal Density"]:
                colors.append(["lightblue", "white"])  # Main analysis results
            elif label in ["Time Period", "Analysis Days"]:
                colors.append(["lightgreen", "white"])  # Query parameters
            else:
                colors.append(["lightgray", "white"])  # Technical info

        if table_data:
            table = ax4.table(
                cellText=table_data,
                cellColours=colors,
                cellLoc="left",
                loc="center",
                colWidths=[0.5, 0.4],
                bbox=[0.0, 0.0, 1.0, 0.82]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.4)

    else:
        ax4.text(
            0.5, 0.5,
            "No statistics available",
            transform=ax4.transAxes,
            ha="center", va="center",
            fontsize=12
        )
        ax4.set_title("Analysis Results", pad=20)
        ax4.axis("off")


# Example usage and testing
if __name__ == "__main__":
    def run_temporal_test():
        """Run the temporal analysis test."""
        try:
            print("Testing PlanetScope-py Temporal Analysis Module")
            print("=" * 50)
            
            from shapely.geometry import box
            import numpy as np
            
            # Create test data
            milan_roi = box(9.1, 45.45, 9.25, 45.5)
            
            # Mock scene data with temporal information
            mock_scenes = []
            base_date = datetime(2025, 1, 1)
            
            for i in range(30):  # 30 scenes over 3 months
                # Random date within 3 months
                days_offset = np.random.randint(0, 90)
                scene_date = base_date + timedelta(days=days_offset)
                
                # Random location within ROI
                center_x = np.random.uniform(9.1, 9.25)
                center_y = np.random.uniform(45.45, 45.5)
                size = 0.01  # Small scene footprint
                
                footprint = box(
                    center_x - size/2, center_y - size/2,
                    center_x + size/2, center_y + size/2
                )
                
                mock_scenes.append({
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [list(footprint.exterior.coords)]
                    },
                    'properties': {
                        'id': f'mock_scene_{i}',
                        'acquired': scene_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        'cloud_cover': np.random.uniform(0, 0.3),
                    }
                })
            
            print(f"Testing with {len(mock_scenes)} mock scenes")
            print(f"Date range: 2025-01-01 to 2025-03-31")
            print(f"ROI area: {milan_roi.area * 111**2:.1f} km²")
            
            # Test temporal analysis configuration
            config = TemporalConfig(
                spatial_resolution=100.0,  # 100m resolution for testing
                temporal_resolution=TemporalResolution.DAILY,
                metrics=[
                    TemporalMetric.COVERAGE_DAYS,
                    TemporalMetric.MEAN_INTERVAL,
                    TemporalMetric.TEMPORAL_DENSITY
                ],
                coordinate_system_fixes=True,
                force_single_chunk=True,
                min_scenes_per_cell=2,
                optimization_method="fast"
            )
            
            analyzer = TemporalAnalyzer(config)
            
            # Run temporal analysis
            result = analyzer.analyze_temporal_patterns(
                scene_footprints=mock_scenes,
                roi_geometry=milan_roi,
                start_date="2025-01-01",
                end_date="2025-03-31",
                clip_to_roi=True
            )
            
            print("\nTemporal Analysis Results:")
            print(f"Computation time: {result.computation_time:.2f}s")
            print(f"Grid size: {result.grid_info['width']}x{result.grid_info['height']}")
            print(f"Coordinate system corrected: {result.coordinate_system_corrected}")
            print(f"Optimization method: {result.grid_info['optimization_method']}")
            print(f"Metrics calculated: {[m.value for m in result.config.metrics]}")
            
            # Display statistics for each metric
            for metric in result.config.metrics:
                if metric in result.metric_arrays:
                    array = result.metric_arrays[metric]
                    valid_data = array[array != result.no_data_value]
                    if len(valid_data) > 0:
                        print(f"\n{metric.value.replace('_', ' ').title()}:")
                        print(f"  Valid cells: {len(valid_data):,}")
                        print(f"  Range: [{np.min(valid_data):.1f}, {np.max(valid_data):.1f}]")
                        print(f"  Mean: {np.mean(valid_data):.2f}")
            
            print(f"\nTemporal Statistics Summary:")
            print(f"Mean coverage days: {result.temporal_stats.get('mean_coverage_days', 0):.1f}")
            print(f"Total scenes in analysis: {result.temporal_stats['scene_summary']['total_scenes']}")
            print(f"Unique dates: {result.temporal_stats['scene_summary']['unique_dates']}")
            
            # Test export functionality
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"\nTesting GeoTIFF export to: {temp_dir}")
                
                exported_files = analyzer.export_temporal_geotiffs(
                    result, temp_dir, roi_polygon=milan_roi, clip_to_roi=True
                )
                
                print(f"Exported files: {list(exported_files.keys())}")
            
            print("\n" + "=" * 50)
            print(" TEMPORAL ANALYSIS MODULE TEST PASSED!")
            print("Complete temporal analysis capabilities are ready!")
            print("=" * 50)
            
            print("\nKey Features Implemented:")
            print("✓ Grid-based temporal pattern analysis (same as spatial density)")
            print("✓ Multiple temporal metrics (coverage days, intervals, density)")
            print("✓ Daily temporal resolution")
            print("✓ Coordinate system fixes integration")
            print("✓ ROI clipping support (same as spatial density)")
            print("✓ Professional GeoTIFF export with QML styling")
            print("✓ Integration with visualization module")
            print("✓ High-level workflow functions")
            print("✓ Comprehensive statistics and metadata")
            print("✓ FAST and ACCURATE optimization methods")
            print("✓ Complete function implementations")
            
            print("\nUsage Examples:")
            print("# Complete temporal analysis")
            print("result = analyze_temporal_patterns(milan_roi, '2025-01-01/2025-03-31')")
            print()
            print("# Custom metrics and resolution")
            print("from planetscope_py import TemporalAnalyzer, TemporalConfig, TemporalMetric")
            print("config = TemporalConfig(spatial_resolution=50, metrics=[TemporalMetric.COVERAGE_DAYS])")
            print("analyzer = TemporalAnalyzer(config)")
            print("result = analyzer.analyze_temporal_patterns(scenes, roi, start, end)")
            print()
            print("# Use FAST optimization for large areas")
            print("result = analyze_temporal_patterns(roi, period, optimization_level='fast')")
            
            return True
            
        except Exception as e:
            print(f" Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Run the test
    test_success = run_temporal_test()
    
    if test_success:
        print("\n Temporal Analysis Module is ready for use!")
    else:
        print("\n Please check the errors above and fix before using.")