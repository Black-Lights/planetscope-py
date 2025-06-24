#!/usr/bin/env python3
"""
PlanetScope-py Enhanced GeoPackage One-Liner Functions
=====================================================
Simple one-line functions for GeoPackage creation following the pattern
from density analysis and visualization modules.

Author: Ammar & Umayr  
Version: 4.0.0 (Enhanced) - FIXED
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
from collections import Counter  # FIXED: Import at top level
from shapely.geometry import Polygon, shape

from .query import PlanetScopeQuery
from .geopackage_manager import GeoPackageManager, GeoPackageConfig
from .exceptions import PlanetScopeError, ValidationError
from .utils import validate_geometry, calculate_area_km2

logger = logging.getLogger(__name__)


def quick_geopackage_export(
    roi: Union[Polygon, list, dict],
    time_period: str = "last_month", 
    output_path: Optional[str] = None,
    clip_to_roi: bool = True,
    schema: str = "standard",
    cloud_cover_max: float = 0.3,
    item_types: Optional[List[str]] = None,
    sun_elevation_min: Optional[float] = None,
    ground_control: Optional[bool] = None,
    quality_category: Optional[str] = None,
    **kwargs
) -> str:
    """
    ONE-LINE function to create GeoPackage from ROI and time period.
    
    ENHANCED with ALL Planet API search parameters and comprehensive metadata.
    
    Usage:
        # Basic usage
        gpkg_path = quick_geopackage_export(milan_polygon, "2025-01-01/2025-01-31")
        
        # With comprehensive Planet API parameters
        gpkg_path = quick_geopackage_export(
            roi, "last_month", "output.gpkg", 
            clip_to_roi=True, cloud_cover_max=0.2, sun_elevation_min=30,
            ground_control=True, quality_category="standard", 
            item_types=["PSScene"], schema="comprehensive"
        )
    
    Args:
        roi: Region of interest (Polygon, coordinate list, or GeoJSON dict)
        time_period: Time period ("last_month", "last_3_months", or "YYYY-MM-DD/YYYY-MM-DD")
        output_path: Output path for GeoPackage (auto-generated if None)
        clip_to_roi: Whether to clip scene footprints to ROI shape
        schema: Attribute schema ("minimal", "standard", "comprehensive")
        cloud_cover_max: Maximum cloud cover threshold (0.0-1.0)
        item_types: Planet item types to search (default: ["PSScene"])
        sun_elevation_min: Minimum sun elevation in degrees
        ground_control: Require ground control points (True/False/None)
        quality_category: Required quality category ("test", "standard", etc.)
        **kwargs: Additional Planet API parameters (see documentation)
    
    Returns:
        str: Path to created GeoPackage file
    """
    try:
        # Parse ROI
        if not isinstance(roi, Polygon):
            if isinstance(roi, list):
                roi_polygon = Polygon(roi)
            elif isinstance(roi, dict) and "coordinates" in roi:
                roi_polygon = Polygon(roi["coordinates"][0])
            else:
                raise ValidationError(f"Unsupported ROI format: {type(roi)}")
        else:
            roi_polygon = roi
        
        # Parse time period
        if time_period == "last_month":
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
        elif time_period == "last_3_months":
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
        elif "/" in time_period:
            start_str, end_str = time_period.split("/")
        else:
            raise ValidationError(f"Invalid time period: {time_period}")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            area_km2 = calculate_area_km2(roi_polygon)
            output_path = f"planetscope_scenes_{area_km2:.0f}km2_{timestamp}.gpkg"
        
        # Search for scenes with ALL Planet API parameters
        logger.info(f"Searching scenes for ROI ({calculate_area_km2(roi_polygon):.0f} km²)")
        query = PlanetScopeQuery()
        
        # Build comprehensive search parameters
        search_params = {
            "geometry": roi_polygon,
            "start_date": start_str,
            "end_date": end_str,
            "cloud_cover_max": cloud_cover_max,
        }
        
        # Add Planet API item types
        if item_types is not None:
            search_params["item_types"] = item_types
        else:
            search_params["item_types"] = ["PSScene"]
        
        # Add comprehensive Planet API search filters
        if sun_elevation_min is not None:
            search_params["sun_elevation_min"] = sun_elevation_min
        
        if ground_control is not None:
            search_params["ground_control"] = ground_control
            
        if quality_category is not None:
            search_params["quality_category"] = quality_category
        
        # Add all additional Planet API parameters from kwargs
        planet_api_params = [
            "visible_percent_min", "clear_percent_min", "usable_data_min",
            "shadow_percent_max", "snow_ice_percent_max", "heavy_haze_percent_max",
            "light_haze_percent_max", "anomalous_pixels_max", "view_angle_max",
            "off_nadir_max", "gsd_min", "gsd_max", "satellite_ids", "instrument",
            "provider", "processing_level", "clear_confidence_percent_min",
            "visible_confidence_percent_min", "pixel_resolution_min", "pixel_resolution_max"
        ]
        
        for param in planet_api_params:
            if param in kwargs:
                search_params[param] = kwargs[param]
        
        results = query.search_scenes(**search_params)
        
        scenes = results.get('features', [])
        if not scenes:
            raise PlanetScopeError(f"No scenes found for the specified criteria")
        
        # Create GeoPackage with smart defaults
        config = GeoPackageConfig(
            clip_to_roi=clip_to_roi,
            attribute_schema=schema,
            include_imagery=False,  # Default to footprints only for speed
        )
        
        manager = GeoPackageManager(config=config)
        final_path = manager.create_scene_geopackage(
            scenes=scenes,
            output_path=output_path,
            roi=roi_polygon if clip_to_roi else None
        )
        
        logger.info(f"GeoPackage created: {final_path} ({len(scenes)} scenes)")
        return final_path
        
    except Exception as e:
        logger.error(f"Quick GeoPackage export failed: {e}")
        raise PlanetScopeError(f"GeoPackage export failed: {e}")


def create_milan_geopackage(
    time_period: str = "2025-01-01/2025-01-31",
    output_path: Optional[str] = None,
    size: str = "large",
    cloud_cover_max: float = 0.3,
    sun_elevation_min: Optional[float] = None,
    ground_control: Optional[bool] = None,
    quality_category: Optional[str] = None,
    **kwargs
) -> str:
    """
    ONE-LINE function to create Milan area GeoPackage (predefined polygon).
    
    ENHANCED with comprehensive Planet API parameters for Milan area analysis.
    
    Usage:
        # Basic Milan analysis
        milan_gpkg = create_milan_geopackage("2025-01-01/2025-01-31")
        
        # High-quality data only
        milan_gpkg = create_milan_geopackage(
            "last_month", size="large", cloud_cover_max=0.1, 
            sun_elevation_min=30, ground_control=True, quality_category="standard"
        )
    
    Args:
        time_period: Time period for scene search
        output_path: Output path (auto-generated if None)
        size: Milan area size ("small", "medium", "large", "city_center")
        cloud_cover_max: Maximum cloud cover threshold (0.0-1.0)
        sun_elevation_min: Minimum sun elevation in degrees
        ground_control: Require ground control points (True/False/None)
        quality_category: Required quality category ("test", "standard", etc.)
        **kwargs: Additional Planet API parameters (same as quick_geopackage_export)
    
    Returns:
        str: Path to created GeoPackage file
    """
    # Predefined Milan polygons of different sizes
    milan_polygons = {
        "city_center": Polygon([
            [9.15, 45.45], [9.25, 45.44], [9.26, 45.50], [9.16, 45.51], [9.15, 45.45]
        ]),  # ~100 km²
        
        "small": Polygon([
            [9.0, 45.4], [9.3, 45.35], [9.35, 45.55], [9.05, 45.6], [9.0, 45.4]  
        ]),  # ~500 km²
        
        "medium": Polygon([
            [8.9, 45.3], [9.4, 45.25], [9.45, 45.65], [8.95, 45.7], [8.9, 45.3]
        ]),  # ~1200 km²
        
        "large": Polygon([
            [8.7, 45.1], [9.8, 44.9], [10.3, 45.3], [10.1, 45.9], 
            [9.5, 46.2], [8.9, 46.0], [8.5, 45.6], [8.7, 45.1]
        ])  # ~2000+ km²
    }
    
    if size not in milan_polygons:
        raise ValidationError(f"Invalid size '{size}'. Options: {list(milan_polygons.keys())}")
    
    milan_roi = milan_polygons[size]
    
    # Generate Milan-specific output path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        area_km2 = calculate_area_km2(milan_roi)
        output_path = f"milan_{size}_{area_km2:.0f}km2_{timestamp}.gpkg"
    
    return quick_geopackage_export(
        roi=milan_roi,
        time_period=time_period,
        output_path=output_path,
        cloud_cover_max=cloud_cover_max,
        sun_elevation_min=sun_elevation_min,
        ground_control=ground_control,
        quality_category=quality_category,
        **kwargs
    )


def create_clipped_geopackage(
    roi: Union[Polygon, list, dict],
    time_period: str = "last_month",
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    ONE-LINE function to create GeoPackage with scenes clipped to ROI shape.
    
    Usage:
        clipped_gpkg = create_clipped_geopackage(polygon, "2025-01-01/2025-01-31")
        clipped_gpkg = create_clipped_geopackage(roi, "last_month", "clipped.gpkg")
    
    This is a shortcut for quick_geopackage_export with clip_to_roi=True.
    """
    return quick_geopackage_export(
        roi=roi,
        time_period=time_period,
        output_path=output_path,
        clip_to_roi=True,  # Force clipping
        **kwargs
    )


def create_full_grid_geopackage(
    roi: Union[Polygon, list, dict],
    time_period: str = "last_month", 
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    ONE-LINE function to create GeoPackage with full scene footprints (no clipping).
    
    Usage:
        full_gpkg = create_full_grid_geopackage(polygon, "2025-01-01/2025-01-31")
        full_gpkg = create_full_grid_geopackage(roi, "last_month", "full.gpkg")
    
    This is a shortcut for quick_geopackage_export with clip_to_roi=False.
    """
    return quick_geopackage_export(
        roi=roi,
        time_period=time_period,
        output_path=output_path,
        clip_to_roi=False,  # No clipping - full footprints
        **kwargs
    )


def export_scenes_to_geopackage(
    scenes: List[Dict],
    output_path: str,
    roi: Optional[Polygon] = None,
    clip_to_roi: bool = True,
    schema: str = "standard"
) -> str:
    """
    ONE-LINE function to export existing scene list to GeoPackage.
    
    Usage:
        gpkg_path = export_scenes_to_geopackage(scenes, "output.gpkg")
        gpkg_path = export_scenes_to_geopackage(scenes, "clipped.gpkg", roi, clip_to_roi=True)
    
    Args:
        scenes: List of Planet scene features (from search results)
        output_path: Path for output GeoPackage
        roi: Optional ROI polygon for clipping
        clip_to_roi: Whether to clip scenes to ROI
        schema: Attribute schema to use
    
    Returns:
        str: Path to created GeoPackage
    """
    try:
        config = GeoPackageConfig(
            clip_to_roi=clip_to_roi,
            attribute_schema=schema,
            include_imagery=False
        )
        
        manager = GeoPackageManager(config=config)
        return manager.create_scene_geopackage(
            scenes=scenes,
            output_path=output_path,
            roi=roi if clip_to_roi else None
        )
        
    except Exception as e:
        logger.error(f"Scene export to GeoPackage failed: {e}")
        raise PlanetScopeError(f"Export failed: {e}")


def quick_scene_search_and_export(
    roi: Union[Polygon, list, dict],
    start_date: str,
    end_date: str,
    output_path: Optional[str] = None,
    cloud_cover_max: float = 0.3,
    sun_elevation_min: Optional[float] = None,
    ground_control: Optional[bool] = None,
    quality_category: Optional[str] = None,
    item_types: Optional[List[str]] = None,
    **search_params
) -> Dict[str, Any]:
    """
    ONE-LINE function for search + export with detailed results.
    
    ENHANCED with ALL Planet API search parameters and comprehensive statistics.
    FIXED: Returns consistent structure whether scenes are found or not.
    
    Usage:
        # Basic search and export
        result = quick_scene_search_and_export(roi, "2025-01-01", "2025-01-31")
        
        # High-quality scenes only with comprehensive filtering
        result = quick_scene_search_and_export(
            roi, "2025-01-01", "2025-01-31", "output.gpkg",
            cloud_cover_max=0.1, sun_elevation_min=30, ground_control=True,
            quality_category="standard", usable_data_min=0.9,
            shadow_percent_max=0.1, item_types=["PSScene"]
        )
    
    Args:
        roi: Region of interest (Polygon, coordinate list, or GeoJSON dict)
        start_date: Start date in "YYYY-MM-DD" format  
        end_date: End date in "YYYY-MM-DD" format
        output_path: Path for output GeoPackage (auto-generated if None)
        cloud_cover_max: Maximum cloud cover threshold (0.0-1.0)
        sun_elevation_min: Minimum sun elevation in degrees
        ground_control: Require ground control points (True/False/None)
        quality_category: Required quality category ("test", "standard", etc.)
        item_types: Planet item types to search (default: ["PSScene"])
        **search_params: Additional Planet API search parameters
    
    Returns:
        dict: CONSISTENT comprehensive results including scene count, area coverage, statistics
    """
    # Initialize default return structure
    search_options = {}
    roi_area_km2 = 0
    
    try:
        # Parse ROI
        if not isinstance(roi, Polygon):
            if isinstance(roi, list):
                roi_polygon = Polygon(roi)
            elif isinstance(roi, dict) and "coordinates" in roi:
                roi_polygon = Polygon(roi["coordinates"][0])
            else:
                roi_polygon = roi
        else:
            roi_polygon = roi
        
        # Calculate ROI area
        roi_area_km2 = calculate_area_km2(roi_polygon)
        
        # Search scenes with comprehensive Planet API parameters
        query = PlanetScopeQuery()
        
        # Build search parameters with all Planet API options
        search_options = {
            "geometry": roi_polygon,
            "start_date": start_date,
            "end_date": end_date,
            "cloud_cover_max": cloud_cover_max,
        }
        
        # Add Planet API item types
        if item_types is not None:
            search_options["item_types"] = item_types
        else:
            search_options["item_types"] = ["PSScene"]
        
        # Add comprehensive Planet API search filters
        if sun_elevation_min is not None:
            search_options["sun_elevation_min"] = sun_elevation_min
        
        if ground_control is not None:
            search_options["ground_control"] = ground_control
            
        if quality_category is not None:
            search_options["quality_category"] = quality_category
        
        # Add all additional Planet API parameters
        planet_search_params = [
            "visible_percent_min", "clear_percent_min", "usable_data_min",
            "shadow_percent_max", "snow_ice_percent_max", "heavy_haze_percent_max",
            "light_haze_percent_max", "anomalous_pixels_max", "view_angle_max",
            "off_nadir_max", "gsd_min", "gsd_max", "satellite_ids", "instrument",
            "provider", "processing_level", "clear_confidence_percent_min",
            "visible_confidence_percent_min", "pixel_resolution_min", "pixel_resolution_max"
        ]
        
        for param in planet_search_params:
            if param in search_params:
                search_options[param] = search_params[param]
        
        # Execute search
        results = query.search_scenes(**search_options)
        scenes = results.get('features', [])
        
        # FIXED: Always return consistent structure
        if scenes:
            # Scenes found - create GeoPackage and calculate statistics
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"scenes_{len(scenes)}_{timestamp}.gpkg"
            
            geopackage_path = export_scenes_to_geopackage(
                scenes, output_path, roi_polygon, 
                clip_to_roi=search_params.get('clip_to_roi', True)
            )
            
            # Calculate coverage statistics with Planet API metadata
            total_scene_area = 0
            cloud_covers = []
            sun_elevations = []
            usable_data_values = []
            quality_categories = []
            acquisition_dates = []
            satellite_ids = []
            
            for scene in scenes:
                try:
                    scene_geom = shape(scene['geometry'])
                    intersection = scene_geom.intersection(roi_polygon)
                    if not intersection.is_empty:
                        total_scene_area += calculate_area_km2(intersection)
                    
                    # Collect comprehensive Planet API metadata
                    props = scene.get('properties', {})
                    
                    # Quality metrics
                    if 'cloud_cover' in props and props['cloud_cover'] is not None:
                        cloud_covers.append(props['cloud_cover'])
                    if 'sun_elevation' in props and props['sun_elevation'] is not None:
                        sun_elevations.append(props['sun_elevation'])
                    if 'usable_data' in props and props['usable_data'] is not None:
                        usable_data_values.append(props['usable_data'])
                    if 'quality_category' in props:
                        quality_categories.append(props['quality_category'])
                    
                    # Temporal and satellite info
                    if 'acquired' in props:
                        acquisition_dates.append(props['acquired'])
                    if 'satellite_id' in props:
                        satellite_ids.append(props['satellite_id'])
                        
                except Exception as e:
                    logger.warning(f"Error processing scene: {e}")
                    continue
            
            # Calculate comprehensive statistics
            statistics = {}
            
            if cloud_covers:
                statistics['cloud_cover'] = {
                    'mean': sum(cloud_covers) / len(cloud_covers),
                    'min': min(cloud_covers),
                    'max': max(cloud_covers),
                    'count': len(cloud_covers)
                }
            
            if sun_elevations:
                statistics['sun_elevation'] = {
                    'mean': sum(sun_elevations) / len(sun_elevations),
                    'min': min(sun_elevations),
                    'max': max(sun_elevations),
                    'count': len(sun_elevations)
                }
            
            if usable_data_values:
                statistics['usable_data'] = {
                    'mean': sum(usable_data_values) / len(usable_data_values),
                    'min': min(usable_data_values),
                    'max': max(usable_data_values),
                    'count': len(usable_data_values)
                }
            
            if quality_categories:
                quality_counts = Counter(quality_categories)
                statistics['quality_distribution'] = dict(quality_counts)
            
            if satellite_ids:
                satellite_counts = Counter(satellite_ids)
                statistics['satellite_distribution'] = dict(satellite_counts)
                statistics['unique_satellites'] = len(set(satellite_ids))
            
            # SUCCESSFUL RESULT - with scenes
            return {
                'success': True,
                'scenes_found': len(scenes),
                'roi_area_km2': roi_area_km2,
                'total_coverage_km2': total_scene_area,
                'coverage_ratio': total_scene_area / roi_area_km2 if roi_area_km2 > 0 else 0,
                'statistics': statistics,
                'date_range': {
                    'start': min(acquisition_dates) if acquisition_dates else None,
                    'end': max(acquisition_dates) if acquisition_dates else None,
                    'unique_dates': len(set(acquisition_dates)) if acquisition_dates else 0
                },
                'geopackage_path': geopackage_path,
                'search_parameters': search_options,
                'message': f'Successfully found {len(scenes)} scenes'
            }
        else:
            # FIXED: No scenes found - return consistent structure with default values
            return {
                'success': False,
                'scenes_found': 0,
                'roi_area_km2': roi_area_km2,
                'total_coverage_km2': 0.0,  # FIXED: Always include
                'coverage_ratio': 0.0,     # FIXED: Always include
                'statistics': {},          # FIXED: Always include (empty)
                'date_range': {            # FIXED: Always include
                    'start': None,
                    'end': None,
                    'unique_dates': 0
                },
                'geopackage_path': None,   # FIXED: Always include (None when no scenes)
                'search_parameters': search_options,
                'message': 'No scenes found for the specified criteria. Try relaxing filters.'
            }
            
    except Exception as e:
        logger.error(f"Scene search and export failed: {e}")
        # FIXED: Error case - return consistent structure
        return {
            'success': False,
            'scenes_found': 0,
            'roi_area_km2': roi_area_km2,
            'total_coverage_km2': 0.0,  # FIXED: Always include
            'coverage_ratio': 0.0,     # FIXED: Always include  
            'statistics': {},          # FIXED: Always include (empty)
            'date_range': {            # FIXED: Always include
                'start': None,
                'end': None,
                'unique_dates': 0
            },
            'geopackage_path': None,   # FIXED: Always include (None on error)
            'search_parameters': search_options,
            'error': str(e),
            'message': f'Search failed: {str(e)}'
        }


def validate_geopackage_output(geopackage_path: str) -> Dict[str, Any]:
    """
    ONE-LINE function to validate created GeoPackage and get summary stats.
    
    Usage:
        validation = validate_geopackage_output("output.gpkg")
        print(f"Features: {validation['feature_count']}")
    
    Returns:
        dict: Validation results and statistics
    """
    try:
        import geopandas as gpd
        import os
        
        if not os.path.exists(geopackage_path):
            return {'valid': False, 'error': 'File does not exist'}
        
        # Read GeoPackage
        gdf = gpd.read_file(geopackage_path)
        
        # Calculate statistics
        validation = {
            'valid': True,
            'file_path': geopackage_path,
            'file_size_mb': os.path.getsize(geopackage_path) / (1024 * 1024),
            'feature_count': len(gdf),
            'attribute_count': len(gdf.columns),
            'crs': str(gdf.crs),
            'bounds': gdf.total_bounds.tolist(),
            'columns': gdf.columns.tolist(),
        }
        
        # Add field-specific statistics
        if 'aoi_km2' in gdf.columns:
            aoi_values = gdf['aoi_km2'].dropna()
            validation['aoi_statistics'] = {
                'total_aoi_km2': float(aoi_values.sum()),
                'mean_aoi_km2': float(aoi_values.mean()) if len(aoi_values) > 0 else 0,
                'scenes_with_aoi': int((aoi_values > 0).sum())
            }
        
        if 'cloud_cover' in gdf.columns:
            cloud_values = gdf['cloud_cover'].dropna()
            validation['cloud_statistics'] = {
                'mean_cloud_cover': float(cloud_values.mean()) if len(cloud_values) > 0 else 0,
                'min_cloud_cover': float(cloud_values.min()) if len(cloud_values) > 0 else 0,
                'max_cloud_cover': float(cloud_values.max()) if len(cloud_values) > 0 else 0
            }
        
        if 'acquired' in gdf.columns:
            dates = gdf['acquired'].dropna()
            if len(dates) > 0:
                validation['temporal_coverage'] = {
                    'start_date': str(dates.min()),
                    'end_date': str(dates.max()),
                    'unique_dates': len(dates.unique())
                }
        
        return validation
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'file_path': geopackage_path
        }


# Enhanced batch processing
def batch_geopackage_export(
    roi_list: List[Union[Polygon, list, dict]],
    time_period: str = "last_month",
    output_dir: str = "./batch_geopackages",
    **kwargs
) -> Dict[str, Any]:
    """
    ONE-LINE function to create GeoPackages for multiple ROIs.
    
    Usage:
        results = batch_geopackage_export([roi1, roi2, roi3], "2025-01-01/2025-01-31")
        results = batch_geopackage_export(roi_list, "last_month", "./output/", clip_to_roi=True)
    
    Returns:
        dict: Results for each ROI with paths and statistics
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for i, roi in enumerate(roi_list):
        try:
            output_path = os.path.join(output_dir, f"roi_{i+1:03d}.gpkg")
            
            geopackage_path = quick_geopackage_export(
                roi=roi,
                time_period=time_period,
                output_path=output_path,
                **kwargs
            )
            
            validation = validate_geopackage_output(geopackage_path)
            
            results[f"roi_{i+1}"] = {
                'success': True,
                'geopackage_path': geopackage_path,
                'validation': validation
            }
            
            logger.info(f"ROI {i+1} completed: {validation.get('feature_count', 0)} scenes")
            
        except Exception as e:
            logger.error(f"ROI {i+1} failed: {e}")
            results[f"roi_{i+1}"] = {
                'success': False,
                'error': str(e)
            }
    
    return results


# NEW: Integration with the main __init__.py exports
def get_geopackage_usage_examples():
    """Display usage examples for GeoPackage one-liner functions."""
    print("GeoPackage One-Liner Function Examples")
    print("=" * 40)
    
    print("\n1. Quick GeoPackage Export:")
    print("   from planetscope_py import quick_geopackage_export")
    print("   gpkg = quick_geopackage_export(milan_polygon, '2025-01-01/2025-01-31')")
    
    print("\n2. Milan Area Presets:")
    print("   from planetscope_py import create_milan_geopackage") 
    print("   milan_gpkg = create_milan_geopackage('last_month', size='large')")
    
    print("\n3. Clipped vs Full Grid:")
    print("   from planetscope_py import create_clipped_geopackage, create_full_grid_geopackage")
    print("   clipped = create_clipped_geopackage(roi, 'last_month')")
    print("   full = create_full_grid_geopackage(roi, 'last_month')")
    
    print("\n4. Search + Export with Stats:")
    print("   from planetscope_py import quick_scene_search_and_export")
    print("   result = quick_scene_search_and_export(roi, '2025-01-01', '2025-01-31')")
    print("   print(f'Found {result[\"scenes_found\"]} scenes, {result[\"coverage_ratio\"]:.1%} coverage')")
    
    print("\n5. Batch Processing:")
    print("   from planetscope_py import batch_geopackage_export")
    print("   results = batch_geopackage_export([roi1, roi2, roi3], 'last_month')")
    
    print("\n6. Validation:")
    print("   from planetscope_py import validate_geopackage_output")
    print("   stats = validate_geopackage_output('output.gpkg')")
    print("   print(f'Valid: {stats[\"valid\"]}, Features: {stats[\"feature_count\"]}')")


# Add these functions to __all__ exports in __init__.py
__all__ = [
    "quick_geopackage_export",
    "create_milan_geopackage", 
    "create_clipped_geopackage",
    "create_full_grid_geopackage",
    "export_scenes_to_geopackage",
    "quick_scene_search_and_export",
    "validate_geopackage_output",
    "batch_geopackage_export",
    "get_geopackage_usage_examples"
]


if __name__ == "__main__":
    print("PlanetScope-py Enhanced GeoPackage One-Liner Functions")
    print("=" * 55)
    get_geopackage_usage_examples()
    
    # Demo the Milan example
    print("\n" + "=" * 55)
    print("🚀 DEMO: Milan Large Polygon")
    
    try:
        # Create the same large polygon from your example
        milan_large = Polygon([
            [8.7, 45.1], [9.8, 44.9], [10.3, 45.3], [10.1, 45.9],
            [9.5, 46.2], [8.9, 46.0], [8.5, 45.6], [8.7, 45.1]
        ])
        
        print(f"Large Milan polygon: {calculate_area_km2(milan_large):.0f} km²")
        
        # This would be your one-liner equivalent
        print("\nOne-liner equivalent of your code:")
        print("gpkg = quick_geopackage_export(milan_large, '2025-01-05/2025-01-25', clip_to_roi=True)")
        
        # Or even simpler with preset
        print("# Or with Milan preset:")
        print("gpkg = create_milan_geopackage('2025-01-05/2025-01-25', size='large')")
        
    except Exception as e:
        print(f"Demo setup failed: {e}")