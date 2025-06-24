#!/usr/bin/env python3
"""
PlanetScope-py Workflows Module - UPDATED WITH FIXES
High-level workflow orchestration with individual plot access functions.

NEW FEATURES:
- Individual plot access functions (1-line usage)
- Increased scene footprint limits (150+ default)
- Fixed coordinate system display issues
- GeoTIFF-only export functions

Author: Ammar & Umayr
Version: 4.0.0 (Updated)
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Union, Dict, Any, Optional, List, Tuple
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, box
import rasterio
from rasterio.transform import Affine
from rasterio.features import rasterize

from .exceptions import PlanetScopeError, ValidationError
from .query import PlanetScopeQuery
from .density_engine import SpatialDensityEngine, DensityConfig, DensityMethod, DensityResult
from .visualization import DensityVisualizer, plot_density_only, plot_footprints_only, export_geotiff_only
from .utils import validate_geometry, calculate_area_km2

logger = logging.getLogger(__name__)


class WorkflowConfig:
    """Default configuration for workflow operations."""
    
    # Analysis defaults - optimized for your corrected workflow
    DEFAULT_RESOLUTION = 30.0
    DEFAULT_CLOUD_COVER_MAX = 0.2
    DEFAULT_METHOD = DensityMethod.RASTERIZATION  # Your corrected default
    DEFAULT_CHUNK_SIZE = 200.0  # Large to avoid merging issues
    DEFAULT_MEMORY_LIMIT = 16.0
    DEFAULT_MAX_SCENES_FOOTPRINT = 150  # NEW: Increased from 50
    
    # Output defaults
    DEFAULT_OUTPUT_DIR = "./planetscope_analysis"
    DEFAULT_CREATE_VISUALIZATIONS = True
    DEFAULT_EXPORT_GEOTIFF = True
    DEFAULT_CLIP_TO_ROI = True
    
    # Time period defaults
    DEFAULT_PERIOD = "2025-01-01/2025-01-31"


# ... [Keep all the existing parse_time_period, parse_roi_input, etc. functions from the previous version] ...

def parse_time_period(time_period: Union[str, tuple]) -> Tuple[str, str]:
    """Parse time period specification into start and end dates."""
    if isinstance(time_period, tuple):
        return str(time_period[0]), str(time_period[1])
    
    if isinstance(time_period, str):
        if "/" in time_period:
            start_date, end_date = time_period.split("/")
            return start_date.strip(), end_date.strip()
        elif time_period == "last_month":
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        elif time_period == "last_3_months":
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    
    raise ValidationError(f"Invalid time period format: {time_period}")


def parse_roi_input(roi_input: Union[str, Polygon, list, dict]) -> Polygon:
    """
    Parse various ROI input formats into a Shapely Polygon.
    
    This function now leverages the enhanced validate_geometry() from utils.py
    which already handles shapefiles, GeoJSON files, WKT strings, etc.
    """
    from .utils import validate_geometry
    from shapely.geometry import shape
    
    try:
        # Use the enhanced validate_geometry function which handles:
        # - File paths (.shp, .geojson, .wkt, .txt)
        # - Shapely objects
        # - GeoJSON dictionaries
        # - WKT strings
        validated_geom = validate_geometry(roi_input)
        
        # Convert the validated GeoJSON geometry to Shapely Polygon
        polygon = shape(validated_geom)
        
        # Ensure it's a Polygon (not MultiPolygon, etc.)
        if polygon.geom_type == 'MultiPolygon':
            # Take the largest polygon
            polygons = list(polygon.geoms)
            polygon = max(polygons, key=lambda p: p.area)
        elif polygon.geom_type != 'Polygon':
            raise ValidationError(f"ROI must be a Polygon, got {polygon.geom_type}")
        
        return polygon
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Failed to parse ROI input: {e}")


def create_output_directory(base_dir: str) -> str:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"planetscope_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_analysis_metadata(
    density_result: DensityResult,
    scenes_result: dict,
    roi_polygon: Polygon,
    start_date: str,
    end_date: str,
    metadata_path: str
) -> None:
    """Save comprehensive analysis metadata."""
    try:
        import json
        
        metadata = {
            "analysis_info": {
                "analysis_type": "spatial_density",
                "method": "rasterization_corrected",
                "timestamp": datetime.now().isoformat(),
                "library_version": "4.0.0",
                "computation_time_seconds": density_result.computation_time,
                "coordinate_system_fixed": True,
            },
            "roi_info": {
                "area_km2": calculate_area_km2(roi_polygon),
                "bounds": list(roi_polygon.bounds),
                "center": [roi_polygon.centroid.x, roi_polygon.centroid.y],
            },
            "query_parameters": {
                "start_date": start_date,
                "end_date": end_date,
                "scenes_found": len(scenes_result['features']),
            },
            "grid_info": density_result.grid_info,
            "density_statistics": density_result.stats,
            "coordinate_system": {
                "crs": density_result.crs,
                "transform_corrected": True,
                "pixel_orientation": "north_to_south",
                "display_orientation": "corrected_for_visualization",
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Metadata saved to: {metadata_path}")
        
    except Exception as e:
        logger.warning(f"Metadata save failed: {e}")


def analyze_density(
    roi_polygon: Union[Polygon, list, str],
    time_period: Union[str, tuple] = WorkflowConfig.DEFAULT_PERIOD,
    resolution: float = WorkflowConfig.DEFAULT_RESOLUTION,
    cloud_cover_max: float = WorkflowConfig.DEFAULT_CLOUD_COVER_MAX,
    output_dir: str = WorkflowConfig.DEFAULT_OUTPUT_DIR,
    create_visualizations: bool = WorkflowConfig.DEFAULT_CREATE_VISUALIZATIONS,
    export_geotiff: bool = WorkflowConfig.DEFAULT_EXPORT_GEOTIFF,
    clip_to_roi: bool = WorkflowConfig.DEFAULT_CLIP_TO_ROI,
    max_scenes_footprint: int = WorkflowConfig.DEFAULT_MAX_SCENES_FOOTPRINT,  # NEW parameter
    show_plots: bool = True,  # NEW: Control plot display in notebooks
    **kwargs
) -> Dict[str, Any]:
    """
    Complete density analysis workflow with FIXED visualization.
    
    NEW PARAMETERS:
    - max_scenes_footprint: Maximum scenes to show in footprint plot (default: 150)
    """
    logger.info("Starting PlanetScope density analysis workflow")
    
    try:
        # 1. Parse and validate inputs
        roi_poly = parse_roi_input(roi_polygon)
        start_date, end_date = parse_time_period(time_period)
        
        # Log workflow parameters
        roi_area = calculate_area_km2(roi_poly)
        logger.info(f"ROI area: {roi_area:.0f} km²")
        logger.info(f"Time period: {start_date} to {end_date}")
        logger.info(f"Resolution: {resolution}m")
        logger.info(f"Max scenes footprint: {max_scenes_footprint}")
        logger.info(f"Output: {output_dir}")
        
        # 2. Create output directory
        output_path = create_output_directory(output_dir)
        results = {
            'scenes_found': 0,
            'density_result': None,
            'visualizations': {},
            'exports': {},
            'summary': {},
            'output_directory': output_path,
            'scene_polygons': []  # NEW: Store for individual access
        }
        
        # 3. Scene Discovery
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
        logger.info(f"Found {scenes_found} scenes")
        
        if scenes_found == 0:
            logger.warning("No scenes found for the specified criteria")
            return results
        
        # Extract scene polygons for later use
        scene_polygons = []
        for feature in scenes_result['features']:
            coords = feature['geometry']['coordinates'][0]
            scene_polygons.append(Polygon(coords))
        results['scene_polygons'] = scene_polygons
        
        # 4. Calculate Spatial Density using the PROPER density engine
        logger.info("Calculating spatial density with coordinate system corrections")
        
        # Create density configuration with corrected defaults
        density_config = DensityConfig(
            resolution=resolution,
            method=DensityMethod.RASTERIZATION,
            chunk_size_km=200.0,
            coordinate_system_fixes=True,
            force_single_chunk=kwargs.get('force_single_chunk', False),
            no_data_value=-9999.0
        )
        
        # Use the PROPER SpatialDensityEngine from density_engine.py
        engine = SpatialDensityEngine(density_config)
        
        # Calculate density using the engine's standard method
        density_result = engine.calculate_density(
        scene_footprints=scenes_result['features'],
        roi_geometry=roi_poly,
        clip_to_roi=clip_to_roi  # Pass the clip_to_roi parameter
    )
        
        results['density_result'] = density_result
        logger.info(f"Density calculated: {density_result.stats['mean']:.1f} avg scenes/pixel")
        analysis_type = "ROI-clipped" if clip_to_roi else "full grid"
        logger.info(f"Density calculated ({analysis_type}): {density_result.stats['mean']:.1f} avg scenes/pixel")

            
        # 5. Create visualizations (FIXED coordinate system)
        if create_visualizations:
            logger.info("Creating enhanced visualizations with coordinate fixes")
            visualizer = DensityVisualizer()
            
            # Create summary plot with time period and cloud cover information
            summary_fig = visualizer.create_summary_plot(
                density_result=density_result,
                scene_polygons=scene_polygons,
                roi_polygon=roi_poly,
                save_path=os.path.join(output_path, "summary_plot.png"),
                clip_to_roi=clip_to_roi,
                max_scenes_footprint=max_scenes_footprint,
                show_plot=show_plots,
                start_date=start_date,  # Pass the start date
                end_date=end_date,      # Pass the end date
                cloud_cover_max=cloud_cover_max  # Pass the cloud cover threshold
            )
            results['visualizations']['summary'] = summary_fig
            
            # Individual plots with FIXES
            density_plot_path = os.path.join(output_path, "density_map.png")
            visualizer.plot_density_map(
                density_result,
                roi_polygon=roi_poly,
                title="Scene Density Map (Coordinate-Corrected)",
                save_path=density_plot_path,
                clip_to_roi=clip_to_roi,
                show_plot=False
            )
            results['visualizations']['density_map'] = density_plot_path
            
            # Scene footprints with increased limits
            footprints_path = os.path.join(output_path, "scene_footprints.png")
            visualizer.plot_scene_footprints(
                scene_polygons,
                roi_poly,
                title="Scene Footprints (Enhanced)",
                max_scenes=max_scenes_footprint,
                save_path=footprints_path,
                show_plot=False
            )
            results['visualizations']['footprints'] = footprints_path
            
            logger.info("Enhanced visualizations generated successfully")
        
        # 6. Export Professional GeoTIFF with FIXES
        if export_geotiff:
            logger.info("Exporting enhanced GeoTIFF with coordinate corrections")
            
            if clip_to_roi:
                geotiff_path = os.path.join(output_path, "density_map_clipped.tif")
            else:
                geotiff_path = os.path.join(output_path, "density_map.tif")
            
            # Use enhanced export with FIXED coordinate handling
            success = export_density_geotiff_robust(
                density_result, 
                geotiff_path,
                roi_polygon=roi_poly if clip_to_roi else None,
                clip_to_roi=clip_to_roi
            )
            
            if success:
                results['exports']['geotiff'] = geotiff_path
                qml_path = geotiff_path.replace('.tif', '.qml')
                if os.path.exists(qml_path):
                    results['exports']['qml_style'] = qml_path
                logger.info("GeoTIFF export completed successfully")
            else:
                logger.warning("GeoTIFF export failed")
        
        # 7. Save Metadata
        metadata_path = os.path.join(output_path, "analysis_metadata.json")
        save_analysis_metadata(
            density_result, scenes_result, roi_poly, 
            start_date, end_date, metadata_path
        )
        results['exports']['metadata'] = metadata_path
        
        # 8. Create Analysis Summary
        results['summary'] = {
            'roi_area_km2': calculate_area_km2(roi_poly),
            'analysis_mode': 'ROI-clipped' if clip_to_roi else 'full grid',
            'scenes_found': scenes_found,
            'analysis_resolution_m': resolution,
            'computation_time_s': density_result.computation_time,
            'mean_density': density_result.stats['mean'],
            'max_density': density_result.stats['max'],
            'total_observations': density_result.stats.get('total_scenes', 0),
            'output_directory': output_path,
            'coordinate_system_corrected': True,
            'max_scenes_displayed': max_scenes_footprint
        }
        
        logger.info("Analysis workflow completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Analysis workflow failed: {e}")
        raise PlanetScopeError(f"Density analysis workflow failed: {e}")


def export_density_geotiff_robust(
    density_result: DensityResult, 
    output_path: str, 
    roi_polygon: Optional[Polygon] = None,
    clip_to_roi: bool = True
) -> bool:
    """Export density as GeoTIFF with robust PROJ error handling and FIXED coordinates."""
    try:
        logger.info(f"Exporting GeoTIFF to: {output_path}")
        
        # Get density data
        density_array = density_result.density_array
        no_data_value = getattr(density_result, 'no_data_value', -9999.0)
        
        # Apply ROI clipping if requested
        export_array = density_array
        if clip_to_roi and roi_polygon is not None:
            # Create ROI mask
            height, width = density_array.shape
            roi_mask = rasterize(
                [(roi_polygon, 1)],
                out_shape=(height, width),
                transform=density_result.transform,
                fill=0,
                dtype=np.uint8,
            )
            
            # Apply mask
            export_array = np.where(roi_mask == 1, density_array, no_data_value)
            logger.info("Applied ROI clipping to export data")
        
        # Try multiple CRS approaches for PROJ compatibility
        crs_options = [
            "EPSG:4326",
            "+proj=longlat +datum=WGS84 +no_defs",
            None  # No CRS fallback
        ]
        
        for crs_option in crs_options:
            try:
                logger.info(f"Attempting export with CRS: {crs_option}")
                
                with rasterio.open(
                    output_path, "w", driver="GTiff",
                    height=export_array.shape[0],
                    width=export_array.shape[1],
                    count=1, dtype=export_array.dtype,
                    crs=crs_option, transform=density_result.transform,
                    compress="lzw", nodata=no_data_value,
                ) as dst:
                    dst.write(export_array, 1)
                    
                    # Add comprehensive metadata
                    dst.update_tags(
                        title="PlanetScope Scene Density Analysis (Coordinate-Corrected)",
                        description="Scene density with FIXED coordinate system and orientation",
                        method="rasterization_corrected",
                        resolution=f"{density_result.grid_info.get('resolution', 'unknown')}m",
                        roi_clipped=str(clip_to_roi and roi_polygon is not None),
                        coordinate_fixes="enabled",
                        display_orientation="corrected",
                        crs_used=str(crs_option) if crs_option else "none",
                        created_by="PlanetScope-py Enhanced Workflow v4.0.0 (Fixed)"
                    )
                
                logger.info(f"GeoTIFF export successful with CRS: {crs_option}")
                
                # Create QML style file
                create_qml_style_file(export_array, output_path, no_data_value)
                
                return True
                
            except Exception as e:
                logger.warning(f"Export failed with CRS {crs_option}: {e}")
                continue
        
        logger.error("All GeoTIFF export attempts failed")
        return False
        
    except Exception as e:
        logger.error(f"GeoTIFF export error: {e}")
        return False


def create_qml_style_file(density_array: np.ndarray, geotiff_path: str, no_data_value: float):
    """Create QGIS style file for the density raster."""
    try:
        qml_path = geotiff_path.replace('.tif', '.qml')
        
        # Calculate data range
        valid_data = density_array[density_array != no_data_value]
        if len(valid_data) == 0:
            min_val, max_val = 0, 1
        else:
            min_val = float(np.min(valid_data))
            max_val = float(np.max(valid_data))
        
        # Create QML content with viridis-like colors
        qml_content = f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1">
      <rastershader>
        <colorrampshader minimumValue="{min_val}" maximumValue="{max_val}" colorRampType="INTERPOLATED">
          <item alpha="255" value="{min_val}" label="{min_val:.1f}" color="68,1,84,255"/>
          <item alpha="255" value="{min_val + (max_val-min_val)*0.25}" label="{min_val + (max_val-min_val)*0.25:.1f}" color="59,82,139,255"/>
          <item alpha="255" value="{min_val + (max_val-min_val)*0.5}" label="{min_val + (max_val-min_val)*0.5:.1f}" color="33,145,140,255"/>
          <item alpha="255" value="{min_val + (max_val-min_val)*0.75}" label="{min_val + (max_val-min_val)*0.75:.1f}" color="94,201,98,255"/>
          <item alpha="255" value="{max_val}" label="{max_val:.1f}" color="253,231,37,255"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
</qgis>'''
        
        with open(qml_path, 'w') as f:
            f.write(qml_content)
        
        logger.info(f"QML style file created: {qml_path}")
        
    except Exception as e:
        logger.warning(f"QML style file creation failed: {e}")


# NEW: ONE-LINE FUNCTIONS for individual outputs

def quick_density_plot(roi_polygon, time_period="last_month", save_path=None, **kwargs):
    """
    ONE-LINE function to get just the density plot.
    
    Usage:
        quick_density_plot(milan_roi, "2025-01-01/2025-01-31", "density.png")
    """
    result = analyze_density(
        roi_polygon, time_period, 
        create_visualizations=False, export_geotiff=False, **kwargs
    )
    
    if result['density_result'] is not None:
        return plot_density_only(
            result['density_result'], 
            roi_polygon=roi_polygon, 
            save_path=save_path,
            **kwargs
        )
    return None


def quick_footprints_plot(roi_polygon, time_period="last_month", save_path=None, max_scenes=300, **kwargs):
    """
    ONE-LINE function to get just the scene footprints plot.
    
    Usage:
        quick_footprints_plot(milan_roi, "last_month", "footprints.png", max_scenes=500)
    """
    result = analyze_density(
        roi_polygon, time_period,
        create_visualizations=False, export_geotiff=False, **kwargs
    )
    
    if result['scene_polygons']:
        return plot_footprints_only(
            result['scene_polygons'],
            roi_polygon,
            save_path=save_path,
            max_scenes=max_scenes,
            **kwargs
        )
    return None


def quick_geotiff_export(roi_polygon, time_period="last_month", output_path="density.tif", **kwargs):
    """
    ONE-LINE function to get just the GeoTIFF + QML files.
    
    Usage:
        quick_geotiff_export(milan_roi, "2025-01-01/2025-01-31", "milan_density.tif")
    """
    result = analyze_density(
        roi_polygon, time_period,
        create_visualizations=False, export_geotiff=False, **kwargs
    )
    
    if result['density_result'] is not None:
        return export_geotiff_only(
            result['density_result'],
            output_path,
            roi_polygon=roi_polygon,
            clip_to_roi=kwargs.get('clip_to_roi', True)
        )
    return False


def quick_analysis(
    roi: Union[str, Polygon, list],
    period: str = "last_month", 
    output_dir: str = "./output",
    max_scenes_footprint: int = 150,  # NEW parameter with increased default
    show_plots: bool = True,  # NEW: Add this parameter
    **config_overrides
) -> Dict[str, Any]:
    """
    Ultra-simplified analysis function with FIXED visualization.
    
    NEW PARAMETERS:
    - max_scenes_footprint: Maximum scenes in footprint plot (default: 150)
    """
    logger.info("Starting quick analysis workflow")
    
    # Parse period shortcuts
    if period in ["last_month", "last_3_months"]:
        time_period = period
    else:
        time_period = period
    
    # Use smart defaults with any overrides
    defaults = {
        'resolution': WorkflowConfig.DEFAULT_RESOLUTION,
        'cloud_cover_max': WorkflowConfig.DEFAULT_CLOUD_COVER_MAX,
        'output_dir': output_dir,
        'create_visualizations': True,
        'export_geotiff': True,
        'clip_to_roi': True,
        'max_scenes_footprint': max_scenes_footprint, # Pass through the parameter
        'show_plots': show_plots  # Pass through the parameter
    }
    defaults.update(config_overrides)
    
    return analyze_density(roi, time_period, **defaults)


# Keep existing batch_analysis and temporal_analysis_workflow functions...
def batch_analysis(
    roi_list: List[Union[Polygon, list]],
    time_period: Union[str, tuple] = WorkflowConfig.DEFAULT_PERIOD,
    base_output_dir: str = "./batch_analysis",
    **kwargs
) -> Dict[str, Any]:
    """Batch analysis for multiple ROIs."""
    logger.info(f"Starting batch analysis for {len(roi_list)} ROIs")
    
    batch_results = {}
    
    for i, roi in enumerate(roi_list):
        try:
            logger.info(f"Processing ROI {i+1}/{len(roi_list)}")
            
            roi_output_dir = os.path.join(base_output_dir, f"roi_{i+1:03d}")
            
            result = analyze_density(
                roi_polygon=roi,
                time_period=time_period,
                output_dir=roi_output_dir,
                **kwargs
            )
            
            batch_results[f"roi_{i+1}"] = result
            logger.info(f"ROI {i+1} completed successfully")
            
        except Exception as e:
            logger.error(f"ROI {i+1} failed: {e}")
            batch_results[f"roi_{i+1}"] = {'error': str(e)}
    
    logger.info("Batch analysis completed")
    return batch_results


def temporal_analysis_workflow(
    roi_polygon: Union[Polygon, list],
    time_periods: List[Union[str, tuple]],
    output_dir: str = "./temporal_analysis",
    **kwargs
) -> Dict[str, Any]:
    """Temporal analysis workflow for multiple time periods."""
    logger.info(f"Starting temporal analysis for {len(time_periods)} periods")
    
    temporal_results = {}
    
    for i, period in enumerate(time_periods):
        try:
            logger.info(f"Processing period {i+1}/{len(time_periods)}: {period}")
            
            period_str = str(period).replace('/', '_').replace(' ', '_')
            period_output_dir = os.path.join(output_dir, f"period_{i+1:02d}_{period_str}")
            
            result = analyze_density(
                roi_polygon=roi_polygon,
                time_period=period,
                output_dir=period_output_dir,
                **kwargs
            )
            
            temporal_results[period_str] = result
            logger.info(f"Period {i+1} completed successfully")
            
        except Exception as e:
            logger.error(f"Period {i+1} failed: {e}")
            temporal_results[period_str] = {'error': str(e)}
    
    logger.info("Temporal analysis workflow completed")
    return temporal_results

def quick_planet_analysis(
    roi: Union[Dict, Polygon, List[Tuple[float, float]]],
    period: str = "last_month",
    output_dir: str = "./analysis_output",
    resolution: float = 30.0,
    cloud_cover_max: float = 0.3,
    method: str = "auto",
    max_scenes_footprint: int = 150,
    clip_to_roi: bool = True,  # NEW PARAMETER: Control clipping behavior
    create_visualizations: bool = True,
    export_geotiff: bool = True,
    show_plots: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Enhanced one-line Planet analysis with TURBO colormap and full grid support.

    This function performs complete PlanetScope scene analysis including search,
    spatial density calculation, and visualization generation with coordinate fixes.

    Args:
        roi: Region of interest (GeoJSON dict, Shapely Polygon, or coordinate list)
        period: Time period ("last_month", "last_3_months", or "YYYY-MM-DD/YYYY-MM-DD")
        output_dir: Output directory for results
        resolution: Spatial resolution in meters (default: 30m)
        cloud_cover_max: Maximum cloud coverage (0.0-1.0)
        method: Density calculation method ("auto", "rasterization", "vector_overlay", "adaptive_grid")
        max_scenes_footprint: Maximum scenes to display in footprint plots
        clip_to_roi: If True, clip analysis to ROI shape. If False, analyze full grid covering all scenes
        create_visualizations: Whether to generate plots
        export_geotiff: Whether to export GeoTIFF
        show_plots: Whether to display plots
        **kwargs: Additional parameters

    Returns:
        Dictionary containing all analysis results
    """
    # Just call the existing analyze_density function with all parameters
    return analyze_density(
        roi_polygon=roi,
        time_period=period,
        output_dir=output_dir,
        resolution=resolution,
        cloud_cover_max=cloud_cover_max,
        max_scenes_footprint=max_scenes_footprint,
        clip_to_roi=clip_to_roi,  # Pass the clip_to_roi parameter
        create_visualizations=create_visualizations,
        export_geotiff=export_geotiff,
        show_plots=show_plots,
        **kwargs
    )