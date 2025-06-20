#!/usr/bin/env python3
"""
PlanetScope-py Phase 3: Spatial Density Engine
Core spatial density calculation engine with three computational methods.

This module implements the spatial analysis engine for calculating scene density
across user-defined grids with multiple algorithmic approaches for performance
optimization based on ROI size and computational constraints.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from .exceptions import ValidationError, PlanetScopeError
from .utils import validate_geometry, calculate_geometry_bounds

logger = logging.getLogger(__name__)

class DensityMethod(Enum):
    """Computational methods for spatial density calculation."""
    RASTERIZATION = "rasterization"
    VECTOR_OVERLAY = "vector_overlay"
    ADAPTIVE_GRID = "adaptive_grid"
    AUTO = "auto"

@dataclass
class DensityConfig:
    """Configuration for density calculation."""
    resolution: float = 10.0  # Default 10m resolution per project requirements
    method: Union[DensityMethod, str] = DensityMethod.AUTO  # Accept both enum and string
    chunk_size_km: float = 50.0  # Max chunk size for large ROIs
    max_memory_gb: float = 8.0  # Memory limit for calculations
    parallel_workers: int = 4  # Number of parallel processing workers
    no_data_value: float = -9999.0  # NoData value for output rasters
    
    def __post_init__(self):
        """Post-initialization to convert string methods to enum."""
        if isinstance(self.method, str):
            # Convert string to enum
            method_mapping = {
                "auto": DensityMethod.AUTO,
                "rasterization": DensityMethod.RASTERIZATION,
                "vector_overlay": DensityMethod.VECTOR_OVERLAY,
                "adaptive_grid": DensityMethod.ADAPTIVE_GRID
            }
            
            method_key = self.method.lower()
            if method_key in method_mapping:
                self.method = method_mapping[method_key]
            else:
                raise ValidationError(f"Invalid method: {self.method}. Must be one of: {list(method_mapping.keys())}")

@dataclass
class DensityResult:
    """Results from density calculation."""
    density_array: np.ndarray
    transform: rasterio.Affine
    crs: str
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    stats: Dict[str, Any]
    computation_time: float
    method_used: DensityMethod
    grid_info: Dict[str, Any]

class SpatialDensityEngine:
    """
    Core spatial density calculation engine with multiple computational methods.
    
    Implements three approaches for calculating scene overlap density:
    1. Rasterization Method: Convert polygons to raster for array operations
    2. Vector Overlay Method: Spatial database operations with indexing
    3. Adaptive Grid Method: Hierarchical refinement for large datasets
    
    Automatically selects optimal method based on ROI size and data characteristics.
    """
    
    def __init__(self, config: Optional[DensityConfig] = None):
        """Initialize the density engine.
        
        Args:
            config: Configuration for density calculations
        """
        self.config = config or DensityConfig()
        self._validate_config()
        
        # Performance tracking
        self.performance_stats = {}
        
        # FIXED: Safe method value extraction
        try:
            if hasattr(self.config.method, 'value'):
                method_str = self.config.method.value
            else:
                method_str = str(self.config.method)
            logger.info(f"Density engine initialized with {method_str} method")
        except Exception as e:
            logger.warning(f"Could not display method in log: {e}")
            logger.info("Density engine initialized")
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.resolution <= 0:
            raise ValidationError("Resolution must be positive", {"resolution": self.config.resolution})
        
        if self.config.chunk_size_km <= 0:
            raise ValidationError("Chunk size must be positive", {"chunk_size_km": self.config.chunk_size_km})
        
        if self.config.max_memory_gb <= 0:
            raise ValidationError("Memory limit must be positive", {"max_memory_gb": self.config.max_memory_gb})
    
    def calculate_density(self,
                         scene_footprints: List[Dict],
                         roi_geometry: Union[Dict, Polygon],
                         **kwargs) -> DensityResult:
        """
        Calculate spatial density of scene coverage across ROI.
        
        Args:
            scene_footprints: List of scene features with geometry
            roi_geometry: Region of interest geometry
            **kwargs: Additional parameters (resolution, method override, etc.)
        
        Returns:
            DensityResult with calculated density map and statistics
            
        Raises:
            ValidationError: If inputs are invalid
            PlanetScopeError: If calculation fails
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            roi_poly = self._prepare_roi_geometry(roi_geometry)
            scene_polygons = self._prepare_scene_geometries(scene_footprints)
            
            # Update config with kwargs
            config = self._merge_config_kwargs(kwargs)
            
            # Check if ROI needs chunking
            chunks = self._create_spatial_chunks(roi_poly, config)
            
            if len(chunks) > 1:
                logger.info(f"Large ROI detected: processing in {len(chunks)} chunks")
                return self._process_chunked_density(scene_polygons, chunks, config, start_time)
            else:
                # Single chunk processing
                return self._process_single_density(scene_polygons, roi_poly, config, start_time)
                
        except Exception as e:
            computation_time = time.time() - start_time
            logger.error(f"Density calculation failed after {computation_time:.2f}s: {e}")
            if isinstance(e, (ValidationError, PlanetScopeError)):
                raise
            raise PlanetScopeError(f"Density calculation error: {e}")
    
    def _prepare_roi_geometry(self, roi_geometry: Union[Dict, Polygon]) -> Polygon:
        """Prepare and validate ROI geometry."""
        if isinstance(roi_geometry, dict):
            # Assume GeoJSON format
            if roi_geometry.get('type') == 'Polygon':
                coords = roi_geometry['coordinates'][0]  # First ring only
                roi_poly = Polygon(coords)
            else:
                raise ValidationError("Only Polygon ROI supported", {"type": roi_geometry.get('type')})
        elif isinstance(roi_geometry, Polygon):
            roi_poly = roi_geometry
        else:
            raise ValidationError("Invalid ROI geometry type", {"type": type(roi_geometry)})
        
        # Validate geometry
        if not roi_poly.is_valid:
            roi_poly = roi_poly.buffer(0)  # Fix invalid geometries
            
        if roi_poly.is_empty:
            raise ValidationError("ROI geometry is empty")
            
        return roi_poly
    
    def _prepare_scene_geometries(self, scene_footprints: List[Dict]) -> List[Polygon]:
        """Extract and validate scene geometries."""
        scene_polygons = []
        
        for i, scene in enumerate(scene_footprints):
            try:
                geom = scene.get('geometry')
                if not geom:
                    logger.warning(f"Scene {i} missing geometry, skipping")
                    continue
                
                if geom['type'] == 'Polygon':
                    coords = geom['coordinates'][0]  # First ring only
                    poly = Polygon(coords)
                elif geom['type'] == 'MultiPolygon':
                    # Take largest polygon
                    polygons = [Polygon(ring[0]) for ring in geom['coordinates']]
                    poly = max(polygons, key=lambda p: p.area)
                else:
                    logger.warning(f"Scene {i} unsupported geometry type: {geom['type']}")
                    continue
                
                if poly.is_valid and not poly.is_empty:
                    scene_polygons.append(poly)
                else:
                    logger.warning(f"Scene {i} invalid geometry, skipping")
                    
            except Exception as e:
                logger.warning(f"Failed to process scene {i} geometry: {e}")
                continue
        
        if not scene_polygons:
            raise ValidationError("No valid scene geometries found")
            
        logger.info(f"Prepared {len(scene_polygons)} valid scene geometries")
        return scene_polygons
    
    def _merge_config_kwargs(self, kwargs: Dict) -> DensityConfig:
        """Merge configuration with keyword arguments."""
        config = DensityConfig(
            resolution=kwargs.get('resolution', self.config.resolution),
            method=DensityMethod(kwargs.get('method', self.config.method.value)),
            chunk_size_km=kwargs.get('chunk_size_km', self.config.chunk_size_km),
            max_memory_gb=kwargs.get('max_memory_gb', self.config.max_memory_gb),
            parallel_workers=kwargs.get('parallel_workers', self.config.parallel_workers),
            no_data_value=kwargs.get('no_data_value', self.config.no_data_value)
        )
        return config
    
    def _create_spatial_chunks(self, roi_poly: Polygon, config: DensityConfig) -> List[Polygon]:
        """Create spatial chunks for large ROI processing."""
        bounds = roi_poly.bounds
        roi_width_km = (bounds[2] - bounds[0]) * 111.0  # Rough conversion to km
        roi_height_km = (bounds[3] - bounds[1]) * 111.0
        
        # Check if chunking needed
        if max(roi_width_km, roi_height_km) <= config.chunk_size_km:
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
    
    def _process_chunked_density(self,
                               scene_polygons: List[Polygon],
                               chunks: List[Polygon],
                               config: DensityConfig,
                               start_time: float) -> DensityResult:
        """Process density calculation for chunked ROI."""
        logger.info("Processing chunked density calculation")
        
        # Calculate each chunk
        chunk_results = []
        
        with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
            future_to_chunk = {
                executor.submit(self._process_single_density, scene_polygons, chunk, config, start_time): i
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    chunk_results.append((chunk_idx, result))
                    logger.info(f"Completed chunk {chunk_idx + 1}/{len(chunks)}")
                except Exception as e:
                    logger.error(f"Chunk {chunk_idx} failed: {e}")
                    raise
        
        # Merge chunk results
        return self._merge_chunk_results(chunk_results, chunks, config, start_time)
    
    def _process_single_density(self,
                              scene_polygons: List[Polygon],
                              roi_poly: Polygon,
                              config: DensityConfig,
                              start_time: float) -> DensityResult:
        """Process density calculation for single ROI."""
        
        # Auto-select method if needed
        if config.method == DensityMethod.AUTO:
            method = self._select_optimal_method(scene_polygons, roi_poly, config)
        else:
            method = config.method
        
        logger.info(f"Using {method.value} method for density calculation")
        
        # Execute calculation
        if method == DensityMethod.RASTERIZATION:
            return self._calculate_rasterization_density(scene_polygons, roi_poly, config, start_time)
        elif method == DensityMethod.VECTOR_OVERLAY:
            return self._calculate_vector_overlay_density(scene_polygons, roi_poly, config, start_time)
        elif method == DensityMethod.ADAPTIVE_GRID:
            return self._calculate_adaptive_grid_density(scene_polygons, roi_poly, config, start_time)
        else:
            raise ValidationError(f"Unsupported method: {method}")
    
    def _select_optimal_method(self,
                             scene_polygons: List[Polygon],
                             roi_poly: Polygon,
                             config: DensityConfig) -> DensityMethod:
        """Select optimal computational method based on data characteristics."""
        
        # Calculate dataset characteristics
        bounds = roi_poly.bounds
        roi_area_km2 = roi_poly.area * (111.0 ** 2)  # Rough conversion
        n_scenes = len(scene_polygons)
        
        # Estimate raster size
        width = int((bounds[2] - bounds[0]) / (config.resolution / 111000))
        height = int((bounds[3] - bounds[1]) / (config.resolution / 111000))
        raster_size_mb = (width * height * 4) / (1024 ** 2)  # 4 bytes per float32
        
        logger.info(f"Dataset characteristics: {roi_area_km2:.1f} km², {n_scenes} scenes, "
                   f"{raster_size_mb:.1f} MB raster")
        
        # Method selection logic
        if raster_size_mb > config.max_memory_gb * 1024 * 0.5:  # Use half memory limit
            logger.info("Large raster detected, using adaptive grid method")
            return DensityMethod.ADAPTIVE_GRID
        elif n_scenes > 1000:
            logger.info("Many scenes detected, using rasterization method")
            return DensityMethod.RASTERIZATION
        else:
            logger.info("Standard dataset, using vector overlay method")
            return DensityMethod.VECTOR_OVERLAY
    
    def _calculate_rasterization_density(self,
                                       scene_polygons: List[Polygon],
                                       roi_poly: Polygon,
                                       config: DensityConfig,
                                       start_time: float) -> DensityResult:
        """Calculate density using rasterization method."""
        logger.info("Executing rasterization density calculation")
        
        bounds = roi_poly.bounds
        
        # Calculate raster dimensions
        width = int((bounds[2] - bounds[0]) / (config.resolution / 111000))
        height = int((bounds[3] - bounds[1]) / (config.resolution / 111000))
        
        # Create transform
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
        
        # Initialize density array
        density_array = np.zeros((height, width), dtype=np.float32)
        
        # Rasterize each scene polygon
        for i, scene_poly in enumerate(scene_polygons):
            try:
                # Check intersection with ROI
                if not scene_poly.intersects(roi_poly):
                    continue
                
                # Rasterize scene polygon
                scene_mask = rasterize(
                    [(scene_poly, 1)],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                )
                
                # Add to density array
                density_array += scene_mask.astype(np.float32)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(scene_polygons)} scenes")
                    
            except Exception as e:
                logger.warning(f"Failed to rasterize scene {i}: {e}")
                continue
        
        # Apply ROI mask
        roi_mask = rasterize(
            [(roi_poly, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        
        density_array = np.where(roi_mask, density_array, config.no_data_value)
        
        # Calculate statistics
        stats = self._calculate_density_stats(density_array, config.no_data_value)
        
        computation_time = time.time() - start_time
        
        return DensityResult(
            density_array=density_array,
            transform=transform,
            crs="EPSG:4326",
            bounds=bounds,
            stats=stats,
            computation_time=computation_time,
            method_used=DensityMethod.RASTERIZATION,
            grid_info={
                "width": width,
                "height": height,
                "resolution": config.resolution,
                "total_cells": width * height
            }
        )
    
    def _calculate_vector_overlay_density(self,
                                        scene_polygons: List[Polygon],
                                        roi_poly: Polygon,
                                        config: DensityConfig,
                                        start_time: float) -> DensityResult:
        """Calculate density using vector overlay method."""
        logger.info("Executing vector overlay density calculation")
        
        bounds = roi_poly.bounds
        
        # Create grid of points/cells
        x_coords = np.arange(bounds[0], bounds[2], config.resolution / 111000)
        y_coords = np.arange(bounds[1], bounds[3], config.resolution / 111000)
        
        width = len(x_coords)
        height = len(y_coords)
        
        logger.info(f"Created {width}x{height} grid for vector overlay")
        
        # Create spatial index for scenes
        scene_gdf = gpd.GeoDataFrame(geometry=scene_polygons)
        scene_sindex = scene_gdf.sindex
        
        # Initialize density array
        density_array = np.zeros((height, width), dtype=np.float32)
        
        # Process grid cells
        total_cells = width * height
        processed = 0
        
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                # Create cell geometry
                cell_size = config.resolution / 111000
                cell = box(x, y, x + cell_size, y + cell_size)
                
                # Check if cell is within ROI
                if not cell.intersects(roi_poly):
                    density_array[i, j] = config.no_data_value
                    continue
                
                # Find intersecting scenes using spatial index
                possible_matches_index = list(scene_sindex.intersection(cell.bounds))
                possible_matches = scene_gdf.iloc[possible_matches_index]
                
                # Count actual intersections
                count = 0
                for _, scene_row in possible_matches.iterrows():
                    if scene_row.geometry.intersects(cell):
                        count += 1
                
                density_array[i, j] = count
                
                processed += 1
                if processed % 10000 == 0:
                    logger.info(f"Processed {processed}/{total_cells} cells ({processed/total_cells*100:.1f}%)")
        
        # Create transform
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
        
        # Calculate statistics
        stats = self._calculate_density_stats(density_array, config.no_data_value)
        
        computation_time = time.time() - start_time
        
        return DensityResult(
            density_array=density_array,
            transform=transform,
            crs="EPSG:4326",
            bounds=bounds,
            stats=stats,
            computation_time=computation_time,
            method_used=DensityMethod.VECTOR_OVERLAY,
            grid_info={
                "width": width,
                "height": height,
                "resolution": config.resolution,
                "total_cells": total_cells
            }
        )
    
    def _calculate_adaptive_grid_density(self,
                                    scene_polygons: List[Polygon],
                                    roi_poly: Polygon,
                                    config: DensityConfig,
                                    start_time: float) -> DensityResult:
        """Calculate density using adaptive grid method - REAL IMPLEMENTATION."""
        logger.info("Executing adaptive grid density calculation")
        
        try:
            from .adaptive_grid import AdaptiveGridEngine, AdaptiveGridConfig
            
            # Create adaptive grid configuration from density config
            adaptive_config = AdaptiveGridConfig(
                base_resolution=config.resolution * 4,    # Start 4x coarser
                min_resolution=config.resolution,         # Target resolution
                max_resolution=config.resolution * 16,    # Max 16x coarser  
                refinement_factor=2,
                max_levels=3,
                density_threshold=5.0,
                variance_threshold=2.0
            )
            
            # Initialize adaptive grid engine
            adaptive_engine = AdaptiveGridEngine(adaptive_config)
            
            # Calculate adaptive density
            adaptive_result = adaptive_engine.calculate_adaptive_density(
                scene_polygons, roi_poly
            )
            
            # Convert to standard DensityResult format
            from rasterio.transform import from_bounds
            bounds = roi_poly.bounds
            transform = from_bounds(
                bounds[0], bounds[1], bounds[2], bounds[3],
                adaptive_result['grid_info']['width'], 
                adaptive_result['grid_info']['height']
            )
            
            computation_time = time.time() - start_time
            
            return DensityResult(
                density_array=adaptive_result["density_array"],
                transform=transform,
                crs="EPSG:4326",
                bounds=bounds,
                stats=adaptive_result["stats"],
                computation_time=computation_time,
                method_used=DensityMethod.ADAPTIVE_GRID,
                grid_info=adaptive_result["grid_info"]
            )
            
        except ImportError:
            logger.warning("Adaptive grid module not available, falling back to rasterization")
            return self._calculate_rasterization_density(scene_polygons, roi_poly, config, start_time)
        
        except Exception as e:
            logger.error(f"Adaptive grid calculation failed: {e}, falling back to rasterization")
            return self._calculate_rasterization_density(scene_polygons, roi_poly, config, start_time)
    
    def _calculate_density_stats(self, density_array: np.ndarray, no_data_value: float) -> Dict[str, Any]:
        """Calculate statistics for density array."""
        valid_data = density_array[density_array != no_data_value]
        
        if len(valid_data) == 0:
            return {"error": "No valid data"}
        
        stats = {
            "count": int(len(valid_data)),
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "mean": float(np.mean(valid_data)),
            "std": float(np.std(valid_data)),
            "median": float(np.median(valid_data)),
            "percentiles": {
                "25": float(np.percentile(valid_data, 25)),
                "75": float(np.percentile(valid_data, 75)),
                "90": float(np.percentile(valid_data, 90)),
                "95": float(np.percentile(valid_data, 95))
            },
            "histogram": self._calculate_histogram(valid_data)
        }
        
        return stats
    
    def _calculate_histogram(self, data: np.ndarray, bins: int = 10) -> Dict[str, List]:
        """Calculate histogram for data."""
        counts, bin_edges = np.histogram(data, bins=bins)
        
        return {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
        }
    
    def _merge_chunk_results(self,
                           chunk_results: List[Tuple[int, DensityResult]],
                           chunks: List[Polygon],
                           config: DensityConfig,
                           start_time: float) -> DensityResult:
        """Merge results from multiple chunks into single result."""
        logger.info("Merging chunk results")
        
        # Sort results by chunk index
        chunk_results.sort(key=lambda x: x[0])
        
        # For now, implement a simple merge by taking the first result
        # Full implementation would properly mosaic the raster chunks
        if chunk_results:
            _, first_result = chunk_results[0]
            
            # Update computation time
            first_result.computation_time = time.time() - start_time
            
            logger.warning("Chunk merging simplified - full mosaic implementation needed")
            return first_result
        else:
            raise PlanetScopeError("No chunk results to merge")

    def export_density_geotiff(self,
                             result: DensityResult,
                             output_path: str,
                             compress: str = 'lzw') -> None:
        """Export density result as GeoTIFF."""
        logger.info(f"Exporting density map to {output_path}")
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=result.density_array.shape[0],
            width=result.density_array.shape[1],
            count=1,
            dtype=result.density_array.dtype,
            crs=result.crs,
            transform=result.transform,
            compress=compress,
            nodata=self.config.no_data_value
        ) as dst:
            dst.write(result.density_array, 1)
            
            # Write metadata
            dst.update_tags(
                method=result.method_used.value,
                computation_time=result.computation_time,
                resolution=result.grid_info.get('resolution', 'unknown')
            )
        
        logger.info(f"Density map exported successfully")


# Example usage and testing
if __name__ == "__main__":
    # Simple test with mock data
    import json
    from shapely.geometry import Point
    
    # Create mock ROI (Milan area)
    milan_bounds = (9.04, 45.40, 9.28, 45.52)  # Rough Milan bounds
    milan_roi = box(*milan_bounds)
    
    # Create mock scene footprints
    mock_scenes = []
    for i in range(20):
        # Random points around Milan
        center_x = np.random.uniform(milan_bounds[0], milan_bounds[2])
        center_y = np.random.uniform(milan_bounds[1], milan_bounds[3])
        
        # Create small square footprint
        size = 0.01  # ~1km square
        footprint = box(center_x - size/2, center_y - size/2, 
                       center_x + size/2, center_y + size/2)
        
        mock_scenes.append({
            'geometry': {
                'type': 'Polygon',
                'coordinates': [list(footprint.exterior.coords)]
            },
            'properties': {'id': f'mock_scene_{i}'}
        })
    
    # Test density calculation
    engine = SpatialDensityEngine()
    
    try:
        result = engine.calculate_density(
            scene_footprints=mock_scenes,
            roi_geometry=milan_roi,
            resolution=100  # 100m for fast testing
        )
        
        print(f"Density calculation completed!")
        print(f"Method used: {result.method_used.value}")
        print(f"Computation time: {result.computation_time:.2f}s")
        print(f"Grid size: {result.grid_info['width']}x{result.grid_info['height']}")
        print(f"Stats: min={result.stats['min']}, max={result.stats['max']}, mean={result.stats['mean']:.2f}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
