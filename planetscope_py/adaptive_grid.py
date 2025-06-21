#!/usr/bin/env python3
"""
PlanetScope-py Phase 3: Adaptive Grid Module
Hierarchical grid refinement for large-scale spatial density analysis.

This module implements adaptive grid refinement, starting with coarse grids
and progressively refining high-activity areas to optimize computation
for large ROIs while maintaining accuracy in data-rich regions.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize

from .exceptions import ValidationError, PlanetScopeError

logger = logging.getLogger(__name__)


class RefinementCriteria(Enum):
    """Criteria for determining grid refinement needs."""

    DENSITY_THRESHOLD = "density_threshold"
    VARIANCE_THRESHOLD = "variance_threshold"
    SCENE_COUNT = "scene_count"
    HYBRID = "hybrid"


@dataclass
class AdaptiveGridConfig:
    """Configuration for adaptive grid refinement."""

    base_resolution: float = 100.0  # Starting resolution in meters
    min_resolution: float = 10.0  # Minimum resolution (max refinement)
    max_resolution: float = 1000.0  # Maximum resolution (coarsest level)
    refinement_factor: int = 2  # Factor by which to refine grid
    max_levels: int = 4  # Maximum refinement levels
    density_threshold: float = 5.0  # Scenes per cell to trigger refinement
    variance_threshold: float = 2.0  # Variance threshold for refinement
    min_cell_scenes: int = 3  # Minimum scenes to consider for refinement
    criteria: RefinementCriteria = RefinementCriteria.HYBRID


@dataclass
class GridCell:
    """Represents a single grid cell with metadata."""

    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    level: int  # Refinement level (0 = coarsest)
    density: float  # Scene count/density
    variance: float  # Density variance from neighbors
    geometry: Polygon
    needs_refinement: bool = False
    children: Optional[List["GridCell"]] = None
    scene_count: int = 0


class AdaptiveGridEngine:
    """
    Adaptive grid engine for hierarchical spatial density calculation.

    Implements a hierarchical approach that starts with coarse grid cells
    and progressively refines areas with high scene density or high variance
    to balance computational efficiency with accuracy.
    """

    def __init__(self, config: Optional[AdaptiveGridConfig] = None):
        """Initialize adaptive grid engine.

        Args:
            config: Configuration for adaptive grid processing
        """
        self.config = config or AdaptiveGridConfig()
        self._validate_config()

        # Grid hierarchy tracking
        self.grid_hierarchy = {}
        self.performance_stats = {}

        logger.info(
            f"Adaptive grid engine initialized with base resolution {self.config.base_resolution}m"
        )

    def _validate_config(self) -> None:
        """Validate adaptive grid configuration."""
        if self.config.min_resolution >= self.config.base_resolution:
            raise ValidationError("Min resolution must be less than base resolution")

        if self.config.max_resolution <= self.config.base_resolution:
            raise ValidationError("Max resolution must be greater than base resolution")

        if self.config.refinement_factor < 2:
            raise ValidationError("Refinement factor must be at least 2")

        if self.config.max_levels < 1:
            raise ValidationError("Must allow at least 1 refinement level")

    def calculate_adaptive_density(
        self, scene_polygons: List[Polygon], roi_polygon: Polygon, **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate density using adaptive grid refinement.

        Args:
            scene_polygons: List of scene footprint polygons
            roi_polygon: Region of interest polygon
            **kwargs: Additional parameters

        Returns:
            Dictionary with density results and grid hierarchy info
        """
        start_time = time.time()

        try:
            logger.info(
                f"Starting adaptive grid calculation for {len(scene_polygons)} scenes"
            )

            # Step 1: Create initial coarse grid
            initial_grid = self._create_initial_grid(roi_polygon)
            logger.info(f"Created initial grid with {len(initial_grid)} cells")

            # Step 2: Calculate initial density for all cells
            self._calculate_initial_density(initial_grid, scene_polygons)

            # Step 3: Iteratively refine high-activity areas
            refined_grid = self._iterative_refinement(initial_grid, scene_polygons)

            # Step 4: Generate final density array
            density_result = self._generate_density_array(refined_grid, roi_polygon)

            computation_time = time.time() - start_time
            density_result["computation_time"] = computation_time
            density_result["method"] = "adaptive_grid"

            logger.info(
                f"Adaptive grid calculation completed in {computation_time:.2f}s"
            )

            return density_result

        except Exception as e:
            logger.error(f"Adaptive grid calculation failed: {e}")
            if isinstance(e, (ValidationError, PlanetScopeError)):
                raise
            raise PlanetScopeError(f"Adaptive grid error: {e}")

    def _create_initial_grid(self, roi_polygon: Polygon) -> List[GridCell]:
        """Create initial coarse grid covering the ROI."""
        bounds = roi_polygon.bounds

        # Calculate grid dimensions based on base resolution
        cell_size_deg = (
            self.config.base_resolution / 111000
        )  # Convert meters to degrees

        x_coords = np.arange(bounds[0], bounds[2] + cell_size_deg, cell_size_deg)
        y_coords = np.arange(bounds[1], bounds[3] + cell_size_deg, cell_size_deg)

        grid_cells = []

        for i in range(len(x_coords) - 1):
            for j in range(len(y_coords) - 1):
                cell_bounds = (
                    x_coords[i],
                    y_coords[j],
                    x_coords[i + 1],
                    y_coords[j + 1],
                )
                cell_geom = box(*cell_bounds)

                # Only include cells that intersect ROI
                if cell_geom.intersects(roi_polygon):
                    cell = GridCell(
                        bounds=cell_bounds,
                        level=0,
                        density=0.0,
                        variance=0.0,
                        geometry=cell_geom,
                        scene_count=0,
                    )
                    grid_cells.append(cell)

        return grid_cells

    def _calculate_initial_density(
        self, grid_cells: List[GridCell], scene_polygons: List[Polygon]
    ) -> None:
        """Calculate initial scene density for all grid cells."""
        logger.info("Calculating initial density for base grid")

        for cell in grid_cells:
            scene_count = 0

            # Count scenes intersecting this cell
            for scene_poly in scene_polygons:
                if scene_poly.intersects(cell.geometry):
                    scene_count += 1

            cell.scene_count = scene_count
            cell.density = float(scene_count)

        # Calculate variance for each cell based on neighbors
        self._calculate_cell_variance(grid_cells)

    def _calculate_cell_variance(self, grid_cells: List[GridCell]) -> None:
        """Calculate density variance for each cell based on neighbors."""

        for cell in grid_cells:
            # Find neighboring cells
            neighbors = self._find_neighbor_cells(cell, grid_cells)

            if len(neighbors) >= 3:  # Need enough neighbors for meaningful variance
                neighbor_densities = [neighbor.density for neighbor in neighbors]
                neighbor_densities.append(cell.density)  # Include self

                cell.variance = float(np.var(neighbor_densities))
            else:
                cell.variance = 0.0

    def _find_neighbor_cells(
        self, target_cell: GridCell, all_cells: List[GridCell]
    ) -> List[GridCell]:
        """Find neighboring cells using geometric proximity."""
        neighbors = []
        target_bounds = target_cell.bounds

        # Define search buffer (slightly larger than cell)
        buffer = (
            max(
                target_bounds[2] - target_bounds[0],  # width
                target_bounds[3] - target_bounds[1],  # height
            )
            * 1.1
        )

        target_center_x = (target_bounds[0] + target_bounds[2]) / 2
        target_center_y = (target_bounds[1] + target_bounds[3]) / 2

        for cell in all_cells:
            if cell == target_cell:
                continue

            cell_bounds = cell.bounds
            cell_center_x = (cell_bounds[0] + cell_bounds[2]) / 2
            cell_center_y = (cell_bounds[1] + cell_bounds[3]) / 2

            # Check if centers are within buffer distance
            distance = np.sqrt(
                (target_center_x - cell_center_x) ** 2
                + (target_center_y - cell_center_y) ** 2
            )

            if distance <= buffer:
                neighbors.append(cell)

        return neighbors

    def _iterative_refinement(
        self, initial_grid: List[GridCell], scene_polygons: List[Polygon]
    ) -> List[GridCell]:
        """Iteratively refine grid cells based on refinement criteria."""
        current_grid = initial_grid.copy()

        for level in range(1, self.config.max_levels + 1):
            logger.info(f"Refinement level {level}/{self.config.max_levels}")

            # Identify cells needing refinement
            cells_to_refine = self._identify_refinement_candidates(current_grid)

            if not cells_to_refine:
                logger.info("No cells need refinement, stopping early")
                break

            logger.info(f"Refining {len(cells_to_refine)} cells at level {level}")

            # Refine identified cells
            new_grid = []

            for cell in current_grid:
                if cell in cells_to_refine:
                    # Replace with refined children
                    children = self._refine_cell(cell, scene_polygons, level)
                    new_grid.extend(children)
                    cell.children = children
                else:
                    # Keep original cell
                    new_grid.append(cell)

            current_grid = new_grid

            # Recalculate variance for new grid
            self._calculate_cell_variance(current_grid)

        return current_grid

    def _identify_refinement_candidates(
        self, grid_cells: List[GridCell]
    ) -> List[GridCell]:
        """Identify cells that need refinement based on criteria."""
        candidates = []

        for cell in grid_cells:
            needs_refinement = False

            # Check if cell meets minimum scenes requirement
            if cell.scene_count < self.config.min_cell_scenes:
                continue

            # Apply refinement criteria
            if self.config.criteria == RefinementCriteria.DENSITY_THRESHOLD:
                if cell.density >= self.config.density_threshold:
                    needs_refinement = True

            elif self.config.criteria == RefinementCriteria.VARIANCE_THRESHOLD:
                if cell.variance >= self.config.variance_threshold:
                    needs_refinement = True

            elif self.config.criteria == RefinementCriteria.SCENE_COUNT:
                if cell.scene_count >= self.config.density_threshold:
                    needs_refinement = True

            elif self.config.criteria == RefinementCriteria.HYBRID:
                # Hybrid approach: density OR high variance
                if (
                    cell.density >= self.config.density_threshold
                    or cell.variance >= self.config.variance_threshold
                ):
                    needs_refinement = True

            if needs_refinement:
                cell.needs_refinement = True
                candidates.append(cell)

        return candidates

    def _refine_cell(
        self, parent_cell: GridCell, scene_polygons: List[Polygon], level: int
    ) -> List[GridCell]:
        """Refine a single cell into smaller children."""
        bounds = parent_cell.bounds

        # Calculate child cell dimensions
        parent_width = bounds[2] - bounds[0]
        parent_height = bounds[3] - bounds[1]

        child_width = parent_width / self.config.refinement_factor
        child_height = parent_height / self.config.refinement_factor

        children = []

        # Create child cells
        for i in range(self.config.refinement_factor):
            for j in range(self.config.refinement_factor):
                child_minx = bounds[0] + i * child_width
                child_miny = bounds[1] + j * child_height
                child_maxx = child_minx + child_width
                child_maxy = child_miny + child_height

                child_bounds = (child_minx, child_miny, child_maxx, child_maxy)
                child_geom = box(*child_bounds)

                # Calculate density for child cell
                child_scene_count = 0
                for scene_poly in scene_polygons:
                    if scene_poly.intersects(child_geom):
                        child_scene_count += 1

                child_cell = GridCell(
                    bounds=child_bounds,
                    level=level,
                    density=float(child_scene_count),
                    variance=0.0,  # Will be calculated later
                    geometry=child_geom,
                    scene_count=child_scene_count,
                )

                children.append(child_cell)

        return children

    def _generate_density_array(
        self, final_grid: List[GridCell], roi_polygon: Polygon
    ) -> Dict[str, Any]:
        """Generate final density array from refined grid."""
        logger.info("Generating final density array from adaptive grid")

        # Determine output resolution (finest level)
        max_level = max(cell.level for cell in final_grid)
        output_resolution = self.config.base_resolution / (
            self.config.refinement_factor**max_level
        )

        # Calculate output dimensions
        roi_bounds = roi_polygon.bounds

        width = int((roi_bounds[2] - roi_bounds[0]) / (output_resolution / 111000))
        height = int((roi_bounds[3] - roi_bounds[1]) / (output_resolution / 111000))

        # Create transform
        transform = from_bounds(
            roi_bounds[0], roi_bounds[1], roi_bounds[2], roi_bounds[3], width, height
        )

        # Initialize density array
        density_array = np.full((height, width), -9999.0, dtype=np.float32)

        # Rasterize each grid cell
        for cell in final_grid:
            cell_mask = rasterize(
                [(cell.geometry, cell.density)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.float32,
            )

            # Update density array (use maximum value where cells overlap)
            density_array = np.maximum(density_array, cell_mask)

        # Apply ROI mask
        roi_mask = rasterize(
            [(roi_polygon, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        density_array = np.where(roi_mask, density_array, -9999.0)

        # Calculate statistics
        valid_data = density_array[density_array != -9999.0]

        if len(valid_data) > 0:
            stats = {
                "count": int(len(valid_data)),
                "min": float(np.min(valid_data)),
                "max": float(np.max(valid_data)),
                "mean": float(np.mean(valid_data)),
                "std": float(np.std(valid_data)),
                "total_scenes": int(np.sum(valid_data)),
            }
        else:
            stats = {"error": "No valid data"}

        # Grid hierarchy statistics
        level_stats = {}
        for level in range(max_level + 1):
            level_cells = [cell for cell in final_grid if cell.level == level]
            level_stats[f"level_{level}"] = {
                "cell_count": len(level_cells),
                "resolution_m": self.config.base_resolution
                / (self.config.refinement_factor**level),
                "avg_density": (
                    np.mean([cell.density for cell in level_cells])
                    if level_cells
                    else 0
                ),
            }

        return {
            "density_array": density_array,
            "transform": transform,
            "crs": "EPSG:4326",
            "bounds": roi_bounds,
            "stats": stats,
            "grid_info": {
                "width": width,
                "height": height,
                "output_resolution": output_resolution,
                "max_level": max_level,
                "total_cells": len(final_grid),
                "level_statistics": level_stats,
            },
            "adaptive_grid": final_grid,  # Include grid for analysis
        }

    def visualize_grid_hierarchy(self, grid_cells: List[GridCell]) -> Dict[str, Any]:
        """Generate visualization data for grid hierarchy."""

        # Group cells by level
        levels = {}
        for cell in grid_cells:
            if cell.level not in levels:
                levels[cell.level] = []
            levels[cell.level].append(cell)

        # Generate GeoJSON for each level
        level_geojson = {}
        for level, cells in levels.items():
            features = []
            for cell in cells:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [list(cell.geometry.exterior.coords)],
                    },
                    "properties": {
                        "level": cell.level,
                        "density": cell.density,
                        "variance": cell.variance,
                        "scene_count": cell.scene_count,
                        "needs_refinement": cell.needs_refinement,
                    },
                }
                features.append(feature)

            level_geojson[f"level_{level}"] = {
                "type": "FeatureCollection",
                "features": features,
            }

        return {
            "grid_levels": level_geojson,
            "summary": {
                "total_levels": len(levels),
                "total_cells": len(grid_cells),
                "cells_per_level": {f"level_{k}": len(v) for k, v in levels.items()},
            },
        }


# Integration with main density engine
def integrate_adaptive_grid(density_engine):
    """Integrate adaptive grid method into main density engine."""

    def _calculate_adaptive_grid_density(
        self, scene_polygons, roi_poly, config, start_time
    ):
        """Enhanced adaptive grid calculation for density engine."""

        # Create adaptive grid configuration
        adaptive_config = AdaptiveGridConfig(
            base_resolution=config.resolution * 4,  # Start 4x coarser
            min_resolution=config.resolution,  # Target resolution
            max_resolution=config.resolution * 16,  # Max 16x coarser
            refinement_factor=2,
            max_levels=3,
            density_threshold=5.0,
            variance_threshold=2.0,
        )

        # Initialize adaptive grid engine
        adaptive_engine = AdaptiveGridEngine(adaptive_config)

        # Calculate adaptive density
        adaptive_result = adaptive_engine.calculate_adaptive_density(
            scene_polygons, roi_poly
        )

        # Convert to standard DensityResult format
        from .density_engine import DensityResult, DensityMethod

        return DensityResult(
            density_array=adaptive_result["density_array"],
            transform=adaptive_result["transform"],
            crs=adaptive_result["crs"],
            bounds=adaptive_result["bounds"],
            stats=adaptive_result["stats"],
            computation_time=adaptive_result["computation_time"],
            method_used=DensityMethod.ADAPTIVE_GRID,
            grid_info=adaptive_result["grid_info"],
        )

    # Replace the placeholder method
    density_engine._calculate_adaptive_grid_density = (
        _calculate_adaptive_grid_density.__get__(density_engine)
    )


# Example usage
if __name__ == "__main__":
    # Test adaptive grid with mock data
    from shapely.geometry import box
    import random

    # Create test ROI
    test_roi = box(9.0, 45.0, 9.5, 45.5)  # Large area around Milan

    # Create mock scenes with clustering
    scenes = []

    # Dense cluster in center
    for _ in range(50):
        x = random.uniform(9.2, 9.3)
        y = random.uniform(45.2, 45.3)
        scene = box(x - 0.01, y - 0.01, x + 0.01, y + 0.01)
        scenes.append(scene)

    # Sparse scenes elsewhere
    for _ in range(20):
        x = random.uniform(9.0, 9.5)
        y = random.uniform(45.0, 45.5)
        scene = box(x - 0.005, y - 0.005, x + 0.005, y + 0.005)
        scenes.append(scene)

    # Test adaptive grid
    engine = AdaptiveGridEngine()

    try:
        result = engine.calculate_adaptive_density(scenes, test_roi)

        print("Adaptive Grid Test Results:")
        print(f"Computation time: {result['computation_time']:.2f}s")
        print(f"Output resolution: {result['grid_info']['output_resolution']:.1f}m")
        print(f"Max refinement level: {result['grid_info']['max_level']}")
        print(f"Total grid cells: {result['grid_info']['total_cells']}")
        print(
            f"Density stats: min={result['stats']['min']}, max={result['stats']['max']}"
        )

        # Show level statistics
        for level, stats in result["grid_info"]["level_statistics"].items():
            print(
                f"{level}: {stats['cell_count']} cells at {stats['resolution_m']:.1f}m resolution"
            )

    except Exception as e:
        print(f"Adaptive grid test failed: {e}")
        import traceback

        traceback.print_exc()
