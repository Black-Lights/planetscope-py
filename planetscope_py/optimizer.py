#!/usr/bin/env python3
"""
PlanetScope-py Phase 3: Performance Optimizer Module
Intelligent method selection and performance optimization for spatial density calculations.

This module provides automatic algorithm selection based on data characteristics,
performance estimation, memory optimization, and adaptive parameter tuning.
"""

import logging
import time
import psutil
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from shapely.geometry import Polygon
import math

from .exceptions import ValidationError, PlanetScopeError

logger = logging.getLogger(__name__)


class PerformanceProfile(Enum):
    """Performance optimization profiles."""

    SPEED = "speed"  # Optimize for fastest computation
    MEMORY = "memory"  # Optimize for minimal memory usage
    ACCURACY = "accuracy"  # Optimize for highest accuracy
    BALANCED = "balanced"  # Balance all factors
    AUTO = "auto"  # Automatic selection


class SystemResources(Enum):
    """System resource levels."""

    LOW = "low"  # Limited resources
    MEDIUM = "medium"  # Standard resources
    HIGH = "high"  # High-performance system
    UNKNOWN = "unknown"  # Unable to determine


@dataclass
class PerformanceEstimate:
    """Performance estimation for different methods."""

    method: str
    estimated_time: float  # Seconds
    estimated_memory: float  # MB
    accuracy_score: float  # 0.0-1.0
    confidence: float  # Estimation confidence 0.0-1.0
    recommended: bool = False


@dataclass
class SystemCapabilities:
    """Current system resource information."""

    available_memory_gb: float
    cpu_cores: int
    memory_usage_percent: float
    resource_level: SystemResources


@dataclass
class DatasetCharacteristics:
    """Characteristics of input dataset."""

    roi_area_km2: float
    scene_count: int
    scene_density_per_km2: float
    roi_complexity: float  # Geometry complexity metric
    target_resolution: float  # Meters
    estimated_raster_size_mb: float


class PerformanceOptimizer:
    """
    Performance optimizer for spatial density calculations.

    Analyzes dataset characteristics and system resources to automatically
    select the optimal computational method and parameters for best performance
    while maintaining accuracy requirements.
    """

    def __init__(self):
        """Initialize performance optimizer."""
        self.system_info = self._analyze_system_resources()
        self.method_benchmarks = self._load_method_benchmarks()
        self.optimization_history = []

        logger.info(
            f"Performance optimizer initialized - {self.system_info.resource_level.value} resource system"
        )

    def _analyze_system_resources(self) -> SystemCapabilities:
        """Analyze current system resources and capabilities."""
        try:
            # Get memory information
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
            memory_usage_percent = memory.percent

            # Get CPU information
            cpu_cores = psutil.cpu_count(logical=False) or 1

            # Determine resource level
            if available_memory_gb >= 16 and cpu_cores >= 8:
                resource_level = SystemResources.HIGH
            elif available_memory_gb >= 8 and cpu_cores >= 4:
                resource_level = SystemResources.MEDIUM
            elif available_memory_gb >= 4:
                resource_level = SystemResources.LOW
            else:
                resource_level = SystemResources.UNKNOWN

            return SystemCapabilities(
                available_memory_gb=available_memory_gb,
                cpu_cores=cpu_cores,
                memory_usage_percent=memory_usage_percent,
                resource_level=resource_level,
            )

        except Exception as e:
            logger.warning(f"Failed to analyze system resources: {e}")
            return SystemCapabilities(
                available_memory_gb=8.0,  # Conservative default
                cpu_cores=4,
                memory_usage_percent=50.0,
                resource_level=SystemResources.UNKNOWN,
            )

    def _load_method_benchmarks(self) -> Dict[str, Dict]:
        """Load performance benchmarks for different methods."""

        # Empirical benchmarks based on testing
        # These would be updated based on real performance data
        return {
            "rasterization": {
                "base_time_per_scene": 0.001,  # seconds per scene
                "memory_per_megapixel": 4.0,  # MB per megapixel
                "accuracy_score": 0.95,
                "scales_with_scenes": "linear",
                "scales_with_pixels": "linear",
                "best_for": ["high_scene_count", "standard_roi"],
            },
            "vector_overlay": {
                "base_time_per_scene": 0.005,  # seconds per scene
                "memory_per_megapixel": 2.0,  # MB per megapixel
                "accuracy_score": 1.0,  # Highest accuracy
                "scales_with_scenes": "quadratic",  # Spatial index overhead
                "scales_with_pixels": "linear",
                "best_for": ["low_scene_count", "high_accuracy"],
            },
            "adaptive_grid": {
                "base_time_per_scene": 0.002,  # seconds per scene
                "memory_per_megapixel": 3.0,  # MB per megapixel
                "accuracy_score": 0.90,  # Good accuracy
                "scales_with_scenes": "logarithmic",  # Adaptive scaling
                "scales_with_pixels": "sublinear",  # Grid refinement
                "best_for": ["large_roi", "variable_density"],
            },
        }

    def analyze_dataset(
        self,
        scene_footprints: List[Dict],
        roi_geometry: Union[Dict, Polygon],
        resolution: float,
    ) -> DatasetCharacteristics:
        """Analyze dataset characteristics for optimization."""

        # Convert ROI to polygon if needed
        if isinstance(roi_geometry, dict):
            from shapely.geometry import shape

            roi_poly = shape(roi_geometry)
        else:
            roi_poly = roi_geometry

        # Calculate ROI area (rough conversion from degrees to km²)
        roi_area_km2 = roi_poly.area * (111.0**2)

        # Count valid scenes
        scene_count = len(scene_footprints)

        # Calculate scene density
        scene_density_per_km2 = scene_count / roi_area_km2 if roi_area_km2 > 0 else 0

        # Estimate ROI complexity (number of vertices as proxy)
        roi_complexity = len(list(roi_poly.exterior.coords))

        # Estimate raster size
        bounds = roi_poly.bounds
        width_pixels = int((bounds[2] - bounds[0]) / (resolution / 111000))
        height_pixels = int((bounds[3] - bounds[1]) / (resolution / 111000))
        total_pixels = width_pixels * height_pixels
        estimated_raster_size_mb = (total_pixels * 4) / (1024**2)  # 4 bytes per float32

        characteristics = DatasetCharacteristics(
            roi_area_km2=roi_area_km2,
            scene_count=scene_count,
            scene_density_per_km2=scene_density_per_km2,
            roi_complexity=roi_complexity,
            target_resolution=resolution,
            estimated_raster_size_mb=estimated_raster_size_mb,
        )

        logger.info(
            f"Dataset analysis: {roi_area_km2:.1f} km², {scene_count} scenes, "
            f"{scene_density_per_km2:.1f} scenes/km², {estimated_raster_size_mb:.1f} MB raster"
        )

        return characteristics

    def estimate_method_performance(
        self, method: str, dataset: DatasetCharacteristics
    ) -> PerformanceEstimate:
        """Estimate performance for a specific method."""

        if method not in self.method_benchmarks:
            raise ValidationError(f"Unknown method: {method}")

        benchmarks = self.method_benchmarks[method]

        # Estimate computation time
        base_time = benchmarks["base_time_per_scene"] * dataset.scene_count

        # Apply scaling factors
        scaling = benchmarks["scales_with_scenes"]
        if scaling == "linear":
            time_factor = 1.0
        elif scaling == "quadratic":
            time_factor = max(1.0, dataset.scene_count / 100)  # Penalty for many scenes
        elif scaling == "logarithmic":
            time_factor = max(0.5, math.log10(max(1, dataset.scene_count / 10)))
        else:  # sublinear
            time_factor = max(0.7, math.sqrt(dataset.scene_count / 100))

        # Pixel scaling
        pixel_factor = dataset.estimated_raster_size_mb / 100  # Base 100MB

        estimated_time = base_time * time_factor * pixel_factor

        # Estimate memory usage
        estimated_memory = (
            benchmarks["memory_per_megapixel"] * dataset.estimated_raster_size_mb / 4
        )  # Convert to megapixels

        # Accuracy score from benchmarks
        accuracy_score = benchmarks["accuracy_score"]

        # Confidence based on dataset size (higher confidence for larger datasets)
        confidence = min(0.95, 0.5 + (dataset.scene_count / 1000) * 0.45)

        return PerformanceEstimate(
            method=method,
            estimated_time=estimated_time,
            estimated_memory=estimated_memory,
            accuracy_score=accuracy_score,
            confidence=confidence,
        )

    def recommend_method(
        self,
        dataset: DatasetCharacteristics,
        profile: PerformanceProfile = PerformanceProfile.BALANCED,
        constraints: Optional[Dict] = None,
    ) -> Tuple[str, PerformanceEstimate]:
        """
        Recommend optimal method based on dataset and performance profile.

        Args:
            dataset: Dataset characteristics
            profile: Performance optimization profile
            constraints: Optional constraints (max_time, max_memory, min_accuracy)

        Returns:
            Tuple of (method_name, performance_estimate)
        """
        constraints = constraints or {}

        # Get estimates for all methods
        estimates = {}
        for method in self.method_benchmarks.keys():
            try:
                estimates[method] = self.estimate_method_performance(method, dataset)
            except Exception as e:
                logger.warning(f"Failed to estimate performance for {method}: {e}")
                continue

        if not estimates:
            raise PlanetScopeError("No valid method estimates available")

        # Apply constraints
        valid_estimates = self._apply_constraints(estimates, constraints)

        if not valid_estimates:
            logger.warning("No methods meet constraints, using best available")
            valid_estimates = estimates

        # Select method based on profile
        if profile == PerformanceProfile.SPEED:
            best_method = min(
                valid_estimates.keys(), key=lambda m: valid_estimates[m].estimated_time
            )
        elif profile == PerformanceProfile.MEMORY:
            best_method = min(
                valid_estimates.keys(),
                key=lambda m: valid_estimates[m].estimated_memory,
            )
        elif profile == PerformanceProfile.ACCURACY:
            best_method = max(
                valid_estimates.keys(), key=lambda m: valid_estimates[m].accuracy_score
            )
        elif profile == PerformanceProfile.AUTO:
            best_method = self._auto_select_method(dataset, valid_estimates)
        else:  # BALANCED
            best_method = self._balanced_method_selection(valid_estimates)

        # Mark as recommended
        valid_estimates[best_method].recommended = True

        logger.info(
            f"Recommended method: {best_method} "
            f"(time: {valid_estimates[best_method].estimated_time:.2f}s, "
            f"memory: {valid_estimates[best_method].estimated_memory:.1f}MB)"
        )

        return best_method, valid_estimates[best_method]

    def _apply_constraints(
        self, estimates: Dict[str, PerformanceEstimate], constraints: Dict
    ) -> Dict[str, PerformanceEstimate]:
        """Filter estimates based on constraints."""
        valid = {}

        for method, estimate in estimates.items():
            valid_method = True

            # Check time constraint
            if "max_time" in constraints:
                if estimate.estimated_time > constraints["max_time"]:
                    logger.debug(
                        f"{method} exceeds time constraint: "
                        f"{estimate.estimated_time:.2f}s > {constraints['max_time']}s"
                    )
                    valid_method = False

            # Check memory constraint
            if "max_memory" in constraints:
                if estimate.estimated_memory > constraints["max_memory"]:
                    logger.debug(
                        f"{method} exceeds memory constraint: "
                        f"{estimate.estimated_memory:.1f}MB > {constraints['max_memory']}MB"
                    )
                    valid_method = False

            # Check accuracy constraint
            if "min_accuracy" in constraints:
                if estimate.accuracy_score < constraints["min_accuracy"]:
                    logger.debug(
                        f"{method} below accuracy constraint: "
                        f"{estimate.accuracy_score:.2f} < {constraints['min_accuracy']}"
                    )
                    valid_method = False

            if valid_method:
                valid[method] = estimate

        return valid

    def _auto_select_method(
        self, dataset: DatasetCharacteristics, estimates: Dict[str, PerformanceEstimate]
    ) -> str:
        """Automatic method selection based on dataset characteristics."""

        # Rule-based selection
        if (
            dataset.estimated_raster_size_mb
            > self.system_info.available_memory_gb * 1024 * 0.3
        ):
            # Large raster - prefer adaptive grid or memory-efficient method
            memory_methods = sorted(
                estimates.keys(), key=lambda m: estimates[m].estimated_memory
            )
            if "adaptive_grid" in memory_methods:
                return "adaptive_grid"
            return memory_methods[0]

        elif dataset.scene_count > 1000:
            # Many scenes - prefer rasterization
            if "rasterization" in estimates:
                return "rasterization"

        elif dataset.scene_count < 50:
            # Few scenes - prefer vector overlay for accuracy
            if "vector_overlay" in estimates:
                return "vector_overlay"

        # Default to balanced selection
        return self._balanced_method_selection(estimates)

    def _balanced_method_selection(
        self, estimates: Dict[str, PerformanceEstimate]
    ) -> str:
        """Select method using balanced scoring."""

        # Normalize metrics for scoring
        times = [est.estimated_time for est in estimates.values()]
        memories = [est.estimated_memory for est in estimates.values()]
        accuracies = [est.accuracy_score for est in estimates.values()]

        max_time = max(times) if times else 1.0
        max_memory = max(memories) if memories else 1.0
        min_accuracy = min(accuracies) if accuracies else 0.0
        accuracy_range = max(accuracies) - min_accuracy if accuracies else 1.0

        best_method = None
        best_score = -float("inf")

        for method, estimate in estimates.items():
            # Normalize scores (0-1, higher is better)
            time_score = (
                1.0 - (estimate.estimated_time / max_time) if max_time > 0 else 1.0
            )
            memory_score = (
                1.0 - (estimate.estimated_memory / max_memory)
                if max_memory > 0
                else 1.0
            )
            accuracy_score = (
                (estimate.accuracy_score - min_accuracy) / accuracy_range
                if accuracy_range > 0
                else 1.0
            )

            # Weighted combination (adjust weights as needed)
            combined_score = (
                0.3 * time_score + 0.3 * memory_score + 0.4 * accuracy_score
            )

            if combined_score > best_score:
                best_score = combined_score
                best_method = method

        return best_method or list(estimates.keys())[0]

    def optimize_parameters(
        self, method: str, dataset: DatasetCharacteristics
    ) -> Dict[str, Any]:
        """Optimize parameters for selected method."""

        optimized_params = {}

        if method == "rasterization":
            optimized_params = self._optimize_rasterization_params(dataset)
        elif method == "vector_overlay":
            optimized_params = self._optimize_vector_overlay_params(dataset)
        elif method == "adaptive_grid":
            optimized_params = self._optimize_adaptive_grid_params(dataset)

        logger.info(f"Optimized parameters for {method}: {optimized_params}")
        return optimized_params

    def _optimize_rasterization_params(
        self, dataset: DatasetCharacteristics
    ) -> Dict[str, Any]:
        """Optimize parameters for rasterization method."""
        params = {}

        # Adjust chunk size based on memory
        available_memory_mb = (
            self.system_info.available_memory_gb * 1024 * 0.5
        )  # Use half

        if dataset.estimated_raster_size_mb > available_memory_mb:
            # Need chunking
            chunk_factor = math.ceil(
                dataset.estimated_raster_size_mb / available_memory_mb
            )
            chunk_size_km = dataset.roi_area_km2**0.5 / chunk_factor
            params["chunk_size_km"] = max(10.0, chunk_size_km)  # Minimum 10km chunks

        # Parallel processing
        params["parallel_workers"] = min(self.system_info.cpu_cores, 8)

        return params

    def _optimize_vector_overlay_params(
        self, dataset: DatasetCharacteristics
    ) -> Dict[str, Any]:
        """Optimize parameters for vector overlay method."""
        params = {}

        # Spatial index parameters
        if dataset.scene_count > 500:
            params["use_spatial_index"] = True
            params["index_resolution"] = min(dataset.target_resolution * 10, 1000)

        # Batch processing for large datasets
        if dataset.scene_count > 1000:
            params["batch_size"] = 100

        # Memory management
        params["max_memory_gb"] = self.system_info.available_memory_gb * 0.7

        return params

    def _optimize_adaptive_grid_params(
        self, dataset: DatasetCharacteristics
    ) -> Dict[str, Any]:
        """Optimize parameters for adaptive grid method."""
        params = {}

        # Base resolution based on target and dataset size
        if dataset.roi_area_km2 > 1000:  # Large ROI
            params["base_resolution"] = dataset.target_resolution * 8
            params["max_levels"] = 4
        elif dataset.roi_area_km2 > 100:  # Medium ROI
            params["base_resolution"] = dataset.target_resolution * 4
            params["max_levels"] = 3
        else:  # Small ROI
            params["base_resolution"] = dataset.target_resolution * 2
            params["max_levels"] = 2

        # Refinement criteria based on scene density
        if dataset.scene_density_per_km2 > 10:  # High density
            params["density_threshold"] = 8.0
            params["criteria"] = "density_threshold"
        else:  # Low to medium density
            params["density_threshold"] = 3.0
            params["criteria"] = "hybrid"

        return params

    def record_performance(
        self,
        method: str,
        dataset: DatasetCharacteristics,
        actual_time: float,
        actual_memory: float,
        quality_metrics: Optional[Dict] = None,
    ) -> None:
        """Record actual performance for future optimization."""

        performance_record = {
            "timestamp": time.time(),
            "method": method,
            "dataset_size": dataset.scene_count,
            "roi_area_km2": dataset.roi_area_km2,
            "resolution": dataset.target_resolution,
            "estimated_time": 0.0,  # Would come from estimate
            "actual_time": actual_time,
            "estimated_memory": 0.0,  # Would come from estimate
            "actual_memory": actual_memory,
            "quality_metrics": quality_metrics or {},
        }

        self.optimization_history.append(performance_record)

        # Update benchmarks if we have enough data
        if len(self.optimization_history) > 10:
            self._update_benchmarks()

    def _update_benchmarks(self) -> None:
        """Update method benchmarks based on recorded performance."""

        # Group by method
        method_data = {}
        for record in self.optimization_history[-50:]:  # Use recent data
            method = record["method"]
            if method not in method_data:
                method_data[method] = []
            method_data[method].append(record)

        # Update benchmarks
        for method, records in method_data.items():
            if len(records) >= 5 and method in self.method_benchmarks:
                # Calculate average performance per scene
                avg_time_per_scene = np.mean(
                    [
                        r["actual_time"] / r["dataset_size"]
                        for r in records
                        if r["dataset_size"] > 0
                    ]
                )

                # Update benchmark (moving average)
                current = self.method_benchmarks[method]["base_time_per_scene"]
                self.method_benchmarks[method]["base_time_per_scene"] = (
                    0.8 * current + 0.2 * avg_time_per_scene
                )

                logger.debug(
                    f"Updated {method} benchmark: {avg_time_per_scene:.4f}s per scene"
                )

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance analysis report."""

        if not self.optimization_history:
            return {"message": "No performance data available"}

        recent_records = self.optimization_history[-20:]  # Recent performance

        # Method usage statistics
        method_usage = {}
        for record in recent_records:
            method = record["method"]
            method_usage[method] = method_usage.get(method, 0) + 1

        # Average performance by method
        method_performance = {}
        for method in method_usage.keys():
            method_records = [r for r in recent_records if r["method"] == method]
            if method_records:
                avg_time = np.mean([r["actual_time"] for r in method_records])
                avg_memory = np.mean([r["actual_memory"] for r in method_records])

                method_performance[method] = {
                    "usage_count": len(method_records),
                    "avg_time": avg_time,
                    "avg_memory": avg_memory,
                    "efficiency_score": 1.0
                    / (avg_time * avg_memory / 1000),  # Simple efficiency metric
                }

        return {
            "system_info": {
                "available_memory_gb": self.system_info.available_memory_gb,
                "cpu_cores": self.system_info.cpu_cores,
                "resource_level": self.system_info.resource_level.value,
            },
            "method_usage": method_usage,
            "method_performance": method_performance,
            "total_optimizations": len(self.optimization_history),
            "recent_optimizations": len(recent_records),
        }


# Integration function for main density engine
def integrate_optimizer(density_engine):
    """Integrate performance optimizer with density engine."""

    optimizer = PerformanceOptimizer()
    density_engine.optimizer = optimizer

    # Enhanced method selection
    original_select_method = density_engine._select_optimal_method

    def _enhanced_method_selection(self, scene_polygons, roi_poly, config):
        """Enhanced method selection using performance optimizer."""

        # Prepare mock scene footprints for analyzer
        mock_footprints = []
        for poly in scene_polygons:
            mock_footprints.append(
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [list(poly.exterior.coords)],
                    }
                }
            )

        # Analyze dataset
        dataset = optimizer.analyze_dataset(
            mock_footprints, roi_poly, config.resolution
        )

        # Get method recommendation
        method_name, estimate = optimizer.recommend_method(
            dataset,
            profile=PerformanceProfile.BALANCED,
            constraints={
                "max_memory": config.max_memory_gb * 1024,  # Convert to MB
            },
        )

        # Convert method name to enum
        from .density_engine import DensityMethod

        method_mapping = {
            "rasterization": DensityMethod.RASTERIZATION,
            "vector_overlay": DensityMethod.VECTOR_OVERLAY,
            "adaptive_grid": DensityMethod.ADAPTIVE_GRID,
        }

        selected_method = method_mapping.get(method_name, DensityMethod.RASTERIZATION)

        logger.info(
            f"Optimizer selected {selected_method.value} method "
            f"(estimated: {estimate.estimated_time:.2f}s, {estimate.estimated_memory:.1f}MB)"
        )

        return selected_method

    # Replace method selection
    density_engine._select_optimal_method = _enhanced_method_selection.__get__(
        density_engine
    )


# Example usage and testing
if __name__ == "__main__":
    # Test performance optimizer
    from shapely.geometry import box

    # Create mock dataset
    mock_roi = box(9.0, 45.0, 9.2, 45.2)  # Milan area
    mock_scenes = []

    for i in range(100):
        x = 9.0 + (i % 10) * 0.02
        y = 45.0 + (i // 10) * 0.02
        scene = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [(x, y), (x + 0.01, y), (x + 0.01, y + 0.01), (x, y + 0.01), (x, y)]
                ],
            }
        }
        mock_scenes.append(scene)

    # Test optimizer
    optimizer = PerformanceOptimizer()

    try:
        # Analyze dataset
        dataset = optimizer.analyze_dataset(mock_scenes, mock_roi, 30.0)

        print("Dataset Analysis:")
        print(f"ROI area: {dataset.roi_area_km2:.1f} km²")
        print(f"Scene count: {dataset.scene_count}")
        print(f"Scene density: {dataset.scene_density_per_km2:.1f} scenes/km²")
        print(f"Estimated raster size: {dataset.estimated_raster_size_mb:.1f} MB")

        # Get method estimates
        methods = ["rasterization", "vector_overlay", "adaptive_grid"]
        print("\nMethod Performance Estimates:")

        for method in methods:
            estimate = optimizer.estimate_method_performance(method, dataset)
            print(f"{method}:")
            print(f"  Time: {estimate.estimated_time:.2f}s")
            print(f"  Memory: {estimate.estimated_memory:.1f}MB")
            print(f"  Accuracy: {estimate.accuracy_score:.2f}")
            print(f"  Confidence: {estimate.confidence:.2f}")

        # Get recommendation
        recommended_method, best_estimate = optimizer.recommend_method(dataset)
        print(f"\nRecommended: {recommended_method}")
        print(
            f"Performance: {best_estimate.estimated_time:.2f}s, {best_estimate.estimated_memory:.1f}MB"
        )

        # Get optimized parameters
        params = optimizer.optimize_parameters(recommended_method, dataset)
        print(f"Optimized parameters: {params}")

        # System report
        report = optimizer.get_performance_report()
        print(
            f"\nSystem: {report['system_info']['resource_level']} "
            f"({report['system_info']['available_memory_gb']:.1f}GB, "
            f"{report['system_info']['cpu_cores']} cores)"
        )

    except Exception as e:
        print(f"Optimizer test failed: {e}")
        import traceback

        traceback.print_exc()
