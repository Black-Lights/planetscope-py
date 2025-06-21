#!/usr/bin/env python3
"""
PlanetScope-py Phase 3: Basic Density Visualization
Simple visualization capabilities for immediate feedback on density calculations.

This module provides basic plotting and visualization for density results to enable
immediate validation of spatial analysis results. More advanced interactive
visualization will be implemented in Phase 4.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MPLPolygon
import rasterio
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import Polygon
import contextily as ctx

logger = logging.getLogger(__name__)


class DensityVisualizer:
    """
    Basic visualization for spatial density results.

    Provides simple plotting capabilities for density maps, histograms,
    and scene footprint overlays for immediate validation and analysis.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer.

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.default_cmap = "viridis"

        # Set up matplotlib for better defaults
        plt.style.use("default")
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["font.size"] = 10

    def plot_density_map(
        self,
        density_result,
        title: str = "Scene Density Map",
        colormap: str = None,
        save_path: Optional[str] = None,
        show_stats: bool = True,
    ) -> plt.Figure:
        """
        Plot density map with optional statistics.

        Args:
            density_result: DensityResult object from density calculation
            title: Plot title
            colormap: Matplotlib colormap name
            save_path: Optional path to save plot
            show_stats: Whether to display statistics

        Returns:
            Matplotlib figure object
        """
        colormap = colormap or self.default_cmap

        fig, ax = plt.subplots(figsize=self.figsize)

        # Get density array and mask no-data values
        density_array = density_result.density_array
        no_data_value = getattr(density_result, "no_data_value", -9999.0)

        # Mask no-data values
        masked_array = np.ma.masked_equal(density_array, no_data_value)

        # Create extent for proper geographic plotting
        bounds = density_result.bounds
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

        # Plot density map
        im = ax.imshow(
            masked_array,
            extent=extent,
            cmap=colormap,
            origin="lower",
            interpolation="nearest",
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Scene Count", rotation=270, labelpad=20)

        # Set labels and title
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)

        # Add grid
        ax.grid(True, alpha=0.3)

        # Add statistics text if requested
        if show_stats and hasattr(density_result, "stats"):
            stats = density_result.stats
            if "error" not in stats:
                stats_text = (
                    f"Min: {stats['min']:.1f}\n"
                    f"Max: {stats['max']:.1f}\n"
                    f"Mean: {stats['mean']:.1f}\n"
                    f"Std: {stats['std']:.1f}"
                )

                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    fontsize=9,
                )

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Density map saved to {save_path}")

        return fig

    def plot_density_histogram(
        self,
        density_result,
        bins: int = 20,
        title: str = "Density Distribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot histogram of density values.

        Args:
            density_result: DensityResult object
            bins: Number of histogram bins
            title: Plot title
            save_path: Optional path to save plot

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Get valid density values
        density_array = density_result.density_array
        no_data_value = getattr(density_result, "no_data_value", -9999.0)
        valid_data = density_array[density_array != no_data_value]

        if len(valid_data) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid data to plot",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title(title)
            return fig

        # Plot histogram
        counts, bin_edges, patches = ax.hist(
            valid_data, bins=bins, alpha=0.7, edgecolor="black"
        )

        # Color bars based on values
        for i, (count, patch) in enumerate(zip(counts, patches)):
            patch.set_facecolor(plt.cm.viridis(i / len(patches)))

        # Add statistics
        mean_val = np.mean(valid_data)
        median_val = np.median(valid_data)

        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.1f}",
        )
        ax.axvline(
            median_val,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:.1f}",
        )

        # Labels and formatting
        ax.set_xlabel("Scene Count")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Histogram saved to {save_path}")

        return fig

    def plot_scene_footprints(
        self,
        scene_polygons: List[Polygon],
        roi_polygon: Polygon,
        title: str = "Scene Footprints",
        max_scenes: int = 100,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot scene footprints over ROI.

        Args:
            scene_polygons: List of scene footprint polygons
            roi_polygon: Region of interest polygon
            title: Plot title
            max_scenes: Maximum number of scenes to plot (for performance)
            save_path: Optional path to save plot

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot ROI
        roi_coords = list(roi_polygon.exterior.coords)
        roi_patch = MPLPolygon(
            roi_coords, fill=False, edgecolor="red", linewidth=2, label="ROI"
        )
        ax.add_patch(roi_patch)

        # Sample scenes if too many
        if len(scene_polygons) > max_scenes:
            import random

            scene_sample = random.sample(scene_polygons, max_scenes)
            logger.info(f"Plotting {max_scenes} of {len(scene_polygons)} scenes")
        else:
            scene_sample = scene_polygons

        # Plot scene footprints
        for i, scene_poly in enumerate(scene_sample):
            try:
                coords = list(scene_poly.exterior.coords)
                patch = MPLPolygon(
                    coords, fill=False, edgecolor="blue", alpha=0.6, linewidth=0.5
                )
                ax.add_patch(patch)
            except Exception as e:
                logger.warning(f"Failed to plot scene {i}: {e}")
                continue

        # Set equal aspect and limits
        ax.set_aspect("equal")
        bounds = roi_polygon.bounds
        margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.05
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)

        # Labels and formatting
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{title} ({len(scene_sample)} scenes shown)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Scene footprints plot saved to {save_path}")

        return fig

    def create_summary_plot(
        self,
        density_result,
        scene_polygons: Optional[List[Polygon]] = None,
        roi_polygon: Optional[Polygon] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create multi-panel summary plot.

        Args:
            density_result: DensityResult object
            scene_polygons: Optional scene polygons for footprint plot
            roi_polygon: Optional ROI polygon
            save_path: Optional path to save plot

        Returns:
            Matplotlib figure object
        """
        # Determine layout
        has_footprints = scene_polygons is not None and roi_polygon is not None

        if has_footprints:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Density map
        density_array = density_result.density_array
        no_data_value = getattr(density_result, "no_data_value", -9999.0)
        masked_array = np.ma.masked_equal(density_array, no_data_value)

        bounds = density_result.bounds
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

        im1 = ax1.imshow(
            masked_array,
            extent=extent,
            cmap="viridis",
            origin="lower",
            interpolation="nearest",
        )
        ax1.set_title("Density Map")
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        plt.colorbar(im1, ax=ax1, shrink=0.8, label="Scene Count")

        # 2. Histogram
        valid_data = density_array[density_array != no_data_value]

        if len(valid_data) > 0:
            ax2.hist(valid_data, bins=20, alpha=0.7, edgecolor="black", color="skyblue")
            mean_val = np.mean(valid_data)
            ax2.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_val:.1f}",
            )
            ax2.set_xlabel("Scene Count")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Density Distribution")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                "No valid data",
                transform=ax2.transAxes,
                ha="center",
                va="center",
            )
            ax2.set_title("Density Distribution")

        # 3. Statistics summary
        if hasattr(density_result, "stats") and "error" not in density_result.stats:
            stats = density_result.stats

            # Create statistics table
            stats_data = [
                ["Count", f"{stats['count']:,}"],
                ["Min", f"{stats['min']:.1f}"],
                ["Max", f"{stats['max']:.1f}"],
                ["Mean", f"{stats['mean']:.2f}"],
                ["Std Dev", f"{stats['std']:.2f}"],
                ["Median", f"{stats['median']:.1f}"],
                ["75th %ile", f"{stats['percentiles']['75']:.1f}"],
                ["95th %ile", f"{stats['percentiles']['95']:.1f}"],
            ]

            # Method and timing info
            method_info = [
                [
                    "Method",
                    (
                        getattr(density_result, "method_used", "Unknown").value
                        if hasattr(
                            getattr(density_result, "method_used", None), "value"
                        )
                        else str(getattr(density_result, "method_used", "Unknown"))
                    ),
                ],
                ["Time", f"{getattr(density_result, 'computation_time', 0):.2f}s"],
                ["Resolution", f"{density_result.grid_info.get('resolution', 'N/A')}m"],
                [
                    "Grid Size",
                    f"{density_result.grid_info.get('width', 0)}×{density_result.grid_info.get('height', 0)}",
                ],
            ]

            all_data = stats_data + [["", ""]] + method_info

            # Remove axis ticks and labels
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis("off")

            # Create table
            table_data = []
            colors = []
            for i, (label, value) in enumerate(all_data):
                if label == "":  # Separator row
                    continue
                table_data.append([label, value])
                if i < len(stats_data):
                    colors.append(["lightblue", "white"])
                else:
                    colors.append(["lightgreen", "white"])

            table = ax3.table(
                cellText=table_data,
                cellColours=colors,
                cellLoc="left",
                loc="center",
                colWidths=[0.4, 0.4],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            ax3.set_title("Statistics Summary")
        else:
            ax3.text(
                0.5,
                0.5,
                "No statistics available",
                transform=ax3.transAxes,
                ha="center",
                va="center",
            )
            ax3.set_title("Statistics Summary")
            ax3.axis("off")

        # 4. Scene footprints (if available)
        if has_footprints:
            # Sample scenes for performance
            max_scenes = 50
            if len(scene_polygons) > max_scenes:
                import random

                scene_sample = random.sample(scene_polygons, max_scenes)
            else:
                scene_sample = scene_polygons

            # Plot ROI
            roi_coords = list(roi_polygon.exterior.coords)
            roi_patch = MPLPolygon(
                roi_coords, fill=False, edgecolor="red", linewidth=2, label="ROI"
            )
            ax4.add_patch(roi_patch)

            # Plot scenes
            for scene_poly in scene_sample:
                try:
                    coords = list(scene_poly.exterior.coords)
                    patch = MPLPolygon(
                        coords, fill=False, edgecolor="blue", alpha=0.6, linewidth=0.5
                    )
                    ax4.add_patch(patch)
                except:
                    continue

            # Set limits and formatting
            ax4.set_aspect("equal")
            bounds_roi = roi_polygon.bounds
            margin = (
                max(bounds_roi[2] - bounds_roi[0], bounds_roi[3] - bounds_roi[1]) * 0.05
            )
            ax4.set_xlim(bounds_roi[0] - margin, bounds_roi[2] + margin)
            ax4.set_ylim(bounds_roi[1] - margin, bounds_roi[3] + margin)
            ax4.set_xlabel("Longitude")
            ax4.set_ylabel("Latitude")
            ax4.set_title(f"Scene Footprints ({len(scene_sample)} shown)")
            ax4.grid(True, alpha=0.3)
            ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Summary plot saved to {save_path}")

        return fig

    def export_density_geotiff_with_style(
        self, density_result, output_path: str, colormap: str = "viridis"
    ) -> None:
        """
        Export density as GeoTIFF with accompanying style file.

        Args:
            density_result: DensityResult object
            output_path: Output path for GeoTIFF
            colormap: Colormap for styling
        """
        # Determine CRS to use (handle PROJ issues gracefully)
        crs_to_use = self._get_safe_crs(density_result.crs)

        # Export basic GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=density_result.density_array.shape[0],
            width=density_result.density_array.shape[1],
            count=1,
            dtype=density_result.density_array.dtype,
            crs=crs_to_use,
            transform=density_result.transform,
            compress="lzw",
            nodata=-9999.0,
        ) as dst:
            dst.write(density_result.density_array, 1)

            # Add metadata
            dst.update_tags(
                title="PlanetScope Scene Density",
                description="Scene overlap density calculation",
                method=(
                    getattr(density_result, "method_used", "unknown").value
                    if hasattr(getattr(density_result, "method_used", None), "value")
                    else str(getattr(density_result, "method_used", "unknown"))
                ),
                resolution=str(density_result.grid_info.get("resolution", "unknown")),
                computation_time=str(getattr(density_result, "computation_time", 0)),
            )

        # Create QGIS style file (.qml)
        qml_path = output_path.replace(".tif", ".qml")
        self._create_qgis_style_file(density_result, qml_path, colormap)

        logger.info(f"Density GeoTIFF exported to {output_path}")
        logger.info(f"QGIS style file created: {qml_path}")

    def _get_safe_crs(self, preferred_crs: str) -> str:
        """Get a safe CRS that works with current PROJ installation."""
        try:
            # Try preferred CRS first
            import rasterio.crs

            crs_obj = rasterio.crs.CRS.from_string(preferred_crs)
            return preferred_crs
        except Exception:
            # Fall back to PROJ4 string for WGS84
            try:
                fallback_crs = "+proj=longlat +datum=WGS84 +no_defs"
                crs_obj = rasterio.crs.CRS.from_string(fallback_crs)
                logger.warning(
                    f"CRS {preferred_crs} failed, using fallback: {fallback_crs}"
                )
                return fallback_crs
            except Exception:
                # Final fallback - no CRS
                logger.warning(
                    "CRS initialization failed, creating GeoTIFF without CRS"
                )
                return None

    def _create_qgis_style_file(
        self, density_result, qml_path: str, colormap: str
    ) -> None:
        """Create QGIS style file for density raster."""

        # Get data range for color ramp
        density_array = density_result.density_array
        no_data_value = getattr(density_result, "no_data_value", -9999.0)
        valid_data = density_array[density_array != no_data_value]

        if len(valid_data) == 0:
            min_val, max_val = 0, 1
        else:
            min_val = float(np.min(valid_data))
            max_val = float(np.max(valid_data))

        # Create QML content
        qml_content = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22" styleCategories="AllStyleCategories">
  <pipe>
    <provider>
      <resampling enabled="false" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer alphaBand="-1" type="singlebandpseudocolor" band="1" nodataColor="" opacity="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" minimumValue="{min_val}" maximumValue="{max_val}" clip="0">
          <colorramp type="gradient" name="[source]">
            <Option type="Map">
              <Option type="QString" name="color1" value="68,1,84,255"/>
              <Option type="QString" name="color2" value="253,231,37,255"/>
              <Option type="QString" name="direction" value="ccw"/>
              <Option type="QString" name="discrete" value="0"/>
              <Option type="QString" name="rampType" value="gradient"/>
              <Option type="QString" name="spec" value="rgb"/>
              <Option type="QString" name="stops" value="0.25;59,82,139,255:0.5;33,145,140,255:0.75;94,201,98,255"/>
            </Option>
          </colorramp>
          <item alpha="255" value="{min_val}" label="{min_val:.1f}" color="68,1,84,255"/>
          <item alpha="255" value="{min_val + (max_val-min_val)*0.25}" label="{min_val + (max_val-min_val)*0.25:.1f}" color="59,82,139,255"/>
          <item alpha="255" value="{min_val + (max_val-min_val)*0.5}" label="{min_val + (max_val-min_val)*0.5:.1f}" color="33,145,140,255"/>
          <item alpha="255" value="{min_val + (max_val-min_val)*0.75}" label="{min_val + (max_val-min_val)*0.75:.1f}" color="94,201,98,255"/>
          <item alpha="255" value="{max_val}" label="{max_val:.1f}" color="253,231,37,255"/>
          <rampLegendSettings minimumLabel="" maximumLabel="" prefix="" suffix="" direction="0" useContinuousLegend="1" orientation="2">
            <numericFormat id="basic">
              <Option type="Map">
                <Option type="QChar" name="decimal_separator" value=""/>
                <Option type="int" name="decimals" value="6"/>
                <Option type="int" name="rounding_type" value="0"/>
                <Option type="bool" name="show_plus" value="false"/>
                <Option type="bool" name="show_thousand_separator" value="true"/>
                <Option type="bool" name="show_trailing_zeros" value="false"/>
                <Option type="QChar" name="thousand_separator" value=""/>
              </Option>
            </numericFormat>
          </rampLegendSettings>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" gamma="1" contrast="0"/>
    <huesaturation colorizeGreen="128" colorizeOn="0" colorizeRed="255" colorizeBlue="128" grayscaleMode="0" saturation="0" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <layerGeometry>
    <wkbType>0</wkbType>
  </layerGeometry>
</qgis>"""

        with open(qml_path, "w") as f:
            f.write(qml_content)


# Integration with density engine
def integrate_visualization(density_engine):
    """Add basic visualization capabilities to density engine."""

    visualizer = DensityVisualizer()
    density_engine.visualizer = visualizer

    def plot_result(self, result, output_dir=None, show_plots=True):
        """Plot density calculation results."""
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create summary plot
        summary_path = (
            os.path.join(output_dir, "density_summary.png") if output_dir else None
        )
        fig = visualizer.create_summary_plot(result, save_path=summary_path)

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        # Export styled GeoTIFF
        if output_dir:
            geotiff_path = os.path.join(output_dir, "density_map.tif")
            visualizer.export_density_geotiff_with_style(result, geotiff_path)

        return fig

    # Add method to density engine
    density_engine.plot_result = plot_result.__get__(density_engine)


# Example usage
if __name__ == "__main__":
    # Test visualization with mock data
    import numpy as np
    from dataclasses import dataclass
    from shapely.geometry import box

    # Create mock density result
    @dataclass
    class MockDensityResult:
        density_array: np.ndarray
        transform: Any
        crs: str
        bounds: tuple
        stats: dict
        computation_time: float
        method_used: str
        grid_info: dict
        no_data_value: float = -9999.0

    # Create test data
    width, height = 100, 80
    test_array = np.random.poisson(3, (height, width)).astype(np.float32)
    test_array[:10, :] = -9999.0  # No data region

    from rasterio.transform import from_bounds

    bounds = (9.0, 45.0, 9.2, 45.1)
    transform = from_bounds(*bounds, width, height)

    # Calculate stats
    valid_data = test_array[test_array != -9999.0]
    stats = {
        "count": len(valid_data),
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "mean": float(np.mean(valid_data)),
        "std": float(np.std(valid_data)),
        "median": float(np.median(valid_data)),
        "percentiles": {
            "25": float(np.percentile(valid_data, 25)),
            "75": float(np.percentile(valid_data, 75)),
            "90": float(np.percentile(valid_data, 90)),
            "95": float(np.percentile(valid_data, 95)),
        },
    }

    mock_result = MockDensityResult(
        density_array=test_array,
        transform=transform,
        crs="EPSG:4326",
        bounds=bounds,
        stats=stats,
        computation_time=1.5,
        method_used="test_method",
        grid_info={"width": width, "height": height, "resolution": 10},
    )

    # Test visualizer
    visualizer = DensityVisualizer()

    try:
        print("Testing density visualization...")

        # Test density map
        fig1 = visualizer.plot_density_map(mock_result, title="Test Density Map")
        print("✓ Density map plot created")

        # Test histogram
        fig2 = visualizer.plot_density_histogram(mock_result, title="Test Histogram")
        print("✓ Histogram plot created")

        # Test scene footprints
        test_roi = box(*bounds)
        test_scenes = [
            box(9.05 + i * 0.02, 45.02 + j * 0.015, 9.07 + i * 0.02, 45.035 + j * 0.015)
            for i in range(5)
            for j in range(3)
        ]

        fig3 = visualizer.plot_scene_footprints(
            test_scenes, test_roi, title="Test Footprints"
        )
        print("✓ Scene footprints plot created")

        # Test summary plot
        fig4 = visualizer.create_summary_plot(mock_result, test_scenes, test_roi)
        print("✓ Summary plot created")

        # Close all figures
        plt.close("all")

        print("All visualization tests passed!")

    except Exception as e:
        print(f"Visualization test failed: {e}")
        import traceback

        traceback.print_exc()
