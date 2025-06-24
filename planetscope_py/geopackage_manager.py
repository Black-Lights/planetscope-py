#!/usr/bin/env python3
"""
PlanetScope-py Phase 4: GeoPackage Management System
Comprehensive GeoPackage creation and management for scene polygons and imagery.

This module implements professional GeoPackage export capabilities including:
- Scene footprint polygons with comprehensive metadata attributes
- Multiple raster imagery inclusion with ROI clipping support
- Cross-platform compatibility and standardized schemas
- Integration with existing spatial analysis workflows
"""

import logging
import sqlite3
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, shape, box
from shapely.ops import transform
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.mask import mask
import fiona

from .metadata import MetadataProcessor
from .exceptions import ValidationError, PlanetScopeError
from .utils import validate_geometry, calculate_area_km2

logger = logging.getLogger(__name__)


@dataclass
class GeoPackageConfig:
    """Configuration for GeoPackage creation."""

    include_imagery: bool = False
    clip_to_roi: bool = False
    imagery_format: str = "GeoTIFF"  # GeoTIFF or COG
    compression: str = "LZW"
    overview_levels: List[int] = None
    target_crs: str = "EPSG:4326"
    attribute_schema: str = "standard"  # CHANGED: Default to enhanced standard schema
    max_raster_size_mb: int = 100  # Maximum raster size per layer

    def __post_init__(self):
        if self.overview_levels is None:
            self.overview_levels = [2, 4, 8, 16]


@dataclass
class LayerInfo:
    """Information about a GeoPackage layer."""

    name: str
    layer_type: str  # vector, raster
    feature_count: int
    geometry_type: str
    crs: str
    bbox: Tuple[float, float, float, float]
    created: datetime


@dataclass
class RasterInfo:
    """Information about processed raster files."""

    original_path: str
    processed_path: str
    scene_id: str
    asset_type: str
    file_size_mb: float
    width: int
    height: int
    band_count: int
    data_type: str
    crs: str
    bounds: Tuple[float, float, float, float]
    clipped: bool


class GeoPackageManager:
    """
    Comprehensive GeoPackage creation and management system.

    Creates professional GeoPackage files containing PlanetScope scene polygons
    with comprehensive metadata attributes, multiple imagery inclusion, and
    standardized schemas for maximum compatibility.
    """

    def __init__(
        self,
        metadata_processor: Optional[MetadataProcessor] = None,
        config: Optional[GeoPackageConfig] = None,
    ):
        """Initialize GeoPackage manager.

        Args:
            metadata_processor: Metadata processor for scene analysis
            config: GeoPackage creation configuration
        """
        self.metadata_processor = metadata_processor or MetadataProcessor()
        self.config = config or GeoPackageConfig()

        # Attribute schemas
        self.attribute_schemas = {
            "minimal": self._get_minimal_schema(),
            "standard": self._get_standard_schema(),
            "comprehensive": self._get_comprehensive_schema(),
        }

        # Track processed rasters
        self.processed_rasters: List[RasterInfo] = []

        logger.info(
            f"GeoPackageManager initialized with {self.config.attribute_schema} schema"
        )

    def create_scene_geopackage(
        self,
        scenes: List[Dict],
        output_path: str,
        roi: Optional[Polygon] = None,
        downloaded_files: Optional[List[str]] = None,
        layer_name: str = "scene_footprints",
    ) -> str:
        """
        Create comprehensive GeoPackage with scene polygons and metadata.

        Args:
            scenes: List of Planet scene features
            output_path: Path for output GeoPackage file
            roi: Region of interest for clipping/filtering
            downloaded_files: List of downloaded imagery files to include
            layer_name: Name for the scene footprints layer

        Returns:
            Path to created GeoPackage file

        Raises:
            ValidationError: Invalid input parameters
            PlanetScopeError: GeoPackage creation failed
        """
        if not scenes:
            raise ValidationError("No scenes provided for GeoPackage creation")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing file if it exists
        if output_file.exists():
            output_file.unlink()
            logger.info(f"Removed existing GeoPackage: {output_file}")

        logger.info(f"Creating GeoPackage with {len(scenes)} scenes...")

        try:
            # Process scene data
            scene_data = self._process_scenes_for_geopackage(scenes, roi)

            if not scene_data:
                raise PlanetScopeError("No valid scenes remain after processing")

            # Create GeoDataFrame
            gdf = self._create_scene_geodataframe(scene_data)

            # Write scene footprints layer
            self._write_footprints_layer(gdf, output_file, layer_name)

            # Add multiple imagery layers if provided
            if downloaded_files and self.config.include_imagery:
                self._add_multiple_imagery_layers(
                    output_file, downloaded_files, scene_data, roi
                )

            # Add metadata and styling
            self._add_geopackage_metadata(output_file, scenes, roi)
            self._add_layer_styles(output_file, layer_name)

            # Create summary layer
            self._create_summary_layer(output_file, gdf, roi)

            # Validate GeoPackage
            self._validate_geopackage(output_file)

            logger.info(f"GeoPackage created successfully: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Failed to create GeoPackage: {e}")
            # Cleanup partial file
            if output_file.exists():
                output_file.unlink()
            raise PlanetScopeError(f"GeoPackage creation failed: {e}")

    def _process_scenes_for_geopackage(self, scenes: List[Dict], roi: Optional[Polygon] = None) -> List[Dict]:
        """
        FIXED: Process scenes with proper clipping, Planet API metadata, AND centroid calculation.
        """
        processed_scenes = []
        
        logger.info(f"Processing {len(scenes)} scenes with clipping={self.config.clip_to_roi}...")
        
        for i, scene in enumerate(scenes):
            try:
                # Extract metadata using the metadata processor
                metadata = self.metadata_processor.extract_scene_metadata(scene)
                
                # Get scene geometry
                scene_geom = shape(scene.get("geometry", {}))
                original_area = calculate_area_km2(scene_geom)
                
                # CRITICAL FIX: Actually clip geometry if clip_to_roi=True
                if self.config.clip_to_roi and roi:
                    try:
                        intersection = scene_geom.intersection(roi)
                        
                        if intersection.is_empty:
                            continue  # Skip scenes that don't overlap ROI
                        
                        # Use CLIPPED geometry as final geometry
                        if intersection.geom_type in ['Polygon', 'MultiPolygon']:
                            final_geometry = intersection
                            clipped_area = calculate_area_km2(intersection)
                            coverage_percentage = (clipped_area / original_area) * 100 if original_area > 0 else 0
                            aoi_km2 = clipped_area
                        else:
                            continue  # Skip if intersection is just a line/point
                            
                    except Exception as e:
                        logger.warning(f"Clipping error for scene {metadata.get('scene_id', 'unknown')}: {e}")
                        continue
                else:
                    # No clipping - use original geometry
                    final_geometry = scene_geom
                    
                    # Calculate AOI for non-clipped case
                    aoi_km2 = 0.0
                    coverage_percentage = 0.0
                    
                    if roi and scene_geom.is_valid:
                        try:
                            intersection = scene_geom.intersection(roi)
                            if not intersection.is_empty:
                                aoi_km2 = calculate_area_km2(intersection)
                                coverage_percentage = (aoi_km2 / original_area) * 100 if original_area > 0 else 0
                        except Exception as e:
                            logger.warning(f"AOI calculation error: {e}")
                
                # CENTROID FIX: Calculate centroid from final geometry
                try:
                    centroid = final_geometry.centroid
                    centroid_lat = centroid.y
                    centroid_lon = centroid.x
                except Exception as e:
                    logger.warning(f"Centroid calculation error for scene {i}: {e}")
                    centroid_lat = None
                    centroid_lon = None
                
                # Extract Planet API properties directly
                properties = scene.get("properties", {})
                
                # Complete Planet API field mapping
                planet_api_fields = {
                    # Core identification
                    "id": properties.get("id"),
                    "item_type": properties.get("item_type"),
                    "satellite_id": properties.get("satellite_id"),
                    "provider": properties.get("provider"),
                    "platform": properties.get("platform"),
                    "spacecraft_id": properties.get("spacecraft_id"),
                    
                    # Temporal information
                    "acquired": properties.get("acquired"),
                    "published": properties.get("published"),
                    "updated": properties.get("updated"),
                    "publishing_stage": properties.get("publishing_stage"),
                    
                    # Quality and coverage metrics (ALL from your Planet API example)
                    "cloud_cover": properties.get("cloud_cover"),
                    "cloud_percent": properties.get("cloud_percent"),
                    "clear_percent": properties.get("clear_percent"),
                    "clear_confidence_percent": properties.get("clear_confidence_percent"),
                    "visible_percent": properties.get("visible_percent"),
                    "visible_confidence_percent": properties.get("visible_confidence_percent"),
                    "shadow_percent": properties.get("shadow_percent"),
                    "snow_ice_percent": properties.get("snow_ice_percent"),
                    "heavy_haze_percent": properties.get("heavy_haze_percent"),
                    "light_haze_percent": properties.get("light_haze_percent"),
                    "anomalous_pixels": properties.get("anomalous_pixels"),
                    "usable_data": properties.get("usable_data"),
                    "quality_category": properties.get("quality_category"),
                    "black_fill": properties.get("black_fill"),
                    
                    # Technical specifications
                    "gsd": properties.get("gsd"),
                    "epsg_code": properties.get("epsg_code"),
                    "pixel_resolution": properties.get("pixel_resolution"),
                    "processing_level": properties.get("processing_level"),
                    "ground_control": properties.get("ground_control"),
                    
                    # Solar and viewing geometry
                    "sun_azimuth": properties.get("sun_azimuth"),
                    "sun_elevation": properties.get("sun_elevation"),
                    "satellite_azimuth": properties.get("satellite_azimuth"),
                    "view_angle": properties.get("view_angle"),
                    "off_nadir": properties.get("off_nadir"),
                    "azimuth_angle": properties.get("azimuth_angle"),
                    
                    # Technical details
                    "instrument": properties.get("instrument"),
                    "strip_id": properties.get("strip_id"),
                    "radiometric_target": properties.get("radiometric_target"),
                }
                
                # Merge Planet API fields into metadata
                for field_name, field_value in planet_api_fields.items():
                    if field_value is not None:
                        if field_name not in metadata or metadata[field_name] is None:
                            metadata[field_name] = field_value
                
                # Add AOI, coverage, and centroid information
                metadata.update({
                    "aoi_km2": aoi_km2,
                    "coverage_percentage": coverage_percentage,
                    "centroid_lat": centroid_lat,  # FIX: Actually calculate centroid
                    "centroid_lon": centroid_lon,  # FIX: Actually calculate centroid
                    "geometry": final_geometry,
                })
                
                # Ensure scene_id is properly set
                if not metadata.get("scene_id"):
                    metadata["scene_id"] = properties.get("id") or scene.get("id") or f"scene_{i:04d}"
                
                processed_scenes.append(metadata)
                
            except Exception as e:
                logger.error(f"Error processing scene {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_scenes)}/{len(scenes)} scenes")
        return processed_scenes


    def _create_scene_geodataframe(self, scene_data: List[Dict]) -> gpd.GeoDataFrame:
        """Enhanced GeoDataFrame creation with better type handling and validation."""
        if not scene_data:
            raise ValueError("No scene data provided")

        # Get the appropriate attribute schema
        schema = self.attribute_schemas[self.config.attribute_schema]
        
        logger.info(f"Creating GeoDataFrame with {self.config.attribute_schema} schema ({len(schema)} fields)")

        # Prepare data with enhanced schema compliance and type conversion
        rows = []
        for scene in scene_data:
            row = {}

            # Map scene data to schema fields with robust type conversion
            for field_name, field_config in schema.items():
                if field_name == "geometry":
                    continue  # Handle geometry separately

                # Get value from scene data
                value = scene.get(field_name)

                # Enhanced type conversion with better error handling
                if value is not None:
                    try:
                        field_type = field_config["type"]
                        
                        if field_type == "TEXT":
                            row[field_name] = str(value)
                        elif field_type == "REAL":
                            # Handle numeric conversion more robustly
                            if isinstance(value, (int, float)):
                                row[field_name] = float(value)
                            elif isinstance(value, str) and value.strip():
                                row[field_name] = float(value)
                            else:
                                row[field_name] = None
                        elif field_type == "INTEGER":
                            if isinstance(value, (int, float)):
                                row[field_name] = int(value)
                            elif isinstance(value, str) and value.strip():
                                row[field_name] = int(float(value))  # Handle "10.0" -> 10
                            else:
                                row[field_name] = None
                        elif field_type == "BOOLEAN":
                            if isinstance(value, bool):
                                row[field_name] = value
                            elif isinstance(value, str):
                                row[field_name] = value.lower() in ("true", "1", "yes", "on")
                            else:
                                row[field_name] = bool(value)
                        elif field_type == "DATE":
                            if isinstance(value, str) and value:
                                # Handle ISO format dates
                                if "T" in value:
                                    row[field_name] = pd.to_datetime(value).date()
                                else:
                                    row[field_name] = pd.to_datetime(value).date()
                            else:
                                row[field_name] = value
                        else:
                            row[field_name] = value
                            
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Type conversion error for {field_name} = {value}: {e}")
                        row[field_name] = None
                else:
                    row[field_name] = None

            rows.append(row)

        # Create GeoDataFrame
        geometries = [scene["geometry"] for scene in scene_data]
        gdf = gpd.GeoDataFrame(rows, geometry=geometries, crs=self.config.target_crs)

        # Ensure consistent column order
        ordered_columns = [col for col in schema.keys() if col != "geometry"] + ["geometry"]
        gdf = gdf.reindex(columns=[col for col in ordered_columns if col in gdf.columns])

        logger.info(f"Created GeoDataFrame with {len(gdf)} features and {len(gdf.columns)} attributes")
        
        # Log summary of non-null values for key fields
        key_fields = ["scene_id", "acquired", "cloud_cover", "aoi_km2", "coverage_percentage"]
        for field in key_fields:
            if field in gdf.columns:
                non_null_count = gdf[field].notna().sum()
                logger.info(f"  {field}: {non_null_count}/{len(gdf)} non-null values")
        
        return gdf

    def _write_footprints_layer(
        self, gdf: gpd.GeoDataFrame, output_file: Path, layer_name: str
    ) -> None:
        """Write scene footprints layer to GeoPackage."""
        try:
            # Ensure GeoDataFrame has the correct CRS before writing
            if gdf.crs is None:
                gdf = gdf.set_crs(self.config.target_crs)
            elif str(gdf.crs) != self.config.target_crs:
                gdf = gdf.to_crs(self.config.target_crs)

            # Write to GeoPackage - DON'T pass crs parameter with pyogrio engine
            # The CRS is already set on the GeoDataFrame
            gdf.to_file(output_file, driver="GPKG", layer=layer_name)

            logger.info(
                f"Written footprints layer '{layer_name}' with {len(gdf)} features"
            )

        except Exception as e:
            raise PlanetScopeError(f"Failed to write footprints layer: {e}")

    def _create_summary_layer(
        self, output_file: Path, gdf: gpd.GeoDataFrame, roi: Optional[Polygon] = None
    ) -> None:
        """Create analysis summary layer with aggregated statistics."""
        try:
            # Calculate summary statistics
            summary_stats = {
                "total_scenes": len(gdf),
                "total_area_km2": (
                    gdf["area_km2"].sum() if "area_km2" in gdf.columns else 0
                ),
                "avg_cloud_cover": (
                    gdf["cloud_cover"].mean() if "cloud_cover" in gdf.columns else None
                ),
                "date_range_start": (
                    gdf["acquisition_date"].min()
                    if "acquisition_date" in gdf.columns
                    else None
                ),
                "date_range_end": (
                    gdf["acquisition_date"].max()
                    if "acquisition_date" in gdf.columns
                    else None
                ),
                "unique_satellites": (
                    gdf["satellite_id"].nunique()
                    if "satellite_id" in gdf.columns
                    else 0
                ),
                "raster_files_included": len(self.processed_rasters),
                "total_raster_size_mb": sum(
                    r.file_size_mb for r in self.processed_rasters
                ),
            }

            # Create summary geometry (union of all scenes or ROI)
            if roi:
                summary_geometry = roi
            else:
                summary_geometry = gdf.geometry.unary_union

            # Create summary GeoDataFrame with proper CRS
            summary_gdf = gpd.GeoDataFrame(
                [summary_stats], geometry=[summary_geometry], crs=self.config.target_crs
            )

            # Write summary layer - no crs parameter needed
            summary_gdf.to_file(output_file, driver="GPKG", layer="analysis_summary")

            logger.info("Created analysis summary layer")

        except Exception as e:
            logger.error(f"Failed to create summary layer: {e}")

    def _add_single_raster_reference(
        self, output_file: Path, raster_path: Path, layer_name: str
    ) -> None:
        """Add a single raster reference to GeoPackage."""
        try:

            with rasterio.open(raster_path) as src:
                # Create polygon representing raster extent
                bounds = src.bounds
                extent_polygon = box(
                    bounds.left, bounds.bottom, bounds.right, bounds.top
                )

                # Get detailed raster information
                file_size_mb = raster_path.stat().st_size / (1024 * 1024)

                # Create comprehensive reference data
                reference_data = {
                    "raster_name": raster_path.name,
                    "raster_path": str(raster_path),
                    "layer_name": layer_name,
                    "width": src.width,
                    "height": src.height,
                    "band_count": src.count,
                    "data_type": str(src.dtypes[0]),
                    "crs": str(src.crs),
                    "pixel_size_x": abs(src.transform[0]),
                    "pixel_size_y": abs(src.transform[4]),
                    "file_size_mb": file_size_mb,
                    "nodata_value": src.nodata,
                    "bounds_minx": bounds.left,
                    "bounds_miny": bounds.bottom,
                    "bounds_maxx": bounds.right,
                    "bounds_maxy": bounds.top,
                    "created_date": datetime.now().isoformat(),
                    "geometry": extent_polygon,
                }

                # Create GeoDataFrame with proper CRS handling
                ref_gdf = gpd.GeoDataFrame([reference_data], crs=src.crs)

                # Reproject to target CRS if necessary
                if str(src.crs) != self.config.target_crs:
                    ref_gdf = ref_gdf.to_crs(self.config.target_crs)

                # Write without crs parameter
                ref_gdf.to_file(output_file, driver="GPKG", layer=layer_name)

        except Exception as e:
            logger.error(f"Failed to create raster reference layer: {e}")

    def _create_raster_inventory_layer(self, output_file: Path) -> None:
        """Create a consolidated inventory layer for all processed rasters."""
        try:
            if not self.processed_rasters:
                return

            # Create inventory data
            inventory_data = []
            for raster_info in self.processed_rasters:
                # Create extent polygon from bounds
                bounds = raster_info.bounds
                extent_polygon = box(bounds[0], bounds[1], bounds[2], bounds[3])

                inventory_data.append(
                    {
                        "scene_id": raster_info.scene_id,
                        "asset_type": raster_info.asset_type,
                        "file_name": Path(raster_info.original_path).name,
                        "file_size_mb": raster_info.file_size_mb,
                        "width": raster_info.width,
                        "height": raster_info.height,
                        "band_count": raster_info.band_count,
                        "data_type": raster_info.data_type,
                        "crs": raster_info.crs,
                        "clipped_to_roi": raster_info.clipped,
                        "pixel_count": raster_info.width * raster_info.height,
                        "geometry": extent_polygon,
                    }
                )

            # Create inventory GeoDataFrame with proper CRS
            inventory_gdf = gpd.GeoDataFrame(inventory_data, crs=self.config.target_crs)

            # Write inventory layer without crs parameter
            inventory_gdf.to_file(output_file, driver="GPKG", layer="raster_inventory")

            logger.info(
                f"Created raster inventory layer with {len(inventory_data)} entries"
            )

        except Exception as e:
            logger.error(f"Failed to create raster inventory layer: {e}")

    def _add_multiple_imagery_layers(
        self,
        output_file: Path,
        downloaded_files: List[str],
        scene_data: List[Dict],
        roi: Optional[Polygon] = None,
    ) -> None:
        """
        Add multiple raster imagery layers to GeoPackage.

        This method handles multiple raster files, processing each one individually
        and creating appropriate reference layers for each.
        """
        if not downloaded_files:
            return

        logger.info(
            f"Processing {len(downloaded_files)} imagery files for GeoPackage..."
        )

        # Group files by scene/asset type for better organization
        file_groups = self._group_raster_files(downloaded_files)

        for group_name, file_list in file_groups.items():
            try:
                logger.info(
                    f"Processing group '{group_name}' with {len(file_list)} files"
                )
                self._process_raster_group(output_file, group_name, file_list, roi)

            except Exception as e:
                logger.error(f"Failed to process raster group {group_name}: {e}")
                continue

        # Create consolidated raster inventory layer
        if self.processed_rasters:
            self._create_raster_inventory_layer(output_file)

    def _group_raster_files(self, downloaded_files: List[str]) -> Dict[str, List[str]]:
        """Group raster files by asset type for organized processing."""
        groups = {}

        for file_path in downloaded_files:
            try:
                file_path_obj = Path(file_path)
                filename = file_path_obj.stem

                # Extract asset type from filename
                # Expected format: {scene_id}_{asset_type}.tif
                # Examples:
                #   scene_001_ortho_analytic_4b.tif -> "ortho_analytic_4b"
                #   scene_002_ortho_visual.tif -> "ortho_visual"
                #   random_file.tif -> goes to misc_rasters

                # Look for common Planet asset type patterns
                if "ortho_analytic_4b" in filename:
                    asset_type = "ortho_analytic_4b"
                elif "ortho_visual" in filename:
                    asset_type = "ortho_visual"
                elif "ortho_analytic_8b" in filename:
                    asset_type = "ortho_analytic_8b"
                elif "ortho_analytic_3b" in filename:
                    asset_type = "ortho_analytic_3b"
                elif "udm2" in filename:
                    asset_type = "udm2"
                elif "udm" in filename:
                    asset_type = "udm"
                else:
                    # Try to extract from filename pattern
                    parts = filename.split("_")
                    if len(parts) >= 3:
                        # Look for ortho_* patterns
                        ortho_indices = [
                            i for i, part in enumerate(parts) if part == "ortho"
                        ]
                        if ortho_indices:
                            # Take from 'ortho' to the end
                            ortho_idx = ortho_indices[0]
                            asset_type = "_".join(parts[ortho_idx:])
                        else:
                            # Fallback: use last 2-3 parts if they look like asset type
                            if len(parts) >= 3 and any(
                                keyword in "_".join(parts[-3:]).lower()
                                for keyword in ["analytic", "visual", "udm"]
                            ):
                                asset_type = "_".join(parts[-3:])
                            elif len(parts) >= 2:
                                asset_type = "_".join(parts[-2:])
                            else:
                                asset_type = "misc_rasters"
                    else:
                        # Doesn't match expected pattern
                        asset_type = "misc_rasters"

                # Group by asset type
                if asset_type not in groups:
                    groups[asset_type] = []
                groups[asset_type].append(file_path)

            except Exception as e:
                logger.warning(f"Could not categorize file {file_path}: {e}")
                # Add to miscellaneous group
                if "misc_rasters" not in groups:
                    groups["misc_rasters"] = []
                groups["misc_rasters"].append(file_path)

        return groups

    def _process_raster_group(
        self,
        output_file: Path,
        group_name: str,
        file_list: List[str],
        roi: Optional[Polygon] = None,
    ) -> None:
        """Process a group of related raster files."""
        for i, file_path in enumerate(file_list):
            try:
                raster_path = Path(file_path)
                if not raster_path.exists():
                    logger.warning(f"Raster file not found: {file_path}")
                    continue

                # Create unique layer name for this raster
                layer_name = f"{group_name}_{i+1}" if len(file_list) > 1 else group_name

                # Process and add this raster
                raster_info = self._add_single_raster(
                    output_file, raster_path, roi, layer_name
                )

                if raster_info:
                    self.processed_rasters.append(raster_info)

            except Exception as e:
                logger.error(f"Failed to add raster {file_path}: {e}")
                continue

    def _add_single_raster(
        self,
        output_file: Path,
        raster_path: Path,
        roi: Optional[Polygon] = None,
        layer_name: Optional[str] = None,
    ) -> Optional[RasterInfo]:
        """
        Add a single raster file to the GeoPackage with proper handling.

        This method processes individual raster files, optionally clips them to ROI,
        and creates appropriate reference layers in the GeoPackage.
        """
        try:
            with rasterio.open(raster_path) as src:
                # Get raster information
                profile = src.profile
                bounds = src.bounds

                # Extract scene ID and asset type from filename
                filename = raster_path.stem
                parts = filename.split("_")
                scene_id = "_".join(parts[:-1]) if len(parts) >= 2 else filename
                asset_type = parts[-1] if len(parts) >= 2 else "unknown"

                layer_name = layer_name or f"raster_{scene_id}_{asset_type}"

                # Check file size
                file_size_mb = raster_path.stat().st_size / (1024 * 1024)
                if file_size_mb > self.config.max_raster_size_mb:
                    logger.warning(
                        f"Raster {raster_path.name} is large ({file_size_mb:.1f} MB)"
                    )

                processed_path = raster_path
                clipped = False

                # Clip to ROI if requested and ROI is provided
                if self.config.clip_to_roi and roi:
                    try:
                        processed_path = self._clip_raster_to_roi(src, roi, raster_path)
                        clipped = True
                        logger.info(f"Clipped raster {raster_path.name} to ROI")
                    except Exception as e:
                        logger.warning(f"Failed to clip raster {raster_path.name}: {e}")
                        processed_path = raster_path
                        clipped = False

                # Create raster reference layer in GeoPackage
                self._create_raster_reference_layer(
                    output_file, processed_path, layer_name, src
                )

                # Create RasterInfo object
                raster_info = RasterInfo(
                    original_path=str(raster_path),
                    processed_path=str(processed_path),
                    scene_id=scene_id,
                    asset_type=asset_type,
                    file_size_mb=file_size_mb,
                    width=src.width,
                    height=src.height,
                    band_count=src.count,
                    data_type=str(src.dtypes[0]),
                    crs=str(src.crs),
                    bounds=bounds,
                    clipped=clipped,
                )

                logger.info(f"Added raster reference layer: {layer_name}")
                return raster_info

        except Exception as e:
            logger.error(f"Failed to process raster {raster_path}: {e}")
            return None

    def _clip_raster_to_roi(
        self, src: rasterio.DatasetReader, roi: Polygon, original_path: Path
    ) -> Path:
        """
        Clip raster to ROI and save to temporary file.

        Args:
            src: Open rasterio dataset
            roi: Region of interest polygon
            original_path: Original raster file path

        Returns:
            Path to clipped raster file
        """
        # Create temporary file for clipped raster
        temp_dir = Path(tempfile.gettempdir()) / "planetscope_py_clipped"
        temp_dir.mkdir(exist_ok=True)

        clipped_filename = f"clipped_{original_path.stem}.tif"
        clipped_path = temp_dir / clipped_filename

        try:
            # Ensure ROI is in same CRS as raster
            roi_crs = CRS.from_epsg(4326)  # Assume ROI is in WGS84
            roi_transformed = roi

            if src.crs != roi_crs:
                # Transform ROI to raster CRS
                from pyproj import Transformer

                transformer = Transformer.from_crs(roi_crs, src.crs, always_xy=True)
                roi_transformed = transform(transformer.transform, roi)

            # Clip raster using rasterio.mask
            clipped_data, clipped_transform = mask(
                src, [roi_transformed], crop=True, nodata=src.nodata or -9999
            )

            # Update profile for clipped raster
            profile = src.profile.copy()
            profile.update(
                {
                    "height": clipped_data.shape[1],
                    "width": clipped_data.shape[2],
                    "transform": clipped_transform,
                    "compress": self.config.compression,
                }
            )

            # Write clipped raster
            with rasterio.open(clipped_path, "w", **profile) as dst:
                dst.write(clipped_data)

            return clipped_path

        except Exception as e:
            logger.error(f"Failed to clip raster: {e}")
            # Return original path if clipping fails
            return original_path

    def _create_raster_reference_layer(
        self,
        output_file: Path,
        raster_path: Path,
        layer_name: str,
        src: rasterio.DatasetReader,
    ) -> None:
        """Create a comprehensive reference layer for raster files."""
        try:
            # Create polygon representing raster extent
            bounds = src.bounds
            extent_polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

            # Get detailed raster information
            file_size_mb = raster_path.stat().st_size / (1024 * 1024)

            # Create comprehensive reference data
            reference_data = {
                "raster_name": raster_path.name,
                "raster_path": str(raster_path),
                "layer_name": layer_name,
                "width": src.width,
                "height": src.height,
                "band_count": src.count,
                "data_type": str(src.dtypes[0]),
                "crs": str(src.crs),
                "pixel_size_x": abs(src.transform[0]),
                "pixel_size_y": abs(src.transform[4]),
                "file_size_mb": file_size_mb,
                "nodata_value": src.nodata,
                "bounds_minx": bounds.left,
                "bounds_miny": bounds.bottom,
                "bounds_maxx": bounds.right,
                "bounds_maxy": bounds.top,
                "created_date": datetime.now().isoformat(),
                "geometry": extent_polygon,
            }

            # Create GeoDataFrame and write to GeoPackage
            ref_gdf = gpd.GeoDataFrame([reference_data], crs=src.crs)

            # Reproject to target CRS if necessary
            if str(src.crs) != self.config.target_crs:
                ref_gdf = ref_gdf.to_crs(self.config.target_crs)

            ref_gdf.to_file(output_file, driver="GPKG", layer=layer_name)

        except Exception as e:
            logger.error(f"Failed to create raster reference layer: {e}")

    def _create_raster_inventory_layer(self, output_file: Path) -> None:
        """Create a consolidated inventory layer for all processed rasters."""
        try:
            if not self.processed_rasters:
                return

            # Create inventory data
            inventory_data = []
            for raster_info in self.processed_rasters:
                # Create extent polygon from bounds
                bounds = raster_info.bounds
                extent_polygon = box(bounds[0], bounds[1], bounds[2], bounds[3])

                inventory_data.append(
                    {
                        "scene_id": raster_info.scene_id,
                        "asset_type": raster_info.asset_type,
                        "file_name": Path(raster_info.original_path).name,
                        "file_size_mb": raster_info.file_size_mb,
                        "width": raster_info.width,
                        "height": raster_info.height,
                        "band_count": raster_info.band_count,
                        "data_type": raster_info.data_type,
                        "crs": raster_info.crs,
                        "clipped_to_roi": raster_info.clipped,
                        "pixel_count": raster_info.width * raster_info.height,
                        "geometry": extent_polygon,
                    }
                )

            # Create inventory GeoDataFrame
            inventory_gdf = gpd.GeoDataFrame(inventory_data, crs=self.config.target_crs)

            # Write inventory layer
            inventory_gdf.to_file(output_file, driver="GPKG", layer="raster_inventory")

            logger.info(
                f"Created raster inventory layer with {len(inventory_data)} entries"
            )

        except Exception as e:
            logger.error(f"Failed to create raster inventory layer: {e}")

    def _add_geopackage_metadata(
        self, output_file: Path, scenes: List[Dict], roi: Optional[Polygon] = None
    ) -> None:
        """Add metadata tables to GeoPackage."""
        conn = sqlite3.connect(output_file)
        cursor = conn.cursor()

        try:
            # Create metadata table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS geopackage_metadata (
                    id INTEGER PRIMARY KEY,
                    created_date TEXT,
                    created_by TEXT,
                    planetscope_py_version TEXT,
                    total_scenes INTEGER,
                    date_range_start TEXT,
                    date_range_end TEXT,
                    roi_area_km2 REAL,
                    processing_config TEXT,
                    raster_count INTEGER,
                    total_file_size_mb REAL
                )
            """
            )

            # Extract metadata
            scene_dates = []
            for scene in scenes:
                acquired = scene.get("properties", {}).get("acquired")
                if acquired:
                    scene_dates.append(acquired)

            date_start = min(scene_dates) if scene_dates else None
            date_end = max(scene_dates) if scene_dates else None
            roi_area = calculate_area_km2(roi) if roi else None

            total_raster_size = sum(r.file_size_mb for r in self.processed_rasters)

            # Insert metadata
            cursor.execute(
                """
                INSERT INTO geopackage_metadata 
                (created_date, created_by, planetscope_py_version, total_scenes, 
                 date_range_start, date_range_end, roi_area_km2, processing_config,
                 raster_count, total_file_size_mb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    "planetscope-py",
                    "4.0.0",  # Version
                    len(scenes),
                    date_start,
                    date_end,
                    roi_area,
                    str(self.config.__dict__),
                    len(self.processed_rasters),
                    total_raster_size,
                ),
            )

            conn.commit()
            logger.info("Added GeoPackage metadata")

        except Exception as e:
            logger.error(f"Failed to add metadata: {e}")
        finally:
            conn.close()

    def _add_layer_styles(self, output_file: Path, layer_name: str) -> None:
        """Add default styling for layers."""
        conn = sqlite3.connect(output_file)
        cursor = conn.cursor()

        try:
            # Create styles extension table (if not exists)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS layer_styles (
                    id INTEGER PRIMARY KEY,
                    f_table_catalog TEXT,
                    f_table_schema TEXT, 
                    f_table_name TEXT,
                    f_geometry_column TEXT,
                    styleName TEXT,
                    styleQML TEXT,
                    styleSLD TEXT,
                    useAsDefault INTEGER,
                    description TEXT,
                    owner TEXT,
                    ui TEXT,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Default QGIS style for scene footprints
            qml_style = self._generate_qgis_style()

            cursor.execute(
                """
                INSERT INTO layer_styles 
                (f_table_name, f_geometry_column, styleName, styleQML, useAsDefault, description)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    layer_name,
                    "geom",
                    "PlanetScope Scenes",
                    qml_style,
                    1,
                    "Default style for PlanetScope scene footprints",
                ),
            )

            conn.commit()
            logger.info("Added layer styling")

        except Exception as e:
            logger.error(f"Failed to add styling: {e}")
        finally:
            conn.close()

    def _create_summary_layer(
        self, output_file: Path, gdf: gpd.GeoDataFrame, roi: Optional[Polygon] = None
    ) -> None:
        """Create summary statistics layer."""
        try:
            # Calculate summary statistics
            summary_stats = {
                "total_scenes": len(gdf),
                "total_area_km2": (
                    gdf["area_km2"].sum() if "area_km2" in gdf.columns else 0
                ),
                "avg_cloud_cover": (
                    gdf["cloud_cover"].mean() if "cloud_cover" in gdf.columns else None
                ),
                "date_range_start": (
                    gdf["acquisition_date"].min()
                    if "acquisition_date" in gdf.columns
                    else None
                ),
                "date_range_end": (
                    gdf["acquisition_date"].max()
                    if "acquisition_date" in gdf.columns
                    else None
                ),
                "unique_satellites": (
                    gdf["satellite_id"].nunique()
                    if "satellite_id" in gdf.columns
                    else 0
                ),
                "raster_files_included": len(self.processed_rasters),
                "total_raster_size_mb": sum(
                    r.file_size_mb for r in self.processed_rasters
                ),
            }

            # Create summary geometry (union of all scenes or ROI)
            if roi:
                summary_geometry = roi
            else:
                summary_geometry = gdf.geometry.unary_union

            # Create summary GeoDataFrame
            summary_gdf = gpd.GeoDataFrame(
                [summary_stats], geometry=[summary_geometry], crs=self.config.target_crs
            )

            # Write summary layer
            summary_gdf.to_file(output_file, driver="GPKG", layer="analysis_summary")

            logger.info("Created analysis summary layer")

        except Exception as e:
            logger.error(f"Failed to create summary layer: {e}")

    def _validate_geopackage(self, output_file: Path) -> bool:
        """Validate the created GeoPackage."""
        try:
            # Check if file exists and is readable
            if not output_file.exists():
                raise FileNotFoundError(f"GeoPackage file not found: {output_file}")

            # Try to read with geopandas
            layers = fiona.listlayers(output_file)

            if not layers:
                raise ValueError("No layers found in GeoPackage")

            # Validate each layer
            for layer in layers:
                try:
                    test_gdf = gpd.read_file(output_file, layer=layer)
                    logger.info(f"Layer '{layer}': {len(test_gdf)} features")
                except Exception as e:
                    logger.warning(f"Issue with layer '{layer}': {e}")

            logger.info(f"GeoPackage validation successful: {len(layers)} layers")
            return True

        except Exception as e:
            logger.error(f"GeoPackage validation failed: {e}")
            return False
    def validate_geopackage_metadata(self, geopackage_path: str) -> Dict:
        """
        NEW METHOD: Validate metadata completeness in created GeoPackage.
        
        Returns a detailed report on metadata quality and completeness.
        """
        try:
            import geopandas as gpd
            
            gdf = gpd.read_file(geopackage_path)
            schema = self.attribute_schemas[self.config.attribute_schema]
            
            report = {
                "file_path": geopackage_path,
                "total_features": len(gdf),
                "total_attributes": len(gdf.columns),
                "schema_type": self.config.attribute_schema,
                "schema_compliance": {},
                "missing_fields": [],
                "null_value_analysis": {},
                "aoi_analysis": {},
                "recommendations": []
            }
            
            # Check schema compliance
            for field_name in schema.keys():
                if field_name == "geometry":
                    continue
                    
                if field_name in gdf.columns:
                    null_count = gdf[field_name].isnull().sum()
                    null_percentage = (null_count / len(gdf)) * 100
                    
                    report["schema_compliance"][field_name] = "present"
                    report["null_value_analysis"][field_name] = {
                        "null_count": int(null_count),
                        "null_percentage": round(null_percentage, 2),
                        "non_null_count": len(gdf) - null_count
                    }
                else:
                    report["missing_fields"].append(field_name)
                    report["schema_compliance"][field_name] = "missing"
            
            # AOI-specific analysis
            if "aoi_km2" in gdf.columns:
                aoi_values = gdf["aoi_km2"].dropna()
                report["aoi_analysis"] = {
                    "scenes_with_aoi": int((aoi_values > 0).sum()),
                    "total_aoi_km2": float(aoi_values.sum()),
                    "average_aoi_km2": float(aoi_values.mean()) if len(aoi_values) > 0 else 0.0,
                    "max_aoi_km2": float(aoi_values.max()) if len(aoi_values) > 0 else 0.0,
                    "scenes_no_overlap": int((aoi_values == 0).sum()),
                    "aoi_calculation_success": True
                }
            else:
                report["aoi_analysis"] = {
                    "aoi_calculation_success": False,
                    "message": "AOI field not found in GeoPackage"
                }
            
            # Generate recommendations
            if report["missing_fields"]:
                report["recommendations"].append(
                    f"Missing {len(report['missing_fields'])} expected fields from schema"
                )
            
            high_null_fields = [
                field for field, stats in report["null_value_analysis"].items()
                if stats["null_percentage"] > 50
            ]
            
            if high_null_fields:
                report["recommendations"].append(
                    f"High null values (>50%) in: {', '.join(high_null_fields[:3])}"
                )
            
            if report["aoi_analysis"].get("aoi_calculation_success"):
                aoi_scenes = report["aoi_analysis"]["scenes_with_aoi"]
                if aoi_scenes == 0:
                    report["recommendations"].append("No scenes overlap with ROI - check geometry alignment")
                elif aoi_scenes < len(gdf) * 0.5:
                    report["recommendations"].append("Less than 50% of scenes overlap with ROI")
                else:
                    report["recommendations"].append(f"AOI calculation successful: {aoi_scenes} scenes overlap ROI")
            
            if not report["missing_fields"] and not high_null_fields:
                report["recommendations"].append("Metadata completeness is excellent!")
            
            return report
            
        except Exception as e:
            return {
                "error": f"Failed to validate metadata: {e}",
                "file_path": geopackage_path
            }

    def get_geopackage_info(self, geopackage_path: str) -> Dict:
        """Get comprehensive information about a GeoPackage."""
        try:
            gp_path = Path(geopackage_path)
            if not gp_path.exists():
                raise FileNotFoundError(f"GeoPackage not found: {geopackage_path}")

            # Get layers using fiona (spatial layers only)
            spatial_layers = fiona.listlayers(gp_path)

            # Get all tables using sqlite (includes non-spatial tables)
            import sqlite3

            conn = sqlite3.connect(gp_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'gpkg_%' AND name NOT LIKE 'rtree_%'"
            )
            all_tables = [row[0] for row in cursor.fetchall()]
            conn.close()

            layer_info = {}

            # Process spatial layers (those that fiona can read)
            for layer_name in spatial_layers:
                try:
                    gdf = gpd.read_file(gp_path, layer=layer_name)

                    # Check if layer has geometry
                    if hasattr(gdf, "geometry") and "geometry" in gdf.columns:
                        geometry_type = (
                            str(gdf.geometry.geom_type.iloc[0])
                            if len(gdf) > 0 and not gdf.geometry.iloc[0] is None
                            else "Unknown"
                        )
                        bbox = (
                            tuple(gdf.total_bounds)
                            if len(gdf) > 0 and not gdf.geometry.empty.all()
                            else (0, 0, 0, 0)
                        )
                        crs = str(gdf.crs) if gdf.crs else "Unknown"
                        layer_type = "vector"
                    else:
                        geometry_type = "None"
                        bbox = (0, 0, 0, 0)
                        crs = "N/A"
                        layer_type = "table"

                    layer_info[layer_name] = {
                        "name": layer_name,
                        "layer_type": layer_type,
                        "feature_count": len(gdf),
                        "geometry_type": geometry_type,
                        "crs": crs,
                        "bbox": bbox,
                        "created": datetime.now(),  # Would need to get from metadata
                    }

                except Exception as e:
                    logger.warning(f"Could not analyze layer {layer_name}: {e}")
                    # Add basic info for layers that can't be analyzed
                    layer_info[layer_name] = {
                        "name": layer_name,
                        "layer_type": "table",
                        "feature_count": 0,
                        "geometry_type": "Unknown",
                        "crs": "N/A",
                        "bbox": (0, 0, 0, 0),
                        "created": datetime.now(),
                        "error": str(e),
                    }

            # Process non-spatial tables that fiona can't read
            non_spatial_tables = set(all_tables) - set(spatial_layers)
            for table_name in non_spatial_tables:
                try:
                    # Get row count for non-spatial tables
                    conn = sqlite3.connect(gp_path)
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                    conn.close()

                    layer_info[table_name] = {
                        "name": table_name,
                        "layer_type": "table",
                        "feature_count": row_count,
                        "geometry_type": "None",
                        "crs": "N/A",
                        "bbox": (0, 0, 0, 0),
                        "created": datetime.now(),
                    }

                except Exception as e:
                    logger.warning(f"Could not analyze table {table_name}: {e}")
                    layer_info[table_name] = {
                        "name": table_name,
                        "layer_type": "table",
                        "feature_count": 0,
                        "geometry_type": "Unknown",
                        "crs": "N/A",
                        "bbox": (0, 0, 0, 0),
                        "created": datetime.now(),
                        "error": str(e),
                    }

            # Get file size
            file_size_mb = gp_path.stat().st_size / (1024 * 1024)

            # Total layers = spatial layers + non-spatial tables
            total_layers = len(spatial_layers) + len(non_spatial_tables)

            return {
                "file_path": str(gp_path),
                "file_size_mb": file_size_mb,
                "total_layers": total_layers,
                "layer_info": layer_info,
                "created": datetime.fromtimestamp(gp_path.stat().st_ctime).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get GeoPackage info: {e}")
            return {"error": str(e)}

    def add_imagery_to_existing_geopackage(
        self,
        geopackage_path: str,
        downloaded_files: List[str],
        scene_mapping: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Add imagery to an existing GeoPackage."""
        try:
            gp_path = Path(geopackage_path)
            if not gp_path.exists():
                raise FileNotFoundError(f"GeoPackage not found: {geopackage_path}")

            # Reset processed rasters list
            self.processed_rasters = []

            # Add imagery layers
            self._add_multiple_imagery_layers(gp_path, downloaded_files, [], None)

            # Update metadata
            self._update_geopackage_metadata(gp_path)

            logger.info(
                f"Added {len(downloaded_files)} imagery files to existing GeoPackage"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add imagery to existing GeoPackage: {e}")
            return False

    def _update_geopackage_metadata(self, output_file: Path) -> None:
        """Update metadata for existing GeoPackage."""
        conn = sqlite3.connect(output_file)
        cursor = conn.cursor()

        try:
            total_raster_size = sum(r.file_size_mb for r in self.processed_rasters)

            cursor.execute(
                """
                UPDATE geopackage_metadata 
                SET raster_count = ?, total_file_size_mb = ?, updated_date = ?
                WHERE id = 1
            """,
                (
                    len(self.processed_rasters),
                    total_raster_size,
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()

        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
        finally:
            conn.close()

    def _get_minimal_schema(self) -> Dict:
        """Enhanced minimal schema with essential Planet API fields."""
        return {
            # Core essentials
            "scene_id": {"type": "TEXT", "description": "Unique scene identifier"},
            "acquired": {"type": "TEXT", "description": "Acquisition datetime"},
            "satellite_id": {"type": "TEXT", "description": "Satellite identifier"},
            "item_type": {"type": "TEXT", "description": "Planet item type"},
            
            # Quality essentials
            "cloud_cover": {"type": "REAL", "description": "Cloud cover percentage (0-1)"},
            "cloud_percent": {"type": "REAL", "description": "Cloud percentage"},
            "clear_percent": {"type": "REAL", "description": "Clear pixels percentage"},
            
            # Area calculations
            "area_km2": {"type": "REAL", "description": "Scene area in km²"},
            "aoi_km2": {"type": "REAL", "description": "Area of intersection with ROI in km²"},
            "coverage_percentage": {"type": "REAL", "description": "ROI coverage percentage"},
        }


    # =============================================================================
    # UPDATE 2: Enhanced _get_standard_schema (most Planet API fields)
    # =============================================================================

    def _get_standard_schema(self) -> Dict:
        """Enhanced standard schema with most Planet API fields as standard."""
        minimal = self._get_minimal_schema()
        additional = {
            # Technical specifications
            "provider": {"type": "TEXT", "description": "Data provider"},
            "instrument": {"type": "TEXT", "description": "Instrument type"},
            "gsd": {"type": "REAL", "description": "Ground sample distance (m)"},
            "pixel_resolution": {"type": "REAL", "description": "Pixel resolution (m)"},
            "ground_control": {"type": "BOOLEAN", "description": "Ground control points available"},
            "quality_category": {"type": "TEXT", "description": "Quality category"},
            "strip_id": {"type": "TEXT", "description": "Strip identifier"},
            
            # Enhanced quality metrics (from your Planet API example)
            "clear_confidence_percent": {"type": "REAL", "description": "Clear confidence percentage"},
            "visible_percent": {"type": "REAL", "description": "Visible pixels percentage"},
            "visible_confidence_percent": {"type": "REAL", "description": "Visible confidence percentage"},
            "shadow_percent": {"type": "REAL", "description": "Shadow percentage"},
            "snow_ice_percent": {"type": "REAL", "description": "Snow/ice percentage"},
            "heavy_haze_percent": {"type": "REAL", "description": "Heavy haze percentage"},
            "light_haze_percent": {"type": "REAL", "description": "Light haze percentage"},
            "anomalous_pixels": {"type": "REAL", "description": "Anomalous pixels"},
            
            # Solar and viewing geometry
            "sun_elevation": {"type": "REAL", "description": "Sun elevation angle"},
            "sun_azimuth": {"type": "REAL", "description": "Sun azimuth angle"},
            "satellite_azimuth": {"type": "REAL", "description": "Satellite azimuth angle"},
            "view_angle": {"type": "REAL", "description": "View angle in degrees"},
            
            # Temporal information
            "published": {"type": "TEXT", "description": "Publication datetime"},
            "updated": {"type": "TEXT", "description": "Last update datetime"},
            "publishing_stage": {"type": "TEXT", "description": "Publishing stage"},
            
            # Calculated geometry
            "acquisition_date": {"type": "DATE", "description": "Acquisition date"},
            "centroid_lat": {"type": "REAL", "description": "Centroid latitude"},
            "centroid_lon": {"type": "REAL", "description": "Centroid longitude"},
        }
        return {**minimal, **additional}


    # =============================================================================
    # UPDATE 3: Enhanced _get_comprehensive_schema (ALL Planet API fields)
    # =============================================================================

    def _get_comprehensive_schema(self) -> Dict:
        """Enhanced comprehensive schema with ALL possible Planet API fields."""
        standard = self._get_standard_schema()
        additional = {
            # Additional technical details
            "platform": {"type": "TEXT", "description": "Satellite platform"},
            "spacecraft_id": {"type": "TEXT", "description": "Spacecraft identifier"},
            "epsg_code": {"type": "INTEGER", "description": "EPSG code"},
            "processing_level": {"type": "TEXT", "description": "Processing level"},
            
            # Enhanced quality and radiometric
            "usable_data": {"type": "REAL", "description": "Usable data percentage (0-1)"},
            "overall_quality": {"type": "REAL", "description": "Overall quality score (0-1)"},
            "suitability": {"type": "TEXT", "description": "Scene suitability rating"},
            "geometric_accuracy": {"type": "TEXT", "description": "Geometric accuracy category"},
            "radiometric_target": {"type": "TEXT", "description": "Radiometric calibration target"},
            "black_fill": {"type": "REAL", "description": "Black fill percentage"},
            
            # Advanced viewing geometry
            "off_nadir": {"type": "REAL", "description": "Off-nadir angle in degrees"},
            "azimuth_angle": {"type": "REAL", "description": "Azimuth angle in degrees"},
            
            # Temporal details (enhanced)
            "acquisition_time": {"type": "TEXT", "description": "Time of acquisition"},
            "day_of_year": {"type": "INTEGER", "description": "Day of year (1-366)"},
            
            # Geometric bounds (calculated from geometry)
            "bounds_west": {"type": "REAL", "description": "Western boundary longitude"},
            "bounds_south": {"type": "REAL", "description": "Southern boundary latitude"}, 
            "bounds_east": {"type": "REAL", "description": "Eastern boundary longitude"},
            "bounds_north": {"type": "REAL", "description": "Northern boundary latitude"},
            
            # Calculated geometric properties
            "perimeter_km": {"type": "REAL", "description": "Scene perimeter in km"},
            "aspect_ratio": {"type": "REAL", "description": "Scene aspect ratio (width/height)"},
        }
        return {**standard, **additional}

    def _generate_qgis_style(self) -> str:
        """Generate QGIS QML style for scene footprints."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<qgis version="3.16">
  <renderer-v2 type="singleSymbol">
    <symbols>
      <symbol type="fill" name="0">
        <layer class="SimpleFill">
          <prop k="border_width_unit" v="MM"/>
          <prop k="color" v="125,139,143,63"/>
          <prop k="outline_color" v="35,35,35,255"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings>
      <text-style fieldName="scene_id" fontSize="8"/>
    </settings>
  </labeling>
</qgis>"""

    def export_geopackage_report(self, geopackage_path: str, output_path: str) -> str:
        """Export comprehensive report about GeoPackage contents."""
        try:
            import json

            gp_info = self.get_geopackage_info(geopackage_path)

            # Create detailed report
            report = {
                "geopackage_analysis": gp_info,
                "processed_rasters": [r.__dict__ for r in self.processed_rasters],
                "configuration": self.config.__dict__,
                "generated_at": datetime.now().isoformat(),
                "generator": "planetscope-py GeoPackageManager v4.0.0",
            }

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"GeoPackage report exported to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export GeoPackage report: {e}")
            raise PlanetScopeError(f"Report export failed: {e}")

    def cleanup_temporary_files(self) -> None:
        """Clean up temporary clipped raster files."""
        temp_dir = Path(tempfile.gettempdir()) / "planetscope_py_clipped"

        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary files")
            except Exception as e:
                logger.warning(f"Could not clean up temporary files: {e}")


# Example usage
if __name__ == "__main__":
    # This would typically be run from a notebook or script
    pass
