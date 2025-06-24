"""Core utility functions for planetscope-py.

This module provides essential validation, transformation, and helper functions
used throughout the library.
"""
import os

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Union

from pyproj import Transformer
from shapely.geometry import shape
from shapely.validation import explain_validity

from .config import PlanetScopeConfig
from .exceptions import ValidationError

# Phase 2 additional imports
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, mapping
from shapely.ops import transform
from shapely.validation import make_valid
import pyproj
from pyproj import Transformer
import logging

logger = logging.getLogger(__name__)


def validate_geometry(geometry: Union[Dict[str, Any], Any, str]) -> Dict[str, Any]:
    """Universal geometry validator with shapefile support.

    This function serves as the universal entry point for all geometry processing
    in the planetscope-py library. It handles file paths, Shapely objects, and 
    traditional GeoJSON dictionaries, automatically handling CRS reprojection
    and validation.

    Args:
        geometry: One of:
            - File path (str): .shp, .geojson, .wkt, .txt files
            - Shapely geometry object
            - GeoJSON geometry dictionary
            - WKT string

    Returns:
        Validated and normalized geometry as GeoJSON dict in WGS84

    Raises:
        ValidationError: If geometry is invalid or file cannot be processed

    Examples:
        # File paths (NEW functionality)
        geom = validate_geometry("./study_area.shp")
        geom = validate_geometry("./roi.geojson")
        
        # Shapely objects (ENHANCED functionality)
        from shapely.geometry import box
        geom = validate_geometry(box(9.1, 45.45, 9.25, 45.5))
        
        # GeoJSON dict (EXISTING functionality - unchanged)
        geom = validate_geometry({
            "type": "Polygon",
            "coordinates": [[[9.1, 45.45], [9.25, 45.45], [9.25, 45.5], [9.1, 45.5], [9.1, 45.45]]]
        })
        
        # WKT string (NEW functionality)
        geom = validate_geometry("POLYGON((9.1 45.45, 9.25 45.45, 9.25 45.5, 9.1 45.5, 9.1 45.45))")
    """
    
    # 1. FILE PATH HANDLING (NEW)
    if isinstance(geometry, str):
        # Check if it's a file path
        if _is_file_path(geometry):
            return _process_geometry_file(geometry)
        else:
            # Try WKT parsing
            return _process_wkt_string(geometry)
    
    # 2. SHAPELY OBJECT HANDLING (ENHANCED)
    elif hasattr(geometry, "__geo_interface__"):
        # Convert Shapely object to GeoJSON
        geojson_geom = geometry.__geo_interface__
        return _validate_and_ensure_wgs84(geojson_geom)
    
    elif hasattr(geometry, "geom_type"):
        # Alternative way to handle Shapely objects
        try:
            from shapely.geometry import mapping
            geojson_geom = mapping(geometry)
            return _validate_and_ensure_wgs84(geojson_geom)
        except ImportError:
            # Fallback if mapping import fails
            geojson_geom = geometry.__geo_interface__
            return _validate_and_ensure_wgs84(geojson_geom)
    
    # 3. GEOJSON DICTIONARY HANDLING (EXISTING - unchanged logic)
    elif isinstance(geometry, dict):
        return _validate_geojson_geometry(geometry)
    
    # 4. UNSUPPORTED TYPE
    else:
        raise ValidationError(
            "Geometry must be a file path, Shapely object, GeoJSON dict, or WKT string",
            {
                "geometry": str(geometry)[:100], 
                "type": type(geometry).__name__,
                "supported_types": ["file_path", "shapely_object", "geojson_dict", "wkt_string"]
            },
        )


def _is_file_path(string_input: str) -> bool:
    """Check if string is likely a file path."""
    import os
    
    # Check if file exists
    if os.path.exists(string_input):
        return True
    
    # Check if it has a file extension
    if '.' in string_input and len(string_input.split('.')[-1]) <= 7:  # reasonable extension length
        # Could be a file path that doesn't exist yet
        supported_extensions = ['.shp', '.geojson', '.wkt', '.txt']
        ext = '.' + string_input.split('.')[-1].lower()
        return ext in supported_extensions
    
    return False


def _process_geometry_file(file_path: str) -> Dict[str, Any]:
    """Process geometry from various file formats."""
    import os
    
    if not os.path.exists(file_path):
        raise ValidationError(
            f"File not found: {file_path}",
            {"file_path": file_path, "suggestion": "Check file path and permissions"}
        )
    
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # Route to appropriate parser
    if ext == '.shp':
        return _process_shapefile(file_path)
    elif ext == '.geojson':
        return _process_geojson_file(file_path)
    elif ext in ['.wkt', '.txt']:
        return _process_wkt_file(file_path)
    else:
        raise ValidationError(
            f"Unsupported file format: {ext}",
            {
                "file_path": file_path,
                "supported_formats": [".shp", ".geojson", ".wkt", ".txt"]
            }
        )


def _process_shapefile(shp_path: str, feature_selection: str = 'union') -> Dict[str, Any]:
    """Process shapefile with automatic CRS handling and feature selection."""
    
    # Check for geopandas
    try:
        import geopandas as gpd
    except ImportError:
        raise ValidationError(
            "Shapefile support requires geopandas. Install with: pip install geopandas",
            {
                "missing_dependency": "geopandas", 
                "install_command": "pip install geopandas",
                "file_path": shp_path
            }
        )
    
    try:
        # Read shapefile
        logger.info(f"Reading shapefile: {shp_path}")
        gdf = gpd.read_file(shp_path)
        
        # Check if empty
        if len(gdf) == 0:
            raise ValidationError(f"Shapefile contains no features: {shp_path}")
        
        # Log original CRS
        original_crs = gdf.crs
        if original_crs:
            logger.info(f"Original CRS: {original_crs}")
        else:
            logger.warning("Shapefile has no CRS information, assuming WGS84")
        
        # Ensure WGS84 (required for Planet API)
        if original_crs is not None and original_crs.to_epsg() != 4326:
            logger.info(f"Reprojecting from {original_crs} to WGS84 (EPSG:4326)")
            gdf = gdf.to_crs('EPSG:4326')
        elif original_crs is None:
            logger.warning("No CRS found, assuming data is already in WGS84")
        
        # Filter to polygons only
        polygon_gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        
        if len(polygon_gdf) == 0:
            raise ValidationError(
                f"Shapefile contains no polygon features: {shp_path}",
                {"available_types": gdf.geometry.type.unique().tolist()}
            )
        
        logger.info(f"Found {len(polygon_gdf)} polygon features")
        
        # Feature selection (default: union all features)
        if len(polygon_gdf) == 1:
            # Single feature - use directly
            result_geom = polygon_gdf.iloc[0].geometry
        else:
            # Multiple features - union them
            logger.info(f"Multiple features found, creating union of {len(polygon_gdf)} polygons")
            from shapely.ops import unary_union
            geometries = polygon_gdf.geometry.tolist()
            result_geom = unary_union(geometries)
        
        # Convert MultiPolygon to largest Polygon if needed
        if result_geom.geom_type == 'MultiPolygon':
            logger.info("MultiPolygon result - selecting largest polygon")
            polygons = list(result_geom.geoms)
            result_geom = max(polygons, key=lambda p: p.area)
        
        # Validate result
        if not result_geom.is_valid:
            logger.warning("Invalid geometry detected, attempting to fix")
            result_geom = result_geom.buffer(0)
            
            if not result_geom.is_valid:
                raise ValidationError("Could not create valid polygon from shapefile")
        
        # Convert to GeoJSON and validate
        from shapely.geometry import mapping
        geojson_geom = mapping(result_geom)
        
        logger.info(f"Successfully processed shapefile: {shp_path}")
        return _validate_geojson_geometry(geojson_geom)
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Failed to process shapefile: {e}")


def _process_geojson_file(geojson_path: str) -> Dict[str, Any]:
    """Process GeoJSON file."""
    try:
        import json
        
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Handle different GeoJSON structures
        if geojson_data.get('type') == 'Feature':
            # Single feature
            return _validate_and_ensure_wgs84(geojson_data['geometry'])
        
        elif geojson_data.get('type') == 'FeatureCollection':
            # Feature collection - union all features
            features = geojson_data['features']
            if len(features) == 0:
                raise ValidationError(f"GeoJSON file contains no features: {geojson_path}")
            
            if len(features) == 1:
                return _validate_and_ensure_wgs84(features[0]['geometry'])
            else:
                # Union multiple features
                from shapely.geometry import shape
                from shapely.ops import unary_union
                geometries = [shape(f['geometry']) for f in features]
                result_geom = unary_union(geometries)
                
                # Convert to GeoJSON
                from shapely.geometry import mapping
                geojson_geom = mapping(result_geom)
                return _validate_and_ensure_wgs84(geojson_geom)
        
        elif geojson_data.get('type') in ['Polygon', 'MultiPolygon', 'Point', 'LineString']:
            # Direct geometry object
            return _validate_and_ensure_wgs84(geojson_data)
        
        else:
            raise ValidationError(f"Unsupported GeoJSON structure in file: {geojson_path}")
            
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Failed to process GeoJSON file: {e}")


def _process_wkt_file(file_path: str) -> Dict[str, Any]:
    """Process WKT from text file."""
    try:
        with open(file_path, 'r') as f:
            wkt_string = f.read().strip()
        return _process_wkt_string(wkt_string)
    except Exception as e:
        raise ValidationError(f"Failed to process WKT file: {e}")


def _process_wkt_string(wkt_string: str) -> Dict[str, Any]:
    """Process WKT string."""
    try:
        from shapely import wkt
        geom = wkt.loads(wkt_string)
        
        # Convert to GeoJSON
        from shapely.geometry import mapping
        geojson_geom = mapping(geom)
        
        return _validate_and_ensure_wgs84(geojson_geom)
    except Exception as e:
        raise ValidationError(f"Invalid WKT string: {e}")


def _validate_and_ensure_wgs84(geojson_geom: Dict[str, Any]) -> Dict[str, Any]:
    """Validate GeoJSON geometry and ensure it's in WGS84."""
    # First validate using existing logic
    validated_geom = _validate_geojson_geometry(geojson_geom)
    
    # Additional CRS validation could go here if needed
    # For now, we assume the geometry is already in the correct CRS
    # since file processing handles reprojection
    
    return validated_geom


def _validate_geojson_geometry(geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Validate GeoJSON geometry object - EXISTING LOGIC (unchanged)."""
    
    # This contains all the existing validation logic from the original function
    if not isinstance(geometry, dict):
        raise ValidationError(
            "Geometry must be a dictionary or Shapely object",
            {"geometry": str(geometry)[:100], "type": type(geometry).__name__},
        )

    required_fields = ["type", "coordinates"]
    for field in required_fields:
        if field not in geometry:
            raise ValidationError(
                f"Geometry missing required field: {field}",
                {"geometry": geometry, "missing_field": field},
            )

    geom_type = geometry["type"]
    coords = geometry["coordinates"]

    # Validate geometry type
    valid_types = [
        "Point",
        "LineString", 
        "Polygon",
        "MultiPoint",
        "MultiLineString",
        "MultiPolygon",
    ]
    if geom_type not in valid_types:
        raise ValidationError(
            f"Invalid geometry type: {geom_type}",
            {"geometry": geometry, "valid_types": valid_types},
        )

    try:
        # Use shapely for detailed validation
        from shapely.geometry import shape
        from shapely.validation import explain_validity
        
        geom_obj = shape(geometry)

        if not geom_obj.is_valid:
            explanation = explain_validity(geom_obj)
            raise ValidationError(
                f"Invalid geometry: {explanation}",
                {"geometry": geometry, "shapely_error": explanation},
            )

        # Check coordinate bounds (WGS84)
        bounds = geom_obj.bounds
        if bounds[0] < -180 or bounds[2] > 180:
            raise ValidationError(
                "Longitude coordinates must be between -180 and 180",
                {"geometry": geometry, "bounds": bounds},
            )
        if bounds[1] < -90 or bounds[3] > 90:
            raise ValidationError(
                "Latitude coordinates must be between -90 and 90",
                {"geometry": geometry, "bounds": bounds},
            )

        # Check polygon closure for Polygon types
        if geom_type == "Polygon":
            for ring in coords:
                if len(ring) < 4:
                    raise ValidationError(
                        "Polygon rings must have at least 4 coordinates",
                        {"geometry": geometry, "ring_length": len(ring)},
                    )
                if ring[0] != ring[-1]:
                    raise ValidationError(
                        "Polygon rings must be closed (first and last coordinates must be the same)",
                        {"geometry": geometry, "first": ring[0], "last": ring[-1]},
                    )

        return geometry

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(
            f"Geometry validation failed: {str(e)}",
            {"geometry": geometry, "error": str(e)},
        )


def validate_date_range(
    start_date: Union[str, datetime], end_date: Union[str, datetime]
) -> Tuple[str, str]:
    """Validate and normalize date range for Planet API.

    Args:
        start_date: Start date (ISO string or datetime object)
        end_date: End date (ISO string or datetime object)

    Returns:
        Tuple of (start_iso, end_iso) in Planet API format

    Raises:
        ValidationError: If dates are invalid or in wrong order

    Example:
        start, end = validate_date_range("2025-01-01", "2025-12-31")
        # Returns: ("2025-01-01T00:00:00.000000Z", "2025-12-31T23:59:59.999999Z")
    """

    def parse_date(
        date_input: Union[str, datetime], is_end_date: bool = False
    ) -> datetime:
        """Parse date input to datetime object with proper time handling."""
        if isinstance(date_input, datetime):
            return date_input

        if isinstance(date_input, str):
            # Check if it's a simple date string (YYYY-MM-DD)
            if (
                len(date_input) == 10
                and date_input.count("-") == 2
                and "T" not in date_input
            ):
                # Simple date format - add appropriate time
                dt = datetime.strptime(date_input, "%Y-%m-%d")

                if is_end_date:
                    # Set to end of day: 23:59:59.999999
                    dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
                else:
                    # Set to start of day: 00:00:00.000000
                    dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)

                return dt

            # Handle datetime strings with time components
            formats = [
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%fZ",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(date_input, fmt)
                except ValueError:
                    continue

            raise ValidationError(
                f"Invalid date format: {date_input}",
                {"date": date_input, "expected_formats": formats},
            )

        raise ValidationError(
            "Date must be string or datetime object",
            {"date": date_input, "type": type(date_input).__name__},
        )

    try:
        # CRITICAL FIX: Pass is_end_date flag to handle end-of-day properly
        start_dt = parse_date(start_date, is_end_date=False)
        end_dt = parse_date(end_date, is_end_date=True)

        # Ensure timezone awareness (assume UTC if naive)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)

        # Validate order
        if start_dt >= end_dt:
            raise ValidationError(
                "Start date must be before end date",
                {"start_date": start_date, "end_date": end_date},
            )

        # Convert to Planet API format (ISO with Z suffix)
        start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        return start_iso, end_iso

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(
            f"Date validation failed: {str(e)}",
            {"start_date": start_date, "end_date": end_date, "error": str(e)},
        )


def validate_roi_size(geometry: Dict[str, Any]) -> float:
    """Validate ROI size is within acceptable limits.

    Args:
        geometry: GeoJSON geometry object

    Returns:
        Area in square kilometers

    Raises:
        ValidationError: If ROI is too large

    Example:
        area_km2 = validate_roi_size(polygon_geometry)
    """
    try:
        geom_obj = shape(geometry)

        # Calculate area in square meters using equal-area projection
        # Use Mollweide projection for global equal-area calculation
        transformer = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
        geom_projected = transform_geometry(geom_obj, transformer)

        area_m2 = geom_projected.area
        area_km2 = area_m2 / 1_000_000  # Convert to km²

        config = PlanetScopeConfig()
        if area_km2 > config.MAX_ROI_AREA_KM2:
            raise ValidationError(
                f"ROI area too large: {area_km2:.2f} km² > {config.MAX_ROI_AREA_KM2} km²",
                {
                    "geometry": geometry,
                    "area_km2": area_km2,
                    "max_allowed": config.MAX_ROI_AREA_KM2,
                },
            )

        return area_km2

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(
            f"ROI size validation failed: {str(e)}",
            {"geometry": geometry, "error": str(e)},
        )


def transform_geometry(geom_obj, transformer) -> Any:
    """Transform geometry coordinates using pyproj transformer.

    Args:
        geom_obj: Shapely geometry object
        transformer: Pyproj transformer

    Returns:
        Transformed shapely geometry
    """
    from shapely.ops import transform

    return transform(transformer.transform, geom_obj)


def validate_cloud_cover(cloud_cover: Union[float, int]) -> float:
    """Validate cloud cover percentage.

    Args:
        cloud_cover: Cloud cover as percentage (0-100) or fraction (0-1)

    Returns:
        Normalized cloud cover as fraction (0-1)

    Raises:
        ValidationError: If cloud cover is invalid
    """
    if not isinstance(cloud_cover, (int, float)):
        raise ValidationError(
            "Cloud cover must be numeric",
            {"cloud_cover": cloud_cover, "type": type(cloud_cover).__name__},
        )

    # Convert percentage to fraction if needed
    if cloud_cover > 1.0:
        if cloud_cover > 100.0:
            raise ValidationError(
                "Cloud cover cannot exceed 100%", {"cloud_cover": cloud_cover}
            )
        cloud_cover = cloud_cover / 100.0

    if cloud_cover < 0.0:
        raise ValidationError(
            "Cloud cover cannot be negative", {"cloud_cover": cloud_cover}
        )

    return float(cloud_cover)


def validate_item_types(item_types: List[str]) -> List[str]:
    """Validate Planet item types.

    Args:
        item_types: List of Planet item type strings

    Returns:
        Validated item types

    Raises:
        ValidationError: If item types are invalid
    """
    if not isinstance(item_types, list):
        raise ValidationError(
            "Item types must be a list",
            {"item_types": item_types, "type": type(item_types).__name__},
        )

    if not item_types:
        raise ValidationError(
            "At least one item type required", {"item_types": item_types}
        )

    # Valid Planet item types (as of 2024)
    valid_types = {
        "PSScene",
        "REOrthoTile",
        "REScene",
        "PSOrthoTile",
        "SkySatScene",
        "SkySatCollect",
        "Landsat8L1G",
        "Sentinel2L1C",
        "MOD09GQ",
        "MYD09GQ",
        "MOD09GA",
        "MYD09GA",
    }

    for item_type in item_types:
        if not isinstance(item_type, str):
            raise ValidationError(
                f"Item type must be string: {item_type}",
                {"item_type": item_type, "type": type(item_type).__name__},
            )

        if item_type not in valid_types:
            raise ValidationError(
                f"Invalid item type: {item_type}",
                {"item_type": item_type, "valid_types": list(valid_types)},
            )

    return item_types


def format_api_url(base_url: str, endpoint: str, **params) -> str:
    """Format Planet API URL with parameters.

    Args:
        base_url: Base API URL
        endpoint: API endpoint path
        **params: URL parameters

    Returns:
        Formatted URL string

    Example:
        url = format_api_url(
            "https://api.planet.com/data/v1",
            "item-types/PSScene/items/12345/assets",
            item_id="12345"
        )
    """
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    if params:
        param_strs = []
        for key, value in params.items():
            if value is not None:
                param_strs.append(f"{key}={value}")
        if param_strs:
            url += "?" + "&".join(param_strs)

    return url


def calculate_geometry_bounds(
    geometry: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    """Calculate bounding box of geometry.

    Args:
        geometry: GeoJSON geometry object

    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat)

    Example:
        bounds = calculate_geometry_bounds(polygon_geometry)
        min_lon, min_lat, max_lon, max_lat = bounds
    """
    geom_obj = shape(geometry)
    return geom_obj.bounds


def create_point_geometry(longitude: float, latitude: float) -> Dict[str, Any]:
    """Create GeoJSON Point geometry.

    Args:
        longitude: Longitude coordinate
        latitude: Latitude coordinate

    Returns:
        GeoJSON Point geometry

    Example:
        point = create_point_geometry(-122.4194, 37.7749)  # San Francisco
    """
    return {"type": "Point", "coordinates": [longitude, latitude]}


def create_bbox_geometry(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> Dict[str, Any]:
    """Create GeoJSON Polygon from bounding box.

    Args:
        min_lon: Minimum longitude
        min_lat: Minimum latitude
        max_lon: Maximum longitude
        max_lat: Maximum latitude

    Returns:
        GeoJSON Polygon geometry

    Example:
        bbox = create_bbox_geometry(-122.5, 37.7, -122.3, 37.8)
    """
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]
        ],
    }


def pretty_print_json(data: Any) -> str:
    """Pretty print JSON data.

    Args:
        data: Data to format as JSON

    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=2, sort_keys=True)


def mask_api_key(api_key: str) -> str:
    """Mask API key for safe logging.

    Args:
        api_key: API key to mask

    Returns:
        Masked API key string
    """
    if len(api_key) > 8:
        return f"{api_key[:4]}...{api_key[-4:]}"
    return "***"


# ============================================================================
# PHASE 2 ADDITIONS - Added for Planet API integration
# ============================================================================


def calculate_area_km2(geometry: Union[Dict, Polygon]) -> float:
    """Calculate area of geometry in square kilometers.

    Enhanced version that works with both GeoJSON dicts and Shapely objects.

    Args:
        geometry: GeoJSON geometry or Shapely polygon

    Returns:
        Area in square kilometers

    Raises:
        ValidationError: If geometry is invalid
    """
    try:
        if isinstance(geometry, dict):
            geom = shape(geometry)
        else:
            geom = geometry

        if not geom.is_valid:
            geom = make_valid(geom)

        # Get centroid for appropriate projection
        centroid = geom.centroid

        # Use appropriate UTM zone for accurate area calculation
        utm_crs = get_utm_crs(centroid.x, centroid.y)

        # Transform to UTM for area calculation
        transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        utm_geom = transform(transformer.transform, geom)

        # Calculate area in square meters, convert to km�
        area_m2 = utm_geom.area
        area_km2 = area_m2 / 1_000_000

        return area_km2

    except Exception as e:
        raise ValidationError(f"Error calculating geometry area: {str(e)}")


def get_utm_crs(longitude: float, latitude: float) -> str:
    """Get appropriate UTM CRS for given coordinates.

    Args:
        longitude: Longitude in decimal degrees
        latitude: Latitude in decimal degrees

    Returns:
        EPSG code for UTM zone
    """
    # Handle None values safely
    if longitude is None or latitude is None:
        # Return WGS84 as safe fallback when coordinates are invalid
        # This allows the function to continue without crashing
        return "EPSG:4326"

    # Ensure values are numeric
    try:
        longitude = float(longitude)
        latitude = float(latitude)
    except (TypeError, ValueError):
        # Return WGS84 as safe fallback if conversion fails
        return "EPSG:4326"

    # Validate coordinate ranges
    if not (-180 <= longitude <= 180) or not (-90 <= latitude <= 90):
        # Return WGS84 for invalid coordinate ranges
        return "EPSG:4326"

    # Calculate UTM zone (1-60)
    utm_zone = int((longitude + 180) / 6) + 1

    # Ensure UTM zone is within valid range
    utm_zone = max(1, min(60, utm_zone))

    # Determine hemisphere - NOW SAFE FROM None COMPARISON
    if latitude >= 0:
        epsg_code = f"EPSG:{32600 + utm_zone}"  # Northern hemisphere
    else:
        epsg_code = f"EPSG:{32700 + utm_zone}"  # Southern hemisphere

    return epsg_code


def transform_geometry_crs(
    geometry: Union[Dict, Polygon],
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:3857",
) -> Dict:
    """Transform geometry between coordinate reference systems.

    Args:
        geometry: Input geometry
        source_crs: Source CRS (default: WGS84)
        target_crs: Target CRS (default: Web Mercator)

    Returns:
        Transformed geometry as GeoJSON dict

    Raises:
        ValidationError: If transformation fails
    """
    try:
        if isinstance(geometry, dict):
            geom = shape(geometry)
        else:
            geom = geometry

        # Create transformer
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

        # Transform geometry
        transformed_geom = transform(transformer.transform, geom)

        return mapping(transformed_geom)

    except Exception as e:
        raise ValidationError(f"Error transforming geometry: {str(e)}")


def create_bounding_box(
    geometry: Union[Dict, Polygon], buffer_meters: float = 0
) -> Dict:
    """Create bounding box from geometry.

    Args:
        geometry: Input geometry
        buffer_meters: Buffer distance in meters (default: 0)

    Returns:
        Bounding box as GeoJSON polygon

    Raises:
        ValidationError: If geometry processing fails
    """
    try:
        if isinstance(geometry, dict):
            geom = shape(geometry)
        else:
            geom = geometry

        # Get bounds
        minx, miny, maxx, maxy = geom.bounds

        # Apply buffer if specified
        if buffer_meters > 0:
            # Convert buffer to degrees (approximate)
            buffer_degrees = buffer_meters / 111000  # rough conversion
            minx -= buffer_degrees
            miny -= buffer_degrees
            maxx += buffer_degrees
            maxy += buffer_degrees

        # Create bounding box polygon
        bbox = Polygon(
            [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
        )

        return mapping(bbox)

    except Exception as e:
        raise ValidationError(f"Error creating bounding box: {str(e)}")


def buffer_geometry(geometry: Union[Dict, Polygon], buffer_meters: float) -> Dict:
    """Buffer geometry by specified distance.

    Args:
        geometry: Input geometry
        buffer_meters: Buffer distance in meters

    Returns:
        Buffered geometry as GeoJSON dict

    Raises:
        ValidationError: If buffering fails
    """
    try:
        if isinstance(geometry, dict):
            geom = shape(geometry)
        else:
            geom = geometry

        # Get centroid for projection
        centroid = geom.centroid
        utm_crs = get_utm_crs(centroid.x, centroid.y)

        # Transform to UTM for accurate buffering
        to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        utm_geom = transform(to_utm.transform, geom)

        # Buffer in UTM
        buffered_utm = utm_geom.buffer(buffer_meters)

        # Transform back to WGS84
        from_utm = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        buffered_geom = transform(from_utm.transform, buffered_utm)

        return mapping(buffered_geom)

    except Exception as e:
        raise ValidationError(f"Error buffering geometry: {str(e)}")


def validate_coordinates(longitude: float, latitude: float) -> Tuple[float, float]:
    """Validate geographic coordinates.

    Args:
        longitude: Longitude in decimal degrees
        latitude: Latitude in decimal degrees

    Returns:
        Tuple of validated coordinates

    Raises:
        ValidationError: If coordinates are invalid
    """
    if not isinstance(longitude, (int, float)):
        raise ValidationError("Longitude must be a number")

    if not isinstance(latitude, (int, float)):
        raise ValidationError("Latitude must be a number")

    if not -180.0 <= longitude <= 180.0:
        raise ValidationError("Longitude must be between -180 and 180 degrees")

    if not -90.0 <= latitude <= 90.0:
        raise ValidationError("Latitude must be between -90 and 90 degrees")

    return float(longitude), float(latitude)


def format_geometry_for_api(geometry: Union[Dict, Polygon]) -> Dict:
    """Format geometry for Planet API compatibility.

    Args:
        geometry: Input geometry

    Returns:
        API-compatible GeoJSON geometry
    """
    # Use existing robust validate_geometry function
    if isinstance(geometry, dict):
        geom_dict = validate_geometry(geometry)
    else:
        geom_dict = validate_geometry(mapping(geometry))

    # Ensure coordinates are properly formatted
    if isinstance(geom_dict.get("coordinates"), list):
        # Round coordinates to reasonable precision
        geom_dict["coordinates"] = _round_coordinates(geom_dict["coordinates"])

    return geom_dict


def _round_coordinates(coords: List, precision: int = 6) -> List:
    """Round coordinates to specified precision.

    Args:
        coords: Coordinate array (nested lists)
        precision: Decimal places to round to

    Returns:
        Rounded coordinate array
    """
    if isinstance(coords[0], (int, float)):
        # Single coordinate pair
        return [round(c, precision) for c in coords]
    else:
        # Nested coordinate array
        return [_round_coordinates(coord_group, precision) for coord_group in coords]


def safe_json_loads(json_str: str) -> Dict:
    """Safely load JSON string with error handling.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed JSON dictionary

    Raises:
        ValidationError: If JSON is invalid
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {str(e)}")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


# Enhanced geometry validation function to utils.py
def validate_geometry_enhanced(geometry):
    """Enhanced geometry validation that handles multiple input types."""

    # Handle Shapely objects
    if hasattr(geometry, "__geo_interface__"):
        return geometry.__geo_interface__

    # Handle regular dicts
    elif isinstance(geometry, dict):
        return validate_geometry(geometry)  # Use existing function

    # Handle other types
    else:
        raise ValidationError(
            "Geometry must be a dictionary or Shapely object",
            {"geometry": str(geometry)[:100], "type": type(geometry).__name__},
        )
