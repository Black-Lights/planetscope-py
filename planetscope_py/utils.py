"""Core utility functions for planetscope-py.

This module provides essential validation, transformation, and helper functions
used throughout the library.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Union

from pyproj import Transformer
from shapely.geometry import shape
from shapely.validation import explain_validity

from .config import PlanetScopeConfig
from .exceptions import ValidationError


def validate_geometry(geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Validate GeoJSON geometry object.

    Args:
        geometry: GeoJSON geometry dictionary

    Returns:
        Validated and normalized geometry

    Raises:
        ValidationError: If geometry is invalid

    Example:
        valid_geom = validate_geometry({
            "type": "Polygon",
            "coordinates": [[[lon1, lat1], [lon2, lat2], ...]]
        })
    """
    if not isinstance(geometry, dict):
        raise ValidationError(
            "Geometry must be a dictionary",
            {"geometry": geometry, "type": type(geometry).__name__},
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
                        "Polygon rings must be closed (first and last coordinates equal)",
                        {
                            "geometry": geometry,
                            "first": ring[0],
                            "last": ring[-1],
                        },
                    )

        # Check vertex limit
        config = PlanetScopeConfig()
        if hasattr(geom_obj, "exterior") and geom_obj.exterior:
            vertex_count = len(geom_obj.exterior.coords)
        else:
            vertex_count = (
                len(coords)
                if geom_type == "Point"
                else len(coords[0]) if coords else 0
            )

        if vertex_count > config.MAX_GEOMETRY_VERTICES:
            raise ValidationError(
                f"Geometry has too many vertices: {vertex_count} > {config.MAX_GEOMETRY_VERTICES}",
                {"geometry": geometry, "vertex_count": vertex_count},
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
    """

    def parse_date(date_input: Union[str, datetime]) -> datetime:
        """Parse date input to datetime object."""
        if isinstance(date_input, datetime):
            return date_input

        if isinstance(date_input, str):
            # Common date formats to try
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
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)

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
        transformer = Transformer.from_crs(
            "EPSG:4326", "ESRI:54009", always_xy=True
        )
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
