"""Tests for planetscope_py.utils module."""

import pytest
from datetime import datetime, timezone

from planetscope_py.utils import (
    validate_geometry, validate_date_range, validate_roi_size,
    validate_cloud_cover, validate_item_types, format_api_url,
    calculate_geometry_bounds, create_point_geometry, create_bbox_geometry,
    pretty_print_json, mask_api_key
)
from planetscope_py.exceptions import ValidationError


class TestValidateGeometry:
    """Test cases for geometry validation."""
    
    def test_valid_point(self):
        """Test validation of valid Point geometry."""
        geometry = {
            "type": "Point",
            "coordinates": [-122.4194, 37.7749]
        }
        result = validate_geometry(geometry)
        assert result == geometry
    
    def test_valid_polygon(self):
        """Test validation of valid Polygon geometry."""
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [-122.5, 37.7],
                [-122.3, 37.7],
                [-122.3, 37.8],
                [-122.5, 37.8],
                [-122.5, 37.7]
            ]]
        }
        result = validate_geometry(geometry)
        assert result == geometry
    
    def test_invalid_type_not_dict(self):
        """Test validation fails for non-dictionary input."""
        with pytest.raises(ValidationError) as exc_info:
            validate_geometry("not a dict")
        
        assert "must be a dictionary" in str(exc_info.value)
    
    def test_missing_required_field(self):
        """Test validation fails for missing required fields."""
        geometry = {"type": "Point"}  # Missing coordinates
        
        with pytest.raises(ValidationError) as exc_info:
            validate_geometry(geometry)
        
        assert "missing required field: coordinates" in str(exc_info.value)
    
    def test_invalid_geometry_type(self):
        """Test validation fails for invalid geometry type."""
        geometry = {
            "type": "InvalidType",
            "coordinates": [0, 0]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_geometry(geometry)
        
        assert "Invalid geometry type" in str(exc_info.value)
    
    def test_coordinates_out_of_bounds_longitude(self):
        """Test validation fails for longitude out of bounds."""
        geometry = {
            "type": "Point",
            "coordinates": [200, 0]  # Invalid longitude
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_geometry(geometry)
        
        assert "Longitude coordinates must be between -180 and 180" in str(exc_info.value)
    
    def test_coordinates_out_of_bounds_latitude(self):
        """Test validation fails for latitude out of bounds."""
        geometry = {
            "type": "Point", 
            "coordinates": [0, 100]  # Invalid latitude
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_geometry(geometry)
        
        assert "Latitude coordinates must be between -90 and 90" in str(exc_info.value)
    
    def test_polygon_not_closed(self):
        """Test validation fails for unclosed polygon."""
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [0, 0],
                [1, 0], 
                [1, 1],
                [0, 1]  # Not closed - missing [0, 0]
            ]]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_geometry(geometry)
        
        assert "must be closed" in str(exc_info.value)
    
    def test_polygon_too_few_coordinates(self):
        """Test validation fails for polygon with too few coordinates."""
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [0, 0],
                [1, 0],
                [0, 0]  # Only 3 coordinates
            ]]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_geometry(geometry)
        
        assert "Too few points" in str(exc_info.value)


class TestValidateDateRange:
    """Test cases for date range validation."""
    
    def test_valid_string_dates(self):
        """Test validation with valid string dates."""
        start, end = validate_date_range("2025-01-01", "2025-12-31")
        
        assert start.endswith("Z")
        assert end.endswith("Z")
        assert "2025-01-01" in start
        assert "2025-12-31" in end
    
    def test_valid_datetime_objects(self):
        """Test validation with datetime objects."""
        start_dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end_dt = datetime(2025, 12, 31, tzinfo=timezone.utc)
        
        start, end = validate_date_range(start_dt, end_dt)
        
        assert start.endswith("Z")
        assert end.endswith("Z")
    
    def test_start_after_end(self):
        """Test validation fails when start date is after end date."""
        with pytest.raises(ValidationError) as exc_info:
            validate_date_range("2025-12-31", "2025-01-01")
        
        assert "Start date must be before end date" in str(exc_info.value)
    
    def test_invalid_date_format(self):
        """Test validation fails for invalid date format."""
        with pytest.raises(ValidationError) as exc_info:
            validate_date_range("invalid-date", "2025-12-31")
        
        assert "Invalid date format" in str(exc_info.value)
    
    def test_invalid_date_type(self):
        """Test validation fails for invalid date type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_date_range(123, "2025-12-31")
        
        assert "must be string or datetime object" in str(exc_info.value)


class TestValidateCloudCover:
    """Test cases for cloud cover validation."""
    
    def test_valid_fraction(self):
        """Test validation with valid fraction (0-1)."""
        result = validate_cloud_cover(0.5)
        assert result == 0.5
    
    def test_valid_percentage(self):
        """Test validation with valid percentage (0-100)."""
        result = validate_cloud_cover(50.0)
        assert result == 0.5
    
    def test_edge_cases(self):
        """Test validation with edge cases."""
        assert validate_cloud_cover(0) == 0.0
        assert validate_cloud_cover(1) == 1.0
        assert validate_cloud_cover(100) == 1.0
    
    def test_invalid_negative(self):
        """Test validation fails for negative values."""
        with pytest.raises(ValidationError) as exc_info:
            validate_cloud_cover(-0.1)
        
        assert "cannot be negative" in str(exc_info.value)
    
    def test_invalid_over_100(self):
        """Test validation fails for values over 100%."""
        with pytest.raises(ValidationError) as exc_info:
            validate_cloud_cover(150)
        
        assert "cannot exceed 100%" in str(exc_info.value)
    
    def test_invalid_type(self):
        """Test validation fails for non-numeric types."""
        with pytest.raises(ValidationError) as exc_info:
            validate_cloud_cover("50%")
        
        assert "must be numeric" in str(exc_info.value)


class TestValidateItemTypes:
    """Test cases for item types validation."""
    
    def test_valid_item_types(self):
        """Test validation with valid item types."""
        item_types = ["PSScene", "REOrthoTile"]
        result = validate_item_types(item_types)
        assert result == item_types
    
    def test_single_item_type(self):
        """Test validation with single item type."""
        item_types = ["PSScene"]
        result = validate_item_types(item_types)
        assert result == item_types
    
    def test_empty_list(self):
        """Test validation fails for empty list."""
        with pytest.raises(ValidationError) as exc_info:
            validate_item_types([])
        
        assert "At least one item type required" in str(exc_info.value)
    
    def test_not_list(self):
        """Test validation fails for non-list input."""
        with pytest.raises(ValidationError) as exc_info:
            validate_item_types("PSScene")
        
        assert "must be a list" in str(exc_info.value)
    
    def test_invalid_item_type(self):
        """Test validation fails for invalid item type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_item_types(["InvalidType"])
        
        assert "Invalid item type: InvalidType" in str(exc_info.value)
        assert "valid_types" in exc_info.value.details
    
    def test_non_string_item_type(self):
        """Test validation fails for non-string item type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_item_types([123])
        
        assert "must be string" in str(exc_info.value)


class TestFormatApiUrl:
    """Test cases for API URL formatting."""
    
    def test_basic_url(self):
        """Test basic URL formatting."""
        url = format_api_url("https://api.planet.com/data/v1", "items")
        assert url == "https://api.planet.com/data/v1/items"
    
    def test_url_with_trailing_slash(self):
        """Test URL formatting with trailing slash in base."""
        url = format_api_url("https://api.planet.com/data/v1/", "items")
        assert url == "https://api.planet.com/data/v1/items"
    
    def test_url_with_leading_slash(self):
        """Test URL formatting with leading slash in endpoint."""
        url = format_api_url("https://api.planet.com/data/v1", "/items")
        assert url == "https://api.planet.com/data/v1/items"
    
    def test_url_with_parameters(self):
        """Test URL formatting with parameters."""
        url = format_api_url(
            "https://api.planet.com/data/v1",
            "items",
            limit=100,
            page=2
        )
        assert "limit=100" in url
        assert "page=2" in url
        assert url.count("?") == 1
    
    def test_url_with_none_parameters(self):
        """Test URL formatting ignores None parameters."""
        url = format_api_url(
            "https://api.planet.com/data/v1",
            "items", 
            limit=100,
            page=None
        )
        assert "limit=100" in url
        assert "page" not in url


class TestCalculateGeometryBounds:
    """Test cases for geometry bounds calculation."""
    
    def test_point_bounds(self):
        """Test bounds calculation for Point geometry."""
        geometry = {
            "type": "Point",
            "coordinates": [-122.4194, 37.7749]
        }
        bounds = calculate_geometry_bounds(geometry)
        
        # Point bounds should be the same coordinate repeated
        assert bounds == (-122.4194, 37.7749, -122.4194, 37.7749)
    
    def test_polygon_bounds(self):
        """Test bounds calculation for Polygon geometry."""
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [-122.5, 37.7],
                [-122.3, 37.7],
                [-122.3, 37.8],
                [-122.5, 37.8],
                [-122.5, 37.7]
            ]]
        }
        bounds = calculate_geometry_bounds(geometry)
        
        # Should return min/max coordinates
        assert bounds == (-122.5, 37.7, -122.3, 37.8)


class TestCreateGeometries:
    """Test cases for geometry creation utilities."""
    
    def test_create_point_geometry(self):
        """Test Point geometry creation."""
        point = create_point_geometry(-122.4194, 37.7749)
        
        assert point["type"] == "Point"
        assert point["coordinates"] == [-122.4194, 37.7749]
    
    def test_create_bbox_geometry(self):
        """Test bounding box geometry creation."""
        bbox = create_bbox_geometry(-122.5, 37.7, -122.3, 37.8)
        
        assert bbox["type"] == "Polygon"
        coords = bbox["coordinates"][0]
        
        # Check that bbox is properly formed
        assert len(coords) == 5  # Closed polygon
        assert coords[0] == coords[-1]  # First equals last
        assert coords[0] == [-122.5, 37.7]  # Bottom-left
        assert coords[2] == [-122.3, 37.8]  # Top-right


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_pretty_print_json(self):
        """Test JSON pretty printing."""
        data = {"key": "value", "number": 42}
        result = pretty_print_json(data)
        
        assert '"key": "value"' in result
        assert '"number": 42' in result
        assert result.count('\n') > 0  # Should be multi-line
    
    def test_mask_api_key_long(self):
        """Test API key masking for long keys."""
        api_key = "very_long_api_key_12345"
        masked = mask_api_key(api_key)
        
        assert "very" in masked
        assert "2345" in masked
        assert "..." in masked
        assert len(masked) < len(api_key)
    
    def test_mask_api_key_short(self):
        """Test API key masking for short keys."""
        api_key = "short"
        masked = mask_api_key(api_key)
        
        assert masked == "***"
    
    def test_mask_api_key_empty(self):
        """Test API key masking for empty string."""
        masked = mask_api_key("")
        assert masked == "***"


class TestValidateRoiSize:
    """Test cases for ROI size validation."""
    
    def test_small_valid_roi(self):
        """Test validation passes for small ROI."""
        # Small polygon around San Francisco
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [-122.5, 37.7],
                [-122.4, 37.7],
                [-122.4, 37.8],
                [-122.5, 37.8],
                [-122.5, 37.7]
            ]]
        }
        
        # Should not raise exception and return area
        area = validate_roi_size(geometry)
        assert isinstance(area, float)
        assert area > 0
    
    def test_point_roi(self):
        """Test validation for Point geometry (zero area)."""
        geometry = {
            "type": "Point",
            "coordinates": [-122.4194, 37.7749]
        }
        
        area = validate_roi_size(geometry)
        assert area == 0.0


# Integration tests that test multiple functions together
class TestIntegration:
    """Integration tests for utility functions."""
    
    def test_geometry_validation_and_bounds(self):
        """Test geometry validation followed by bounds calculation."""
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [-122.5, 37.7],
                [-122.3, 37.7],
                [-122.3, 37.8],
                [-122.5, 37.8],
                [-122.5, 37.7]
            ]]
        }
        
        # Validate geometry
        validated = validate_geometry(geometry)
        
        # Calculate bounds
        bounds = calculate_geometry_bounds(validated)
        
        assert bounds == (-122.5, 37.7, -122.3, 37.8)
    
    def test_create_and_validate_bbox(self):
        """Test creating bbox and then validating it."""
        bbox = create_bbox_geometry(-122.5, 37.7, -122.3, 37.8)
        
        # Should validate without errors
        validated = validate_geometry(bbox)
        assert validated == bbox
    
    @pytest.mark.parametrize("item_type", [
        "PSScene", "REOrthoTile", "REScene", "PSOrthoTile",
        "SkySatScene", "SkySatCollect", "Landsat8L1G"
    ])
    def test_all_valid_item_types(self, item_type):
        """Test that all documented item types are valid."""
        result = validate_item_types([item_type])
        assert result == [item_type]