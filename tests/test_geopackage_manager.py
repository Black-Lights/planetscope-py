#!/usr/bin/env python3
"""
Tests for geopackage_manager.py module.

Comprehensive test suite for GeoPackage creation and management functionality.
"""

import pytest
import tempfile
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

try:
    import geopandas as gpd
    import fiona
    import rasterio

    GEOPACKAGE_DEPS_AVAILABLE = True
except ImportError:
    GEOPACKAGE_DEPS_AVAILABLE = False

from shapely.geometry import box, Polygon

from planetscope_py.geopackage_manager import (
    GeoPackageManager,
    GeoPackageConfig,
    LayerInfo,
    RasterInfo,
)
from planetscope_py.metadata import MetadataProcessor
from planetscope_py.exceptions import ValidationError, PlanetScopeError


class TestGeoPackageConfig:
    """Test GeoPackageConfig dataclass."""

    def test_geopackage_config_defaults(self):
        """Test GeoPackageConfig default values."""
        config = GeoPackageConfig()

        assert config.include_imagery is False
        assert config.clip_to_roi is False
        assert config.imagery_format == "GeoTIFF"
        assert config.compression == "LZW"
        assert config.target_crs == "EPSG:4326"
        assert config.attribute_schema == "comprehensive"
        assert config.max_raster_size_mb == 100
        assert config.overview_levels == [2, 4, 8, 16]

    def test_geopackage_config_custom(self):
        """Test GeoPackageConfig with custom values."""
        config = GeoPackageConfig(
            include_imagery=True,
            clip_to_roi=True,
            imagery_format="COG",
            compression="DEFLATE",
            target_crs="EPSG:3857",
            attribute_schema="minimal",
            max_raster_size_mb=50,
            overview_levels=[2, 4],
        )

        assert config.include_imagery is True
        assert config.clip_to_roi is True
        assert config.imagery_format == "COG"
        assert config.compression == "DEFLATE"
        assert config.target_crs == "EPSG:3857"
        assert config.attribute_schema == "minimal"
        assert config.max_raster_size_mb == 50
        assert config.overview_levels == [2, 4]


class TestLayerInfo:
    """Test LayerInfo dataclass."""

    def test_layer_info_creation(self):
        """Test LayerInfo object creation."""
        layer_info = LayerInfo(
            name="test_layer",
            layer_type="vector",
            feature_count=100,
            geometry_type="Polygon",
            crs="EPSG:4326",
            bbox=(9.0, 45.4, 9.2, 45.6),
            created=datetime.now(),
        )

        assert layer_info.name == "test_layer"
        assert layer_info.layer_type == "vector"
        assert layer_info.feature_count == 100
        assert layer_info.geometry_type == "Polygon"
        assert layer_info.crs == "EPSG:4326"
        assert len(layer_info.bbox) == 4


class TestRasterInfo:
    """Test RasterInfo dataclass."""

    def test_raster_info_creation(self):
        """Test RasterInfo object creation."""
        raster_info = RasterInfo(
            original_path="/path/to/original.tif",
            processed_path="/path/to/processed.tif",
            scene_id="scene_001",
            asset_type="ortho_analytic_4b",
            file_size_mb=45.2,
            width=2048,
            height=2048,
            band_count=4,
            data_type="uint16",
            crs="EPSG:32632",
            bounds=(100000, 5000000, 102000, 5002000),
            clipped=True,
        )

        assert raster_info.original_path == "/path/to/original.tif"
        assert raster_info.processed_path == "/path/to/processed.tif"
        assert raster_info.scene_id == "scene_001"
        assert raster_info.asset_type == "ortho_analytic_4b"
        assert raster_info.file_size_mb == 45.2
        assert raster_info.width == 2048
        assert raster_info.height == 2048
        assert raster_info.band_count == 4
        assert raster_info.data_type == "uint16"
        assert raster_info.crs == "EPSG:32632"
        assert len(raster_info.bounds) == 4
        assert raster_info.clipped is True


@pytest.mark.skipif(
    not GEOPACKAGE_DEPS_AVAILABLE, reason="GeoPackage dependencies not available"
)
class TestGeoPackageManager:
    """Test GeoPackageManager class."""

    @pytest.fixture
    def mock_metadata_processor(self):
        """Mock MetadataProcessor for testing."""
        processor = Mock(spec=MetadataProcessor)
        processor.extract_scene_metadata.return_value = {
            "scene_id": "test_scene_001",
            "acquired": "2024-01-15T10:30:00Z",
            "cloud_cover": 0.15,
            "satellite_id": "Planet_001",
            "item_type": "PSScene",
            "overall_quality": 0.85,
            "acquisition_date": "2024-01-15",
            "sun_elevation": 35.2,
            "sun_azimuth": 145.8,
        }
        return processor

    @pytest.fixture
    def geopackage_manager(self, mock_metadata_processor):
        """Create GeoPackageManager instance for testing."""
        config = GeoPackageConfig(attribute_schema="standard")
        return GeoPackageManager(mock_metadata_processor, config)

    @pytest.fixture
    def sample_scenes(self):
        """Create sample scenes for testing."""
        scenes = []
        for i in range(5):
            scene = {
                "properties": {
                    "id": f"scene_{i:03d}",
                    "acquired": f"2024-01-{15+i:02d}T10:30:00Z",
                    "cloud_cover": 0.1 + i * 0.05,
                    "item_type": "PSScene",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [9.0 + i * 0.01, 45.4 + i * 0.01],
                            [9.1 + i * 0.01, 45.4 + i * 0.01],
                            [9.1 + i * 0.01, 45.5 + i * 0.01],
                            [9.0 + i * 0.01, 45.5 + i * 0.01],
                            [9.0 + i * 0.01, 45.4 + i * 0.01],
                        ]
                    ],
                },
            }
            scenes.append(scene)
        return scenes

    @pytest.fixture
    def sample_roi(self):
        """Create sample ROI for testing."""
        return box(9.0, 45.4, 9.15, 45.55)

    def test_geopackage_manager_initialization(self, geopackage_manager):
        """Test GeoPackageManager initialization."""
        assert geopackage_manager.config.attribute_schema == "standard"
        assert geopackage_manager.metadata_processor is not None
        assert "minimal" in geopackage_manager.attribute_schemas
        assert "standard" in geopackage_manager.attribute_schemas
        assert "comprehensive" in geopackage_manager.attribute_schemas
        assert len(geopackage_manager.processed_rasters) == 0

    def test_create_scene_geopackage_empty_scenes(self, geopackage_manager):
        """Test GeoPackage creation with empty scenes list."""
        with pytest.raises(ValidationError, match="No scenes provided"):
            geopackage_manager.create_scene_geopackage([], "test.gpkg")

    def test_process_scenes_for_geopackage(
        self, geopackage_manager, sample_scenes, sample_roi
    ):
        """Test scene processing for GeoPackage creation."""
        processed = geopackage_manager._process_scenes_for_geopackage(
            sample_scenes, sample_roi
        )

        assert len(processed) > 0
        assert len(processed) <= len(sample_scenes)

        # Check that each processed scene has required fields
        for scene_data in processed:
            assert "geometry" in scene_data
            assert "area_km2" in scene_data
            assert "centroid_lat" in scene_data
            assert "centroid_lon" in scene_data
            assert "scene_id" in scene_data

    def test_create_scene_geodataframe(
        self, geopackage_manager, sample_scenes, sample_roi
    ):
        """Test GeoDataFrame creation from scene data."""
        processed_scenes = geopackage_manager._process_scenes_for_geopackage(
            sample_scenes, sample_roi
        )
        gdf = geopackage_manager._create_scene_geodataframe(processed_scenes)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == len(processed_scenes)
        assert gdf.crs.to_string() == geopackage_manager.config.target_crs

        # Check that schema fields are present
        schema = geopackage_manager.attribute_schemas["standard"]
        for field_name in schema.keys():
            if field_name != "geometry":
                assert field_name in gdf.columns or field_name in [
                    "scene_id"
                ]  # May be mapped

    def test_create_scene_geopackage_success(
        self, geopackage_manager, sample_scenes, sample_roi
    ):
        """Test successful GeoPackage creation."""
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            result_path = geopackage_manager.create_scene_geopackage(
                scenes=sample_scenes,
                output_path=output_path,
                roi=sample_roi,
                layer_name="test_scenes",
            )

            assert result_path == output_path
            assert Path(output_path).exists()

            # Check that GeoPackage contains expected layers
            layers = fiona.listlayers(output_path)
            assert "test_scenes" in layers
            assert "analysis_summary" in layers

            # Verify scene layer content
            gdf = gpd.read_file(output_path, layer="test_scenes")
            assert len(gdf) > 0
            assert "geometry" in gdf.columns

        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_group_raster_files(self, geopackage_manager):
        """Test raster file grouping functionality."""
        files = [
            "/path/scene_001_ortho_analytic_4b.tif",
            "/path/scene_002_ortho_analytic_4b.tif",
            "/path/scene_001_ortho_visual.tif",
            "/path/scene_003_ortho_analytic_4b.tif",
            "/path/random_file.tif",
        ]

        groups = geopackage_manager._group_raster_files(files)

        assert "ortho_analytic_4b" in groups
        assert "ortho_visual" in groups
        assert len(groups["ortho_analytic_4b"]) == 3  # Three analytic files
        assert len(groups["ortho_visual"]) == 1  # One visual file

        # Check that ungrouped files go to misc
        if "misc_rasters" in groups:
            assert "/path/random_file.tif" in groups["misc_rasters"]

    def test_attribute_schemas(self, geopackage_manager):
        """Test attribute schema definitions."""
        minimal_schema = geopackage_manager._get_minimal_schema()
        standard_schema = geopackage_manager._get_standard_schema()
        comprehensive_schema = geopackage_manager._get_comprehensive_schema()

        # Check minimal schema
        assert "scene_id" in minimal_schema
        assert "acquired" in minimal_schema
        assert "cloud_cover" in minimal_schema
        assert "area_km2" in minimal_schema

        # Check that standard includes minimal
        for field in minimal_schema:
            assert field in standard_schema

        # Check that comprehensive includes standard
        for field in standard_schema:
            assert field in comprehensive_schema

        # Check that comprehensive has additional fields
        assert len(comprehensive_schema) > len(standard_schema)
        assert len(standard_schema) > len(minimal_schema)

        # Check field type specifications
        for schema in [minimal_schema, standard_schema, comprehensive_schema]:
            for field_name, field_config in schema.items():
                assert "type" in field_config
                assert field_config["type"] in ["TEXT", "REAL", "INTEGER", "DATE"]
                assert "description" in field_config

    def test_get_geopackage_info(self, geopackage_manager, sample_scenes, sample_roi):
        """Test GeoPackage information extraction."""
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Create a test GeoPackage
            geopackage_manager.create_scene_geopackage(
                scenes=sample_scenes, output_path=output_path, roi=sample_roi
            )

            # Get info
            info = geopackage_manager.get_geopackage_info(output_path)

            assert "file_path" in info
            assert "file_size_mb" in info
            assert "total_layers" in info
            assert "layer_info" in info
            assert "created" in info

            assert info["file_path"] == output_path
            assert info["file_size_mb"] > 0
            assert info["total_layers"] > 0
            assert len(info["layer_info"]) == info["total_layers"]

        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_get_geopackage_info_nonexistent(self, geopackage_manager):
        """Test GeoPackage info for nonexistent file."""
        info = geopackage_manager.get_geopackage_info("nonexistent.gpkg")
        assert "error" in info

    def test_export_geopackage_report(
        self, geopackage_manager, sample_scenes, sample_roi
    ):
        """Test GeoPackage report export."""
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp_file:
            gpkg_path = tmp_file.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            report_path = tmp_file.name

        try:
            # Create GeoPackage
            geopackage_manager.create_scene_geopackage(
                scenes=sample_scenes, output_path=gpkg_path, roi=sample_roi
            )

            # Export report
            result_path = geopackage_manager.export_geopackage_report(
                gpkg_path, report_path
            )

            assert result_path == report_path
            assert Path(report_path).exists()

            # Check report content
            import json

            with open(report_path, "r") as f:
                report = json.load(f)

            assert "geopackage_analysis" in report
            assert "processed_rasters" in report
            assert "configuration" in report
            assert "generated_at" in report
            assert "generator" in report

        finally:
            for path in [gpkg_path, report_path]:
                if Path(path).exists():
                    Path(path).unlink()


@pytest.mark.skipif(
    not GEOPACKAGE_DEPS_AVAILABLE, reason="GeoPackage dependencies not available"
)
class TestGeoPackageManagerWithRasters:
    """Test GeoPackageManager with raster functionality."""

    @pytest.fixture
    def mock_metadata_processor(self):
        """Mock MetadataProcessor for testing."""
        processor = Mock(spec=MetadataProcessor)
        processor.extract_scene_metadata.return_value = {
            "scene_id": "test_scene_001",
            "acquired": "2024-01-15T10:30:00Z",
            "cloud_cover": 0.15,
        }
        return processor

    @pytest.fixture
    def geopackage_manager_with_imagery(self, mock_metadata_processor):
        """Create GeoPackageManager with imagery enabled."""
        config = GeoPackageConfig(
            include_imagery=True, clip_to_roi=True, max_raster_size_mb=50
        )
        return GeoPackageManager(mock_metadata_processor, config)

    @pytest.fixture
    def mock_raster_file(self):
        """Create mock raster file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
            raster_path = tmp_file.name

        # Create a simple mock raster
        import numpy as np

        # Create a proper mock that supports context manager protocol
        class MockRasterDataset:
            def __init__(self):
                self.profile = {
                    "driver": "GTiff",
                    "dtype": "uint16",
                    "width": 100,
                    "height": 100,
                    "count": 4,
                    "crs": "EPSG:4326",
                }
                self.bounds = (9.0, 45.4, 9.1, 45.5)
                self.width = 100
                self.height = 100
                self.count = 4
                self.dtypes = ["uint16"] * 4
                self.nodata = None
                self.transform = [0.001, 0.0, 9.0, 0.0, -0.001, 45.5]

                # Mock CRS without using real rasterio CRS
                mock_crs = Mock()
                mock_crs.__str__ = Mock(return_value="EPSG:4326")
                mock_crs.to_string = Mock(return_value="EPSG:4326")
                self.crs = mock_crs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_dataset = MockRasterDataset()

        yield Path(raster_path), mock_dataset

        # Cleanup
        try:
            Path(raster_path).unlink()
        except FileNotFoundError:
            pass

    def test_add_single_raster_reference(
        self, geopackage_manager_with_imagery, mock_raster_file
    ):
        """Test adding single raster reference to GeoPackage."""
        raster_path, mock_dataset = mock_raster_file

        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp_file:
            gpkg_path = Path(tmp_file.name)

        try:
            with patch("rasterio.open", return_value=mock_dataset):
                raster_info = geopackage_manager_with_imagery._add_single_raster(
                    output_file=gpkg_path,
                    raster_path=Path(raster_path),
                    roi=None,
                    layer_name="test_raster",
                )

                assert raster_info is not None
                assert isinstance(raster_info, RasterInfo)
                assert raster_info.width == 100
                assert raster_info.height == 100
                assert raster_info.band_count == 4
                assert raster_info.data_type == "uint16"

        finally:
            for path in [raster_path, gpkg_path]:
                if Path(path).exists():
                    Path(path).unlink()

    def test_create_raster_inventory_layer(self, geopackage_manager_with_imagery):
        """Test raster inventory layer creation."""
        # Add some mock raster info
        raster_info1 = RasterInfo(
            original_path="/path/scene_001_ortho_analytic_4b.tif",
            processed_path="/path/scene_001_ortho_analytic_4b.tif",
            scene_id="scene_001",
            asset_type="ortho_analytic_4b",
            file_size_mb=25.5,
            width=1024,
            height=1024,
            band_count=4,
            data_type="uint16",
            crs="EPSG:4326",
            bounds=(9.0, 45.4, 9.1, 45.5),
            clipped=False,
        )

        raster_info2 = RasterInfo(
            original_path="/path/scene_002_ortho_visual.tif",
            processed_path="/path/scene_002_ortho_visual.tif",
            scene_id="scene_002",
            asset_type="ortho_visual",
            file_size_mb=18.2,
            width=512,
            height=512,
            band_count=3,
            data_type="uint8",
            crs="EPSG:4326",
            bounds=(9.05, 45.45, 9.15, 45.55),
            clipped=True,
        )

        geopackage_manager_with_imagery.processed_rasters = [raster_info1, raster_info2]

        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp_file:
            gpkg_path = Path(tmp_file.name)

        try:
            geopackage_manager_with_imagery._create_raster_inventory_layer(gpkg_path)

            # Check that inventory layer was created
            layers = fiona.listlayers(gpkg_path)
            assert "raster_inventory" in layers

            # Check inventory content
            inventory_gdf = gpd.read_file(gpkg_path, layer="raster_inventory")
            assert len(inventory_gdf) == 2
            assert "scene_id" in inventory_gdf.columns
            assert "asset_type" in inventory_gdf.columns
            assert "file_size_mb" in inventory_gdf.columns
            assert "clipped_to_roi" in inventory_gdf.columns

            # Check values
            assert inventory_gdf["scene_id"].tolist() == ["scene_001", "scene_002"]
            assert inventory_gdf["asset_type"].tolist() == [
                "ortho_analytic_4b",
                "ortho_visual",
            ]
            assert inventory_gdf["clipped_to_roi"].tolist() == [False, True]

        finally:
            if gpkg_path.exists():
                gpkg_path.unlink()

    def test_add_imagery_to_existing_geopackage(
        self, geopackage_manager_with_imagery, mock_raster_file
    ):
        """Test adding imagery to existing GeoPackage."""
        raster_path, mock_dataset = mock_raster_file

        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp_file:
            gpkg_path = tmp_file.name

        try:
            # Create initial GeoPackage (empty)
            Path(gpkg_path).touch()

            with patch("rasterio.open", return_value=mock_dataset):
                result = (
                    geopackage_manager_with_imagery.add_imagery_to_existing_geopackage(
                        geopackage_path=gpkg_path, downloaded_files=[raster_path]
                    )
                )

                assert result is True
                assert len(geopackage_manager_with_imagery.processed_rasters) > 0

        finally:
            for path in [raster_path, gpkg_path]:
                if Path(path).exists():
                    Path(path).unlink()

    def test_cleanup_temporary_files(self, geopackage_manager_with_imagery):
        """Test temporary file cleanup."""
        # Create temporary directory structure
        import tempfile

        temp_dir = Path(tempfile.gettempdir()) / "planetscope_py_clipped"
        temp_dir.mkdir(exist_ok=True)

        # Create some test files
        test_file = temp_dir / "test_clipped.tif"
        test_file.touch()

        assert test_file.exists()

        # Cleanup
        geopackage_manager_with_imagery.cleanup_temporary_files()

        # Directory should be removed
        assert not temp_dir.exists()


class TestGeoPackageManagerHelperMethods:
    """Test helper methods of GeoPackageManager."""

    @pytest.fixture
    def geopackage_manager(self):
        """Create basic GeoPackageManager for testing."""
        return GeoPackageManager()

    def test_generate_qgis_style(self, geopackage_manager):
        """Test QGIS style generation."""
        style_xml = geopackage_manager._generate_qgis_style()

        assert isinstance(style_xml, str)
        assert '<?xml version="1.0" encoding="UTF-8"?>' in style_xml
        assert "<qgis version=" in style_xml
        assert '<renderer-v2 type="singleSymbol">' in style_xml
        assert "SimpleFill" in style_xml

    def test_validate_geopackage_nonexistent(self, geopackage_manager):
        """Test validation of nonexistent GeoPackage."""
        result = geopackage_manager._validate_geopackage(Path("nonexistent.gpkg"))
        assert result is False

    @pytest.mark.skipif(
        not GEOPACKAGE_DEPS_AVAILABLE, reason="GeoPackage dependencies not available"
    )
    def test_validate_geopackage_empty(self, geopackage_manager):
        """Test validation of empty GeoPackage."""
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp_file:
            gpkg_path = Path(tmp_file.name)

        try:
            # Create empty GeoPackage
            gpkg_path.touch()

            result = geopackage_manager._validate_geopackage(gpkg_path)
            assert result is False  # Empty file should fail validation

        finally:
            if gpkg_path.exists():
                gpkg_path.unlink()


@pytest.mark.skipif(
    not GEOPACKAGE_DEPS_AVAILABLE, reason="GeoPackage dependencies not available"
)
class TestGeoPackageManagerIntegration:
    """Integration tests for GeoPackageManager."""

    def test_full_workflow_with_clipping(self):
        """Test complete GeoPackage workflow with ROI clipping."""
        # Create test data
        scenes = [
            {
                "properties": {
                    "id": "integration_scene_001",
                    "acquired": "2024-01-15T10:30:00Z",
                    "cloud_cover": 0.1,
                    "item_type": "PSScene",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [8.9, 45.3],
                            [9.3, 45.3],
                            [9.3, 45.7],
                            [8.9, 45.7],
                            [8.9, 45.3],
                        ]
                    ],
                },
            },
            {
                "properties": {
                    "id": "integration_scene_002",
                    "acquired": "2024-01-16T10:30:00Z",
                    "cloud_cover": 0.15,
                    "item_type": "PSScene",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [9.0, 45.4],
                            [9.2, 45.4],
                            [9.2, 45.6],
                            [9.0, 45.6],
                            [9.0, 45.4],
                        ]
                    ],
                },
            },
        ]

        roi = box(9.05, 45.45, 9.15, 45.55)  # Smaller ROI for clipping test

        # Configure manager
        config = GeoPackageConfig(clip_to_roi=True, attribute_schema="minimal")

        # Mock metadata processor
        mock_processor = Mock(spec=MetadataProcessor)
        mock_processor.extract_scene_metadata.side_effect = [
            {
                "scene_id": "integration_scene_001",
                "acquired": "2024-01-15T10:30:00Z",
                "cloud_cover": 0.1,
                "area_km2": 100.0,
            },
            {
                "scene_id": "integration_scene_002",
                "acquired": "2024-01-16T10:30:00Z",
                "cloud_cover": 0.15,
                "area_km2": 50.0,
            },
        ]

        manager = GeoPackageManager(mock_processor, config)

        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Create GeoPackage
            result_path = manager.create_scene_geopackage(
                scenes=scenes,
                output_path=output_path,
                roi=roi,
                layer_name="integration_test",
            )

            assert result_path == output_path
            assert Path(output_path).exists()

            # Verify content
            layers = fiona.listlayers(output_path)
            assert "integration_test" in layers

            # Check that clipping worked (should have fewer/smaller geometries)
            gdf = gpd.read_file(output_path, layer="integration_test")

            # All geometries should be within or intersect the ROI
            for geom in gdf.geometry:
                assert geom.intersects(roi) or roi.contains(geom)

            # Check metadata table
            conn = sqlite3.connect(output_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='geopackage_metadata'"
            )
            assert cursor.fetchone() is not None

            cursor.execute("SELECT * FROM geopackage_metadata")
            metadata_row = cursor.fetchone()
            assert metadata_row is not None

            conn.close()

        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_error_handling_and_recovery(self):
        """Test error handling during GeoPackage creation."""
        # Test with invalid scenes
        invalid_scenes = [
            {
                "properties": {"id": "bad_scene"},
                "geometry": "invalid_geometry",  # Invalid geometry
            }
        ]

        manager = GeoPackageManager()

        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Should handle errors gracefully
            with pytest.raises(PlanetScopeError):
                manager.create_scene_geopackage(
                    scenes=invalid_scenes, output_path=output_path
                )

            # Output file should be cleaned up after failure
            assert not Path(output_path).exists()

        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
