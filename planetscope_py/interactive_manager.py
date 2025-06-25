#!/usr/bin/env python3
"""
Interactive Manager for PlanetScope-py
Simple interactive ROI selection using Folium maps.

This module provides an easy way to select regions of interest (ROI) 
interactively using web maps, eliminating the need to manually define coordinates.

Features:
- Interactive map with drawing tools
- Click to select ROI polygons
- Export to Shapely geometries
- Integration with all PlanetScope-py analysis functions
- Support for multiple ROI selection
- Coordinate validation and area calculation

Author: Ammar & Umayr
Version: 4.1.0 (Enhanced + Metadata Fixes + JSON Serialization)
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import json
from pathlib import Path

try:
    import folium
    from folium import plugins
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False

try:
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union
    _SHAPELY_AVAILABLE = True
except ImportError:
    _SHAPELY_AVAILABLE = False

try:
    import geopandas as gpd
    _GEOPANDAS_AVAILABLE = True
except ImportError:
    _GEOPANDAS_AVAILABLE = False

from .exceptions import ValidationError, PlanetScopeError
from .utils import validate_geometry, calculate_area_km2

logger = logging.getLogger(__name__)


class InteractiveManager:
    """
    Interactive ROI selection manager using Folium maps.
    
    Provides easy-to-use interactive map interface for selecting
    regions of interest for PlanetScope analysis.
    
    Features:
    - Interactive web maps with drawing tools
    - ROI polygon selection and validation
    - Export to multiple formats (Shapely, GeoJSON, Shapefile)
    - Integration with analysis functions
    - Area calculation and coordinate validation
    """
    
    def __init__(self, center_location: Optional[Tuple[float, float]] = None):
        """
        Initialize the interactive manager.
        
        Args:
            center_location: (lat, lon) tuple for map center (default: Milan, Italy)
        """
        self._check_dependencies()
        
        # Default to Milan, Italy if no center provided
        self.center_location = center_location or (45.4642, 9.1900)
        self.selected_polygons = []
        self.current_map = None
        
        logger.info("Interactive manager initialized")
        logger.info(f"Map center: {self.center_location}")
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        missing_deps = []
        
        if not _FOLIUM_AVAILABLE:
            missing_deps.append("folium")
        if not _SHAPELY_AVAILABLE:
            missing_deps.append("shapely")
        
        if missing_deps:
            raise ImportError(
                f"Missing required dependencies for interactive manager: {', '.join(missing_deps)}. "
                f"Install with: pip install {' '.join(missing_deps)}"
            )
    
    def create_selection_map(
        self, 
        zoom_start: int = 10,
        width: str = "100%",
        height: str = "600px",
        tiles: str = "OpenStreetMap"
    ) -> folium.Map:
        """
        Create an interactive map for ROI selection.
        
        Args:
            zoom_start: Initial zoom level (default: 10)
            width: Map width (default: "100%")
            height: Map height (default: "600px") 
            tiles: Map tiles to use (default: "OpenStreetMap")
            
        Returns:
            folium.Map: Interactive map with drawing tools
        """
        # Create base map
        map_obj = folium.Map(
            location=self.center_location,
            zoom_start=zoom_start,
            width=width,
            height=height,
            tiles=tiles
        )
        
        # Add drawing tools
        draw = plugins.Draw(
            export=True,
            filename='roi_selection.geojson',
            position='topleft',
            draw_options={
                'polyline': False,
                'polygon': {
                    'allowIntersection': False,
                    'showArea': True,
                    'metric': ['km', 'm'],
                    'shapeOptions': {
                        'color': '#ff0000',
                        'weight': 2,
                        'fillOpacity': 0.2
                    }
                },
                'circle': False,
                'rectangle': {
                    'shapeOptions': {
                        'color': '#0000ff',
                        'weight': 2,
                        'fillOpacity': 0.2
                    }
                },
                'marker': False,
                'circlemarker': False,
            },
            edit_options={
                'edit': True,
                'remove': True
            }
        )
        
        map_obj.add_child(draw)
        
        # Add fullscreen capability
        plugins.Fullscreen(
            position='topleft',
            title='Open fullscreen map',
            title_cancel='Close fullscreen map',
            force_separate_button=True
        ).add_to(map_obj)
        
        # Add measurement tool
        plugins.MeasureControl(
            position='topright',
            primary_length_unit='kilometers',
            secondary_length_unit='meters',
            primary_area_unit='sqkilometers',
            secondary_area_unit='acres'
        ).add_to(map_obj)
        
        # Store map reference
        self.current_map = map_obj
        
        logger.info("Interactive selection map created")
        return map_obj
    
    def display_map_with_instructions(self) -> folium.Map:
        """
        Create and display map with user instructions.
        
        Returns:
            folium.Map: Map with instructions overlay
        """
        map_obj = self.create_selection_map()
        
        # Add Jupyter-optimized instructions popup
        instructions_html = """
        <div style="font-family: Arial, sans-serif; padding: 15px; max-width: 450px;">
            <h3 style="color: #2E86AB; margin-top: 0;"> Jupyter ROI Selection</h3>
            
            <h4> How to Select Your Region of Interest:</h4>
            <ol style="line-height: 1.6;">
                <li><strong>Polygon Tool:</strong> Click the polygon icon (⬟) to draw custom shapes</li>
                <li><strong>Rectangle Tool:</strong> Click the rectangle icon (□) for simple boxes</li>
                <li><strong>Draw ROI:</strong> Click points on the map to create your region</li>
                <li><strong>Edit:</strong> Use edit tools to modify your selection</li>
                <li><strong>Delete:</strong> Use trash icon to remove shapes</li>
            </ol>
            
            <h4> Jupyter Workflow:</h4>
            <ol style="line-height: 1.6; background-color: #FFF3CD; padding: 10px; border-radius: 5px;">
                <li><strong>Draw ROI</strong> on this map</li>
                <li><strong>Export GeoJSON</strong> using the export button</li>
                <li><strong>Load in next cell:</strong> <code>selector.load_roi_from_file('roi_selection.geojson')</code></li>
                <li><strong>Run analysis:</strong> <code>analyze_roi_density(roi, period)</code></li>
            </ol>
            
            <h4> Features Available:</h4>
            <ul style="line-height: 1.6;">
                <li> <strong>Fullscreen:</strong> Top-left button</li>
                <li> <strong>Measurements:</strong> Top-right tool</li>
                <li> <strong>Area Display:</strong> Shown while drawing</li>
                <li> <strong>Export:</strong> Download as GeoJSON</li>
            </ul>
            
            <div style="background-color: #E8F4FD; padding: 12px; border-radius: 5px; margin-top: 15px;">
                <strong> Jupyter Tips:</strong><br>
                • This map displays directly in your notebook cell<br>
                • Export your ROI, then load it in the next cell<br>
                • Use <code>quick_roi_analysis()</code> for instant results<br>
                • All visualizations will appear in notebook outputs
            </div>
            
            <div style="background-color: #D4EDDA; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <strong> Quick Start:</strong><br>
                <code style="font-size: 11px;">
                # After drawing ROI:<br>
                roi = selector.get_selected_roi()<br>
                result = analyze_roi_density(roi, "last_month")<br>
                </code>
            </div>
        </div>
        """
        
        # Add instructions popup to center of map
        folium.Popup(
            folium.Html(instructions_html, script=True),
            max_width=450,
            show=True
        ).add_to(
            folium.Marker(
                self.center_location,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(map_obj)
        )
        
        return map_obj
    
    def extract_polygons_from_geojson(self, geojson_data: Union[str, dict]) -> List[Polygon]:
        """
        Extract Shapely polygons from GeoJSON data.
        
        Args:
            geojson_data: GeoJSON as string or dictionary
            
        Returns:
            List[Polygon]: List of Shapely polygon objects
        """
        if isinstance(geojson_data, str):
            try:
                geojson_data = json.loads(geojson_data)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid GeoJSON string: {e}")
        
        polygons = []
        
        if 'features' in geojson_data:
            features = geojson_data['features']
        elif 'type' in geojson_data and geojson_data['type'] == 'Feature':
            features = [geojson_data]
        else:
            raise ValidationError("Invalid GeoJSON structure")
        
        for feature in features:
            try:
                geometry = feature.get('geometry', {})
                geom_type = geometry.get('type')
                
                if geom_type == 'Polygon':
                    coords = geometry['coordinates'][0]  # Exterior ring
                    polygon = Polygon(coords)
                    
                    # Validate geometry
                    if validate_geometry(polygon):
                        polygons.append(polygon)
                        area_km2 = calculate_area_km2(polygon)
                        logger.info(f"Extracted polygon: {area_km2:.2f} km²")
                    else:
                        logger.warning("Invalid polygon geometry, skipping")
                
                elif geom_type == 'MultiPolygon':
                    for poly_coords in geometry['coordinates']:
                        polygon = Polygon(poly_coords[0])  # Exterior ring of each polygon
                        if validate_geometry(polygon):
                            polygons.append(polygon)
                
            except Exception as e:
                logger.warning(f"Failed to extract polygon from feature: {e}")
                continue
        
        self.selected_polygons = polygons
        logger.info(f"Extracted {len(polygons)} valid polygons")
        
        return polygons
    
    def load_roi_from_file(self, file_path: Union[str, Path]) -> List[Polygon]:
        """
        Load ROI polygons from file (GeoJSON, Shapefile, etc.).
        
        Args:
            file_path: Path to geometry file
            
        Returns:
            List[Polygon]: List of loaded polygons
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.geojson':
                with open(file_path, 'r') as f:
                    geojson_data = json.load(f)
                return self.extract_polygons_from_geojson(geojson_data)
            
            elif _GEOPANDAS_AVAILABLE and file_path.suffix.lower() in ['.shp', '.gpkg']:
                gdf = gpd.read_file(file_path)
                polygons = []
                
                for geometry in gdf.geometry:
                    if geometry.geom_type == 'Polygon':
                        if validate_geometry(geometry):
                            polygons.append(geometry)
                    elif geometry.geom_type == 'MultiPolygon':
                        for poly in geometry.geoms:
                            if validate_geometry(poly):
                                polygons.append(poly)
                
                self.selected_polygons = polygons
                logger.info(f"Loaded {len(polygons)} polygons from {file_path}")
                return polygons
            
            else:
                raise ValidationError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            raise PlanetScopeError(f"Failed to load ROI from file: {e}")
    
    def create_preset_locations_map(self) -> folium.Map:
        """
        Create map with preset location options for quick selection.
        
        Returns:
            folium.Map: Map with preset ROI options
        """
        map_obj = self.create_selection_map()
        
        # Define preset locations
        preset_locations = {
            "Milan, Italy": {
                "center": (45.4642, 9.1900),
                "polygon": Polygon([
                    [8.7, 45.1], [9.8, 44.9], [10.3, 45.3], [10.1, 45.9],
                    [9.5, 46.2], [8.9, 46.0], [8.5, 45.6], [8.7, 45.1]
                ]),
                "color": "red"
            },
            "Rome, Italy": {
                "center": (41.9028, 12.4964),
                "polygon": Polygon([
                    [12.3, 41.8], [12.7, 41.8], [12.7, 42.0], [12.3, 42.0], [12.3, 41.8]
                ]),
                "color": "blue"
            },
            "San Francisco, USA": {
                "center": (37.7749, -122.4194),
                "polygon": Polygon([
                    [-122.5, 37.7], [-122.3, 37.7], [-122.3, 37.8], [-122.5, 37.8], [-122.5, 37.7]
                ]),
                "color": "green"
            },
            "London, UK": {
                "center": (51.5074, -0.1278),
                "polygon": Polygon([
                    [-0.2, 51.4], [0.0, 51.4], [0.0, 51.6], [-0.2, 51.6], [-0.2, 51.4]
                ]),
                "color": "purple"
            }
        }
        
        # Add preset locations to map
        for name, location_data in preset_locations.items():
            center = location_data["center"]
            polygon = location_data["polygon"]
            color = location_data["color"]
            
            # Add polygon to map
            folium.GeoJson(
                {
                    "type": "Feature",
                    "geometry": polygon.__geo_interface__
                },
                style_function=lambda feature, color=color: {
                    'fillColor': color,
                    'color': color,
                    'weight': 2,
                    'fillOpacity': 0.3,
                },
                popup=folium.Popup(
                    f"<b>{name}</b><br>Area: {calculate_area_km2(polygon):.1f} km²<br>Click to select this ROI",
                    max_width=200
                ),
                tooltip=name
            ).add_to(map_obj)
            
            # Add marker at center
            folium.Marker(
                center,
                popup=f"<b>{name}</b><br>Preset ROI Location",
                icon=folium.Icon(color=color.replace('purple', 'blue'), icon='map-marker')
            ).add_to(map_obj)
        
        return map_obj
    
    def export_roi_to_formats(
        self, 
        polygons: Optional[List[Polygon]] = None,
        output_dir: str = "./roi_exports"
    ) -> Dict[str, str]:
        """
        Export selected ROI to multiple formats.
        
        Args:
            polygons: List of polygons to export (uses selected_polygons if None)
            output_dir: Output directory for exports
            
        Returns:
            Dict[str, str]: Dictionary of format -> file_path
        """
        polygons = polygons or self.selected_polygons
        
        if not polygons:
            raise ValidationError("No polygons to export")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        exported_files = {}
        
        try:
            # Export as GeoJSON
            geojson_path = output_path / "roi_selection.geojson"
            geojson_data = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": polygon.__geo_interface__,
                        "properties": {
                            "id": i,
                            "area_km2": calculate_area_km2(polygon)
                        }
                    }
                    for i, polygon in enumerate(polygons)
                ]
            }
            
            with open(geojson_path, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            exported_files["geojson"] = str(geojson_path)
            
            # Export as Shapefile (if GeoPandas available)
            if _GEOPANDAS_AVAILABLE:
                shapefile_path = output_path / "roi_selection.shp"
                gdf = gpd.GeoDataFrame({
                    'id': range(len(polygons)),
                    'area_km2': [calculate_area_km2(p) for p in polygons],
                    'geometry': polygons
                }, crs='EPSG:4326')
                
                gdf.to_file(shapefile_path)
                exported_files["shapefile"] = str(shapefile_path)
            
            logger.info(f"Exported ROI to {len(exported_files)} formats")
            return exported_files
            
        except Exception as e:
            raise PlanetScopeError(f"Failed to export ROI: {e}")
    
    def get_selected_roi(self, union_polygons: bool = False) -> Union[Polygon, List[Polygon]]:
        """
        Get selected ROI polygon(s).
        
        Args:
            union_polygons: If True, return single union polygon; if False, return list
            
        Returns:
            Union[Polygon, List[Polygon]]: Selected ROI(s)
        """
        if not self.selected_polygons:
            raise ValidationError("No ROI selected. Please select polygons first.")
        
        if union_polygons and len(self.selected_polygons) > 1:
            return unary_union(self.selected_polygons)
        elif union_polygons:
            return self.selected_polygons[0]
        else:
            return self.selected_polygons
    
    def quick_roi_analysis(
        self, 
        time_period: str = "last_month",
        analysis_type: str = "spatial",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Quick analysis using selected ROI.
        
        Args:
            time_period: Time period for analysis
            analysis_type: "spatial", "temporal", or "both"
            **kwargs: Additional analysis parameters
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        if not self.selected_polygons:
            raise ValidationError("No ROI selected. Please select polygons first.")
        
        # Use first polygon or union of all polygons
        roi = self.get_selected_roi(union_polygons=True)
        
        try:
            # Import analysis functions dynamically
            if analysis_type == "spatial":
                from . import analyze_roi_density
                return analyze_roi_density(roi, time_period, **kwargs)
            
            elif analysis_type == "temporal":
                from . import analyze_roi_temporal_patterns
                return analyze_roi_temporal_patterns(roi, time_period, **kwargs)
            
            elif analysis_type == "both":
                from . import analyze_roi_density, analyze_roi_temporal_patterns
                
                spatial_result = analyze_roi_density(roi, time_period, **kwargs)
                temporal_result = analyze_roi_temporal_patterns(roi, time_period, **kwargs)
                
                return {
                    "spatial_analysis": spatial_result,
                    "temporal_analysis": temporal_result,
                    "roi_area_km2": calculate_area_km2(roi)
                }
            
            else:
                raise ValidationError(f"Invalid analysis type: {analysis_type}")
                
        except ImportError as e:
            raise PlanetScopeError(f"Analysis functions not available: {e}")
        
    def export_roi_to_formats(
        self, 
        polygons: Optional[List[Polygon]] = None,
        output_dir: str = "./roi_exports"
    ) -> Dict[str, str]:
        """
        Export selected ROI to multiple formats including Shapely objects.
        
        Args:
            polygons: List of polygons to export (uses selected_polygons if None)
            output_dir: Output directory for exports
            
        Returns:
            Dict[str, str]: Dictionary of format -> file_path or object
        """
        polygons = polygons or self.selected_polygons
        
        if not polygons:
            raise ValidationError("No polygons to export")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        exported_files = {}
        
        try:
            # Export as GeoJSON
            geojson_path = output_path / "roi_selection.geojson"
            geojson_data = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": polygon.__geo_interface__,
                        "properties": {
                            "id": i,
                            "area_km2": calculate_area_km2(polygon)
                        }
                    }
                    for i, polygon in enumerate(polygons)
                ]
            }
            
            with open(geojson_path, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            exported_files["geojson"] = str(geojson_path)
            
            # Export as Shapefile (if GeoPandas available)
            if _GEOPANDAS_AVAILABLE:
                shapefile_path = output_path / "roi_selection.shp"
                gdf = gpd.GeoDataFrame({
                    'id': range(len(polygons)),
                    'area_km2': [calculate_area_km2(p) for p in polygons],
                    'geometry': polygons
                }, crs='EPSG:4326')
                
                gdf.to_file(shapefile_path)
                exported_files["shapefile"] = str(shapefile_path)
            
            # NEW: Export as Shapely objects directly
            exported_files["shapely_polygons"] = polygons.copy()
            
            # NEW: Export as individual Shapely objects with metadata
            shapely_objects = []
            for i, polygon in enumerate(polygons):
                shapely_obj = {
                    'id': i,
                    'geometry': polygon,
                    'area_km2': calculate_area_km2(polygon),
                    'bounds': polygon.bounds,
                    'centroid': polygon.centroid,
                    'area_degrees': polygon.area
                }
                shapely_objects.append(shapely_obj)
            
            exported_files["shapely_objects"] = shapely_objects
            
            # NEW: Export as Well-Known Text (WKT)
            wkt_path = output_path / "roi_selection.wkt"
            with open(wkt_path, 'w') as f:
                for i, polygon in enumerate(polygons):
                    f.write(f"# Polygon {i} (Area: {calculate_area_km2(polygon):.2f} km²)\n")
                    f.write(f"{polygon.wkt}\n\n")
            exported_files["wkt"] = str(wkt_path)
            
            # NEW: Export as Python code for direct use
            python_code_path = output_path / "roi_polygons.py"
            with open(python_code_path, 'w') as f:
                f.write("# ROI Polygons as Shapely objects\n")
                f.write("from shapely.geometry import Polygon\n\n")
                
                for i, polygon in enumerate(polygons):
                    coords = list(polygon.exterior.coords)
                    f.write(f"# Polygon {i} - Area: {calculate_area_km2(polygon):.2f} km²\n")
                    f.write(f"roi_polygon_{i} = Polygon({coords})\n\n")
                
                if len(polygons) == 1:
                    f.write("# Main ROI polygon\n")
                    f.write("roi_polygon = roi_polygon_0\n")
                else:
                    f.write("# All ROI polygons\n")
                    f.write(f"all_roi_polygons = [{', '.join([f'roi_polygon_{i}' for i in range(len(polygons))])}]\n")
            
            exported_files["python_code"] = str(python_code_path)
            
            logger.info(f"Exported ROI to {len(exported_files)} formats including Shapely objects")
            return exported_files
            
        except Exception as e:
            raise PlanetScopeError(f"Failed to export ROI: {e}")


    def get_shapely_polygons(self) -> List[Polygon]:
        """
        Get selected ROI as list of Shapely polygon objects.
        
        Returns:
            List[Polygon]: List of Shapely polygons
        """
        if not self.selected_polygons:
            raise ValidationError("No ROI selected. Please select polygons first.")
        
        return self.selected_polygons.copy()


    def get_single_shapely_polygon(self, union_if_multiple: bool = True) -> Polygon:
        """
        Get selected ROI as single Shapely polygon.
        
        Args:
            union_if_multiple: If True and multiple polygons exist, return their union
            
        Returns:
            Polygon: Single Shapely polygon
        """
        if not self.selected_polygons:
            raise ValidationError("No ROI selected. Please select polygons first.")
        
        if len(self.selected_polygons) == 1:
            return self.selected_polygons[0]
        elif union_if_multiple:
            from shapely.ops import unary_union
            return unary_union(self.selected_polygons)
        else:
            raise ValidationError(
                f"Multiple polygons selected ({len(self.selected_polygons)}). "
                "Set union_if_multiple=True to combine them, or use get_shapely_polygons() for all."
            )


    def load_shapely_polygons(self, shapely_objects: Union[Polygon, List[Polygon]]) -> List[Polygon]:
        """
        Load Shapely polygon objects directly into the manager.
        
        Args:
            shapely_objects: Single Polygon or list of Polygons
            
        Returns:
            List[Polygon]: Loaded polygons
        """
        if isinstance(shapely_objects, Polygon):
            polygons = [shapely_objects]
        elif isinstance(shapely_objects, list):
            # Validate all are Polygon objects
            for obj in shapely_objects:
                if not isinstance(obj, Polygon):
                    raise ValidationError(f"Expected Polygon object, got {type(obj)}")
            polygons = shapely_objects
        else:
            raise ValidationError(f"Expected Polygon or list of Polygons, got {type(shapely_objects)}")
        
        # Validate geometries
        valid_polygons = []
        for polygon in polygons:
            if validate_geometry(polygon):
                valid_polygons.append(polygon)
                area_km2 = calculate_area_km2(polygon)
                logger.info(f"Loaded Shapely polygon: {area_km2:.2f} km²")
            else:
                logger.warning("Invalid Shapely polygon geometry, skipping")
        
        if not valid_polygons:
            raise ValidationError("No valid polygons provided")
        
        self.selected_polygons = valid_polygons
        logger.info(f"Loaded {len(valid_polygons)} Shapely polygons")
        
        return valid_polygons


# Jupyter-specific convenience functions
def jupyter_roi_selector(location: str = "milan", instructions: bool = True) -> folium.Map:
    """
    Create Jupyter-optimized ROI selector with enhanced workflow.
    
    Args:
        location: Location name for map center
        instructions: Whether to show instruction popup
        
    Returns:
        folium.Map: Interactive map optimized for Jupyter notebooks
    """
    manager = InteractiveManager()
    
    if instructions:
        return manager.display_map_with_instructions()
    else:
        return manager.create_selection_map()


def jupyter_quick_analysis(
    geojson_file: str = "roi_selection.geojson",
    time_period: str = "last_month",
    analysis_type: str = "spatial"
) -> Dict[str, Any]:
    """
    Quick analysis optimized for Jupyter workflow.
    
    Args:
        geojson_file: Path to exported GeoJSON from interactive map
        time_period: Analysis time period
        analysis_type: "spatial", "temporal", or "both"
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    try:
        # Load ROI from exported file
        manager = InteractiveManager()
        polygons = manager.load_roi_from_file(geojson_file)
        
        if not polygons:
            raise ValidationError("No valid polygons found in file")
        
        # Use first polygon (or union if multiple)
        roi = polygons[0] if len(polygons) == 1 else unary_union(polygons)
        
        # Run analysis
        if analysis_type == "spatial":
            from . import analyze_roi_density
            return analyze_roi_density(roi, time_period)
        
        elif analysis_type == "temporal":
            from . import analyze_roi_temporal_patterns
            return analyze_roi_temporal_patterns(roi, time_period)
        
        elif analysis_type == "both":
            from . import analyze_roi_density, analyze_roi_temporal_patterns
            
            print(" Running spatial analysis...")
            spatial_result = analyze_roi_density(roi, time_period)
            
            print(" Running temporal analysis...")
            temporal_result = analyze_roi_temporal_patterns(roi, time_period)
            
            return {
                "spatial_analysis": spatial_result,
                "temporal_analysis": temporal_result,
                "roi_info": {
                    "area_km2": calculate_area_km2(roi),
                    "polygon_count": len(polygons)
                }
            }
        
    except Exception as e:
        print(f" Analysis failed: {e}")
        print(" Make sure you've exported your ROI as 'roi_selection.geojson'")
        raise


def display_jupyter_workflow_example():
    """Display complete Jupyter workflow example."""
    from IPython.display import display, HTML
    
    workflow_html = """
    <div style="border: 2px solid #28a745; border-radius: 10px; padding: 20px; background-color: #f8f9fa; font-family: Arial, sans-serif;">
        <h2 style="color: #28a745; margin-top: 0;"> Complete Jupyter Workflow Example</h2>
        
        <h3> Step-by-Step Process:</h3>
        
        <div style="background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4> Step 1: Create Interactive Map</h4>
            <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto;"><code>from planetscope_py import jupyter_roi_selector

# Create interactive map for ROI selection
map_obj = jupyter_roi_selector("milan")
map_obj  # This displays the interactive map</code></pre>
        </div>
        
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4> Step 2: Draw Your ROI</h4>
            <ul>
                <li>Use the polygon or rectangle tools on the map above</li>
                <li>Draw your region of interest</li>
                <li>Click the export button to download 'roi_selection.geojson'</li>
            </ul>
        </div>
        
        <div style="background-color: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4> Step 3: Run Analysis</h4>
            <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto;"><code>from planetscope_py import jupyter_quick_analysis

# Quick spatial analysis
result = jupyter_quick_analysis(
    "roi_selection.geojson", 
    "2025-01-01/2025-01-31", 
    "spatial"
)

print(f"Found {result['scenes_found']} scenes")
print(f"Mean density: {result['density_result'].stats['mean']:.1f}")

# Display visualization
result['visualizations']</code></pre>
        </div>
        
        <div style="background-color: #d1ecf1; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4> Step 4: Advanced Analysis</h4>
            <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto;"><code># Run both spatial and temporal analysis
comprehensive_result = jupyter_quick_analysis(
    "roi_selection.geojson",
    "2025-01-01/2025-03-31",
    "both"
)

# Access results
spatial = comprehensive_result['spatial_analysis']
temporal = comprehensive_result['temporal_analysis']

print(f"Spatial: {spatial['scenes_found']} scenes")
print(f"Temporal: {temporal['scenes_found']} scenes")
print(f"Mean coverage days: {temporal['temporal_result'].temporal_stats['mean_coverage_days']:.1f}")

# View all visualizations
spatial['visualizations']
temporal['visualizations']</code></pre>
        </div>
        
        <div style="background-color: #f8d7da; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4> Step 5: Export Results</h4>
            <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto;"><code># All outputs are automatically saved to directories
print(f"Spatial outputs: {spatial['output_directory']}")
print(f"Temporal outputs: {temporal['output_directory']}")

# GeoTIFF files, QML styles, and metadata are automatically created
print(f"Exported files: {list(spatial['exports'].keys())}")
print(f"Temporal files: {list(temporal['exports'].keys())}")</code></pre>
        </div>
        
        <div style="background-color: #e2e3e5; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #6c757d;">
            <h4> Pro Tips for Jupyter:</h4>
            <ul>
                <li><strong>Widget Display:</strong> Maps and plots display directly in cell outputs</li>
                <li><strong>File Management:</strong> GeoJSON files are saved to your notebook directory</li>
                <li><strong>Error Handling:</strong> Clear error messages help with troubleshooting</li>
                <li><strong>Memory Management:</strong> Large analyses automatically use chunking</li>
                <li><strong>Progress Tracking:</strong> Real-time progress bars for long operations</li>
            </ul>
        </div>
    </div>
    """
    
    display(HTML(workflow_html))


# Jupyter-optimized Shapely functions
def jupyter_get_shapely_roi(geojson_file: str = "roi_selection.geojson") -> Polygon:
    """
    Quick function to get Shapely polygon from exported GeoJSON in Jupyter.
    
    Args:
        geojson_file: Path to exported GeoJSON file
        
    Returns:
        Polygon: First polygon from the file as Shapely object
    """
    try:
        manager = InteractiveManager()
        polygons = manager.load_roi_from_file(geojson_file)
        
        if not polygons:
            raise ValidationError(f"No polygons found in {geojson_file}")
        
        polygon = polygons[0]
        area_km2 = calculate_area_km2(polygon)
        
        print(f"Loaded ROI polygon:")
        print(f"  Area: {area_km2:.2f} km²")
        print(f"  Bounds: {polygon.bounds}")
        print(f"  Type: {type(polygon)}")
        
        return polygon
        
    except Exception as e:
        print(f"Failed to load Shapely ROI: {e}")
        raise


def export_shapely_objects(polygons: List[Polygon], output_dir: str = "./shapely_export") -> Dict[str, Any]:
    """
    Export Shapely polygons to various formats.
    
    Args:
        polygons: List of Shapely polygons
        output_dir: Output directory
        
    Returns:
        Dict with exported formats and objects
    """
    manager = InteractiveManager()
    manager.selected_polygons = polygons
    
    return manager.export_roi_to_formats(polygons, output_dir)


def create_shapely_polygon_from_coords(coordinates: List[List[float]]) -> Polygon:
    """
    Create Shapely polygon from coordinate list.
    
    Args:
        coordinates: List of [lon, lat] coordinate pairs
        
    Returns:
        Polygon: Shapely polygon object
        
    Example:
        coords = [[9.1, 45.4], [9.2, 45.4], [9.2, 45.5], [9.1, 45.5], [9.1, 45.4]]
        polygon = create_shapely_polygon_from_coords(coords)
    """
    try:
        polygon = Polygon(coordinates)
        
        if not validate_geometry(polygon):
            raise ValidationError("Created polygon is invalid")
        
        area_km2 = calculate_area_km2(polygon)
        logger.info(f"Created Shapely polygon: {area_km2:.2f} km²")
        
        return polygon
        
    except Exception as e:
        raise ValidationError(f"Failed to create polygon from coordinates: {e}")


# Clean version of quick_preview_with_shapely function
# Replace the existing function in interactive_manager.py with this version

def quick_preview_with_shapely(
    query_instance,
    roi_polygon: Polygon,
    time_period: str,
    max_scenes: int = 10,
    **kwargs
) -> Optional[Any]:
    """
    Quick preview creation using Shapely polygon directly.
    
    Args:
        query_instance: PlanetScopeQuery instance
        roi_polygon: Shapely polygon object
        time_period: Time period string (e.g., "2025-01-01/2025-01-31")
        max_scenes: Maximum scenes to display
        **kwargs: Additional search parameters:
            - cloud_cover_max (float): Maximum cloud cover threshold (default: 0.2)
            - sun_elevation_min (float): Minimum sun elevation in degrees
            - ground_control (bool): Require ground control points
            - quality_category (str): Required quality category
            - item_types (list): Planet item types to search
        
    Returns:
        Folium map with preview
    
    Example:
        >>> preview_map = quick_preview_with_shapely(
        ...     query, roi_polygon, "2025-01-01/2025-01-31",
        ...     max_scenes=20, cloud_cover_max=0.15, sun_elevation_min=30
        ... )
    """
    try:
        # Import required modules within the function
        from .preview_manager import PreviewManager
        from .exceptions import ValidationError
        from .utils import calculate_area_km2
        
        # Parse time period
        if "/" in time_period:
            start_date, end_date = time_period.split("/")
        else:
            raise ValidationError("Time period must be in format 'YYYY-MM-DD/YYYY-MM-DD'")
        
        # Extract search parameters from kwargs
        cloud_cover_max = kwargs.get('cloud_cover_max', 0.2)
        
        # Search for scenes using the provided query instance with additional parameters
        print(f"Searching for scenes in time period: {start_date} to {end_date}")
        print(f"Cloud cover filter: <={cloud_cover_max*100:.0f}%")
        
        # Pass all search parameters to search_scenes
        results = query_instance.search_scenes(
            roi_polygon, 
            start_date, 
            end_date,
            **kwargs
        )
        
        if not results or not results.get('features'):
            print("No scenes found for the given criteria")
            return None
        
        # Create preview manager and interactive map
        print(f"Creating preview with {len(results['features'])} scenes...")
        preview = PreviewManager(query_instance)
        preview_map = preview.create_interactive_map(
            search_results=results,
            roi_geometry=roi_polygon,
            max_scenes=max_scenes
        )
        
        # Calculate and display statistics
        area_km2 = calculate_area_km2(roi_polygon)
        print(f"Created preview with {len(results['features'])} scenes")
        print(f"ROI area: {area_km2:.2f} km²")
        print(f"Time period: {start_date} to {end_date}")
        print(f"Cloud cover: <={cloud_cover_max*100:.0f}%")
        print(f"Displaying up to {max_scenes} scenes on map")
        
        # Display additional filter info if provided
        if kwargs.get('sun_elevation_min'):
            print(f"Sun elevation: >={kwargs['sun_elevation_min']} degrees")
        if kwargs.get('ground_control'):
            print(f"Ground control: Required")
        if kwargs.get('quality_category'):
            print(f"Quality: {kwargs['quality_category']}")
        
        return preview_map
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure preview_manager.py exists and contains PreviewManager class")
        return None
    except Exception as e:
        print(f"Quick preview failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Add to the convenience functions
def create_roi_selector(center_location: Optional[Tuple[float, float]] = None) -> InteractiveManager:
    """
    Create interactive ROI selector (convenience function).
    
    Args:
        center_location: (lat, lon) tuple for map center
        
    Returns:
        InteractiveManager: Configured interactive manager
    """
    return InteractiveManager(center_location)


def quick_roi_map(location: str = "milan") -> folium.Map:
    """
    Create quick ROI selection map for common locations.
    
    Args:
        location: Location name ("milan", "rome", "london", "sf")
        
    Returns:
        folium.Map: Interactive map centered on location
    """
    location_centers = {
        "milan": (45.4642, 9.1900),
        "rome": (41.9028, 12.4964),
        "london": (51.5074, -0.1278),
        "sf": (37.7749, -122.4194),
        "san_francisco": (37.7749, -122.4194),
        "paris": (48.8566, 2.3522),
        "berlin": (52.5200, 13.4050),
        "tokyo": (35.6762, 139.6503),
        "sydney": (-33.8688, 151.2093)
    }
    
    center = location_centers.get(location.lower(), (45.4642, 9.1900))
    manager = InteractiveManager(center)
    return manager.display_map_with_instructions()


# Example usage functions
def demo_interactive_selection():
    """Demonstrate interactive ROI selection capabilities."""
    print("🗺️  Interactive ROI Selection Demo")
    print("=" * 40)
    
    print("\n📍 FEATURES:")
    print("✓ Interactive web maps with drawing tools")
    print("✓ Polygon and rectangle drawing")
    print("✓ Area measurement and validation")
    print("✓ Export to multiple formats (GeoJSON, Shapefile)")
    print("✓ Preset locations for quick selection")
    print("✓ Direct integration with analysis functions")
    
    print("\n🚀 USAGE EXAMPLES:")
    
    print("\n1. Basic ROI Selection:")
    print("   from planetscope_py import create_roi_selector")
    print("   ")
    print("   # Create interactive selector")
    print("   selector = create_roi_selector()")
    print("   map_obj = selector.display_map_with_instructions()")
    print("   # map_obj  # Display in Jupyter notebook")
    
    print("\n2. Quick Location Maps:")
    print("   from planetscope_py import quick_roi_map")
    print("   ")
    print("   # Quick maps for common locations")
    print("   milan_map = quick_roi_map('milan')")
    print("   london_map = quick_roi_map('london')")
    print("   sf_map = quick_roi_map('sf')")
    
    print("\n3. Load ROI from File:")
    print("   # Load existing ROI")
    print("   polygons = selector.load_roi_from_file('my_roi.geojson')")
    print("   ")
    print("   # Export to multiple formats")
    print("   files = selector.export_roi_to_formats(polygons)")
    print("   print(f'Exported: {list(files.keys())}')")
    
    print("\n4. Direct Analysis Integration:")
    print("   # After selecting ROI interactively")
    print("   result = selector.quick_roi_analysis('last_month', 'spatial')")
    print("   print(f'Found {result[\"scenes_found\"]} scenes')")
    
    print("\n📋 WORKFLOW:")
    print("1. Create interactive map")
    print("2. Draw ROI polygons using tools")
    print("3. Export ROI for use in analysis")
    print("4. Run spatial/temporal analysis")
    print("5. View results and visualizations")
    
    dependencies_status = "✅ READY" if _FOLIUM_AVAILABLE and _SHAPELY_AVAILABLE else "❌ MISSING DEPS"
    print(f"\n🔧 STATUS: {dependencies_status}")
    
    if not _FOLIUM_AVAILABLE or not _SHAPELY_AVAILABLE:
        missing = []
        if not _FOLIUM_AVAILABLE:
            missing.append("folium")
        if not _SHAPELY_AVAILABLE:
            missing.append("shapely")
        print(f"📦 Install missing dependencies: pip install {' '.join(missing)}")


if __name__ == "__main__":
    demo_interactive_selection()