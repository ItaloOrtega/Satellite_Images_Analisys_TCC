from typing import Union

from rasterio import CRS
from rasterio.warp import transform_geom
from shapely import Polygon
from shapely.geometry import shape, mapping


def epsg_transform(
        geometry: Union[dict, Polygon], from_epsg: Union[int, CRS] = None, to_epsg: Union[int, CRS] = None
) -> Union[dict, Polygon]:
    """
    Reproject EPSG to another EPSG

    :param geometry: Geojson or Polygon with the geometry
    :param from_epsg: EPSG code of the input Geojson/Polygon
    :param to_epsg: EPSG code of the output Geojson/Polygon
    :return: Return the input format transformed to another projection.
    """
    if to_epsg is None:
        to_epsg = 3857
    if from_epsg is None:
        from_epsg = 4326

    if isinstance(from_epsg, int):
        from_crs = CRS({'init': f'EPSG:{from_epsg}'})
    else:
        from_crs = from_epsg

    if isinstance(to_epsg, int):
        to_crs = CRS({'init': f'EPSG:{to_epsg}'})
    else:
        to_crs = to_epsg

    if isinstance(geometry, Polygon):
        return shape(transform_geom(from_crs, to_crs, mapping(geometry)))

    return transform_geom(from_crs, to_crs, geometry)


def create_a_square(side_in_meters: int, center_lat: float, center_lon: float):
    """
    Creates a square Polygon from a center point and the max size of it sides

    :param side_in_meters: Max size of each size of the polygon square
    :param center_lat: center point latitude
    :param center_lon: center point longitude
    :return: Polygon of a square geometry
    """
    one_meter = 1 / 111_111
    half_side = side_in_meters / 2

    tile_coordinates = {
        "type": "Polygon",
        "coordinates": [
            [
                [(center_lat - half_side * one_meter), (center_lon - half_side * one_meter)],
                [(center_lat - half_side * one_meter), (center_lon + half_side * one_meter)],
                [(center_lat + half_side * one_meter), (center_lon + half_side * one_meter)],
                [(center_lat + half_side * one_meter), (center_lon - half_side * one_meter)],
                [(center_lat - half_side * one_meter), (center_lon - half_side * one_meter)],
            ]
        ],
    }
    return shape(tile_coordinates)
