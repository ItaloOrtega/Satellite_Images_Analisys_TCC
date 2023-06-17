import time
from datetime import datetime

from geom_functions import create_a_square
from models import SourceType, IndexExpressionType, ColorMaps
from scenes_functions import get_scenes_ids_from_microsoft_planetary, get_scenes_urls
from service import open_multiple_images, create_images_files, create_images_datasets, create_affected_area_information

_FIRST_POSITION = 0


def get_images_with_index_from_middle_point(
        source: str,
        index: str,
        color_map: str,
        point_latitude: float,
        point_longitude: float,
        start_date_str: str,
        end_date_str: str,
        days_gap: int = 1,
        max_distance_meters: int = 5000,
        image_size: int = 256,
        max_cloud_coverage: float = 40.0
):
    """
    Calls the functions to create geometry area from center point, get Scenes IDs and data that intersects with the
    geometry, create Scenes URLs, open Images in teh geometry area, apply Index in the Images,
    and write Images locally.

    :param source: Source of the Scenes
    :param index: Index to be applied on the Images
    :param color_map: Color Map to be used in the visualization of the Image
    :param point_latitude: Latitude of the center point of the geometry area
    :param point_longitude: Longitude of the center point of the geometry area
    :param start_date_str: Start date string
    :param end_date_str: End date string
    :param days_gap: Gap of the days between Scenes
    :param max_distance_meters: Max size of the side of geometry area
    :param image_size: Image size in pixels
    :param max_cloud_coverage: Limit of max cloud coverage of the Scenes
    :return: Writes locally the Images with the Index applied and the color map
    """

    source_bands = SourceType[source]

    index_type = IndexExpressionType[index].value

    color_map_object = ColorMaps[color_map].value

    if color_map_object.type not in index_type.color_maps:
        print('Color Map is not available to this index')
        raise ValueError

    if index_type == IndexExpressionType.rgb.value:
        print('RGB is not valid index.')
        raise ValueError

    area_from_geom = create_a_square(max_distance_meters, point_latitude, point_longitude)
    date_format = "%Y-%m-%d"
    try:
        start_date = datetime.strptime(start_date_str, date_format).date()
        end_date = datetime.strptime(end_date_str, date_format).date()
    except ValueError as e:
        print('Dates does not match the format.')
        raise e

    list_scenes = get_scenes_ids_from_microsoft_planetary(
        start_date, end_date, area_from_geom, source_bands.value.source_name, max_cloud_coverage, days_gap
    )
    time_start = time.perf_counter()
    list_scenes_informations = get_scenes_urls(list_scenes, source_bands)
    print(f'Time to create VRTs for images = {time.perf_counter() - time_start} s')

    print(f'Creating {len(list_scenes_informations)} images ...')
    time_start = time.perf_counter()
    opened_images = open_multiple_images(
        list_scenes_informations, area_from_geom, image_size, source_bands.value
    )
    print(
        f'Time to open all {len(opened_images)} images from {sum([len(scenes_same_day) for scenes_same_day in list_scenes_informations])}= {time.perf_counter() - time_start} s')

    index_images, rgb_images = create_images_datasets(opened_images, index_type, area_from_geom, max_cloud_coverage)

    fig_means_stdev_max_min, fig_affected_area_graph, afected_area_image, fig_deforestation_area_graph, \
        list_deforestation_diff_area_obj = create_affected_area_information(index_images, rgb_images[_FIRST_POSITION],
                                                                            area_from_geom, image_size)

    # TODO: add all images to a PDF file and save locally and zip the files

    create_images_files(index_images, ColorMaps.raw.value, 'tiff')

    create_images_files(rgb_images, ColorMaps.truecolor.value, 'tiff')

    if index == 'raw' or color_map == 'raw':
        print('Not able to create PNG images.')

    else:
        create_images_files(index_images, color_map_object, 'png')

        create_images_files(rgb_images, ColorMaps.truecolor.value, 'png')
