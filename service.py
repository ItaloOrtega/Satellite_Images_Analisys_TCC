import os
import time
from typing import List

import numpy
from shapely import Polygon

from geom_functions import epsg_transform
from image_functions import open_single_image, merge_datasets, get_image_id, create_image_with_rasterio
from models import SceneInformation, SourceBands, Image, Index, ColorMap
from indices_functions import get_image_with_index

_FIRST_POSITION = 0
_LINES_INDEX = 1
_COLUMNS_INDEX = 2
_ORIGINAL_POLYGON_EPSG = 4326


def open_multiple_images(
        list_scenes_informations: List[List[SceneInformation]],
        area_geom: Polygon,
        image_size: int,
        source: SourceBands,
        index_name: str
) -> List[Image]:
    """
    Open multiple scenes from the same day as Datasets and merge them to become single image.

    :param list_scenes_informations: List of Scene Information from the same day
    :param area_geom: Polygon object of the area geometry
    :param image_size: Size of the image in pixels
    :param source: Source of the Images
    :param index_name: Index name
    :return: List of Images from the Scenes of the same day
    """
    opened_images = []
    for scenes_from_same_day in list_scenes_informations:
        opened_datasets = []
        time_start_opening = time.perf_counter()
        geometry = epsg_transform(
            geometry=area_geom, from_epsg=_ORIGINAL_POLYGON_EPSG, to_epsg=source.epsg
        )
        for scene_information in scenes_from_same_day:
            dataset = open_single_image(scene_information.url, image_size, geometry)
            opened_datasets.append(dataset)
        if len(opened_datasets) > 1:  # Case for more than one Scene from the same day
            image_bands, metadata = merge_datasets(opened_datasets, geometry)
            for dataset in opened_datasets:
                dataset.close()
        else:  # Case when only one image from that day
            image_bands = opened_datasets[_FIRST_POSITION].read()
            metadata = opened_datasets[_FIRST_POSITION].profile
            opened_datasets[_FIRST_POSITION].close()
        acquisition_date = scenes_from_same_day[_FIRST_POSITION].acquisition_date
        opened_images.append(
            Image(
                data=image_bands,
                mask=numpy.ones((image_bands.shape[_LINES_INDEX], image_bands.shape[_COLUMNS_INDEX]))*255,
                cloud_mask=numpy.ones((image_bands.shape[_LINES_INDEX], image_bands.shape[_COLUMNS_INDEX]))*255,
                metadata=metadata,
                id=get_image_id(source.source_name, index_name, acquisition_date),
                source=source
            )
        )
        print(f'Time to OPEN and merge {len(scenes_from_same_day)} images from the same day = {time.perf_counter() - time_start_opening} s')
    return opened_images


def calculate_images_indices(list_images: List[Image], index: Index, geometry: Polygon, max_cloud_coverage: float):
    """
    Calculate images with index applied on it.

    :param list_images: List of Images
    :param index: Index to be applied on the images
    :param geometry: Polygon object of the area geometry
    :return: List of Images with the index applied
    """
    area_geometry = epsg_transform(
            geometry=geometry, from_epsg=_ORIGINAL_POLYGON_EPSG, to_epsg=list_images[_FIRST_POSITION].source.epsg
        )
    list_images_with_index = []
    for image in list_images:
        image_with_index = get_image_with_index(image, index, area_geometry, max_cloud_coverage)
        if image_with_index is not None:
            list_images_with_index.append(image_with_index)

    return list_images_with_index


def create_images_files(list_images: List[Image], color_map: ColorMap, file_extension: str):
    """
    Creates the images with a color map representation and writes it locally.

    :param list_images: List of Images
    :param color_map: Color Map objected to be applied in the bands
    :param file_extension: File extension to save the image
    :return: None
    """
    if os.path.exists('./images') is False:
        os.mkdir('./images')
    for image in list_images:
        image_file = create_image_with_rasterio(image, color_map, file_extension)
        with open(f'images/{image.id}.{file_extension}', 'wb') as file:
            file.write(image_file)
