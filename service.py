import os
import time
from typing import List

import numpy
from shapely import Polygon

from geom_functions import epsg_transform
from image_functions import open_single_image, merge_datasets, create_image_with_rasterio, get_scene_id
from models import SceneInformation, SourceBands, Image, Index, ColorMap
from indices_functions import get_image_with_index, get_rgb_image, calculate_cloud_mask

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
        image_bands = None
        metadata = None
        opened_datasets = []
        time_start_opening = time.perf_counter()
        geometry = epsg_transform(
            geometry=area_geom, from_epsg=_ORIGINAL_POLYGON_EPSG, to_epsg=source.epsg
        )
        for scene_information in scenes_from_same_day:
            dataset = open_single_image(scene_information.url, image_size, geometry, scene_information.scene_id)
            # dataset will be None if something wrong happened while opening a Scene
            if dataset is not None:
                opened_datasets.append(dataset)
        if len(opened_datasets) > 1:  # Case for more than one Scene from the same day
            image_bands, metadata = merge_datasets(opened_datasets, geometry)
            for dataset in opened_datasets:
                dataset.close()
        elif len(opened_datasets) == 1:  # Case when only one image from that day
            image_bands = opened_datasets[_FIRST_POSITION].read()
            metadata = opened_datasets[_FIRST_POSITION].profile
            opened_datasets[_FIRST_POSITION].close()

        if image_bands is not None and metadata is not None:
            acquisition_date = scenes_from_same_day[_FIRST_POSITION].acquisition_date
            opened_images.append(
                Image(
                    data=image_bands,
                    mask=numpy.ones((image_bands.shape[_LINES_INDEX], image_bands.shape[_COLUMNS_INDEX]))*255,
                    cloud_mask=numpy.ones((image_bands.shape[_LINES_INDEX], image_bands.shape[_COLUMNS_INDEX]))*255,
                    metadata=metadata,
                    id=get_scene_id(source.source_name, acquisition_date),
                    source=source
                )
            )
        print(f'Time to OPEN and merge {len(scenes_from_same_day)} images from the same day = {time.perf_counter() - time_start_opening} s')
    return opened_images


def create_images_datasets(list_images: List[Image], index: Index, geometry: Polygon, max_cloud_coverage: float):
    """
    Calculate the dataset to be analised, creating RGB images and images with the choosen Index.

    :param list_images: List of Images
    :param index: Index to be applied on the images
    :param geometry: Polygon object of the area geometry
    :return: List of Images with the index applied
    """
    area_geometry = epsg_transform(
            geometry=geometry, from_epsg=_ORIGINAL_POLYGON_EPSG, to_epsg=list_images[_FIRST_POSITION].source.epsg
        )

    rgb_images_list = []
    valid_scenes = []
    for scene_image in list_images:
        rgb_image = get_rgb_image(scene_image, area_geometry, max_cloud_coverage)
        if rgb_image is not None:
            rgb_image.cloud_mask, percentage_of_clouds = calculate_cloud_mask(rgb_image.cloud_mask, rgb_image.data, rgb_image.mask)
            if percentage_of_clouds <= max_cloud_coverage:
                rgb_images_list.append(rgb_image)
                scene_image.cloud_mask = rgb_image.cloud_mask
                scene_image.mask = rgb_image.mask
                valid_scenes.append(scene_image)
            else:
                print(f'Scene {scene_image.id} has clouds over the threshold, {percentage_of_clouds}.')

    list_images_with_index = []
    for image in valid_scenes:
        image_with_index = get_image_with_index(image, index)
        if image_with_index is not None:
            list_images_with_index.append(image_with_index)

    return list_images_with_index, rgb_images_list


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
