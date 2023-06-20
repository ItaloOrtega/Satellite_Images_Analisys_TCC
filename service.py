import os
import time
from typing import List, Optional

import numpy
from shapely import Polygon

from analyser import create_means_stdev_max_min_plot_figure, create_affected_area_plot_figure, \
    create_affected_area_image, calculate_list_difference_between_days, create_deforestation_area_throught_dates_figure, \
    get_most_changed_area_figures, calculate_afected_area, DifferenceMask
from files_manager import save_file_locally, create_gif_image
from geom_functions import epsg_transform
from image_functions import open_single_image, merge_datasets, get_scene_id, \
    create_raster_image
from models import SceneInformation, SourceBands, Image, Index, ColorMap, ColorMaps
from indices_functions import get_image_with_index, get_rgb_image, calculate_cloud_mask

_FIRST_POSITION = 0
_LAST_POSITION = -1
_LINES_INDEX = 1
_COLUMNS_INDEX = 2
_ORIGINAL_POLYGON_EPSG = 4326


def open_multiple_images(
        list_scenes_informations: List[List[SceneInformation]],
        area_geom: Polygon,
        image_size: int,
        source: SourceBands
) -> List[Image]:
    """
    Open multiple scenes from the same day as Datasets and merge them to become single image.

    :param list_scenes_informations: List of Scene Information from the same day
    :param area_geom: Polygon object of the area geometry
    :param image_size: Size of the image in pixels
    :param source: Source of the Images
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
                    mask=numpy.ones((image_bands.shape[_LINES_INDEX], image_bands.shape[_COLUMNS_INDEX])) * 255,
                    cloud_mask=numpy.ones((image_bands.shape[_LINES_INDEX], image_bands.shape[_COLUMNS_INDEX])) * 255,
                    metadata=metadata,
                    id=get_scene_id(source.source_name, acquisition_date),
                    acquisition_date=acquisition_date,
                    source=source
                )
            )
        print(
            f'Time to OPEN and merge {len(scenes_from_same_day)} images from the same day = {time.perf_counter() - time_start_opening} s')
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
            rgb_image.cloud_mask, percentage_of_clouds = calculate_cloud_mask(rgb_image.cloud_mask, rgb_image.data,
                                                                              rgb_image.mask)
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


def create_index_images_files(list_images: List[Image], color_map: ColorMap, file_extension: str):
    """
    Creates all the RasterImages from the Images list with a color map representation and writes it locally.

    :param list_images: List of Images
    :param color_map: Color Map objected to be applied in the bands
    :param file_extension: File extension to save the image
    :return: List of RasterImages
    """
    raster_images_list = []
    for image in list_images:
        raster_image = create_raster_image(image, color_map)
        raster_image_data = raster_image.data
        metadata = None
        if file_extension == 'tiff':
            raster_image_data = numpy.append(raster_image_data, [raster_image.cloud_mask], axis=0)
            metadata = raster_image.metadata
            metadata.update(count=raster_image_data.shape[0], dtype=raster_image_data.dtype)
        raster_images_list.append(raster_image)
        if file_extension == 'png':
            parent_folder = 'visualizer'
        else:
            parent_folder = 'dataset'
        if os.path.exists(f'./images/{parent_folder}') is False:
            os.mkdir(f'./images/{parent_folder}')
        folder_path = f'./images/{parent_folder}/{file_extension}'
        save_file_locally(raster_image_data, folder_path, image.id, file_extension, metadata)
    return raster_images_list


def create_affected_area_information(
        ndvi_dataset: List[Image], initial_rgb_image: Image, original_geometry: Polygon, image_size: int
):
    """
    Creates all graphs and figures necessary to analyse the user's area, by creating lost and gain green areas in
     hectares graph, percentage of loss area due to deforestation to the months, and other data.
    :param ndvi_dataset: List of the NDVI Images
    :param initial_rgb_image: First RGB Image of the area being analysed
    :param original_geometry: Original geometry of the area
    :param image_size: Size being used to create the images
    :return: Graphs and figures
    """
    masked_ndvi_images = []
    for ndvi_img in ndvi_dataset:
        masked_ndvi_images.append(ndvi_img.create_masked_array_dataset())

    fig_means_stdev_max_min = create_means_stdev_max_min_plot_figure(masked_ndvi_images)

    interpolated_deforestation_area, interpolated_recovered_area = calculate_afected_area(masked_ndvi_images)

    initial_date = masked_ndvi_images[_FIRST_POSITION].acquisition_date
    final_date = masked_ndvi_images[_LAST_POSITION].acquisition_date

    fig_affected_area_graph = create_affected_area_plot_figure(
        interpolated_deforestation_area, interpolated_recovered_area,
        original_geometry, image_size, initial_date, final_date
    )

    afected_area_image = create_affected_area_image(initial_rgb_image, interpolated_deforestation_area,
                                                    interpolated_recovered_area)

    difference_mask_list = calculate_list_difference_between_days(masked_ndvi_images,
                                                                  initial_rgb_image.metadata.get('transform'),
                                                                  original_geometry, image_size)

    fig_deforestation_area_graph = create_deforestation_area_throught_dates_figure(difference_mask_list)

    list_deforestation_diff_area_obj = get_most_changed_area_figures(difference_mask_list, initial_rgb_image)

    return fig_means_stdev_max_min, fig_affected_area_graph, afected_area_image, fig_deforestation_area_graph, \
        list_deforestation_diff_area_obj


def save_affected_area_analisys_files(
        fig_means_stdev_max_min: numpy.array, fig_affected_area_graph: numpy.array,
        afected_area_image: numpy.array, fig_deforestation_area_graph: Optional[numpy.array],
        list_deforestation_diff_area_obj: List[DifferenceMask]
):
    """
    Saves all the graphs and images of the analisys of the area
    :param fig_means_stdev_max_min: Graph representing the means, stdev, max, and min values throught the months
    :param fig_affected_area_graph: Graph representing the gain and loss of green area throught the months
    :param afected_area_image: Image representing the gain and loss of green area throught the months
    :param fig_deforestation_area_graph: Graph representing the affected areas by deforestation throught the months
    :param list_deforestation_diff_area_obj: List of deforestation differences bettween the months
    """
    if os.path.exists('./images') is False:
        os.mkdir('./images')

    save_file_locally(fig_means_stdev_max_min, './images/graphs', 'means_stdev_max_min_graph', 'png')

    save_file_locally(fig_affected_area_graph, './images/graphs', 'affected_area_graph', 'png')

    save_file_locally(afected_area_image, './images/visualizer', 'afected_area_image', 'png')

    if fig_deforestation_area_graph is not None:
        save_file_locally(fig_deforestation_area_graph, './images/graphs', 'deforestation_area_graph', 'png')

    if os.path.exists('./images/visualizer/deforestation_areas') is False:
        os.mkdir('./images/visualizer/deforestation_areas')
    for index_diff, diff_mask in enumerate(list_deforestation_diff_area_obj):
        file_path = f'./images/visualizer/deforestation_areas/{index_diff + 1}-{diff_mask.difference_date}'
        save_file_locally(
            diff_mask.lost_area_image, file_path, f'lost_area_{diff_mask.difference_date}', 'png')
        save_file_locally(
            diff_mask.lost_area_geojson, file_path, f'lost_area_{diff_mask.difference_date}_geojson', 'json')


def save_index_rgb_images(index_images: List[Image], rgb_images: List[Image], original_colormap: ColorMap):
    """
    Saves locally all the PNG, TIFF and GIF of the RGB and Index images locally
    """
    raster_index_images = create_index_images_files(index_images, original_colormap, 'png')

    raster_rbg_images = create_index_images_files(rgb_images, ColorMaps.truecolor.value, 'png')

    create_gif_image(raster_index_images, 'ndvi')
    create_gif_image(raster_rbg_images, 'rgb')

    _ = create_index_images_files(index_images, ColorMaps.raw.value, 'tiff')

    # _ = create_index_images_files(rgb_images, ColorMaps.raw.value, 'tiff')
