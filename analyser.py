import datetime
import io
import os
from collections import namedtuple, defaultdict
from copy import copy
from typing import List

import matplotlib
from area import area
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio import transform, MemoryFile
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import measure
import numpy
from affine import Affine
from matplotlib import pyplot as plt
from rasterio.features import rasterize
from shapely import Polygon, intersects, MultiPolygon
from shapely.geometry import mapping

from geom_functions import epsg_transform
from indices_functions import create_grey_scale_img_from_rgb
from models import Image, MaskedImage, ColorMaps
from service import create_analisys_files

ImageInfo = namedtuple('ImageInfo', ['acquisition_date', 'mean', 'stdev', 'max', 'min'])
DifferenceMask = namedtuple('DifferenceMask', [
    'difference_date', 'lost_mask', 'lost_area', 'lost_area_geojson', 'lost_area_image'
])

plt.rcParams['figure.constrained_layout.use'] = True

FIRST_POSITION = 0
SECOND_POSITION = 1


def create_masked_array_dataset(index_image: Image):
    bool_cloud_mask = numpy.invert(index_image.cloud_mask.astype('bool'))
    image_masked_array = numpy.ma.masked_array(data=index_image.data, mask=bool_cloud_mask)
    masked_image = MaskedImage(image_masked_array, index_image.acquisition_date, index_image.id)
    return masked_image


def calculate_means_stdev_max_min(list_masked_images: List[MaskedImage]):
    list_images_info = []
    for masked_image in list_masked_images:
        image_mean = masked_image.data.mean()
        image_stdev = masked_image.data.std()
        image_max = masked_image.data.max()
        image_min = masked_image.data.min()
        image_acquisition_date = masked_image.acquisition_date
        list_images_info.append(ImageInfo(image_acquisition_date, image_mean, image_stdev, image_max, image_min))

    return list_images_info


def create_polygon_from_image_contour(image_contour_array: numpy.array, original_transform: Affine):
    lines_values = [x for x, y in image_contour_array]
    columns_values = [y for x, y in image_contour_array]
    x_coordinates, y_coordinates = transform.xy(original_transform, lines_values, columns_values)
    transformed_contour = [[x_coordinates[coord_index], y_coordinates[coord_index]] for coord_index in
                           range(len(x_coordinates))]
    numpy_contour = numpy.asarray(transformed_contour)
    polygon_contour = Polygon(numpy_contour)
    return polygon_contour


def plot_means_stdev_max_min(list_images_info: List[ImageInfo]):
    list_dates = [img.acquisition_date for img in list_images_info]
    image_info_sub_lists = defaultdict(list)

    for img_info in list_images_info:
        image_info_sub_lists[(img_info.acquisition_date.year, img_info.acquisition_date.month)].append(img_info)

    sorted_image_info_sub_lists = sorted(image_info_sub_lists.items())

    months_list = ['{}-{}'.format(*img_info[0]) for img_info in sorted_image_info_sub_lists]
    list_means = []
    list_stdevs = []
    list_maxs = []
    list_mins = []
    for img_info_sub_list in sorted_image_info_sub_lists:
        sub_list_mean = []
        sub_list_stdevs = []
        sub_list_maxs = []
        sub_list_mins = []
        for img_info in img_info_sub_list[1]:
            sub_list_mean.append(img_info.data.mean())
            sub_list_stdevs = [numpy.std(img_info.data)]
            sub_list_maxs = [img_info.data.max()]
            sub_list_mins = [img_info.data.min()]
        list_means.append(sum(sub_list_mean) / len(sub_list_mean))
        list_stdevs.append(sum(sub_list_stdevs) / len(sub_list_stdevs))
        list_maxs.append(sum(sub_list_maxs) / len(sub_list_maxs))
        list_mins.append(sum(sub_list_mins) / len(sub_list_mins))

    figsize = (len(list_dates) * 0.5, 6)

    plt.figure(figsize=figsize, layout="constrained")
    plt.plot(months_list, list_means, marker='o', label='Means Values')
    plt.plot(months_list, list_stdevs, marker='o', label='Standart Deviation Values')
    plt.plot(months_list, list_maxs, marker='o', label='Max Values')
    plt.plot(months_list, list_mins, marker='o', label='Min Values')

    plt.xlabel('Images months')
    plt.ylabel('Images Informations Values')
    plt.legend()
    plt.title('Images Informations throught months')
    plt.xticks(months_list, rotation=45)
    with io.BytesIO() as memf:
        extent = plt.gcf().get_window_extent()
        extent = extent.transformed(plt.gcf().dpi_scale_trans.inverted())
        plt.gcf().savefig(memf, format='PNG', bbox_inches=extent)
        memf.seek(0)
        with MemoryFile(memf) as memfile:
            with memfile.open() as dataset:
                figure_array = dataset.read()
        memf.close()
        plt.close()

    return figure_array


def plot_afect_area(deforestation_area_array: numpy.array, recovered_area_array: numpy.array,
                    original_geometry: Polygon, image_size: int, initial_date: datetime, final_date: datetime):
    pixel_area = area(mapping(original_geometry)) / (image_size * image_size)
    deforestation_area = ((deforestation_area_array != 0).sum() * pixel_area) / 10000
    recovered_area = ((recovered_area_array != 0).sum() * pixel_area) / 10000

    area_in_hectare = (deforestation_area, recovered_area)
    bar_labels = ('Lost Area to Deforestation', 'Recovered Area')

    plt.bar(bar_labels, area_in_hectare, color=('red', 'green'), label=('Lost Area to Deforestation', 'Recovered Area'),
            align='center')
    for i in range(len(area_in_hectare)):
        plt.annotate('{:.2f} Ha\n'.format(area_in_hectare[i]), xy=(bar_labels[i], area_in_hectare[i]), ha='center',
                     va='center', fontweight='bold')
    plt.xlabel('Affected Area')
    plt.ylabel('Affected Area Rate in Hectare (Ha)')
    plt.legend()
    plt.title(f'Affected Hectares of the Area from {initial_date} -> {final_date}')

    with io.BytesIO() as memf:
        extent = plt.gcf().get_window_extent()
        extent = extent.transformed(plt.gcf().dpi_scale_trans.inverted())
        plt.gcf().savefig(memf, format='PNG', bbox_inches=extent)
        memf.seek(0)
        with MemoryFile(memf) as memfile:
            with memfile.open() as dataset:
                figure_array = dataset.read()
        memf.close()
        plt.close()

    return figure_array


def calculate_afected_area(list_masked_images: List[MaskedImage]):
    area_of_deforestation = numpy.zeros(list_masked_images[FIRST_POSITION].data.shape)
    area_of_recoveration = numpy.zeros(list_masked_images[FIRST_POSITION].data.shape)
    initial_deforestad_area = None
    initial_green_area = None
    for img_index, masked_image in enumerate(list_masked_images):

        deforestation_image_area = masked_image.data.filled(0)
        recovered_image_area = masked_image.data.filled(0)

        deforestation_image_area[deforestation_image_area > 0.5] = 0
        deforestation_image_area[deforestation_image_area <= 0] = 0
        deforestation_image_area[deforestation_image_area != 0] = 1

        recovered_image_area[recovered_image_area < 0.5] = 0
        recovered_image_area[recovered_image_area != 0] = 1

        if img_index != 0:
            area_of_deforestation += deforestation_image_area
            area_of_recoveration += recovered_image_area
        else:
            initial_deforestad_area = deforestation_image_area
            initial_green_area = recovered_image_area

    area_of_deforestation[area_of_deforestation < 2] = 0
    area_of_deforestation[initial_deforestad_area != 0] = 0

    area_of_recoveration[area_of_recoveration < len(list_masked_images) * 0.15] = 0
    area_of_recoveration[initial_green_area != 0] = 0

    interpolated_recovered_area = numpy.interp(area_of_recoveration,
                                               [0,
                                                (area_of_recoveration[area_of_recoveration != 0]).min(),
                                                area_of_recoveration.max()], [0, 0.5, 1])

    interpolated_deforestation_area = numpy.interp(area_of_deforestation,
                                                   [0, (area_of_deforestation[area_of_deforestation != 0]).min(),
                                                    area_of_deforestation.max()], [0, 0.5, 0.05])

    return interpolated_deforestation_area, interpolated_recovered_area


def calculate_list_difference_between_days(masked_images_list: List[MaskedImage], original_transform: Affine,
                                           original_geometry: Polygon, image_size: int):
    pixel_area = area(mapping(original_geometry)) / (image_size * image_size)
    list_difference_mask = []
    lost_area = numpy.zeros((image_size, image_size))
    gain_area = numpy.zeros((image_size, image_size))
    print(f'Total area = {area(mapping(original_geometry))} | Pixel Area = {pixel_area} m²')

    minimal_contour_array = numpy.zeros((image_size, image_size))
    minimal_contour_array[120:124, 120:124] = 255
    minimal_contour = measure.find_contours(minimal_contour_array)
    minimal_contour_geometry = create_polygon_from_image_contour(minimal_contour[FIRST_POSITION], original_transform)
    area_minimal_contour_geometry = area(mapping(epsg_transform(minimal_contour_geometry, 32722, 4326)))
    print(f'Minimal area = {area_minimal_contour_geometry}')

    for img_index, masked_image in enumerate(masked_images_list):
        if img_index != len(masked_images_list) - 1:
            change_date = str(masked_image.acquisition_date) + '->' + str(
                masked_images_list[img_index + 1].acquisition_date)
            diff_mask_img = numpy.subtract(masked_images_list[img_index + 1].data, masked_image.data).data
            int_mask = numpy.add(masked_image.data.mask, masked_images_list[img_index + 1].data.mask).astype('uint8')
            diff_mask_img[int_mask == 1] = 0

            img_gain = numpy.copy(diff_mask_img)
            img_loss = numpy.copy(diff_mask_img)

            img_gain[img_gain < 0.3] = 0
            img_gain[gain_area != 0] = 0
            gain_area[img_gain != 0] = 1

            img_loss[img_loss > -0.3] = 0
            img_loss[lost_area != 0] = 0
            lost_area[img_loss != 0] = 1

            list_difference_mask.append((change_date, img_loss, img_gain))

    change_date_index = 0
    lost_mask_index = 1
    gain_mask_index = 2

    difference_mask_list = []

    for img_index, difference_mask in enumerate(list_difference_mask):
        original_loss_mask = copy(difference_mask[lost_mask_index])
        loss_count = 0
        if img_index != len(list_difference_mask) - 1:
            difference_mask[lost_mask_index][list_difference_mask[img_index + 1][gain_mask_index] != 0] = 0
            list_difference_mask[img_index + 1][gain_mask_index][original_loss_mask != 0] = 0

        structure_dilation = numpy.array([[1, 1, 1],
                                          [1, 1, 1],
                                          [1, 1, 1]])
        bool_lost_mask = copy(difference_mask[lost_mask_index])

        bool_lost_mask[bool_lost_mask != 0] = 255

        dilated_lost_mask = binary_dilation(bool_lost_mask.astype('bool'),
                                            structure=structure_dilation).astype('uint8')

        dilated_lost_mask[dilated_lost_mask == 1] = 255
        loss_geojsons = []
        image_contours_list = measure.find_contours(dilated_lost_mask)
        for img_contour in image_contours_list:
            loss_geojsons.append(create_polygon_from_image_contour(img_contour, original_transform))

        valid_loss_geometries_list = []
        for geometry in loss_geojsons:
            poly_area = area(mapping(epsg_transform(geometry, 32722, 4326)))
            if poly_area >= area_minimal_contour_geometry:
                valid_loss_geometries_list.append(geometry)

        final_valid_geometries_list = copy(valid_loss_geometries_list)
        for valid_loss_geometry in valid_loss_geometries_list:
            for geometry in loss_geojsons:
                if geometry not in valid_loss_geometries_list and intersects(valid_loss_geometry, geometry) is True:
                    final_valid_geometries_list.append(geometry)

        default_numpy_array = numpy.zeros((image_size, image_size))

        if len(final_valid_geometries_list) > 0:
            result_valid = rasterize(
                shapes=final_valid_geometries_list, transform=original_transform, out=default_numpy_array, fill=0,
                dtype=float
            )
            result_valid[result_valid == 1] = 255

            structure_erosion = numpy.array([[1, 1, 1, 1],
                                             [1, 1, 1, 1],
                                             [1, 1, 1, 1],
                                             [1, 1, 1, 1]])

            final_lost_mask = binary_erosion(result_valid.astype('bool'),
                                             structure=structure_erosion).astype('uint8')

            final_lost_mask[difference_mask[lost_mask_index] == 0] = 0

            loss_count = (final_lost_mask != 0).sum()

            lost_area_geojson = epsg_transform(mapping(MultiPolygon(final_valid_geometries_list)), 32722, 4326)

            difference_mask_list.append(DifferenceMask(
                difference_mask[change_date_index],
                difference_mask[lost_mask_index],
                loss_count * pixel_area,
                lost_area_geojson,
                None
            )
            )

    return difference_mask_list


def plot_deforestation_area_throught_dates(lost_mask_list: List[DifferenceMask]):
    image_info_sub_lists = defaultdict(list)

    for diff_mask in lost_mask_list:
        acquisition_date = datetime.datetime.strptime(
            diff_mask.difference_date.split('->')[SECOND_POSITION], '%Y-%m-%d'
        )
        image_info_sub_lists[(acquisition_date.year, acquisition_date.month)].append(diff_mask.lost_area)

    sorted_image_info_sub_lists = sorted(image_info_sub_lists.items())

    months_list = ['{}-{}'.format(*img_info[FIRST_POSITION]) for img_info in sorted_image_info_sub_lists]
    lost_areas_hac = [round(sum(img_info[SECOND_POSITION]) / 10000, 2) for img_info in sorted_image_info_sub_lists]

    figsize = (len(months_list) * 0.9, len(lost_areas_hac) * 0.7)

    plt.figure(figsize=figsize, layout="constrained")
    plt.plot(months_list, lost_areas_hac, marker='o', color='red', label='Lost Area in Hectare (Ha)')

    for x, y in zip(months_list, lost_areas_hac):
        plt.annotate(f'{y} Ha',  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 5),  # distance from text to points (x,y)
                     ha='center',
                     fontweight='bold')  # horizontal alignment can be left, right or center

    plt.xlabel('Images months')
    plt.ylabel('Affected Area Rate in Hectare (Ha)')
    plt.legend()
    plt.title('Lost Area to Deforestation Throught Months')
    plt.xticks(months_list, rotation=45)

    with io.BytesIO() as memf:
        extent = plt.gcf().get_window_extent()
        extent = extent.transformed(plt.gcf().dpi_scale_trans.inverted())
        plt.gcf().savefig(memf, format='PNG', bbox_inches=extent)
        memf.seek(0)
        with MemoryFile(memf) as memfile:
            with memfile.open() as dataset:
                figure_array = dataset.read()
        memf.close()
        plt.close()

    return figure_array


def get_most_changed_area_figures(lost_mask_list: List[DifferenceMask], initial_rgb_image: Image):
    ordered_mask_list = sorted(lost_mask_list, key=lambda x: x.lost_area, reverse=True)
    max_list_size = 10
    if len(ordered_mask_list) < max_list_size:
        max_list_size = int(len(ordered_mask_list) * 0.15)
    selected_lost_mask_list = ordered_mask_list[:max_list_size]

    final_lost_mask_list = []

    for lost_mask in selected_lost_mask_list:
        gray_scale_img = create_grey_scale_img_from_rgb(initial_rgb_image, [lost_mask.lost_mask])
        deforestation_mask = copy(lost_mask.lost_mask)
        deforestation_mask[deforestation_mask != 0] = 0.05
        img_deforestation_with_color_map = ColorMaps.contrast_original.value.apply_color_map(
            numpy.asarray([deforestation_mask]))
        bool_deforestation_mask = deforestation_mask.astype('bool').__invert__()
        img_deforestation_with_color_map[
            numpy.asarray([bool_deforestation_mask, bool_deforestation_mask, bool_deforestation_mask])] = 0

        afected_area_image = numpy.append(
            (gray_scale_img + img_deforestation_with_color_map),
            numpy.ones((1, 256, 256), dtype='uint8') * 255, axis=0)

        fig, ax = plt.subplots()
        ax.imshow(numpy.transpose(afected_area_image, (1, 2, 0)).astype('uint8'))

        legend_elements = [matplotlib.patches.Patch(facecolor='red', label='Deforestaded Area')]
        ax.legend(handles=legend_elements, loc='lower right')
        ax.axis('off')

        with io.BytesIO() as memf:
            extent = plt.gcf().get_window_extent()
            extent = extent.transformed(plt.gcf().dpi_scale_trans.inverted())
            plt.gcf().savefig(memf, format='PNG', bbox_inches=extent)
            memf.seek(0)
            with MemoryFile(memf) as memfile:
                with memfile.open() as dataset:
                    figure_array = dataset.read()
            memf.close()
            plt.close()

        final_lost_mask_list.append(DifferenceMask(
            lost_mask.difference_date,
            lost_mask.lost_mask,
            lost_mask.lost_area,
            lost_mask.lost_area_geojson,
            figure_array
        ))

    return final_lost_mask_list


def create_afected_area_image(initial_rgb_image: Image, interpolated_deforestation_area: numpy.array,
                              interpolated_recovered_area: numpy.array):
    gray_scale_img = create_grey_scale_img_from_rgb(
        initial_rgb_image, [interpolated_recovered_area, interpolated_deforestation_area]
    )

    mask_deforestation = interpolated_deforestation_area == 0
    mask_recoverage = interpolated_recovered_area == 0

    img_deforestation_with_color_map = ColorMaps.contrast_original.value.apply_color_map(
        numpy.asarray([interpolated_deforestation_area]))
    img_recovered_with_color_map = ColorMaps.contrast_original.value.apply_color_map(
        numpy.asarray([interpolated_recovered_area]))

    img_deforestation_with_color_map[numpy.asarray([mask_deforestation, mask_deforestation, mask_deforestation])] = 0
    img_recovered_with_color_map[numpy.asarray([mask_recoverage, mask_recoverage, mask_recoverage])] = 0

    afected_area_image = numpy.append(
        (gray_scale_img + img_recovered_with_color_map + img_deforestation_with_color_map),
        numpy.ones((1, 256, 256), dtype='uint8') * 255, axis=0)

    fig, ax = plt.subplots(nrows=2, ncols=1, height_ratios=[3, 0.2], width_ratios=[0.5], layout="constrained")

    custom_cmap = (matplotlib.colors.LinearSegmentedColormap.from_list("custom", ['red', 'yellow', 'green']))
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    ax[1].axis('off')
    fig.colorbar(matplotlib.cm.ScalarMappable(cmap=custom_cmap),
                 cax=cax, orientation='horizontal', label="Most Deforestade Area to Most Recovered Area", )

    cax.set_xticks([])

    ax[0].imshow(numpy.transpose(afected_area_image, (1, 2, 0)))
    ax[0].axis('off')

    with io.BytesIO() as memf:
        extent = plt.gcf().get_window_extent()
        extent = extent.transformed(plt.gcf().dpi_scale_trans.inverted())
        plt.gcf().savefig(memf, format='PNG', bbox_inches=extent)
        memf.seek(0)
        with MemoryFile(memf) as memfile:
            with memfile.open() as dataset:
                figure_array = dataset.read()
        memf.close()
        plt.close()

    return figure_array


def create_afected_area_informations(masked_ndvi_images: List[MaskedImage], initial_rgb_image: Image,
                                     original_geometry: Polygon, image_size: int):
    fig_means_stdev_max_min = plot_means_stdev_max_min(masked_ndvi_images)

    # os.mkdir('./images/graphs')
    # os.mkdir('./images/visualizer')
    # os.mkdir('./images/visualizer/deforestation_areas')
    create_analisys_files('graphs/means_stdev_max_min_graph', fig_means_stdev_max_min)

    interpolated_deforestation_area, interpolated_recovered_area = calculate_afected_area(masked_ndvi_images)
    initial_date = masked_ndvi_images[FIRST_POSITION].acquisition_date
    final_date = masked_ndvi_images[-1].acquisition_date
    fig_affected_area_graph = plot_afect_area(interpolated_deforestation_area, interpolated_recovered_area,
                                              original_geometry, image_size, initial_date, final_date)
    create_analisys_files('graphs/affected_area_graph', fig_affected_area_graph)

    afected_area_image = create_afected_area_image(initial_rgb_image, interpolated_deforestation_area,
                                                   interpolated_recovered_area)
    create_analisys_files('visualizer/afected_area_image', afected_area_image)

    difference_mask_list = calculate_list_difference_between_days(masked_ndvi_images,
                                                                  initial_rgb_image.metadata.get('transform'),
                                                                  original_geometry, image_size)

    fig_deforestation_area_graph = plot_deforestation_area_throught_dates(difference_mask_list)
    create_analisys_files('graphs/deforestation_area_graph', fig_deforestation_area_graph)

    list_deforestation_diff_area_obj = get_most_changed_area_figures(difference_mask_list, initial_rgb_image)

    for index_diff, diff_mask in enumerate(list_deforestation_diff_area_obj):
        if os.path.exists(f'./images/visualizer/deforestation_areas/{index_diff+1}') is False:
            os.mkdir(f'./images/visualizer/deforestation_areas/{index_diff+1}')
        create_analisys_files(
            f'visualizer/deforestation_areas/{index_diff+1}/lost_area_{index_diff+1}', diff_mask.lost_area_image, 'png')
        create_analisys_files(
            f'visualizer/deforestation_areas/{index_diff + 1}/lost_area_{index_diff + 1}_geojson',
            diff_mask.lost_area_geojson, 'json')
