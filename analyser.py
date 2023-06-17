import datetime
import os
from collections import namedtuple, defaultdict
from copy import copy
from typing import List

from area import area
from rasterio import transform
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import measure
import numpy
from affine import Affine
from rasterio.features import rasterize
from shapely import Polygon, intersects, MultiPolygon
from shapely.geometry import mapping

from geom_functions import epsg_transform
from graphs_functions import plot_full_affected_area_figure, plot_single_deforestation_affected_area_figure, \
    plot_deforestation_area_throught_dates_figure, plot_lost_gain_area_graph_figure, plot_mean_stdev_max_min_figure
from indices_functions import create_gray_scale_img_from_rgb
from models import Image, MaskedImage, ColorMaps

ImageInfo = namedtuple('ImageInfo', ['acquisition_date', 'mean', 'stdev', 'max', 'min'])
DifferenceMask = namedtuple('DifferenceMask', [
    'difference_date', 'lost_mask', 'lost_area', 'lost_area_geojson', 'lost_area_image'
])

FIRST_POSITION = 0
SECOND_POSITION = 1


def create_image_information_from_masked_image(list_masked_images: List[MaskedImage]) -> List[ImageInfo]:
    """
    Calculate the information (max, min, mean, stdev) and creates an ImageInfo object from a MaskedImage object
    """
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
    """
    Creates a Polygon shape from the contour of the valid pixels of the numpy array of the image and using the
    original image transform.
    """
    lines_values = [x for x, y in image_contour_array]
    columns_values = [y for x, y in image_contour_array]
    x_coordinates, y_coordinates = transform.xy(original_transform, lines_values, columns_values)
    transformed_contour = [[x_coordinates[coord_index], y_coordinates[coord_index]] for coord_index in
                           range(len(x_coordinates))]
    numpy_contour = numpy.asarray(transformed_contour)
    polygon_contour = Polygon(numpy_contour)
    return polygon_contour


def create_means_stdev_max_min_plot_figure(list_images_info: List[ImageInfo]):
    """
    Calculates the max, min, mean and standart deviation of the images grouped by months and them plots those
    information in a graph.
    """
    image_info_sub_lists = defaultdict(list)

    for img_info in list_images_info:
        image_info_sub_lists[(img_info.acquisition_date.year, img_info.acquisition_date.month)].append(copy(img_info))

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
            img_info.data.mask[img_info.data.data < 0] = True
            sub_list_mean.append(img_info.data.mean())
            sub_list_stdevs = [numpy.std(img_info.data)]
            sub_list_maxs = [img_info.data.max()]
            sub_list_mins = [img_info.data.min()]
        list_means.append(sum(sub_list_mean) / len(sub_list_mean))
        list_stdevs.append(sum(sub_list_stdevs) / len(sub_list_stdevs))
        list_maxs.append(sum(sub_list_maxs) / len(sub_list_maxs))
        list_mins.append(sum(sub_list_mins) / len(sub_list_mins))

    return plot_mean_stdev_max_min_figure(months_list, list_means, list_stdevs, list_maxs, list_mins)


def create_affected_area_plot_figure(
        deforestation_area_array: numpy.array, recovered_area_array: numpy.array,
        original_geometry: Polygon, image_size: int, initial_date: datetime, final_date: datetime
):
    """
    Calculates the deforestation and reforestation area in hectares of the user's selected area and creates
    the graph of those information
    :param deforestation_area_array: Numpy array containing the pixels that represents the deforestation area
    :param recovered_area_array: Numpy array containing the pixels that represents the recovered area
    :param original_geometry: Original Polygon geometry of the selected area
    :param image_size: Image Size
    :param initial_date: Initial of the date window
    :param final_date: Final of the date window
    :return: Affected Area graph figure
    """
    pixel_area = area(mapping(original_geometry)) / (image_size * image_size)
    deforestation_area = ((deforestation_area_array != 0).sum() * pixel_area) / 10000
    recovered_area = ((recovered_area_array != 0).sum() * pixel_area) / 10000
    affected_area_hectare_figure = plot_lost_gain_area_graph_figure(deforestation_area, recovered_area, initial_date, final_date)

    return affected_area_hectare_figure


def calculate_afected_area(list_masked_images: List[MaskedImage]):
    """
    Calculates the affected area of the wanted user area, by getting the inital values of the area and the occurences
    of deforestation and recoverage of green area.
    """
    # Initializes the deforestaion and reforestation arrays
    area_of_deforestation = numpy.zeros(list_masked_images[FIRST_POSITION].data.shape)
    area_of_reforestation = numpy.zeros(list_masked_images[FIRST_POSITION].data.shape)
    # Initials arrays representing areas already deforestaded/green
    initial_deforestad_area = None
    initial_green_area = None

    # For LOOP to get the occurences of lost and gained green area
    for img_index, masked_image in enumerate(list_masked_images):
        # Creates the numpy array from the MaskedImage
        deforestation_image_area = masked_image.data.filled(0)
        reforestation_image_area = masked_image.data.filled(0)

        # Gets only pixels that have lower of 0.5 and greater than 0, that represent non-green area
        deforestation_image_area[deforestation_image_area > 0.5] = 0
        deforestation_image_area[deforestation_image_area <= 0] = 0
        deforestation_image_area[deforestation_image_area != 0] = 1

        # Occurence of Pixels only greater 0.5, that represents the green area
        reforestation_image_area[reforestation_image_area < 0.5] = 0
        reforestation_image_area[reforestation_image_area != 0] = 1

        # Adds the occurences to the respectives arrays
        if img_index != 0:
            area_of_deforestation += deforestation_image_area
            area_of_reforestation += reforestation_image_area
        # If the index == 0, it is the first image of the window date. where that all pixels in those arrays position
        # are disconsidered
        else:
            initial_deforestad_area = deforestation_image_area
            initial_green_area = reforestation_image_area

    # Reset pixels with less occurences of deforestation/recoverage of areas
    occurency_threshold = 3
    if len(list_masked_images) < 10:
        occurency_threshold = 2
    area_of_deforestation[area_of_deforestation < occurency_threshold] = 0
    area_of_reforestation[area_of_reforestation < occurency_threshold] = 0

    # Removes areas that was already deforestaded/recovered from the respective mask arrays
    area_of_deforestation[initial_deforestad_area != 0] = 0
    area_of_reforestation[initial_green_area != 0] = 0

    # Interpolates the numpy arrays to have similar values to the NDVI index for better visualization in the final image
    interpolated_recovered_area = numpy.interp(area_of_reforestation,
                                               [0,
                                                (area_of_reforestation[area_of_reforestation != 0]).min(),
                                                area_of_reforestation.max()], [0, 0.5, 1])

    interpolated_deforestation_area = numpy.interp(area_of_deforestation,
                                                   [0, (area_of_deforestation[area_of_deforestation != 0]).min(),
                                                    area_of_deforestation.max()], [0, 0.45, 0.05])

    return interpolated_deforestation_area, interpolated_recovered_area


def calculate_list_difference_between_days(masked_images_list: List[MaskedImage], original_transform: Affine,
                                           original_geometry: Polygon, image_size: int):
    """
    Calculates the difference of gain and loss of green area in consecutive days images, to see what are the days with
    most deforestation in the date window. The calculation of the area affected is possible by checking the negative and
    positive difference between the images, then using the contours function to create a Polygon object of those pixels
    to see how muc hectares in area that change made and if it's valid to the minimal chage to be added to the list.
    """
    # Calculates the area of one pixel in a numpy.array and the minimal area valid to be added as a valid change
    pixel_area = area(mapping(original_geometry)) / (image_size * image_size)
    print(f'Total area = {area(mapping(original_geometry))} | Pixel Area = {pixel_area} m²')
    minimal_contour_array = numpy.zeros((image_size, image_size))
    minimal_contour_array[120:124, 120:124] = 255
    minimal_contour = measure.find_contours(minimal_contour_array)
    minimal_contour_geometry = create_polygon_from_image_contour(minimal_contour[FIRST_POSITION], original_transform)
    area_minimal_contour_geometry = area(mapping(epsg_transform(minimal_contour_geometry, 32722, 4326)))
    print(f'Minimal area = {area_minimal_contour_geometry}')

    list_difference_mask = []
    lost_area = numpy.zeros((1, image_size, image_size))
    gain_area = numpy.zeros((1, image_size, image_size))

    # For Loop to create the difference mask of day X to day X+1
    for img_index, masked_image in enumerate(masked_images_list):
        if img_index != len(masked_images_list) - 1:
            change_date = str(masked_image.acquisition_date) + '->' + str(
                masked_images_list[img_index + 1].acquisition_date)
            diff_mask_img = numpy.subtract(masked_images_list[img_index + 1].data, masked_image.data).data
            int_mask = numpy.add(masked_image.data.mask, masked_images_list[img_index + 1].data.mask).astype('uint8')
            diff_mask_img[int_mask == 1] = 0

            # Creates the masks related only to loss and gain area
            img_gain = numpy.copy(diff_mask_img)
            img_loss = numpy.copy(diff_mask_img)

            # Only differences higher or equal to +0.3 are recognized as gain of green area
            img_gain[img_gain < 0.3] = 0
            img_gain[gain_area != 0] = 0
            gain_area[img_gain != 0] = 1

            # Only differences lower or equal to -0.3 are recognized as loss of green area
            img_loss[img_loss > -0.3] = 0
            img_loss[lost_area != 0] = 0
            lost_area[img_loss != 0] = 1

            list_difference_mask.append((change_date, img_loss, img_gain))

    change_date_index = 0
    lost_mask_index = 1
    gain_mask_index = 2

    difference_mask_list = []

    # For Loop to validate the difference mask
    for img_index, difference_mask in enumerate(list_difference_mask):
        original_loss_mask = copy(difference_mask[lost_mask_index])
        if img_index != len(list_difference_mask) - 1:
            # Mask of loss area reset the pixels values if the next day gain mask haves the same values
            # Because if a loss of area happened in day X->X+1, is impossible to be recovered in X+1->X+2
            difference_mask[lost_mask_index][list_difference_mask[img_index + 1][gain_mask_index] != 0] = 0
            # The same case happens to gain masks, but the other way around
            list_difference_mask[img_index + 1][gain_mask_index][original_loss_mask != 0] = 0

        # A dilatation occurs in the lost area mask to be possible to create more easily the countours of the array
        structure_dilation = numpy.array([[1, 1, 1],
                                          [1, 1, 1],
                                          [1, 1, 1]])
        bool_lost_mask = copy(difference_mask[lost_mask_index])

        bool_lost_mask[bool_lost_mask != 0] = 255

        dilated_lost_mask = binary_dilation(bool_lost_mask[FIRST_POSITION].astype('bool'),
                                            structure=structure_dilation).astype('uint8')

        dilated_lost_mask[dilated_lost_mask == 1] = 255
        loss_geojsons = []
        # After the dilatation and change of the values of the mask a list of contours is created and then transformed
        # to valid geojson using the original transform
        image_contours_list = measure.find_contours(dilated_lost_mask)
        for img_contour in image_contours_list:
            loss_geojsons.append(create_polygon_from_image_contour(img_contour, original_transform))

        # Then the valid geometries are selected, valid geometries have the area bigger or equal to the minimal area
        valid_loss_geometries_list = []
        for geometry in loss_geojsons:
            poly_area = area(mapping(epsg_transform(geometry, 32722, 4326)))
            if poly_area >= area_minimal_contour_geometry:
                valid_loss_geometries_list.append(geometry)

        # After having a list of valid geometries, it checks if any other remaining geometries intercts with them
        # and if they do, are added to the list of final values for geometries
        final_valid_geometries_list = copy(valid_loss_geometries_list)
        for valid_loss_geometry in valid_loss_geometries_list:
            for geometry in loss_geojsons:
                if geometry not in valid_loss_geometries_list and intersects(valid_loss_geometry, geometry) is True:
                    final_valid_geometries_list.append(geometry)

        default_numpy_array = numpy.zeros((image_size, image_size))

        if len(final_valid_geometries_list) > 0:
            # Applys the valid geometries to a new numpy array via the rasterize function
            result_valid = rasterize(
                shapes=final_valid_geometries_list, transform=original_transform, out=default_numpy_array, fill=0,
                dtype=float
            )
            result_valid[result_valid == 1] = 255

            # Having the new rasterized array, its applied on the rasterized array an erosion to remove unnecessary
            # pixels
            structure_erosion = numpy.array([[1, 1, 1, 1],
                                             [1, 1, 1, 1],
                                             [1, 1, 1, 1],
                                             [1, 1, 1, 1]])

            final_lost_mask = binary_erosion(result_valid.astype('bool'),
                                             structure=structure_erosion).astype('uint8')
            final_lost_mask = numpy.asarray([final_lost_mask])
            final_lost_mask[difference_mask[lost_mask_index] == 0] = 0

            # Counts the pixels of the lost areas
            loss_count = (final_lost_mask != 0).sum()

            # Creates a geojson containing all the geometries used to rasterize the image
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


def create_deforestation_area_throught_dates_figure(lost_mask_list: List[DifferenceMask]):
    """
    Calculates the sum of all area lost due to deforestation through the months of the date window and plots the figure
    """
    image_info_sub_lists = defaultdict(list)

    for diff_mask in lost_mask_list:
        acquisition_date = datetime.datetime.strptime(
            diff_mask.difference_date.split('->')[SECOND_POSITION], '%Y-%m-%d'
        )
        image_info_sub_lists[(acquisition_date.year, acquisition_date.month)].append(diff_mask.lost_area)

    sorted_image_info_sub_lists = sorted(image_info_sub_lists.items())

    months_list = ['{}-{}'.format(*img_info[FIRST_POSITION]) for img_info in sorted_image_info_sub_lists]
    lost_areas_hac = [round(sum(img_info[SECOND_POSITION]) / 10000, 2) for img_info in sorted_image_info_sub_lists]

    deforestation_area_throught_dates_figure = plot_deforestation_area_throught_dates_figure(
        months_list, lost_areas_hac
    )

    return deforestation_area_throught_dates_figure


def get_most_changed_area_figures(lost_mask_list: List[DifferenceMask], initial_rgb_image: Image):
    """
    Filters the most deforestaded affected areas from a day to another, creates a gray-scale image to be used as
    background, applys the pixels representing deforestaded area to the image.
    """
    # Gets the most area affected images to deforestation
    ordered_mask_list = sorted(lost_mask_list, key=lambda x: x.lost_area, reverse=True)
    max_list_size = 10
    if len(ordered_mask_list) < max_list_size:
        max_list_size = len(ordered_mask_list)
    selected_lost_mask_list = ordered_mask_list[:max_list_size]

    final_lost_mask_list = []

    for lost_mask in selected_lost_mask_list:
        # Creates gray image using the pixels as mask to not change the values when creating the final image
        gray_scale_img = create_gray_scale_img_from_rgb(initial_rgb_image, [lost_mask.lost_mask])
        deforestation_mask = copy(lost_mask.lost_mask)
        # Applys the colormap to the mask
        deforestation_mask[deforestation_mask != 0] = 0.05
        img_deforestation_with_color_map = ColorMaps.contrast_original.value.apply_color_map(deforestation_mask)
        bool_deforestation_mask = deforestation_mask.astype('bool').__invert__()
        # Excludes all 0 values, that in the colormap in RGB are 128
        img_deforestation_with_color_map[
            numpy.concatenate(
                [bool_deforestation_mask, bool_deforestation_mask, bool_deforestation_mask], axis=0
            )] = 0

        # Creates the image and adds to the list
        affected_area_image_array = numpy.append(
            (gray_scale_img + img_deforestation_with_color_map),
            numpy.ones((1, 256, 256), dtype='uint8') * 255, axis=0)

        deforestation_affected_area_figure_array = plot_single_deforestation_affected_area_figure(
            affected_area_image_array, lost_mask.difference_date
        )

        final_lost_mask_list.append(DifferenceMask(
            lost_mask.difference_date,
            lost_mask.lost_mask,
            lost_mask.lost_area,
            lost_mask.lost_area_geojson,
            deforestation_affected_area_figure_array
        ))

    return final_lost_mask_list


def create_affected_area_image(
        initial_rgb_image: Image, interpolated_deforestation_area: numpy.array, interpolated_recovered_area: numpy.array
):
    """
    Creates a gray scale of the initial RGB image to use as templeate to show all the lost and gain green area of the
    users' wanted image.
    """
    gray_scale_img = create_gray_scale_img_from_rgb(
        initial_rgb_image, [interpolated_recovered_area, interpolated_deforestation_area]
    )
    # Creates a mask for all non-zero values
    mask_deforestation = interpolated_deforestation_area == 0
    mask_recoverage = interpolated_recovered_area == 0

    # Applys the wanted color map in the affected area numpy arrays
    img_deforestation_with_color_map = ColorMaps.contrast_original.value.apply_color_map(
        interpolated_deforestation_area)
    img_recovered_with_color_map = ColorMaps.contrast_original.value.apply_color_map(
        interpolated_recovered_area)

    # Removes all values that are 0, which are now is transfomed to 128 in all three bands
    img_deforestation_with_color_map[numpy.concatenate(
        [mask_deforestation, mask_deforestation, mask_deforestation], axis=0
    )] = 0
    img_recovered_with_color_map[numpy.concatenate(
        [mask_recoverage, mask_recoverage, mask_recoverage], axis=0
    )] = 0

    # Adds all the arrays to gray image to finalize it
    afected_area_image = numpy.append(
        (gray_scale_img + img_recovered_with_color_map + img_deforestation_with_color_map),
        numpy.ones((1, 256, 256), dtype='uint8') * 255, axis=0)

    return plot_full_affected_area_figure(afected_area_image)
