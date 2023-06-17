from copy import copy
from typing import Optional, List

import numpy
from expression import Expression_Parser as ExpressionParser
from matplotlib import pyplot as plt
from rasterio.features import geometry_mask
from scipy.ndimage import binary_dilation
from shapely import Polygon

from bands_interpolator import __interpolate_bands
from models import Image, Index, SourceBands, IndexExpressionType

LENGTH_OF_SHAPE = 3
_MATRICES_AXIS = 0


def calculate_ndvi(red: numpy.array, nir: numpy.array):
    return numpy.divide(numpy.subtract(nir, red), numpy.add(nir, red))


def calculate_savi(red: numpy.array, nir: numpy.array):
    return numpy.divide(numpy.multiply(numpy.subtract(nir, red), 1.5), numpy.add(numpy.add(nir, red), 0.5))


def calculate_rgb(red: numpy.array, green: numpy.array, blue: numpy.array):
    return numpy.asarray([red, green, blue])


def calculate_evi(red: numpy.array, blue: numpy.array, nir: numpy.array, c1: float, c2: float):
    return numpy.divide(
        numpy.multiply(2.5, numpy.subtract(nir, red)),
        numpy.add(
            numpy.subtract(
                numpy.add(
                    nir,
                    numpy.multiply(
                        c1,
                        red
                    )
                ),
                numpy.multiply(
                    c2,
                    blue
                )
            ),
            10000
        )
    )


def calculate_ndwi(green: numpy.array, nir: numpy.array):
    return numpy.divide(numpy.subtract(green, nir), numpy.add(nir, green))


def calculate_ndvig(green: numpy.array, nir: numpy.array):
    return numpy.divide(numpy.subtract(nir, green), numpy.add(nir, green))


def calculate_rvi(red: numpy.array, nir: numpy.array):
    return numpy.divide(nir, red)


def calculate_arvi(red: numpy.array, blue: numpy.array, nir: numpy.array):
    return numpy.divide(
        numpy.subtract(nir, numpy.add(numpy.subtract(red, blue), red)),
        numpy.add(nir, numpy.add(numpy.subtract(red, blue), red))
    )


def calculate_nir(green: numpy.array, blue: numpy.array, nir: numpy.array):
    return numpy.asarray([nir, blue, green])


def calculate_vari(red: numpy.array, green: numpy.array, blue: numpy.array):
    return numpy.divide(numpy.subtract(green, red), numpy.subtract(numpy.add(green, red), blue))


def calculate_raw(red: numpy.array, green: numpy.array, blue: numpy.array, nir: numpy.array, cloud_mask: numpy.array):
    return numpy.asarray([red, blue, green, nir, cloud_mask])


INDEX_FUNCS = {
    "ndvi": calculate_ndvi,
    "savi": calculate_savi,
    "rgb": calculate_rgb,
    "evi": calculate_evi,
    "ndwi": calculate_ndwi,
    "ndvig": calculate_ndvig,
    "rvi": calculate_rvi,
    "arvi": calculate_arvi,
    "nir": calculate_nir,
    "vari": calculate_vari,
    "raw": calculate_raw
}


def update_array_shape(array: numpy.ndarray):
    """
    Adds a new axis to the array, so it goes from a 2D array to a 3D one.

    :param array: The original numpy array
    :return: Returns the array with the updated shape
    """
    if len(array.shape) < LENGTH_OF_SHAPE:
        array = array[numpy.newaxis, ...]
    return array


def __interpolate_values(bands: numpy.ndarray, ceiling_value: int):
    """
    Sets values greater than 1 to the ceiling value of the index used on the bands

    :param bands: Bands with the index applied
    :param ceiling_value: Ceiling value of the index
    """
    result = bands
    result[result > 1] = ceiling_value
    return result


def calculate_cloud_mask(original_cloud_mask: numpy.array, rgb_data: numpy.array, image_mask: numpy.array):
    """
    Calculates a cloud mask using the SCL band mask
    """

    new_cloud_mask = copy(original_cloud_mask)
    new_cloud_mask[new_cloud_mask <= 6] = 255
    new_cloud_mask[new_cloud_mask <= 11] = 0

    rgb_interpolated = copy(rgb_data)

    white_values_mask = (rgb_interpolated >= 160).sum(axis=0)

    plt.imshow(white_values_mask, cmap='Greys')
    plt.title(f'cloud mask 2')
    plt.show()

    white_values_mask[white_values_mask != 3] = 255
    white_values_mask[white_values_mask == 3] = 0

    final_cloud_mask = new_cloud_mask + white_values_mask
    final_cloud_mask[final_cloud_mask <= 255] = 0
    final_cloud_mask[final_cloud_mask > 255] = 255

    final_cloud_mask += image_mask

    final_cloud_mask[final_cloud_mask == 255] = 0
    final_cloud_mask[final_cloud_mask > 0] = 255

    structuring_element = numpy.array([[1, 1, 1],
                                       [1, 1, 1],
                                       [1, 1, 1]])

    dilated_cloud_mask = binary_dilation(final_cloud_mask.astype('bool').__invert__(), structure=structuring_element)

    dilated_cloud_mask = dilated_cloud_mask.astype('uint8')
    dilated_cloud_mask[dilated_cloud_mask == 0] = 255
    dilated_cloud_mask[dilated_cloud_mask == 1] = 0

    valid_pixels_count = numpy.count_nonzero(dilated_cloud_mask == 255)
    percentage_of_cloud = 100 - (
                valid_pixels_count / (dilated_cloud_mask.shape[0] * dilated_cloud_mask.shape[1])) * 100
    return dilated_cloud_mask, percentage_of_cloud


def create_gray_scale_img_from_rgb(initial_rgb_image: Image, list_mask: List[numpy.array]):
    """
    Creates a gray-scale numpy array image to be used as background, and reset the pixel values as 0 if a list of masks
    is sent.
    """
    gray_scale_band = numpy.dot(initial_rgb_image.data.transpose(1, 2, 0), [0.2989, 0.5870, 0.1140])
    for mask in list_mask:
        gray_scale_band[mask != 0] = 0
    gray_scale_img = numpy.asarray([gray_scale_band, gray_scale_band, gray_scale_band], dtype='uint8')
    return gray_scale_img


def create_image_mask(image: Image, geometry: Polygon):
    """
    Creates a no-data mask from the image data.

    :param image: Object Image containing the data and metadata of the image
    :param geometry: The polygon geometry of the image area
    :return: Returns the mask numpy.array
    """
    # Creates a mask from the image raster, where 0 has data on that pixel and 1 doesn't
    res_mask = geometry_mask(geometries=[geometry], out_shape=image.mask.shape,
                             transform=image.metadata.get('transform')).astype(
        int
    )
    # Change the original values to 0 and 255
    res_mask[res_mask == 0] = 255
    res_mask[res_mask == 1] = 0

    data = image.data

    res_values_sum = res_mask + data.sum(axis=_MATRICES_AXIS)

    res_values_sum[res_values_sum == 255] = 0
    res_values_sum[res_values_sum > 0] = 255

    res_mask = res_values_sum.astype('uint8')

    return res_mask


def calculate_image_index(index_name: str, image_vars: dict) -> numpy.array:
    """
    Calculates the index using the bands from the image and the function related to the index.

    :param index_name: Name of the index requested
    :param image_vars: Dict containing the bands arrays as vars
    :return: Numpy array of the image with the index applied on it.
    """
    parser = ExpressionParser(variables=image_vars, functions=INDEX_FUNCS)
    index_function = INDEX_FUNCS[index_name]
    necessary_bands = str(index_function.__code__.co_varnames).replace("'", "")
    image_with_index = parser.parse(f"{index_name}{necessary_bands}")
    return image_with_index


def filter_interpolate_image_data(
        image_with_index: numpy.array, index: Index, source: SourceBands
) -> numpy.array:
    """
    Filter and interpolate the values of the image after the index being applied,
    to remove nan values and other managements.

    :param image_with_index: Image numpy array with the index applied
    :param index: Index object
    :param source: Source of the image
    :return: Image numpy array
    """
    if numpy.isnan(image_with_index).any():
        image_with_index = numpy.nan_to_num(image_with_index)

    if index.ceiling_value is not None:
        image_with_index = __interpolate_values(
            bands=image_with_index,
            ceiling_value=index.ceiling_value)

    if index.name in ('rgb', 'nir'):
        image_with_index = __interpolate_bands(bands=image_with_index, source=source)

    image_with_index = image_with_index.astype('float32')

    image_with_index = update_array_shape(image_with_index)

    return image_with_index


def get_rgb_image(scene_image: Image, geometry: Polygon, source: SourceBands):
    """
    Creates an RGB image and mask from the Scene image.

    :param scene_image: Scene image object contaning all the bands
    :param geometry: Polygon geometry of the image area
    :param source: Source of the Scenes
    :return: RGB Image object
    """
    _LINES_INDEX = 1
    _COLUMNS_INDEX = 2

    rgb_index = IndexExpressionType.rgb.value

    rgb_image_metadata = copy(scene_image.metadata)

    band_vars = scene_image.create_bands_vars()

    if 0 in [color_band.max() for color_band in list(band_vars.values())[:4]]:
        print(f"Image {scene_image.id} has no valid values.")
        return None

    image_with_index = calculate_image_index(rgb_index.name, band_vars)

    filtered_image_with_index = filter_interpolate_image_data(image_with_index, rgb_index, scene_image.source)

    rgb_image_metadata.update({'count': filtered_image_with_index.shape[_MATRICES_AXIS], 'dtype': 'float32'})

    rgb_image = Image(
        metadata=rgb_image_metadata,
        data=filtered_image_with_index,
        mask=numpy.ones(
            (filtered_image_with_index.shape[_LINES_INDEX], filtered_image_with_index.shape[_COLUMNS_INDEX])) * 255,
        cloud_mask=band_vars['cloud_mask'],
        id=f'IMG_{rgb_index.name.upper()}_{scene_image.id}',
        acquisition_date=scene_image.acquisition_date,
        source=source
    )

    rgb_image.mask = create_image_mask(rgb_image, geometry)

    valid_pixels_count = numpy.count_nonzero(rgb_image.mask == 255)
    percentage_of_invalid_pixels = 100 - (
                valid_pixels_count / (rgb_image.mask.shape[0] * rgb_image.mask.shape[1])) * 100

    if percentage_of_invalid_pixels > 20.0:
        print(f'Image {rgb_image.id} have insufficient pixels for analyzes.')
        return None

    return rgb_image


def get_image_with_index(original_image: Image, index: Index) -> Optional[Image]:
    """
    Get image with index applied on it, with a no-data mask add it.

    :param original_image: Scene object with numpy array and metadata
    :param index: Index to be applied in the original Image
    :return: Image with the index applied and the no-data mask
    """

    band_vars = original_image.create_bands_vars()

    index_image_metadata = original_image.metadata

    if 0 in [color_band.max() for color_band in list(band_vars.values())[:4]]:
        print(f"Image {original_image.id} has no valid values.")
        return None

    image_with_index = calculate_image_index(index.name, original_image.create_bands_vars())

    filtered_image_with_index = filter_interpolate_image_data(image_with_index, index, original_image.source)

    index_image_metadata.update({'count': filtered_image_with_index.shape[_MATRICES_AXIS], 'dtype': 'float32'})

    index_image = Image(
        metadata=index_image_metadata,
        data=filtered_image_with_index,
        mask=original_image.mask,
        cloud_mask=original_image.cloud_mask,
        id=f'IMG_{index.name.upper()}_{original_image.id}',
        acquisition_date=original_image.acquisition_date,
        source=original_image.source
    )

    return index_image


def get_image_with_index_old(image: Image, index: Index, geometry: Polygon, max_cloud_coverage: float) -> Optional[
    Image]:
    """
    Get image with index applied on it, with a no-data mask add it.

    :param image: Image object with numpy array and metadata
    :param index: Index to be applied in the original Image
    :param geometry: Polygon geometry of the image area
    :param max_cloud_coverage: Max of percentage of cloud of the scene
    :return: Image with the index applied and the no-data mask
    """

    band_vars = image.create_bands_vars()

    if 0 in [color_band.max() for color_band in list(band_vars.values())[:4]]:
        print(f"Image {image.id} has no valid values.")
        return None

    image.cloud_mask, percentage_of_clouds = calculate_cloud_mask(band_vars, image.source)

    if percentage_of_clouds <= max_cloud_coverage:
        image_with_index = calculate_image_index(index.name, image.create_bands_vars())

        filtered_image_with_index = filter_interpolate_image_data(image_with_index, index, image.source)

        image.data = filtered_image_with_index

        image.metadata.update({'count': filtered_image_with_index.shape[_MATRICES_AXIS], 'dtype': 'float32'})

        image.mask = create_image_mask(image, geometry)

        return image

    print(f"Image {image.id} has clouds over the threshold, {percentage_of_clouds}.")

    return None
