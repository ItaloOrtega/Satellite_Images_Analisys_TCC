import numpy
from expression import Expression_Parser as ExpressionParser
from rasterio.features import geometry_mask
from shapely import Polygon

from models import Image, Index, SourceBands

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
    "vari": calculate_vari
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


def __interpolate_one_band(band: numpy.ndarray, source: SourceBands, color_name: str):
    """
    Interpolates one color band to the correct values for presentation of its color.

    :param band: Image Bands with the dbx applied
    :param source: Where the image bands comes from
    :param color_name: The color of that band
    :return: The bands with the values interpolated
    """

    interpolation_value = source.band_interp_values[color_name]
    if interpolation_value is None:
        return band

    band_interpolated = numpy.interp(band, interpolation_value, [0, 255])
    return band_interpolated


def __interpolate_bands(bands: numpy.ndarray, source: SourceBands):
    """
    Interpolates the RGB bands to the correct values for presentation.

    :param bands: Image Bands with the dbx applied
    :param source: Where the image bands comes from
    :return: The bands with the values interpolated
    """

    bands_interpolated = numpy.zeros(bands.shape)
    bands_interpolated[0] = __interpolate_one_band(bands[0], source, 'red')
    bands_interpolated[1] = __interpolate_one_band(bands[1], source, 'green')
    bands_interpolated[2] = __interpolate_one_band(bands[2], source, 'blue')
    return bands_interpolated


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
    res_values_sum[res_values_sum > 255] = 255

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


def get_image_with_index(image: Image, index: Index, geometry: Polygon) -> Image:
    """
    Get image with index applied on it, with a no-data mask add it.

    :param image: Image object with numpy array and metadata
    :param index: Index to be applied in the original Image
    :param geometry: Polygon geometry of the image area
    :return: Image with the index applied and the no-data mask
    """

    image_with_index = calculate_image_index(index.name, image.create_bands_vars())

    filtered_image_with_index = filter_interpolate_image_data(image_with_index, index, image.source)

    image.data = filtered_image_with_index

    image.metadata.update({'count': filtered_image_with_index.shape[_MATRICES_AXIS], 'dtype': 'float32'})

    image.mask = create_image_mask(image, geometry)

    return image
