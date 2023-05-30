from typing import List, Optional

import numpy

from models import SourceBands

max_reflectance = 3.0
mid_reflectance = 0.13
saturation_value = 1.2
gamma = 1.8
gamma_offset = 0.01
gamma_offset_pow = gamma_offset ** gamma
gamma_offset_range = (1 + gamma_offset) ** gamma - gamma_offset_pow


def interpolate_color_band_to_16bit(color_band: numpy.array):
    """
    Interpolate a color band values from 0-1 to 0-255
    """

    return numpy.interp(color_band, [0, 1.0], [0, 255])


def truecolor_calibration_representation(red_band: numpy.array, green_band: numpy.array, blue_band: numpy.array):
    """
    Interpolates the values of the RGB/NIR to bring more clearer values from Sentinel L2A images.

    :param red_band: Red color band
    :param green_band: Green color band
    :param blue_band: Blue color band
    :return: New interpolated RGB/NIR image
    """
    rgbLin = enhance_color_band_saturation(
        adjust_band_saturation(red_band),
        adjust_band_saturation(green_band),
        adjust_band_saturation(blue_band)
    )

    return numpy.asarray([
        interpolate_color_band_to_16bit(create_sRGB_band(rgbLin[0])),
        interpolate_color_band_to_16bit(create_sRGB_band(rgbLin[1])),
        interpolate_color_band_to_16bit(create_sRGB_band(rgbLin[2]))
    ])


def adjust_band_saturation(color_band: numpy.array):
    """
    Adjust the color band saturation using reflectance.

    """
    return adjust_band_gamma(enhance_band_contrast(color_band, mid_reflectance, 1, max_reflectance))


def adjust_band_gamma(color_band: numpy.array):
    """
    Adjust the color bands using gamma values.

    :param color_band: Color band numpy array
    :return: Color band with gamma adjusted
    """
    return ((color_band + gamma_offset) ** gamma - gamma_offset_pow) / gamma_offset_range


def enhance_color_band_saturation(red_band: numpy.array, green_band: numpy.array, blue_band: numpy.array):
    """
    Enhance the RGB band values with saturation.
    """
    avgS = (red_band + green_band + blue_band) / 3.0 * (1 - saturation_value)
    return [
        (avgS + red_band * saturation_value),
        (avgS + green_band * saturation_value),
        (avgS + blue_band * saturation_value)
    ]


def enhance_band_contrast(color_band: numpy.array, tx: float, ty: float, max_contrast: float):
    """
    Enhance the color band values with contrast.
    """
    band_with_contrast = (color_band / max_contrast)
    return band_with_contrast * (band_with_contrast * (tx / max_contrast + ty - 1) - ty) / (band_with_contrast * (2 * tx / max_contrast - 1) - tx / max_contrast)


def create_sRGB_band(color_band: numpy.array):
    """
    Create a numpy color band into the new RGB values.
    """
    return numpy.where(
        color_band <= 0.0031308, 12.92 * color_band, 1.055 * numpy.power(color_band, 0.41666666666) - 0.055
    )


def __interpolate_one_band(band: numpy.ndarray, source: SourceBands, color_name: str, data_range: Optional[List[int]]):
    """
    Interpolates one color band to the correct values for presentation of its color.

    :param band: Image Bands with the dbx applied
    :param source: Where the image bands comes from
    :param color_name: The color of that band
    :return: The bands with the values interpolated
    """

    if data_range is None:
        data_range = [0, 255]

    interpolation_value = source.band_interp_values[color_name]
    if interpolation_value is None:
        return band

    band_interpolated = numpy.interp(band, interpolation_value, data_range)
    return band_interpolated


def __interpolate_bands(bands: numpy.ndarray, source: SourceBands):
    """
    Interpolates the RGB bands to the correct values for presentation.

    :param bands: Image Bands with the dbx applied
    :param source: Where the image bands comes from
    :return: The bands with the values interpolated
    """
    data_range = None

    if source.source_name == 'sentinel-2-l2a':
        data_range = [0, 1.0]

    bands_interpolated = numpy.zeros(bands.shape)
    bands_interpolated[0] = __interpolate_one_band(bands[0], source, 'red', data_range)
    bands_interpolated[1] = __interpolate_one_band(bands[1], source, 'green', data_range)
    bands_interpolated[2] = __interpolate_one_band(bands[2], source, 'blue', data_range)

    if source.source_name == 'sentinel-2-l2a':
        bands_interpolated = truecolor_calibration_representation(
            bands_interpolated[0], bands_interpolated[1], bands_interpolated[2]
        )

    return bands_interpolated
