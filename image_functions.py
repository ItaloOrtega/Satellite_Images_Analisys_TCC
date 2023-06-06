import datetime
import time
from typing import Tuple, List

import numpy
import rasterio
from rasterio import MemoryFile, DatasetReader
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject
from shapely import Polygon

from rasterio.windows import from_bounds as window_from_bounds
from rasterio.transform import from_bounds as transform_from_bounds

from models import Image, ColorMap

import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

_FIRST_POSITION = 0
_FIRST_MASK_POSITION = 1
_WIDTH_POSITION = 2
_HEIGHT_POSITION = 1
_NODATA_VALUE = 0


def to_dataset(data: numpy.ndarray, metadata: dict, mask: numpy.ndarray) -> DatasetReader:
    """
    Returns a open Image as a dataset.

    :param data: The numpy.array containing the values of the bands of the image
    :param metadata: The metadata from the image, like resolution, transform equation, etc
    :param mask: The numpy.array of the mask of the image bands
    :return: The opened image as dataset
    """
    memory_file = MemoryFile()
    with memory_file.open(**metadata) as dataset:
        if mask is not None:
            dataset.write_mask(mask)
        dataset.write(data)
    return memory_file.open()


def get_scene_id(source: str, acquisition_date: datetime.date):
    """
    Creates a Scene ID from the opened scenes, using the acquisition date.

    :param source: Source of the Scene
    :param acquisition_date: Date of the acquisition of the Scene that the image is in
    :return: Scene ID of the merged scenes
    """
    return f"{source.replace('-', '_').upper()}_{str(acquisition_date).replace('-', '_').upper()}"


def transform_epsg_image(data: numpy.ndarray, mask: numpy.ndarray, profile: dict, raster_bounds: Tuple,
                         to_epsg: int = 3857
                         ):
    """
    Transforms an image that is in a EPSG to another.

    :param data: The image bands as a ndarray
    :param mask: The mask of the image
    :param profile: The metadata of the image
    :param raster_bounds: The bounds where the image is inside
    :param to_epsg: The EPSG that the image is being transform to
    :returns: Returns the transformed image
    """
    transform, width, height = calculate_default_transform(
        profile['crs'], {'init': f'EPSG:{to_epsg}'}, profile['width'], profile['height'], *raster_bounds
    )
    data, transform = reproject(
        source=data,
        destination=numpy.zeros((profile['count'], height, width), profile['dtype']),
        src_transform=profile['transform'],
        src_crs=profile['crs'],
        dst_transform=transform,
        dst_crs={'init': f'EPSG:{to_epsg}'},
        resampling=Resampling.nearest,
    )
    profile.update(
        driver='GTiff',
        height=height,
        width=width,
        transform=transform,
        crs=rasterio.crs.CRS.from_epsg(code=to_epsg),
    )
    return to_dataset(data, profile, mask)


def merge_datasets(list_datasets: List[DatasetReader], geom: Polygon):
    """
    Merge datasets of the scenes from the same day in one single numpy.array, and updates the metadata to be equal to
    the merged data.

    :param list_datasets: List of Scenes Datasets to be merged
    :param geom: Polygon object of the area requested
    :return: Merged numpy.array and metadata of the Scenes from the same day
    """
    merged_bands, transform = merge(datasets=list_datasets, bounds=geom.bounds)
    profile = list_datasets[_FIRST_POSITION].profile.copy()

    profile.update(
        transform=transform,
        driver='GTiff',
        height=merged_bands.shape[_HEIGHT_POSITION],
        width=merged_bands.shape[_WIDTH_POSITION],
    )

    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as dataset_raster:
            metadata = dataset_raster.profile

    return merged_bands, metadata


def windowed_read_dataset(
        dataset: DatasetReader,
        geom: Polygon,
        no_data_value: int,
        first_mask_position: int,
        image_size: int
) -> DatasetReader:
    """
    Reads the image inside a window
    :param dataset: Dataset of the image read of the bucket
    :param geom: Polygon object of the area requested
    :param no_data_value: The no-data value of the image
    :param first_mask_position: Position of the first mask of the image
    :param image_size: Max size of the image
    :return: Returns the dataset of the read image inside a window
    """
    profile = dataset.profile.copy()

    bounds = geom.bounds
    min_lon, min_lat, max_lon, max_lat = bounds

    tile_window = window_from_bounds(
        left=min_lon, bottom=min_lat, right=max_lon, top=max_lat, transform=dataset.transform
    )

    output_height = image_size
    output_width = image_size

    transform_with_bounds = transform_from_bounds(
        north=max_lat,
        south=min_lat,
        west=min_lon,
        east=max_lon,
        height=output_height,
        width=output_width,
    )
    time_start = time.perf_counter()
    data = dataset.read(
        window=tile_window,
        boundless=False,
        out_shape=(output_height, output_width),
        fill_value=no_data_value,
        resampling=Resampling.nearest
    )
    print(f'Time to read image = {time.perf_counter() - time_start} s')

    profile.update(
        driver='GTiff',
        height=output_height,
        width=output_width,
        transform=transform_with_bounds,
        count=data.shape[_FIRST_POSITION],
    )

    mask = dataset.read_masks(
        first_mask_position, window=tile_window, boundless=False, out_shape=(output_height, output_width)
    )

    return to_dataset(data, profile, mask)


def open_single_image(image_url: str, image_size: int, geometry: Polygon, scene_id: str) -> DatasetReader:
    """
    Opens a single image with rasterio.open()

    :param image_url: Image URL
    :param image_size: Max size of the image
    :param geometry: Polygon object of the area requested
    :return: Dataset of the opened image
    """
    with rasterio.Env(aws_unsigned=True):
        with rasterio.open(image_url) as dataset:
            with WarpedVRT(dataset) as vrt:
                try:
                    windowed_dataset = windowed_read_dataset(
                        dataset=vrt,
                        geom=geometry,
                        no_data_value=_NODATA_VALUE,
                        first_mask_position=_FIRST_MASK_POSITION,
                        image_size=image_size
                    )
                except Exception as e:
                    print(f'Error when opening the scene {scene_id}.')
                    rasterio.Env().cache = None
                    return None

    rasterio.Env().cache = None

    return windowed_dataset


def create_image_with_rasterio(image: Image, color_map: ColorMap, file_extension: str):
    """
    Applies the raster type to the bands of the image and creates an image of the file extension asked by the user
    using rasterio.

    :param image: Object RasterImage containing the data and metadata of the tile
    :param color_map: Color Map that will be applied in the image
    :param file_extension: NamedTuple of the type of the file that will be returned to the user
    :return: A bytes array of the image file in the file_extension format
    """
    image_with_color_map = color_map.apply_color_map(image.data)
    raster_image = numpy.append(image_with_color_map, [image.mask], axis=0)
    with MemoryFile() as memfile:
        if file_extension == 'tiff':
            raster_image = numpy.append(raster_image, [image.cloud_mask], axis=0)
            metadata = image.metadata
            metadata.update(count=raster_image.shape[0], dtype=raster_image.dtype)
            if color_map.type == 'raw':
                metadata.update(nodata=0)
            with memfile.open(**metadata) as opened_memfile:
                opened_memfile.write(raster_image)
        else:
            with memfile.open(
                driver=file_extension,
                count=raster_image.shape[0],
                height=raster_image.shape[1],
                width=raster_image.shape[2],
                dtype=raster_image.dtype,
                nodata=0,
            ) as dst:
                dst.write(raster_image)
        return memfile.read()
