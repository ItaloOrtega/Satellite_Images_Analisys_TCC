import json
import os
from typing import List, Union, Optional

import numpy
from matplotlib import pyplot as plt, animation
from rasterio import MemoryFile
from rasterio.profiles import Profile
from models import RasterImage

_FIRST_POSITION = 0
_LAST_POSITION = -1


def create_image_with_rasterio(raster_image: RasterImage, file_extension: str):
    """
    Creates an image of the file extension asked by the user using rasterio.

    :param raster_image: Object RasterImage containing the data and metadata of the tile
    :param file_extension: Type of the file that will be returned to the user
    :return: A bytes array of the image file in the file_extension format
    """
    with MemoryFile() as memfile:
        if file_extension == 'tiff':
            raster_image_array = numpy.append(raster_image.data, [raster_image.cloud_mask], axis=0)
            metadata = raster_image.metadata
            metadata.update(count=raster_image_array.shape[_FIRST_POSITION], dtype=raster_image_array.dtype)
            if raster_image.used_colormap == 'raw':
                metadata.update(nodata=0)
            with memfile.open(**metadata) as opened_memfile:
                opened_memfile.write(raster_image_array)
        else:
            with memfile.open(
                    driver=file_extension,
                    count=raster_image.data.shape[0],
                    height=raster_image.data.shape[1],
                    width=raster_image.data.shape[2],
                    dtype=raster_image.data.dtype,
                    nodata=0,
            ) as dst:
                dst.write(raster_image.data)
        return memfile.read()


def create_gif_image(images_list: List[RasterImage], index_name: str):
    """
    Creates a GIF from a list of RasterImages, Images with Index and ColorMap applied.
    """
    date_window = f'{images_list[_FIRST_POSITION].acquisition_date} -> {images_list[_LAST_POSITION].acquisition_date}'

    def load_image(image_obj: RasterImage):
        png_image = image_obj.data.transpose((1, 2, 0))
        img_figure = plt.imshow(png_image)
        plt.axis('off')
        return img_figure,

    fig = plt.figure()
    plt.title(f'{index_name.upper()} Images from {date_window}')
    ani = animation.FuncAnimation(fig, load_image, frames=images_list, blit=True, interval=400)

    # Save the animation as a GIF
    ani.save(f'./images/visualizer/{index_name}.gif', writer='imagemagick')


def save_file_locally(
        original_file: Union[numpy.ndarray, dict], folder_path: str, filename: str, file_type: str,
        metadata: Optional[Profile] = None
):
    """
    Writes locally a file (PNG, TIFF or JSON) to a given folder path locally

    :param original_file: File that will be saved locally. Can be a PNG, TIFF or JSON file
    :param folder_path: The path were the file will be stored in
    :param filename: The name the file
    :param file_type: The type choosen to save the file. Can be a PNG, TIFF or JSON file
    :param metadata: [Optional] Metadata of the original image to save TIFF files
    """
    writing_option = 'wb'
    if file_type in ['png', 'tiff'] and isinstance(original_file, numpy.ndarray):
        with MemoryFile() as memfile:
            if metadata and file_type == 'tiff':
                with memfile.open(**metadata) as opened_memfile:
                    opened_memfile.write(original_file)
            else:
                with memfile.open(
                        driver=file_type,
                        count=original_file.shape[0],
                        height=original_file.shape[1],
                        width=original_file.shape[2],
                        dtype=original_file.dtype,
                        nodata=0,
                ) as dst:
                    dst.write(original_file)
            file_obj = memfile.read()

    elif file_type == 'json' and isinstance(original_file, dict):
        file_obj = json.dumps(original_file)
        writing_option = 'w'

    else:
        print(f'Invalid file format! The file {filename} was not able to be saved.')
        return

    if os.path.exists(folder_path) is False:
        os.mkdir(folder_path)

    with open(f'{folder_path}/{filename}.{file_type}', writing_option) as outputfile:
        outputfile.write(file_obj)
