import datetime
from collections import namedtuple
from typing import List

import numpy
from matplotlib import pyplot as plt

from models import Image, MaskedImage

ImageInfo = namedtuple('ImageInfo', ['mean', 'stdev', 'max', 'min'])


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
        list_images_info.append(ImageInfo(image_mean, image_stdev, image_max, image_min))

    return list_images_info


def plot_means_stdev_max_min(list_images_info: List[ImageInfo], list_dates: List[datetime.date]):
    list_means = [img_info.mean for img_info in list_images_info]
    list_stdevs = [img_info.stdev for img_info in list_images_info]
    list_maxs = [img_info.max for img_info in list_images_info]
    list_mins = [img_info.min for img_info in list_images_info]

    figsize = (len(list_dates) * 0.5, 6)

    plt.figure(figsize=figsize)
    plt.plot(list_dates, list_means, marker='o', label='Means Values')
    plt.plot(list_dates, list_stdevs, marker='o', label='Standart Deviation Values')
    plt.plot(list_dates, list_maxs, marker='o', label='Max Values')
    plt.plot(list_dates, list_mins, marker='o', label='Min Values')

    plt.xlabel('Images Dates')
    plt.ylabel('Images Informations Values')
    plt.legend()
    plt.title('Images Informations throught dates')
    plt.xticks(list_dates, rotation=45)
    plt.tight_layout()
    plt.show()


def calculate_afected_area(list_masked_images: List[MaskedImage]):
    FIRST_POSITION = 0
    diff_masks_list = []
    area_of_deforestation = numpy.zeros(list_masked_images[FIRST_POSITION].data.shape)
    area_of_recoveration = numpy.zeros(list_masked_images[FIRST_POSITION].data.shape)
    initial_deforestad_area = None
    initial_green_area = None
    for img_index, masked_image in enumerate(list_masked_images):
        if img_index != len(list_masked_images)-1:
            diff_masks_list.append(numpy.subtract(list_masked_images[img_index+1].data, list_masked_images[img_index].data))

        deforestation_image_area = masked_image.data.filled(0)
        deforestation_image_area[deforestation_image_area > 0.5] = 0
        deforestation_image_area[deforestation_image_area <= 0] = 0
        deforestation_image_area[deforestation_image_area != 0] = 1
        if img_index != 0:
            area_of_deforestation += deforestation_image_area
        else:
            initial_deforestad_area = deforestation_image_area

        recovered_image_area = masked_image.data.filled(0)
        recovered_image_area[recovered_image_area < 0.5] = 0
        recovered_image_area[recovered_image_area != 0] = 1
        if img_index != 0:
            area_of_recoveration += recovered_image_area
        else:
            initial_green_area = recovered_image_area

    area_of_deforestation[area_of_deforestation < len(list_masked_images)*0.15] = 0
    area_of_deforestation[initial_deforestad_area != 0] = 0
    plt.imshow(area_of_deforestation, cmap='magma')
    plt.title('Areas afected by deforestation')
    plt.show()

    area_of_recoveration[area_of_recoveration < len(list_masked_images)*0.15] = 0
    area_of_recoveration[initial_green_area != 0] = 0
    plt.imshow(area_of_recoveration, cmap='viridis')
    plt.title('Recovered Green Areas')
    plt.show()
