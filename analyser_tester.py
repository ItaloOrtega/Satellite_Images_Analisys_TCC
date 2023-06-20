import time
from copy import copy
from datetime import datetime

import numpy
import rasterio
import os

from geom_functions import create_a_square
from models import Image, SourceType, ColorMaps
from service import create_affected_area_information, save_affected_area_analisys_files, save_index_rgb_images

ndvi_images = []
rgb_images = []
for filename in os.listdir('images/tiff_images'):
    with rasterio.open(f'images/tiff_images/{filename}') as dataset:
        all_image_data = dataset.read()
        if 'NDVI' in filename:
            image_metadata = dataset.profile.copy()
            ndvi_data = numpy.asarray([all_image_data[0]])
            image_mask = all_image_data[1]
            cloud_mask = all_image_data[2]
            ndvi_images.append(
                Image(
                    data=ndvi_data,
                    mask=image_mask,
                    cloud_mask=cloud_mask,
                    acquisition_date=datetime.strptime(filename[24:34], '%Y_%m_%d').date(),
                    metadata=image_metadata,
                    id=filename[:-5],
                    source=SourceType.sentinel_l2a.value
                ))
        else:
            image_metadata = dataset.profile.copy()
            rgb_data = all_image_data[:3]
            image_mask = all_image_data[3]
            cloud_mask = all_image_data[4]
            rgb_images.append(
                Image(
                    data=rgb_data,
                    mask=image_mask,
                    cloud_mask=cloud_mask,
                    acquisition_date=datetime.strptime(filename[23:33], '%Y_%m_%d').date(),
                    metadata=image_metadata,
                    id=filename[:-5],
                    source=SourceType.sentinel_l2a.value
                ))
    dataset.close()

ndvi_images.sort(key=lambda x: x.acquisition_date)
rgb_images.sort(key=lambda x: x.acquisition_date)

center_point_latitude = -49.187900455436534
center_point_longitude = -22.623691718299597

max_distance_meters = 10000

original_geometry = create_a_square(max_distance_meters, center_point_latitude, center_point_longitude)

fig_means_stdev_max_min, fig_affected_area_graph, recoverage_affected_area_image, deforestation_affected_area_image, \
    fig_deforestation_area_graph, list_deforestation_diff_area_obj = create_affected_area_information(
        copy(ndvi_images[21:]), copy(rgb_images[0]), original_geometry, 256
)

save_affected_area_analisys_files(
    fig_means_stdev_max_min, fig_affected_area_graph, recoverage_affected_area_image, deforestation_affected_area_image,
    fig_deforestation_area_graph, list_deforestation_diff_area_obj
)

save_index_rgb_images(ndvi_images[21:], rgb_images[21:], ColorMaps.contrast_original.value)
