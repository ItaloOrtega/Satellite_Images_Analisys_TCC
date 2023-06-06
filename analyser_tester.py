from datetime import datetime

import rasterio
import os

from analyser import create_masked_array_dataset, calculate_means_stdev_max_min, plot_means_stdev_max_min
from models import Image, SourceType

ndvi_images = []
rgb_images = []
for filename in os.listdir('tiff_images'):
    with rasterio.open(f'tiff_images/{filename}') as dataset:
        all_image_data = dataset.read()
        if 'NDVI' in filename:
            image_metadata = dataset.profile.copy()
            ndvi_data = all_image_data[0]
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

masked_ndvi_images = []
for ndvi_img in ndvi_images:
    masked_ndvi_images.append(create_masked_array_dataset(ndvi_img))

ndvi_images_infos = calculate_means_stdev_max_min(masked_ndvi_images)
dates_images = [img.acquisition_date for img in masked_ndvi_images]

plot_means_stdev_max_min(ndvi_images_infos, dates_images)

print(1)
