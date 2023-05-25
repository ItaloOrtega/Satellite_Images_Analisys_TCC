from use_cases import get_images_with_index_from_middle_point

start_date_str = '2023-05-01'
end_date_str = '2023-05-24'

file_extension = 'png'
source = 'sentinel_l2a'
index = 'ndvi'
color_map = 'contrast_original'

center_point_latitude = -49.44225758848097
center_point_longitude = -20.84224166518645

max_cloud_coverage = 60.0

if __name__ == '__main__':
    get_images_with_index_from_middle_point(
        source=source,
        index=index,
        color_map=color_map,
        point_latitude=center_point_latitude,
        point_longitude=center_point_longitude,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        file_extension=file_extension,
        max_cloud_coverage=max_cloud_coverage
    )
