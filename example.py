from use_cases import get_images_with_index_from_middle_point

start_date_str = '2023-01-01'
end_date_str = '2023-05-03'

source = 'sentinel_l2a'
subproduct = 'ndvi'
color_map = 'contrast_original'

center_point_latitude = -49.44225758848097
center_point_longitude = -20.84224166518645

if __name__ == '__main__':
    get_images_with_index_from_middle_point(
        source=source,
        index=subproduct,
        color_map=color_map,
        point_latitude=center_point_latitude,
        point_longitude=center_point_longitude,
        start_date_str=start_date_str,
        end_date_str=end_date_str
    )
