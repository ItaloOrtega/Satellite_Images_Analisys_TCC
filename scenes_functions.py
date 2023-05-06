import datetime
import json
from typing import List

import rasterio
import requests
from shapely import Polygon
from shapely.geometry import mapping, shape
from geom_functions import epsg_transform
from models import SceneInformation, SourceBands, SourceType

_FIRST_POSITION = 0
_FIRST_MASK_POSITION = 1
_WIDTH_POSITION = 1
_HEIGHT_POSITION = 0
_NODATA_VALUE = 0


def check_percentage_of_cloud_coverage(list_scenes: List[SceneInformation], max_cloud_percentage: float) -> bool:
    """
    Checks if the images have the mean percentage under the limit to be used.

    :param list_scenes: List of SceneInformation from the same day
    :param max_cloud_percentage: Max percentage of clouds of the scene
    :return: True, if the scenes are less-equal max_cloud_percentage. Otherwise returns False
    """
    percentages_of_cloud = [scene_information.cloud_coverage for scene_information in list_scenes]
    mean_cloud_percentage = sum(percentages_of_cloud)/len(list_scenes)

    if mean_cloud_percentage > max_cloud_percentage:
        return False
    print(f'Mean percentage of clouds: {mean_cloud_percentage}')
    return True


def old_get_simple_metadata_from_image(image_path: str):
    with rasterio.open(image_path) as dataset:
        transform_string = f'{dataset.transform.c}, {dataset.transform.a}, {dataset.transform.b}, {dataset.transform.f}, {dataset.transform.d}, {dataset.transform.e}'
        dataset.close()
    return transform_string


def get_transform_from_scene(scene_information: SceneInformation, source_bands: SourceBands):
    """
    Gets the transform from the original geometry of the Scene.

    :param scene_information: Scene data and information
    :param source_bands: Source Bands
    :return: Transform as string of the Scene
    """
    _geom_polygon = shape(epsg_transform(scene_information.geometry, 4326, source_bands.epsg))
    dataset_transform = rasterio.transform.from_bounds(*_geom_polygon.bounds, source_bands.width, source_bands.height)
    transform_string = f'{dataset_transform.c}, {dataset_transform.a}, {dataset_transform.b}, {dataset_transform.f}, {dataset_transform.d}, {dataset_transform.e}'

    return transform_string


def sentinel_scene_id_to_path(scene_id: str) -> str:
    """
    Creates part of the URL to get access to the Sentinel L2-A scene from its ID.

    :param scene_id: ID of the scene
    :return: Path of the scene in the URL
    """
    parts = scene_id.split('_')
    tile_utm = parts[-2]
    temp_date = parts[2].split("T")[0]
    u, t, m = tile_utm[1:3], tile_utm[3:4], tile_utm[4:6]
    if u.startswith('0'):
        u = u[1:]
    date = datetime.datetime.strptime(temp_date, '%Y%m%d')
    year, month = date.year, date.month
    return f'/{u}/{t}/{m}/{year}/{month}/{parts[0]}_{u}{t}{m}_{temp_date}_0_L2A'


def landsat_scene_id_to_path(scene_id: str) -> str:
    raise ValueError


def get_band_urls(scene_id: str, source_band: SourceType) -> List[str]:
    """
    Gets URL of ALL the wanted bands from the scene ID and the source.

    :param scene_id: Scene ID
    :param source_band: Source of the Scene
    :return: List of URLs of each band of the Scene
    """
    sentinel_bands_list = source_band.value.bands_sequence
    if source_band == SourceType.sentinel_l2a:
        tile_path = f'/vsis3/sentinel-cogs/sentinel-s2-l2a-cogs{sentinel_scene_id_to_path(scene_id)}'
    else:
        # Not Implemented
        tile_path = f'/vsis3/modis-061-cogs/landsat-c2-l2-cogs{landsat_scene_id_to_path(scene_id)}'
    return [f'{tile_path}/{band}.tif' for band in sentinel_bands_list]


def build_vrt_band(band_url: str, band_index: int, width, height):
    """
    Creates a band VRT from one scene's band.

    :param band_url: URL of the band
    :param band_index: Index of the band, position where the band is originally
    :param width: Band Width
    :param height: Band Height
    :return: VRT of one band, with URL, index and shape
    """
    return f'''
    <VRTRasterBand dataType="UInt16" band="{band_index}">
        <SimpleSource>
          <SourceFilename relativeToVRT="0">{band_url}</SourceFilename>
          <SourceBand>1</SourceBand>
          <SourceProperties RasterXSize="{width}" RasterYSize="{height}" DataType="UInt16" BlockXSize="1024" BlockYSize="1024" />
          <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
          <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
        </SimpleSource>
    </VRTRasterBand>
    '''


def build_vrt(scene_information: SceneInformation, source_bands: SourceType) -> SceneInformation:
    """
    Create VRT of the scenes, by using the ID of the Scene, Bands position in the Bucket and Scene metadata(height,
    width, epsg, etc).

    :param scene_information: Metadata of the Scene
    :param source_bands: Source of the Scene
    :return: Scene URL with the Bands
    """
    band_urls = get_band_urls(scene_information.scene_id, source_bands)
    transform_string = get_transform_from_scene(scene_information, source_bands.value)
    # transform_string = old_get_simple_metadata_from_image(band_urls[0])
    geo_transform = f'<GeoTransform> {transform_string} </GeoTransform>'
    srs = f'<SRS dataAxisToSRSAxisMapping="1,2">{source_bands.value.epsg}</SRS>'

    bands = '\n'.join(
        [build_vrt_band(band, index + 1, source_bands.value.width, source_bands.value.height) for index, band in
         enumerate(band_urls)])

    scene_information.url = f'''
    <VRTDataset rasterXSize="{source_bands.value.width}" rasterYSize="{source_bands.value.height}">
      {srs}
      {geo_transform}
      {bands}
    </VRTDataset>
    '''
    return scene_information


def get_scenes_urls(
        list_scenes: List[List[SceneInformation]], source_bands: SourceType
) -> List[List[SceneInformation]]:
    """
    Get Scenes URLs to be able to open in rasterio.open()

    :param list_scenes: List of SceneInformation from the same day
    :param source_bands: Source of the scenes
    :return: List of SceneInformation from the same day with the URLs add it
    """
    list_scenes_informations = []
    for scenes_from_same_day in list_scenes:
        same_day_scenes_informations = []
        for scene_information in scenes_from_same_day:
            same_day_scenes_informations.append(build_vrt(scene_information, source_bands))
        list_scenes_informations.append(same_day_scenes_informations)
    return list_scenes_informations


def get_scenes_ids_from_microsoft_planetary(
        start_date: datetime.date, end_date: datetime.date, area_geom: Polygon, scenes_source: str,
        max_cloud_coverage: float = 50.0, days_gap: int = 1
) -> List[List[SceneInformation]]:
    """
    Get scenes ID and information that intersect with the given geometry, puts Scenes from the same day in a List, and
    verifies if the Scenes have a mean percentage of clouds on them less-equal to the limit put by the user.

    :param start_date: Start date to get the images
    :param end_date: End date to get the images
    :param area_geom: Polygon geometry of the requested area
    :param scenes_source: Source of the scenes
    :param max_cloud_coverage: Max of percentage of cloud of the scene
    :param days_gap: Gap of days
    :return: List of SceneInformation from the same day
    """
    planetary_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    token_cursor = ""
    list_scene_ids = []
    last_scene_date = None
    while token_cursor is not None:
        payload = {
            "filter-lang": "cql2-json",
            "filter": {
                "op": "and",
                "args": [
                    {
                        "op": "=",
                        "args": [
                            {
                                "property": "collection"
                            },
                            scenes_source
                        ]
                    },
                    {
                        "op": "s_intersects",
                        "args": [
                            {
                                "property": "geometry"
                            },
                            mapping(area_geom)
                        ]
                    },
                    {
                        "op": "anyinteracts",
                        "args": [
                            {
                                "property": "datetime"
                            },
                            {
                                "interval": [
                                    start_date.strftime("%Y-%m-%d"),
                                    end_date.strftime("%Y-%m-%d")
                                ]
                            }
                        ]
                    }
                ]
            },
            "sortby": [
                {
                    "field": "datetime",
                    "direction": "desc"
                }
            ]
        }

        if token_cursor != "":
            payload["token"] = token_cursor

        headers = {
            'accept': 'application/json, text/plain, */*',
            'authority': 'planetarycomputer.microsoft.com',
            'content-type': 'application/json',
            'origin': 'https://planetarycomputer.microsoft.com'
        }

        response = requests.request("POST", planetary_url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            resp_json = response.json()
            list_scenes_from_same_day = []
            if len(resp_json['features']) == 0:
                print('There is no Scenes!')
                raise ValueError
            for scene_data in resp_json['features']:
                scene_date_acquisition = datetime.datetime.strptime(
                    scene_data['properties']['datetime'][:10], "%Y-%m-%d"
                ).date()

                if not last_scene_date:
                    last_scene_date = scene_date_acquisition
                    list_scenes_from_same_day.append(
                        SceneInformation(
                            scene_data['id'],
                            scene_data['geometry'],
                            scene_date_acquisition,
                            scene_data['properties']['eo:cloud_cover']
                        )
                    )
                else:  # Checks to see if the images are from the same day
                    if (last_scene_date - scene_date_acquisition).days == 0:
                        list_scenes_from_same_day.append(
                            SceneInformation(
                                scene_data['id'],
                                scene_data['geometry'],
                                scene_date_acquisition,
                                scene_data['properties']['eo:cloud_cover']
                            )
                        )
                    else:
                        # The list of scenes from the same day having data in this point,
                        # means that there is no more scenes from the same day.
                        if len(list_scenes_from_same_day) > 0:
                            if check_percentage_of_cloud_coverage(list_scenes_from_same_day, max_cloud_coverage):
                                list_scene_ids.append(list_scenes_from_same_day)
                            list_scenes_from_same_day = []
                        # Checks if the current Scene is bigger-equal to the number of days gap between images
                        elif (last_scene_date - scene_date_acquisition).days >= days_gap:
                            list_scenes_from_same_day.append(
                                SceneInformation(
                                    scene_data['id'],
                                    scene_data['geometry'],
                                    scene_date_acquisition,
                                    scene_data['properties']['eo:cloud_cover']
                                )
                            )
                            last_scene_date = scene_date_acquisition
            try:
                # Token to get subsequent images
                token_cursor = resp_json['links'][_FIRST_POSITION]['body']['token']
                if token_cursor is None or token_cursor[:4] != 'next':
                    token_cursor = None
            except (KeyError, IndexError):
                token_cursor = None
                print('There is no more scenes to be requested.')
        else:
            print('Error! Something wrong happened while handling the request')

    return list_scene_ids
