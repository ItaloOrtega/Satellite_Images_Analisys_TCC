import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, NamedTuple

import numpy
from pyproj import CRS
from rasterio.profiles import Profile
from shapely import Polygon

ExtraBands = NamedTuple('ExtraBands', [('band_name', str), ('band_position', str)])
RGBColor = NamedTuple('RGBColor', [('red', int), ('green', int), ('blue', int)])


@dataclass
class RGBColorRepresentation:
    min: float
    max: float
    color: RGBColor


@dataclass
class ColorMap:
    type: str
    colors: List[RGBColorRepresentation]

    def get_rgb_color(self, original_value: float):
        for color_representation in self.colors:
            if color_representation.min <= original_value <= color_representation.max:
                return color_representation.color

    def apply_color_map(self, image_data: numpy.array):
        _RED_POSITION = 0
        _GREEN_POSITION = 1
        _BLUE_POSITION = 2
        _MATRICES_INDEX = 0
        _HEIGHT_INDEX = 1
        _WIDTH_INDEX = 2

        if self.type == 'raw':
            return image_data

        if self.type == 'truecolor':
            return image_data.astype('uint8')

        image_with_color_map = numpy.zeros(
            shape=(3, image_data.shape[_HEIGHT_INDEX], image_data.shape[_WIDTH_INDEX]), dtype='uint8'
        )

        for index_line in range(image_data.shape[_HEIGHT_INDEX]):
            for index_column in range(image_data.shape[_WIDTH_INDEX]):
                rgb_values = self.get_rgb_color(image_data[_MATRICES_INDEX][index_line][index_column])
                image_with_color_map[_RED_POSITION][index_line][index_column] = rgb_values[_RED_POSITION]
                image_with_color_map[_GREEN_POSITION][index_line][index_column] = rgb_values[_GREEN_POSITION]
                image_with_color_map[_BLUE_POSITION][index_line][index_column] = rgb_values[_BLUE_POSITION]

        return image_with_color_map


class ColorMaps(Enum):
    truecolor = ColorMap(type='truecolor', colors=[])
    raw = ColorMap(type='raw', colors=[])
    contrast_original = ColorMap(
        type='contrast_original',
        colors=[
            RGBColorRepresentation(min=-1, max=0, color=RGBColor(red=128, green=128, blue=128)),
            RGBColorRepresentation(min=0.00, max=0.05, color=RGBColor(red=255, green=0, blue=0)),
            RGBColorRepresentation(min=0.05, max=0.10, color=RGBColor(red=248, green=26, blue=0)),
            RGBColorRepresentation(min=0.10, max=0.15, color=RGBColor(red=242, green=51, blue=0)),
            RGBColorRepresentation(min=0.15, max=0.20, color=RGBColor(red=235, green=74, blue=0)),
            RGBColorRepresentation(min=0.20, max=0.25, color=RGBColor(red=228, green=96, blue=0)),
            RGBColorRepresentation(min=0.25, max=0.30, color=RGBColor(red=222, green=117, blue=0)),
            RGBColorRepresentation(min=0.30, max=0.35, color=RGBColor(red=215, green=136, blue=0)),
            RGBColorRepresentation(min=0.35, max=0.40, color=RGBColor(red=208, green=153, blue=0)),
            RGBColorRepresentation(min=0.40, max=0.45, color=RGBColor(red=202, green=170, blue=0)),
            RGBColorRepresentation(min=0.45, max=0.50, color=RGBColor(red=195, green=185, blue=0)),
            RGBColorRepresentation(min=0.50, max=0.55, color=RGBColor(red=178, green=188, blue=0)),
            RGBColorRepresentation(min=0.55, max=0.60, color=RGBColor(red=153, green=181, blue=0)),
            RGBColorRepresentation(min=0.60, max=0.65, color=RGBColor(red=129, green=175, blue=0)),
            RGBColorRepresentation(min=0.65, max=0.70, color=RGBColor(red=106, green=168, blue=0)),
            RGBColorRepresentation(min=0.70, max=0.75, color=RGBColor(red=85, green=161, blue=0)),
            RGBColorRepresentation(min=0.75, max=0.80, color=RGBColor(red=65, green=155, blue=0)),
            RGBColorRepresentation(min=0.80, max=0.85, color=RGBColor(red=47, green=148, blue=0)),
            RGBColorRepresentation(min=0.85, max=0.90, color=RGBColor(red=30, green=141, blue=0)),
            RGBColorRepresentation(min=0.90, max=0.95, color=RGBColor(red=14, green=135, blue=0)),
            RGBColorRepresentation(min=0.95, max=1.00, color=RGBColor(red=0, green=128, blue=0)),
        ],
    )
    contrast_heat = ColorMap(
        type='contrast_heat',
        colors=[
            RGBColorRepresentation(min=-1, max=0, color=RGBColor(red=128, green=128, blue=128)),
            RGBColorRepresentation(min=0.00, max=0.05, color=RGBColor(red=128, green=0, blue=38)),
            RGBColorRepresentation(min=0.05, max=0.10, color=RGBColor(red=153, green=0, blue=38)),
            RGBColorRepresentation(min=0.10, max=0.15, color=RGBColor(red=179, green=0, blue=38)),
            RGBColorRepresentation(min=0.15, max=0.20, color=RGBColor(red=199, green=6, blue=35)),
            RGBColorRepresentation(min=0.20, max=0.25, color=RGBColor(red=215, green=17, blue=31)),
            RGBColorRepresentation(min=0.25, max=0.30, color=RGBColor(red=229, green=31, blue=29)),
            RGBColorRepresentation(min=0.30, max=0.35, color=RGBColor(red=240, green=53, blue=35)),
            RGBColorRepresentation(min=0.35, max=0.40, color=RGBColor(red=250, green=75, blue=41)),
            RGBColorRepresentation(min=0.40, max=0.45, color=RGBColor(red=252, green=101, blue=48)),
            RGBColorRepresentation(min=0.45, max=0.50, color=RGBColor(red=252, green=127, blue=56)),
            RGBColorRepresentation(min=0.50, max=0.55, color=RGBColor(red=253, green=148, blue=63)),
            RGBColorRepresentation(min=0.55, max=0.60, color=RGBColor(red=253, green=164, blue=70)),
            RGBColorRepresentation(min=0.60, max=0.65, color=RGBColor(red=254, green=180, blue=78)),
            RGBColorRepresentation(min=0.65, max=0.70, color=RGBColor(red=254, green=196, blue=95)),
            RGBColorRepresentation(min=0.70, max=0.75, color=RGBColor(red=254, green=212, blue=113)),
            RGBColorRepresentation(min=0.75, max=0.80, color=RGBColor(red=254, green=223, blue=131)),
            RGBColorRepresentation(min=0.80, max=0.85, color=RGBColor(red=254, green=231, blue=148)),
            RGBColorRepresentation(min=0.85, max=0.90, color=RGBColor(red=255, green=239, blue=166)),
            RGBColorRepresentation(min=0.90, max=0.95, color=RGBColor(red=255, green=247, blue=185)),
            RGBColorRepresentation(min=0.95, max=1.00, color=RGBColor(red=255, green=255, blue=204)),

        ],
    )
    contrast_viridis = ColorMap(
        type='contrast_viridis',
        colors=[
            RGBColorRepresentation(min=-1, max=0, color=RGBColor(red=128, green=128, blue=128)),
            RGBColorRepresentation(min=0.00, max=0.05, color=RGBColor(red=253, green=231, blue=36)),
            RGBColorRepresentation(min=0.05, max=0.10, color=RGBColor(red=220, green=226, blue=24)),
            RGBColorRepresentation(min=0.10, max=0.15, color=RGBColor(red=186, green=222, blue=39)),
            RGBColorRepresentation(min=0.15, max=0.20, color=RGBColor(red=149, green=215, blue=63)),
            RGBColorRepresentation(min=0.20, max=0.25, color=RGBColor(red=116, green=208, blue=84)),
            RGBColorRepresentation(min=0.25, max=0.30, color=RGBColor(red=85, green=198, blue=102)),
            RGBColorRepresentation(min=0.30, max=0.35, color=RGBColor(red=61, green=187, blue=116)),
            RGBColorRepresentation(min=0.35, max=0.40, color=RGBColor(red=41, green=175, blue=127)),
            RGBColorRepresentation(min=0.40, max=0.45, color=RGBColor(red=31, green=163, blue=134)),
            RGBColorRepresentation(min=0.45, max=0.50, color=RGBColor(red=31, green=150, blue=139)),
            RGBColorRepresentation(min=0.50, max=0.55, color=RGBColor(red=34, green=137, blue=141)),
            RGBColorRepresentation(min=0.55, max=0.60, color=RGBColor(red=39, green=124, blue=142)),
            RGBColorRepresentation(min=0.60, max=0.65, color=RGBColor(red=44, green=112, blue=142)),
            RGBColorRepresentation(min=0.65, max=0.70, color=RGBColor(red=50, green=98, blue=141)),
            RGBColorRepresentation(min=0.70, max=0.75, color=RGBColor(red=57, green=85, blue=139)),
            RGBColorRepresentation(min=0.75, max=0.80, color=RGBColor(red=63, green=69, blue=135)),
            RGBColorRepresentation(min=0.80, max=0.85, color=RGBColor(red=69, green=54, blue=129)),
            RGBColorRepresentation(min=0.85, max=0.90, color=RGBColor(red=71, green=37, blue=117)),
            RGBColorRepresentation(min=0.90, max=0.95, color=RGBColor(red=71, green=20, blue=102)),
            RGBColorRepresentation(min=0.95, max=1.00, color=RGBColor(red=68, green=1, blue=84)),

        ],
    )
    contrast_magma = ColorMap(
        type='contrast_magma',
        colors=[
            RGBColorRepresentation(min=-1, max=0, color=RGBColor(red=128, green=128, blue=128)),
            RGBColorRepresentation(min=0.00, max=0.05, color=RGBColor(red=251, green=252, blue=191)),
            RGBColorRepresentation(min=0.05, max=0.10, color=RGBColor(red=252, green=229, blue=166)),
            RGBColorRepresentation(min=0.10, max=0.15, color=RGBColor(red=253, green=205, blue=144)),
            RGBColorRepresentation(min=0.15, max=0.20, color=RGBColor(red=254, green=179, blue=123)),
            RGBColorRepresentation(min=0.20, max=0.25, color=RGBColor(red=253, green=155, blue=106)),
            RGBColorRepresentation(min=0.25, max=0.30, color=RGBColor(red=250, green=128, blue=94)),
            RGBColorRepresentation(min=0.30, max=0.35, color=RGBColor(red=244, green=104, blue=91)),
            RGBColorRepresentation(min=0.35, max=0.40, color=RGBColor(red=231, green=82, blue=98)),
            RGBColorRepresentation(min=0.40, max=0.45, color=RGBColor(red=214, green=68, blue=108)),
            RGBColorRepresentation(min=0.45, max=0.50, color=RGBColor(red=192, green=58, blue=117)),
            RGBColorRepresentation(min=0.50, max=0.55, color=RGBColor(red=171, green=51, blue=124)),
            RGBColorRepresentation(min=0.55, max=0.60, color=RGBColor(red=148, green=43, blue=128)),
            RGBColorRepresentation(min=0.60, max=0.65, color=RGBColor(red=127, green=36, blue=129)),
            RGBColorRepresentation(min=0.65, max=0.70, color=RGBColor(red=105, green=28, blue=128)),
            RGBColorRepresentation(min=0.70, max=0.75, color=RGBColor(red=85, green=19, blue=125)),
            RGBColorRepresentation(min=0.75, max=0.80, color=RGBColor(red=62, green=15, blue=114)),
            RGBColorRepresentation(min=0.80, max=0.85, color=RGBColor(red=40, green=17, blue=89)),
            RGBColorRepresentation(min=0.85, max=0.90, color=RGBColor(red=21, green=14, blue=56)),
            RGBColorRepresentation(min=0.90, max=0.95, color=RGBColor(red=7, green=5, blue=27)),
            RGBColorRepresentation(min=0.95, max=1.00, color=RGBColor(red=0, green=0, blue=3)),

        ],
    )


_SINGLE_BAND_LIST_OF_COLOR_MAP = ['contrast_original', 'contrast_heat', 'contrast_viridis', 'contrast_magma', 'raw']

_TRIPLE_BAND_LIST_OF_COLOR_MAP = ['truecolor', 'raw']


class IndexType(Enum):
    NDVI = 'ndvi'
    SAVI = 'savi'
    RGB = 'rgb'
    EVI = 'evi'
    NDWI = 'ndwi'
    NDVIG = 'ndvig'
    RVI = 'rvi'
    ARVI = 'arvi'
    NIR = 'nir'
    VARI = 'vari'


@dataclass
class SceneInformation:
    scene_id: str
    geometry: dict
    acquisition_date: datetime.date
    cloud_coverage: float
    url: Optional[str] = None

    def get_bounds_from_geometry(self):
        _geom_polygon = Polygon(self.geometry)
        return _geom_polygon.bounds


@dataclass
class SourceBands:
    source_name: str
    blue: str
    green: str
    red: str
    nir: str
    cloud_mask: str
    c1: float
    c2: float
    epsg: CRS
    width: int
    height: int
    mask_width: int
    mask_height: int
    bands_sequence: List[str]
    extra_bands: Optional[List[ExtraBands]] = None
    band_interp_values: Optional[dict] = None


class SourceType(Enum):
    sentinel_l2a = SourceBands(
        source_name='sentinel-2-l2a',
        blue='b3',
        green='b2',
        red='b1',
        nir='b4',
        cloud_mask='b5',
        c1=6,
        c2=7.5,
        epsg=CRS(32722),
        width=10980,
        height=10980,
        mask_width=5490,
        mask_height=5490,
        bands_sequence=['B04', 'B03', 'B02', 'B08', 'SCL'],
        band_interp_values={'red': [0, 4000], 'green': [0, 4000], 'blue': [0, 4000]},
    )

    landsat = SourceBands(
        source_name='landsat-c2-l2',
        blue='b3',
        green='b2',
        red='b1',
        nir='b4',
        cloud_mask='b5',
        c1=6,
        c2=7.5,
        epsg=CRS(32722),
        width=10980,
        height=10980,
        mask_width=5490,
        mask_height=5490,
        bands_sequence=['B03', 'B13', 'B04', 'B14', 'B08', 'B09'],
        extra_bands=[ExtraBands('thermal_rad', 'b5'), ExtraBands('surface_temp', 'b6')],
        band_interp_values={'red': [5140, 20560], 'green': [5140, 20560], 'blue': [5140, 20560]},
    )


@dataclass
class Image:
    data: numpy.array
    mask: numpy.array
    cloud_mask: numpy.array
    metadata: Profile
    id: str
    source: SourceBands

    def create_bands_vars(self):
        """
        Creates a var dict representing each of the bands in the numpy.array as the represented color
        :return:
        """
        _BAND_POSITION = 1
        band_keys = ['red', 'green', 'blue', 'nir', 'cloud_mask', 'c1', 'c2']
        source_dict = self.source.__dict__
        band_vars = {}
        for key in band_keys:
            if key in ('c1', 'c2'):
                value = source_dict[key]
            else:
                value = self.data[int(source_dict[key][_BAND_POSITION])-1]
            band_vars[key] = value
        if self.source.extra_bands:
            for extra_bands in self.source.extra_bands:
                band_vars[extra_bands.band_name] = self.data[int(extra_bands.band_position[_BAND_POSITION])]

        return band_vars


@dataclass
class Index:
    name: str
    expression: str
    color_maps: List[str]
    ceiling_value: Optional[int] = None


class IndexExpressionType(Enum):
    ndvi = Index(
        name='ndvi',
        expression='(nir-red)/(nir+red)',
        color_maps=_SINGLE_BAND_LIST_OF_COLOR_MAP,
        ceiling_value=-1,
    )
    savi = Index(
        name='savi',
        expression='1.5*(nir-red)/(nir+red+0.5)',
        color_maps=_SINGLE_BAND_LIST_OF_COLOR_MAP,
        ceiling_value=1,
    )
    rgb = Index(
        name='rgb',
        expression='red, green, blue',
        color_maps=_TRIPLE_BAND_LIST_OF_COLOR_MAP,
    )
    evi = Index(
        name='evi',
        expression='2.5*(nir-red)/(nir+c1*red-c2*blue+10000)',
        color_maps=_SINGLE_BAND_LIST_OF_COLOR_MAP,
        ceiling_value=1,
    )
    ndwi = Index(
        name='ndwi',
        expression='(green-nir)/(nir+green)',
        color_maps=_SINGLE_BAND_LIST_OF_COLOR_MAP,
        ceiling_value=-1,
    )
    ndvig = Index(
        name='ndvig',
        expression='(nir-green)/(nir+green)',
        color_maps=_SINGLE_BAND_LIST_OF_COLOR_MAP,
        ceiling_value=-1,
    )
    rvi = Index(
        name='rvi',
        expression='nir/red',
        color_maps=_SINGLE_BAND_LIST_OF_COLOR_MAP,
        ceiling_value=1,
    )
    arvi = Index(
        name='arvi',
        expression='(nir-(red-blue+red))/(nir+(red-blue+red))',
        color_maps=_SINGLE_BAND_LIST_OF_COLOR_MAP,
        ceiling_value=1,
    )
    nir = Index(
        name='nir',
        expression='nir, blue, green',
        color_maps=_TRIPLE_BAND_LIST_OF_COLOR_MAP,
    )
    vari = Index(
        name='vari',
        expression='(green-red)/(green+red-blue)',
        color_maps=_SINGLE_BAND_LIST_OF_COLOR_MAP,
        ceiling_value=1,
    )
    raw = Index(
        name='raw',
        expression='red, blue, green, nir',
        color_maps=['raw'],
    )
