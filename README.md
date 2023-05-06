# Italo's TCC

## Apresentation

A project that aims to generate satellite images with index, such as NDVI (Normalized Difference Vegetation Index), for a specific area within a time window to facilitate academic analysis and data retrieval.


## Word Summary

### Scenes
Scenes are images of large areas of the Earth's surface captured by satellites. They are typically saved in .TIFF format, which is a geolocated format.

### Bands
Bands are numeric matrices of commonly used values, such as Blue and Infrared (NIR).

### Index
Index refers to mathematical expressions performed using image bands to generate matrices that provide important information, such as NDVI (Normalized Difference Vegetation Index) which indicates the health of plantations.

### Images
Images are representations of band matrices after applying a Index and using a color map for visualization.

### Geometry
A geolocated area Polygon represents a specific geographic region defined by a set of coordinates. It is commonly used to describe the boundaries of an area on the Earth's surface.

### Color Map
Visual representation of image values through gradual colors.


## Set up environment

```bash
pip install -r requirements.txt
```

## Example code

Run `main` inside the `example.py`

```bash
python3 example.py  
```

### Indices 

| Index Name | Description                                                                                                                                                                                                       | Color Maps Available                                                      |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| **RGB**    | The representation of the true colors of the image, just for visualization.                                                                                                                                       | [Truecolor, Raw]                                                          |
| **NIR**    | Near-Infrared is a Index composed of Infra-Red, Blue, and Green bands. It is used to detect vegetation cover and monitor plant health.                                                                            | [Truecolor, Raw]                                                          |
| **NDVI**   | Normalized Difference Vegetation Index is a commonly used vegetation index that measures the difference between the reflectance of near-infrared and red light to assess vegetation health and density.           | [Contrast Original, Contrast Heat, Contrast Viridis, Contrast Magma, Raw] |
| **SAVI**   | Soil-Adjusted Vegetation Index is a vegetation index that corrects for the effects of soil brightness on NDVI, making it more suitable for areas with exposed soil or sparse vegetation.                          | [Contrast Original, Contrast Heat, Contrast Viridis, Contrast Magma, Raw] |
| **EVI**    | Enhanced Vegetation Index is another vegetation index that accounts for atmospheric interference and is more sensitive to changes in vegetation density than NDVI.                                                | [Contrast Original, Contrast Heat, Contrast Viridis, Contrast Magma, Raw] |
| **NDWI**   | Normalized Difference Water Index is a vegetation index that uses the difference in reflectance between near-infrared and shortwave infrared light to detect water content in vegetation.                         | [Contrast Original, Contrast Heat, Contrast Viridis, Contrast Magma, Raw] |
| **NDVIG**  | Normalized Difference Vegetation Index Green is a modification of NDVI that uses the green band instead of the red band to reduce soil background noise.                                                          | [Contrast Original, Contrast Heat, Contrast Viridis, Contrast Magma, Raw] |
| **RVI**    | Ratio Vegetation Index is a vegetation index that uses the ratio between red and near-infrared bands to estimate vegetation cover and productivity.                                                               | [Contrast Original, Contrast Heat, Contrast Viridis, Contrast Magma, Raw] |
| **ARVI**   | Atmospherically Resistant Vegetation Index is a vegetation index that minimizes the influence of atmospheric scattering and absorption on NDVI.                                                                   | [Contrast Original, Contrast Heat, Contrast Viridis, Contrast Magma, Raw] |
| **VARI**   | Visible Atmospherically Resistant Index is a vegetation index that uses the difference between blue and red bands to estimate vegetation cover, while being more resistant to atmospheric interference than NDVI. | [Contrast Original, Contrast Heat, Contrast Viridis, Contrast Magma, Raw] |
