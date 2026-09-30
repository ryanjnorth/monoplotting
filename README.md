# Historical Photo Monoplotting
Simple guided workflow for georectifying line features within single historical photos ('monoplotting') (via Python, Jupyter Notebook).

Download both the Jupyter Notebook (.ipynb) file which explains and walks through the process of monoplotting as well as the Python 'engine' script (.py) that contains the functions the drive the Jupyter Notebook.
The Python file needs to be saved in the same directory as the Jupyter notebook. The Python file also contains detailed comments so that anyone with a basic understanding of code should be able to follow it. 

Built by Dr Ryan North [rnorth.academic @ gmail . com] and Elena Disilvestro, Monash University, with assistance from Gemini 3.1 Pro and Claude Opus 4.8. 

## Input Data
| File | Format | Notes |
|---|---|---|
| Oblique photo | JPG / PNG / TIFF | Colour or greyscale |
| DEM | GeoTIFF | **Projected CRS in metres** (e.g. UTM). Must cover everything visible in the photo |
| GCP table | CSV | `GCP, row, col, X, Y, Z` — at least 6 points, 8+ recommended |
| Line features | CSV | `feature_name, col, row` — only needed for Part 2 |
| Custom exclusion mask | Image | *Optional*. Image-space mask for lens flare, dust, aircraft struts… |
| Stable terrain mask | GeoTIFF or shapefile | *Optional*. Only for the warp-fill in Cell 15 |

## Requirements
Python packages:
> pip install numpy opencv-python rasterio open3d matplotlib scipy geopandas shapely

*Note:* Python 3.12 is required for Open3D to work. 

