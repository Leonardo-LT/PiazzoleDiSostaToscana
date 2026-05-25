import geopandas as gpd
import pandas as pd
from evalPerRoad import findRoadFromPoint
from evalPerZone import pointToZone

df = pd.read_csv("results.csv")

import geopandas as gpd
import shapely.geometry as geom

extracted = df["tileId"].str.extract(r"(\d+\.\d+)\.(\d+\.\d+)\.png")

extracted.columns = ["lat", "lon"]
extracted = extracted.astype(float)

gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(extracted["lon"], extracted["lat"]), crs="EPSG:4326"
)

TILE_SIZE_IN_SQM = 1430.0
PIXELS_NUM = 224**2
SQM_PER_PIXEL = (1 / PIXELS_NUM) * TILE_SIZE_IN_SQM

zones = {"0.0": "NW", "0.1": "NE", "1.0": "SW", "1.1": "SE"}


def toZone(coords):
    res = pointToZone(
        coords.x,
        coords.y,
        [9.6867692, 42.2376150, 12.3722747, 44.4725419],
        2,
        2,
    )

    if res is None:
        return "None"

    row, col = res

    return zones[f"{row}.{col}"]


gdf["group"] = gdf["geometry"].apply(lambda x: toZone(x))
gdf["extArea"] = gdf["extArea"] * SQM_PER_PIXEL
gdf["gtArea"] = gdf["gtArea"] * SQM_PER_PIXEL
gdf["areaErr"] = (gdf["extArea"] - gdf["gtArea"]).abs()

grouped = gdf.groupby("group")

print(grouped["IoU"].mean())
print(grouped["areaErr"].mean())
print(grouped["areaErr"].std())

shp = gpd.read_file(
    "/home/leon/Documenti/Tesi/iternet/iternet_c91b903823ae2a975c539cc65f880af9/iternet/shp/",
    layer="strade",
)

shp.to_crs("EPSG:4326", inplace=True)

gdf["group"] = gdf["geometry"].apply(lambda x: findRoadFromPoint(shp, x))

grouped = gdf.groupby("group")

print(grouped["IoU"].mean())
print(grouped["areaErr"].mean())
print(grouped["areaErr"].std())
