from shapely.geometry import Point


def findRoadFromPoint(gdf, point):
    distances = gdf["geometry"].distance(point)
    nearest_idx = distances.idxmin()
    return gdf.loc[nearest_idx]["den_estesa"]
