from shapefileLineIter import shapefileLineIter
from toscanaStrategy import ToscanaTileStrategy
from requestTileContext import requestTileContext
from tilesMetadataManager import tilesMetadataManager

from PIL import Image
from io import BytesIO
import os
import shapely
import datetime

tipiStrade = ["AA", "SS", "SR"]

def filter(el):
  if el is None:
    print("Elemento None")
    return None
  
  return el["tipostrada"].isin(tipiStrade)

sli = shapefileLineIter("/home/leon/Documenti/Tesi/iternet/iternet_c91b903823ae2a975c539cc65f880af9/iternet/shp/", "strade", 0.0008)
sli.setGdfCRS("EPSG:4326")
sli.setGdfFilter(filter)
shapeiter = iter(sli)

reqTileContext = requestTileContext(ToscanaTileStrategy(0.0008, 0.0008))
metadataManager = tilesMetadataManager("./tilesMetadata.geojson")
print(metadataManager.gdf.head()["tileBBOX"])
exit()

print("eccoic")
def downloadTile(lat, long, path):
  bytes = reqTileContext.requestTileBytes(lat, long)

  try:
    img = Image.open(BytesIO(bytes))
    img.save(path)
    img.close()
  except Exception as e:
    print(f"Error saving image: {e}")
    return False

  return True

def getNextCoordinates():
  point = next(shapeiter)
  return point.y, point.x

def getAllTiles():
  print("dentro")
  i = 0
  while(True):
    lat, long = getNextCoordinates()

    if (lat, long) == (None, None):
      print("Shapefile iteration completed")
      break

    res = downloadTile(lat, long, f"tiles/{lat}.{long}")
    if res:
      i+=1
      bboxPolygon = shapely.geometry.Polygon([(long-0.0008, lat-0.0008), (long+0.0008, lat-0.0008), (long+0.0008, lat+0.0008), (long-0.0008, lat+0.0008)])
      metadataManager.addTileMetadata(f"{lat}.{long}", bboxPolygon, None, None, None, datetime.datetime.now(), None, None)


try:
  #os.mkdir("tiles")
  print("eccolo")
  getAllTiles()
except KeyboardInterrupt as e:
  print(f"Error occurred: {e}")
  metadataManager.saveToCSV()