from shapefileLineIter import shapefileLineIter
from toscanaStrategy import ToscanaTileStrategy
from requestTileContext import requestTileContext
from tilesMetadataManager import tilesMetadataManager

from PIL import Image
from io import BytesIO
import os
import shapely
import datetime
from cnnInference import inferFromBytes as infer

tipiStrade = ["AA", "SS", "SR"]

def filter(el):
  if el is None:
    return False
  
  return el["tipostrada"].isin(tipiStrade)

sli = shapefileLineIter("/home/leon/Documenti/Tesi/iternet/iternet_c91b903823ae2a975c539cc65f880af9/iternet/shp/", "strade", 0.0008)
sli.setGdfCRS("EPSG:4326")
sli.setGdfFilter(filter)
shapeiter = iter(sli)

reqTileContext = requestTileContext(ToscanaTileStrategy(0.0008, 0.0008))
metadataManager = tilesMetadataManager("./tilesMetadata.csv")

def downloadTile(lat, long, path):
  bytes = reqTileContext.requestTileBytes(lat, long)
  if bytes == None:
    print("Not image")
    return 0

  try:
    img = Image.open(BytesIO(bytes))
    res = infer(img)

    if (res >= 0.75):
      img.save(path)

    img.close()
  except Exception as e:
    print(f"Error saving image: {e}")
    return 0

  return res

def getNextCoordinates():
  point = next(shapeiter)
  return point.y, point.x

def getAllTiles():
  i = 0 # contatore per salvare i metadati ogni tot file scaricati
  while(True):
    lat, long = getNextCoordinates()

    if (lat, long) == (None, None):
      print("Shapefile iteration completed")
      break

    res = downloadTile(lat, long, f"tiles/{lat}.{long}.png")
    if res >= 0.65:
      i+=1
      print("Piaz count: ", i)
      bboxPolygon = shapely.geometry.Polygon([(long-0.0008, lat-0.0008), (long+0.0008, lat-0.0008), (long+0.0008, lat+0.0008), (long-0.0008, lat+0.0008)])
      metadataManager.addTileMetadata(f"{lat}.{long}", bboxPolygon, None, None, None, datetime.datetime.now(), None, None)


try:
  #os.mkdir("tiles")
  getAllTiles()
except KeyboardInterrupt as e:
  print(f"Error occurred: {e}")
  metadataManager.saveToCSV()
