import geopandas # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
import skimage as ski
from PIL import Image
from io import BytesIO
import numpy as np

from toscanaStrategy import ToscanaTileStrategy
from requestTileContext import requestTileContext

context = requestTileContext(ToscanaTileStrategy(0.0008, 0.0008))

def downloadTile(lat, long, fileName):
  print(fileName, "\n")
  return context.requestTileBytes(lat, long)


def iterateLine(line, n, name):
  name = "_".join(name.split(" "))
  #print("lunghezza: ", line.length, "\n")
  step =  0.0008 
  currStep = 0
  i = 0

  #sum = 0

  try:
    os.mkdir(f"Immagini/{name}")
  except Exception as e:
    print(f"Errore: {e}")

  while True:
    point = geometry.line_interpolate_point(currStep)
    bytes = downloadTile(point.y, point.x, f"{name}/{n}.{i}")
    i+=1
    if bytes != None:
      img = Image.open(BytesIO(bytes))
      imgArray = np.array(img)
      # is_grey = ((imgArray >= 110) & (imgArray <= 160)).any(axis=2)
      
      # count = np.count_nonzero(is_grey)
      # sum += count

      img.save(f"./Immagini/{name}/{n}.{i}.png")
      # print("Count: ", count)

    currStep+=step
    if currStep >= line.length: 
      print("lunghezza: ", line.length)
      break
  
  #return sum

pathToShapefiles = "/home/leon/Documenti/Tesi/iternet/iternet_c91b903823ae2a975c539cc65f880af9/iternet/shp/"

gdf = geopandas.read_file(pathToShapefiles, layer="strade")

#grouped_df = gdf.groupby(["tipostrada"])

#for key, item in grouped_df:
  #print(key)
  #print(grouped_df.get_group(key), "\n\n")

tipistrada = ["AA", "SS", "SR"] #, "SP", "SC", "SV"

# dug_piazzole = [
#   "AUTOSTRADA", "SUPERSTRADA", "TANGENZIALE", "S.G.C.", 
#   "S.S.", "EX S.S.", "S.R.", "EX S.R.", "S.P.", "EX S.P.",
  
#   #"GALLERIA", "VIADOTTO", "PONTE", "CAVALCAVIA", "RACCORDO", 
#   #"RACCORDO AUTOSTRADALE", "BRETELLA", "BRETELLA DI COLLEGAMENTO", 
#   #"SVINCOLO", "COMPLANARE", "BYPASS", "VARIANTE",
  
#   #"STRADA", "VIA", "CIRCONVALLAZIONE", "ASSE STRADALE", 
#   "STRADA COMUNALE", "S.C.", "EX S.C."
# ]

gdf = gdf[(gdf["tipostrada"].isin(tipistrada))]

# gdf.plot()
# plt.show()

gdf.to_crs("EPSG:4326", inplace=True)

#iter = gdf.iterfeatures()
n = 0

greySum = 0

for i, row in gdf.iterrows():
  #print(n, "\n")
  #print(row["tipostrada"])
  #print(row)
  geometry = row["geometry"]
  #greySum += iterateLine(geometry, n, row["indirizzo"])
  iterateLine(geometry, n, row["indirizzo"])
  n+=1

  # la media su circa 1000 tiles è di circa 80000 pixel grigi
  # average = greySum / (n*10)
  # print("Average: ", average)
  # print("Sum: ", greySum, "\n")
  
  #if (n >= 20): break
  


#print(geopandas.GeoSeries(gdf.iloc[1000]["geometry"]).get_coordinates())

