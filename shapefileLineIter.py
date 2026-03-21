# funzione shapefileReader
# implementa un generatore che restituisce le coordinate dei punti lungo le geometrie LineString o MultiLineString, con un certo step
import geopandas

class shapefileLineIter:
  def __init__(self, pathToShapefile, layer, stepSize):
    self.pathToShapefile = pathToShapefile
    self.layer = layer
    self.stepSize = stepSize
    self.gdf = geopandas.read_file(self.pathToShapefile, layer=self.layer)

  def setStepSize(self, newStepSize):
    self.stepSize = newStepSize
  
  def setGdfCRS(self, newCRS):
    self.gdf.to_crs(newCRS, inplace=True)

  def setGdfFilter(self, filterFunc):
    self.gdf = self.gdf[filterFunc(self.gdf)]

  def __iter__(self):
    for _, row in self.gdf.iterrows():
      currStep = 0
      geometry = row["geometry"]

      if geometry.geom_type == "LineString":
        while currStep < geometry.length:
          point = geometry.interpolate(currStep)
          yield point
          currStep += self.stepSize
    
      elif geometry.geom_type == "MultiLineString":
        for line in list(geometry.geoms):
          while currStep < line.length:
            point = line.interpolate(currStep)
            yield point
            currStep += self.stepSize

  

# modificare la classe in modo che permette di 
# ricominciare dall'ultima riga letta, 
# in modo da poter riprendere l'iterazione dopo una pausa 
# o un'interruzione del processo.
