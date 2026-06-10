
class requestTileContext:
  def __init__(self, tileStrategy):
    self.tileStrategy = tileStrategy

  
  def setTileStrategy(self, tileStrategy):
    self.tileStrategy = tileStrategy


  def requestTileBytes(self, lat, long):
    return self.tileStrategy.requestTileBytes(lat, long)


  def requestCustomBBoxBytes(self, min_lat, min_lon, max_lat, max_lon, width_px, height_px):
    return self.tileStrategy.requestCustomBBoxBytes(min_lat, min_lon, max_lat, max_lon, width_px, height_px)
  

  def requestAndSaveTile(self, lat, long, fileName, path):
    self.tileStrategy.requestAndSaveTile(lat, long, fileName, path)