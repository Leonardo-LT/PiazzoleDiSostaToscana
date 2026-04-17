
class requestTileContext:
  def __init__(self, tileStrategy):
    self.tileStrategy = tileStrategy

  
  def setTileStrategy(self, tileStrategy):
    self.tileStrategy = tileStrategy


  def requestTileBytes(self, lat, long):
    return self.tileStrategy.requestTileBytes(lat, long)
  

  def requestAndSaveTile(self, lat, long, fileName, path):
    self.tileStrategy.requestAndSaveTile(lat, long, fileName, path)