from PIL import Image
from io import BytesIO

class RequestTileStrategy:
  #def setTileCoordinates(self, lat, long):
    #pass


  def requestTileBytes(self, lat, long):
    pass


  def requestAndSaveTile(self, lat, long, fileName, path="./"):
    bytes = self.requestTileBytes(lat, long)

    if bytes == None:
      print("Not an image")

    img = Image.open(BytesIO(bytes))
    img.save(f"{path}/{fileName}.png")
    #print(f"File: {fileName}")

    

    