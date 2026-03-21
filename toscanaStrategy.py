import requests
from requestTileStrategy import RequestTileStrategy

class ToscanaTileStrategy(RequestTileStrategy):
  def __init__(self, tileHeight, tileWidth):
    self.wmsUrl = "https://www502.regione.toscana.it/ows_ofc/com.rt.wms.RTmap/wms"

    self.params = {
      "map": "owsofc_rt",
      "SERVICE": "WMS",
      "VERSION": "1.3.0",
      "REQUEST": "GetMap",
      "LAYERS": "rt_ofc.5k24.32bit",
      "CRS": "EPSG:4326",
      "BBOX": None,
      "WIDTH": "224",
      "HEIGHT": "224",
      "FORMAT": "image/jpeg",
    }

    self.tileHeight, self.tileWidth = tileHeight, tileWidth

  def setBBOX(self, lat, long):
    min_lat = lat - self.tileHeight
    max_lat = lat + self.tileHeight
    min_lon = long -self.tileWidth
    max_lon = long + self.tileWidth

    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    self.params["BBOX"] = bbox


  def setTileHeight(self, height):
    self.tileHeight = height


  def setTileWidth(self, width):
    self.tileWidth = width


  def requestTileBytes(self, lat, long):
    self.setBBOX(lat, long)

    try:
        print(f"Coordinate: {lat}, {long}...")
        
        response = requests.get(self.wmsUrl, params=self.params, timeout=5)
        #print(response.headers, "\n")

        if response.headers["Content-Type"] != "image/jpeg":
          return None
        
        #print(response.status_code)

        if response.status_code != 200:
          print(f"Error: code {response.status_code} \n")
          #print(response.text)
          return

        #print(response.text)

        return response.content
        
    except Exception as e:
        print(f"Error: {e}")
