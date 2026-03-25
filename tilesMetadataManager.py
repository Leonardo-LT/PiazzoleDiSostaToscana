import geopandas as gpd
import pandas as pd
import pyogrio
import shapely
import datetime

class tilesMetadataManager:
  def __init__(self, pathToCSV):
    self.pathToCSV = pathToCSV
    self.gdf = None
    self.counter = 0
    self.rowBuffer = []

    try:
      self.gdf = gpd.read_file(self.pathToCSV)
      self.gdf["tileBBOX"] = self.gdf["tileBBOX"].apply(lambda x: shapely.wkt.loads(x))
    except pyogrio.errors.DataSourceError:
      print("File not found, creating new one")
      # valutare se è meglio effettuare un crop dell'immagine in corrispondenza delle piazzole
      # in modo da avere metadati singoli per ogni piazzola,
      # oppure creare un altro file csv con i metadati delle piazzole, e collegarlo a quello delle tiles tramite idtile
      # o ancora, avere tutto in questo file, 
      # avendo un booleano che indica se la tile è una piazzola (quindi un immagine croppata sulla piazzola) o meno 
      # (assumo che tutte le immagini contengano piazzole),
 
      self.gdf = gpd.GeoDataFrame(columns=["tileID", "tileBBOX", "piazzola", "piazzolaBBOXs", "sizeEstimated", "downloadDate", "firstCNNconfidence", "secondCNNconfidence"])
    except Exception as e:
      print(e)
    
    self.gdf = self.gdf.set_geometry("tileBBOX")

  def getTileMetadata(self, tileID):
    row = self.gdf[self.gdf["tileID"] == tileID]
    if row.empty:
      return None
    else:
      return row.iloc[0].to_dict()
    
  def addTileMetadata(self, tileID, tileBBOX, piazzola, piazzolaBBOXs, sizeEstimated, downloadDate, firstCNNconfidence, secondCNNconfidence):
    newRow = {
      "tileID": tileID,
      "tileBBOX": tileBBOX,
      "piazzola": piazzola,
      "piazzolaBBOXs": piazzolaBBOXs,
      "sizeEstimated": sizeEstimated,
      "downloadDate": downloadDate.isoformat(),
      "firstCNNconfidence": firstCNNconfidence,
      "secondCNNconfidence": secondCNNconfidence
    }

    self.rowBuffer.append(newRow)

  def modifyTileMetadata(self, tileID, **kwargs):
    index = self.gdf[self.gdf["tileID"] == tileID].index
    if not index.empty:
      for key, value in kwargs.items():
        if key in self.gdf.columns:
          self.gdf.at[index[0], key] = value
        else:
          print(f"Column {key} is invalid")
    else:
      print(f"Tile with ID {tileID} not found")
    
  def saveToCSV(self):
    try:
      otherGdf = gpd.GeoDataFrame(self.rowBuffer)
      self.gdf = gpd.GeoDataFrame(pd.concat([self.gdf, otherGdf], ignore_index=True))
      self.gdf.to_csv(self.pathToCSV, index=False)
    except Exception as e:
      print(f"Error saving to CSV: {e}")
