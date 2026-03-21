import geopandas as gpd
import pandas as pd
import pyogrio

class tilesMetadataManager:
  def __init__(self, pathToCSV):
    self.pathToCSV = pathToCSV
    self.gdf = None
    self.counter = 0

    try:
      self.gdf = gpd.read_file(self.pathToCSV)
    except pyogrio.errors.DataSourceError:
      print("File not found, creating new one")
      # valutare se è meglio effettuare un crop dell'immagine in corrispondenza delle piazzole
      # in modo da avere metadati singoli per ogni piazzola,
      # oppure creare un altro file csv con i metadati delle piazzole, e collegarlo a quello delle tiles tramite idtile
      # o ancora avere tutto in questo file, 
      # avendo un booleano che indica se la tile è una piazzola o meno 
      # (assumo che tutte le immagini contengano piazzole),
 
      self.gdf = gpd.GeoDataFrame(columns=["tileID", "tileBBOX", "piazzola", "piazzolaBBOXs", "sizeEstimated", "downloadDate", "firstCNNconfidence", "secondCNNconfidence"])
    except Exception as e:
      print(e)

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
      "downloadDate": downloadDate,
      "firstCNNconfidence": firstCNNconfidence,
      "secondCNNconfidence": secondCNNconfidence
    }
    self.gdf = pd.concat([self.gdf, gpd.GeoDataFrame([newRow])], ignore_index=True)

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
      self.gdf.to_file(self.pathToCSV, index=False)
    except Exception as e:
      print(f"Error saving to CSV: {e}")