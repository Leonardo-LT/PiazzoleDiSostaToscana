import datetime
import os
import sys
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cv2
import numpy as np
import shapely
from PIL import Image, ImageDraw
from requestTileContext import requestTileContext
from shapefileLineIter import shapefileLineIter
from tilesMetadataManager import tilesMetadataManager
from toscanaStrategy import ToscanaTileStrategy

from cnnInference import inferFromBytes as infer

# from RoadExtraction.filterImages import filterImage
from test import infer as inferBB

tipiStrade = ["AA", "SS", "SR"]  # , "SS", "SR"
treshold = 0.6


def filter(el):
    if el is None:
        return False

    return el["tipostrada"].isin(tipiStrade)


tileSize = 0.0004

sli = shapefileLineIter(
    "/home/leon/Documenti/Tesi/iternet/iternet_c91b903823ae2a975c539cc65f880af9/iternet/shp/",
    "strade",
    tileSize,
)
sli.setGdfCRS("EPSG:4326")
sli.setGdfFilter(filter)
shapeiter = iter(sli)


def saveImg(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def saveAnn(img, pred, path, filename):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    found = False
    for box, label_idx, score in zip(
        pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]
    ):
        if score > treshold:
            found = True
            b = box.cpu().numpy().tolist()
            draw.rectangle(b, outline="red", width=3)

            label_text = f"piazzola: {score:.2f}"
            draw.text((b[0], b[1] - 10), label_text, fill="red")

    if found:
        print("here")
        img.save(path + filename)
    else:
        print(pred[0]["scores"])
        # img.save(path + "/no" + filename)
    return found


reqTileContext = requestTileContext(ToscanaTileStrategy(tileSize, tileSize))
metadataManager = tilesMetadataManager("./tilesMetadata.csv")


def downloadTile(lat, long, path, filename):
    bytes = reqTileContext.requestTileBytes(lat, long)
    if bytes == None:
        print("Not image")
        return 0

    try:
        bytes = np.asarray(bytearray(bytes), dtype="uint8")
        img = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # filteredImg = filterImage(img)
        # res = infer(img)
        res = inferBB(img)
        # res = 1

        # if res >= treshold:
        found = saveAnn(img, res, path, filename)

    except Exception as e:
        print(f"Error saving image: {e}")
        return 0

    return found


def getNextCoordinates():
    point = next(shapeiter)
    return point.y, point.x


def getAllTiles():
    i = 0  # contatore per salvare i metadati ogni tot file scaricati
    while True:
        lat, long = getNextCoordinates()
        if metadataManager.checkTileIdDup(f"{lat}.{long}"):
            continue

        if (lat, long) == (None, None):
            print("Shapefile iteration completed")
            break

        res = downloadTile(lat, long, "./tiles", f"/{lat}.{long}.png")
        if res:  # res >= treshold:
            i += 1
            print(f"{lat}.{long}, Piaz count: ", i)
            bboxPolygon = shapely.geometry.Polygon(
                [
                    (long - tileSize, lat - tileSize),
                    (long - tileSize, lat - tileSize),
                    (long + tileSize, lat + tileSize),
                    (long - tileSize, lat + tileSize),
                ]
            )
            metadataManager.addTileMetadata(
                f"{lat}.{long}",
                bboxPolygon,
                None,
                None,
                None,
                datetime.datetime.now(),
                res,
                None,
            )


try:
    # os.mkdir("tiles")
    getAllTiles()
except KeyboardInterrupt as e:
    print(f"Error occurred: {e}")
    metadataManager.saveToCSV()
