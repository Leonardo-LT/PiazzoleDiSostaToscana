import datetime
import os
import sys
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cv2
import numpy as np
import shapely
from PIL import Image, ImageDraw
from pyproj import Transformer
from requestTileContext import requestTileContext
from shapefileLineIter import shapefileLineIter
from tilesMetadataManager import tilesMetadataManager
from toscanaStrategy import ToscanaTileStrategy

from areaExtimate.areaExtimate import runEdgePipeline
from cnnInference import inferFromBytes as infer
from cnnInference import inferSegmentation
from piazzolaLocation import (
    classifyPoint,
    getMedianGradient,
    getRoadSide,
    pixelGradientToAngle,
    roadAngleAt,
)

# from RoadExtraction.filterImages import filterImage
from test import infer as inferBB
from utility.geo_utils import sqmPerPixel

tipiStrade = ["AA"]  # , "SS", "SR"
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


PIXEL_SIZE = tileSize * 2 / 224
SQM_PER_PIXEL = sqmPerPixel()


def saveImg(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def saveAnn(img, pred, path, filename):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    found = False
    for box, _, score in zip(pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]):
        if score > treshold:
            found = True
            b = box.cpu().numpy().tolist()
            draw.rectangle(b, outline="red", width=3)

            label_text = f"piazzola: {score:.2f}"
            draw.text((b[0], b[1] - 10), label_text, fill="red")

    if found:
        img.save(path + filename)
    else:
        print(pred[0]["scores"])
        # img.save(path + "/no" + filename)
    return found


reqTileContext = requestTileContext(ToscanaTileStrategy(tileSize, tileSize))
metadataManager = tilesMetadataManager("./tilesMetadata.csv")


def getNextCoordinates():
    try:
        point, lineGeom, dist, row = next(shapeiter)
        lat, lon = point.y, point.x
        angle = roadAngleAt(lineGeom, dist)
        segmentId = row.get("pk_uid", row.name) if hasattr(row, "get") else row.name
        return lat, lon, angle, segmentId
    except StopIteration:
        return None, None, None, None


def getContextImage(pending):
    if len(pending) == 1:
        p = pending[0]
        img = p["detections"][0]["img"]
        return img, PIXEL_SIZE, (p["lon"] - tileSize, p["lat"] + tileSize)

    allGeoBboxes = [bb for p in pending for bb in p["geoBboxes"]]
    union = shapely.ops.unary_union(allGeoBboxes)
    padding = PIXEL_SIZE * 10
    minLon, minLat, maxLon, maxLat = [
        union.bounds[0] - padding,
        union.bounds[1] - padding,
        union.bounds[2] + padding,
        union.bounds[3] + padding,
    ]

    widthPx = round((maxLon - minLon) / PIXEL_SIZE)
    heightPx = round((maxLat - minLat) / PIXEL_SIZE)

    imgBytes = reqTileContext.requestCustomBBoxBytes(
        minLat, minLon, maxLat, maxLon, widthPx, heightPx
    )
    if imgBytes is None:
        return None

    raw = np.asarray(bytearray(imgBytes), dtype="uint8")
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, PIXEL_SIZE, (minLon, maxLat)


def processDetections(img, origin, tileId, counter, roadAngle, roadSegmentId):
    originLon, originLat = origin

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = runEdgePipeline(gray)["lines"]
    gradient = getMedianGradient(lines)
    pred = inferBB(img)
    i = counter

    for bbox, _, score in zip(pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]):
        if score < treshold:
            continue
        i += 1
        xMin, yMin, xMax, yMax = map(int, bbox)

        location = classifyPoint((xMin + xMax) // 2, (yMin + yMax) // 2, gradient)

        roadSide = None
        if gradient is not None and roadAngle is not None:
            pixelAngle = pixelGradientToAngle(gradient)
            roadSide = getRoadSide(location, pixelAngle, roadAngle)

        print(f"{tileId} → piazzola #{i}, location: {location}, roadSide: {roadSide}")

        geoXMin = originLon + xMin * PIXEL_SIZE
        geoXMax = originLon + xMax * PIXEL_SIZE
        geoYMax = originLat - yMin * PIXEL_SIZE
        geoYMin = originLat - yMax * PIXEL_SIZE
        geoBbox = shapely.geometry.box(geoXMin, geoYMin, geoXMax, geoYMax)

        cropImg = img[yMin:yMax, xMin:xMax]
        extAreaSqm = inferSegmentation(cropImg) * SQM_PER_PIXEL

        saveImg(f"./tiles/{tileId}.{location}.png", img)
        metadataManager.addTileMetadata(
            tileId,
            geoBbox,
            None,
            None,
            extAreaSqm,
            datetime.datetime.now(),
            score.item(),
            None,
            roadAngle=roadAngle,
            roadSide=roadSide,
            roadSegmentId=roadSegmentId,
        )

    return i


def computeGeoBbox(bboxPixels, lat, lon):
    xMin, yMin, xMax, yMax = map(int, bboxPixels)
    topLeftLon = lon - tileSize
    topLeftLat = lat + tileSize
    geoXMin = topLeftLon + (xMin * PIXEL_SIZE)
    geoXMax = topLeftLon + (xMax * PIXEL_SIZE)
    geoYMax = topLeftLat - (yMin * PIXEL_SIZE)
    geoYMin = topLeftLat - (yMax * PIXEL_SIZE)
    return shapely.geometry.box(geoXMin, geoYMin, geoXMax, geoYMax)


def downloadTile(lat, long, path, filename):
    imgBytes = reqTileContext.requestTileBytes(lat, long)
    results = []

    if imgBytes is None:
        print("Not image")
        return []

    try:
        raw = np.asarray(bytearray(imgBytes), dtype="uint8")
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = inferBB(img)
        pred2 = infer(img)

        for bbox, _, score in zip(
            pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]
        ):
            results.append({"score": score, "bbox": bbox, "img": img, "pred2": pred2})

    except Exception as e:
        print(f"Error saving image: {e}")
        return []

    return results


def getAllTiles():
    i = 0
    pending = []
    emptyStreak = 0

    while True:
        lat, long, roadAngle, roadSegmentId = getNextCoordinates()

        if (lat, long) == (None, None):
            break
        if metadataManager.checkTileIdDup(f"{lat}.{long}"):
            continue

        results = downloadTile(lat, long, "./tiles", f"/{lat}.{long}.png")
        detections = [r for r in results if r["score"] >= treshold]

        if detections:
            emptyStreak = 0
            geoBboxes = [computeGeoBbox(d["bbox"], lat, long) for d in detections]
            pending.append(
                {
                    "lat": lat,
                    "lon": long,
                    "detections": results,
                    "geoBboxes": geoBboxes,
                    "roadAngle": roadAngle,
                    "roadSegmentId": roadSegmentId,
                }
            )
        else:
            emptyStreak += 1
            if emptyStreak >= 2 and pending:
                tileId = "|".join(f"{p['lat']}.{p['lon']}" for p in pending)
                pRoadAngle = pending[0]["roadAngle"]
                pRoadSegmentId = pending[0]["roadSegmentId"]
                ctx = getContextImage(pending)
                if ctx:
                    img, _, origin = ctx
                    i = processDetections(
                        img, origin, tileId, i, pRoadAngle, pRoadSegmentId
                    )
                pending = []
                emptyStreak = 0

    if pending:
        tileId = "|".join(f"{p['lat']}.{p['lon']}" for p in pending)
        pRoadAngle = pending[0]["roadAngle"]
        pRoadSegmentId = pending[0]["roadSegmentId"]
        ctx = getContextImage(pending)
        if ctx:
            img, _, origin = ctx
            i = processDetections(img, origin, tileId, i, pRoadAngle, pRoadSegmentId)

    print(f"Finished, found: {i}")


try:
    # os.mkdir("tiles")
    getAllTiles()
except KeyboardInterrupt as e:
    print(f"Error occurred: {e}")
    metadataManager.saveToCSV()
