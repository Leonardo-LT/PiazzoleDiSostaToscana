import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from canny import runEdgePipeline
from sklearn.cluster import KMeans

from colorDistribution import clustering


def filterImageOTSU(img):
    img = cv2.bilateralFilter(img, d=7, sigmaColor=90, sigmaSpace=90)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, color_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return color_mask


def filterImage(img, kmeans):
    imgKmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
    imgFilt = cv2.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)
    image_ycrcb = cv2.cvtColor(imgFilt, cv2.COLOR_BGR2YCrCb)
    ab_pixels = image_ycrcb.reshape((-1, 3))[:, 1:3]
    kResults = imgKmeans.fit(ab_pixels)

    local_labels = kResults.labels_.reshape(imgFilt.shape[:2])
    highlight_mask = np.zeros(imgFilt.shape[:2], dtype=np.uint8)

    for local_idx, center in enumerate(kResults.cluster_centers_):
        global_idx = kmeans.predict(center.reshape(1, -1))[0]
        if global_idx == 0 or global_idx == 3:
            highlight_mask[local_labels == local_idx] = 255

    image_bgr = img.copy().astype(np.float32)

    highlighted_pixels = image_bgr[highlight_mask == 255]
    highlighted_pixels *= 1.15
    image_bgr[highlight_mask == 255] = np.clip(highlighted_pixels, 0, 255)

    non_highlighted_pixels = image_bgr[highlight_mask == 0]
    non_highlighted_pixels *= 0.35
    image_bgr[highlight_mask == 0] = np.clip(non_highlighted_pixels, 0, 255)

    return image_bgr.astype(np.uint8)


def filterImageFromPath(imgPath, kmeans):
    img = cv2.imread(imgPath)
    return filterImage(img, kmeans)


def highlight_roads(img, mask):
    if mask is None:
        raise ValueError("Mask is None")

    # Ensure mask is single-channel.
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # OpenCV requires 8-bit mask for bitwise ops.
    if mask.dtype != np.uint8:
        if np.issubdtype(mask.dtype, np.bool_):
            mask = mask.astype(np.uint8) * 255
        else:
            mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Enforce exact spatial match with source image.
    h, w = img.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    masked_image = cv2.bitwise_and(img, img, mask=mask)
    return masked_image


def filterImages(path, dirName, kmeans):
    for dirPath in os.listdir(path):
        if "txt" in dirPath:
            continue
        for innerDirPath in os.listdir(os.path.join(path, dirPath)):
            imgPath = os.path.join(path, dirPath, innerDirPath)  # , imgName
            # img = filterImageFromPath(imgPath, kmeans)
            if "png" not in imgPath:
                continue

            img = cv2.imread(imgPath)
            # mask = filterImageFromPath(imgPath, kmeans)

            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            s_channel = hsv_img[:, :, 1]
            edge_results = runEdgePipeline(gray)
            # mask = edge_results["road_mask"]
            mask = edge_results["raw_edges"]

            res = highlight_roads(img, mask)
            print(os.path.join(f"{dirName}", dirPath, innerDirPath))
            cv2.imwrite(
                os.path.join(f"{dirName}", dirPath, innerDirPath),
                mask,  # , imgName
            )
        # for imgName in os.listdir(os.path.join(path, dirPath, innerDirPath)):


# testImage = filterImageFromPath("/home/leon/Documenti/TestTesi/FilteredDataset/evaluate/1/46.82.png")

# kmeans = clustering("/home/leon/Documenti/TestTesi/datasets/nuovoDataset(0.0004,0.0004)/train")
kmeans = None
filterImages(
    "/home/leon/Documenti/TestTesi/datasets/21-04.dataset(0.0004, 0.0004)Ann (Copia)",
    "/home/leon/Documenti/TestTesi/datasets/21-04.dataset(0.0004, 0.0004)Ann (Copia)(simplified)",
    kmeans,
)

# plt.figure(figsize=(8, 8))
# plt.imshow(testImage)
# plt.title("Road Brightened by 1.15x")
# plt.show()
# plt.axis('off')
