import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from  sklearn.cluster import KMeans 
from colorDistribution import clustering



def filterImage(img, kmeans):
    imgKmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
    img = cv2.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)
    image_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ab_pixels = image_ycrcb.reshape((-1, 3))[:, 1:3]
    kResults = imgKmeans.fit(ab_pixels)

    local_labels = kResults.labels_.reshape(img.shape[:2])
    highlight_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for local_idx, center in enumerate(kResults.cluster_centers_):
        global_idx = kmeans.predict(center.reshape(1, -1))[0]
        if global_idx == 0 or global_idx == 3:
            highlight_mask[local_labels == local_idx] = 255

    image_bgr = img.copy().astype(np.float32)

    highlighted_pixels = image_bgr[highlight_mask == 255]
    highlighted_pixels *= 1.35
    image_bgr[highlight_mask == 255] = np.clip(highlighted_pixels, 0, 255)

    non_highlighted_pixels = image_bgr[highlight_mask == 0]
    non_highlighted_pixels *= 0.35
    image_bgr[highlight_mask == 0] = np.clip(non_highlighted_pixels, 0, 255)

    return image_bgr.astype(np.uint8)


def filterImageFromPath(imgPath, kmeans):
    img = cv2.imread(imgPath)
    return filterImage(img, kmeans)


def filterImages(path, dirName, kmeans):
    for dirPath in os.listdir(path):
        for innerDirPath in os.listdir(os.path.join(path, dirPath)):
            imgPath = os.path.join(path, dirPath, innerDirPath)  # , imgName
            img = filterImageFromPath(imgPath, kmeans)
            print(os.path.join(f"{dirName}", dirPath, innerDirPath))
            cv2.imwrite(
                os.path.join(f"./{dirName}", dirPath, innerDirPath),
                img,  # , imgName
            )
        # for imgName in os.listdir(os.path.join(path, dirPath, innerDirPath)):


# testImage = filterImageFromPath("/home/leon/Documenti/TestTesi/FilteredDataset/evaluate/1/46.82.png")

kmeans = clustering("/home/leon/Documenti/TestTesi/datasets/nuovoDataset(0.0004,0.0004)/train")

filterImages("/home/leon/Documenti/TestTesi/datasets/smallTest", "./FilteredDatasetSma", kmeans)

# plt.figure(figsize=(8, 8))
# plt.imshow(testImage)
# plt.title("Road Brightened by 1.15x")
# plt.show()
# plt.axis('off')
