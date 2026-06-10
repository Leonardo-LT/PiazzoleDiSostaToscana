import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def centroidsToRgbPalette(ab_centers):
    y_channel = np.full((ab_centers.shape[0], 1), 80.0)
    ycrcb_centers = np.hstack([y_channel, ab_centers]).astype(np.uint8)
    return cv2.cvtColor(np.uint8([ycrcb_centers]), cv2.COLOR_YCrCb2RGB)[0]


def sortByPercentage(percentages, palette):
    sort_indices = np.argsort(percentages)[::-1]

    sorted_percentages = percentages[sort_indices]
    sorted_palette = palette[sort_indices]

    return sorted_percentages, sorted_palette


def calculatePaletteDifference(percentages1, palette1, percentages2, palette2):
    palette1 = np.asarray(palette1)
    palette2 = np.asarray(palette2)

    dims1 = palette1.shape[1]
    dims2 = palette2.shape[1]

    if dims1 != dims2:
        raise ValueError(f"Palette dimensions do not match: {dims1} vs {dims2}")

    sig1 = np.empty((len(percentages1), dims1 + 1), dtype=np.float32)
    sig1[:, 0] = percentages1
    sig1[:, 1:] = palette1

    sig2 = np.empty((len(percentages2), dims2 + 1), dtype=np.float32)
    sig2[:, 0] = percentages2
    sig2[:, 1:] = palette2

    cost, lower_bound, flow = cv2.EMD(sig1, sig2, cv2.DIST_L2)

    return cost


def calculateAllPaletteDifferences(percentages, palettes, names):
    results = []
    for i in range(len(percentages)):
        if i + 1 == len(percentages):
            break
        for j in range(i + 1, len(percentages)):
            cost = calculatePaletteDifference(
                percentages[i], palettes[i], percentages[j], palettes[j]
            )
            results.append({"pal1": names[i], "pal2": names[j], "cost": cost})

    return results


def clustering(folderPath):
    ABpixels = []

    for imageName in os.listdir(folderPath):
        if "png" not in imageName:
            continue
        image = cv2.imread(os.path.join(folderPath, imageName))
        image = cv2.bilateralFilter(image, d=7, sigmaColor=75, sigmaSpace=75)

        # image = cv2.resize(image, (50, 50))
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        pixels = image_lab.reshape((-1, 3))[:, 1:3]
        ABpixels.append(pixels)

    kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, n_init="auto")
    ABpixels = np.vstack(ABpixels)

    estimator = kmeans.fit(ABpixels)
    ab_centers = estimator.cluster_centers_
    eslabels = estimator.labels_

    labels, counts = np.unique(eslabels, return_counts=True)
    totalPixels = len(ABpixels)
    percentages = (counts / totalPixels) * 100

    plt.pie(
        x=percentages,
        labels=range(len(kmeans.cluster_centers_)),
        colors=centroidsToRgbPalette(ab_centers) / 255.0,
    )
    plt.plot()
    plt.show()

    return ab_centers, percentages  # kmeans


def findElbowPoint(folderPath, max_k=10):
    ABpixels = []
    for dirpath in os.listdir(folderPath):
        for imageName in os.listdir(os.path.join(folderPath, dirpath)):
            image = cv2.imread(os.path.join(folderPath, dirpath, imageName))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
            pixels = image_lab.reshape((-1, 3))
            ABpixels.append(pixels[:, 1:])

    ABpixels = np.vstack(ABpixels)
    inertia_values = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)

        kmeans.fit(ABpixels)

        inertia_values.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia_values, marker="o", linestyle="-", color="g")
    plt.title("Elbow Method (Shadow-Invariant: a & b channels only)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Errors)")
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()


def rgbToHEX(rgb):
    return "#%x%x%x" % (int(255 * rgb[0]), int(255 * rgb[1]), int(255 * rgb[2]))


# findElbowPoint("./Test2", max_k=10) #### elbow : 3 ####
# exit()

# findElbowPoint("./dataset/train", max_k=10) #### elbow : 3 ####
# exit()

# findElbowPoint("./dataset/evaluate", max_k=10) #### elbow : 3 ####
# exit()

if __name__ == "__main__":
    # test2Palette, test2ClusterPercentages = clustering("./datasetMix/Test2")
    # trainPalette, trainClusterPercentages = clustering("./datasetMix/train")
    # evalPalette, evalClusterPercentages = clustering("./datasetMix/eval")
    # testPalette, testClusterPercentages = clustering("./datasetMix/test")

    # test2Palette, test2ClusterPercentages = clustering("./FilteredDataset/FilteredTest")
    trainPalette, trainClusterPercentages = clustering(
        "/home/leon/Documenti/TestTesi/datasets/21-04.dataset(0.0004, 0.0004)Ann (Copia)/train"
    )
    evalPalette, evalClusterPercentages = clustering(
        "/home/leon/Documenti/TestTesi/datasets/21-04.dataset(0.0004, 0.0004)Ann (Copia)/eval"
    )
    testPalette, testClusterPercentages = clustering(
        "/home/leon/Documenti/TestTesi/datasets/21-04.dataset(0.0004, 0.0004)Ann (Copia)/test"
    )

    # res = calculateAllPaletteDifferences([test2ClusterPercentages, trainClusterPercentages, evalClusterPercentages, testClusterPercentages], [test2Palette, trainPalette, evalPalette, testPalette], ["test2", "train", "eval", "test"])
    res = calculateAllPaletteDifferences(
        [trainClusterPercentages, evalClusterPercentages, testClusterPercentages],
        [trainPalette, evalPalette, testPalette],
        ["train", "eval", "test"],
    )
    total = 0
    print(res)
    for cost in res:
        total += cost["cost"]

    print(total / len(res))

    # est2Palette = centroidsToRgbPalette(test2Palette)
    trainPalette = centroidsToRgbPalette(trainPalette)
    evalPalette = centroidsToRgbPalette(evalPalette)
    testPalette = centroidsToRgbPalette(testPalette)

    # test2ClusterPercentages, test2Palette = sortByPercentage(test2ClusterPercentages, test2Palette)
    trainClusterPercentages, trainPalette = sortByPercentage(
        trainClusterPercentages, trainPalette
    )
    evalClusterPercentages, evalPalette = sortByPercentage(
        evalClusterPercentages, evalPalette
    )
    testClusterPercentages, testPalette = sortByPercentage(
        testClusterPercentages, testPalette
    )

    # test2Palette = test2Palette / 255.0
    trainPalette = trainPalette / 255.0
    evalPalette = evalPalette / 255.0
    testPalette = testPalette / 255.0

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # axs[0, 0].pie(test2ClusterPercentages, colors=test2Palette, labels=list(map(rgbToHEX, test2Palette)))
    # axs[0, 0].set_title("Test2 Palette")

    axs[0, 1].pie(
        trainClusterPercentages,
        colors=trainPalette,
        labels=list(map(rgbToHEX, trainPalette)),
    )
    axs[0, 1].set_title("Train Palette")

    axs[1, 0].pie(
        evalClusterPercentages,
        colors=evalPalette,
        labels=list(map(rgbToHEX, evalPalette)),
    )
    axs[1, 0].set_title("Eval Palette")

    axs[1, 1].pie(
        testClusterPercentages,
        colors=testPalette,
        labels=list(map(rgbToHEX, testPalette)),
    )
    axs[1, 1].set_title("Test Palette")

    fig.suptitle("Palettes Distribution", fontsize=16)

    plt.show()
