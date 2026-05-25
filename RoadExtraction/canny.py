import cv2
import numpy as np
from matplotlib import pyplot as plt


def cleanEdges(edges, span=0.1):
    h, w = edges.shape
    img_diag = np.hypot(h, w)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edges, connectivity=8
    )
    filtered_edges = np.zeros_like(edges)

    for lbl in range(1, num_labels):
        x, y, bw, bh, _ = stats[lbl]

        touch_left = x <= 4
        touch_top = y <= 4
        touch_right = x + bw >= w - 4
        touch_bottom = y + bh >= h - 4

        touches = sum([touch_left, touch_top, touch_right, touch_bottom])

        if (touches >= 2) or np.hypot(bw, bh) >= span * img_diag:
            filtered_edges[labels == lbl] = 255

    return filtered_edges


def runEdgePipeline(single_channel):
    # equalized_image = cv2.equalizeHist(single_channel)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(single_channel)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # background = cv2.morphologyEx(single_channel, cv2.MORPH_DILATE, kernel)
    # shadow_removed = cv2.subtract(background, single_channel)

    filtered = cv2.bilateralFilter(gray_clahe, d=11, sigmaColor=90, sigmaSpace=95)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # enhanced = clahe.apply(filtered)

    lower = 25
    upper = 100
    raw_edges = cv2.Canny(filtered, lower, upper)
    clean = cleanEdges(raw_edges)

    return {
        "channel": filtered,
        "raw_edges": raw_edges,
        "final_edges": raw_edges,
        "clean_edges": clean,
    }


if __name__ == "__main__":
    img_bgr = cv2.imread(
        "/home/leon/Scaricati/43.748565385154166.11.171925343279899 (modificato).png"
    )
    # img_bgr = cv2.imread('/home/leon/Documenti/TestTesi/datasets/smallTest/1/43.96950384945796.10.184779412210856.png')

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv_img[:, :, 1]

    s_results = runEdgePipeline(s_channel)
    gray_results = runEdgePipeline(gray)

    s_mask = s_results["clean_edges"]
    gray_mask = gray_results["clean_edges"]

    fig, axes = plt.subplots(2, 3, figsize=(24, 9))

    axes[0, 0].imshow(s_results["channel"], cmap="gray")
    axes[0, 0].set_title("S Channel")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(s_results["final_edges"], cmap="gray")
    axes[0, 1].set_title("S: Edges")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(s_results["clean_edges"], cmap="gray")
    axes[0, 2].set_title("S: Mask")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(gray_results["channel"], cmap="gray")
    axes[1, 0].set_title("Grayscale")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gray_results["final_edges"], cmap="gray")
    axes[1, 1].set_title("Edges")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(gray_results["clean_edges"], cmap="gray")
    axes[1, 2].set_title("Mask")
    axes[1, 2].axis("off")

    plt.tight_layout()

    plt.show()
