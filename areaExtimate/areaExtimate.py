import cv2
import numpy as np
from matplotlib import pyplot as plt


def build_max_distance_mask(lines, img_shape):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if lines is None or len(lines) < 2:
        return mask

    def parse(line):
        x1, y1, x2, y2 = line[0]
        return ((x1, y1, x2, y2), np.hypot(x2 - x1, y2 - y1))

    top_2 = sorted(map(parse, lines), key=lambda x: x[1], reverse=True)[:2]

    c1 = top_2[0][0]
    c2 = top_2[1][0]

    def top_bottom(coords):
        x1, y1, x2, y2 = coords
        return ((x1, y1), (x2, y2)) if y1 <= y2 else ((x2, y2), (x1, y1))

    l1_top, l1_bot = top_bottom(c1)
    l2_top, l2_bot = top_bottom(c2)

    if l1_top[0] > l2_top[0]:
        l1_top, l1_bot, l2_top, l2_bot = l2_top, l2_bot, l1_top, l1_bot

    polygon = np.array([[l1_top, l1_bot, l2_bot, l2_top]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)

    return mask


def cleanEdges(edges, span=0.35):
    h, w = edges.shape
    img_diag = np.hypot(h, w)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edges, connectivity=8
    )

    filtered_edges = np.zeros_like(edges)

    for lbl in range(1, num_labels):
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]

        comp_diag = np.hypot(bh, bw)

        if comp_diag >= (span * img_diag):
            filtered_edges[labels == lbl] = 255

    return filtered_edges


def buildMask(edges, size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return
    contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(cleaned)
    cv2.fillPoly(mask, pts=[contour], color=255)

    return mask


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

    lower = 40
    upper = 120
    raw_edges = cv2.Canny(filtered, lower, upper)
    clean = cleanEdges(raw_edges)

    lines = cv2.HoughLinesP(
        clean,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=50,
        maxLineGap=30,
    )

    line_mask = np.zeros_like(clean)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

    mask = buildMask(clean, 21)
    mask = build_max_distance_mask(lines, clean.shape)

    return {
        "channel": filtered,
        "raw_edges": raw_edges,
        "final_edges": raw_edges,
        "clean_edges": line_mask,
        "mask": mask,
        "lines": lines,
    }


def runEdgePipelineWPad(img_bgr, bbox, padding=0):
    h, w = img_bgr.shape[:2]
    bx, by, bw, bh = [int(v) for v in bbox]

    x1 = max(0, bx - padding)
    y1 = max(0, by - padding)
    x2 = min(w, bx + bw + padding)
    y2 = min(h, by + bh + padding)

    crop_bgr = img_bgr[y1:y2, x1:x2]
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    result = runEdgePipeline(crop_gray)
    return result


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

    axes[1, 2].imshow(gray_results["mask"], cmap="gray")
    axes[1, 2].set_title("Mask")
    axes[1, 2].axis("off")

    plt.tight_layout()

    plt.show()
