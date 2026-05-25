import json
import os

import cv2
import numpy as np
import pandas as pd

from areaExtimate import runEdgePipelineWPad


def compute_iou(pred, gt):
    p, g = pred > 127, gt > 127
    union = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum()) / union if union > 0 else float("nan")


def calculate_mask_iou(coco_json_path, img_dir, save_dir=None):
    with open(coco_json_path, "r") as f:
        data = json.load(f)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results = []
    ious = []

    for img in data["images"]:
        anns = [a for a in data["annotations"] if a["image_id"] == img["id"]]
        if not anns:
            continue

        img_bgr = cv2.imread(os.path.join(img_dir, img["file_name"]))
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        bx, by, bw, bh = map(int, anns[0]["bbox"])
        x1, y1 = max(0, bx - 5), max(0, by - 5)
        x2, y2 = min(w, bx + bw + 5), min(h, by + bh + 5)

        pred = runEdgePipelineWPad(img_bgr, anns[0]["bbox"], padding=5)
        pred_mask = pred["mask"]
        edges = pred["final_edges"]

        if pred_mask is None:
            continue

        gt_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

        for ann in anns:
            for seg in ann["segmentation"]:
                poly = np.array(seg, dtype=np.int32).reshape(-1, 2) - [x1, y1]
                cv2.fillPoly(gt_mask, [poly], 255)

        if gt_mask.shape != pred_mask.shape:
            gt_mask = cv2.resize(
                gt_mask,
                (pred_mask.shape[1], pred_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        extArea = np.sum(pred_mask > 127)
        gtArea = np.sum(gt_mask > 127)

        iou = compute_iou(pred_mask, gt_mask)
        if not np.isnan(iou):
            ious.append(iou)
            results.append(
                {
                    "tileId": img["file_name"],
                    "IoU": iou,
                    "extArea": extArea,
                    "gtArea": gtArea,
                    "bboxArea": (x2 - x1) * (y2 - y1),
                }
            )

        if save_dir:
            name = os.path.splitext(img["file_name"])[0] + "_masks.png"
            panel = np.hstack(
                [
                    img_bgr[y1:y2, x1:x2],
                    cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
                ]
            )
            cv2.imwrite(os.path.join(save_dir, name), panel)

    df = pd.DataFrame(results)
    df.to_csv("ED_results.csv", index=False, mode="w")

    return sum(ious) / len(ious) if ious else None


if __name__ == "__main__":
    print(
        calculate_mask_iou(
            "/home/leon/Documenti/TestTesi/datasets/21-04.dataset(0.0004, 0.0004)Ann (Copia)/test/annotations.json",
            "/home/leon/Documenti/TestTesi/datasets/21-04.dataset(0.0004, 0.0004)Ann (Copia)/test",
            "/home/leon/Documenti/TestTesi/areaExtimate/outImages",
        )
    )
