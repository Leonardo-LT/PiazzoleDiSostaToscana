import json
import os

import cv2
import numpy as np
from canny import runEdgePipeline


def calculate_mask_iou(coco_json_path, img_dir):
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    images_info = {img["id"]: img for img in coco_data["images"]}

    total_iou = 0
    valid_images = 0

    for img_id, img_info in images_info.items():
        gt_mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)

        annotations = [
            ann for ann in coco_data["annotations"] if ann["image_id"] == img_id
        ]

        for ann in annotations:
            for segment in ann["segmentation"]:
                poly = np.array(segment, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(gt_mask, [poly], color=255)

        img = cv2.imread(os.path.join(img_dir, img_info["file_name"]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prediction = runEdgePipeline(gray)

        pred_mask = prediction["clean_edges"]

        gt_binary = (gt_mask > 127).astype(bool)
        pred_binary = (pred_mask > 127).astype(bool)

        intersection = np.logical_and(gt_binary, pred_binary).sum()
        union = np.logical_or(gt_binary, pred_binary).sum()

        if union > 0:
            iou = intersection / union
            total_iou += iou
            valid_images += 1

    if valid_images > 0:
        return total_iou / valid_images


res = calculate_mask_iou(
    "/home/leon/Documenti/TestTesi/RoadExtraction/TestImages/labels.json",
    "/home/leon/Documenti/TestTesi/RoadExtraction/TestImages/",
)

print(res)
