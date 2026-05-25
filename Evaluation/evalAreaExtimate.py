import os
import time

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None, positive_cat_ids=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.getImgIds())
        self.transform = transform
        self.positive_cat_ids = (
            set(positive_cat_ids) if positive_cat_ids is not None else None
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        img = Image.open(img_path).convert("RGB")
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            if (
                self.positive_cat_ids is None
                or ann["category_id"] in self.positive_cat_ids
            ):
                ann_mask = self.coco.annToMask(ann)
                mask = np.maximum(mask, ann_mask)

        img_tensor = tv_tensors.Image(img)
        mask_tensor = tv_tensors.Mask(torch.from_numpy(mask))

        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor, img_info["file_name"]


def testSegmentation(path, net, bSize):
    results = []

    startTime = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformEvalData = v2.Compose(
        [
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    testing_imgs = SegmentationDataset(
        img_dir=os.path.join(path, "train"),
        ann_file=os.path.join(path, "train", "annotations.json"),
        transform=transformEvalData,
    )

    test_iter = torch.utils.data.DataLoader(
        testing_imgs, batch_size=bSize, shuffle=False
    )

    net.to(device)
    net.eval()

    total_intersection = 0.0
    total_union = 0.0

    with torch.no_grad():
        for X, y, img_name in test_iter:
            X = X.to(device)
            y = y.to(device).float()

            if len(y.shape) == 3:
                y = y.unsqueeze(1)

            out = net(X)

            preds = (out > 0).float()

            batch_intersection = (preds * y).sum().item()
            batch_union = preds.sum().item() + y.sum().item() - batch_intersection

            total_intersection += batch_intersection
            total_union += batch_union

            intersections = (preds * y).sum(dim=(1, 2, 3))
            unions = preds.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) - intersections

            gt_areas = y.sum(dim=(1, 2, 3))
            ext_areas = preds.sum(dim=(1, 2, 3))

            for i in range(len(img_name)):
                if gt_areas[i] == 0:
                    continue

                img_iou = (intersections[i].item() + 1e-6) / (unions[i].item() + 1e-6)

                results.append(
                    {
                        "tileId": img_name[i],
                        "IoU": img_iou,
                        "extArea": ext_areas[i].item(),
                        "gtArea": gt_areas[i].item(),
                        "group": "test",
                    }
                )

    endTime = time.time()
    print(f"Evaluation took {endTime - startTime:.2f} seconds")

    avg_iou = (total_intersection + 1e-6) / (total_union + 1e-6)

    df = pd.DataFrame(results)
    df.to_csv("SEG_results.csv", index=False, mode="w")

    return avg_iou


net = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)

loaded = torch.load(
    "/home/leon/Documenti/Tesi/ultimi/bestUnet.pth",
    map_location=torch.device("cpu"),
)
net.load_state_dict(loaded)

testSegmentation(
    "/home/leon/Documenti/TestTesi/datasets/21-04.dataset(0.0004, 0.0004)Ann (Copia)",
    net,
    32,
)
