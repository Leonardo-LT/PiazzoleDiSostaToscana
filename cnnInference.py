import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2

# pretrained_vgg19 = torchvision.models.vgg19()
# pretrained_vgg19.classifier[6] = nn.Linear(pretrained_vgg19.classifier[6].in_features, 2)
# state_dict = torch.load("/home/leon/Documenti/Tesi/bestModelVgg19.pth", map_location=torch.device('cpu'))
# pretrained_vgg19.load_state_dict(state_dict)

pretrained_resnet50 = torchvision.models.resnet50()
pretrained_resnet50.fc = nn.Linear(pretrained_resnet50.fc.in_features, 2)
state_dict = torch.load(
    "/home/leon/Documenti/Tesi/bestModelResnet(ds0.0004_21-04).pth",
    map_location=torch.device("cpu"),
)
pretrained_resnet50.load_state_dict(state_dict)

transformEvalData = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def inferFromBytes(imgBytes, net=pretrained_resnet50):
    startTime = time.time()

    device = torch.device("cpu")
    net.to(device)
    net.eval()

    with torch.no_grad():
        imageT = transformEvalData(imgBytes)
        imageT.unsqueeze_(0)
        out = net(imageT)
        confidence = nn.functional.softmax(out, dim=1)
        idx, preds = torch.max(out, dim=1)
        # print("confidence: ", confidence)

        res = preds.item()
        endTime = time.time()

        return confidence[0][1].item()


def inferFromPath(path, net=pretrained_resnet50):
    imgBytes = Image.open(path).convert("RGB")
    return inferFromBytes(imgBytes, net)


path = "/home/leon/Documenti/TestTesi/tileRequestManagement/tiles/43.73815410152688.11.290228026961914.png"


print(inferFromPath(path))
# img = Image.open(path)
# img.show()


def inferSegmentation(img):
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

    # startTime = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformEvalData = v2.Compose(
        [
            v2.ToImage(),
            # v2.Resize(size=(224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tensor = transformEvalData(img)
    batch_img = tensor.unsqueeze(0)

    net.to(device)
    net.eval()

    with torch.no_grad():
        batch_img = batch_img.to(device)

        out = net(batch_img)

        print(out)

        prob_mask = torch.sigmoid(out).squeeze().cpu().numpy()
        binary_mask = (prob_mask > 0.5).astype(np.uint8) * 255

        points = cv2.findNonZero(binary_mask)
        print(points)

        if points is None:
            return 0.0

        hull = cv2.convexHull(points)

        hull_mask = np.zeros_like(binary_mask)
        cv2.fillConvexPoly(hull_mask, hull, 255)

        area = cv2.contourArea(hull)

        combined_image = cv2.hconcat([binary_mask, hull_mask])
        cv2.imwrite("./immComp.png", combined_image)

        # print(area)

        return area

    # endTime = time.time()
    # print(f"Evaluation took {endTime - startTime:.2f} seconds")
