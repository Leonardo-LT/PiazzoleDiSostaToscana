import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

num_classes = 2
label_list = ["piazzola"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = fasterrcnn_resnet50_fpn_v2(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(
    torch.load(
        "/home/leon/Scaricati/model_epoch_30(faster 0.0004).pth",
        map_location=torch.device("cpu"),
    )
)
model.to(device)
model.eval()

# img_path = "./Test2/1/43.97621031547912.10.17341508659892.png"
# image_pil = Image.open(img_path).convert("RGB")
# transform = transforms.Compose([transforms.ToTensor()])
# image_tensor = transform(image_pil).unsqueeze(0).to(device)
# with torch.no_grad():
#     predictions = model(image_tensor)
#     print(predictions)
# draw = ImageDraw.Draw(image_pil)
# for box, label_idx, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
#     if score > 0.55:
#         b = box.cpu().numpy().tolist()
#         draw.rectangle(b, outline="red", width=3)

#         label_text = f"{label_list[label_idx-1]}: {score:.2f}"
#         draw.text((b[0], b[1] - 10), label_text, fill="red")

# image_pil.show()


def plotMetricsComparison(res, dataSize):
    labels = ["Accuracy", "Recall", "Precision", "F1-score"]
    resnet50Metrics = [
        accuracy(res["TP"], res["TN"], dataSize),
        recall(res["TP"], res["FN"]),
        precision(res["TP"], res["FP"]),
        f1(res["TP"], res["FP"], res["FN"]),
    ]

    plt.bar(labels, resnet50Metrics, label="Faster-R-CNN")
    plt.ylim(0, 1)
    plt.title("Metrics Comparison")
    plt.show()


def metrics(pred, act):
    metrics = [0, 0, 0, 0]  # TP, FP, FN, TN
    for i in range(len(pred)):
        if pred[i] == 1:
            if act[i] == 0:
                metrics[1] += 1
            else:
                metrics[0] += 1
        else:
            if act[i] == 0:
                metrics[3] += 1
            else:
                metrics[2] += 1
    return metrics


def accuracy(tp, tn, dsLen):
    return (tp + tn) / dsLen

    # return (tp + tn) / 152


def recall(tp, fn):
    return tp / (tp + fn)


def precision(tp, fp):
    return tp / (tp + fp)


def f1(tp, fp, fn):
    return 2 * (
        (precision(tp, fp) * recall(tp, fn)) / (precision(tp, fp) + recall(tp, fn))
    )


def printAllMetrics(metrics, dsLen):
    print("Acc: ", accuracy(metrics[0], metrics[3], dsLen))
    print("Recall: ", recall(metrics[0], metrics[2]))
    print("Precision: ", precision(metrics[0], metrics[1]))
    print("F1: ", f1(metrics[0], metrics[1], metrics[2]))


def testCNN(path, net):
    startTime = time.time()
    transformEvalData = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    total_metrics = np.array([0, 0, 0, 0])

    bSize = 10

    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            os.path.join(path), transform=transformEvalData
        ),
        batch_size=bSize,
    )

    net.eval()

    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            out = net(X)

            # preds = np.zeros(bSize)
            results = map(
                lambda res: len(res["scores"]) > 0 and any(res["scores"] > 0.5), out
            )

            total_metrics = total_metrics + np.array(
                metrics(list(results), y.data.numpy())
            )
            # print(total_metrics)

        endTime = time.time()
        print(test_iter.dataset.__len__())
        printAllMetrics(total_metrics, test_iter.dataset.__len__())
        return {
            "TP": total_metrics[0],
            "FP": total_metrics[1],
            "FN": total_metrics[2],
            "TN": total_metrics[3],
        }, endTime - startTime


# path = "./dataset/test"
# metrics, infTime = testCNN(path, model)
# test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
#     os.path.join(path)),
#     batch_size=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def infer(img):
    transformEvalData = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    with torch.no_grad():
        t = transformEvalData(img)
        t = t.unsqueeze(0).to(device)
        out = model(t)
        return out

        results = map(
            lambda res: len(res["scores"]) > 0 and any(res["scores"] > 0.5), out
        )
        return any(results)


# print(torchvision.__version__)
# image_pil = Image.open("../43.7359392738655.11.297472097413014.png").convert("RGB")
# predictions = infer(image_pil)

# draw = ImageDraw.Draw(image_pil)
# for box, label_idx, score in zip(
#     predictions[0]["boxes"], predictions[0]["labels"], predictions[0]["scores"]
# ):
#     if score > 0.55:
#         b = box.cpu().numpy().tolist()
#         draw.rectangle(b, outline="red", width=3)
#         label_text = f"{label_list[label_idx - 1]}: {score:.2f}"
#         draw.text((b[0], b[1] - 10), label_text, fill="red")
# image_pil.show()


# plotMetricsComparison(metrics, test_iter.dataset.__len__())

# print("Average inference time: ", infTime / test_iter.__len__())
