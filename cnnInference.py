import torch
import torchvision
import time
from torchvision import transforms
import os
import torch.nn as nn
from PIL import Image

# pretrained_vgg19 = torchvision.models.vgg19()
# pretrained_vgg19.classifier[6] = nn.Linear(pretrained_vgg19.classifier[6].in_features, 2)
# state_dict = torch.load("/home/leon/Documenti/Tesi/bestModelVgg19.pth", map_location=torch.device('cpu'))
# pretrained_vgg19.load_state_dict(state_dict)

pretrained_resnet50 = torchvision.models.resnet50()
pretrained_resnet50.fc = nn.Linear(pretrained_resnet50.fc.in_features, 2)
state_dict = torch.load("/home/leon/Documenti/Tesi/nuovodataset04/1/bestModelResnet.pth", map_location=torch.device('cpu'))
pretrained_resnet50.load_state_dict(state_dict)

transformEvalData = transforms.Compose([
  #transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def inferFromBytes(imgBytes, net = pretrained_resnet50):
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
    #print("confidence: ", confidence)

    res = preds.item()
    endTime = time.time()

    return confidence[0][1].item()

def inferFromPath(path, net = pretrained_resnet50):
  imgBytes = Image.open(path).convert("RGB")
  return inferFromBytes(imgBytes, net)
    

# path = "/home/leon/Documenti/TestTesi/tiles/43.98889135203641.10.15488072042202.png"

# print(inferFromPath(path))
# img = Image.open(path)
# img.show()