from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

num_classes = 2
label_list = ["piazzola"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = fasterrcnn_resnet50_fpn_v2(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("./model_epoch_21.pth", map_location=torch.device('cpu')))
model.to(device)
model.eval()

img_path = "./cocoDataset/eval/46.44.png"
image_pil = Image.open(img_path).convert("RGB")

transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image_pil).unsqueeze(0).to(device)

with torch.no_grad():
    predictions = model(image_tensor)

draw = ImageDraw.Draw(image_pil)
for box, label_idx, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
    if score > 0.7:
        b = box.cpu().numpy().tolist()
        draw.rectangle(b, outline="red", width=3)
        
        label_text = f"{label_list[label_idx-1]}: {score:.2f}"
        draw.text((b[0], b[1] - 10), label_text, fill="red")

image_pil.show()
