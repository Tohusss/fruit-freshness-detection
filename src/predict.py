import os
import torch
from torchvision import transforms
from PIL import Image
from src.model import MultiTaskResNet

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "multitask_resnet18.pth")

def load_model(model_path=MODEL_PATH):
    model = MultiTaskResNet(num_fruits=3, num_freshness=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def predict_result(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    model = load_model()

    fruit_output, freshness_output = model(image)

    fruit_probs = torch.softmax(fruit_output, dim=1)[0]
    freshness_probs = torch.softmax(freshness_output, dim=1)[0]

    fruit_classes = ["Apple", "Banana", "Strawberry"]
    freshness_classes = ["Fresh", "Rotten"]

    fruit_idx = torch.argmax(fruit_probs).item()
    freshness_idx = torch.argmax(freshness_probs).item()

    fruit_conf = round(float(fruit_probs[fruit_idx] * 100), 2)
    freshness_conf = round(float(freshness_probs[freshness_idx] * 100), 2)

    return (
        fruit_classes[fruit_idx],
        freshness_classes[freshness_idx],
        fruit_conf,
        freshness_conf
    )
