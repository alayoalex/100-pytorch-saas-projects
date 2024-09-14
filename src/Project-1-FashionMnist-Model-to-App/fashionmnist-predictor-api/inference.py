import torch
from model import NeuralNetwork, device
from utils import transform_image


model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model-v1-20240913.pth", weights_only=True))
model.eval()

def get_prediction(image_bytes):
    with torch.no_grad():
        tensor = transform_image(image_bytes=image_bytes)
        tensor = tensor.to(device)
        outputs = model.forward(tensor)
        _, y_hat = outputs.max(1)
        return y_hat
