import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from first_model.model_engine import NeuralNetwork, device
from first_model.dataloader import test_data


model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model-v1-20240913.pth", weights_only=True))
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose(transforms.ToTensor())
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat


def get_prediction_name(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat.item()


def open_image(image_path="../fashion_mnist_images/Sandal_0.png"):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return image_bytes
