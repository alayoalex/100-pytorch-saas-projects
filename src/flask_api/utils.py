import io
import torch
import torchvision.transforms as transforms
from PIL import Image


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def open_image(image_path="../fashion_mnist_images/Sandal_0.png"):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return image_bytes
