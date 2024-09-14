# app.py (FastAPI API)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from model import NeuralNetwork, device  # Assume your model is defined here

app = FastAPI()

# Allow CORS for all domains (you can restrict to your React app later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can set this to your React app domain, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model (assumes you've already trained it)
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model-v1-20240913.pth", map_location=device))
model.eval()

# Define classes corresponding to FashionMNIST
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28, 28)),  # Resize to the FashionMNIST format
                                    transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        tensor = transform_image(image_bytes)
        tensor = tensor.to(device)

        # Perform inference
        outputs = model(tensor)
        _, predicted = outputs.max(1)
        predicted_class = classes[predicted.item()]

        return {"prediction": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
