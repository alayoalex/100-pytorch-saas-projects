import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.utils as vutils


# Create a directory to save the images
output_dir = './fashion_mnist_examples'
os.makedirs(output_dir, exist_ok=True)

# Load the FashionMNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.FashionMNIST(root='./fashion_mnist_dataset/', train=True, download=True, transform=transform)

# Create a DataLoader for easier batch processing
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

# Class labels in FashionMNIST
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Replace invalid characters in file names
def clean_label(label):
    return label.replace('/', '_')

# Retrieve and save 10 images
count = 0
for images, labels in trainloader:
    # Stop after saving 10 images
    if count >= 10:
        break

    # Get the label name
    label = clean_label(classes[labels[0]])

    # Convert tensor to PIL image and save
    img = transforms.ToPILImage()(images[0])  # Convert the image tensor to a PIL image
    img.save(os.path.join(output_dir, f'{label}_{count}.png'))  # Save the image

    print(f'Saved image {count+1}: {label}_{count}.png')
    count += 1

# Optionally, you can visualize the saved images
def show_saved_images():
    for i in range(10):
        img = Image.open(os.path.join(output_dir, f'{classes[labels[i]]}_{i}.png'))
        plt.imshow(img)
        plt.title(f'Image {i+1}: {classes[labels[i]]}')
        plt.show()

# Call this function if you want to visualize them
# show_saved_images()
