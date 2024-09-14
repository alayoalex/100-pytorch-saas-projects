import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the FashionMNIST dataset
transform = transforms.ToTensor()  # Convert to tensor
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# Step 2: Get a batch of data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# Step 3: Define classes for labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# Step 4: Function to show images
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()  # Convert to numpy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Transpose to get the right dimensions (HWC)
    plt.show()

# Step 5: Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Step 6: Show the images
imshow(torchvision.utils.make_grid(images))

# Step 7: Print labels for the images
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
