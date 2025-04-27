import os
import random
import numpy as np
import joblib
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image  
import matplotlib.pyplot as plt  

# Define dataset directories
dataset_dirs = [
    "D:/sam/dataset/Potato__Early_blight/", 
    "D:/sam/dataset/Potato__healthy/", 
    "D:/sam/dataset/Potato__Late_blight/"
]

# Randomly select a directory and image
random_dir = random.choice(dataset_dirs)
random_image = random.choice(os.listdir(random_dir))
image_path = os.path.join(random_dir, random_image)

print(f"Selected Image: {image_path}")

# Load ResNet18 model for feature extraction
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18.fc = torch.nn.Identity()  # Remove the classification layer
resnet18.eval()

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load and convert image to PIL format
image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to read image!")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
image_pil = Image.fromarray(image)  # Convert to PIL Image

# Apply transforms
image_tensor = transform(image_pil).unsqueeze(0)

# Extract features
with torch.no_grad():
    features = resnet18(image_tensor).numpy()

# Load trained classifier
clf = joblib.load("potato_classifier.pkl")

# Predict class
prediction = clf.predict(features)[0]
print(f"Prediction: {prediction}")

# Display the image with prediction
plt.figure(figsize=(6, 6))  # Bigger display
plt.imshow(image)
plt.axis("off")  
plt.title(f"Prediction: {prediction}", fontsize=14, fontweight='bold')
plt.show()
