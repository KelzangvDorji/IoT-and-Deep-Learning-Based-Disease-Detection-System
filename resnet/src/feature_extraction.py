import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import os

# Load Pretrained ResNet18 Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
model.to(device)
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset Paths
dataset_path = "D:/sam/dataset/"
categories = ["Potato__Early_blight", "Potato__healthy", "Potato__Late_blight"]
features = []
labels = []

# Feature Extraction Loop
for idx, category in enumerate(categories):
    category_path = os.path.join(dataset_path, category)
    
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        
        # Read and preprocess image
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            feature = model(image).squeeze().cpu().numpy()
        
        features.append(feature)
        labels.append(idx)  # 0=Early Blight, 1=Healthy, 2=Late Blight

# Convert to NumPy arrays and save
features = np.array(features)
labels = np.array(labels)
np.save("features.npy", features)
np.save("labels.npy", labels)

print("Feature Extraction Completed! Saved 'features.npy' & 'labels.npy'")
