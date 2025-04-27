import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

class PotatoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.categories = ["Potato__Early_blight", "Potato__healthy", "Potato__Late_blight"]
        self.image_paths = []
        self.labels = []
        
        print(f"Initializing dataset from directory: {data_dir}")
        
        for idx, category in enumerate(self.categories):
            category_path = os.path.join(data_dir, category)
            print(f"Processing category: {category} from path: {category_path}")
            
            if not os.path.exists(category_path):
                print(f"Warning: Category path does not exist: {category_path}")
                continue
                
            try:
                files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"Found {len(files)} images in category {category}")
                
                for img_name in files:
                    img_path = os.path.join(category_path, img_name)
                    if not os.path.exists(img_path):
                        print(f"Warning: Image file not found: {img_path}")
                        continue
                    self.image_paths.append(img_path)
                    self.labels.append(idx)
            except Exception as e:
                print(f"Error processing category {category}: {str(e)}")
        
        if len(self.image_paths) == 0:
            print("Warning: No valid images found in any category")
        else:
            print(f"Total images loaded: {len(self.image_paths)}")
            print(f"Category distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
                
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a default image in case of error
            default_image = torch.zeros((3, 224, 224))
            return default_image, self.labels[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
        # Save best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            model_save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
    
    return model, train_losses, val_losses, train_accs, val_accs

def main():
    print("Starting the training pipeline...")
    
    # Data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Create datasets
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    print(f"Loading dataset from: {data_dir}")
    
    try:
        full_dataset = PotatoDataset(data_dir, transform=data_transforms['train'])
        if len(full_dataset) == 0:
            raise ValueError("Dataset is empty. Please check if the data directory contains valid images.")
        print(f"Dataset loaded successfully with {len(full_dataset)} images")
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        print(f"Split dataset into {train_size} training and {val_size} validation samples")
        
        # Create dataloaders with minimal configuration
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        print("Created data loaders")
        
        # Initialize model
        print("Initializing ResNet18 model...")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)  # 3 classes
        
        if torch.cuda.is_available():
            print("CUDA is available, using GPU")
            model = model.cuda()
        else:
            print("CUDA is not available, using CPU")
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        # Train model
        print("Starting training...")
        model, train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, criterion, optimizer, num_epochs=10
        )
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.title('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.legend()
        plt.title('Accuracy')
        
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'training_history.png'))
        plt.close()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 