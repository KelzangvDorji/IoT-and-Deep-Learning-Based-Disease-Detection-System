import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

# Add U-2-Net directory to Python path
u2net_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'U-2-Net')
sys.path.append(u2net_dir)
from model.u2net import U2NET

def load_u2net_model(model_path):
    """Load the U-2-Net model"""
    print(f"Loading model from: {model_path}")
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    return net

def normPRED(d):
    """Normalize the predicted SOD probability map"""
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name, pred, output_dir):
    """Save the output image with removed background"""
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()
    
    # Load and resize original image
    image = cv2.imread(image_name)
    original_size = image.shape[:2]  # Store original size
    image = cv2.resize(image, (320, 320))  # Resize to match model input size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create mask (320x320 to match resized image)
    mask = predict_np > 0.5
    mask = mask.astype(np.uint8) * 255
    
    # Apply mask to resized image
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Resize back to original size
    result = cv2.resize(result, (original_size[1], original_size[0]))
    
    # Save the result
    img_name = image_name.split(os.sep)[-1]
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

def process_dataset(input_dir, output_dir, model_path):
    """Process all images in the dataset directory"""
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load U-2-Net model
    net = load_u2net_model(model_path)
    
    # Process each category
    categories = ["Potato__Early_blight", "Potato__healthy", "Potato__Late_blight"]
    
    for category in categories:
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)
        
        print(f"\nProcessing category: {category}")
        print(f"Category path: {category_path}")
        print(f"Output category path: {output_category_path}")
        
        # Check if directory exists and list contents
        if os.path.exists(category_path):
            print(f"Directory exists. Contents:")
            print(os.listdir(category_path))
        else:
            print(f"Directory does not exist: {category_path}")
            continue
            
        for img_name in os.listdir(category_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Make case-insensitive
                img_path = os.path.join(category_path, img_name)
                print(f"\nProcessing image: {img_path}")
                
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                    
                print(f"Successfully loaded image: {img_path}")
                print(f"Image shape: {img.shape}")
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (320, 320))
                img = img.astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                
                if torch.cuda.is_available():
                    img = img.cuda()
                    print("Using CUDA for processing")
                else:
                    print("Using CPU for processing")
                
                # Get prediction
                try:
                    with torch.no_grad():
                        d1, d2, d3, d4, d5, d6, d7 = net(img)
                        print("Successfully got prediction")
                    
                    # Normalize prediction
                    pred = d1[:, 0, :, :]
                    pred = normPRED(pred)
                    
                    # Save output
                    output_path = os.path.join(output_category_path, img_name)
                    save_output(img_path, pred, output_category_path)
                    print(f"Successfully saved output to: {output_path}")
                except Exception as e:
                    print(f"Error processing image: {e}")
                
                print(f"Processed: {img_path}")

if __name__ == "__main__":
    # Paths
    input_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'dataset')  # Path to original dataset
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')  # Path to save processed images
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'U-2-Net', 'saved_models', 'u2net', 'u2net.pth')  # Path to U-2-Net model
    
    print("Starting background removal process...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model path: {model_path}")
    
    # Process the dataset
    process_dataset(input_dir, output_dir, model_path) 