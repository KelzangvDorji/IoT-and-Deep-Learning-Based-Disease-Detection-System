import os
import sys
import subprocess

def main():
    # Add src directory to Python path
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    sys.path.append(src_dir)
    
    # Step 1: Run background removal
    print("Step 1: Running background removal...")
    subprocess.run(['python', os.path.join(src_dir, 'background_removal.py')])
    
    # Step 2: Train classifier
    print("\nStep 2: Training classifier...")
    subprocess.run(['python', os.path.join(src_dir, 'train_classifier.py')])

if __name__ == "__main__":
    main() 