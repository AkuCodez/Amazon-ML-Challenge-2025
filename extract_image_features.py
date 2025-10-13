# Quick script to generate ONLY the test image features

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
TEST_CSV = "dataset/test.csv"
IMAGE_DIR = "dataset/images/" 
OUTPUT_FILE = "dataset/test_image_features.npy" # The file we need to create
BATCH_SIZE = 64 # Use a larger batch size for speed

# --- MAIN SCRIPT ---
print("="*60)
print("🚀 Generating MISSING TEST IMAGE FEATURES...")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Use a fast and powerful model like EfficientNet-B0
model = models.efficientnet_b0(pretrained=True)
model.classifier = nn.Identity() # Remove the final layer
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_df = pd.read_csv(TEST_CSV)
image_paths = [os.path.join(IMAGE_DIR, f"{sid}.jpg") for sid in test_df['sample_id']]

all_features = []
with torch.no_grad():
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Processing test images"):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_tensors = []
        for path in batch_paths:
            try:
                if os.path.exists(path):
                    img = Image.open(path).convert('RGB')
                    batch_tensors.append(transform(img))
                else:
                    batch_tensors.append(torch.zeros(3, 224, 224)) # Zero tensor for missing images
            except Exception:
                batch_tensors.append(torch.zeros(3, 224, 224))

        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)
            features = model(batch)
            all_features.append(features.cpu().numpy())

final_features = np.vstack(all_features)
np.save(OUTPUT_FILE, final_features)

print(f"\n✅ SUCCESS! Created '{OUTPUT_FILE}' with shape {final_features.shape}")
print("You are now ready to run the ultimate model.")