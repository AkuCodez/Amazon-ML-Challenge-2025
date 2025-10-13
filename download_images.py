# ============================================================
# download_images.py
# Phase 2, Step 2: Download all product images (TRAIN SET ONLY)
# ============================================================

import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
TRAIN_CSV_PATH = "dataset/train.csv"
# TEST_CSV_PATH = "dataset/test.csv" # No longer needed for now
TRAIN_IMAGE_DIR = "dataset/train_images"
# TEST_IMAGE_DIR = "dataset/test_images" # No longer needed for now
MAX_WORKERS = 8
TIMEOUT = 10

# --- Create directories ---
os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
# os.makedirs(TEST_IMAGE_DIR, exist_ok=True) # No longer needed

# --- Download function ---
def download_and_save_image(args):
    """Downloads a single image, verifies it, and saves it."""
    sample_id, url, save_dir = args
    image_path = os.path.join(save_dir, f"{sample_id}.jpg")

    if os.path.exists(image_path):
        return "skipped"

    try:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()

        if 'image' not in response.headers.get('Content-Type', ''):
            return "not_image"

        with Image.open(BytesIO(response.content)) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(image_path, "JPEG", quality=90)
        return "success"
    
    except requests.exceptions.RequestException:
        return "download_error"
    except Exception:
        return "processing_error"

# --- Main script ---
def process_dataset(csv_path, image_dir):
    """Processes a CSV file to download all associated images in parallel."""
    df = pd.read_csv(csv_path)
    tasks = []
    for _, row in df.iterrows():
        tasks.append((row['sample_id'], row['image_link'], image_dir))

    success_count, skipped_count, error_count = 0, 0, 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(download_and_save_image, task): task for task in tasks}
        progress = tqdm(as_completed(future_to_url), total=len(tasks), desc=f"Downloading to {os.path.basename(image_dir)}")
        
        for future in progress:
            result = future.result()
            if result == "success":
                success_count += 1
            elif result == "skipped":
                skipped_count += 1
            else:
                error_count += 1
            
            progress.set_postfix({
                "success": success_count, "skipped": skipped_count, "errors": error_count,
            })

    print(f"\n✅ Finished {os.path.basename(image_dir)}:")
    print(f"   Successfully downloaded: {success_count}")
    print(f"   Skipped (already exist): {skipped_count}")
    print(f"   Errors: {error_count}")


if __name__ == "__main__":
    print("="*70)
    print("🖼️ Starting Image Download Process (TRAIN SET ONLY)")
    print("="*70)
    
    print("\nDownloading training images...")
    process_dataset(TRAIN_CSV_PATH, TRAIN_IMAGE_DIR)
    
    # --- Test set download is now disabled ---
    # print("\nDownloading test images...")
    # process_dataset(TEST_CSV_PATH, TEST_IMAGE_DIR)
    
    print("\n" + "="*70)
    print("✅ All training image downloads attempted!")
    print("="*70)