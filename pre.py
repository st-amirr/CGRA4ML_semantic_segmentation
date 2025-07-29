import os
import numpy as np
import cv2
import gc
import psutil
import signal
import sys
import glob
from pathlib import Path
from train import INPUT_WIDTH, INPUT_HEIGHT, NUM_CLASSES, prepare_data
import albumentations as A

#  python3 /workspace/pre.py

def cleanup_files(output_dir):
    """Clean up any existing preprocessing files"""
    print("Cleaning up existing preprocessing files...")
    
    # Patterns to match previous preprocessing files
    patterns = [
        'train_images_chunk_*.npy',
        'train_masks_chunk_*.npy', 
        'val_images_chunk_*.npy',
        'val_masks_chunk_*.npy',
        'train_images.npy',
        'train_masks.npy',
        'val_images.npy', 
        'val_masks.npy'
    ]
    
    deleted_count = 0
    for pattern in patterns:
        files = glob.glob(os.path.join(output_dir, pattern))
        for file in files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"Could not delete {file}: {e}")
    
    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} files")
    else:
        print("No previous preprocessing files found")

def signal_handler(sig, frame):
    """Handle Ctrl+C to clean up files"""
    print("\n" + "="*60)
    print("INTERRUPTED BY USER (Ctrl+C)")
    print("="*60)
    print("Cleaning up temporary files...")
    
    # Clean up any partial files
    output_dir = 'models'
    cleanup_files(output_dir)
    
    print("Cleanup complete. Exiting safely.")
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def create_optimized_data_index(cityscapes_root, output_dir):
    """
    Create an optimized data index instead of preprocessing everything.
    This creates a small index file with file paths and basic info.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_images, train_masks, val_images, val_masks = prepare_data(cityscapes_root)
    
    print("Creating optimized data index...")
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    # Create index files with file paths and metadata
    train_index = []
    for i, (img_path, mask_path) in enumerate(zip(train_images, train_masks)):
        train_index.append({
            'index': i,
            'image_path': str(img_path),
            'mask_path': str(mask_path),
            'image_size': (INPUT_HEIGHT, INPUT_WIDTH, 3),
            'mask_size': (INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES)
        })
    
    val_index = []
    for i, (img_path, mask_path) in enumerate(zip(val_images, val_masks)):
        val_index.append({
            'index': i,
            'image_path': str(img_path),
            'mask_path': str(mask_path),
            'image_size': (INPUT_HEIGHT, INPUT_WIDTH, 3),
            'mask_size': (INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES)
        })
    
    # Save index files (very small, just metadata)
    train_index_file = os.path.join(output_dir, 'train_index.npy')
    val_index_file = os.path.join(output_dir, 'val_index.npy')
    
    np.save(train_index_file, train_index)
    np.save(val_index_file, val_index)
    
    print(f"Saved train index: {len(train_index)} samples")
    print(f"Saved val index: {len(val_index)} samples")
    print(f"Index files size: ~{os.path.getsize(train_index_file) / 1024:.1f} KB each")
    
    return len(train_index), len(val_index)

def create_small_cache_sample(cityscapes_root, output_dir, num_samples=50):
    """
    Create a small cache of a few samples for quick testing.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_images, train_masks, val_images, val_masks = prepare_data(cityscapes_root)
    
    print(f"Creating small cache with {num_samples} samples...")
    
    # Take first few samples from each set
    cache_train_images = train_images[:num_samples]
    cache_train_masks = train_masks[:num_samples]
    cache_val_images = val_images[:num_samples//3]  # 10 validation samples (30/3)
    cache_val_masks = val_masks[:num_samples//3]
    
    def process_small_batch(image_paths, mask_paths, split_name):
        processed_images = []
        processed_masks = []
        
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            print(f"Processing {split_name} sample {i+1}/{len(image_paths)}...")
            
            try:
                # Load and process image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32) / 255.0
                
                # Load and process mask
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                
                mask = cv2.resize(mask, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
                mask_one_hot = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES), dtype=np.float32)
                np.put_along_axis(mask_one_hot, mask[..., np.newaxis], 1, axis=-1)
                
                processed_images.append(img)
                processed_masks.append(mask_one_hot)
                
                # Memory cleanup
                del img, mask, mask_one_hot
                gc.collect()
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        return np.array(processed_images), np.array(processed_masks)
    
    # Process training cache
    print("Processing training cache...")
    train_imgs, train_msks = process_small_batch(cache_train_images, cache_train_masks, 'train')
    
    # Process validation cache
    print("Processing validation cache...")
    val_imgs, val_msks = process_small_batch(cache_val_images, cache_val_masks, 'val')
    
    # Save cache files
    cache_train_img_file = os.path.join(output_dir, 'cache_train_images.npy')
    cache_train_mask_file = os.path.join(output_dir, 'cache_train_masks.npy')
    cache_val_img_file = os.path.join(output_dir, 'cache_val_images.npy')
    cache_val_mask_file = os.path.join(output_dir, 'cache_val_masks.npy')
    
    np.save(cache_train_img_file, train_imgs)
    np.save(cache_train_mask_file, train_msks)
    np.save(cache_val_img_file, val_imgs)
    np.save(cache_val_mask_file, val_msks)
    
    print(f"Cache created successfully!")
    print(f"Training cache: {len(train_imgs)} samples")
    print(f"Validation cache: {len(val_imgs)} samples")
    print(f"Cache size: ~{os.path.getsize(cache_train_img_file) / 1024 / 1024:.1f} MB each")

def create_augmented_dataset(cityscapes_root, output_dir, augmentations_per_image=5):
    """
    Process images one by one, apply augmentations, save to individual files,
    then combine them at the end.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_images, train_masks, val_images, val_masks = prepare_data(cityscapes_root)
    
    print(f"Creating augmented dataset with {augmentations_per_image} augmentations per image...")
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    # Create augmentation pipeline
    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.RandomGamma(gamma_limit=(90, 110), p=0.2),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.15),
    ])
    
    def process_single_image_with_augmentations(img_path, mask_path, sample_idx, split_name, aug_count):
        """Process a single image with multiple augmentations"""
        try:
            # Load original image and mask
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                return None, None
            
            # Resize
            img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
            
            # Apply augmentations
            augmented_images = []
            augmented_masks = []
            
            # Original image (no augmentation)
            img_orig = img.astype(np.float32) / 255.0
            mask_one_hot = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES), dtype=np.float32)
            mask_one_hot[np.arange(INPUT_HEIGHT)[:, None], np.arange(INPUT_WIDTH), mask] = 1.0
            
            augmented_images.append(img_orig)
            augmented_masks.append(mask_one_hot)
            
            # Apply augmentations
            for aug_idx in range(aug_count - 1):  # -1 because we already have original
                augmented = aug(image=img, mask=mask)
                aug_img = augmented['image'].astype(np.float32) / 255.0
                aug_mask = augmented['mask']
                
                # Convert mask to one-hot
                aug_mask_one_hot = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES), dtype=np.float32)
                aug_mask_one_hot[np.arange(INPUT_HEIGHT)[:, None], np.arange(INPUT_WIDTH), aug_mask] = 1.0
                
                augmented_images.append(aug_img)
                augmented_masks.append(aug_mask_one_hot)
            
            # Save individual files
            for i, (aug_img, aug_mask) in enumerate(zip(augmented_images, augmented_masks)):
                img_filename = f"{split_name}_img_{sample_idx:04d}_aug_{i:02d}.npy"
                mask_filename = f"{split_name}_mask_{sample_idx:04d}_aug_{i:02d}.npy"
                
                img_filepath = os.path.join(output_dir, img_filename)
                mask_filepath = os.path.join(output_dir, mask_filename)
                
                np.save(img_filepath, aug_img)
                np.save(mask_filepath, aug_mask)
            
            return len(augmented_images), len(augmented_masks)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None, None
    
    # Process training images one by one
    print("Processing training images...")
    train_total_images = 0
    train_total_masks = 0
    
    for i, (img_path, mask_path) in enumerate(zip(train_images, train_masks)):
        print(f"Processing train image {i+1}/{len(train_images)}...")
        
        img_count, mask_count = process_single_image_with_augmentations(
            img_path, mask_path, i, 'train', augmentations_per_image
        )
        
        if img_count is not None:
            train_total_images += img_count
            train_total_masks += mask_count
        
        # Memory cleanup
        gc.collect()
        
        # Show progress every 100 images
        if (i + 1) % 100 == 0:
            process = psutil.Process()
            print(f"  Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
            print(f"  Processed: {i+1}/{len(train_images)} images")
    
    # Process validation images one by one
    print("Processing validation images...")
    val_total_images = 0
    val_total_masks = 0
    
    for i, (img_path, mask_path) in enumerate(zip(val_images, val_masks)):
        print(f"Processing val image {i+1}/{len(val_images)}...")
        
        img_count, mask_count = process_single_image_with_augmentations(
            img_path, mask_path, i, 'val', augmentations_per_image
        )
        
        if img_count is not None:
            val_total_images += img_count
            val_total_masks += mask_count
        
        # Memory cleanup
        gc.collect()
    
    print("=" * 60)
    print("AUGMENTED DATASET CREATED!")
    print("=" * 60)
    print(f"Training images: {train_total_images}")
    print(f"Training masks: {train_total_masks}")
    print(f"Validation images: {val_total_images}")
    print(f"Validation masks: {val_total_masks}")
    print(f"Augmentations per image: {augmentations_per_image}")
    print(f"Files saved to: {output_dir}")
    print("=" * 60)
    print("Next: Combine individual files into large arrays")
    print("=" * 60)
    
    return train_total_images, val_total_images

if __name__ == "__main__":
    cityscapes_root = '/data'
    output_dir = 'models'
    
    print("=" * 60)
    print("OPTIMIZED DATA PREPARATION")
    print("=" * 60)
    print("Method: Create data index for on-the-fly loading")
    print("Training will use 10% random subset each epoch")
    print("=" * 60)
    
    # Clean up any previous preprocessing files
    cleanup_files(output_dir)
    
    # Create optimized index (very small files)
    train_count, val_count = create_optimized_data_index(cityscapes_root, output_dir)
    
    print("=" * 60)
    print("PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"Training samples indexed: {train_count}")
    print(f"Validation samples indexed: {val_count}")
    print(f"Storage used: < 1 MB")
    print(f"Training will use 10% random subset each epoch")
    print("=" * 60)
    print("Ready to start training with on-the-fly loading!")
    print("=" * 60) 