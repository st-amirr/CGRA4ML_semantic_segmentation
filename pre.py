import os
import numpy as np
import cv2
import gc
from pathlib import Path
from train import INPUT_WIDTH, INPUT_HEIGHT, NUM_CLASSES, prepare_data

def preprocess_and_save(cityscapes_root, output_dir, batch_size=50):
    """
    Preprocess Cityscapes dataset with memory-efficient processing.
    Processes images in batches to reduce RAM usage.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_images, train_masks, val_images, val_masks = prepare_data(cityscapes_root)

    def process_images_batch(image_paths, mask_paths, split_name, batch_size=50):
        """Process images in batches to reduce memory usage"""
        total_samples = len(image_paths)
        processed_count = 0
        
        # Calculate total size for pre-allocation
        sample_img_shape = (INPUT_HEIGHT, INPUT_WIDTH, 3)
        sample_mask_shape = (INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES)
        
        # Create memory-mapped files for large arrays
        img_file = os.path.join(output_dir, f'{split_name}_images.npy')
        mask_file = os.path.join(output_dir, f'{split_name}_masks.npy')
        
        # Pre-allocate arrays using memory mapping
        images = np.memmap(img_file, dtype=np.float32, mode='w+', 
                          shape=(total_samples,) + sample_img_shape)
        masks = np.memmap(mask_file, dtype=np.float32, mode='w+', 
                         shape=(total_samples,) + sample_mask_shape)
        
        print(f"Processing {split_name} set ({total_samples} samples) in batches of {batch_size}...")
        
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_size_actual = end_idx - start_idx
            
            # Process batch
            for i in range(batch_size_actual):
                idx = start_idx + i
                img_path = image_paths[idx]
                mask_path = mask_paths[idx]
                
                try:
                    # Load and process image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"Failed to load image: {img_path}")
                        continue
                    
                    img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
                    img = img.astype(np.float32) / 255.0
                    
                    # Load and process mask
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        print(f"Failed to load mask: {mask_path}")
                        continue
                    
                    mask = cv2.resize(mask, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    mask_one_hot = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES), dtype=np.float32)
                    np.put_along_axis(mask_one_hot, mask[..., np.newaxis], 1, axis=-1)
                    
                    # Store in memory-mapped arrays
                    images[idx] = img
                    masks[idx] = mask_one_hot
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            # Progress update
            print(f"Processed {processed_count}/{total_samples} {split_name} samples (batch {start_idx//batch_size + 1})")
            
            # Force garbage collection after each batch
            gc.collect()
        
        # Flush memory-mapped arrays to disk
        images.flush()
        masks.flush()
        del images, masks
        gc.collect()
        
        print(f"Completed {split_name} set: {processed_count}/{total_samples} samples processed")
        return processed_count

    # Process training set
    train_count = process_images_batch(train_images, train_masks, 'train', batch_size)
    
    # Process validation set
    val_count = process_images_batch(val_images, val_masks, 'val', batch_size)
    
    print(f"\nPreprocessing complete!")
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    cityscapes_root = '/data'  # Change if needed
    output_dir = 'models'      # Save preprocessed arrays here
    batch_size = 50           # Process 50 images at a time to control memory usage
    
    print("Starting Cityscapes preprocessing with memory optimization...")
    preprocess_and_save(cityscapes_root, output_dir, batch_size) 