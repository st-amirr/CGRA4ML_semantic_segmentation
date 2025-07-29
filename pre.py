import os
import numpy as np
import cv2
import gc
import tempfile
import shutil
from pathlib import Path
from train import INPUT_WIDTH, INPUT_HEIGHT, NUM_CLASSES, prepare_data

#  python3 /workspace/pre.py

def preprocess_and_save(cityscapes_root, output_dir, batch_size=20):
    """
    Ultra memory-efficient preprocessing that saves batches as temp files,
    clears RAM after each batch, then combines at the end.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_images, train_masks, val_images, val_masks = prepare_data(cityscapes_root)

    def process_split_with_temp_batches(image_paths, mask_paths, split_name, batch_size=20):
        """Process images in small batches, save as temp files, then combine"""
        total_samples = len(image_paths)
        processed_count = 0
        temp_files = []
        
        print(f"Processing {split_name} set ({total_samples} samples) in batches of {batch_size}...")
        
        # Process in small batches and save as temp files
        for batch_idx, start_idx in enumerate(range(0, total_samples, batch_size)):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_size_actual = end_idx - start_idx
            
            print(f"Processing batch {batch_idx + 1}/{(total_samples + batch_size - 1) // batch_size}...")
            
            # Create temporary arrays for this batch
            batch_images = np.zeros((batch_size_actual, INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.float32)
            batch_masks = np.zeros((batch_size_actual, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES), dtype=np.float32)
            
            # Process each image in the batch
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
                    
                    # Store in batch arrays
                    batch_images[i] = img
                    batch_masks[i] = mask_one_hot
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            # Save batch as temporary file
            temp_img_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
            temp_mask_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
            
            np.save(temp_img_file.name, batch_images)
            np.save(temp_mask_file.name, batch_masks)
            
            temp_files.append((temp_img_file.name, temp_mask_file.name, batch_size_actual))
            
            # Clear batch arrays and force garbage collection
            del batch_images, batch_masks
            gc.collect()
            
            print(f"Saved batch {batch_idx + 1} to temp files. Processed: {processed_count}/{total_samples}")
        
        # Combine all temporary files into final arrays using memory mapping
        print(f"Combining {len(temp_files)} temporary batch files...")
        
        final_img_file = os.path.join(output_dir, f'{split_name}_images.npy')
        final_mask_file = os.path.join(output_dir, f'{split_name}_masks.npy')
        
        # Create final arrays using memory mapping (never loaded into RAM)
        final_images = np.memmap(final_img_file, dtype=np.float32, mode='w+', 
                                shape=(processed_count, INPUT_HEIGHT, INPUT_WIDTH, 3))
        final_masks = np.memmap(final_mask_file, dtype=np.float32, mode='w+', 
                               shape=(processed_count, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES))
        
        # Copy data from temp files to final arrays using memory mapping
        current_idx = 0
        for temp_img_path, temp_mask_path, batch_size_actual in temp_files:
            # Load temp batch as memory-mapped (not into RAM)
            temp_images = np.load(temp_img_path, mmap_mode='r')
            temp_masks = np.load(temp_mask_path, mmap_mode='r')
            
            # Copy to final arrays (memory-mapped, so no RAM usage)
            final_images[current_idx:current_idx + batch_size_actual] = temp_images
            final_masks[current_idx:current_idx + batch_size_actual] = temp_masks
            
            current_idx += batch_size_actual
            
            # Clean up temp batch (just close the memory mapping)
            del temp_images, temp_masks
            gc.collect()
        
        # Flush and close final arrays
        final_images.flush()
        final_masks.flush()
        del final_images, final_masks
        gc.collect()
        
        # Clean up temporary files
        for temp_img_path, temp_mask_path, _ in temp_files:
            try:
                os.unlink(temp_img_path)
                os.unlink(temp_mask_path)
            except:
                pass
        
        print(f"Completed {split_name} set: {processed_count}/{total_samples} samples processed")
        return processed_count

    # Process training set
    print("=" * 50)
    print("PROCESSING TRAINING SET")
    print("=" * 50)
    train_count = process_split_with_temp_batches(train_images, train_masks, 'train', batch_size)
    
    # Clear training data from memory
    del train_images, train_masks
    gc.collect()
    
    # Process validation set
    print("=" * 50)
    print("PROCESSING VALIDATION SET")
    print("=" * 50)
    val_count = process_split_with_temp_batches(val_images, val_masks, 'val', batch_size)
    
    # Clear validation data from memory
    del val_images, val_masks
    gc.collect()
    
    print("=" * 50)
    print("PREPROCESSING COMPLETE!")
    print("=" * 50)
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Files saved to: {output_dir}")
    print(f"Memory usage should be minimal now.")

if __name__ == "__main__":
    cityscapes_root = '/data'  # Change if needed
    output_dir = 'models'      # Save preprocessed arrays here
    batch_size = 20           # Reduced batch size for lower memory usage
    
    print("Starting ultra memory-efficient Cityscapes preprocessing...")
    print(f"Batch size: {batch_size} (very small to minimize RAM usage)")
    print(f"Temporary files will be used to avoid loading everything into memory")
    preprocess_and_save(cityscapes_root, output_dir, batch_size) 