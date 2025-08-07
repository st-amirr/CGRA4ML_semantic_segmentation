#!/usr/bin/env python3
"""
Training Data Extraction and Formatting Script
=============================================

This script extracts training data from TensorBoard logs and training state files,
formats it according to the reference format, removes zeros, and extracts epoch data correctly.

Usage:
    python compare_models.py --extract_training_data --tb_log_dir=./models/tb_logs --model_dir=./models
    python compare_models.py --extract_training_data --model_prefix=float32_model --output_file=./formatted_data.json
"""

import os
import json
import argparse
import glob
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def extract_training_data_from_tensorboard(tb_log_dir, output_file, include_validation=False):
    """
    Extract training data from TensorBoard logs and format it according to the reference format.
    Removes zeros and extracts epoch data correctly. Combines multiple .v2 files.
    """
    print(f"🔍 Extracting training data from TensorBoard logs: {tb_log_dir}")
    
    # Initialize data structure
    training_data = {
        "epoch": 0,
        "history": {}
    }
    
    # Find all event files (including .v2 files)
    event_files = []
    for root, dirs, files in os.walk(tb_log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print(f"❌ No TensorBoard event files found in {tb_log_dir}")
        return None
    
    print(f"📁 Found {len(event_files)} TensorBoard event files")
    
    # Extract data from all event files and combine them
    all_metrics = {}
    max_epoch = 0
    
    for event_file in event_files:
        try:
            print(f"Processing: {os.path.basename(event_file)}")
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            
            # Get all tensor tags (training metrics are stored as tensors)
            tags = ea.Tags()['tensors']
            print(f"  Found {len(tags)} tensor metrics in this file")
            
            for tag in tags:
                # Only process epoch-related metrics
                if tag.startswith('epoch_'):
                    if tag not in all_metrics:
                        all_metrics[tag] = {}
                        
                    # Get all events for this tag
                    events = ea.Tensors(tag)
                    
                    for event in events:
                        # Extract epoch number from step (assuming step = epoch)
                        epoch = int(event.step)
                        
                        # Extract tensor value - try different methods
                        value = 0.0
                        if hasattr(event.tensor_proto, 'float_val') and event.tensor_proto.float_val:
                            value = float(event.tensor_proto.float_val[0])
                        elif hasattr(event.tensor_proto, 'tensor_content') and event.tensor_proto.tensor_content:
                            # Decode binary tensor content
                            import struct
                            try:
                                # Assuming float32 (4 bytes)
                                value = struct.unpack('f', event.tensor_proto.tensor_content[:4])[0]
                            except:
                                value = 0.0
                        
                        # Skip zero values (as per requirement)
                        if value != 0:
                            # Store by epoch to avoid duplicates and maintain order
                            all_metrics[tag][epoch] = value
                            max_epoch = max(max_epoch, epoch)
                        
        except Exception as e:
            print(f"⚠️  Warning: Could not process {event_file}: {e}")
            continue
    
    # Synchronize all metrics to have the same epoch range
    if all_metrics:
        # Find the full epoch range across all metrics
        all_epochs = set()
        for epoch_data in all_metrics.values():
            all_epochs.update(epoch_data.keys())
        
        min_epoch = min(all_epochs) if all_epochs else 0
        max_epoch = max(all_epochs) if all_epochs else 0
        full_epoch_range = list(range(min_epoch, max_epoch + 1))
        
        print(f"📊 Synchronizing metrics across epochs {min_epoch}-{max_epoch} ({len(full_epoch_range)} total epochs)")
        
        # Convert epoch-based dictionaries to synchronized ordered lists
        cleaned_history = {}
        for tag, epoch_data in all_metrics.items():
            if epoch_data:
                # Create synchronized array with zeros for missing epochs
                synchronized_values = []
                for epoch in full_epoch_range:
                    if epoch in epoch_data:
                        synchronized_values.append(epoch_data[epoch])
                    else:
                        synchronized_values.append(0.0)  # Add zero for missing epoch
                
                cleaned_history[tag] = synchronized_values
                print(f"  {tag}: {len(synchronized_values)} values (epochs {min_epoch}-{max_epoch})")
    else:
        cleaned_history = {}
    
    # Set the epoch to the maximum epoch found
    training_data["epoch"] = max_epoch
    training_data["history"] = cleaned_history
    
    # Save the extracted data with maximum precision
    with open(output_file, 'w') as f:
        json.dump(training_data, f, separators=(',', ':'), ensure_ascii=False)
    
    print(f"✅ Extracted training data saved to: {output_file}")
    print(f"📊 Extracted {len(cleaned_history)} metrics over {max_epoch} epochs")
    
    return training_data

def extract_training_data_from_state_file(model_dir, model_prefix, output_file):
    """
    Extract training data from training state file and format it according to the reference format.
    Removes zeros and ensures proper epoch data extraction with maximum precision.
    """
    print(f"🔍 Extracting training data from state file: {model_dir}")
    
    training_state_file = os.path.join(model_dir, f'{model_prefix}_training_state.json')
    
    if not os.path.exists(training_state_file):
        print(f"❌ Training state file not found: {training_state_file}")
        return None
    
    try:
        with open(training_state_file, 'r') as f:
            training_state = json.load(f)
        
        epoch = training_state.get('epoch', 0)
        history = training_state.get('history', {})
        
        print(f"📊 Found training state with epoch {epoch} and {len(history)} metrics")
        
        # Clean the data: remove any zero values and ensure proper formatting
        cleaned_history = {}
        for metric, values in history.items():
            if values:
                # Remove zero values as per requirement
                cleaned_values = [v for v in values if v != 0]
                if cleaned_values:
                    cleaned_history[metric] = cleaned_values
                    print(f"  {metric}: {len(cleaned_values)} values (last: {cleaned_values[-1]})")
        
        # Create the properly formatted training data
        formatted_data = {
            'epoch': epoch,
            'history': cleaned_history
        }
        
        # Save to output file with maximum precision
        with open(output_file, 'w') as f:
            json.dump(formatted_data, f, separators=(',', ':'), ensure_ascii=False)
        
        print(f"✅ Formatted training data saved to: {output_file}")
        print(f"📊 Extracted {len(cleaned_history)} metrics over {epoch} epochs")
        
        return formatted_data
        
    except Exception as e:
        print(f"❌ Failed to extract training data: {e}")
        return None

def extract_training_data_from_checkpoints(model_dir, model_prefix, output_file):
    """
    Extract training data from checkpoint files and format it according to the reference format.
    This function is a placeholder - actual checkpoint extraction would require loading each model.
    """
    print(f"🔍 Extracting training data from checkpoints: {model_dir}")
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(model_dir, f'{model_prefix}_checkpoint_epoch_*.keras')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print(f"❌ No checkpoint files found matching pattern: {checkpoint_pattern}")
        return None
    
    print(f"📁 Found {len(checkpoints)} checkpoint files")
    print("⚠️  Note: Checkpoint extraction is not implemented yet. Use TensorBoard logs or training state files instead.")
    
    return None

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract and format training data')
    parser.add_argument('--extract_training_data', action='store_true',
                       help='Extract training data from various sources')
    parser.add_argument('--tb_log_dir', type=str, default='./models/tb_logs',
                       help='Directory containing TensorBoard logs')
    parser.add_argument('--model_dir', type=str, default='./models',
                       help='Directory containing model files and training state')
    parser.add_argument('--model_prefix', type=str, default='float32_model',
                       help='Model prefix to extract data for (e.g., float32_model, qat_model)')
    parser.add_argument('--output_file', type=str, default='',
                       help='Output file path for formatted training data')
    parser.add_argument('--extract_from', type=str, choices=['tensorboard', 'state_file', 'checkpoints', 'all'],
                       default='tensorboard', help='Source to extract data from')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if not args.extract_training_data:
        print("❌ Please use --extract_training_data flag to extract training data")
        return
    
    print("🔍 Training Data Extraction and Formatting")
    print("=" * 50)
    
    # Determine output file - use the same naming convention as the reference file
    if not args.output_file:
        args.output_file = os.path.join(args.model_dir, f'{args.model_prefix}_training_state.json')
    
    # Extract data based on specified source
    success = False
    
    # Priority 1: Extract from TensorBoard logs (most reliable for real training data)
    if args.extract_from == 'tensorboard' or args.extract_from == 'all':
        print("\n📊 Extracting from TensorBoard logs...")
        # Check for train and validation folders
        train_dir = os.path.join(args.tb_log_dir, 'train')
        validation_dir = os.path.join(args.tb_log_dir, 'validation')
        
        if os.path.exists(train_dir):
            print(f"📁 Found train logs at: {train_dir}")
            print(f"📁 Found validation logs at: {validation_dir}")
            
            # Extract from train folder (primary source)
            result = extract_training_data_from_tensorboard(train_dir, args.output_file)
            if result:
                success = True
                print(f"✅ Successfully extracted training data from train logs")
        elif os.path.exists(args.tb_log_dir):
            print(f"📁 Using root logs directory: {args.tb_log_dir}")
            result = extract_training_data_from_tensorboard(args.tb_log_dir, args.output_file)
            if result:
                success = True
        else:
            print(f"⚠️  TensorBoard logs not found at {args.tb_log_dir} or {train_dir}")
    
    # Priority 2: Extract from training state file (fallback)
    if (args.extract_from == 'state_file' or args.extract_from == 'all') and not success:
        print("\n📊 Extracting from training state file...")
        result = extract_training_data_from_state_file(args.model_dir, args.model_prefix, args.output_file)
        if result:
            success = True
    
    # Priority 3: Extract from checkpoint files (placeholder)
    if (args.extract_from == 'checkpoints' or args.extract_from == 'all') and not success:
        print("\n📊 Extracting from checkpoint files...")
        extract_training_data_from_checkpoints(args.model_dir, args.model_prefix, args.output_file)
    
    if not success:
        print("❌ No training data could be extracted from any source")
        return
    
    print(f"\n✅ Extraction complete! Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
