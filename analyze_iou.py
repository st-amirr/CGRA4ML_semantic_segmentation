#!/usr/bin/env python3
"""
Analyze IoU divergence across classes from TensorBoard .v2 files
"""

import os
import struct
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def analyze_iou_divergence():
    """Analyze IoU progression across all classes from TensorBoard logs"""
    
    tb_log_dir = '/workspace/models/tb_logs/train'
    
    # Find all event files
    event_files = []
    for file in os.listdir(tb_log_dir):
        if file.startswith('events.out.tfevents'):
            event_files.append(os.path.join(tb_log_dir, file))
    
    if not event_files:
        print("❌ No TensorBoard event files found")
        return
    
    print(f"📁 Found {len(event_files)} TensorBoard event files")
    
    # Use the latest event file
    latest_event_file = max(event_files, key=os.path.getmtime)
    print(f"📊 Analyzing: {os.path.basename(latest_event_file)}")
    
    # Load event accumulator
    ea = event_accumulator.EventAccumulator(latest_event_file)
    ea.Reload()
    
    # Get all tensor tags
    tags = ea.Tags()['tensors']
    iou_tags = [tag for tag in tags if tag.startswith('epoch_IoU_Classes/')]
    iou_tags.sort()
    
    print("="*80)
    print("IoU DIVERGENCE ANALYSIS FROM TENSORBOARD LOGS")
    print("="*80)
    
    # Analyze each class
    class_data = {}
    for tag in iou_tags:
        class_name = tag.split('/')[-1]
        events = ea.Tensors(tag)
        
        if events:
            # Extract values from tensor events
            values = []
            epochs = []
            for event in events:
                epoch = int(event.step)
                # Decode tensor content
                if hasattr(event.tensor_proto, 'tensor_content') and event.tensor_proto.tensor_content:
                    try:
                        value = struct.unpack('f', event.tensor_proto.tensor_content[:4])[0]
                        values.append(value)
                        epochs.append(epoch)
                    except:
                        continue
            
            if values:
                class_data[class_name] = {
                    'epochs': epochs,
                    'values': values,
                    'initial': values[0],
                    'final': values[-1],
                    'max': max(values),
                    'min': min(values),
                    'improvement': values[-1] - values[0]
                }
                
                print(f"\n{class_name.upper()}:")
                print(f"  Initial IoU: {values[0]:.4f}")
                print(f"  Final IoU:   {values[-1]:.4f}")
                print(f"  Max IoU:     {max(values):.4f}")
                print(f"  Min IoU:     {min(values):.4f}")
                print(f"  Improvement:  {values[-1] - values[0]:+.4f}")
                print(f"  Epochs:      {len(values)} (from {epochs[0]} to {epochs[-1]})")
                
                # Check for dramatic drops
                if values[-1] < values[0] * 0.5:
                    print(f"  ⚠️  WARNING: Dramatic drop detected!")
                elif values[-1] < 0.1:
                    print(f"  ❌ CRITICAL: Very low final IoU!")
                elif values[-1] > 0.8:
                    print(f"  ✅ EXCELLENT: High final IoU")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if class_data:
        final_ious = [data['final'] for data in class_data.values()]
        print(f"Mean final IoU: {np.mean(final_ious):.4f}")
        print(f"Std final IoU:  {np.std(final_ious):.4f}")
        print(f"Min final IoU:  {min(final_ious):.4f}")
        print(f"Max final IoU:  {max(final_ious):.4f}")
        
        # Classes with very low IoU
        low_iou_classes = [name for name, data in class_data.items() if data['final'] < 0.1]
        if low_iou_classes:
            print(f"\n❌ Classes with very low IoU (< 0.1): {', '.join(low_iou_classes)}")
        
        # Classes with very high IoU
        high_iou_classes = [name for name, data in class_data.items() if data['final'] > 0.8]
        if high_iou_classes:
            print(f"✅ Classes with high IoU (> 0.8): {', '.join(high_iou_classes)}")
        
        # Classes with dramatic drops
        dramatic_drops = [name for name, data in class_data.items() if data['final'] < data['initial'] * 0.5]
        if dramatic_drops:
            print(f"⚠️  Classes with dramatic drops (>50% decrease): {', '.join(dramatic_drops)}")
        
        # Show progression for problematic classes
        print(f"\n📊 DETAILED ANALYSIS OF PROBLEMATIC CLASSES:")
        for name, data in class_data.items():
            if data['final'] < 0.1 or data['final'] < data['initial'] * 0.5:
                print(f"\n{name.upper()} - IoU Progression:")
                for i, (epoch, value) in enumerate(zip(data['epochs'], data['values'])):
                    if i % 5 == 0 or i == len(data['values']) - 1:  # Show every 5th epoch and last
                        print(f"  Epoch {epoch}: {value:.4f}")

if __name__ == "__main__":
    analyze_iou_divergence()
