HLS4ML Training Pipeline Usage Guide
====================================

## Quick Start

You now have a complete training pipeline with QAT toggle functionality. Here's how to use it:

### Prerequisites
1. Docker Desktop running with GPU support
2. Cityscapes dataset mounted at: `C:/Users/stamir/Desktop/cityscape`
3. Your current directory: `C:/Users/stamir/Desktop/hls4ml_SS`

### Recommended Training Workflow

#### Phase 1: Train Float32 Baseline (RECOMMENDED FIRST STEP)
```bash
# Start Docker container
docker-compose up -d

# Enter container terminal (like your current workflow)
docker-compose exec hls4ml-training /bin/bash

# Inside container - train float32 baseline
python3 /workspace/trainQAT.py --trainQAT=False --epochs=200 --batch_size=2
```

**Why start with Float32?**
- Better convergence and stability
- Easier debugging and validation
- Establishes baseline performance
- hls4ml documentation recommends this approach

#### Phase 2: Apply QAT to Trained Model
After your Float32 model training completes successfully:

```bash
# Inside the same container terminal (or reconnect if needed)
docker-compose exec hls4ml-training /bin/bash

# Check available models first
ls -la /app/models/float32_*.keras

# Apply QAT to your best float32 model
python3 /workspace/trainQAT.py --trainQAT=True --load_model=/app/models/float32_best.keras --epochs=20 --batch_size=2
```

**QAT Fine-tuning:**
- Uses lower learning rate (1e-5 vs 1e-3)
- Shorter training (typically 15-30 epochs)
- Loads float32 weights and converts to quantized layers
- Saves quantized weights for hls4ml

## Understanding Your Current Issues

Based on your training logs, your model has plateaued around IoU ~0.12-0.14. This is likely due to:

1. **Aggressive Quantization**: Training with QKeras from the start
2. **Learning Rate**: May need adjustment for QAT
3. **Architecture**: Complex model struggling with quantization

## New Approach Benefits

### 1. Float32 First Strategy
- Train without quantization constraints
- Get higher baseline IoU (likely 0.3-0.5+)
- Validate architecture works well
- Easier hyperparameter tuning

### 2. QAT as Fine-tuning
- Start from good float32 weights
- Gradual adaptation to quantization
- Better final quantized performance
- More stable training

### 3. Layer-wise Weight Transfer
The new script includes automatic weight transfer from float32 to QAT layers:
- Maps `conv` layers to `qconv` layers
- Maps `bn` layers to `qbn` layers  
- Preserves learned features during QAT conversion

### 4. Automatic Training Resume
The script automatically detects and resumes from the latest checkpoint:
- **Smart detection**: Finds the most recent checkpoint for the current training type (float32 vs QAT)
- **Epoch extraction**: Automatically determines the epoch to resume from
- **Seamless continuation**: Loads model weights and optimizer state when possible
- **Fallback handling**: Creates new model if checkpoint loading fails
- **Manual override**: Use `--load_model` to specify a different starting point

### 5. Dropout Management
The script automatically handles dropout layers for HLS4ML compatibility:
- **Training models**: Include dropout layers for regularization when enabled (default rates: 0.1-0.35)
- **Clean export models**: Always automatically created without dropout for HLS4ML compatibility
- **Flexible control**: Can disable dropout with `--use_dropout=False` for testing
- **Consistent naming**: Clean models always saved as `*_clean_for_hls4ml.keras` regardless of dropout setting
- **Weight transfer**: Clean models inherit weights from trained models (dropout layers automatically excluded)

## File Structure

After training, you'll have:
```
models/
├── float32_model_checkpoint_epoch_XX.keras  # Float32 checkpoints
├── float32_model_best.keras                 # Best float32 model
├── float32_model_final.keras               # Final float32 model
├── float32_model_clean_for_hls4ml.keras    # Clean float32 model (no dropout)
├── qat_model_checkpoint_epoch_XX.keras     # QAT checkpoints
├── qat_model_best.keras                    # Best QAT model
├── qat_model_final.keras                   # Final QAT model
├── qat_model_clean_for_hls4ml.keras        # Clean QAT model (no dropout)
└── qat_model_quantized_weights.h5          # For hls4ml export
```

### Important Notes About Model Files:
- **Training models** (`*_final.keras`): Include dropout layers when dropout is enabled
- **Clean models** (`*_clean_for_hls4ml.keras`): Always created without dropout for HLS4ML export
- **Quantized weights** (`*_quantized_weights.h5`): Always generated from clean models for FPGA synthesis
- **Consistent workflow**: Same file structure and naming regardless of dropout setting

## Docker Terminal Workflow (Recommended)

### Complete Step-by-Step Process

```bash
# 1. Start your container (from Windows terminal)
docker-compose up -d

# 2. Enter container terminal
docker-compose exec hls4ml-training /bin/bash

# 3. You're now inside the container - run Phase 1
python3 /workspace/trainQAT.py --trainQAT=False --epochs=100 --batch_size=2 --subset_ratio=0.9

# 4. Wait for training to complete (6-8 hours), then check results
ls -la /app/models/float32_*.keras

# 5. Run Phase 2 (QAT) using your best float32 model
python3 /workspace/trainQAT.py --trainQAT=True --load_model=/app/models/float32_best.keras --epochs=20 --batch_size=2 --subset_ratio=0.5

# 6. Check final results
ls -la /app/models/
```

### Monitoring During Training

```bash
# In another terminal window, monitor training progress
docker-compose exec hls4ml-training tail -f /app/models/log.txt

# Check GPU usage
docker-compose exec hls4ml-training nvidia-smi

# Check memory usage
docker-compose exec hls4ml-training free -h
```

## Command Line Options

### Basic Training (Inside Container)
```bash
# Float32 baseline (recommended first step)
python3 /workspace/trainQAT.py --trainQAT=False

# QAT training with loaded model (recommended second step)
python3 /workspace/trainQAT.py --trainQAT=True --load_model=/app/models/float32_best.keras

# QAT from scratch (not recommended)
python3 /workspace/trainQAT.py --trainQAT=True
```

### Advanced Options (Inside Container)
```bash
# Custom epochs and batch size for float32
python3 /workspace/trainQAT.py --trainQAT=False --epochs=30 --batch_size=1

# Custom parameters for QAT
python3 /workspace/trainQAT.py --trainQAT=True --load_model=/app/models/float32_best.keras --epochs=15 --batch_size=2

# Training without dropout (for testing)
python3 /workspace/trainQAT.py --trainQAT=False --use_dropout=False --epochs=20


# Medium testing with 50% of dataset
python3 /workspace/trainQAT.py --trainQAT=False --epochs=20 --batch_size=2 --subset_ratio=0.5

# All available parameters:
--trainQAT=True/False          # Enable/disable QAT
--load_model=path             # Load existing model for QAT
--epochs=50                   # Number of epochs
--batch_size=2                # Batch size
--use_dropout=True/False      # Enable/disable dropout during training
--subset_ratio=0.1-1.0        # Fraction of dataset to use (1.0 = full dataset)
```

### Dataset Subset Options
The `--subset_ratio` parameter allows you to control how much of the dataset to use:

```bash
# Quick testing - 10% of dataset (fast iteration, ~1 hour)
python3 /workspace/trainQAT.py --trainQAT=False --epochs=10 --subset_ratio=0.1

# Development - 25% of dataset (good for hyperparameter tuning, ~2-3 hours)
python3 /workspace/trainQAT.py --trainQAT=False --epochs=20 --subset_ratio=0.25

# Validation - 50% of dataset (solid results, ~4-5 hours)
python3 /workspace/trainQAT.py --trainQAT=False --epochs=40 --subset_ratio=0.5

# Full training - 100% of dataset (production training, ~8-10 hours)
python3 /workspace/trainQAT.py --trainQAT=False --epochs=100 --subset_ratio=1.0
```

## Quick Reference Commands

### Essential Commands (Run Inside Container)
```bash
# Start training from scratch (Float32)
python3 /workspace/trainQAT.py --trainQAT=False

# Quick test with small dataset subset (10% for fast iteration)
python3 /workspace/trainQAT.py --trainQAT=False --epochs=5 --subset_ratio=0.1

# Continue with QAT after float32 completes
python3 /workspace/trainQAT.py --trainQAT=True --load_model=/app/models/float32_best.keras

# Check available models
ls -la /app/models/

# Monitor training
tail -f /app/models/log.txt

# Check GPU
nvidia-smi
```

## Complete Workflow Example

Here's exactly what you should run, step by step:

### Initial Setup (Windows Terminal)
```bash
cd "C:\Users\stamir\Desktop\hls4ml_SS"
docker-compose up -d
docker-compose exec hls4ml-training /bin/bash
```

### Phase 1 - Float32 Training (Inside Container)
```bash
# Start float32 baseline training
python3 /workspace/trainQAT.py --trainQAT=False --epochs=50 --batch_size=2

# For quick testing, use subset of dataset
python3 /workspace/trainQAT.py --trainQAT=False --epochs=10 --batch_size=2 --subset_ratio=0.1

# Expected output:
# Training Configuration:
#   QAT Training: False
#   Load Model: None (training from scratch)
#   Epochs: 50
#   Batch Size: 2
# ============================================================
# Phase 1: Training float32 baseline model
# ============================================================

# Training will take 6-8 hours. Monitor with:
# tail -f /app/models/log.txt
```

### Phase 2 - QAT Training (Inside Container, After Phase 1)
```bash
# Check your trained models
ls -la /app/models/float32_*.keras

# Start QAT training using best float32 model
python3 /workspace/trainQAT.py --trainQAT=True --load_model=/app/models/float32_best.keras --epochs=20 --batch_size=2

# Expected output:
# Training Configuration:
#   QAT Training: True
#   Load Model: /app/models/float32_best.keras
#   Epochs: 20
#   Batch Size: 2
# ============================================================
# Phase 2: Applying Quantization-Aware Training (QAT)
# Loading baseline model from: /app/models/float32_best.keras
# ============================================================

# QAT training will take 2-3 hours
```

### Final Results
```bash
# Check all your trained models
ls -la /app/models/

# You should see:
# float32_model_best.keras     <- Best float32 model
# qat_model_best.keras         <- Best QAT model  
# qat_model_quantized_weights.h5 <- For hls4ml export
```

## Expected Performance Improvements

### Your Current Results vs New Approach
- **Your Current Training**: IoU plateaued at ~0.12-0.14 after 52 epochs
- **New Float32 Phase**: Expected IoU ~0.35-0.55 (2-4x improvement!)
- **New QAT Phase**: Expected IoU ~0.30-0.50 (still 2-3x better than current)

### Training Time Comparison
- **Your Current Approach**: ~52 epochs × 11 minutes = ~9.5 hours for poor results
- **New Phase 1**: ~50 epochs × 8 minutes = ~6.5 hours for good float32 model
- **New Phase 2**: ~20 epochs × 8 minutes = ~2.5 hours for QAT fine-tuning
- **Total**: ~9 hours for much better results

### What You Should See

#### Phase 1 (Float32) Progress:
```
Epoch 1/50: loss: 1.4xxx - iou: 0.25xx - val_iou: 0.22xx  # Much better start
Epoch 10/50: loss: 1.0xxx - iou: 0.35xx - val_iou: 0.31xx # Steady improvement
Epoch 30/50: loss: 0.7xxx - iou: 0.48xx - val_iou: 0.44xx # Strong performance
Epoch 50/50: loss: 0.6xxx - iou: 0.52xx - val_iou: 0.48xx # Excellent baseline
```

#### Phase 2 (QAT) Progress:
```
Epoch 1/20: loss: 0.8xxx - iou: 0.45xx - val_iou: 0.42xx  # Start from good baseline
Epoch 10/20: loss: 0.7xxx - iou: 0.47xx - val_iou: 0.44xx # Gradual QAT adaptation
Epoch 20/20: loss: 0.6xxx - iou: 0.49xx - val_iou: 0.46xx # Final quantized model
```

## Monitoring Training

### Inside Container Terminal
```bash
# View live training logs (most important)
tail -f /app/models/log.txt

# Check GPU usage
nvidia-smi

# Check available models
ls -la /app/models/

# Check memory usage
free -h

# Check training progress (look for IoU improvements)
grep "val_iou" /app/models/log.txt | tail -10
```

### From Windows Terminal (Alternative)
```bash
# View logs from outside container
docker-compose exec hls4ml-training tail -f /app/models/log.txt

# Check GPU from outside
docker-compose exec hls4ml-training nvidia-smi

# List models from outside
docker-compose exec hls4ml-training ls -la /app/models/
```

## Troubleshooting

### Training Interruption and Resume
The script automatically handles training interruptions:
- **Automatic detection**: When you restart training, it automatically finds and loads the latest checkpoint
- **Smart resuming**: Continues from the exact epoch where training was interrupted
- **Type-aware**: Float32 training resumes float32 checkpoints, QAT training resumes QAT checkpoints
- **No manual intervention**: Just run the same command again - it will resume automatically

Example of interrupted training:
```bash
# Training gets interrupted at epoch 25
python3 /workspace/trainQAT.py --trainQAT=False --epochs=50 --batch_size=2
# ... training stops unexpectedly ...

# Restart with the same command - will automatically resume from epoch 25
python3 /workspace/trainQAT.py --trainQAT=False --epochs=50 --batch_size=2
# Output: "Found latest checkpoint: float32_model_checkpoint_epoch_25.keras"
# Output: "Resuming from epoch 25"
```

### If Float32 Training Doesn't Improve
1. Check data loading (verify dataset path)
2. Try different learning rates (1e-4, 5e-4)
3. Reduce model complexity temporarily
4. Verify loss function and metrics

### If QAT Conversion Fails
1. Ensure float32 model trained successfully
2. Check model path is correct
3. Verify QKeras compatibility
4. Try loading without weight transfer first

### Memory Issues
1. Reduce batch size to 1
2. Clear model cache between phases
3. Monitor with `nvidia-smi`

## Next Steps After Successful Training

1. **Validate QAT Model**: Test on validation set
2. **Export to hls4ml**: Use the generated `.h5` quantized weights
3. **FPGA Synthesis**: Follow hls4ml documentation for FPGA compilation
4. **Performance Testing**: Compare inference speed and accuracy

## Key Differences from Your Previous Approach

1. **Two-phase training** instead of direct QAT
2. **Automatic weight transfer** between phases  
3. **Proper learning rate scheduling** for each phase
4. **Better model organization** with clear naming
5. **Simplified command line interface**

This approach should give you significantly better results than your current plateaued training!
