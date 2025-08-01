import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from qkeras import QActivation, QConv2D, QBatchNormalization
from train import create_segmentation_model, MODEL_CONFIG, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES

def analyze_parameter(param, name=""):
    """Analyze any parameter array and return detailed statistics"""
    stats = {
        'min': float(np.min(param)),
        'max': float(np.max(param)),
        'mean': float(np.mean(param)),
        'std': float(np.std(param)),
        'abs_mean': float(np.mean(np.abs(param))),
        'sparsity': float(np.sum(np.abs(param) < 1e-6) / param.size),
        'shape': param.shape
    }
    
    range_size = stats['max'] - stats['min']
    precision_need = stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else stats['std']
    
    print(f"\n=== {name} ===")
    print(f"Shape: {stats['shape']}")
    print(f"Value Range: [{stats['min']:.6f}, {stats['max']:.6f}] (size: {range_size:.6f})")
    print(f"Mean: {stats['mean']:.6f} (abs: {stats['abs_mean']:.6f})")
    print(f"Std Dev: {stats['std']:.6f}")
    print(f"Sparsity: {stats['sparsity']*100:.2f}% near-zero values")
    print(f"Precision Need: {precision_need:.6f}")
    
    # Determine recommended bit width
    if range_size > 1000 or precision_need < 0.0001:
        print("Recommendation: Use 32-bit precision (large range or high precision need)")
    elif range_size > 100 or precision_need < 0.001:
        print("Recommendation: Use 16-bit precision (moderate range or precision need)")
    else:
        print("Recommendation: 8-bit precision is sufficient")
    
    # Histogram analysis
    hist, bins = np.histogram(param.flatten(), bins=50)
    print("\nValue Distribution:")
    max_bar_width = 40
    max_count = np.max(hist)
    for count, bin_start, bin_end in zip(hist, bins[:-1], bins[1:]):
        bar_width = int((count / max_count) * max_bar_width)
        print(f"[{bin_start:6.3f}, {bin_end:6.3f}] {'#' * bar_width}")
    
    return stats, range_size, precision_need

def analyze_layer_weights(layer, name=""):
    """Analyze weights of a layer and print detailed statistics"""
    weights = layer.get_weights()
    if len(weights) == 0:
        return
    
    # Analyze kernel weights
    kernel = weights[0]
    stats, range_size, precision_need = analyze_parameter(kernel, f"Layer: {name} (Kernel)")
    
    print(f"\n=== Layer: {name} ===")
    print(f"Shape: {stats['shape']}")
    print(f"Value Range: [{stats['min']:.6f}, {stats['max']:.6f}] (size: {range_size:.6f})")
    print(f"Mean: {stats['mean']:.6f} (abs: {stats['abs_mean']:.6f})")
    print(f"Std Dev: {stats['std']:.6f}")
    print(f"Sparsity: {stats['sparsity']*100:.2f}% near-zero weights")
    print(f"Precision Need: {precision_need:.6f}")
    
    # Determine recommended bit width
    if range_size > 1000 or precision_need < 0.0001:
        print("Recommendation: Use 32-bit precision (large range or high precision need)")
    elif range_size > 100 or precision_need < 0.001:
        print("Recommendation: Use 16-bit precision (moderate range or precision need)")
    else:
        print("Recommendation: 8-bit precision is sufficient")
    
    # Histogram analysis for weight distribution
    hist, bins = np.histogram(kernel.flatten(), bins=50)
    print("\nWeight Distribution:")
    max_bar_width = 40
    max_count = np.max(hist)
    for count, bin_start, bin_end in zip(hist, bins[:-1], bins[1:]):
        bar_width = int((count / max_count) * max_bar_width)
        print(f"[{bin_start:6.3f}, {bin_end:6.3f}] {'#' * bar_width}")

def main():
    print("Creating model architecture...")
    model = create_segmentation_model()
    
    print("\nLoading weights...")
    # Try to load the best model first, fall back to latest checkpoint if not found
    try:
        model.load_weights('models/best_model.keras')
        print("Loaded weights from best_model.keras")
    except:
        try:
            model.load_weights('models/best_model.h5')
            print("Loaded weights from best_model.h5")
        except:
            print("Could not find best model, looking for latest checkpoint...")
            # Find the latest checkpoint
            import glob
            checkpoints = glob.glob('models/checkpoint_epoch_*.keras') + glob.glob('models/checkpoint_epoch_*.h5')
            if checkpoints:
                latest = max(checkpoints)
                model.load_weights(latest)
                print(f"Loaded weights from {latest}")
            else:
                raise Exception("No model weights found!")

    print("\nAnalyzing model architecture and weights...")
    print(f"Total layers: {len(model.layers)}")
    
    # Group layers by architectural blocks
    blocks = {
        'input': [],
        'encoder1': [],
        'encoder2': [],
        'encoder3': [],
        'bridge': [],
        'decoder3': [],
        'decoder2': [],
        'decoder1': [],
        'decoder0': [],
        'output': []
    }
    
    # Categorize layers by their names
    for layer in model.layers:
        if any(isinstance(layer, t) for t in [QConv2D, QBatchNormalization, QActivation]):
            if 'init_conv' in layer.name or 'downsample1' in layer.name:
                blocks['input'].append(layer)
            elif 'encoder1' in layer.name:
                blocks['encoder1'].append(layer)
            elif 'encoder2' in layer.name:
                blocks['encoder2'].append(layer)
            elif 'encoder3' in layer.name:
                blocks['encoder3'].append(layer)
            elif 'bridge' in layer.name:
                blocks['bridge'].append(layer)
            elif 'decoder3' in layer.name:
                blocks['decoder3'].append(layer)
            elif 'decoder2' in layer.name:
                blocks['decoder2'].append(layer)
            elif 'decoder1' in layer.name:
                blocks['decoder1'].append(layer)
            elif 'decoder0' in layer.name or 'boundary_enhance' in layer.name or 'final' in layer.name:
                blocks['decoder0'].append(layer)
            elif 'output' in layer.name:
                blocks['output'].append(layer)
    
    print("\n=== Architecture Analysis ===")
    for block_name, layers in blocks.items():
        if not layers:
            continue
            
        print(f"\n=== {block_name.upper()} BLOCK ===")
        print(f"Total layers in block: {len(layers)}")
        
        # Analyze layers in this block
        for layer in layers:
            if isinstance(layer, QConv2D):
                analyze_layer_weights(layer, layer.name)
                if hasattr(layer, 'kernel_quantizer'):
                    print(f"\nQuantization settings:")
                    print(f"Kernel quantizer: {layer.kernel_quantizer}")
                    print(f"Bias quantizer: {layer.bias_quantizer}")
            
            elif isinstance(layer, QBatchNormalization):
                weights = layer.get_weights()
                if len(weights) == 4:
                    analyze_parameter(weights[0], f"Layer: {layer.name} (Gamma/Scale)")
                    analyze_parameter(weights[1], f"Layer: {layer.name} (Beta/Shift)")
                    analyze_parameter(weights[2], f"Layer: {layer.name} (Moving Mean)")
                    analyze_parameter(weights[3], f"Layer: {layer.name} (Moving Variance)")
                if hasattr(layer, 'gamma_quantizer'):
                    print(f"\nQuantization settings:")
                    print(f"Gamma quantizer: {layer.gamma_quantizer}")
                    print(f"Beta quantizer: {layer.beta_quantizer}")
                    print(f"Mean quantizer: {layer.mean_quantizer}")
                    print(f"Variance quantizer: {layer.variance_quantizer}")
            
            elif isinstance(layer, QActivation):
                print(f"\n=== Layer: {layer.name} ===")
                if hasattr(layer, 'quantizer'):
                    print("Activation Quantizer Settings:")
                    quantizer = layer.quantizer
                    print(f"Quantizer type: {type(quantizer).__name__}")
                    print(f"Configuration: {quantizer}")
                    if hasattr(quantizer, 'bits'):
                        print(f"Bits: {quantizer.bits}")
                        print(f"Value range: [-2^{quantizer.bits-1}, 2^{quantizer.bits-1}-1] (for signed)")
                    if hasattr(quantizer, 'integer'):
                        print(f"Integer bits: {quantizer.integer}")
                    if hasattr(quantizer, 'symmetric'):
                        print(f"Symmetric: {quantizer.symmetric}")
                    if hasattr(quantizer, 'alpha'):
                        print(f"Alpha: {quantizer.alpha}")
    
    print("\n=== Block-wise Memory Analysis ===")
    # Calculate block-wise memory requirements
    block_memory = {}
    for block_name, layers in blocks.items():
        block_params = sum(layer.count_params() for layer in layers)
        block_memory[block_name] = {
            '8bit': block_params,
            '16bit': block_params * 2,
            '32bit': block_params * 4
        }
    
    print("\nBlock-wise parameter distribution:")
    for block_name, memory in block_memory.items():
        if memory['8bit'] > 0:
            print(f"\n{block_name.upper()} Block:")
            print(f"Parameters: {memory['8bit']:,}")
            print(f"Memory usage:")
            print(f"- 8-bit:  {memory['8bit']/1024/1024:.3f} MB")
            print(f"- 16-bit: {memory['16bit']/1024/1024:.3f} MB")
            print(f"- 32-bit: {memory['32bit']/1024/1024:.3f} MB")
    
    # Total memory footprint analysis
    total_params = model.count_params()
    total_bytes_8bit = total_params
    total_bytes_16bit = total_params * 2
    total_bytes_32bit = total_params * 4
    
    print("\n=== Memory Footprint Analysis ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated memory usage:")
    print(f"- 8-bit quantization:  {total_bytes_8bit/1024/1024:.2f} MB")
    print(f"- 16-bit quantization: {total_bytes_16bit/1024/1024:.2f} MB")
    print(f"- 32-bit full precision: {total_bytes_32bit/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()
