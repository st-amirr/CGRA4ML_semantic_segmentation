# LMIINet Semantic Segmentation with CGRA4ML

This project implements LMIINet semantic segmentation using CGRA4ML (Coarse-Grained Reconfigurable Array for Machine Learning), a framework specifically designed for deep neural networks with skip connections and complex dataflow architectures.

## Why CGRA4ML for Semantic Segmentation?

### Key Advantages Over HLS4ML

1. **Skip Connection Optimization**: CGRA4ML is specifically designed to handle skip connections efficiently, which are the core of semantic segmentation architectures
2. **Single Reconfigurable Engine**: Unlike HLS4ML's layer-per-engine approach, CGRA4ML uses a single reconfigurable engine for better resource utilization
3. **Memory-Optimized**: Efficient handling of large feature maps and intermediate results
4. **Custom Processing Elements**: Tailored for semantic segmentation operations like feature fusion and upsampling

### Problem Solved

Traditional HLS4ML implementations struggle with:
- Complex skip connections in semantic segmentation
- Memory bottlenecks with large feature maps
- Dataflow scheduling issues
- Resource inefficiency with layer-per-engine approach

CGRA4ML addresses these issues through:
- Memory-buffered skip connections
- Streaming architecture for efficient data flow
- Custom processing elements for semantic segmentation operations
- Optimized memory management

## Project Structure

```
hls4ml_SS/
├── build_cgra4ml_model.py          # Main CGRA4ML model builder
├── requirements_cgra4ml.txt         # CGRA4ML dependencies
├── setup_cgra4ml.bat               # Windows setup script
├── CGRA4ML_README.md               # This file
├── reports/
│   └── cgra4ml_compatibility_report.txt  # Detailed compatibility analysis
├── cgra4ml_models/                 # Generated CGRA4ML models
└── models/                         # Input QAT models
```

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Git** for cloning CGRA4ML
3. **Hardware Tools**:
    - Verilator 5.014+ (for simulation)
    - Xilinx Vivado (for synthesis)
    - C++ compiler (GCC/Clang)

## Custom Board Integration (NEW)

### How to Add a Custom Board

You can now add and target custom FPGA boards for synthesis and deployment. This enables support for new devices beyond the default CGRA4ML boards.

#### Steps to Add a Custom Board:
1. **Edit the board database/config file**:
    - Add your board's part number, pin count, and resource specs to the supported boards list (e.g., `compatible_boards.csv` in `reports/`).
    - Example entry for Kintex-7:
      ```csv
      Part,I/O Pin Count,Available IOBs,LUT Elements,FlipFlops,Block RAMs,Ultra RAMs,DSPs,...
      xc7k70tfbv676-1,676,400,41000,82000,135,0,240,...
      ```
2. **Update synthesis scripts**:
    - Ensure your board part is referenced in TCL scripts (e.g., `vivado_flow.tcl`).
    - Example:
      ```tcl
      set_part xc7k70tfbv676-1
      ```
3. **Model build command**:
    - Use the `--board` argument (if available) or ensure your custom part is set in the build script.
    - Example:
      ```bash
      python build_cgra4ml_model.py --model_path ./models/cgra4ml_lmiinet_qat_final.h5 --output_dir ./cgra4ml_models --synthesize --board xc7k70tfbv676-1
      ```
4. **Resource validation**:
    - The build flow will check your board's resources against the model requirements and warn if there are mismatches.

#### Example: Adding Kintex-7 Board
- Added `xc7k70tfbv676-1` to `compatible_boards.csv` with full resource specs.
- Updated `vivado_flow.tcl` to use `set_part xc7k70tfbv676-1`.
- Ran synthesis and confirmed successful build and implementation for Kintex-7.

#### Notes
- You can add any Xilinx, Intel, or custom FPGA part supported by Vivado or your synthesis tool.
- For advanced boards, update both the CSV and TCL scripts for full compatibility.

**This project now supports custom board integration for flexible deployment!**

### Quick Setup (Windows)

```bash
# Run the automated setup script
setup_cgra4ml.bat
```

### Manual Setup

```bash
# 1. Create virtual environment
python -m venv cgra4ml_env
cgra4ml_env\Scripts\activate.bat

# 2. Install dependencies
pip install -r requirements_cgra4ml.txt

# 3. Clone and install CGRA4ML
git clone https://github.com/KastnerRG/cgra4ml.git
cd cgra4ml
pip install -e .
cd ..
```

## Usage

### Basic Model Conversion

```bash
# Convert QAT model to CGRA4ML
python build_cgra4ml_model.py \
    --model_path ./models/qat_model_clean_for_compatible.h5 \
    --output_dir ./cgra4ml_models \
    --validate
```

### Advanced Options

```bash
# Full synthesis with custom parameters
python build_cgra4ml_model.py \
    --model_path ./models/qat_model_clean_for_compatible.h5 \
    --output_dir ./cgra4ml_models \
    --data_width 16 \
    --skip_connection_strategy memory_buffered \
    --validate \
    --synthesize \
    --simulate
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to QAT model (.h5) | Required |
| `--output_dir` | Output directory | `./cgra4ml_models` |
| `--data_width` | Data width for quantization | `16` |
| `--skip_connection_strategy` | Skip connection implementation | `memory_buffered` |
| `--validate` | Validate model conversion | False |
| `--synthesize` | Run synthesis | False |
| `--simulate` | Run simulation | False |
| `--verbose` | Enable verbose output | False |

## Skip Connection Strategies

### 1. Memory Buffered (Recommended)
- Stores intermediate features in memory buffers
- Efficient for large feature maps
- Good balance of performance and resource usage

### 2. Streaming
- Processes data in streaming fashion
- Lower memory usage
- May have higher latency

### 3. Hybrid
- Combines memory buffering and streaming
- Optimized for specific architectures
- Requires careful tuning

## Custom Processing Elements

CGRA4ML implements custom processing elements for semantic segmentation:

### Skip Connection PE
- **Purpose**: Handles skip connections with memory buffering
- **Features**: Configurable buffer size, streaming I/O
- **Optimization**: Memory-efficient feature storage

### Feature Fusion PE
- **Purpose**: Performs feature fusion for skip connections
- **Features**: Element-wise addition, configurable precision
- **Optimization**: Optimized for semantic segmentation workloads

### Upsampling PE
- **Purpose**: Performs nearest neighbor upsampling
- **Features**: Configurable scale factors, interpolation methods
- **Optimization**: Hardware-optimized upsampling

## Performance Comparison

### Expected Improvements Over HLS4ML

| Metric | HLS4ML | CGRA4ML | Improvement |
|--------|--------|---------|-------------|
| Skip Connection Efficiency | Low | High | 3-5x |
| Memory Usage | High | Low | 40-60% |
| Resource Utilization | Low | High | 2-3x |
| Synthesis Success Rate | Variable | High | Consistent |
| Dataflow Optimization | Limited | Advanced | Significant |

### Resource Usage (Estimated)

| Resource | HLS4ML | CGRA4ML |
|----------|--------|---------|
| LUTs | ~150K | ~100K |
| BRAMs | ~200 | ~120 |
| DSPs | ~50 | ~40 |
| FFs | ~80K | ~60K |

## Troubleshooting

### Common Issues

1. **CGRA4ML Import Error**
   ```bash
   # Solution: Reinstall CGRA4ML
   cd cgra4ml
   pip install -e . --force-reinstall
   ```

2. **Synthesis Failures**
   - Check timing constraints
   - Verify FPGA resource availability
   - Review skip connection memory requirements

3. **Memory Issues**
   - Reduce data width
   - Adjust buffer sizes
   - Use streaming strategy

### Debug Commands

```bash
# Test CGRA4ML installation
python -c "import cgra4ml; print('CGRA4ML OK')"

# Check model structure
python build_cgra4ml_model.py --model_path model.h5 --verbose

# Validate conversion
python build_cgra4ml_model.py --model_path model.h5 --validate
```

## Integration with Existing Workflow

### Migration from HLS4ML

1. **Model Preparation**: Same QAT model format
2. **Configuration**: CGRA4ML-specific config
3. **Synthesis**: CGRA4ML synthesis flow
4. **Deployment**: Same FPGA deployment process

### Workflow Comparison

| Step | HLS4ML | CGRA4ML |
|------|--------|---------|
| Model Loading | ✅ | ✅ |
| Skip Connection Analysis | ❌ | ✅ |
| Custom PE Generation | ❌ | ✅ |
| Synthesis | Variable | ✅ |
| Validation | ✅ | ✅ |

## Advanced Configuration

### Memory Optimization

```python
config = {
    "memory": {
        "bram_usage": "optimize",
        "array_partition": True,
        "partition_factor": 2,
        "buffer_strategy": "double_buffering"
    }
}
```

### Performance Tuning

```python
config = {
    "performance": {
        "pipeline_depth": 10,
        "parallel_factor": 4,
        "dataflow": True,
        "optimization_level": 3
    }
}
```

## Future Enhancements

1. **ASIC Implementation**: Full ASIC flow support
2. **Runtime Reconfiguration**: Dynamic model switching
3. **Multi-Model Support**: Multiple models on single hardware
4. **Advanced Optimizations**: ML-based optimization

## Support and Resources

- **CGRA4ML Repository**: https://github.com/KastnerRG/cgra4ml
- **Documentation**: Available in CGRA4ML repository
- **Examples**: `/run/work/example.py` in CGRA4ML
- **Hardware Flow**: `/run/work/vivado_flow.tcl` in CGRA4ML

## License

This project uses CGRA4ML which is licensed under Apache-2.0.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Citation

If you use this implementation, please cite:

```bibtex
@misc{cgra4ml_lmiinet,
  title={LMIINet Semantic Segmentation with CGRA4ML},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/hls4ml_SS}
}
```
