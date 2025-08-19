# LMIINet FPGA Deployment Guide

## Model Information
- **Model Type**: Quantized (QAT)
- **Input Shape**: 512x1024x3
- **Output Shape**: 512x1024x20
- **FPGA Part**: xczu7ev-ffvc1156-2-e
- **Clock Period**: 5.0ns
- **I/O Type**: io_stream
- **Strategy**: Latency
- **Reuse Factor**: 2

## Generated Files
The following files were generated in: ./fpga_models/optimized_qat/lmiinet_qat_1755618160

### Core Files
- `myproject.cpp` - Main HLS implementation
- `myproject.h` - Header file
- `myproject_test.cpp` - Testbench
- `myproject_tb.cpp` - Testbench implementation

### Project Files
- `myproject_prj/` - Vivado HLS project directory
- `myproject_prj/solution1/` - Synthesis solution
- `myproject_prj/solution1/syn/` - Synthesis reports
- `myproject_prj/solution1/impl/` - Implementation files

## Usage Instructions

### 1. Simulation
```bash
cd ./fpga_models/optimized_qat/lmiinet_qat_1755618160
./myproject
```

### 2. Synthesis
```bash
cd ./fpga_models/optimized_qat/lmiinet_qat_1755618160
vivado_hls -f run_hls.tcl
```

### 3. Co-simulation
```bash
cd ./fpga_models/optimized_qat/lmiinet_qat_1755618160
./myproject
```

### 4. IP Export
The synthesized IP core can be found in:
`./fpga_models/optimized_qat/lmiinet_qat_1755618160/myproject_prj/solution1/impl/ip/`

## Integration with Vivado
1. Open Vivado and create a new project
2. Add the generated IP core to your project
3. Instantiate the IP core in your design
4. Connect input/output ports as needed
5. Generate bitstream

## Performance Notes
- The model is optimized for Latency strategy
- Reuse factor: 2
- Expected latency and resource usage can be found in synthesis reports

## Troubleshooting
- Check synthesis logs in `myproject_prj/solution1/syn/report/`
- Verify timing constraints are met
- Ensure sufficient FPGA resources are available
- Check I/O timing for your specific application

## Support
For issues with HLS4ML conversion, check:
- HLS4ML documentation: https://hls4ml.readthedocs.io/
- Model compatibility with supported layers
- FPGA resource constraints
- Timing requirements
