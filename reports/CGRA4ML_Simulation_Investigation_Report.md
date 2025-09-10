# CGRA4ML Simulation Investigation Report

**Date:** September 10, 2025  
**Objective:** Execute CGRA4ML inference simulation with model `cgra4ml_lmiinet_qat_final.h5`  
**Command Requested:** `docker-compose exec hls4ml-training` with CGRA4ML model simulation  

## Executive Summary

This investigation revealed that while CGRA4ML model compilation and Verilator simulation setup can be completed successfully, the actual simulation execution has fundamental issues. The `build.py` script successfully generates all required configuration files and compiles a functional simulation executable, but the simulation itself fails to produce expected output files, indicating incomplete simulation execution.

## Technical Environment

- **Container:** Docker with hls4ml-training environment
- **Simulator:** Verilator 5.028 2024-08-21
- **Model:** cgra4ml_lmiinet_qat_final.h5 (QKeras quantized model)
- **Framework:** CGRA4ML with deepsocflow backend
- **Hardware Config:** 16x96 processing elements @ 200MHz, 8-bit precision

## Investigation Timeline & Findings

### 1. Initial Compilation Issues (Resolved ✅)

**Problem:** SystemVerilog compatibility failures with original Verilator 4.038
```
%Error: Unsupported: Packed array field reference requires SystemVerilog
```

**Solution:** Upgraded to Verilator 5.028, resolved SystemVerilog `Bundle_t` structure extensions

**Files Modified:**
- Verilator upgrade via package management
- SystemVerilog compatibility fixes in RTL modules

### 2. Missing Configuration Files (Resolved ✅)

**Problem:** `config_fw.h` not found during compilation
```
fatal error: config_fw.h: No such file or directory
```

**Solution:** Used proper `build.py` script instead of manual compilation approach

**Discovery:** The CGRA4ML framework requires a specific build workflow:
1. `export_inference()` - generates configuration files
2. `verify_inference()` - runs simulation with proper setup

### 3. Build Script Success (Partial ✅)

**Successful Execution:**
```bash
python build.py --input_weights ./models/cgra4ml_lmiinet_qat_final.h5 --simulate --verbose
```

**Generated Files:**
- ✅ `config_fw.h` (5,391 bytes) - Hardware configuration header
- ✅ `config_hw.svh` (1,729 bytes) - SystemVerilog hardware parameters  
- ✅ `Vaxi_sys_tb` (8.1MB executable) - Compiled simulation binary
- ✅ Input vectors: `*_x_sim.bin`, `*_w.txt`, `*_y_exp.txt`
- ✅ Expected outputs: `*_y_nhwc_exp.txt`

**Compilation Statistics:**
- 227 SystemVerilog/C++ files compiled
- Build time: ~10.999 seconds
- Executable size: 8,149,360 bytes
- VCD tracing enabled

### 4. Simulation Execution Failure (❌)

**Problem:** Simulation completes but produces no output files

**Expected:** `vectors/0_0_0_y_raw_sim.txt` and other `*_y_raw_sim.txt` files  
**Actual:** Files not generated, simulation exits immediately

**Diagnostics Performed:**
```bash
./Vaxi_sys_tb                    # Runs without output
ls -la *.vcd                     # No VCD trace files
ls -la vectors/*y_raw_sim*       # No simulation output files
```

### 5. Model Loading Issues (❌)

**Problem:** QKeras custom layers and loss functions not recognized
```
ValueError: Unknown layer: 'QConv2D'
ValueError: Unknown loss function: 'combined_loss'
```

**Attempted Solutions:**
- Added QKeras custom_objects scope
- Used weights-only loading approach
- Both approaches failed due to training-specific components

## Key Technical Discoveries

### 1. CGRA4ML Architecture Understanding

**Processing Elements:** 16 rows × 96 columns = 1,536 total PEs  
**Precision Configuration:**
- Input: 8-bit activations
- Weights: 8-bit quantized  
- Accumulator: 20-bit sum
- Bias: 20-bit

**Memory Architecture:**
- Bundle-based processing with 5 main computation bundles
- Each bundle handles different network layers
- Hardware utilization: 19.49% - 99.48% across bundles

### 2. Simulation Infrastructure

**Verilator Integration:**
- DPI-C interface for C/SystemVerilog communication
- VCD tracing capability (but no traces generated)
- Complex testbench structure with `axi_sys_tb.sv` top-level

**File Structure:**
```
/workspace/cgra4ml_models/lmiinet_cgra4ml_1757474660/
├── Vaxi_sys_tb*              # Main simulation executable
├── config_fw.h               # Hardware parameters
├── vectors/                  # Input/output data
│   ├── *_x_sim.bin          # Input tensors
│   ├── *_w.txt              # Weight matrices  
│   ├── *_y_exp.txt          # Expected outputs
│   └── [missing] *_y_raw_sim.txt  # Simulation results
└── build/                    # Verilator build artifacts
```

### 3. Performance Projections

**Theoretical Performance (if working):**
- Latency per frame: 1.23 ms
- Throughput: 814.97 FPS  
- GOPS: 5.90
- Efficiency: 0.5 (50% utilization)

## Root Cause Analysis

### Primary Issue: Simulation Execution Gap

The fundamental problem appears to be that while the Verilator compilation successfully creates a simulation executable, the actual simulation logic is not executing properly. Evidence:

1. **No VCD traces:** Despite VCD tracing being enabled during compilation
2. **No output files:** Expected `*_y_raw_sim.txt` files are never created
3. **Silent failure:** Executable runs and exits without error messages
4. **Missing DPI calls:** C simulation functions likely never invoked

### Secondary Issues

1. **Model Loading:** QKeras integration requires custom objects not easily accessible in verification context
2. **Documentation Gap:** Limited examples of end-to-end CGRA4ML simulation workflow
3. **Error Handling:** Simulation failures are not reported clearly

## Recommendations

### Immediate Actions

1. **Debug Simulation Execution:**
   - Add debug prints to C simulation code
   - Verify DPI-C function calls are being made
   - Check if SystemVerilog testbench is actually running

2. **Alternative Verification:**
   - Use Python-only verification via `verify_inference()` 
   - Create minimal test case with known-good model
   - Test with pre-built CGRA4ML examples

3. **Model Compatibility:**
   - Export model weights separately from training metadata
   - Use CGRA4ML's native model conversion tools
   - Test with simpler, non-QKeras models first

### Long-term Solutions

1. **Framework Integration:**
   - Improve error reporting in simulation pipeline
   - Add simulation progress indicators
   - Provide clearer documentation for end-users

2. **Testing Infrastructure:**
   - Create automated tests for simulation pipeline
   - Validate against known-good reference implementations
   - Add continuous integration for simulation workflows

## Conclusion

This investigation successfully demonstrated the CGRA4ML build pipeline up to simulation compilation, but revealed fundamental issues with simulation execution. The framework generates all necessary files and compiles successfully, but the actual inference simulation does not produce expected results.

**Status:** Compilation ✅ | Configuration ✅ | Simulation ❌

**Next Steps:** Focus on debugging the simulation execution gap, potentially through alternative verification methods or simpler test cases to isolate the root cause.

---

## Appendix: Commands Executed

### Successful Commands
```bash
# Verilator version check
verilator --version  # → 5.028

# Successful build
python build.py --input_weights ./models/cgra4ml_lmiinet_qat_final.h5 --simulate --verbose

# File verification  
ls -la /workspace/cgra4ml_models/lmiinet_cgra4ml_1757474660/
```

### Failed Commands
```bash
# Direct simulation execution
./Vaxi_sys_tb  # → No output files generated

# Python verification
python -c "verify_inference(...)"  # → Model loading failures
```

### File Artifacts
- **Build Output:** `/workspace/cgra4ml_models/lmiinet_cgra4ml_1757474660/`
- **Log Files:** Embedded in build.py output
- **Configuration:** `config_fw.h`, `config_hw.svh`, `hardware.json`
