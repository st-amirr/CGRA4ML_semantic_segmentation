# Full Project Report: LMIINet Semantic Segmentation on CGRA4ML

Date: 2025-09-07
Workspace: hls4ml_SS (branch: CGRA4ML)
Target OS/Shell: Windows, PowerShell

## Executive summary

This project trains a lightweight, CGRA4ML-friendly semantic segmentation model (LMIINet) with quantization-aware training (QAT) and builds a CGRA accelerator using the CGRA4ML/DeepSoCFlow toolchain. The training script produces a QKeras model that is then converted to a hardware project via a Python builder that emits SystemVerilog/TCL artifacts and optional Vivado synthesis/simulation flows. The end-to-end pipeline is reproducible on Windows with PowerShell and Docker.

Highlights
- Model: LMIINet, 512×1024 input, 19 Cityscapes trainIds, aux supervision, QAT-ready.
- Quantization: 8-bit activations/weights (QKeras); BN folding for QAT weight transfer.
- Training: Combined loss (CE+Dice) with aux λ=0.3, poly LR schedule, strong augmentation.
- Build: Generates config_hw.svh/config_hw.tcl, Vivado scripts, reports; updates RTL/TCL includes.
- Hardware: Tunable PE array and frequency, I/O-optimized defaults for ZCU104 with 8-bit datapaths.

What this report provides
- A single, accurate reference that merges and supersedes prior reports in the `reports/` folder.
- How everything fits together: training → export → hardware build → validation/synthesis.
- Concrete run instructions and troubleshooting for Windows/PowerShell.


## System at a glance
- Inputs: Cityscapes images (RGB) and masks mapped to trainIds (0..18, 255=ignore)
- Outputs: Per-pixel softmax over 19 classes, optional aux head for training
- Success criteria: High mIoU with strict parameter/GFLOPs budget, CGRA4ML-compatible ops
- Tooling: TensorFlow/Keras, QKeras, Albumentations, CGRA4ML, DeepSoCFlow, Vivado/Verilator (optional)


## Repository map (relevant)
- `trainQAT_CGRA4ML.py` — Train LMIINet (float/QAT), aux loss, callbacks, export final model
- `build_cgra4ml_model.py` — Convert QAT .h5 to CGRA4ML project; generate config/TCL; synth/sim (optional)
- `cgra4ml/`, `deepsocflow/` — Framework sources used by the build
- `models/` — Checkpoints and final .h5 models
- `cgra4ml_models/` — Generated CGRA4ML build outputs (timestamped)
- `reports/` — Documentation (this consolidated report replaces per-topic fragments)
- `tools/print_model_summary.py` — Utility to print Keras/QKeras model summaries by folder


## Training pipeline (trainQAT_CGRA4ML.py)

### Model architecture (CGRA4ML-friendly)
- Stem + Encoder: Lightweight Feature Interaction Bottlenecks (LFIB) and Feature Enhancement (FE) blocks
- Context: Improved Flatten Transformer = FLAM + CAB + CC (implemented with convs/SE-like ops)
- Decoder: Nearest-neighbor UpSampling2D + refinement (DW + PW) + FE, with streamlined skip fusions
- Segmentation head: Multi-scale fuse of top decoder features → 1×1 conv → softmax (19 classes)
- Auxiliary output: Supervision at reduced scale; combined into total loss with λ=0.3
- All ops are selected to map cleanly to CGRA4ML: Conv2D/DepthwiseConv2D, BN, ReLU, Add, Multiply, Concatenate, UpSampling2D, GAP/GMP, Reshape

### Quantization (QKeras)
- Activations/weights: 8-bit (quantized_relu, quantized_bits) with fractional representation
- Bias: 8–16 bit (builder uses 16 for hardware accumulations)
- QAT support: Utility folds float BN into Conv before transferring to a QAT-constructed model
- Two modes: float32 baseline or QAT; both produce the same graph structure where possible

### Data pipeline
- Dataset: Cityscapes (mapped to 19 trainIds; 255 as ignore)
- Generator: Yields images in [0,1], int32 masks; provides main (full-res) and aux (1/16) masks
- Augmentations (albumentations): crop/flip/rotate/affine, brightness/contrast, color jitter, HSV, gamma, noise, blur, rain/shadow, coarse dropout (probabilities tuned conservatively)

### Losses and metrics
- Combined loss = (CE + Dice)main + λ × (CE + Dice)aux; pixels with label 255 are masked out
- Metrics: pixel accuracy (ignore 255), mean IoU (per-class IoU averaged, ignoring absent classes)

### Training schedule, callbacks, and exports
- LR: Polynomial decay over E epochs; optional ReduceLROnPlateau kept as fallback
- Stability: Optional BN freeze cycles; optional decoder freeze cycles
- Monitoring: TensorBoard scalars/histograms, periodic validation visualizations colorized with Cityscapes palette
- Checkpoints: Weights-only per epoch; resume supported by filename epoch parsing
- Export: Final single-output inference model (.h5) without aux head and dropout disabled

### Typical runs (PowerShell)
- Baseline (float32)
```powershell
Set-Location -Path "C:\Users\stamir\Desktop\hls4ml_SS";
docker-compose exec hls4ml-training python3 /workspace/trainQAT_CGRA4ML.py --trainQAT=False --epochs=100 --batch_size=2 --subset_ratio=0.9 --LR=1e-4
```
- QAT training from baseline weights
```powershell
Set-Location -Path "C:\Users\stamir\Desktop\hls4ml_SS";
docker-compose exec hls4ml-training python3 /workspace/trainQAT_CGRA4ML.py --trainQAT=True --epochs=200 --batch_size=2 --subset_ratio=0.9 --LR=1e-4 --freeze_bn=True --load_model /workspace/models/cgra4ml_lmiinet_baseline_checkpoint_epoch_81.weights.h5
```
- TensorBoard
```powershell
Set-Location -Path "C:\Users\stamir\Desktop\hls4ml_SS";
docker-compose exec hls4ml-training tensorboard --logdir "/app/models/tb_logs" --host 0.0.0.0 --port 6006
```


## Model summary utility
To print a Keras/QKeras model summary from a folder, use the included tool.

- Script: `tools/print_model_summary.py`
- Behavior: probes for `model.h5`, then SavedModel under `saved_model/`, then any `*.h5`
- QKeras custom_objects auto-registered if package is installed

PowerShell
```powershell
# Using the project’s venv, if desired
C:\Users\stamir\Desktop\hls4ml_SS\cgra4ml_env\Scripts\python.exe tools\print_model_summary.py --model_dir .\models\some_model_dir --json
```


## Build pipeline (build_cgra4ml_model.py)

Purpose
- Loads a trained QKeras .h5 model
- Verifies structure and skip connections
- Creates a CGRA4ML hardware config and exports SystemVerilog/TCL artifacts
- Optionally runs Vivado synthesis and verification simulation (Verilator/xsim)

Core flow
1) Load QAT model (with QKeras custom_objects); log shapes and params
2) Analyze skip connections by traversing Keras graph
3) Create minimal config (PE grid, frequency, data width, skip strategy)
4) Convert/export: instantiate Hardware with I/O-optimized settings; emit config headers and Vivado scripts; export inference vectors (best-effort)
5) Optional validate: compare predictions, write `validation_results.json`
6) Predict performance (DeepSoCFlow if available, else heuristic fallback)
7) Generate performance report JSON and deployment guide
8) Rewrite include paths:
   - Update `deepsocflow/rtl/defines.svh` to include the newly generated `config_hw.svh`
   - Update `deepsocflow/tcl/fpga/zcu104.tcl` to set `MODEL_DIR` to the latest output and add the absolute `config_hw.svh`
9) Optional: run synthesis and/or simulation

CLI (key arguments)
- `--model_path`: required, .h5 only
- `--output_dir`: default `./cgra4ml_models`
- `--part`: default `xczu7ev-ffvc1156-2-e` (ZCU104)
- `--freq_mhz`: default 200
- `--pe_rows`, `--pe_cols`: default 16×64 (fit-friendly starting point)
- `--data_width`: default 16 (data path width in config; weights/activations are 8-bit in hardware)
- `--skip_connection_strategy`: memory_buffered|streaming|hybrid (default memory_buffered)
- `--validate`, `--synthesize`, `--simulate`, `--impl`, `--verbose`

PowerShell examples
- Build + synth-only on ZCU104 Tcl
```powershell
python build_cgra4ml_model.py --model_path .\models\cgra4ml_lmiinet_qat_final.h5 --output_dir .\cgra4ml_models --synthesize --pe_rows 16 --pe_cols 64 --freq_mhz 200
```
- Build + validate (prediction compare) without synthesis
```powershell
python build_cgra4ml_model.py --model_path .\models\cgra4ml_lmiinet_qat_final.h5 --output_dir .\cgra4ml_models --validate
```
- Direct Vivado batch (from generated folder)
```powershell
cd .\cgra4ml_models\lmiinet_cgra4ml_<timestamp>
vivado -mode batch -source vivado_flow.tcl
```

Generated artifacts (in `cgra4ml_models/lmiinet_cgra4ml_<timestamp>/`)
- `config_hw.svh`, `config_hw.tcl`, `config_tb.svh`: hardware configuration
- `vivado_flow.tcl`: batch Vivado flow; ZCU104 synth-only Tcl updated and used by default
- `sources.txt`: reference list of RTL sources
- `cgra4ml_performance_report.json`: performance and config metadata
- `CGRA4ML_DEPLOYMENT_GUIDE.md`: quick how-to
- Logs: `vivado.log`, `vivado.jou`; plus project directories under `my_project/` when using `vivado_flow.tcl`

I/O optimization defaults (why synth fits better)
- Narrower AXI/header widths, reduced burst length
- Capped max channels/kernel size/bundles
- Tighter RAM depths to reduce address pin pressure


## Known synthesis results and fit guidance
- A very large PE grid (e.g., 16×192 @ 250 MHz) over-utilizes mid-range parts (e.g., Kintex-7 xc7k70t): LUTs and BRAM exceed device capacity.
- For ZCU104 (xczu7ev), start with 16×64 @ ~200 MHz; iterate using `--pe_rows/--pe_cols/--freq_mhz` until timing/utilization are acceptable.
- If pin utilization is high, prefer the provided I/O-optimized defaults (AXI=64–128, header=32, smaller RAM depths, smaller bundles).


## Performance and reporting
- The builder writes `cgra4ml_performance_report.json` describing:
  - Hardware configuration, tool versions, and performance predictions
  - Optional simulation results merged post-verify
- When DeepSoCFlow predictors are unavailable, a conservative heuristic estimation is recorded.
- For accurate results, enable `--simulate` (requires Verilator or Vivado xsim) after a successful export.


## Troubleshooting
- “Cannot load model (.keras)”: convert to or save as `.h5`; the builder currently requires HDF5.
- Missing QKeras on host: install `qkeras` in the active Python environment for both training and report tooling.
- `export_inference()` fails: XModel-based path may be required; hardware export still emits config/TCL, so Vivado synthesis can proceed.
- Pathing on Windows: the builder normalizes include paths to forward slashes for Verilog/Tcl.
- Vivado not found: install Vivado and ensure it’s on PATH; otherwise skip synthesis.
- Over-utilization: reduce PE grid and/or frequency; consider smaller AXI/header widths and memory depths.


## Optimization recommendations (optional)
Model-level
- Reduce segmentation head channels if it dominates MACs without hurting mIoU (e.g., lower `seghead_filters`).
- Consider modest encoder/decoder filter scaling (−20–30%) if GFLOPs pressure is high.
- Keep nearest-neighbor upsampling and DW+PW refinement for CGRA-friendliness.

Hardware-level
- Start with 16×64 @ 200 MHz; downscale to 12×48 or 8×64 if timing/utilization misses.
- Bias precision of 16 bits is a safe default with 8-bit activations/weights for accumulators.
- Maintain narrowed AXI/header widths and smaller RAM depths for I/O pin reductions.


## End-to-end “Try it” on Windows/PowerShell
1) Train (float → QAT)
```powershell
# Float32
Set-Location -Path "C:\Users\stamir\Desktop\hls4ml_SS";
docker-compose exec hls4ml-training python3 /workspace/trainQAT_CGRA4ML.py --trainQAT=False --epochs=60 --batch_size=2 --LR=1e-4;

# QAT from best float checkpoint
Set-Location -Path "C:\Users\stamir\Desktop\hls4ml_SS";
docker-compose exec hls4ml-training python3 /workspace/trainQAT_CGRA4ML.py --trainQAT=True --epochs=120 --batch_size=2 --LR=1e-4 --freeze_bn=True --load_model /workspace/models/cgra4ml_lmiinet_baseline_checkpoint_epoch_60.weights.h5
```
2) Build hardware project
```powershell
python build_cgra4ml_model.py --model_path .\models\cgra4ml_lmiinet_qat_final.h5 --output_dir .\cgra4ml_models --pe_rows 16 --pe_cols 64 --freq_mhz 200 --synthesize
```
3) Inspect report and iterate PE/freq if needed
```powershell
Get-Content .\cgra4ml_models\lmiinet_cgra4ml_*\cgra4ml_performance_report.json
```
4) (Optional) Validate conversion
```powershell
python build_cgra4ml_model.py --model_path .\models\cgra4ml_lmiinet_qat_final.h5 --output_dir .\cgra4ml_models --validate
```


## Appendix A — CGRA4ML vs HLS4ML (compat highlights)
- CGRA4ML advantages
  - Single reconfigurable engine with strong support for skip connections and streaming dataflow
  - Tunable processing elements and custom PEs for segmentation ops; SoC-friendly integration
- HLS4ML considerations
  - Layer-per-engine approach can struggle with many skip connections and large feature maps
  - Ensure only supported layers/quantizers are used (e.g., prefer quantized_bits/quantized_relu over *_po2 variants)


## Appendix B — Files and responsibilities
- Training
  - `trainQAT_CGRA4ML.py`: model creation, data generator, losses/metrics, callbacks, QAT folding utility, export
  - Outputs: checkpoints in `models/`, final `*_final.h5`
- Build
  - `build_cgra4ml_model.py`: load .h5, analyze, configure hardware, export SV/TCL, optional validate/sim/synth, generate reports
  - Outputs: `cgra4ml_models/lmiinet_cgra4ml_<ts>/` with configs, TCL, logs, JSON, and deployment guide
- Utilities
  - `tools/print_model_summary.py`: print summaries for Keras/QKeras models given a directory


---
Completion summary
- Consolidated all prior reports into this single reference aligned with the current scripts.
- Included step-by-step training and build flows, Windows-friendly commands, artifacts mapping, and troubleshooting.
- Provided pragmatic optimization guidance for model/hardware iteration until fit/timing goals are met.
