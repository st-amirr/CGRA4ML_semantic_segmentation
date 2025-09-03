# CGRA4ML Build and Integration Report

Date: 2025-09-03
Target device: xc7k70tfbv676-1 (Kintex-7)
Host OS: Windows (PowerShell)

## Overview

This report captures how the LMIINet QAT model is exported and built for CGRA4ML, the roles and linkages between files (Python build script, DeepSoCFlow RTL/TCL, generated configs), and what Vivado synthesized. It also summarizes quantization settings, hardware configuration, and utilization results with guidance for fitting on xc7k70t.

## Model lifecycle and training context

- Training/inference architecture (high-level):
  - Two-output training (main + auxiliary) with a combined loss: Loss = Lmain + λ·Laux (λ≈0.3).
  - Dataset generator yields: X, [Y_main, Y_aux].
  - Final inference model is single-output; dropout and aux output disabled for export.
- Quantization: QKeras/QAT model with 8-bit weights/activations for CGRA4ML deployment.

Files involved during training/export (not exhaustive):
- `trainQAT_CGRA4ML.py` – builds the model (aux output for training, single output for inference export), exports final .h5.
- `models/` – contains produced checkpoints and the final `cgra4ml_lmiinet_qat_final.h5` used by the build.

## Build script anatomy (`build_cgra4ml_model.py`)

Entry point when invoked as:
- `python build_cgra4ml_model.py --model_path ./models/cgra4ml_lmiinet_qat_final.h5 --output_dir ./cgra4ml_models [--synthesize] [--validate]`

Key functions and responsibilities:
- `setup_cgra4ml_environment()`
  - Sets env for CGRA4ML and checks for `vivado`/`verilator`.
- `parse_arguments()`
  - Handles CLI (model path, output dir, FPGA part, data width, clock period, validate/synthesize/simulate flags, etc.).
- `load_qat_model(path)`
  - Loads the final Keras/QKeras .h5 with required custom objects.
- `verify_model_structure(model)`
  - Logs input/output shapes and total params; warns if mismatch vs expected.
- `analyze_skip_connections(model)`
  - Inspects Keras graph to list non-adjacent connections.
- `create_cgra4ml_config(args, skip_connections)`
  - Produces minimal metadata (quantization is set on Hardware directly).
- `convert_to_cgra4ml(model, config, output_dir)`
  - Core flow: creates `Hardware(...)`, calls `export()` and `export_vivado_tcl()`, tries `export_inference(...)`, optionally verification.
  - Writes CGRA4ML artifacts in the timestamped `output_dir`.
- `run_cgra4ml_synthesis(output_dir, args)`
  - `vivado -mode batch -source vivado_flow.tcl` from the output dir.
- `run_cgra4ml_simulation(...)` (unused in this flow)
- `generate_performance_report(...)` and `create_deployment_guide(...)`
  - Write `cgra4ml_performance_report.json` and `CGRA4ML_DEPLOYMENT_GUIDE.md`.
- `main()`
  - Drives the full pipeline; also rewrites pointers in DeepSoCFlow to reference the latest generated config.

Important in-script linkages:
- Defines include path rewrite:
  - Edits `deepsocflow/rtl/defines.svh` so ``\`include ".../config_hw.svh"`` points to the newest `output_dir/config_hw.svh` (absolute path with forward slashes).
- Vivado TCL rewrite:
  - Edits `deepsocflow/tcl/fpga/xc7k70tfbv676-1.tcl` to set `set MODEL_DIR` to the newest `output_dir` and ensures `add_files ".../config_hw.svh"` uses the absolute path.

Notes:
- `export_inference(model, hw, ...)` expects an XModel graph; with a plain Keras model it can fail. Hardware export and Vivado TCL generation are still valid and proceed.

## CGRA4ML and DeepSoCFlow artifacts

Artifacts in a build folder (example: `cgra4ml_models/lmiinet_cgra4ml_1756866341/`):
- `config_hw.svh` – Verilog/SV macros defining hardware limits and parameters
  - Example values from this run:
    - `ROWS 16`, `COLS 192` – PE grid size
    - `X_BITS 8`, `K_BITS 8`, `Y_BITS 20`, `Y_OUT_BITS 32`
    - Max constraints: `KH_MAX 9`, `KW_MAX 9`, `XH_MAX 512`, `XW_MAX 512`, `XN_MAX 64`, `CI_MAX 512`, `MAX_N_BUNDLES 64`
    - Memory sizing: `RAM_WEIGHTS_DEPTH 512`, `RAM_EDGES_DEPTH 3584`
    - Bus: `AXI_WIDTH 128`, header, baseaddr, burst len
- `config_hw.tcl` – TCL with matching params for Vivado
- `config_tb.svh` – testbench-related config
- `vivado_flow.tcl` – top-level non-project batch flow used by the builder
- `sources.txt` – source list for reference
- `CGRA4ML_DEPLOYMENT_GUIDE.md` – quick how-to
- `cgra4ml_performance_report.json` – metadata
- `vivado.log`, `vivado.jou` – run log/journal
- `my_project/` – Vivado run dir; includes `synth_1/axi_cgra4ml_utilization_synth.rpt`

DeepSoCFlow static sources referenced by Vivado (subset):
- `deepsocflow/rtl/` – SystemVerilog/Verilog for the CGRA fabric/top, DMA, AXI interfaces, etc.
  - `axi_cgra4ml.v`, `dnn_engine.v`, `proc_engine.sv`, `axis_*`, `ram.sv`, ext cores, etc.
  - `rtl/defines.svh` – includes the generated `config_hw.svh` so macros are visible across RTL.
    - Robustness note: a fallback `BITS_KW2` macro was added (via previous fix) if missing in generated config: compute from `KW_MAX`.
- `deepsocflow/tcl/fpga/xc7k70tfbv676-1.tcl` – project creation and source addition for this Kintex-7 part. The builder updates `MODEL_DIR` and the `add_files` path for `config_hw.svh`.

## Quantization and precision

- The builder sets quantization via `Hardware(...)`:
  - `bits_input=8`, `bits_weights=8` – 8-bit data path (matching CGRA4ML typical usage)
  - Summation/accumulation: `bits_sum=20`, `bits_bias=16` by default in the script
- The minimal `config` dict returned by `create_cgra4ml_config()` is metadata only; bitwidths are controlled by the `Hardware` constructor.

## Current hardware configuration used

From the checked-in `build_cgra4ml_model.py` at the time of the run, `convert_to_cgra4ml()` creates:
- `Hardware(processing_elements=(16, 192), frequency_mhz=250, bits_input=8, bits_weights=8, ...)`
- This aligns with the generated `config_hw.svh` showing `ROWS=16`, `COLS=192`, and explains the large utilization (see below).

Tip: If targeting xc7k70t, reduce the grid (e.g., `(8,24)`) and/or clock to fit device resources.

## Vivado synthesis results (post-synthesis utilization)

From `my_project/my_project.runs/synth_1/axi_cgra4ml_utilization_synth.rpt` and `vivado.log`:
- Slice LUTs: 406,148 used / 41,000 available → ~990.6% (over-utilized)
- Slice Registers (FF): 375,593 used / 82,000 available → ~458% (over-utilized)
- Block RAM Tiles: 135 used / 135 available → 100% (fully used)
  - RAMB18E1: 254; RAMB36E1: 8
- DSPs: 0 used / 240 available

Interpretation:
- The PE grid and/or intermediate buffering generate far more logic and BRAM than the xc7k70t can host. Numbers can decrease a bit after `opt_design` and impl, but not by an order of magnitude.

## How generated files link together

- `build_cgra4ml_model.py`
  - Generates `config_hw.svh/tcl`, `vivado_flow.tcl`, and other outputs in `cgra4ml_models/lmiinet_cgra4ml_<timestamp>/`.
  - Rewrites:
    - `deepsocflow/rtl/defines.svh` to inline-include the generated `config_hw.svh` by absolute path.
    - `deepsocflow/tcl/fpga/xc7k70tfbv676-1.tcl` to set `MODEL_DIR` and add the generated `config_hw.svh` explicitly.
- `deepsocflow/rtl/*.sv` and `axi_cgra4ml.v`
  - Consume macros from `defines.svh` → indirectly use generated `config_hw.svh` macros (e.g., `ROWS`, `COLS`, bit widths, RAM depths) to parameterize the fabric.
- `vivado_flow.tcl`
  - Sources `config_hw.tcl` and the board TCL, collects RTL sources, and launches `synth_1`.

## Running the build and synthesis (Windows PowerShell)

- Build (export artifacts; optional `--synthesize` to invoke Vivado):
```powershell
python build_cgra4ml_model.py --model_path .\models\cgra4ml_lmiinet_qat_final.h5 --output_dir .\cgra4ml_models --synthesize
```
- Optional validation run (CPU inference vs exported model stub):
```powershell
python build_cgra4ml_model.py --model_path .\models\cgra4ml_lmiinet_qat_final.h5 --output_dir .\cgra4ml_models --validate
```
- Direct Vivado batch from an output folder:
```powershell
cd .\cgra4ml_models\lmiinet_cgra4ml_<timestamp>
vivado -mode batch -source vivado_flow.tcl
```

Notes:
- Windows path length warnings may appear; placing the workspace closer to a drive root (or using `subst`) can help.

## Troubleshooting and fit guidance

- Over-utilization on xc7k70t:
  - Reduce PE grid and/or frequency in `Hardware(...)` (e.g., `(8,24)` at ~200 MHz) and regenerate.
  - Re-run synthesis and compare utilization.
- `export_inference(...)` fails:
  - It expects an XModel graph; Hardware export is still valid. You can proceed with Vivado synthesis, or add an XModel conversion step if you need vectors.
- Missing macro `BITS_KW2`:
  - A fallback macro was added in `defines.svh` to derive from `KW_MAX` if the generator doesn’t define it.
- Pathing:
  - Keep forward slashes when writing absolute paths in Tcl/Verilog includes.

## File map and responsibilities (workspace)

- `build_cgra4ml_model.py`
  - Orchestrates model load, hardware export, report/guide generation, and Vivado launch; rewrites DeepSoCFlow include/TCL to latest output.
- `deepsocflow/rtl/defines.svh`
  - Project-wide defines; includes generated `config_hw.svh`. Contains fallback macro fixes.
- `deepsocflow/tcl/fpga/xc7k70tfbv676-1.tcl`
  - Vivado project script for the Kintex-7 part; builder updates `MODEL_DIR` and `add_files` for the generated config.
- `cgra4ml_models/lmiinet_cgra4ml_<timestamp>/`
  - Generated configuration and Vivado scripts; contains logs and report after synthesis.

## Next steps

- Adjust `Hardware(processing_elements=(R,C), frequency_mhz=F)` in `convert_to_cgra4ml()` to a smaller grid and re-run synthesis.
- Optionally parameterize these as CLI args (e.g., `--pe_rows`, `--pe_cols`, `--frequency_mhz`) for quick iteration.
- After a successful fit, lock down the configuration and proceed to integration on the target board.

---
Completion summary:
- Documented end-to-end build, artifacts, linkages, and Vivado utilization.
- Identified that the current 16x192 grid at 250 MHz does not fit xc7k70t; recommended smaller grid for fit.
