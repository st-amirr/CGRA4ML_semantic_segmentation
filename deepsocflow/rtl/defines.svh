`ifndef DEEPSOCFLOW_DEFINES_SVH
`define DEEPSOCFLOW_DEFINES_SVH

// Auto-updated by the builder to the latest generated config_hw.svh
// Include hardware config; builder may rewrite the path, or include_dirs may resolve it
`ifdef VERILATOR_SIMULATION
    // Use local config for Verilator simulation
    `include "config_hw.svh"
`else
    // Use the original path for Vivado synthesis
    `include "C:/Users/stamir/Desktop/hls4ml_SS/cgra4ml_models/lmiinet_cgra4ml_1757384338/config_hw.svh"
`endif

typedef struct packed {
    logic [`BITS_KW2-1:0] kw2;
    logic                 is_config;
    logic                 is_cin_last;
    logic                 is_w_first_clk;
    logic                 is_w_first_kw2;
    logic                 is_w_last;
} tuser_st;

`define TUSER_WIDTH (`BITS_KW2 + 5)

`endif // DEEPSOCFLOW_DEFINES_SVHW_DEFINES_SVH
`define DEEPSOCFLOW_DEFINES_SVH

// Auto-updated by the builder to the latest generated config_hw.svh
// Include hardware config; builder may rewrite the path, or include_dirs may resolve it
`include "C:/Users/stamir/Desktop/hls4ml_SS/cgra4ml_models/lmiinet_cgra4ml_1757200044/config_hw.svh"
typedef struct packed {
    logic [`BITS_KW2-1:0] kw2;
    logic                 is_config;
    logic                 is_cin_last;
    logic                 is_w_first_clk;
    logic                 is_w_first_kw2;
    logic                 is_w_last;
} tuser_st;

`define TUSER_WIDTH (`BITS_KW2 + 5)

`endif // DEEPSOCFLOW_DEFINES_SVH