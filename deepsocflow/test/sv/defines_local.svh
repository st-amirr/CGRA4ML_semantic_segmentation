`ifndef DEEPSOCFLOW_DEFINES_SVH
`define DEEPSOCFLOW_DEFINES_SVH

// Use local config_hw.svh instead of external path
`include "config_hw.svh"

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
