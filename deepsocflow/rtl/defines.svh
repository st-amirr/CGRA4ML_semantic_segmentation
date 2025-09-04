`ifndef DEEPSOCFLOW_DEFINES_SVH
`define DEEPSOCFLOW_DEFINES_SVH

// Auto-updated by the builder to the latest generated config_hw.svh
// Prefer relative include; builder may rewrite this line to an absolute path.
`include "config_hw.svh"

// Fallback for BITS_KW2 if not defined by config_hw.svh
`ifndef BITS_KW2
    `define BITS_KW2 $clog2((`KW_MAX+1)/2)
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

`endif // DEEPSOCFLOW_DEFINES_SVH