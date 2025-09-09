// Hardware Configuration for CGRA4ML Verilator Simulation
// Based on ZCU104 configuration - 16x96 PEs @ 200MHz

`define OR_NEGEDGE(RSTN)    or negedge RSTN

`define ROWS                16          // PE rows
`define COLS                96          // PE cols  
`define X_BITS              8           // Input bits
`define K_BITS              8           // Weight bits
`define Y_BITS              20          // Output bits
`define Y_OUT_BITS          32          // Padded output bits

`define KH_MAX              9           // Max kernel height
`define KW_MAX              9           // Max kernel width
`define XH_MAX              512         // Max input height
`define XW_MAX              512         // Max input width
`define XN_MAX              32          // Max batch size
`define CI_MAX              256         // Max input channels
`define MAX_N_BUNDLES       32          // Max bundles
`define CONFIG_BEATS        0           // Config beats
`define RAM_WEIGHTS_DEPTH   256         // Weight RAM depth
`define RAM_EDGES_DEPTH     1792        // Edge RAM depth
`define W_BPT               32          // Width bytes per transfer

`define DELAY_MUL           3           // Multiplier delay
`define DELAY_W_RAM         2           // Weight RAM delay

`define AXI_WIDTH           128         // AXI bus width (from ZCU104)
`define HEADER_WIDTH        32          // Header width
`define AXI_MAX_BURST_LEN   8           // Max burst length
`define CONFIG_BASEADDR     40'hB0000000  // Base address (from ZCU104)

`define BITS_KW2            3           // ceil(log2((KW_MAX+1)/2))
`define BITS_KH2            3           // ceil(log2((KH_MAX+1)/2))
`define BITS_CIN_MAX        8           // ceil(log2(CI_MAX))
