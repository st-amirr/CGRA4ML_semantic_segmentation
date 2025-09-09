`include "config_hw.svh"

module cgra4ml_synth_test (
    input  logic clk,
    input  logic rst_n,
    output logic [`ROWS-1:0] pe_rows_test,
    output logic [`COLS-1:0] pe_cols_test,
    output logic [`AXI_WIDTH-1:0] axi_data_test
);

    // Test CGRA4ML configuration parameters
    assign pe_rows_test = `ROWS;
    assign pe_cols_test = `COLS;
    assign axi_data_test = `AXI_WIDTH'h0;
    
    // Simple counter to show activity
    logic [31:0] counter;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            counter <= 0;
        else
            counter <= counter + 1;
    end

endmodule
