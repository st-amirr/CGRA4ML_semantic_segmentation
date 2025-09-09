
module cgra4ml_lmiinet_qat_final_tb;
    reg clk = 0;
    reg rst_n = 0;
    
    always #2.5 clk = ~clk; // 200MHz clock
    
    // Add your CGRA4ML top module instantiation here
    // This will depend on the actual generated RTL
    
    initial begin
        $display("=== CGRA4ML_LMIINET_QAT_FINAL CGRA4ML RTL SIMULATION ===");
        
        rst_n = 0;
        #100;
        rst_n = 1;
        
        // Run simulation
        #50000;
        
        $display("=== RTL SIMULATION COMPLETE ===");
        $finish;
    end
endmodule
