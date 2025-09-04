set BOARD zcu104
set CONFIG_BASEADDR 0x00B0000000

# Basic project vars
if {![info exists PROJECT_NAME]} { set PROJECT_NAME dsf_zcu104 }
if {![info exists PART]} { set PART xczu7ev-ffvc1156-2-e }
if {![info exists TOP]} { set TOP axi_cgra4ml }

create_project ${PROJECT_NAME} ${PROJECT_NAME} -part ${PART} -force
set_property board_part xilinx.com:zcu104:part0:1.1 [current_project]

# Provide defaults for freq/AXI width
if {![info exists FREQ]} { set FREQ 200 }
if {![info exists AXI_WIDTH]} { set AXI_WIDTH 128 }

# Try to create Zynq MPSoC PS only if the IP is available
set ps_ip_defs [lsort -dictionary [get_ipdefs -all xilinx.com:ip:zynq_ultra_ps_e:*]]
set has_ps_ip [llength $ps_ip_defs]
if {$has_ps_ip > 0} {
	# Pick the highest available version of the PS IP
	set ps_vlnv [lindex $ps_ip_defs end]
	puts "INFO: Using PS IP VLNV: $ps_vlnv"
	create_bd_design "design_1"
	if {[catch {create_bd_cell -type ip -vlnv $ps_vlnv zynq_ultra_ps_e_0} create_err]} {
		puts "WARNING: Failed to instantiate PS IP ($ps_vlnv): $create_err"
	} else {
		if {[catch {apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells zynq_ultra_ps_e_0]} bd_auto_err]} {
			puts "WARNING: Failed to apply BD automation: $bd_auto_err"
		}
		set_property -dict [list \
			CONFIG.PSU__USE__M_AXI_GP1 {1} \
			CONFIG.PSU__USE__S_AXI_GP0 {1} \
			CONFIG.PSU__USE__S_AXI_GP1 {1} \
			CONFIG.PSU__USE__S_AXI_GP2 {1} \
			CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $FREQ \
			CONFIG.PSU__USE__M_AXI_GP0 {0} \
			CONFIG.PSU__QSPI__PERIPHERAL__ENABLE {0} \
			CONFIG.PSU__SAXIGP0__DATA_WIDTH $AXI_WIDTH \
			CONFIG.PSU__SAXIGP1__DATA_WIDTH $AXI_WIDTH \
			CONFIG.PSU__SAXIGP2__DATA_WIDTH $AXI_WIDTH \
		] [get_bd_cells zynq_ultra_ps_e_0]

		set PS_IRQ           "zynq_ultra_ps_e_0/pl_ps_irq0"
		set PS_M_AXI_LITE    "/zynq_ultra_ps_e_0/M_AXI_HPM1_FPD"
		set PS_S_AXI_OUTPUT  "/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD"
		set PS_S_AXI_PIXELS  "/zynq_ultra_ps_e_0/S_AXI_HPC1_FPD"
		set PS_S_AXI_WEIGHTS "/zynq_ultra_ps_e_0/S_AXI_HP0_FPD"
		set PS_CLK           "/zynq_ultra_ps_e_0/pl_clk0"
	}
} else {
	puts "WARNING: MPSoC PS IP not installed. Skipping PS BD creation."
}

# Sources (builder will rewrite MODEL_DIR and config_hw.svh path)
if {![info exists MODEL_DIR]} { set MODEL_DIR ./cgra4ml_models/latest }
add_files [glob C:/Users/stamir/Desktop/hls4ml_SS/deepsocflow/rtl/*.sv]
add_files [glob C:/Users/stamir/Desktop/hls4ml_SS/deepsocflow/rtl/*.v]
add_files "C:/Users/stamir/Desktop/hls4ml_SS/cgra4ml_models/lmiinet_cgra4ml_1756981720/config_hw.svh"
add_files [glob C:/Users/stamir/Desktop/hls4ml_SS/deepsocflow/rtl/ext/*.sv]
add_files [glob C:/Users/stamir/Desktop/hls4ml_SS/deepsocflow/rtl/ext/*.v]

set_property top $TOP [current_fileset]

reset_run synth_1
launch_runs synth_1
wait_on_run synth_1

# Skip implementation; report on synthesized netlist
open_run synth_1
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 100 -input_pins -routable_nets -name timing_1 -file timing_report_synth.txt
report_utilization -file utilization_report_synth.txt -name utilization_1
report_power -file power_report_synth.txt -name {power_1}
report_drc -name drc_1 -file drc_report_synth.txt -ruledecks {default opt_checks placer_checks router_checks bitstream_checks incr_eco_checks eco_checks abs_checks}
