
create_project my_project ./my_project -part xc7k70tfbv676-1
# Vivado TCL for Kintex-7 xc7k70tfbv676-1
set PART xc7k70tfbv676-1
set TOP axi_cgra4ml

# Add RTL sources
set MODEL_DIR C:/Users/stamir/Desktop/hls4ml_SS/cgra4ml_models/lmiinet_cgra4ml_1756308838
add_files [glob C:/Users/stamir/Desktop/hls4ml_SS/deepsocflow/rtl/*.sv]
add_files [glob C:/Users/stamir/Desktop/hls4ml_SS/deepsocflow/rtl/*.v]
add_files "C:/Users/stamir/Desktop/hls4ml_SS/cgra4ml_models/lmiinet_cgra4ml_1756308838/config_hw.svh"
add_files [glob C:/Users/stamir/Desktop/hls4ml_SS/deepsocflow/rtl/ext/*.sv]
add_files [glob C:/Users/stamir/Desktop/hls4ml_SS/deepsocflow/rtl/ext/*.v]

# Set top module
set_property top $TOP [current_fileset]

# Synthesis and implementation
reset_run synth_1
launch_runs synth_1
wait_on_run synth_1

reset_run impl_1
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

# Reports
open_run impl_1
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 100 -input_pins -routable_nets -name timing_1 -file timing_report.txt
report_utilization -file utilization_report.txt -name utilization_1
report_power -file power_report.txt -name {power_1}
report_drc -name drc_1 -file drc_report.txt -ruledecks {default opt_checks placer_checks router_checks bitstream_checks incr_eco_checks eco_checks abs_checks}
