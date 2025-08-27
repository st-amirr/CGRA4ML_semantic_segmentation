# Vivado TCL script for KC705 board
# This script creates a Vivado project for the KC705 board (xc7k70tfbv676-1).

set PROJECT_NAME cgra4ml_project
set PART xc7k70tfbv676-1
set TOP_MODULE_NAME top  ;# Change 'top' to your actual top module name if needed

create_project $PROJECT_NAME $PROJECT_NAME -part $PART -force

# Add source files (update paths as needed)
# add_files ../src/top.v
# add_files ../src/top.xdc

# Set the top module for synthesis
set_property top $TOP_MODULE_NAME [current_fileset]

# Run synthesis
launch_runs synth_1
wait_on_run synth_1

# Generate bitstream
launch_runs impl_1
wait_on_run impl_1
write_bitstream -force $PROJECT_NAME/$PROJECT_NAME.bit

puts "Vivado project for KC705 ($PART) created and synthesized. Top module: $TOP_MODULE_NAME."
