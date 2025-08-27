# Check available FPGA parts
puts "Checking available FPGA parts..."
puts "=============================================="

# Get all available parts
set all_parts [get_parts]

if {[llength $all_parts] > 0} {
    puts "Found [llength $all_parts] total parts:"
    foreach part $all_parts {
        puts "  $part"
    }
} else {
    puts "No parts found!"
}

puts "=============================================="
puts "Checking for common Zynq parts:"
puts "Zynq-7000 series:"
if {[catch {get_parts -filter "FAMILY == 'zynq'"} result]} {
    puts "  Error getting Zynq-7000 parts: $result"
} else {
    puts "  $result"
}

puts "Zynq UltraScale+ series:"
if {[catch {get_parts -filter "FAMILY == 'zynquplus'"} result]} {
    puts "  Error getting Zynq UltraScale+ parts: $result"
} else {
    puts "  $result"
}

puts "Artix-7 series:"
if {[catch {get_parts -filter "FAMILY == 'artix7'"} result]} {
    puts "  Error getting Artix-7 parts: $result"
} else {
    puts "  $result"
}

puts "Kintex-7 series:"
if {[catch {get_parts -filter "FAMILY == 'kintex7'"} result]} {
    puts "  Error getting Kintex-7 parts: $result"
} else {
    puts "  $result"
}
