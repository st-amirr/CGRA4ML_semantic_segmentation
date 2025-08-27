# Check available FPGA parts
puts "Available FPGA parts:"
get_parts

puts "\nChecking for common Zynq parts:"
puts "Zynq-7000 series:"
get_parts -filter "FAMILY == 'zynq'"

puts "\nZynq UltraScale+ series:"
get_parts -filter "FAMILY == 'zynquplus'"

puts "\nArtix-7 series:"
get_parts -filter "FAMILY == 'artix7'"

puts "\nKintex-7 series:"
get_parts -filter "FAMILY == 'kintex7'"

puts "\nVirtex-7 series:"
get_parts -filter "FAMILY == 'virtex7'"

puts "\nChecking specific common parts:"
set common_parts {
    xc7z020clg484-1
    xc7z010clg400-1
    xc7z020clg400-1
    xc7z010clg484-1
    xczu3eg-sbva484-1-e
    xczu7ev-ffvc1156-2-e
    xczu9eg-ffvb1156-2-e
    xc7a35tcsg324-1
    xc7a100tcsg324-1
}

foreach part $common_parts {
    if {[catch {get_parts $part} result]} {
        puts "Part $part: NOT AVAILABLE"
    } else {
        puts "Part $part: AVAILABLE"
    }
}
