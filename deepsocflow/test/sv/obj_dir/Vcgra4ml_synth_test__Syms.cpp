// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vcgra4ml_synth_test__Syms.h"
#include "Vcgra4ml_synth_test.h"



// FUNCTIONS
Vcgra4ml_synth_test__Syms::Vcgra4ml_synth_test__Syms(Vcgra4ml_synth_test* topp, const char* namep)
    // Setup locals
    : __Vm_namep(namep)
    , __Vm_didInit(false)
    // Setup submodule names
{
    // Pointer to top level
    TOPp = topp;
    // Setup each module's pointers to their submodules
    // Setup each module's pointer back to symbol table (for public functions)
    TOPp->__Vconfigure(this, true);
}
