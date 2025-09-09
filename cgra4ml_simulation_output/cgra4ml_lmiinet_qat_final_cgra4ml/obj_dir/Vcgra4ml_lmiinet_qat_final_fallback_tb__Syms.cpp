// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms.h"
#include "Vcgra4ml_lmiinet_qat_final_fallback_tb.h"



// FUNCTIONS
Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms::Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms(Vcgra4ml_lmiinet_qat_final_fallback_tb* topp, const char* namep)
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
