// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef _VCGRA4ML_LMIINET_QAT_FINAL_FALLBACK_TB__SYMS_H_
#define _VCGRA4ML_LMIINET_QAT_FINAL_FALLBACK_TB__SYMS_H_  // guard

#include "verilated_heavy.h"

// INCLUDE MODULE CLASSES
#include "Vcgra4ml_lmiinet_qat_final_fallback_tb.h"

// SYMS CLASS
class Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_didInit;
    
    // SUBCELL STATE
    Vcgra4ml_lmiinet_qat_final_fallback_tb* TOPp;
    
    // CREATORS
    Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms(Vcgra4ml_lmiinet_qat_final_fallback_tb* topp, const char* namep);
    ~Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

#endif  // guard
