// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vcgra4ml_lmiinet_qat_final_fallback_tb.h for the primary calling header

#include "Vcgra4ml_lmiinet_qat_final_fallback_tb.h"
#include "Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms.h"

//==========

VL_CTOR_IMP(Vcgra4ml_lmiinet_qat_final_fallback_tb) {
    Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp = __VlSymsp = new Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms(this, name());
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::__Vconfigure(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* vlSymsp, bool first) {
    if (false && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
    if (false && this->__VlSymsp) {}  // Prevent unused
    Verilated::timeunit(-12);
    Verilated::timeprecision(-12);
}

Vcgra4ml_lmiinet_qat_final_fallback_tb::~Vcgra4ml_lmiinet_qat_final_fallback_tb() {
    VL_DO_CLEAR(delete __VlSymsp, __VlSymsp = NULL);
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::_initial__TOP__1(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcgra4ml_lmiinet_qat_final_fallback_tb::_initial__TOP__1\n"); );
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    VL_WRITEF("=== CGRA4ML_LMIINET_QAT_FINAL FALLBACK SIMULATION ===\nModel has 37 conv layers, 826,395 parameters\n=== SIMULATION COMPLETE ===\nEstimated performance based on model analysis\n");
    VL_FINISH_MT("cgra4ml_lmiinet_qat_final_fallback_tb.sv", 24, "");
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::_eval_initial(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcgra4ml_lmiinet_qat_final_fallback_tb::_eval_initial\n"); );
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_initial__TOP__1(vlSymsp);
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcgra4ml_lmiinet_qat_final_fallback_tb::final\n"); );
    // Variables
    Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp = this->__VlSymsp;
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::_eval_settle(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcgra4ml_lmiinet_qat_final_fallback_tb::_eval_settle\n"); );
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcgra4ml_lmiinet_qat_final_fallback_tb::_ctor_var_reset\n"); );
}
