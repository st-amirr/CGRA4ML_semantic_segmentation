// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vcgra4ml_synth_test.h for the primary calling header

#include "Vcgra4ml_synth_test.h"
#include "Vcgra4ml_synth_test__Syms.h"

//==========

VL_CTOR_IMP(Vcgra4ml_synth_test) {
    Vcgra4ml_synth_test__Syms* __restrict vlSymsp = __VlSymsp = new Vcgra4ml_synth_test__Syms(this, name());
    Vcgra4ml_synth_test* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void Vcgra4ml_synth_test::__Vconfigure(Vcgra4ml_synth_test__Syms* vlSymsp, bool first) {
    if (false && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
    if (false && this->__VlSymsp) {}  // Prevent unused
    Verilated::timeunit(-12);
    Verilated::timeprecision(-12);
}

Vcgra4ml_synth_test::~Vcgra4ml_synth_test() {
    VL_DO_CLEAR(delete __VlSymsp, __VlSymsp = NULL);
}

void Vcgra4ml_synth_test::_initial__TOP__1(Vcgra4ml_synth_test__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcgra4ml_synth_test::_initial__TOP__1\n"); );
    Vcgra4ml_synth_test* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->pe_rows_test = 0x10U;
    vlTOPp->pe_cols_test[0U] = 0x60U;
    vlTOPp->pe_cols_test[1U] = 0U;
    vlTOPp->pe_cols_test[2U] = 0U;
    vlTOPp->axi_data_test[0U] = 0U;
    vlTOPp->axi_data_test[1U] = 0U;
    vlTOPp->axi_data_test[2U] = 0U;
    vlTOPp->axi_data_test[3U] = 0U;
}

void Vcgra4ml_synth_test::_eval_initial(Vcgra4ml_synth_test__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcgra4ml_synth_test::_eval_initial\n"); );
    Vcgra4ml_synth_test* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_initial__TOP__1(vlSymsp);
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
    vlTOPp->__Vclklast__TOP__rst_n = vlTOPp->rst_n;
}

void Vcgra4ml_synth_test::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcgra4ml_synth_test::final\n"); );
    // Variables
    Vcgra4ml_synth_test__Syms* __restrict vlSymsp = this->__VlSymsp;
    Vcgra4ml_synth_test* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vcgra4ml_synth_test::_eval_settle(Vcgra4ml_synth_test__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcgra4ml_synth_test::_eval_settle\n"); );
    Vcgra4ml_synth_test* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vcgra4ml_synth_test::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcgra4ml_synth_test::_ctor_var_reset\n"); );
    // Body
    clk = VL_RAND_RESET_I(1);
    rst_n = VL_RAND_RESET_I(1);
    pe_rows_test = VL_RAND_RESET_I(16);
    VL_RAND_RESET_W(96, pe_cols_test);
    VL_RAND_RESET_W(128, axi_data_test);
}
