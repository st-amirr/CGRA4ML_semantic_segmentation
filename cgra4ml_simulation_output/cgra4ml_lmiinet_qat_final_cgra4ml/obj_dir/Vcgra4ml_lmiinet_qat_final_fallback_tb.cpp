// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vcgra4ml_lmiinet_qat_final_fallback_tb.h for the primary calling header

#include "Vcgra4ml_lmiinet_qat_final_fallback_tb.h"
#include "Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms.h"

//==========

void Vcgra4ml_lmiinet_qat_final_fallback_tb::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vcgra4ml_lmiinet_qat_final_fallback_tb::eval\n"); );
    Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp = this->__VlSymsp;  // Setup global symbol table
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
#ifdef VL_DEBUG
    // Debug assertions
    _eval_debug_assertions();
#endif  // VL_DEBUG
    // Initialize
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) _eval_initial_loop(vlSymsp);
    // Evaluate till stable
    int __VclockLoop = 0;
    QData __Vchange = 1;
    do {
        VL_DEBUG_IF(VL_DBG_MSGF("+ Clock loop\n"););
        _eval(vlSymsp);
        if (VL_UNLIKELY(++__VclockLoop > 100)) {
            // About to fail, so enable debug to see what's not settling.
            // Note you must run make with OPT=-DVL_DEBUG for debug prints.
            int __Vsaved_debug = Verilated::debug();
            Verilated::debug(1);
            __Vchange = _change_request(vlSymsp);
            Verilated::debug(__Vsaved_debug);
            VL_FATAL_MT("cgra4ml_lmiinet_qat_final_fallback_tb.sv", 5, "",
                "Verilated model didn't converge\n"
                "- See DIDNOTCONVERGE in the Verilator manual");
        } else {
            __Vchange = _change_request(vlSymsp);
        }
    } while (VL_UNLIKELY(__Vchange));
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::_eval_initial_loop(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp) {
    vlSymsp->__Vm_didInit = true;
    _eval_initial(vlSymsp);
    // Evaluate till stable
    int __VclockLoop = 0;
    QData __Vchange = 1;
    do {
        _eval_settle(vlSymsp);
        _eval(vlSymsp);
        if (VL_UNLIKELY(++__VclockLoop > 100)) {
            // About to fail, so enable debug to see what's not settling.
            // Note you must run make with OPT=-DVL_DEBUG for debug prints.
            int __Vsaved_debug = Verilated::debug();
            Verilated::debug(1);
            __Vchange = _change_request(vlSymsp);
            Verilated::debug(__Vsaved_debug);
            VL_FATAL_MT("cgra4ml_lmiinet_qat_final_fallback_tb.sv", 5, "",
                "Verilated model didn't DC converge\n"
                "- See DIDNOTCONVERGE in the Verilator manual");
        } else {
            __Vchange = _change_request(vlSymsp);
        }
    } while (VL_UNLIKELY(__Vchange));
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::_eval(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcgra4ml_lmiinet_qat_final_fallback_tb::_eval\n"); );
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

VL_INLINE_OPT QData Vcgra4ml_lmiinet_qat_final_fallback_tb::_change_request(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcgra4ml_lmiinet_qat_final_fallback_tb::_change_request\n"); );
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    return (vlTOPp->_change_request_1(vlSymsp));
}

VL_INLINE_OPT QData Vcgra4ml_lmiinet_qat_final_fallback_tb::_change_request_1(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcgra4ml_lmiinet_qat_final_fallback_tb::_change_request_1\n"); );
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // Change detection
    QData __req = false;  // Logically a bool
    return __req;
}

#ifdef VL_DEBUG
void Vcgra4ml_lmiinet_qat_final_fallback_tb::_eval_debug_assertions() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcgra4ml_lmiinet_qat_final_fallback_tb::_eval_debug_assertions\n"); );
}
#endif  // VL_DEBUG
