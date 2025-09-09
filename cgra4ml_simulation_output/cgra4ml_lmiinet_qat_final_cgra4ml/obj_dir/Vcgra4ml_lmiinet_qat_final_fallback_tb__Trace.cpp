// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms.h"


void Vcgra4ml_lmiinet_qat_final_fallback_tb::traceChgTop0(void* userp, VerilatedVcd* tracep) {
    Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp = static_cast<Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms*>(userp);
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    {
        vlTOPp->traceChgSub0(userp, tracep);
    }
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::traceChgSub0(void* userp, VerilatedVcd* tracep) {
    Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp = static_cast<Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms*>(userp);
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    vluint32_t* const oldp = tracep->oldp(vlSymsp->__Vm_baseCode + 1);
    if (false && oldp) {}  // Prevent unused
    // Body
    {
        if (VL_UNLIKELY(vlTOPp->__Vm_traceActivity[0U])) {
            tracep->chgBit(oldp+0,(vlTOPp->cgra4ml_lmiinet_qat_final_fallback_tb__DOT__rst_n));
        }
        tracep->chgBit(oldp+1,(vlTOPp->cgra4ml_lmiinet_qat_final_fallback_tb__DOT__clk));
    }
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::traceCleanup(void* userp, VerilatedVcd* /*unused*/) {
    Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp = static_cast<Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms*>(userp);
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    {
        vlSymsp->__Vm_activity = false;
        vlTOPp->__Vm_traceActivity[0U] = 0U;
    }
}
