// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms.h"


//======================

void Vcgra4ml_lmiinet_qat_final_fallback_tb::trace(VerilatedVcdC* tfp, int, int) {
    tfp->spTrace()->addInitCb(&traceInit, __VlSymsp);
    traceRegister(tfp->spTrace());
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::traceInit(void* userp, VerilatedVcd* tracep, uint32_t code) {
    // Callback from tracep->open()
    Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp = static_cast<Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms*>(userp);
    if (!Verilated::calcUnusedSigs()) {
        VL_FATAL_MT(__FILE__, __LINE__, __FILE__,
                        "Turning on wave traces requires Verilated::traceEverOn(true) call before time 0.");
    }
    vlSymsp->__Vm_baseCode = code;
    tracep->module(vlSymsp->name());
    tracep->scopeEscape(' ');
    Vcgra4ml_lmiinet_qat_final_fallback_tb::traceInitTop(vlSymsp, tracep);
    tracep->scopeEscape('.');
}

//======================


void Vcgra4ml_lmiinet_qat_final_fallback_tb::traceInitTop(void* userp, VerilatedVcd* tracep) {
    Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp = static_cast<Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms*>(userp);
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    {
        vlTOPp->traceInitSub0(userp, tracep);
    }
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::traceInitSub0(void* userp, VerilatedVcd* tracep) {
    Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp = static_cast<Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms*>(userp);
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    const int c = vlSymsp->__Vm_baseCode;
    if (false && tracep && c) {}  // Prevent unused
    // Body
    {
        tracep->declBit(c+2,"cgra4ml_lmiinet_qat_final_fallback_tb clk", false,-1);
        tracep->declBit(c+1,"cgra4ml_lmiinet_qat_final_fallback_tb rst_n", false,-1);
    }
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::traceRegister(VerilatedVcd* tracep) {
    // Body
    {
        tracep->addFullCb(&traceFullTop0, __VlSymsp);
        tracep->addChgCb(&traceChgTop0, __VlSymsp);
        tracep->addCleanupCb(&traceCleanup, __VlSymsp);
    }
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::traceFullTop0(void* userp, VerilatedVcd* tracep) {
    Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp = static_cast<Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms*>(userp);
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    {
        vlTOPp->traceFullSub0(userp, tracep);
    }
}

void Vcgra4ml_lmiinet_qat_final_fallback_tb::traceFullSub0(void* userp, VerilatedVcd* tracep) {
    Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp = static_cast<Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms*>(userp);
    Vcgra4ml_lmiinet_qat_final_fallback_tb* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    vluint32_t* const oldp = tracep->oldp(vlSymsp->__Vm_baseCode);
    if (false && oldp) {}  // Prevent unused
    // Body
    {
        tracep->fullBit(oldp+1,(vlTOPp->cgra4ml_lmiinet_qat_final_fallback_tb__DOT__rst_n));
        tracep->fullBit(oldp+2,(vlTOPp->cgra4ml_lmiinet_qat_final_fallback_tb__DOT__clk));
    }
}
