// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary design header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef _VCGRA4ML_LMIINET_QAT_FINAL_FALLBACK_TB_H_
#define _VCGRA4ML_LMIINET_QAT_FINAL_FALLBACK_TB_H_  // guard

#include "verilated_heavy.h"

//==========

class Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms;
class Vcgra4ml_lmiinet_qat_final_fallback_tb_VerilatedVcd;


//----------

VL_MODULE(Vcgra4ml_lmiinet_qat_final_fallback_tb) {
  public:
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    CData/*0:0*/ cgra4ml_lmiinet_qat_final_fallback_tb__DOT__clk;
    CData/*0:0*/ cgra4ml_lmiinet_qat_final_fallback_tb__DOT__rst_n;
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    CData/*0:0*/ __Vchglast__TOP__cgra4ml_lmiinet_qat_final_fallback_tb__DOT__clk;
    CData/*0:0*/ __Vm_traceActivity[1];
    
    // INTERNAL VARIABLES
    // Internals; generally not touched by application code
    Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __VlSymsp;  // Symbol table
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(Vcgra4ml_lmiinet_qat_final_fallback_tb);  ///< Copying not allowed
  public:
    /// Construct the model; called by application code
    /// The special name  may be used to make a wrapper with a
    /// single model invisible with respect to DPI scope names.
    Vcgra4ml_lmiinet_qat_final_fallback_tb(const char* name = "TOP");
    /// Destroy the model; called (often implicitly) by application code
    ~Vcgra4ml_lmiinet_qat_final_fallback_tb();
    /// Trace signals in the model; called by application code
    void trace(VerilatedVcdC* tfp, int levels, int options = 0);
    
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval() { eval_step(); }
    /// Evaluate when calling multiple units/models per time step.
    void eval_step();
    /// Evaluate at end of a timestep for tracing, when using eval_step().
    /// Application must call after all eval() and before time changes.
    void eval_end_step() {}
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    
    // INTERNAL METHODS
  private:
    static void _eval_initial_loop(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp);
  public:
    void __Vconfigure(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* symsp, bool first);
  private:
    static QData _change_request(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp);
    static QData _change_request_1(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp);
  public:
    static void _combo__TOP__2(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp);
  private:
    void _ctor_var_reset() VL_ATTR_COLD;
  public:
    static void _eval(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp);
  private:
#ifdef VL_DEBUG
    void _eval_debug_assertions();
#endif  // VL_DEBUG
  public:
    static void _eval_initial(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _eval_settle(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _initial__TOP__1(Vcgra4ml_lmiinet_qat_final_fallback_tb__Syms* __restrict vlSymsp) VL_ATTR_COLD;
  private:
    static void traceChgSub0(void* userp, VerilatedVcd* tracep);
    static void traceChgTop0(void* userp, VerilatedVcd* tracep);
    static void traceCleanup(void* userp, VerilatedVcd* /*unused*/);
    static void traceFullSub0(void* userp, VerilatedVcd* tracep) VL_ATTR_COLD;
    static void traceFullTop0(void* userp, VerilatedVcd* tracep) VL_ATTR_COLD;
    static void traceInitSub0(void* userp, VerilatedVcd* tracep) VL_ATTR_COLD;
    static void traceInitTop(void* userp, VerilatedVcd* tracep) VL_ATTR_COLD;
    void traceRegister(VerilatedVcd* tracep) VL_ATTR_COLD;
    static void traceInit(void* userp, VerilatedVcd* tracep, uint32_t code) VL_ATTR_COLD;
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

//----------


#endif  // guard
