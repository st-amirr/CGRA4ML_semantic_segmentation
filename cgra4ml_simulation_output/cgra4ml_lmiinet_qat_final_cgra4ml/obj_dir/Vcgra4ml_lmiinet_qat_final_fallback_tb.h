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

//----------

VL_MODULE(Vcgra4ml_lmiinet_qat_final_fallback_tb) {
  public:
    
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
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

//----------


#endif  // guard
