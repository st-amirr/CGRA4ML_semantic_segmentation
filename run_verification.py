#!/usr/bin/env python3

import sys
import os
sys.path.append('/workspace/cgra4ml')
os.chdir('/workspace/cgra4ml_models/lmiinet_cgra4ml_1757474660')

from deepsocflow import *
import tensorflow as tf
from qkeras import QActivation, QConv2D, QBatchNormalization, quantized_bits, quantized_relu, QDense, QDepthwiseConv2D

print("Loading model...")
# Load with QKeras custom objects
custom_objects = {
    'QConv2D': QConv2D,
    'QActivation': QActivation,
    'QBatchNormalization': QBatchNormalization,
    'quantized_bits': quantized_bits,
    'quantized_relu': quantized_relu,
    'QDense': QDense,
    'QDepthwiseConv2D': QDepthwiseConv2D
}
model = tf.keras.models.load_model("/workspace/models/cgra4ml_lmiinet_qat_final.h5", custom_objects=custom_objects)

print("Creating hardware configuration...")
hw = Hardware(
    processing_elements=(16, 96),
    frequency_mhz=200,
    bits_input=8,
    bits_weights=8,
    bits_sum=20,
    bits_bias=20
)

print("Running verify_inference...")
try:
    verify_inference(model, hw, SIM="verilator", SIM_PATH=".")
    print("✅ Simulation completed successfully!")
except Exception as e:
    print(f"❌ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
