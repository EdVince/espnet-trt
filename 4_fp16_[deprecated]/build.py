import sys
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import os
import cv2
from glob import glob
from datetime import datetime as dt
import torch as t
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from cuda import cudart
import tensorrt as trt


onnxFile = 'baseline.onnx'
trtFile = 'baseline-fp16.plan'

logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
config.clear_flag(trt.BuilderFlag.TF32)
parser = trt.OnnxParser(network, logger)
if not os.path.exists(onnxFile):
    print("Failed finding onnx file!")
    exit()
print("Succeeded finding onnx file!")
with open(onnxFile, 'rb') as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing onnx file!")

'''
129792 130304 0.003929273084479371
79872 79872 0.0
31488 31744 0.008064516129032258
'''


all_layer_type = []
for i in range(network.num_layers):
    layer = network.get_layer(i)


    if layer.type in [
            trt.LayerType.CONVOLUTION,
            trt.LayerType.ELEMENTWISE,
            trt.LayerType.MATRIX_MULTIPLY,
        ] and layer.precision == trt.DataType.FLOAT:

        layer.precision = trt.DataType.FLOAT

        for o in range(layer.num_outputs):
            if layer.get_output_type(o) == trt.DataType.FLOAT:
                layer.set_output_type(o,trt.DataType.FLOAT)


inputTensor = network.get_input(0)
profile.set_shape(inputTensor.name, trt.tensorrt.Dims([4]), trt.tensorrt.Dims([36]), trt.tensorrt.Dims([68]))
config.add_optimization_profile(profile)

engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, 'wb') as f:
    f.write(engineString)
