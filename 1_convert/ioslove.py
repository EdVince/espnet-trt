import os
import ctypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

onnxFile = "kan-bayashi/ljspeech_vits/full/tts_model.onnx"
onnxSurgeonFile = "./tts_model.onnx"
graph = gs.import_onnx(onnx.load(onnxFile))

# clean useless output
graph.outputs = graph.outputs[:1]

# merge input
all_nodes = graph.nodes
shape_node = None
length_node1 = None
length_node2 = None
unsqueeze_node = None
for node in all_nodes:
    if node.name == 'Gather_30':
        shape_node = node
    if node.name == 'Sub_7':
        length_node1 = node
    if node.name == 'ReduceMax_5':
        length_node2 = node
    if node.name == 'Unsqueeze_11':
        unsqueeze_node = node.copy()
        unsqueeze_node.inputs.append(node.inputs[1])
unsqueeze_node.name = 'unsqueeze_text_length'
unsqueeze_node.inputs.append(shape_node.outputs[0])
unsqueeze_node.outputs.append(graph.inputs[1])
unsqueeze_node.outputs[0].shape = None
unsqueeze_node.outputs[0].dtype = None
graph.inputs = graph.inputs[:1]
unsqueeze_node.inputs[1], unsqueeze_node.inputs[0] = unsqueeze_node.inputs[0], unsqueeze_node.inputs[1]
graph.nodes.append(unsqueeze_node)

# clean
graph.cleanup()
onnx.save(gs.export_onnx(graph), onnxSurgeonFile)