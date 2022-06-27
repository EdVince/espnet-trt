import sys
import numpy as np
import onnx
import onnx_graphsurgeon as gs

graph = gs.import_onnx(onnx.load('../2_baseline/baseline.onnx'))
graph.outputs = [graph.outputs[1]]
graph.cleanup()
onnx.save(gs.export_onnx(graph), 'baseline-cut.onnx')
