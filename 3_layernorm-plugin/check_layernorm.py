import sys
import numpy as np
import onnx
import onnx_graphsurgeon as gs

def checkLayerNorm(node):
    return node.o(0).op == 'ReduceMean' and \
            node.o(1).op == 'Sub' and node.o(0).o(0).op == 'Sub' and \
            node.o(1).o(1).op == 'Div' and \
            node.o(1).o(1).i(1).op == 'Sqrt' and \
            node.o(1).o(1).i(1).i(0).op == 'Add' and \
            node.o(1).o(1).i(1).i(0).i(0).op == 'ReduceMean' and \
            node.o(1).o(1).i(1).i(0).i(0).i(0).op == 'Pow'

if __name__ == '__main__':

    graph = gs.import_onnx(onnx.load('../2_baseline/baseline.onnx'))

    all_nodes = graph.nodes
    for node in all_nodes:
        try:
            if checkLayerNorm(node):
                print(node.outputs[0].name)
        except:
            continue