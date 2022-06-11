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


def fusing(onnxFile, onnxSurgeonFile):
    graph = gs.import_onnx(onnx.load(onnxFile))

    nLayerNorm = 0

    all_nodes = graph.nodes
    for node in all_nodes:
        # check LayerNorm
        try:
            if checkLayerNorm(node):
                nLayerNorm += 1
                pluginVariable = gs.Variable("MyLayerNorm-%d" % nLayerNorm, np.dtype(np.float32), None)
                pluginNode = gs.Node("LayerNorm", "MyLayerNorm-%d" % nLayerNorm, inputs=[node.outputs[0]], outputs=[pluginVariable], attrs={"epsilon": node.o(1).o(1).i(1).i(0).inputs[1].values.reshape(1)})
                pluginNode.version = '1'
                graph.nodes.append(pluginNode)
                node.o(1).o(1).o(0).inputs[0] = pluginVariable
                node.o(1).o(1).outputs.clear()
                print('Fused LayerNorm [{}], begin with [{}]'.format(nLayerNorm,node.name))
        except:
            continue

    graph.cleanup()
    onnx.save(gs.export_onnx(graph), onnxSurgeonFile)



if __name__ == '__main__':

    onnxFile = '../2_baseline/baseline.onnx'
    onnxSurgeonFile = 'model-layernorm.onnx'

    print('--------------- Fusing Onnx:[{}] ---------------'.format(onnxFile))
    fusing(onnxFile, onnxSurgeonFile)