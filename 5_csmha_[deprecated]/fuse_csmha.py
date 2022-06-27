import sys
import numpy as np
import onnx
import onnx_graphsurgeon as gs

def checkCMHA(node):
    return node.o(0).op == 'MatMul' and node.o(1).op == 'MatMul' and node.o(2).op == 'MatMul'


def fusing(onnxFile, onnxSurgeonFile):
    graph = gs.import_onnx(onnx.load(onnxFile))

    nCSMHA = 0

    all_nodes = graph.nodes
    for node in all_nodes:
        # check LayerNorm
        try:
            if checkCMHA(node):

                nCSMHA += 1

                k = np.array([192]).reshape(1).astype(np.int32)
                n = np.array([192]).reshape(1).astype(np.int32)
                qw = node.o(0).inputs[1].values.astype(np.float32).reshape(192*192)
                qb = node.o(0).o(0).inputs[0].values.astype(np.float32).reshape(192)
                kw = node.o(1).inputs[1].values.astype(np.float32).reshape(192*192)
                kb = node.o(1).o(0).inputs[0].values.astype(np.float32).reshape(192)
                vw = node.o(2).inputs[1].values.astype(np.float32).reshape(192*192)
                vb = node.o(2).o(0).inputs[0].values.astype(np.float32).reshape(192)
                qkb = node.o(0).o(0).o(0).o(0).inputs[1].values.astype(np.float32).reshape(192)
                qcb = node.o(0).o(0).o(0).o(1).inputs[1].values.astype(np.float32).reshape(192)
                ow = node.o(2).o(0).o(0).o(0).o(0).o(0).o(0).o(0).inputs[1].values.astype(np.float32).reshape(192*192)
                ob = node.o(2).o(0).o(0).o(0).o(0).o(0).o(0).o(0).o(0).inputs[0].values.astype(np.float32).reshape(192)

                in_ = node.outputs[0]
                cross_ = node.o(0).o(0).o(0).o(1).o(0).o(0).inputs[1]
                mask_ = node.o(2).o(0).o(0).o(0).o(0).i(0).i(0).i(0).i(0).inputs[0]
                out_ = node.o(2).o(0).o(0).o(0).o(0).o(0).o(0).o(0).o(0).o(0).outputs[0]

                pluginNode = gs.Node("CSMHA", "CSMHA_%d" % nCSMHA, inputs=[in_,cross_,mask_], outputs=[out_], attrs={"k": k, "n": n, "qw": qw, "qb": qb, "kw": kw, "kb": kb, "vw": vw, "vb": vb, "ow": ow, "ob": ob, "qkb": qkb, "qcb": qcb})
                pluginNode.version = '1'
                graph.nodes.append(pluginNode)

                node.o(2).o(0).o(0).o(0).o(0).o(0).o(0).o(0).o(0).o(0).outputs.clear()

                print('Fused CSMHA [{}], begin with [{}]'.format(nCSMHA,node.name))
        except:
            continue

    graph.cleanup()
    onnx.save(gs.export_onnx(graph), onnxSurgeonFile)



if __name__ == '__main__':

    onnxFile = '../2_baseline/baseline.onnx'
    onnxSurgeonFile = 'model-csmha.onnx'

    print('--------------- Fusing Onnx:[{}] ---------------'.format(onnxFile))
    fusing(onnxFile, onnxSurgeonFile)