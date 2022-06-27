#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import ctypes
import numpy as np
from cuda import cudart
import tensorrt as trt
import torch

soFile = "./CSMHAPlugin.so"
epsilon = 1.0e-3
m = 64
mm = 2*m-1
np.random.seed(2)

globalData = np.random.rand(1 * m * 192).astype(np.float32).reshape(1, m, 192) * 2.0 - 1.0
globalCross = np.random.randint(0,2,(1*2*96*mm)).astype(np.float32).reshape(1,2,96,mm)
globalMask = np.random.randint(0,2,(1*1*m)).astype(np.float32).reshape(1,1,m)

globalQWeight = np.random.rand(192 * 192).astype(np.float32).reshape(192, 192) * 2.0 - 1.0
globalQBias = np.random.rand(192).astype(np.float32).reshape(192) * 2.0 - 1.0
globalKWeight = np.random.rand(192 * 192).astype(np.float32).reshape(192, 192) * 2.0 - 1.0
globalKBias = np.random.rand(192).astype(np.float32).reshape(192) * 2.0 - 1.0
globalVWeight = np.random.rand(192 * 192).astype(np.float32).reshape(192, 192) * 2.0 - 1.0
globalVBias = np.random.rand(192).astype(np.float32).reshape(192) * 2.0 - 1.0
globalOWeight = np.random.rand(192 * 192).astype(np.float32).reshape(192, 192) * 2.0 - 1.0
globalOBias = np.random.rand(192).astype(np.float32).reshape(192) * 2.0 - 1.0
globalQKBias = np.random.rand(192).astype(np.float32).reshape(192) * 2.0 - 1.0
globalQCBias = np.random.rand(192).astype(np.float32).reshape(192) * 2.0 - 1.0

def printArrayInfo(x, description=""):
    print( '%s: %s\n  Mean=%.5e,SumAbs=%.5e,Var=%.5e,Max=%.5f,Min=%.5f,SAD=%.5e'%( \
        description,str(x.shape),np.mean(x),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print("\t", x.reshape(-1)[:16])

def check(a, b, weak=False):
    if weak:
        res = np.all(np.abs(a - b) < epsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.mean(np.abs(a - b) / (np.abs(b) + epsilon))
    print("check:", res, "maxAbsDiff:", diff0, "meanRelDiff:", diff1)

def CSMHACPU(input, cross, mask, qw, qb, kw, kb, vw, vb, ow, ob, qkb, qcb):

    # q
    _1 = np.matmul(input, qw) + qb
    # k
    _2 = np.matmul(input, kw) + kb
    # v
    _3 = np.matmul(input, vw) + vb

    _4 = _1.reshape(1,m,2,96)
    # qk
    _5 = _4 + qkb.reshape(2,96)
    _5 = _5.transpose(0,2,1,3) # transposeQ

    _6 = _2.reshape(1,m,2,96)
    _6 = _6.transpose(0,2,3,1) # transposeK
    _7 = torch.matmul(torch.from_numpy(_5),torch.from_numpy(_6)).numpy()

    # qc
    _8 = _4 + qcb.reshape(2,96)
    _8 = _8.transpose(0,2,1,3)

    _9 = torch.matmul(torch.from_numpy(_8),torch.from_numpy(cross)).numpy()
    _91 = np.zeros((1,2,m,1))
    _92 = np.concatenate([_91,_9.reshape(1,2,m,mm)],axis=3)
    _93 = _92.reshape(1,2,2*m,m)
    _94 = _93[:,:,1:,:]
    _95 = _94.reshape(1,2,m,2*m-1)
    _96 = _95[:,:,:,:m]

    _10 = _7 + _96
    _11 = _10 / 9.797959327697754

    mask = np.expand_dims(mask,1).repeat(2,1).repeat(m,2)
    _11[mask==0] = -3.4028234663852886e+38
    _12 = torch.softmax(torch.from_numpy(_11),-1).numpy()

    _13 = _3.reshape(1,m,2,96)
    _13 = _13.transpose(0,2,1,3)
    _14 = torch.matmul(torch.from_numpy(_12).float(),torch.from_numpy(_13).float()).numpy()
    _14 = _14.transpose(0,2,1,3)
    _14 = _14.reshape(1,m,192)
    _15 = np.matmul(_14, ow) + ob

    return _15

def getCSMHAPlugin(qweight, qbias, kweight, kbias, vweight, vbias, oweight, obias, qkbias, qcbias):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'CSMHA':
            parameterList = []
            parameterList.append(trt.PluginField("qw", np.float32(qweight), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("qb", np.float32(qbias), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("kw", np.float32(kweight), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("kb", np.float32(kbias), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("vw", np.float32(vweight), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("vb", np.float32(vbias), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("ow", np.float32(oweight), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("ob", np.float32(obias), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("qkb", np.float32(qkbias), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("qcb", np.float32(qcbias), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("k", np.int32(qweight.shape[0]), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("n", np.int32(qweight.shape[1]), trt.PluginFieldType.INT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run():
    trtFile = "./model.plan"
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()

        inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, [1, m, 192])
        profile.set_shape(inputT0.name, [1, m, 192], [1, m, 192], [1, m, 192])
        inputT1 = network.add_input('inputT1', trt.DataType.FLOAT, [1,2,96,mm])
        profile.set_shape(inputT1.name, [1,2,96,mm], [1,2,96,mm], [1,2,96,mm])
        inputT2 = network.add_input('inputT2', trt.DataType.FLOAT, [1,1,m])
        profile.set_shape(inputT2.name, [1,1,m], [1,1,m], [1,1,m])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2([inputT0,inputT1,inputT2], 
                        getCSMHAPlugin(globalQWeight, globalQBias, 
                                        globalKWeight, globalKBias, 
                                        globalVWeight, globalVBias, 
                                        globalOWeight, globalOBias,
                                        globalQKBias, globalQCBias))
        pluginLayer.get_output(0).name = "CSMHA-Plugin-Output"

        network.mark_output(pluginLayer.get_output(0))
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, 'wb') as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine (engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0, [1, m, 192])
    context.set_binding_shape(1, [1,2,96,mm])
    context.set_binding_shape(2, [1,1,m])
    #print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("Bind[%2d]:i[%d]->" % (i, i) if engine.binding_is_input(i) else "Bind[%2d]:o[%d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    bufferH = []
    bufferH.append(globalData)
    bufferH.append(globalCross)
    bufferH.append(globalMask)
    # print(nInput,nOutput)
    for i in range(nOutput):
        bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nOutput):
        cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    outputCPU = CSMHACPU(bufferH[0], bufferH[1], bufferH[2], 
                            globalQWeight, globalQBias, 
                            globalKWeight, globalKBias, 
                            globalVWeight, globalVBias,
                            globalOWeight, globalOBias,
                            globalQKBias, globalQCBias)

    printArrayInfo(bufferH[-1], "TensorRT result")
    printArrayInfo(outputCPU, "CPU result")
    check(bufferH[-1], outputCPU, True)

    for buffer in bufferD:
        cudart.cudaFree(buffer)

if __name__ == '__main__':
    os.system('rm ./*.plan')
    np.set_printoptions(precision=3, linewidth=100, suppress=True)


    outputCPU = CSMHACPU(globalData, globalCross, globalMask, 
                        globalQWeight, globalQBias, 
                        globalKWeight, globalKBias, 
                        globalVWeight, globalVBias,
                        globalOWeight, globalOBias,
                        globalQKBias, globalQCBias)


    run()  # 创建 TensorRT 引擎并推理
    # run()  # 读取 TensorRT 引擎并推理

    # print("test finish!")
