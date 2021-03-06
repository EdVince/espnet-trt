import os
import sys
import ctypes
import numpy as np
from glob import glob 
from time import time_ns
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt
import wave


dataFilePath = "/root/trt2022_espnet/espnet-trt/0_data/"
ttsPlanFile  = "/root/trt2022_espnet/espnet-trt/3_layernorm-plugin/model-layernorm.plan"
ttsScoreFile = "/root/trt2022_espnet/espnet-trt/3_layernorm-plugin/Score.txt"
soFileList = glob("./*.so")

tableHead = \
"""
tl: Text Length
lt: Latency (ms)
tp: throughput (word/s)
gl: ground truth of wav length
pl: prediction of wav length
----+--------+---------+---------+---------+---------+---------+---------+---------+---------+---------
  tl|      lt|       tp|  max abs| mean abs|  med abs|  max rel| mean rel|  med rel|   pd len|   Check
----+--------+---------+---------+---------+---------+---------+---------+---------+---------+---------
"""

def check(a, b, weak=False, epsilon = 1e-5):

    if weak:
        res = np.all( np.abs(a - b) < epsilon )
    else:
        res = np.all( a == b )

    _abs_ = np.abs(a - b)
    _rel_ = np.abs(a - b) / (np.abs(b) + epsilon)

    return res,np.max(_abs_),np.mean(_abs_),np.median(_abs_),np.max(_rel_),np.mean(_rel_),np.median(_rel_)

#-------------------------------------------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')

if len(soFileList) > 0:
    print("Find Plugin %s!"%soFileList)
else:
    print("No Plugin!")
for soFile in soFileList:
    ctypes.cdll.LoadLibrary(soFile)

#-------------------------------------------------------------------------------
with open(ttsScoreFile, 'w') as f:

    if os.path.isfile(ttsPlanFile):
        with open(ttsPlanFile, 'rb') as ttsF:
            engine = trt.Runtime(logger).deserialize_cuda_engine(ttsF.read())
        if engine is None:
            print("Failed loading %s"%ttsPlanFile)
            exit()
        print("Succeeded loading %s"%ttsPlanFile)
    else:
        print("Failed finding %s"%ttsPlanFile)
        exit()

    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    context = engine.create_execution_context()

    print(tableHead)

    npzs = sorted(glob(dataFilePath + "*.npz"))
    for numFile in range(len(npzs)):
        ioFile = npzs[numFile]

        ioData = np.load(ioFile,allow_pickle=True)
        text = ioData['text']
        textLength = text.shape[0]

        context.set_binding_shape(0, text.shape)

        bufferH = []
        bufferH.append( text.astype(np.int32).reshape(-1) )
        for i in range(nInput, nInput + nOutput):
            bufferH.append( np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))) )

        bufferD = []
        for i in range(nInput + nOutput):                
            bufferD.append( cudart.cudaMalloc(bufferH[i].nbytes)[1] )

        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        context.execute_v2(bufferD)

        for i in range(nInput, nInput + nOutput):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # warm up
        for i in range(10):
            context.execute_v2(bufferD)

        # test infernece time
        t0 = time_ns()
        for i in range(30):
            context.execute_v2(bufferD)
        t1 = time_ns()
        timePerInference = (t1-t0)/1000/1000/30

        indexTTSOut = engine.get_binding_index('wav')
        indexTTSOutLens = engine.get_binding_index('y_length')

        gt_len = ioData['y_length'][0]
        pd_len = bufferH[indexTTSOutLens][0]

        crash = True
        if gt_len == pd_len:
            crash = False
            check0 = check(bufferH[indexTTSOut][:gt_len],ioData['wav'][:gt_len],True,5e-5)
        else:
            check0 = check(bufferH[indexTTSOut][:min(gt_len,pd_len)],ioData['wav'][:min(gt_len,pd_len)],True,5e-5)

        string = "%4d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9d,%9d"%(
                                                textLength,
                                                timePerInference,
                                                textLength/timePerInference*1000,
                                                check0[1],check0[2],check0[3],
                                                check0[4],check0[5],check0[6],
                                                gt_len,
                                                pd_len)
        print(string+",   %s"%("Crash" if crash is True else " "))
        f.write(string + "\n")

        for i in range(nInput + nOutput):                
            cudart.cudaFree(bufferD[i])