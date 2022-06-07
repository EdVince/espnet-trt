# Onnx转TRT

### 运行脚本

1. script1.sh会调用polygraphy做常量折叠，不折叠的话转trt会报错
2. testOnnx_and_generateData.py会调用常量折叠后的onnx进行三个句子的推理生成，会在0_data目录生成真实输入输出数据和真实音频
3. script2.sh会调用trtexec生成基础的trt模型
4. testTRT.py会调用第三步生成的trt模型，使用第二步onnx中生成的真实数据来测试误差
5. testTRT_and_generateWav.py会调用trt模型，生成真实音频

PS：可通过修改testOnnx_and_generateData.py中的三个测试text来生成不同的测试数据给testTRT用

baseline运行结果：
```
tl: Text Length
lt: Latency (ms)
tp: throughput (word/s)
aw: maximum of absolute difference of wav
rw: median of relative difference of wav
gl: ground truth of wav length
pl: prediction of wav length
----+--------+---------+---------+---------+---------+---------+---------
  tl|      lt|       tp|       aw|       rw|       gl|   pd len|   Check
----+--------+---------+---------+---------+---------+---------+---------
  68,  24.327,2.795e+03,1.021e-01,7.447e-03,   130304,   130304,    
  37,  24.170,1.531e+03,4.180e-02,6.257e-03,    79872,    79872,    
  16,  24.092,6.641e+02,1.743e-02,6.678e-03,    31744,    31744,    
```