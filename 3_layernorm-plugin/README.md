# LayerNorm Plugin

脚本：

1. check_layernorm.py：检查模型中那些节点位置是layernorm
2. check_shape.sh：检查模型中layernorm节点的输入特征的尺寸
3. fuse_layernorm.py：融合替换模型中零碎的layernorm节点
4. LayerNormPlugin/Makefile：编译生成LayerNormPlugin.so
5. build.sh：使用LayerNormPlugin.so来生成engine
6. testTRT.py：测试带LayerNormPlugin的engine的性能

速度(麻了，就快了那么0.5ms的样子)：

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
  68,  24.045,2.828e+03,1.372e-01,8.132e-03,   130304,   130304,    
  37,  23.807,1.554e+03,6.380e-02,5.624e-03,    79872,    79872,    
  16,  23.687,6.755e+02,7.508e-03,5.895e-03,    31744,    31744,  
```