# 基于espnet的文本转语音模型的TensorRT部署与加速

本仓库是 “英伟达TensorRT加速AI推理 Hackathon 2022 —— Transformer模型优化赛” 的 “edvince” 队伍的复赛仓库。

## 总述

使用TensorRT部署及加速espnet的文本转语音(TTS)模型。

 - 原始模型：[espnet](https://github.com/espnet/espnet)
 - 优化效果：把模型在trt上跑起来了，优化提升基本没有
 - 运行步骤：```bash script.sh```

## 原始模型

### 模型简介

 - 用途以及效果：
    - 用途：espnet是一个端到端的语音处理工具链，支持ASR(自动语音识别)，TTS(文字转语音)，SE(语音增强)，ST(语音翻译)，MT(机器翻译)，VC(语音转换)，SLU(语音语言理解)，SUM(语音摘要)。与wenet官方提供了完善的部署流程不同，espnet没有提供，因此对espnet进行部署和加速是十分有价值的。本工作主要部署加速espnet的TTS(文本转语音)模型。
    - 效果：
        - 文本：**"I am excited to participate in the NVIDIA Tensor R T Accelerated A I Inference Hackathon."**
        - espnet的TTS结果：

https://user-images.githubusercontent.com/18224516/170708125-259bb48e-6279-4d14-ae0a-39b8c9763b18.mov

 - 业界应用情况：ESPnet在github上已经收获了5.1k的star，可见其受欢迎程度。由于ESPnet是一个非流式模型，因此其常被用于离线识别，已知的有京东等公司在使用。
 - 模型结构：本仓库部署espnet(工具名字)的TTS(任务名字)的VITS(具体模型)，模型的Pytorch定义在[这](https://github.com/espnet/espnet/blob/5fa6dcc4e649dc66397c629d0030d09ecef36b80/espnet2/gan_tts/vits/vits.py#L52)，VITS是一个“端到端文本转语音的带对抗学习的条件变分自动编码器”，paper在[这](https://arxiv.org/abs/2006.04558)，模型结构如下图所示，是一个结构比较均衡的网络，前端编码器是Transformer，后端解码是1D的卷积操作，是一个综合性很强的网络。

![image](./resources/VITS.png)

### 模型优化的难点

 - 存在大量的动态shape问题，模型中存在随机数节点以实现输出语音的差异性
 - 前端transformer编码部分的特征大小较小
 - 音频长度是模型计算得到的，启用fp16容易导致音频长度计算不准确，倒是无法定量比较精度

## 优化过程

 - 固定模型的推理shape
     - 描述：模型会输出int32的音频长度和具有音频长度大小的float32数组的音频向量。也就是说，模型最后的输出是动态的，其具体长度由模型计算得到，这样**trt因为动态shape的关系，无法转换模型**。在pytorch中，保持音频长度的输出不变，但把音频向量的计算手动设置为最大值，是的模型每次都全部计算最长音频，再将最终结果截断至计算得到的音频长度。是的trt能成功转换。
 - layernorm插件
     - 描述：融合零碎的layernorm算子为整体，并编写plugin进行实现。

## 精度与加速效果

 - 精度

| Text Length | max abs  | mean abs | med abs  | max rel  | mean rel | med rel   |
| ----------- | -------- | -------- | -------- | -------- | -------- | --------- |
| 68          | 1.372e-01| 8.823e-04| 1.761e-04| 1.937e+02| 6.684e-02| 8.132e-03 |
| 37          | 6.380e-02| 3.201e-04| 6.455e-05| 2.717e+01| 2.629e-02| 5.624e-03 |
| 16          | 7.508e-03| 3.061e-04| 1.260e-04| 2.040e+01| 2.392e-02| 5.895e-03 |

 - 速度

***基础模型的速度***

| Text Length | Latency(ms) | throughput (word/s) |
| ----------- | ----------- | ------------------- |
| 68          | 24.854      | 2.736e+03           | 
| 37          | 24.739      | 1.496e+03           | 
| 16          | 24.640      | 6.493e+02           |

***优化模型的速度***

| Text Length | Latency(ms) | throughput (word/s) |
| ----------- | ----------- | ------------------- |
| 68          | 24.975      | 2.723e+03           | 
| 37          | 24.835      | 1.490e+03           | 
| 16          | 24.642      | 6.493e+02           |

***优化前后速度没变***


## 经验与体会

1. 动态shape的问题不好处理，能把shape固定下来的还是要尽量固定下来。