# 基于espnet的文本转语音模型的TensorRT部署与加速

本仓库是 “英伟达TensorRT加速AI推理 Hackathon 2022 —— Transformer模型优化赛” 的 “edvince” 队伍的复赛仓库。

## 总述

使用TensorRT部署及加速espnet的文本转语音模型。

 - 原始模型：[espnet](https://github.com/espnet/espnet)
 - 优化效果：暂无
 - 运行步骤：暂无

## 原始模型

### 模型简介

 - 用途以及效果：
    - 用途：[espnet](https://github.com/espnet/espnet)是一个端到端的语音处理工具链，与[wenet](https://github.com/wenet-e2e/wenet)类似，不同的是wenet官方提供了完善的部署流程，而espnet没有，因此对espnet进行部署和加速是十分有必要的。本工作主要部署加速espnet的文本转语音(Text2Speech, 缩写tts)。
    - 效果：文本："I am excited to participate in the NVIDIA Tensor R T Accelerated A I Inference Hackathon."，生成的语音：

https://user-images.githubusercontent.com/18224516/170708125-259bb48e-6279-4d14-ae0a-39b8c9763b18.mov

 - 业界应用情况：

### 模型优化的难点

