# Pytorch转Onnx

### pytorch模型

1. 转换模型用到了两个仓库：[espnet](https://github.com/espnet/espnet)和[espnet_onnx](https://github.com/Masao-Someki/espnet_onnx)，这两个仓库已经手动放到了本目录下并进行了一定的修复和修改
2. 原始模型在外网，已经手动下载并放到了```/root/trt2022_espnet/tts_train_vits_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave.zip```


### 运行脚本

1. export.py会完成pytorch->onnx的转换
2. ioslove.py会融合onnx的输入节点


### 修复以下问题点以确保模型能在pytorch->onnx->trt中成功转换

 - RandomNormalLike
    - espnet_onnx/export/tts/models/tts_models/vits.py的noise_scale位置
    - espnet2/gan_tts/vits/duration_predictor.py的noise_scale位置
 - NonZero
    - espnet2/gan_tts/vits/transform.py的unconstrained_rational_quadratic_spline位置
 - compute shape tensor
    - espnet_onnx/export/tts/models/tts_models/vits.py的y_lengths的make_pad_mask位置
 - float range operator
    - espnet2/gan_tts/vits/generator.py的torch.arange位置
 - if operator
    - 砍掉网络中att_w和dur的输出
 - matrices with mismatching dimensions
    - 用onnx_graphsurgeon把text_length输入改成从text获取
