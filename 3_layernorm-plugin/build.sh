# trtexec --onnx=model-layernorm.onnx \
#  --saveEngine=model-layernorm.plan \
#  --minShapes=text:4 \
#  --optShapes=text:36 \
#  --maxShapes=text:68 \
#  --plugins=LayerNormPlugin.so

trtexec --onnx=model-layernorm.onnx \
 --saveEngine=model-layernorm-fp16.plan \
 --minShapes=text:4 \
 --optShapes=text:36 \
 --maxShapes=text:68 \
 --fp16 \
 --plugins=LayerNormPlugin.so