trtexec --onnx=baseline.onnx \
 --saveEngine=baseline.plan \
 --minShapes=text:4 \
 --optShapes=text:36 \
 --maxShapes=text:68

# trtexec --onnx=baseline.onnx \
#  --saveEngine=baseline-fp16.plan \
#  --minShapes=text:4 \
#  --optShapes=text:36 \
#  --maxShapes=text:68 \
#  --fp16