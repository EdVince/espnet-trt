trtexec --onnx=baseline.onnx \
 --saveEngine=baseline.plan \
 --minShapes=text:4 \
 --optShapes=text:36 \
 --maxShapes=text:68