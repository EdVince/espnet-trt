trtexec --onnx=model-csmha.onnx \
 --saveEngine=model-csmha.plan \
 --minShapes=text:4 \
 --optShapes=text:36 \
 --maxShapes=text:68 \
 --verbose \
 --plugins=CSMHAPlugin.so