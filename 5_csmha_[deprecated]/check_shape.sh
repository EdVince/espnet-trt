# polygraphy run /root/trt2022_espnet/espnet-trt/2_baseline/baseline.onnx --onnxrt \
#     --input-shapes test:[68] \
#     --onnx-outputs onnx::Add_1381 onnx::Add_1454 onnx::Div_1455 matrix_bd.3

polygraphy run /root/trt2022_espnet/espnet-trt/2_baseline/baseline.onnx --onnxrt \
    --input-shapes text:[16] \
    --onnx-outputs matrix_bd.3 onnx::Concat_1403 onnx::Reshape_1404