from utils import Text2Speech
import wave
import numpy as np

text1 = 'I am excited to participate in the NVIDIA Tensor R T Accelerated A I Inference Hackathon.'
text2 = 'This net is an end to end speech processing toolkit.'
text3 = 'This is the edvince team.'

text2speech = Text2Speech()


waveData1, trueData1 = text2speech(text1)
print('text:',trueData1['text'].shape,'wav:',trueData1['wav'].shape,'length',trueData1['y_length'])
np.savez('/root/trt2022_espnet/espnet-trt/0_data/data1.npz',**trueData1)
waveDataBytes = waveData1 / np.abs(waveData1).max()
waveDataBytes = 32768.0 * waveDataBytes
waveDataBytes = waveDataBytes.astype(np.int16).tolist()
waveDataBytes = [i.to_bytes(2, byteorder='little', signed=True) for i in waveDataBytes]
waveFile=wave.open('/root/trt2022_espnet/espnet-trt/0_data/onnx_test1.wav','wb')
waveFile.setparams((1,2,22050,len(waveDataBytes),'NONE','Tsinghua'))
waveFile.writeframes(b''.join(waveDataBytes))
waveFile.close()


waveData2, trueData2 = text2speech(text2)
print('text:',trueData2['text'].shape,'wav:',trueData2['wav'].shape,'length',trueData2['y_length'])
np.savez('/root/trt2022_espnet/espnet-trt/0_data/data2.npz',**trueData2)
waveDataBytes = waveData2 / np.abs(waveData2).max()
waveDataBytes = 32768.0 * waveDataBytes
waveDataBytes = waveDataBytes.astype(np.int16).tolist()
waveDataBytes = [i.to_bytes(2, byteorder='little', signed=True) for i in waveDataBytes]
waveFile=wave.open('/root/trt2022_espnet/espnet-trt/0_data/onnx_test2.wav','wb')
waveFile.setparams((1,2,22050,len(waveDataBytes),'NONE','Tsinghua'))
waveFile.writeframes(b''.join(waveDataBytes))
waveFile.close()


waveData3, trueData3 = text2speech(text3)
print('text:',trueData3['text'].shape,'wav:',trueData3['wav'].shape,'length',trueData3['y_length'])
np.savez('/root/trt2022_espnet/espnet-trt/0_data/data3.npz',**trueData3)
waveDataBytes = waveData3 / np.abs(waveData3).max()
waveDataBytes = 32768.0 * waveDataBytes
waveDataBytes = waveDataBytes.astype(np.int16).tolist()
waveDataBytes = [i.to_bytes(2, byteorder='little', signed=True) for i in waveDataBytes]
waveFile=wave.open('/root/trt2022_espnet/espnet-trt/0_data/onnx_test3.wav','wb')
waveFile.setparams((1,2,22050,len(waveDataBytes),'NONE','Tsinghua'))
waveFile.writeframes(b''.join(waveDataBytes))
waveFile.close()