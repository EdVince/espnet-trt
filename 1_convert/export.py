from espnet_onnx.export import TTSModelExport
import os

tag_name = 'kan-bayashi/ljspeech_vits'
zip_file = '/root/trt2022_espnet/tts_train_vits_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave.zip'

m = TTSModelExport(cache_dir='./')
m.export_from_pretrained(tag_name=tag_name, zip_file=zip_file)
