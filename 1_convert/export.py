from espnet_onnx.export import TTSModelExport
import os

tag_name = 'kan-bayashi/ljspeech_vits'
zip_file = 'tts_train_vits_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave.zip'

if os.path.exists(zip_file):
    print('exporting from local zipfile')
    m = TTSModelExport(cache_dir='./')
    m.export_from_pretrained(zip_file=zip_file)
else:
    print('exporting from online tag')
    m = TTSModelExport(cache_dir='./')
    m.export_from_pretrained(tag_name=tag_name)