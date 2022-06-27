cd 1_convert/
python3 export.py
python3 ioslove.py

cd ../2_baseline/
bash slove.sh
bash build.sh
python3 testOnnx_and_generateData.py
python3 testTRT_and_generateWave.py

cd ../3_layernorm-plugin/
cd LayerNormPlugin/
make
cp LayerNormPlugin.so ..
cd ..
python3 fuse_layernorm.py
bash build.sh
python3 testTRT.py

cd ..

