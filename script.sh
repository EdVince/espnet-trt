cd 1_convert/
python3 export.py
python3 ioslove.py

cd ../2_baseline/
bash script1.sh
bash script2.sh
python3 testOnnx_and_generateData.py
python3 testTRT_and_generateWave.py
