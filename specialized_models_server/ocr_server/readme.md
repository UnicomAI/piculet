
### installation:
pip install -r requirements.txt

### run:
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=./cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib/:$LD_LIBRARY_PATH python -u ocr_server.py >out.log 2>&1 &


### test:
test.py

