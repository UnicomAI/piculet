We utilize `specialized traditional light-weight deep learning models` to detect factual information of input image, then formulate these descriptions, which, alongside the user’s query and image, are input into MLLMs. MLLMs, given the formulated input, then generate results with reduced hallucinations. Our method utilizes these specialized models to generate factual external knowledge apart from the single input image, which provides a reliable basis for decision-making in the outputs of the MLLMs.


### installation:
```bash
conda create -n specialized_models python==3.9.0
conda activate specialized_models
pip install -r requirements.txt
```

### run:  
#### prerequisite:
- download `cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive` and place it somewhere.
- download `PaddleDetection-release-2.7` and place it under [detect_server](./detect_server/)

#### 1.ocr_server:
```bash
cd ocr_server
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=path/to/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib/:$LD_LIBRARY_PATH python -u ocr_server.py >out.log 2>&1 &
```
#### 2.face_server

Note: The face recognition service requires models from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#model-zoo), users can follow the link and download the `glint360k_cosface_r100_fp16_0.1`'s `backbone.pth`, and utilize our [specialized_models_server/face_server/iresnet.py](specialized_models_server/face_server/iresnet.py) to convert it into onnx model, and input the correct path in `specialized_models_server/face_server/gender_age/align_img_gender_age.py`. (We are using a proprietary facial model different from the insightface; therefore, if you employ this public model, the results you measure may vary.)

And then you need to manually upload the face library. The code for the face_upload is in [specialized_models_server/face_server/gender_age/facetest.py](specialized_models_server/face_server/gender_age/facetest.py).

Once the above steps are completed, you can use the following command to start the face service:

```bash
cd face_server
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=path/to/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib/:$LD_LIBRARY_PATH python -u face_server.py >out.log 2>&1 &
```


#### detect_server
```bash
cd detect_server
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=path/to/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib/:$LD_LIBRARY_PATH python -u yoloe_server.py >out.log 2>&1 &
```

