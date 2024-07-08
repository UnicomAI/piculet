# -*- coding: utf-8 -*-

from flask import Flask,render_template,request,redirect,url_for,send_from_directory
import sys
sys.path.append("/root/exec/")
import os
os.environ['CUDA_VISIBLE_DEVICES']='6'
import time
import base64
from PIL import Image
import cv2
#token验证，根据需要判断要不要加
#import jwt
import numpy as np
import json
import threading
import logging
import random
import string
from logging.handlers import TimedRotatingFileHandler
#引入调用的算法
from save_history import save_info
from transformers import set_seed 
set_seed(1234)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import requests

logger = logging.getLogger('plantqwen')
logger.setLevel(logging.INFO)
ch = TimedRotatingFileHandler("./logs/qwenvl.log", when='D', encoding="utf-8")
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

mutex = threading.Lock()
def generate_random_str(randomlength=16):
    str_list = [random.choice(string.digits + string.ascii_letters) for i in range(randomlength)]
    random_str = ''.join(str_list)
    return random_str
    
device='cuda:0'

model_path ='path/to/Qwen/Qwen-VL-Chat'
data_path="./data/"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True, do_sample=False)
port = 49021

app = Flask(__name__)


def process_hallucination(img):
    t1 = time.time()
    url = "http://127.0.0.1:5015/face_recognition"
    f = open(img, 'rb')
    files = {'imagefile': ('hg.jpeg', f, 'image/jpeg')}
    ret = requests.post(url=url, files=files)
    f.close()
    faces = []
    unknow = []
    if ret.status_code == 200:
        for item in json.loads(ret.text)['info']['recognition_results']:
            if item['id']!='unknown':
                faces.append(item['id'])
            else:
                unknow.append('unknown')
    prompt_face=''
    if len(faces)>0:
        if len(faces) ==1:
            prompt_face = 'The celebrity in the image is '+'、'.join(faces)+'。\n'
        else:
            prompt_face = 'The celebrities in the image are '+'、'.join(faces)+'。\n'

    
    url = "http://127.0.0.1:5001/ocr"
    f = open(img, 'rb')
    files = {'imagefile': ('testocr_2.jpg', f, 'image/jpeg')}
    ret = requests.post(url=url, files=files, verify=False)
    f.close()
    content = ''
    if ret.status_code == 200:
        for text in json.loads(ret.text)['info']['words_results']:
            content+=text['words']+'\n'
    prompt_ocr = ''
    if len(content)>0:

        prompt_ocr = 'The text content contained in the image.:\n'+ content+'\n'
      
    url = "http://127.0.0.1:5012/detect"
    f = open(img, 'rb')
    files = {'imagefile': ('test.jpg', f, 'image/jpeg')}

    ret = requests.post(url=url, files=files)
    f.close()
    content=''
    if ret.status_code == 200:
        items = json.loads(ret.text)['info']
        num_info={}
        if len(items)>0:
            for obj in items.split(','):
                if obj in num_info.keys():
                    num_info[obj]+=1
                else:
                    num_info[obj]=1
            content='the image contains these objects:'
            for k,v in num_info.items():
                # content+=str(v)+'个'+tags_ez[k]+','
                content+='there is '+str(v) +' '+k+'; ' if v==1 else 'there are '+str(v)+' '+k+';'
            content = content[:-1]+'\n'
    # print("prompt_face:{}, ,prompt_ocr:{}, ,conten:{}".format(prompt_face,prompt_ocr,content))
    return prompt_face,prompt_ocr,content


def process_hallucination_detectOnly(img):
    url = "http://127.0.0.1:5012/detect"
    f = open(img, 'rb')
    files = {'imagefile': ('test.jpg', f, 'image/jpeg')}
    ret = requests.post(url=url, files=files)
    f.close()
    content=''
    if ret.status_code == 200:
        items = json.loads(ret.text)['info']
        num_info={}
        if len(items)>0:
            for obj in items.split(','):
                if obj in num_info.keys():
                    num_info[obj]+=1
                else:
                    num_info[obj]=1
            content='the image contains these objects:'
            for k,v in num_info.items():
                content+='there is '+str(v) +' '+k+'; ' if v==1 else 'there are '+str(v)+' '+k+';'
            content = content[:-1]+'\n'
    # print("prompt_face:{}, ,prompt_ocr:{}, ,conten:{}".format(prompt_face,prompt_ocr,content))
    return content

#定义一个url地址
@app.route("qwen_server",methods=['POST']) #表明是post服务
def check():
    logger.info("--------------------------------------------")
    print('------------------------------------------------')
    auth=True
    bsave_info=True
    code=7
    message=''
    res={}

    # Post method
    if request.method == 'POST':
        time_start=time.time()

        if auth:
            dateTime = time.strftime("%Y%m%d-%H.%M.%S")
            t = time.strftime("%Y%m%d")
            img_path=os.path.join(data_path, t, generate_random_str(24)+'.jpg')
            prompt = request.form['prompt']
            print(request.files)
            if 'img' in request.files.keys():
                f = request.files['img']
                if not os.path.exists(os.path.join(data_path, t)):
                    os.makedirs(os.path.join(data_path, t))
                f.save(img_path)

            logger.info(prompt)
            
            time0=time.time()
            mutex.acquire()
            try:
                if 'img' in request.files.keys():
                    prompt_face,prompt_ocr,content = process_hallucination(img_path)
                    # content = process_hallucination_detectOnly(img_path)
                    
                    prompt_temp =  'You are a highly intelligent visual language expert, please fully understand the image and answer the question based on the above content:\n'
                    total_prompt = prompt_ocr+prompt_face+content+prompt_temp+prompt
                    # face
                    # total_prompt = prompt_ocr+content+prompt_temp+prompt
                    # #no Detect
                    # total_prompt = prompt_ocr+prompt_face+prompt_temp+prompt
                    #OCR
                    # total_prompt = prompt_face+content+prompt_temp+prompt
                    # detect only
                    # total_prompt = content+prompt_temp+prompt
                    # total_prompt = prompt_temp+prompt  # no auxiliary models
                    query = tokenizer.from_list_format([
                        {'image': img_path}, # Either a local path or an url
                        {'text': total_prompt},
                    ])
                    print(total_prompt)
                else:
                    query = tokenizer.from_list_format([
                        {'text': prompt},
                    ])
                response, history = model.chat(tokenizer, query=query, history=None, do_sample=False)

                print(response)

                res['text'] = response
                
                code = 0
                message="success"

            except Exception as e:
                print("Unexpected Error:{}".format(e))
                logger.info("Unexpected Error:{}".format(e))
                message=str(e)

            time1=time.time()
            logger.info("cost time: "+str(time1-time0))
        #返回jsonarr，就是算法的结果
        arr={}
        arr['code']=code
        arr['message']=message
        arr['result']=res
        jsonarr = json.dumps(arr,ensure_ascii=False)
        logger.info({"code":code,'message':message,'result':res})
        time_end=time.time()
        logger.info("total cost time: "+str(time_end-time_start))

        mutex.release()
        return jsonarr

    else:
        logger.info("not post method")
        code=2
        message="request failed"
        arr={}
        arr['code']=code
        arr['message']=message
        arr['result']=res
        jsonarr = json.dumps(arr)
        return jsonarr


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port,threaded=True)
