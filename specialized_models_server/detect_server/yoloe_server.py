# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from yoloe_detect_eh import Yoloe

yoloe = Yoloe()

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob
import ast

import paddle

import cv2
#coding=utf8

import json
from flask import Flask, make_response, request
import os
import sys
app = Flask(__name__)
import random
import string

#app = Flask(__name__)

import time
import random
import string
#import exceptions
import threading
import json

lock = threading.Lock()

def write2json(bboxes, texts):

    results = []
    for axis, word in zip(bboxes, texts):
        # loc = {"x": axis[0], "y": axis[1], "width": axis[2], "height": axis[3]}
        tmp_json = {"location": axis, "words": word}
        results.append(tmp_json)

    whole = {"words_results_num":len(bboxes), "words_results":results}

    return whole

def generate_random_str(randomlength=16):
    str_list = [random.choice(string.digits + string.ascii_letters) for i in range(randomlength)]
    random_str = ''.join(str_list)
    return random_str

UPLOAD_PATH = os.path.join(os.path.dirname(__file__),'images')
import time
@app.route('/detect', methods=['POST'])
def check():

    text=""
    rz=""
    stus={'code':0,'info':''}
    if request.method == 'POST':
        f = request.files['imagefile']
        basepath = './'
        time00=time.time()
        upload_path = os.path.join(basepath, 'data',generate_random_str(24)+".jpg")
        f.save(upload_path)
        time01=time.time()
        print('save time',time00-time01)
        time0=time.time()

        print(upload_path)
        lock.acquire()
        text=''
        if True:
        # try:
            img = cv2.imread(upload_path)

            result = yoloe.detect(img)
            print(result)

            text = ','.join(result)
            print(text)
            # text_js = json.loads(whole)
            #stus['time0']=end1-start1
        # except Exception as e:
        #     print("Unexpected Error: {}".format(e))
        #     stus['info']=''.format(e)
        #     stus['code']=1
        lock.release()
        time1=time.time()
    #               print text
        print("cost time: ",time1-time0)
    #               stus["time1"]=time00-time01
    #               stus["time2"]=time1-time0
        stus['info']=text
        rz=json.dumps(stus,ensure_ascii=False, indent=4)
    
    return rz


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5011, threaded=False)
