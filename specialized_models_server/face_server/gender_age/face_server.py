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
import pickle
import cv2
import numpy as np
from align_img_gender_age import FaceModel

net = FaceModel()
def test(net):
    img = cv2.imread('sjh_20221018162735.jpg')
    faces = net.getFeature(img)
    f1 = faces[0].feature
    f_list = []
    f_list.append(f1)
    f_list.append(f1)
    f_list = np.array(f_list)
    print(f_list.shape)
    out = np.dot(f_list,f1)
    print(out.shape)
test(net)

featureLib={}

def update(featureLib):
    if os.path.exists("face_data.pkl"):
        with open("face_data.pkl", "rb") as file:
            face_datas = pickle.load(file)
        featureslist=[]
        idlist = []
        for k,v in face_datas.items():
            idlist.append(k)
            featureslist.append(v)
        feature = np.array(featureslist)
        print("更新成功")
        featureLib['face_datas']=face_datas
        featureLib['idlist']=idlist
        featureLib['featureslist']=featureslist
        featureLib['feature']=feature

update(featureLib)
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

#    string.digits=0123456789
#    string.ascii_letters=abcdefghigklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ

    str_list = [random.choice(string.digits + string.ascii_letters) for i in range(randomlength)]
    random_str = ''.join(str_list)
    return random_str



import time
@app.route('/face_upload', methods=['POST'])
def upload():
    global face_datas,featureslist
    text=""
    rz=""
    stus={'code':0,'info':''}

    if request.method == 'POST':
        f = request.files['imagefile']
        id = request.form['id']
        print(id)
        basepath = './'
        time00=time.time()
        upload_path = os.path.join(basepath, 'facelib',generate_random_str(24)+".jpg")
        f.save(upload_path)
        time01=time.time()
        print('save time',time00-time01)
        time0=time.time()

        print(upload_path)
        lock.acquire()
        text_js=''
        text=''
        try:
            img = cv2.imread(upload_path)
            faces = net.getFeature(img)
            if len(id)>0 and len(faces)>0:
                face_datas = featureLib['face_datas']
                face_datas[id] = faces[0].feature
                with open("face_data.pkl", "wb") as file:
                    pickle.dump(face_datas,file)
                update(featureLib)
                
        except Exception as e:
            print("Unexpected Error: {}".format(e))
            stus['info']=''.format(e)
            stus['code']=1
        lock.release()
        time1=time.time()
        print("cost time: ",time1-time0)
        stus['info']=text_js
        rz=json.dumps(stus,ensure_ascii=False, indent=4)
    
    return rz


@app.route('/face_recognition', methods=['POST'])
def recognition():
    text=""
    rz=""
    stus={'code':0,'info':'','message':'success'}
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
        text_js=''
        text=''
        try:
            img = cv2.imread(upload_path)
            faces = net.getFeature(img)
            results=[]
            feature = featureLib['feature']
            idlist = featureLib['idlist']
            facesmap = {}
            for face in faces:
                facesmap[face.bbox[0]+face.bbox[1]]=face
            facesmap = sorted(facesmap.items(), key=lambda x: x[0])
            for _,face in facesmap:
                out = np.dot(feature,face.feature)
                sim = np.max(out)
                index = np.argmax(out)
                if sim>0.4:
                    reg_id = {'sim':str(sim), 'id':idlist[index]}
                else:
                    reg_id = {'sim':str(sim), 'id':'unknown'}
                results.append(reg_id)
            text_js = {"recognition_results":results}

        except Exception as e:
            print("Unexpected Error: {}".format(e))
            stus['info']=''.format(e)
            stus['code']=1
            stus['message']='fail'
        lock.release()
        time1=time.time()
        print("cost time: ",time1-time0)
        stus['info']=text_js
        rz=json.dumps(stus,ensure_ascii=False, indent=4)
    
    return rz


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5015, threaded=False)
