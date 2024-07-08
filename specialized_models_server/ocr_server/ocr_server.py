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

#    string.digits=0123456789
#    string.ascii_letters=abcdefghigklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ

    str_list = [random.choice(string.digits + string.ascii_letters) for i in range(randomlength)]
    random_str = ''.join(str_list)
    return random_str

from paddleocr import PaddleOCR, draw_ocr

# # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
# img_path = './imgs/11.jpg'
# result = ocr.ocr(img_path, cls=True)
# for idx in range(len(result)):
#     res = result[idx]
#     for line in res:
#         print(line)

# # 显示结果
# from PIL import Image
# result = result[0]
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')
UPLOAD_PATH = os.path.join(os.path.dirname(__file__),'images')
import time
@app.route('/ocr', methods=['POST'])
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
        text_js=''
        text=''
        try:
            result = ocr.ocr(upload_path, cls=False)
            # boxes = [line[0] for line in result]
            # txts = [line[1][0] for line in result]
            # text = write2json(boxes,txts)
            # print(text)
            results = []
            for idx in range(len(result)):
                res = result[idx]
                if res is not None:
                    for line in res:
                        axis = line[0]
                        word = line[1][0]
                        
                        loc = {"x1": str(axis[0][0]), "y1": str(axis[0][1]), "x2": str(axis[1][0]), "y2": str(axis[1][1]), "x3": str(axis[2][0]), "y3": str(axis[2][1]), "x4": str(axis[3][0]), "y4": str(axis[3][1])}
                        tmp_json = {"location": loc, "words": word}
                        results.append(tmp_json)
            text_js = {"words_results_num":len(results), "words_results":results}
            # text_js = json.loads(whole)
            #stus['time0']=end1-start1
        except Exception as e:
            print("Unexpected Error: {}".format(e))
            stus['info']=''.format(e)
            stus['code']=1
        lock.release()
        time1=time.time()
    #               print text
        print("cost time: ",time1-time0)
    #               stus["time1"]=time00-time01
    #               stus["time2"]=time1-time0
        stus['info']=text_js
        rz=json.dumps(stus,ensure_ascii=False, indent=4)
    
    return rz


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, threaded=False)
