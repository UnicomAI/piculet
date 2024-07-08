# import base64
import datetime
import hashlib
import hmac
import json
import sys
import wave
import time

import requests

#注册
url = "http://127.0.0.1:5015/face_upload"

files = {'imagefile': ('hg.jpeg', open('hg.jpeg', 'rb'), 'image/jpeg')}
Data = {'id': 'HuGe'}
ret = requests.post(url=url, files=files,data=Data)
print(ret.text)

#识别
url = "http://127.0.0.1:5015/face_recognition"
files = {'imagefile': ('hg.jpeg', open('hg.jpeg', 'rb'), 'image/jpeg')}
ret = requests.post(url=url, files=files)
print(ret.text)