# import base64
import datetime
import hashlib
import hmac
import json
import sys
import wave
import time

# import jwt
import requests

# 通用文字识别
url = "http://10.1.3.5:5002/ocr"
# headers = {
#            "Authorization": "Bearer {}".format(jwt_token)
# }
files = {'imagefile': ('testocr_2.jpg', open('testocr_2.jpg', 'rb'), 'image/jpeg')}

ret = requests.post(url=url, files=files, verify=False)
if ret.status_code == 200:
    text = json.loads(ret.text)
    print(text)

