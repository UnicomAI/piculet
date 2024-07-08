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

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob
import ast

import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_mlu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.slim import build_slim_model

from ppdet.utils.logger import setup_logger
logger = setup_logger('train')


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="Whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/image",
        help='VisualDL logging directory for image.')
    parser.add_argument(
        "--save_results",
        type=bool,
        default=False,
        help="Whether to save inference results to output_dir.")
    parser.add_argument(
        "--slice_infer",
        action='store_true',
        help="Whether to slice the image and merge the inference results for small object detection."
    )
    parser.add_argument(
        '--slice_size',
        nargs='+',
        type=int,
        default=[640, 640],
        help="Height of the sliced image.")
    parser.add_argument(
        "--overlap_ratio",
        nargs='+',
        type=float,
        default=[0.25, 0.25],
        help="Overlap height ratio of the sliced image.")
    parser.add_argument(
        "--combine_method",
        type=str,
        default='nms',
        help="Combine method of the sliced images' detection results, choose in ['nms', 'nmm', 'concat']."
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.6,
        help="Combine method matching threshold.")
    parser.add_argument(
        "--match_metric",
        type=str,
        default='ios',
        help="Combine method matching metric, choose in ['iou', 'ios'].")
    parser.add_argument(
        "--visualize",
        type=ast.literal_eval,
        default=True,
        help="Whether to save visualize results to output_dir.")
    args = parser.parse_args()
    return args


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def run(FLAGS, cfg):
    # build trainer
    trainer = Trainer(cfg, mode='test')

    # load weights
    trainer.load_weights(cfg.weights)

    # get inference images
    images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)

    # inference
    if FLAGS.slice_infer:
        trainer.slice_predict(
            images,
            slice_size=FLAGS.slice_size,
            overlap_ratio=FLAGS.overlap_ratio,
            combine_method=FLAGS.combine_method,
            match_threshold=FLAGS.match_threshold,
            match_metric=FLAGS.match_metric,
            draw_threshold=FLAGS.draw_threshold,
            output_dir=FLAGS.output_dir,
            save_results=FLAGS.save_results,
            visualize=FLAGS.visualize)
    else:
        trainer.predict(
            images,
            draw_threshold=FLAGS.draw_threshold,
            output_dir=FLAGS.output_dir,
            save_results=FLAGS.save_results,
            visualize=FLAGS.visualize)
import pickle

with open('FLAGS_yoloe.pkl', 'rb') as file:
    FLAGS = pickle.load(file)
cfg = load_config(FLAGS.config)
merge_args(cfg, FLAGS)
merge_config(FLAGS.opt)

# disable npu in config by default
if 'use_npu' not in cfg:
    cfg.use_npu = False

# disable xpu in config by default
if 'use_xpu' not in cfg:
    cfg.use_xpu = False

if 'use_gpu' not in cfg:
    cfg.use_gpu = False

# disable mlu in config by default
if 'use_mlu' not in cfg:
    cfg.use_mlu = False

if cfg.use_gpu:
    place = paddle.set_device('gpu')
elif cfg.use_npu:
    place = paddle.set_device('npu')
elif cfg.use_xpu:
    place = paddle.set_device('xpu')
elif cfg.use_mlu:
    place = paddle.set_device('mlu')
else:
    place = paddle.set_device('cpu')

if FLAGS.slim_config:
    cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

check_config(cfg)
check_gpu(cfg.use_gpu)
check_npu(cfg.use_npu)
check_xpu(cfg.use_xpu)
check_mlu(cfg.use_mlu)
check_version()

print(cfg)
with open('cfg_yoloe.pkl', 'wb') as file:
    pickle.dump(cfg, file)
with open('cfg_yoloe.pkl', 'rb') as file:
    cfg = pickle.load(file)
trainer = Trainer(cfg, mode='test')

# load weights
trainer.load_weights(cfg.weights)

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
        try:

            images = get_test_images(None, upload_path)
            results,results_text = trainer.predict(
                    images,
                    draw_threshold=FLAGS.draw_threshold,
                    output_dir=FLAGS.output_dir,
                    save_results=FLAGS.save_results,
                    visualize=False)
            text = ','.join(results_text)
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
        stus['info']=text
        rz=json.dumps(stus,ensure_ascii=False, indent=4)
    
    return rz


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5011, threaded=False)
