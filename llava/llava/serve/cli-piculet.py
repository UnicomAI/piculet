import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

import requests
import json
import os
def process_hallucination(img):
    url = "http://127.0.0.1:5015/face_recognition"
    f = open(img, 'rb')
    files = {'imagefile': ('hg.jpeg', f, 'image/jpeg')}
    ret = requests.post(url=url, files=files)
    f.close()
    faces = []
    unknow = []
    if ret.status_code == 200:
        for item in json.loads(ret.text)['info']['recognition_results']:
            if item['id'] != 'unknown':
                faces.append(item['id'])
            else:
                unknow.append('unknown')
    prompt_face = ''
    if len(faces) > 0:
        if len(faces) == 1:
            prompt_face = 'The celebrity in the image is ' + '、'.join(faces) + '。\n'
        else:
            prompt_face = 'The celebrities in the image are ' + '、'.join(faces) + '。\n'

    url = "http://127.0.0.1:5001/ocr"
    f = open(img, 'rb')
    files = {'imagefile': ('testocr_2.jpg', f, 'image/jpeg')}
    ret = requests.post(url=url, files=files, verify=False)
    f.close()
    content = ''
    if ret.status_code == 200:
        for text in json.loads(ret.text)['info']['words_results']:
            content += text['words'] + '\n'
    prompt_ocr = ''
    if len(content) > 0:
        prompt_ocr = 'The text content contained in the image.：\n' + content + '\n'

    url = "http://127.0.0.1:5012/detect"
    f = open(img, 'rb')
    files = {'imagefile': ('test.jpg', f, 'image/jpeg')}
    ret = requests.post(url=url, files=files)
    f.close()
    content = ''
    if ret.status_code == 200:
        items = json.loads(ret.text)['info']
        num_info = {}
        if len(items) > 0:
            for obj in items.split(','):
                if obj in num_info.keys():
                    num_info[obj] += 1
                else:
                    num_info[obj] = 1
            content = 'the image contains these objects：'
            for k, v in num_info.items():
                content += 'there is ' + str(v) + ' ' + k + '; ' if v == 1 else 'there are ' + str(v) + ' ' + k + ';'
            content = content[:-1] + '\n'
    return prompt_face, prompt_ocr, content


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    while True:
        try:
            image_name = input(f"image name: ")
            inp = input(f"{roles[0]}: ")
        except EOFError:
            image_name = ""
            inp = ""
        if not inp or not image_name:
            print("exit...")
            break
        print(f"image_name:{image_name}", end="\t")
        whole_image_path = os.path.join('path/to/sampled_coco/', 'COCO_val2014_'+image_name)
        print(f"whole_image_path:{whole_image_path}", end="\t")
        image = load_image(whole_image_path)
        image_size = image.size
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        print(f"{roles[1]}: ", end="")

        prompt_face, prompt_ocr, content = process_hallucination(whole_image_path)
        prompt_template = 'please fully understand the image and answer the question based on the above content:\n'
        print("input before warbler:{}".format(inp))
        # inp = prompt_ocr + prompt_face + content + prompt_template + inp
        # inp = prompt_template + inp
        # print("input after warbler:{}".format(inp))

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        # conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print("oriignal prompt:{}".format(prompt))
        print("oriignal prompt end.......")
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        print("input_ids:{}, IMAGE_TOKEN_INDEX:{}".format(input_ids, IMAGE_TOKEN_INDEX))
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs
        conv = conv_templates[args.conv_mode].copy()
        
        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

# python -m llava.serve.cli-piculet --model-path weights/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3/ --image-file MME_Benchmark_release_version/count/000000067213.jpg
# python -m llava.serve.cli-piculet --image-file sampled_coco/COCO_val2014_000000441147.jpg
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="path/to/models--liuhaotian--llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
