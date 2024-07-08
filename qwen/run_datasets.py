import time
import json
import requests
import time
import os
import re



"""
POPE

read json file and run QWen-VL-Chat to get results.
"""
def read_json_file_and_MLLM_POPE(json_file_path_, output_dir_, url_="http://127.0.0.1:49020qwen_server"):
    with open(json_file_path_, 'r') as file:
        total_lines = file.readlines()
        image_list = [json.loads(q)['image'] for q in total_lines]
        question_list = [json.loads(q)['text'] for q in total_lines]
        print("len(image):{}, len(question):{}".format(len(image_list), len(question_list)))
        output_file_path = os.path.join(output_dir_,"PlainQWen+Detect+OCR+Face-"+os.path.basename(json_file_path_))
        output_file = open(output_file_path, 'w')
        total_dict_list = []
        for item_index, image_name in enumerate(image_list):
            temp_answer_dict = {}
            if(item_index+1)% 100 ==0:
                print("{}/{} processing...".format(item_index+1, len(image_list)))
            image_path = os.path.join('path/to/coco/datasets/val2014', image_name)
            f = open(image_path, 'rb')
            files = {'img': (image_name, f, 'image/jpeg')}
            Data = {'prompt': question_list[item_index]+', simply answer yes or no.'}
            ret2 = requests.post(url=url_, files=files,data=Data)
            answer = json.loads(ret2.text)['result']['text']
            # {"question": "is there a bird in the image?", "answer": "yes"}
            temp_answer_dict['question']  = question_list[item_index]
            temp_answer_dict['answer']  = answer
            total_dict_list.append(temp_answer_dict)
        json.dump(total_dict_list, output_file, indent=1,ensure_ascii=False)
        output_file.close()

"""
MME
"""
txt_and_image_dict={
    'OCR.txt': "OCR",
    'celebrity.txt': "celebrity/images",
    'color.txt': "color",
    'count.txt': "count",
    'existence.txt': "existence",
    'position.txt':'position'
}
print_interval=20
def read_txt_file_and_MLLM_MME(txt_file_path_, output_dir_, image_src_dir_='path/to/MME_Benchmark_release_version/',url_="http://127.0.0.1:49021/qwen_server"):
    output_file_path = os.path.join(output_dir_,"QWen-Detect-OCR-Face_"+os.path.basename(txt_file_path_))
    output_file = open(output_file_path, 'w') 
    total_dict_list = []
    # 读取txt文件并解析每一行
    with open(txt_file_path_, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            # 分割每一行
            items = line.strip().split('\t')
            if len(items) == 4:
                # 获取图片路径、问题、正确答案和错误答案
                image_path, question, correct_answer, wrong_answer = items
                original_question = question

                line_count = line_count +1
                whole_image_path = os.path.join(image_src_dir_, image_path)
                f = open(whole_image_path, 'rb')
                files = {'img': (image_path, f, 'image/jpeg')}
                if line_count %print_interval ==0:
                    print("line:", line_count, image_path, question, correct_answer, wrong_answer)
                Data = {'prompt': question}
                ret2 = requests.post(url=url_, files=files,data=Data)
                answer = json.loads(ret2.text)['result']['text'].strip()
                output_file.write("{}\t{}\t{}\t{}\n".format(image_path, original_question, correct_answer, answer))
        output_file.close()


"""
LLaVa-QA90
"""
print_interval=20
def read_json_file_and_MLLM_LLaVa_QA90(json_file_path_, output_dir_, url_="http://127.0.0.1:49021/qwen_server"):
    with open(json_file_path_, 'r') as file:
        total_lines = file.readlines()
        image_list = [json.loads(q)['image'] for q in total_lines]
        question_list = [json.loads(q)['text'] for q in total_lines]
        print("len(iamge):{}, len(question):{}".format(len(image_list), len(question_list)))
        output_file_path = os.path.join(output_dir_,"PlainQWen+Detect+OCR+Face"+os.path.basename(json_file_path_))
        output_file = open(output_file_path, 'w')
        total_dict_list = []
        for item_index, image_name in enumerate(image_list):
            temp_answer_dict = {}
            print("{}/{} processing...".format(item_index+1, len(image_list)))
            image_name = 'COCO_val2014_'+ image_name
            image_path = os.path.join('path/to/coco/datasets/val2014', image_name)
            f = open(image_path, 'rb')
            files = {'img': (image_name, f, 'image/jpeg')}
            Data = {'prompt': question_list[item_index]}
            ret2 = requests.post(url=url_, files=files,data=Data)
            answer = json.loads(ret2.text)['result']['text']
            # {"question": "is there a bird in the image?", "answer": "yes"}
            temp_answer_dict['question']  = question_list[item_index]
            temp_answer_dict['answer']  = answer
            total_dict_list.append(temp_answer_dict)
        json.dump(total_dict_list, output_file, indent=1,ensure_ascii=False)
        output_file.close()



if __name__ == "__main__":

    # POPE
    output_dir = 'path/to/tested_results/POPE'
    json_file_path = 'path/to/POPE/coco/coco_pope_random.json'
    read_json_file_and_MLLM_POPE(json_file_path, output_dir)

    #MME
    outpur_dir = 'path/to/output'
    file_path = 'path/to/MME_Benchmark_release_version/eval_tool/LaVIN/'
    for txt_name in txt_and_image_dict:
        image_sub_dir = txt_and_image_dict[txt_name]
        print('txt:{}, image_sub_dir:{}'.format(txt_name, image_sub_dir))
        read_txt_file_and_MLLM_MME(os.path.join(file_path, txt_name), outpur_dir, os.path.join('path/to/MME_Benchmark_release_version/', image_sub_dir))

    #LLaVa-QA90
    output_dir = 'path/to/tested_results/LLaVaQA90/tested'
    json_file_path = 'path/to/LLaVaQA90/qa90_questions-description.jsonl'
    read_json_file_and_MLLM_LLaVa_QA90(json_file_path, output_dir)

