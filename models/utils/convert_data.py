import json
import torch
from torch.utils.data import random_split
import numpy as np
import re


def seed(n:int):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.cuda.manual_seed_all(n)


def idx2a_d(idx):
    if idx < 0 or idx > 3 or isinstance(idx, int) is False:
        return None
    switcher = {0: "A", 1: "B", 2: "C", 3: "D"}
    return switcher.get(idx)


def a_d2id(a_d):
    if a_d not in ["A", "B", "C", "D"]:
        return None
    switcher = {"A":0, "B":1, "C":2, "D":3}
    return switcher.get(a_d)


def sentence2a_d(sentence):
    if len(sentence)==1 and sentence in ['A','B','C','D']:
        return sentence
    num_a,num_b,num_c,num_d=0,0,0,0
    sentence=' '+sentence+'.'
    matches = re.findall(r'[^A-Za-z][A-D][^A-Za-z]', sentence)
    # print(type(matches),matches)
    for match in matches:
        if 'A' in match:
            num_a+=1
        elif 'B' in match:
            num_b+=1
        elif 'C' in match:
            num_c+=1
        elif 'D' in match:
            num_d+=1
    # print(num_a, num_b, num_c, num_d)
    if num_a==0 and num_b==0 and num_c==0 and num_d==0:
        return 'N'
    else:
        if max(num_a,num_b,num_c,num_d)==num_a and max(num_a,num_b,num_c,num_d)!=num_b and max(num_a,num_b,num_c,num_d)!=num_c and max(num_a,num_b,num_c,num_d)!=num_d:
            return 'A'
        elif max(num_a,num_b,num_c,num_d)==num_b and max(num_a,num_b,num_c,num_d)!=num_a and max(num_a,num_b,num_c,num_d)!=num_c and max(num_a,num_b,num_c,num_d)!=num_d:
            return 'B'
        elif max(num_a,num_b,num_c,num_d)==num_c and max(num_a,num_b,num_c,num_d)!=num_b and max(num_a,num_b,num_c,num_d)!=num_a and max(num_a,num_b,num_c,num_d)!=num_d:
            return 'C'
        elif max(num_a,num_b,num_c,num_d)==num_d and max(num_a,num_b,num_c,num_d)!=num_b and max(num_a,num_b,num_c,num_d)!=num_c and max(num_a,num_b,num_c,num_d)!=num_a:
            return 'D'
        else:
            return 'N'


def read_json(path):
    with open(path, "r", encoding='utf-8') as file:
        data_json = json.load(file)
    return data_json


def write2json(path, data):
    with open(path, "w", encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def divide_data(filepath):
    with open(filepath, "r", encoding='utf-8') as file:
        list_data = json.load(file)
    total_size = len(list_data)
    train_size = int(total_size * 0.8)
    eval_size = int((total_size - train_size) / 2)
    test_size = total_size - train_size - eval_size
    train_dataset, eval_dataset, test_dataset = random_split(list_data, [train_size, eval_size, test_size])
    return list(train_dataset), list(eval_dataset), list(test_dataset)


def convert_for_cls(list_data, lang="zh"):
    # with open('img2caption.json', "r", encoding='utf-8') as file:
    #     img2cap = json.load(file)
    if lang == "en" or lang == "zh":
        dataset = []
        for one_img in list_data:
            img_path = one_img["image"].get("image_path")
            list_qa = one_img[f"test_{lang}"]
            for num, qa in enumerate(list_qa):
                for idx in range(7):
                    dict_no_his = {}
                    dict_no_his["image"] = img_path
                    dict_no_his["qa"] = []
                    dict_qa = {}
                    if lang == "zh":
                        # dict_qa['question'] = one_img.get('report_zh') + img2cap.get(img_path).get('zh') + '\n问题：' + qa[idx].get('Question') + "\nA: " + qa[idx].get(
                        #     'A') + "\nB: " + qa[idx].get('B') + "\nC: " + qa[idx].get('C') + "\nD: " + qa[idx].get(
                        #     'D') + f"\n请根据以上信息，选择能回答问题的正确选项，仅输出字母(A, B, C, D)其中之一。"
                        dict_qa['question'] = one_img.get('report_zh') + '\n问题：' + qa[idx].get('Question') + "\nA: " + qa[idx].get(
                            'A') + "\nB: " + qa[idx].get('B') + "\nC: " + qa[idx].get('C') + "\nD: " + qa[idx].get(
                            'D') + f"\n请根据以上信息，选择能回答问题的正确选项，仅输出字母(A, B, C, D)其中之一。"
                    elif lang == "en":
                        # dict_qa['question'] = one_img.get('report') + img2cap.get(img_path).get('en') + '\nQuestion: ' + qa[idx].get('Question') + "\nA: " + qa[idx].get(
                        #     'A') + "\nB: " + qa[idx].get('B') + "\nC: " + qa[idx].get('C') + "\nD: " + qa[idx].get(
                        #     'D') + f"\nBased on the above information, please select the correct option that can answer the question and output only one of the letters (A, B, C, D)."
                        dict_qa['question'] = one_img.get('report_en') + '\nQuestion: ' + qa[idx].get('Question') + "\nA: " + qa[idx].get(
                            'A') + "\nB: " + qa[idx].get('B') + "\nC: " + qa[idx].get('C') + "\nD: " + qa[idx].get(
                            'D') + f"\nBased on the above information, please select the correct option that can answer the question and output only one of the letters (A, B, C, D)."

                    dict_qa['answer'] = qa[idx].get('GT')
                    dict_no_his["qa"].append(dict_qa)
                    dataset.append(dict_no_his)
        return dataset
    else:
        print("Please select 'en' or 'zh'!")
        return None


def convert_for_gen(list_data, lang="zh"):
    # with open('img2caption.json', "r", encoding='utf-8') as file:
    #     img2cap = json.load(file)
    if lang == "en" or lang == "zh":
        dataset = []
        for one_img in list_data:
            img_path = one_img["image"].get("image_path")
            list_qa = one_img[f"qa_{lang}"]
            for num, qa in enumerate(list_qa):
                for idx in range(7):
                    dict_no_his = {}
                    dict_no_his["image"] = img_path
                    dict_no_his["qa"] = []
                    dict_qa = {}
                    if lang == "zh":
                        # dict_qa['question'] = one_img.get('report_zh') + img2cap.get(img_path).get('zh') + f"\n请根据以上信息，回答问题。" + '\n问题：' + qa[idx].get('Question')
                        dict_qa['question'] = one_img.get('report_zh') + f"\n请根据以上信息，回答问题。" + '\n问题：' + qa[idx].get('Question')

                    elif lang == "en":
                        # dict_qa['question'] = one_img.get('report') + img2cap.get(img_path).get('en') + f"\nBased on the above information, answer the question." + '\nQuestion: ' + qa[idx].get('Question')
                        dict_qa['question'] = one_img.get('report_en') + f"\nBased on the above information, answer the question." + '\nQuestion: ' + qa[idx].get('Question')


                    dict_qa['answer'] = qa[idx].get('Answer')
                    dict_no_his["qa"].append(dict_qa)
                    dataset.append(dict_no_his)
        return dataset
    else:
        print("Please select 'en' or 'zh'!")
        return None


def to_trained_data(list_data,filepath):
    data = []
    for sample in list_data:
        one_sample = {}
        one_sample["messages"] = []
        one_sample["messages"].append(
                    {
                        "role": "user",
                        "content": sample.get('qa')[0].get('question')
                    })
        one_sample["messages"].append(
            {
                "role": "assistant",
                "content": sample.get('qa')[0].get('answer')
            })
        data.append(one_sample)
    
    write2json(filepath, data)
    return data
