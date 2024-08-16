from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
from typing import List, Dict
import time
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sibling_directory = os.path.join(current_path, '..')
sys.path.append(sibling_directory)
from utils.convert_data import sentence2a_d, read_json, write2json


def predict(output_path, inputs: List[Dict], model_path, max_new_tokens=512) -> List[Dict]:
    torch.manual_seed(42)
    path = model_path
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)
    try:
        outputs=read_json(output_path)
    except:
        outputs = list()
    l = len(outputs)
    for index, one in enumerate(inputs[l:]):
        query = one['messages'][0]['content']
        gt = one['messages'][1]['content']
        start = time.time()
        response = model.chat(tokenizer, query, history=None, eos_token_id=2, pad_token_id=2, temperature=0.3, top_p=0.8, max_length=None, max_new_tokens=max_new_tokens)[0]
        end = time.time()
        if 'cls' in output_path:
            response = sentence2a_d(response)
        print(f"[{l+index + 1}/{len(inputs)}] | {os.path.basename(output_path)} | {(end - start):.2f}s")
        outputs.append({
            'input': query,
            'output': response,
            'ground truth': gt,
        })
        if index % 30 == 0:
            write2json(output_path,outputs)
    write2json(output_path,outputs)
    return outputs


if __name__ == '__main__':
    test_data_path = 'data/test_data.json'
    max_new_tokens = 512
    dir_name, file_name = os.path.split(test_data_path)
    file_base_name = os.path.splitext(file_name)[0]
    if not os.path.exists('./result'):
        os.makedirs('./result')
    path_after_train = os.path.join('./result', file_base_name + '-result-after-train.json')
    trained_model_path = 'MEDSQ'
    
    predict(path_after_train, read_json(test_data_path), trained_model_path, max_new_tokens)
        
