from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import single_meteor_score
from pycocoevalcap.cider.cider import Cider
import bert_score
import torch
import os
import jieba
import json

# 设置缓存文件路径到用户主目录
cache_dir = os.path.expanduser("~/jieba_cache")

# 创建目录（如果不存在）
os.makedirs(cache_dir, exist_ok=True)

# 设置临时目录
jieba.dt.tmp_dir = cache_dir

# 初始化 jieba
jieba.initialize()

def bleu(reference: list[str], candidate: list[str], n: int=4) -> float:
    weights = tuple()
    if n == 1:
        weights = (1, 0, 0, 0)
    elif n == 2:
        weights = (0.5, 0.5, 0, 0)
    elif n == 3:
        weights = (0.33, 0.33, 0.33, 0)
    elif n == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    else:
        return None
    return sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1, weights=weights)

def rouge(reference: list[str], hypothesis: list[str]):
    reference = ' '.join(reference)
    hypothesis = ' '.join(hypothesis)
    return Rouge().get_scores(hypothesis, reference)[0]
    
    
def meteor(reference: list[str], candidate: list[str]):
    return single_meteor_score(reference, candidate)


def avg_cider(references, candidates):
    assert len(references) == len(candidates), 'error.'
    refs, tgts = {}, {}
    for idx, (ref, tgt) in enumerate(zip(references, candidates)):
        refs[f'{idx}'] = [" ".join(ref)]
        tgts[f'{idx}'] = [" ".join(tgt)]
    return Cider().compute_score(refs, tgts)[0]
    
    

def avg_bleu(references: list[list[str]], candidates: list[list[str]]) -> tuple[float, float, float, float]:
    """bleu1~4"""
    assert len(references) == len(candidates), 'error.'
    bleus1 = [bleu(a, b, 1) for a, b in zip(references, candidates)]
    bleus2 = [bleu(a, b, 2) for a, b in zip(references, candidates)]
    bleus3 = [bleu(a, b, 3) for a, b in zip(references, candidates)]
    bleus4 = [bleu(a, b, 4) for a, b in zip(references, candidates)]
    return sum(bleus1) / len(bleus1), sum(bleus2) / len(bleus2), sum(bleus3) / len(bleus3), sum(bleus4) / len(bleus4)

def avg_rouge(references: list[str], candidates: list[str]) -> tuple[float, float, float]:
    rouges = [rouge(a, b) for a, b in zip(references, candidates)]
    rouges_1 = [one['rouge-1']['f'] for one in rouges]
    rouges_2 = [one['rouge-2']['f'] for one in rouges]
    rouges_l = [one['rouge-l']['f'] for one in rouges]
    avg_rouge1 = sum(rouges_1) / len(rouges_1)
    avg_rouge2 = sum(rouges_2) / len(rouges_2)
    avg_rougel = sum(rouges_l) / len(rouges_l)
    return avg_rouge1, avg_rouge2, avg_rougel

def avg_meteor(references: list[str], candidates: list[str]) -> float:
    meteors = [meteor(a, b) for a, b in zip(references, candidates)]
    return sum(meteors) / len(meteors)

def avg_bert_score(references: list[str], candidates: list[str], language: str) -> tuple[float, float, float]:
    """精确度，召回率，F1分数"""
    if language == 'en':
        P, R, F1 = bert_score.score(candidates, references, lang=language, verbose=False, model_type="bert-base-uncased")
    else:
        P, R, F1 = bert_score.score(candidates, references, lang=language, verbose=False, model_type="bert-base-chinese")
    return P.mean().item(), R.mean().item(), F1.mean().item()

def show_score(references: list[str], candidates: list[str], language: str):
    def drop_space(x: list[str]) -> list[str]:
        return [one for one in x if one != ' ']
    cut_references = [drop_space(jieba.lcut(reference)) for reference in references]
    cut_candidates = [drop_space(jieba.lcut(candidate)) for candidate in candidates]
    # print(cut_candidates[0])
    print('bleu:', avg_bleu(cut_references, cut_candidates))
    print('meteor:', avg_meteor(cut_references, cut_candidates))
    print('rouge:', avg_rouge(cut_references, cut_candidates))
    print('bert score:', avg_bert_score(references, candidates, language))
    print('cider:', avg_cider(cut_references, cut_candidates))

def setrecursionlimit(limit: int=5000):
    import sys
    sys.setrecursionlimit(limit)


def json2metrics(json_output, lang):
    with open(json_output,'r',encoding='utf-8') as f:
        list_data = json.load(f)
    true_labels,predictions=[],[]
    for sample in list_data:
        true_labels.append(sample.get('ground truth'))
        predictions.append(sample.get('output'))
    show_score(true_labels,predictions,lang)
    return true_labels,predictions


def get_acc(json_output):
    with open(json_output,'r',encoding='utf-8') as f:
        list_data = json.load(f)
    right=0
    for sample in list_data:
        label=sample.get('ground truth')
        prediction=sample.get('output')
        if label == prediction:
            right+=1
        del label,prediction
    return right/len(list_data)
