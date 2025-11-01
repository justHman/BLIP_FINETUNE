import re

import json
import os
from collections import defaultdict, Counter

import torch.distributed as dist

import utils
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url
import unicodedata

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def vn_pre_caption(caption, max_words=50):
    # Loại bỏ các ký tự không cần thiết, giữ lại dấu tiếng Việt
    caption = re.sub(
        r"([.!\"()*#:;~])",  # Loại bỏ các ký tự không cần thiết
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",  # Loại bỏ khoảng trắng thừa
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')  # Loại bỏ ký tự xuống dòng
    caption = caption.strip(' ')  # Loại bỏ khoảng trắng ở đầu và cuối

    # Cắt ngắn caption nếu vượt quá số từ tối đa
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption

def unaccented_vn_pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",  # Loại bỏ các ký tự không cần thiết
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",  # Loại bỏ khoảng trắng thừa
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')  # Loại bỏ ký tự xuống dòng
    caption = caption.strip(' ')  # Loại bỏ khoảng trắng ở đầu và cuối

    # Loại bỏ dấu tiếng Việt và chuyển đổi 'đ' thành 'd'
    def remove_accent_and_convert_d(text):
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')  # Loại bỏ dấu
        text = text.replace('đ', 'd')  # Chuyển đổi 'đ' thành 'd'
        return text

    caption = remove_accent_and_convert_d(caption)

    # Cắt ngắn caption nếu vượt quá số từ tối đa
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question

def save_result(result, result_dir, distributed, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

    if distributed:
        dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file

def coco_caption_eval(gt_root_file, results_file, split):
    annotation_file = os.path.join(gt_root_file, f'uitvic_{split}_gt.json')
    
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()

    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval

def caption_eval(gt_root_file, results_file, dataset, split):
    if split not in ['valid', 'test']:
        raise ValueError("split must be either 'valid' or 'test'")
    
    gt_root_file = os.path.join(gt_root_file, f'{dataset}_{split}_gt.json')

    # Load ground truth and results
    with open(gt_root_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Create ground truth mapping
    gt_captions = defaultdict(list)
    for ann in gt_data['annotations']:
        gt_captions[ann['image_id']].append(ann['caption'])

    # Tokenizer
    def tokenize(text):
        return text.lower().split()

    # BLEU-n calculation
    def bleu_n(pred, refs, n):
        pred_tokens = tokenize(pred)
        pred_ngrams = Counter(tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1))

        ref_ngrams = Counter()
        for ref in refs:
            ref_tokens = tokenize(ref)
            ref_ngrams.update(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))

        overlap = sum((pred_ngrams & ref_ngrams).values())
        total_pred = sum(pred_ngrams.values())

        return overlap / total_pred if total_pred > 0 else 0.0

    # ROUGE-L calculation
    def rouge_l(pred, refs):
        def lcs(X, Y):
            m, n = len(X), len(Y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i - 1] == Y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]

        pred_tokens = tokenize(pred)
        scores = []
        for ref in refs:
            ref_tokens = tokenize(ref)
            lcs_len = lcs(pred_tokens, ref_tokens)
            precision = lcs_len / len(pred_tokens) if pred_tokens else 0
            recall = lcs_len / len(ref_tokens) if ref_tokens else 0
            if precision + recall > 0:
                scores.append(2 * precision * recall / (precision + recall))
        return max(scores) if scores else 0.0

    # CIDEr calculation
    def cider(pred, refs):
        pred_tokens = tokenize(pred)
        pred_ngrams = Counter(tuple(pred_tokens[i:i + 4]) for i in range(len(pred_tokens) - 4 + 1))

        ref_ngrams = Counter()
        for ref in refs:
            ref_tokens = tokenize(ref)
            ref_ngrams.update(tuple(ref_tokens[i:i + 4]) for i in range(len(ref_tokens) - 4 + 1))

        overlap = sum((pred_ngrams & ref_ngrams).values())
        total_pred = sum(pred_ngrams.values())

        return overlap / total_pred if total_pred > 0 else 0.0

    # Calculate metrics
    bleu_scores = [0.0, 0.0, 0.0, 0.0]
    rouge_scores, cider_scores = 0.0, 0.0
    valid_predictions = 0

    for result in results:
        image_id = result['image_id']
        pred_caption = result['caption']
        refs = gt_captions.get(image_id, [])

        if refs:
            valid_predictions += 1
            for n in range(4):
                bleu_scores[n] += bleu_n(pred_caption, refs, n + 1)
            rouge_scores += rouge_l(pred_caption, refs)
            cider_scores += cider(pred_caption, refs)

    # Average scores
    if valid_predictions > 0:
        bleu_scores = [score / valid_predictions for score in bleu_scores]
        rouge_scores /= valid_predictions
        cider_scores /= valid_predictions

    # Return results
    return {
        'Bleu_1': bleu_scores[0],
        'Bleu_2': bleu_scores[1],
        'Bleu_3': bleu_scores[2],
        'Bleu_4': bleu_scores[3],
        'ROUGE_L': rouge_scores,
        'CIDEr': cider_scores
    }
    
if __name__ == "__main__":
    caption = "Một bức ảnh về con mèo đáng yêu!"
    processed_pre_caption = pre_caption(caption, max_words=20)
    processed_vn_pre_caption = vn_pre_caption(caption, max_words=20)
    processed_unaccented_vn_pre_caption = unaccented_vn_pre_caption(caption, max_words=20)
    print("Pre Caption:", processed_pre_caption)
    print("VN Pre Caption:", processed_vn_pre_caption)
    print("Unaccented VN Pre Caption:", processed_unaccented_vn_pre_caption)
    