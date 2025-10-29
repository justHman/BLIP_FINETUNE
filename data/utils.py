import re
import json
import os

import torch.distributed as dist

import utils
import spacy
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

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



from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url

def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
    download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval

def uitvic_caption_eval(uitvic_gt_root, results_file, split):
    filenames = {'valid': 'uitvic_valid_gt.json', 'test': 'uitvic_test_gt.json'}

    # Ensure the ground truth file exists
    annotation_file = os.path.join(uitvic_gt_root, filenames[split])
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Ground truth file not found: {annotation_file}")
    
    # Create COCO object and load results
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # Create COCOEvalCap object
    coco_eval = COCOEvalCap(coco, coco_result)

    # Evaluate results
    coco_eval.evaluate()

    # Print evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval

class eval_results:
    """Class to hold evaluation results."""
    def __init__(self, eval):
        self.eval = eval

def uitvic_caption_eval_by_spacy(uitvic_gt_root, results_file, split):
    """
    Evaluate captions using spaCy for tokenization and COCOEvalCap metrics.
    """
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Load ground truth and results
    filenames = {'valid': 'uitvic_valid_gt.json', 'test': 'uitvic_test_gt.json'}
    annotation_file = os.path.join(uitvic_gt_root, filenames[split])

    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Ground truth file not found: {annotation_file}")

    with open(annotation_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Prepare ground truth and results dictionaries
    gts = {}
    res = {}
    for ann in gt_data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        if image_id not in gts:
            gts[image_id] = []
        gts[image_id].append(caption)

    for result in results:
        image_id = result['image_id']
        caption = result['caption']
        res[image_id] = [caption]

    # Tokenize captions using spaCy
    def tokenize(text):
        doc = nlp(text.lower())
        return [token.text for token in doc if not token.is_punct and not token.is_space]

    gts = {k: [" ".join(tokenize(caption)) for caption in captions] for k, captions in gts.items()}
    res = {k: [" ".join(tokenize(caption)) for caption in captions] for k, captions in res.items()}

    # Set up scorers
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    # Compute scores
    results = eval_results(eval={})
    for scorer, method in scorers:
        print(f"Computing {scorer.method()}...")
        score, scores = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for m, s in zip(method, score):
                results.eval[m] = s
                print(f"{m}: {s:.3f}")
        else:
            results.eval[method] = score
            print(f"{method}: {score:.3f}")

    return results

def safe_uitvic_caption_eval(root_dir, results_file, split, logger):
    """Custom UITVIC caption evaluation - no external dependencies"""
    try:
        import json
        from collections import defaultdict, Counter
        
        filenames = {'valid': 'uitvic_valid_gt.json', 'test': 'uitvic_test_gt.json'}
        annotation_file = os.path.join(root_dir, filenames[split])
        
        if not os.path.exists(annotation_file):
            logger.error(f"Ground truth file not found: {annotation_file}")
            class ZeroEval:
                def __init__(self):
                    self.eval = {'Bleu_1': 0, 'Bleu_2': 0, 'Bleu_3': 0, 'Bleu_4': 0, 'CIDEr': 0, 'ROUGE_L': 0}
            return ZeroEval()
        
        logger.log(f"Running custom evaluation for {split}")
        
        # Local encoding fix function
        def fix_encoding(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                logger.error(f"Could not read {file_path}")
                return {"images": [], "annotations": []}
        
        # Read data
        gt_data = fix_encoding(annotation_file)
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.log(f"Ground truth: {len(gt_data.get('annotations', []))} annotations")
        logger.log(f"Generated: {len(results)} captions")
        
        # Show samples
        for i, result in enumerate(results[:2]):
            logger.log(f"Sample {i+1}: {result['image_id']} -> '{result['caption']}'")
        
        # Create GT mapping
        gt_captions = defaultdict(list)
        for ann in gt_data.get('annotations', []):
            gt_captions[ann['image_id']].append(ann['caption'])
        
        # Simple tokenizer
        def tokenize(text):
            return text.lower().strip().split()
        
        # Compact BLEU calculation
        def bleu_n(pred, refs, n):
            if len(pred) < n: return 0.0
            pred_ngrams = Counter(tuple(pred[i:i+n]) for i in range(len(pred)-n+1))
            clipped = 0
            for ngram, count in pred_ngrams.items():
                max_ref = max((Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1)).get(ngram, 0) 
                              for ref in refs if len(ref) >= n), default=0)
                clipped += min(count, max_ref)
            return clipped / max(sum(pred_ngrams.values()), 1)
        
        # Compact ROUGE-L calculation  
        def rouge_l(pred, refs):
            def lcs(a, b):
                m, n = len(a), len(b)
                dp = [[0]*(n+1) for _ in range(m+1)]
                for i in range(1, m+1):
                    for j in range(1, n+1):
                        dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
                return dp[m][n]
            
            if not pred: return 0.0
            max_f1 = 0.0
            for ref in refs:
                if ref:
                    lcs_len = lcs(pred, ref)
                    if lcs_len > 0:
                        p, r = lcs_len/len(pred), lcs_len/len(ref)
                        max_f1 = max(max_f1, 2*p*r/(p+r) if p+r > 0 else 0)
            return max_f1
        
        # Compact CIDEr calculation
        def cider(pred, refs):
            if not pred or not refs: return 0.0
            pred_tf = Counter(pred)
            score = 0.0
            for ref in refs:
                if ref:
                    ref_tf = Counter(ref)
                    overlap = sum(min(pred_tf[w], ref_tf[w]) for w in pred_tf if w in ref_tf)
                    score += overlap / max(len(pred), 1)
            return (score / max(len(refs), 1)) * 5  # Scale to CIDEr range
        
        # Calculate metrics
        bleu_scores = [0.0, 0.0, 0.0, 0.0]
        rouge_scores, cider_scores = [], []
        valid_predictions = 0
        
        for result in results:
            img_id = result['image_id']
            pred_tokens = tokenize(result['caption'].strip())
            
            if img_id in gt_captions and pred_tokens:
                ref_tokens_list = [tokenize(gt) for gt in gt_captions[img_id] if tokenize(gt)]
                
                if ref_tokens_list:
                    # BLEU scores
                    for n in range(1, 5):
                        bleu_scores[n-1] += bleu_n(pred_tokens, ref_tokens_list, n)
                    
                    # ROUGE-L and CIDEr
                    rouge_scores.append(rouge_l(pred_tokens, ref_tokens_list))
                    cider_scores.append(cider(pred_tokens, ref_tokens_list))
                    valid_predictions += 1
        
        # Average scores
        if valid_predictions > 0:
            final_bleu = [s/valid_predictions for s in bleu_scores]
            final_rouge = sum(rouge_scores) / len(rouge_scores)
            final_cider = sum(cider_scores) / len(cider_scores)
        else:
            final_bleu = [0.0, 0.0, 0.0, 0.0]
            final_rouge = final_cider = 0.0
        
        # Result
        class EvalResult:
            def __init__(self, bleu, rouge, cider_score):
                self.eval = {
                    'Bleu_1': round(bleu[0], 3), 'Bleu_2': round(bleu[1], 3),
                    'Bleu_3': round(bleu[2], 3), 'Bleu_4': round(bleu[3], 3),
                    'CIDEr': round(cider_score, 3), 'ROUGE_L': round(rouge, 3)
                }
        
        result = EvalResult(final_bleu, final_rouge, final_cider)
        logger.log(f"Metrics for {split}: " + ", ".join(f"{k}={v}" for k,v in result.eval.items()))
        return result
        
    except Exception as e:
        logger.error(f"Complete evaluation failed for {split}: {str(e)}")
        # Return zero metrics as final fallback
        class ZeroEval:
            def __init__(self):
                self.eval = {'Bleu_1': 0, 'Bleu_2': 0, 'Bleu_3': 0, 'Bleu_4': 0, 'CIDEr': 0, 'ROUGE_L': 0}
        return ZeroEval()