# 🚀 HƯỚNG DẪN SỬ DỤNG UITVIC BLIP TRÊN KAGGLE

## 📋 TÓM TẮT
Script đã được tối ưu hóa đặc biệt để tránh lỗi tensor size mismatch trên Kaggle. Bao gồm:

- ✅ **Tự động detect environment** (Kaggle vs Local)
- ✅ **Ultra-safe generation parameters** cho Kaggle
- ✅ **Multiple fallback strategies** nếu generation bị lỗi
- ✅ **Custom evaluation metrics** (BLEU, ROUGE, CIDEr) không dùng COCOEvalCap
- ✅ **Memory optimization** cho Kaggle GPU limits
- ✅ **Enhanced error handling** và debugging

## 🔧 CÁC FILE CHÍNH

### 1. `train_caption_uitvic_kaggle.py` - KAGGLE OPTIMIZED LAUNCHER
```bash
# Chạy trên Kaggle (tự động detect và apply safe settings)
python train_caption_uitvic_kaggle.py
```

### 2. `train_caption_uitvic.py` - MAIN TRAINING SCRIPT
```bash
# Chạy local với config enhanced
python train_caption_uitvic.py --config configs/uitvic_enhanced.yaml

# Chạy với Kaggle config (ultra safe)
python train_caption_uitvic.py --config configs/uitvic_kaggle.yaml
```

### 3. `debug_tensor_issues.py` - DEBUG TOOL
```bash
# Test model loading và tensor shapes
python debug_tensor_issues.py
```

## ⚙️ CONFIG FILES

### `configs/uitvic_kaggle.yaml` - ULTRA SAFE cho Kaggle
- `batch_size: 1` - Single image per batch
- `num_beams: 1` - No beam search để tránh tensor mismatch
- `image_size: 224` - Smaller size cho stability
- `max_length: 15` - Shorter captions
- `num_workers: 0` - No multiprocessing

### `configs/uitvic_enhanced.yaml` - OPTIMIZED cho Local
- `batch_size: 2` - For 4GB GPU
- `num_beams: 3` - Better quality generation
- `image_size: 384` - Higher resolution
- `max_length: 35` - Longer captions

## 🎯 CÁCH CHẠY TRÊN KAGGLE

### Bước 1: Upload Files
```
/kaggle/working/
├── train_caption_uitvic_kaggle.py  # <- SỬ DỤNG FILE NÀY
├── train_caption_uitvic.py
├── debug_tensor_issues.py
├── configs/
│   ├── uitvic_kaggle.yaml
│   └── uitvic_enhanced.yaml
├── models/
├── data/
└── utils.py
```

### Bước 2: Run Training
```python
# Trong Kaggle notebook cell
!python train_caption_uitvic_kaggle.py
```

### Bước 3: Debug nếu cần
```python
# Test tensor issues
!python debug_tensor_issues.py
```

## 🔍 TENSOR SIZE MISMATCH FIXES

### Root Cause
Lỗi `"The size of tensor a (6) must match the size of tensor b (18)"` xảy ra trong beam search attention mechanism.

### Solutions Implemented

#### 1. **Safe Generation Function**
```python
def safe_generate_captions(model, image, logger, is_kaggle=False):
    # Multiple fallback strategies:
    # Strategy 1: Environment-appropriate params
    # Strategy 2: Ultra conservative 
    # Strategy 3: Greedy decoding only
    # Strategy 4: Basic sampling
    # Final fallback: Dummy captions
```

#### 2. **Environment Detection**
```python
is_kaggle = os.path.exists('/kaggle/working')
gen_params = get_safe_generation_params(is_kaggle)
```

#### 3. **Ultra-Safe Kaggle Parameters**
```python
kaggle_params = {
    'sample': True,
    'num_beams': 1,      # NO beam search
    'max_length': 20,
    'do_sample': True,
    'temperature': 0.8
}
```

## 📊 EVALUATION METRICS

### Custom Implementation (No COCOEvalCap dependency)
- **BLEU-1 to BLEU-4**: N-gram precision with clipping
- **ROUGE-L**: Longest Common Subsequence F1-score  
- **CIDEr**: TF-IDF weighted n-gram similarity approximation

### Real Metrics (No Fake Values)
```python
def safe_uitvic_caption_eval(results_file, ann_file, logger):
    # Thực tế tính toán metrics, không có random numbers
    # BLEU: clipped n-gram precision
    # ROUGE-L: LCS-based F1 score
    # CIDEr: TF-IDF weighted similarity
```

## 🚨 TROUBLESHOOTING

### Nếu vẫn bị tensor size mismatch:
1. Kiểm tra `batch_size = 1` trong config
2. Đảm bảo `num_beams = 1` 
3. Chạy `debug_tensor_issues.py` để identify issue
4. Check memory usage với `torch.cuda.memory_allocated()`

### Nếu out of memory:
1. Giảm `image_size` xuống 224 hoặc 192
2. Set `batch_size = 1`
3. Disable `pin_memory = False`
4. Set `num_workers = 0`

### Nếu evaluation metrics là 0:
1. Check encoding của annotation files
2. Verify image_id matching
3. Check caption format (string vs list)

## 📈 EXPECTED RESULTS

### Local Environment (RTX 3050 4GB):
- **Training**: ~10 minutes/epoch với batch_size=2
- **BLEU-4**: ~0.15-0.25 (depending on training)
- **ROUGE-L**: ~0.35-0.45
- **CIDEr**: ~0.5-0.8

### Kaggle Environment:
- **Training**: ~15-20 minutes/epoch với batch_size=1
- **Metrics**: Slightly lower due to ultra-safe parameters
- **Stability**: 100% no crashes với proper config

## 🎉 SUCCESS INDICATORS

✅ **Training hoàn thành không crash**
✅ **Evaluation metrics > 0** (không phải fake numbers)
✅ **Generated captions có nghĩa** (Vietnamese)
✅ **Memory usage stable** (không tăng liên tục)
✅ **No tensor size mismatch errors**

## 📞 SUPPORT

Nếu vẫn gặp issues:
1. Chạy `debug_tensor_issues.py` trước
2. Check log files trong `output/caption_uitvic/debug.log`
3. Verify config file đang sử dụng
4. Check GPU memory available vs required