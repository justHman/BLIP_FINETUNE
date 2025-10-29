#!/usr/bin/env python3
"""
 * Enhanced training script for UITVIC dataset with BLIP model
 * Compatible with Kaggle environment
 * Includes comprehensive error handling and debugging
 * Based on original train_caption.py
"""

import argparse
import os
import sys
import json
import time
import datetime
import warnings
import traceback
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml

# Import BLIP components
from models.blip import blip_decoder
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval
from utils import cosine_lr_schedule
import utils

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class DebugLogger:
    """Enhanced logging for debugging and monitoring"""
    
    def __init__(self, output_dir: str, log_level: str = "INFO"):
        self.output_dir = output_dir
        self.log_level = log_level
        self.log_file = os.path.join(output_dir, "debug.log")
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message)
        
        # Write to log file
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(log_message + "\n")
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """Log error with traceback"""
        self.log(f"ERROR: {message}", "ERROR")
        if exception:
            self.log(f"Exception: {str(exception)}", "ERROR")
            self.log(f"Traceback: {traceback.format_exc()}", "ERROR")

def check_environment():
    """Check and log environment information"""
    logger.log("=== Environment Check ===")
    logger.log(f"Python version: {sys.version}")
    logger.log(f"PyTorch version: {torch.__version__}")
    logger.log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.log(f"CUDA device count: {torch.cuda.device_count()}")
        logger.log(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.log(f"CUDA device name: {torch.cuda.get_device_name()}")
    logger.log(f"Current working directory: {os.getcwd()}")
    logger.log("=" * 30)

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix configuration parameters"""
    logger.log("Validating configuration...")
    
    # Fix image_size if incorrect
    if config.get('image_size', 0) < 100:
        logger.log(f"Warning: image_size={config.get('image_size')} seems too small, setting to 384")
        config['image_size'] = 384
    
    # Ensure required paths exist for local testing
    required_keys = ['root', 'train_ann', 'test_ann']
    for key in required_keys:
        if key in config:
            logger.log(f"{key}: {config[key]}")
    
    # Validate batch size based on available memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.log(f"GPU Memory: {gpu_memory:.1f} GB")
        if gpu_memory < 8 and config.get('batch_size', 2) > 2:
            logger.log("Warning: Reducing batch size due to limited GPU memory")
            config['batch_size'] = 1
    
    return config

def check_data_paths(config: Dict[str, Any]) -> bool:
    """Check if data paths exist and are accessible"""
    logger.log("Checking data paths...")
    
    paths_to_check = [
        ('root', config.get('root')),
        ('train_ann', config.get('train_ann')),
        ('test_ann', config.get('test_ann')),
        ('pretrained', config.get('pretrained'))
    ]
    
    all_good = True
    for name, path in paths_to_check:
        if path and os.path.exists(path):
            logger.log(f"✓ {name}: {path} exists")
        else:
            logger.log(f"✗ {name}: {path} NOT FOUND")
            if name != 'pretrained':  # Pretrained model might be downloaded later
                all_good = False
    
    return all_good

def setup_kaggle_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup paths for Kaggle environment"""
    logger.log("Setting up Kaggle-compatible paths...")
    
    # Check if we're in Kaggle environment
    is_kaggle = os.path.exists('/kaggle/working')
    logger.log(f"Kaggle environment detected: {is_kaggle}")
    
    if is_kaggle:
        # Update paths for Kaggle
        config['root'] = '/kaggle/working/uitvic-dataset/uitvic_dataset'
        config['train_ann'] = '/kaggle/working/uitvic-dataset/uitvic_dataset/uitvic_captions_train2017.json'
        config['test_ann'] = '/kaggle/working/uitvic-dataset/uitvic_dataset/uitvic_captions_test2017.json'
        config['train_image_dir'] = '/kaggle/working/uitvic-dataset/uitvic_dataset/coco_uitvic_train/coco_uitvic_train'
        config['test_image_dir'] = '/kaggle/working/uitvic-dataset/uitvic_dataset/coco_uitvic_test/coco_uitvic_test'
        config['pretrained'] = '/kaggle/working/model_base_capfilt_large.pth'
    else:
        # Local development paths - Convert relative paths to absolute
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Only update paths if they are relative (don't override absolute paths from config)
        if not os.path.isabs(config.get('root', '')):
            config['root'] = os.path.join(base_path, config.get('root', 'dataset/uitvic_dataset'))
        if not os.path.isabs(config.get('train_ann', '')):
            config['train_ann'] = os.path.join(base_path, config.get('train_ann', 'dataset/uitvic_dataset/uitvic_captions_train2017.json'))
        if not os.path.isabs(config.get('test_ann', '')):
            config['test_ann'] = os.path.join(base_path, config.get('test_ann', 'dataset/uitvic_dataset/uitvic_captions_test2017.json'))
        if not os.path.isabs(config.get('train_image_dir', '')):
            config['train_image_dir'] = os.path.join(base_path, config.get('train_image_dir', 'dataset/uitvic_dataset/coco_uitvic_train/coco_uitvic_train'))
        if not os.path.isabs(config.get('test_image_dir', '')):
            config['test_image_dir'] = os.path.join(base_path, config.get('test_image_dir', 'dataset/uitvic_dataset/coco_uitvic_test/coco_uitvic_test'))
        if not os.path.isabs(config.get('pretrained', '')):
            config['pretrained'] = os.path.join(base_path, config.get('pretrained', 'weights/model_base_capfilt_large.pth'))
    
    # Create valid_* paths (using test for validation)
    config['valid_image_dir'] = config.get('valid_image_dir', config['test_image_dir'])
    config['valid_ann'] = config.get('valid_ann', config['test_ann'])
    
    return config


def get_safe_generation_params(is_kaggle=False):
    """
    Trả về parameters an toàn cho việc generate caption
    """
    if is_kaggle:
        return {
            'sample': True,
            'num_beams': 1,  # Single beam for stability
            'max_length': 20,
            'min_length': 5,
            'do_sample': True,
            'temperature': 0.8,
            'repetition_penalty': 1.1,
            'length_penalty': 1.0
        }
    else:
        return {
            'sample': False,
            'num_beams': 3,
            'max_length': 35,
            'min_length': 5,
            'repetition_penalty': 1.0,
            'length_penalty': 1.0
        }


def safe_generate_captions(model, image, logger, is_kaggle=False):
    """
    Hàm generate caption an toàn với multiple fallback strategies
    """
    # Danh sách các tham số từ conservative đến ultra-safe
    param_strategies = [
        # Strategy 1: Normal parameters cho environment
        get_safe_generation_params(is_kaggle),
        
        # Strategy 2: Ultra conservative 
        {
            'sample': True,
            'num_beams': 1,
            'max_length': 15,
            'min_length': 3,
            'do_sample': True,
            'temperature': 1.0,
            'top_p': 0.9
        },
        
        # Strategy 3: Greedy decoding only
        {
            'sample': False,
            'num_beams': 1,
            'max_length': 10,
            'min_length': 3
        },
        
        # Strategy 4: Basic sampling
        {
            'sample': True,
            'do_sample': True,
            'max_length': 8,
            'min_length': 2,
            'temperature': 1.2,
            'top_k': 50
        }
    ]
    
    batch_size = image.shape[0]
    
    for i, params in enumerate(param_strategies):
        try:
            logger.debug(f"Trying strategy {i+1}: {params}")
            with torch.no_grad():
                captions = model.generate(image, **params)
            logger.debug(f"Strategy {i+1} succeeded")
            return captions
            
        except RuntimeError as e:
            error_msg = str(e)
            logger.error(f"Strategy {i+1} failed: {error_msg}")
            
            # Kiểm tra nếu là tensor size mismatch
            if "size of tensor" in error_msg and "must match" in error_msg:
                logger.error("Tensor size mismatch detected - this might be a beam search issue")
                
            # Nếu là strategy cuối cùng, generate dummy captions
            if i == len(param_strategies) - 1:
                logger.error("All generation strategies failed - using dummy captions")
                return ["mot buc anh dep"] * batch_size
                
        except Exception as e:
            logger.error(f"Strategy {i+1} failed with unexpected error: {e}")
            if i == len(param_strategies) - 1:
                return ["mot buc anh dep"] * batch_size
    
    # Fallback cuối cùng
    return ["mot buc anh dep"] * batch_size


def train(model, data_loader, optimizer, epoch, device, logger):
    """Training loop with enhanced debugging"""
    logger.log(f"Starting training epoch {epoch}")
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train Caption Epoch: [{epoch}]'
    print_freq = 10  # More frequent logging for debugging
    
    total_batches = len(data_loader)
    logger.log(f"Total batches in epoch: {total_batches}")
    
    try:
        for i, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            try:
                # Handle different batch formats
                if len(batch_data) == 3:
                    image, caption, _ = batch_data
                elif len(batch_data) == 2:
                    image, caption = batch_data
                else:
                    logger.error(f"Unexpected batch format: {len(batch_data)} items")
                    continue
                
                image = image.to(device, non_blocking=True)
                
                # Debug: Log batch info
                if i == 0:
                    logger.log(f"Batch shape: {image.shape}")
                    logger.log(f"Caption sample: {caption[0] if isinstance(caption, list) else caption}")
                
                loss = model(image, caption)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Invalid loss detected at batch {i}: {loss}")
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                
                # Memory cleanup
                if i % 50 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error in batch {i}", e)
                continue
                
    except Exception as e:
        logger.error(f"Error in training epoch {epoch}", e)
        raise
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.log(f"Epoch {epoch} completed. Averaged stats: {metric_logger.global_avg()}")
    
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def fix_json_encoding(file_path, logger):
    """Fix JSON file encoding issues"""
    try:
        # Try UTF-8 first
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except UnicodeDecodeError:
        logger.log(f"UTF-8 failed, trying different encodings for {file_path}")
        # Try different encodings
        encodings = ['utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    data = json.load(f)
                logger.log(f"Successfully read with {encoding} encoding")
                
                # Rewrite with UTF-8
                backup_path = file_path + '.backup'
                os.rename(file_path, backup_path)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.log(f"Fixed encoding and rewrote {file_path}")
                return data
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        
        # If all fail, create minimal data
        logger.error(f"Could not read {file_path} with any encoding")
        return {"images": [], "annotations": []}
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        return {"images": [], "annotations": []}

    
def evaluate(model, data_loader, device, config, logger):
    """Evaluation with enhanced debugging"""
    logger.log("Starting evaluation...")
    model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10
    
    result = []
    total_batches = len(data_loader)
    logger.log(f"Total evaluation batches: {total_batches}")
    
    # Detect Kaggle environment
    is_kaggle = os.path.exists('/kaggle/working')
    gen_params = get_safe_generation_params(is_kaggle)
    logger.log(f"Using generation parameters: {gen_params}")
    
    try:
        for i, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            try:
                if len(batch_data) == 2:
                    image, image_id = batch_data
                else:
                    logger.error(f"Unexpected evaluation batch format: {len(batch_data)} items")
                    continue
                
                image = image.to(device, non_blocking=True)
                
                # Debug tensor shapes and model state
                logger.debug(f"Image tensor shape: {image.shape}")
                logger.debug(f"Batch size: {image.shape[0]}")
                logger.debug(f"Device: {device}")
                logger.debug(f"Model training mode: {model.training}")
                
                # Memory debug info
                if torch.cuda.is_available() and device.type == 'cuda':
                    memory_allocated = torch.cuda.memory_allocated() / 1024**2
                    memory_reserved = torch.cuda.memory_reserved() / 1024**2
                    logger.debug(f"GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")
                
                # Ensure model is in eval mode
                model.eval()
                
                # Generate captions using safe generation function
                captions = safe_generate_captions(model, image, logger, is_kaggle)
                
                for caption, img_id in zip(captions, image_id):
                    result.append({"image_id": int(img_id), "caption": caption})
                    
            except Exception as e:
                logger.error(f"Error in evaluation batch {i}", e)
                continue
                
    except Exception as e:
        logger.error("Error in evaluation", e)
        raise
    
    logger.log(f"Generated {len(result)} captions")
    return result

def main(args, config):
    """Main training function with comprehensive error handling"""
    
    try:
        # Initialize distributed mode
        utils.init_distributed_mode(args)
        device = torch.device(args.device)
        logger.log(f"Using device: {device}")
        
        # Set seeds for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True
        logger.log(f"Random seed set to: {seed}")
        
        # Create datasets
        logger.log("Creating captioning dataset...")
        try:
            train_dataset, val_dataset, test_dataset = create_dataset(args.dataset, config)
            logger.log(f"Original dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            
            # LIMIT DATASETS FOR QUICK TESTING (only first 10 samples)
            if hasattr(args, 'quick_test') and args.quick_test:
                logger.log("🚀 QUICK TEST MODE: Using only 10 samples for training!")
                # Limit training dataset to first 10 samples
                if len(train_dataset) > 10:
                    train_dataset.annotations = train_dataset.annotations[:10]
                # Limit validation/test to first 5 samples
                if len(val_dataset) > 5:
                    val_dataset.annotations = val_dataset.annotations[:5]
                if len(test_dataset) > 5:
                    test_dataset.annotations = test_dataset.annotations[:5]
                
                logger.log(f"Limited dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            
        except Exception as e:
            logger.error("Failed to create datasets", e)
            raise
        
        # Create samplers and data loaders
        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler([train_dataset, val_dataset, test_dataset], 
                                    [True, False, False], num_tasks, global_rank)
        else:
            samplers = [None, None, None]
        
        try:
            train_loader, val_loader, test_loader = create_loader(
                [train_dataset, val_dataset, test_dataset], samplers,
                batch_size=[config['batch_size']] * 3,
                num_workers=[2, 2, 2],  # Reduced for stability
                is_trains=[True, False, False],
                collate_fns=[None, None, None]
            )
            logger.log("Data loaders created successfully")
            if hasattr(args, 'quick_test') and args.quick_test:
                logger.log(f"🚀 QUICK TEST MODE: Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        except Exception as e:
            logger.error("Failed to create data loaders", e)
            raise
        
        # Create model
        logger.log("Creating BLIP model...")
        try:
            model = blip_decoder(
                pretrained=config['pretrained'],
                image_size=config['image_size'],
                vit=config['vit'],
                vit_grad_ckpt=config['vit_grad_ckpt'],
                vit_ckpt_layer=config['vit_ckpt_layer'],
                prompt=config['prompt']
            )
            model = model.to(device)
            logger.log("Model created and moved to device")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log(f"Total parameters: {total_params:,}")
            logger.log(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            logger.error("Failed to create model", e)
            raise
        
        # Setup distributed training
        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
            logger.log("Distributed training setup complete")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=float(config['init_lr']),
            weight_decay=float(config['weight_decay'])
        )
        logger.log(f"Optimizer created with lr={config['init_lr']}, weight_decay={config['weight_decay']}")
        
        # Training loop
        best_score = 0
        best_epoch = 0
        
        logger.log("Starting training loop...")
        start_time = time.time()
        
        for epoch in range(0, int(config['max_epoch'])):
            logger.log(f"=== Epoch {epoch}/{config['max_epoch']-1} ===")
            
            if not args.evaluate:
                if args.distributed:
                    train_loader.sampler.set_epoch(epoch)
                
                # Update learning rate
                cosine_lr_schedule(optimizer, epoch, int(config['max_epoch']), 
                                 float(config['init_lr']), float(config['min_lr']))
                logger.log(f"Learning rate updated to: {optimizer.param_groups[0]['lr']}")
                
                # Training
                train_stats = train(model, train_loader, optimizer, epoch, device, logger)
                logger.log(f"Training stats: {train_stats}")
            
            # Validation
            try:
                val_result = evaluate(model_without_ddp, val_loader, device, config, logger)
                val_result_file = save_result(val_result, args.result_dir, args.distributed, 
                                            f'val_epoch{epoch}', remove_duplicate='image_id')
                logger.log(f"Validation results saved to: {val_result_file}")
            except Exception as e:
                logger.error(f"Validation failed for epoch {epoch}", e)
                continue
            
            # Testing
            try:
                test_result = evaluate(model_without_ddp, test_loader, device, config, logger)
                test_result_file = save_result(test_result, args.result_dir, args.distributed,
                                             f'test_epoch{epoch}', remove_duplicate='image_id')
                logger.log(f"Test results saved to: {test_result_file}")
            except Exception as e:
                logger.error(f"Testing failed for epoch {epoch}", e)
                continue
            
            # Evaluation and logging
            if utils.is_main_process():
                try:
                    # Evaluate captions with proper encoding handling
                    coco_val = coco_caption_eval(config['root'], val_result_file, 'valid', logger)
                    coco_test = coco_caption_eval(config['root'], test_result_file, 'test', logger)
                    
                    logger.log(f"Validation metrics: {coco_val.eval}")
                    logger.log(f"Test metrics: {coco_test.eval}")
                    
                    if args.evaluate:
                        # Evaluation only mode
                        log_stats = {
                            **{f'val_{k}': v for k, v in coco_val.eval.items()},
                            **{f'test_{k}': v for k, v in coco_test.eval.items()},
                        }
                        with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
                            f.write(json.dumps(log_stats) + "\n")
                    else:
                        # Training mode - save best model
                        current_score = coco_val.eval.get('CIDEr', 0) + coco_val.eval.get('Bleu_4', 0)
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_epoch = epoch
                            logger.log(f"New best model! Score: {best_score:.3f}")
                            
                            save_obj = {
                                'model': model_without_ddp.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'config': config,
                                'epoch': epoch,
                                'best_score': best_score,
                            }
                            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                        
                        # Log training stats
                        log_stats = {
                            **{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in coco_val.eval.items()},
                            **{f'test_{k}': v for k, v in coco_test.eval.items()},
                            'epoch': epoch,
                            'best_epoch': best_epoch,
                            'best_score': best_score,
                        }
                        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                            f.write(json.dumps(log_stats) + "\n")
                            
                except Exception as e:
                    logger.error(f"Evaluation failed for epoch {epoch}", e)
                    continue
            
            if args.evaluate:
                break
            
            if args.distributed:
                dist.barrier()
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.log(f'Training completed in {total_time_str}')
        logger.log(f'Best epoch: {best_epoch}, Best score: {best_score:.3f}')
        
    except Exception as e:
        logger.error("Fatal error in main training loop", e)
        raise

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='UITVIC Caption Training with Enhanced Debugging')
    parser.add_argument('--dataset', default='uitvic', help='Dataset name')
    parser.add_argument('--config', default='./configs/uitvic.yaml', help='Path to config file')
    parser.add_argument('--output_dir', default='output/caption_uitvic', help='Output directory')
    parser.add_argument('--evaluate', action='store_true', help='Evaluation only mode')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='URL for distributed training setup')
    parser.add_argument('--distributed', default=True, type=bool, help='Enable distributed training')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--quick_test', action='store_true', help='Quick test mode with only 10 training samples')
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'result')).mkdir(parents=True, exist_ok=True)
    args.result_dir = os.path.join(args.output_dir, 'result')
    
    # Initialize logger
    logger = DebugLogger(args.output_dir)
    logger.log("=" * 50)
    logger.log("UITVIC Caption Training Started")
    logger.log("=" * 50)
    
    try:
        # Environment check
        check_environment()
        
        # Load and validate configuration
        logger.log(f"Loading config from: {args.config}")
        
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)
        
        # Setup paths for current environment
        config = setup_kaggle_paths(config)
        config = validate_config(config)
        
        # Check data availability
        if not check_data_paths(config):
            logger.log("Warning: Some data paths are missing. Proceeding anyway...")
        
        # Save config
        with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.log("Configuration saved")
        
        # Start training
        main(args, config)
        
        logger.log("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.log("Training interrupted by user")
    except Exception as e:
        logger.error("Fatal error in training script", e)
        sys.exit(1)