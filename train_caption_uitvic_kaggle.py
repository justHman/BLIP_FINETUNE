#!/usr/bin/env python3
"""
KAGGLE-OPTIMIZED UITVIC Training Script
======================================

Script đặc biệt được tối ưu hóa để chạy trên Kaggle environment
Tự động detect environment và áp dụng config phù hợp

Usage:
    python train_caption_uitvic_kaggle.py

Features:
- Tự động detect Kaggle environment
- Chọn config file phù hợp (kaggle vs local)
- Ultra-safe generation parameters cho Kaggle
- Enhanced error handling và logging
- Memory optimization cho Kaggle GPU limits
"""

import os
import sys
import argparse
import warnings
import torch
import yaml

# Suppress warnings
warnings.filterwarnings("ignore")

def detect_environment():
    """Detect if running on Kaggle or local environment"""
    is_kaggle = os.path.exists('/kaggle/working')
    is_colab = 'COLAB_GPU' in os.environ
    
    env_info = {
        'is_kaggle': is_kaggle,
        'is_colab': is_colab,
        'is_local': not (is_kaggle or is_colab),
        'platform': 'kaggle' if is_kaggle else ('colab' if is_colab else 'local')
    }
    
    return env_info

def setup_kaggle_environment():
    """Setup Kaggle-specific environment"""
    print("🚀 Setting up Kaggle environment...")
    
    # Set environment variables for Kaggle
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("✓ Kaggle environment setup complete")

def get_config_path(env_info):
    """Get appropriate config file based on environment"""
    if env_info['is_kaggle']:
        return 'configs/uitvic_kaggle.yaml'
    else:
        return 'configs/uitvic_enhanced.yaml'

def main():
    """Main function with environment auto-detection"""
    print("=" * 60)
    print("🎯 UITVIC BLIP Training - Kaggle Optimized")
    print("=" * 60)
    
    # Detect environment
    env_info = detect_environment()
    print(f"🌍 Environment detected: {env_info['platform'].upper()}")
    
    # Setup environment-specific configurations
    if env_info['is_kaggle']:
        setup_kaggle_environment()
    
    # Get appropriate config
    config_path = get_config_path(env_info)
    print(f"📋 Using config: {config_path}")
    
    # Setup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=config_path)
    parser.add_argument('--output_dir', default='output/caption_uitvic')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--distributed', default=False)
    
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        print(f"✓ Config loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return 1
    
    # Environment-specific adjustments
    if env_info['is_kaggle']:
        print("🔧 Applying Kaggle-specific optimizations...")
        
        # Force ultra-safe settings for Kaggle
        config['batch_size'] = 1
        config['num_workers'] = 0
        config['pin_memory'] = False
        config['image_size'] = 224
        config['max_length'] = 15
        config['num_beams'] = 1
        
        # Memory optimization
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        
        print("✓ Kaggle optimizations applied")
    
    # Import and run main training script
    try:
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import main training function
        from train_caption_uitvic import main as train_main
        
        print("🚀 Starting training...")
        result = train_main(args, config)
        
        print("✅ Training completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)