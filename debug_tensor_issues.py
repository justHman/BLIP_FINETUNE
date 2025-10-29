#!/usr/bin/env python3
"""
BLIP Model Tensor Debug Script
=============================

Script để debug và test tensor size issues trong BLIP model
Đặc biệt cho việc troubleshoot Kaggle environment

Usage:
    python debug_tensor_issues.py

Features:
- Test model loading và generation
- Debug tensor shapes trong attention mechanism
- Test các generation parameters khác nhau
- Memory usage monitoring
"""

import os
import sys
import torch
import yaml
import warnings
from PIL import Image
import torchvision.transforms as transforms

# Suppress warnings
warnings.filterwarnings("ignore")

def setup_paths():
    """Setup paths for current environment"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

def load_config():
    """Load config based on environment"""
    is_kaggle = os.path.exists('/kaggle/working')
    
    config_file = 'configs/uitvic_kaggle.yaml' if is_kaggle else 'configs/uitvic_enhanced.yaml'
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    return config, is_kaggle

def create_dummy_image(image_size=224):
    """Create a dummy image for testing"""
    # Create a dummy RGB image
    image = torch.randn(3, image_size, image_size)
    
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    
    return image

def test_model_loading(config):
    """Test model loading"""
    print("🔄 Testing model loading...")
    
    try:
        # Import models
        from models.blip import blip_decoder
        
        # Load model
        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
        model = blip_decoder(pretrained=model_url, image_size=config['image_size'], vit=config['vit'])
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_tensor_shapes(model, device, image_size=224):
    """Test tensor shapes through the model"""
    print(f"🔍 Testing tensor shapes with image size {image_size}...")
    
    model.eval()
    model = model.to(device)
    
    # Create dummy image batch
    batch_sizes = [1, 2] if device.type == 'cuda' else [1]
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing batch size: {batch_size} ---")
        
        try:
            # Create dummy image batch
            images = torch.stack([create_dummy_image(image_size) for _ in range(batch_size)])
            images = images.to(device)
            
            print(f"Input image shape: {images.shape}")
            
            with torch.no_grad():
                # Test với các parameters khác nhau
                generation_configs = [
                    {'sample': False, 'num_beams': 1, 'max_length': 10, 'min_length': 3},
                    {'sample': True, 'do_sample': True, 'max_length': 10, 'min_length': 3, 'temperature': 1.0},
                    {'sample': False, 'num_beams': 2, 'max_length': 15, 'min_length': 5},
                ]
                
                for i, gen_config in enumerate(generation_configs):
                    try:
                        print(f"  Testing config {i+1}: {gen_config}")
                        
                        # Monitor memory before generation
                        if torch.cuda.is_available():
                            memory_before = torch.cuda.memory_allocated() / 1024**2
                            print(f"  Memory before: {memory_before:.1f}MB")
                        
                        # Generate captions
                        captions = model.generate(images, **gen_config)
                        
                        # Monitor memory after generation
                        if torch.cuda.is_available():
                            memory_after = torch.cuda.memory_allocated() / 1024**2
                            print(f"  Memory after: {memory_after:.1f}MB")
                        
                        print(f"  ✓ Config {i+1} succeeded - Generated {len(captions)} captions")
                        print(f"  Sample caption: {captions[0][:50]}...")
                        
                    except RuntimeError as e:
                        error_msg = str(e)
                        print(f"  ❌ Config {i+1} failed: {error_msg}")
                        
                        if "size of tensor" in error_msg and "must match" in error_msg:
                            print(f"  → Tensor size mismatch detected!")
                            print(f"  → This is likely a beam search attention issue")
                        
                    except Exception as e:
                        print(f"  ❌ Config {i+1} unexpected error: {e}")
            
        except Exception as e:
            print(f"❌ Batch size {batch_size} failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Clear cache between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Main debug function"""
    print("=" * 60)
    print("🔧 BLIP Tensor Debug Tool")
    print("=" * 60)
    
    # Setup
    setup_paths()
    
    # Detect environment
    is_kaggle = os.path.exists('/kaggle/working')
    print(f"Environment: {'Kaggle' if is_kaggle else 'Local'}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load config
    try:
        config, is_kaggle = load_config()
        print(f"✓ Config loaded: {config.get('image_size', 224)}px images")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return 1
    
    # Test model loading
    model = test_model_loading(config)
    if model is None:
        return 1
    
    # Test tensor shapes
    image_sizes = [224, 384] if not is_kaggle else [224]
    
    for img_size in image_sizes:
        print(f"\n{'='*40}")
        test_tensor_shapes(model, device, img_size)
    
    print("\n" + "="*60)
    print("🎯 Debug completed!")
    print("="*60)
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)