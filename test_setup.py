#!/usr/bin/env python3
"""
Test script for UITVIC training setup
Validates dataset, configuration, and model loading
"""

import os
import sys
import json
import torch
from pathlib import Path
from ruamel.yaml import YAML

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        import torchvision
        import transformers
        import timm
        from PIL import Image
        print("✓ Core packages imported successfully")
        
        # Test BLIP specific imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from models.blip import blip_decoder
        from data.uitvic_dataset import UITVIC_DATASET
        from data import create_dataset
        print("✓ BLIP modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_dataset_structure():
    """Test dataset structure and files"""
    print("\nTesting dataset structure...")
    
    # Determine base path
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_path, 'dataset', 'uitvic_dataset')
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset directory not found: {dataset_path}")
        return False
    
    required_files = [
        'train_ann.json',
        'test_ann.json'
    ]
    
    required_dirs = [
        'train/train',
        'test/test'
    ]
    
    all_good = True
    
    # Check files
    for file_name in required_files:
        file_path = os.path.join(dataset_path, file_name)
        if os.path.exists(file_path):
            print(f"✓ Found {file_name}")
            
            # Check JSON structure
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'images' in data and 'annotations' in data:
                    print(f"  - {len(data['images'])} images, {len(data['annotations'])} annotations")
                else:
                    print(f"  - Warning: Unexpected JSON structure in {file_name}")
            except Exception as e:
                print(f"  - Error reading {file_name}: {e}")
                all_good = False
        else:
            print(f"✗ Missing {file_name}")
            all_good = False
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if os.path.exists(dir_path):
            image_count = len([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"✓ Found {dir_name} with {image_count} images")
        else:
            print(f"✗ Missing directory {dir_name}")
            all_good = False
    
    return all_good

def test_configuration():
    """Test configuration files"""
    print("\nTesting configuration...")
    
    config_files = [
        'configs/uitvic.yaml',
        'configs/uitvic_enhanced.yaml'
    ]
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    yaml = YAML(typ='rt')
    
    for config_file in config_files:
        config_path = os.path.join(base_path, config_file)
        if os.path.exists(config_path):
            try:
                config = yaml.load(open(config_path, 'r'))
                print(f"✓ {config_file} loaded successfully")
                
                # Check key parameters
                key_params = ['image_size', 'batch_size', 'max_epoch', 'prompt']
                for param in key_params:
                    if param in config:
                        print(f"  - {param}: {config[param]}")
                    else:
                        print(f"  - Warning: {param} not found in config")
                        
            except Exception as e:
                print(f"✗ Error loading {config_file}: {e}")
                return False
        else:
            print(f"✗ Config file not found: {config_file}")
    
    return True

def test_model_loading():
    """Test model creation and loading"""
    print("\nTesting model loading...")
    
    try:
        from models.blip import blip_decoder
        
        # Test model creation without pretrained weights
        model = blip_decoder(
            pretrained='',  # No pretrained weights for testing
            image_size=384,
            vit='base',
            prompt='một bức ảnh về '
        )
        
        print("✓ Model created successfully")
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # Test forward pass with dummy data
        dummy_image = torch.randn(1, 3, 384, 384)
        dummy_caption = ["một bức ảnh về con mèo"]
        
        with torch.no_grad():
            # Test loss computation
            loss = model(dummy_image, dummy_caption)
            print(f"✓ Forward pass successful, loss: {loss.item():.4f}")
            
            # Test generation
            captions = model.generate(dummy_image, sample=False, num_beams=3, max_length=20, min_length=5)
            print(f"✓ Generation successful: {captions[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test dataset loading"""
    print("\nTesting dataset loading...")
    
    try:
        import torchvision.transforms as transforms
        from torchvision.transforms import InterpolationMode
        
        # Create transform
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                       (0.26862954, 0.26130258, 0.27577711))
        transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        
        # Test dataset creation
        from data.uitvic_dataset import UITVIC_DATASET
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        dataset = UITVIC_DATASET(
            transform=transform,
            root=os.path.join(base_path, 'dataset', 'uitvic_dataset'),
            split='train',
            image_dir=os.path.join(base_path, 'dataset', 'uitvic_dataset', 'train', 'train'),
            ann_path=os.path.join(base_path, 'dataset', 'uitvic_dataset', 'train_ann.json'),
            prompt='một bức ảnh về '
        )
        
        print(f"✓ Dataset created successfully with {len(dataset)} samples")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            if len(sample) == 3:  # training mode
                image, caption, img_id = sample
                print(f"✓ Sample loaded - Image shape: {image.shape}, Caption: {caption[:50]}...")
            else:  # evaluation mode
                image, img_id = sample
                print(f"✓ Sample loaded - Image shape: {image.shape}, Image ID: {img_id}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script():
    """Test training script import and basic functionality"""
    print("\nTesting training script...")
    
    try:
        # Import training script components
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Try to import the main training functions
        exec(open('train_caption_uitvic.py').read(), {'__name__': '__test__'})
        print("✓ Training script syntax is valid")
        
        return True
        
    except Exception as e:
        print(f"✗ Training script error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("UITVIC Training Setup Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Dataset Structure", test_dataset_structure),
        ("Configuration Files", test_configuration),
        ("Model Loading", test_model_loading),
        ("Dataset Loading", test_dataset_loading),
        ("Training Script", test_training_script),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to start training.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues before training.")
        
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)