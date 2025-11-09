import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval, coco_karpathy_retrieval_eval
from data.vic_dataset import UITVIC_DATASET, KTVIC_DATASET, CUSTOM_DATASET
from transform.randaugment import RandomAugment

def create_dataset(dataset, config, min_scale=0.5):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
    
    if dataset.lower() == 'uitvic':
        train_dataset = UITVIC_DATASET(
            dataset, transform=transform_train,
            root=config["root"], split='train',
            image_dir=config.get("train_image_dir", None), ann_path=config.get("train_ann", None),
            tokenizer=config['tokenizer'], prompt=config.get("prompt", 'một bức ảnh về ')
        )
        test_dataset = UITVIC_DATASET(
            dataset, transform=transform_test,
            root=config["root"], split='test',
            image_dir=config.get("test_image_dir", None), ann_path=config.get("test_ann", None),
            tokenizer=config['tokenizer'], prompt=config.get("prompt", 'một bức ảnh về ')
        )
        val_dataset = UITVIC_DATASET(
            dataset, transform=transform_test,
            root=config["root"], split='valid',
            image_dir=config.get("valid_image_dir", None), ann_path=config.get("valid_ann", None),
            tokenizer=config['tokenizer'], prompt=config.get("prompt", 'một bức ảnh về ')
        )

        return train_dataset, val_dataset, test_dataset

    elif dataset == 'ktvic':
        train_dataset = KTVIC_DATASET(
            dataset=dataset,
            transform=transform_train,
            root=config["root"], split='train',
            image_dir=config.get("train_image_dir", None), ann_path=config.get("train_ann", None),
            tokenizer=config['tokenizer'], prompt=config.get("prompt", 'một bức ảnh về ')
        )
        val_dataset = KTVIC_DATASET(
            dataset=dataset,
            transform=transform_train,
            root=config["root"], split='valid',
            image_dir=config.get("valid_image_dir", None), ann_path=config.get("valid_ann", None),
            tokenizer=config['tokenizer'], prompt=config.get("prompt", 'một bức ảnh về ')
        )
        test_dataset = KTVIC_DATASET(
            dataset=dataset,
            transform=transform_train,
            root=config["root"], split='test',
            image_dir=config.get("test_image_dir", None), ann_path=config.get("test_ann", None),
            tokenizer=config['tokenizer'], prompt=config.get("prompt", 'một bức ảnh về ')
        )
        return train_dataset, val_dataset, test_dataset
    
    elif dataset == 'custom':
        train_dataset = CUSTOM_DATASET(
            dataset=dataset,
            transform=transform_train,
            root=config["root"], split='train',
            image_dir=config.get("train_image_dir", None), ann_path=config.get("train_ann", None),
            tokenizer=config['tokenizer'], prompt=config.get("prompt", 'mot buc anh ve ')
        )
        val_dataset = CUSTOM_DATASET(
            dataset=dataset,
            transform=transform_train,
            root=config["root"], split='valid',
            image_dir=config.get("valid_image_dir", None), ann_path=config.get("valid_ann", None),
            tokenizer=config['tokenizer'], prompt=config.get("prompt", 'mot buc anh ve ')
        )
        test_dataset = CUSTOM_DATASET(
            dataset=dataset,
            transform=transform_train,
            root=config["root"], split='test',
            image_dir=config.get("test_image_dir", None), ann_path=config.get("test_ann", None),
            tokenizer=config['tokenizer'], prompt=config.get("prompt", 'mot buc anh ve ')
        )
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='caption_coco':   
        train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'], prompt=config['prompt'])
        val_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='retrieval_coco':          
        train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'])
        val_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val') 
        test_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          
        return train_dataset, val_dataset, test_dataset 
    
    
# def create_sampler(datasets, shuffles, num_tasks, global_rank):
#     samplers = []
#     for dataset,shuffle in zip(datasets,shuffles):
#         sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
#         samplers.append(sampler)
#     return samplers     


# def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
#     loaders = []
#     for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
#         if is_train:
#             shuffle = (sampler is None)
#             drop_last = True
#         else:
#             shuffle = False
#             drop_last = False
#         loader = DataLoader(
#             dataset,
#             batch_size=bs,
#             num_workers=n_worker,
#             pin_memory=True,
#             sampler=sampler,
#             shuffle=shuffle,
#             collate_fn=collate_fn,
#             drop_last=drop_last,
#         )              
#         loaders.append(loader)
#     return loaders  

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        if dataset is None:  # Bỏ qua nếu dataset là None
            samplers.append(None)
            continue
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if dataset is None:  # Bỏ qua nếu dataset là None
            loaders.append(None)
            continue
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
  

