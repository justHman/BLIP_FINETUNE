import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from torch.utils.data import Dataset
from PIL import Image

from data.utils import vn_pre_caption, unaccented_vn_pre_caption

class UITVIC_DATASET(Dataset):
    def __init__(
        self, 
        dataset, transform, 
        root, split='train',
        image_dir=None, ann_path=None,
        max_words=30, 
        tokenizer= 'bert-base-uncased', prompt='một bức ảnh về '
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.root = root
        self.split = split
        self.max_words = max_words
        self.prompt = prompt if prompt is not None else ''

        self.image_dir = image_dir
        self.ann_path = ann_path

        ann_path = self.ann_path
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann_file = json.load(f)

        images = ann_file.get('images', []) 
        self.imgid2f = {}
        for image in images:
            fname = image['file_name']
            image_id = image['id']
            if fname is None or image_id is None:
                continue
            self.imgid2f[image_id] = fname

        annotations = ann_file.get('annotations', [])
        self.annotations = []
        for ann in annotations:
            image_id = ann['image_id']

            fname = self.imgid2f[image_id]
            caption = ann.get('caption', '')
            id = ann["id"]
            if fname is None:
                # try building filename from id (zero-padded 12 by COCO style)
                try:
                    fname = f"{image_id:012d}.jpg"
                except Exception:
                    continue
            self.annotations.append({'image': fname, 'caption': caption, 'image_id': image_id, 'id': id})

        # build internal img index mapping (unique images)
        self.img_ids = {}
        order = 0
        for ann_path in self.annotations:
            image_id = ann_path['image_id']
            if image_id not in self.img_ids:
                self.img_ids[image_id] = order
                order += 1
        
        if split != "train":
            self._create_ground_truth_files()

    def _create_ground_truth_files(self):
        """Create COCO-style ground truth files for validation and test splits."""
        gt_file = os.path.join(self.root, f"{self.dataset}_{self.split}_gt.json")

        # Create COCO-style ground truth
        images = []
        for ann in self.annotations:
            image_id = ann['image_id']
            if {'id': image_id} not in images:
                images.append({'id': image_id})

        annotations = []
        for ann in self.annotations:
            image_id = ann['image_id']
            id = ann['id']
            caption = ann['caption']
            if self.tokenizer == 'bert-base-uncased':
                caption = unaccented_vn_pre_caption(ann.get('caption', ''), max_words=100)
            
            annotations.append({"image_id": image_id, "caption": caption, "id": id})

        gt_data = {
            "info": {"description": f"{self.dataset} dataset"},
            "images": images,
            "annotations": annotations
        }

        # Save to file
        with open(gt_file, 'w', encoding='utf-8') as f:
            json.dump(gt_data, f, ensure_ascii=False, indent=4)
        print(f"Ground truth file created: {gt_file}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        if self.image_dir:
            image_dir = self.image_dir
        else:
            image_dir = self.root

        image_path = os.path.join(image_dir, ann['image'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.split == 'train':
            if self.tokenizer == 'bert-base-uncased':
                caption = self.prompt + unaccented_vn_pre_caption(ann.get('caption', ''), self.max_words)
            else:
                caption = self.prompt + vn_pre_caption(ann.get('caption', ''), self.max_words)
            return image, caption, self.img_ids[ann['image_id']]

        else:
            return image, ann['image_id']
        
class KTVIC_DATASET(Dataset):
    def __init__(
        self, 
        dataset, transform, 
        root, split='train',
        image_dir=None, ann_path=None,
        max_words=30, 
        tokenizer= 'bert-base-uncased', prompt='mot buc anh ve '
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.split = split
        self.max_words = max_words
        self.prompt = prompt if prompt is not None else ''

        self.root = root
        self.image_dir = image_dir
        self.ann_path = ann_path

        ann_path = self.ann_path
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann_file = json.load(f)

        images = ann_file.get('images', []) 
        self.imgid2f = {}
        for image in images:
            fname = image['filename']
            image_id = image['id']
            if fname is None or image_id is None:
                continue
            self.imgid2f[image_id] = fname

        annotations = ann_file.get('annotations', [])
        self.annotations = []
        for ann in annotations:
            image_id = ann['image_id']

            fname = self.imgid2f[image_id]
            caption = ann.get('caption', '')
            id = ann["id"]
            if fname is None:
                # try building filename from id (zero-padded 12 by COCO style)
                try:
                    fname = f"{image_id:012d}.jpg"
                except Exception:
                    continue
            self.annotations.append({'image': fname, 'caption': caption, 'image_id': image_id, 'id': id})

        # build internal img index mapping (unique images)
        self.img_ids = {}
        order = 0
        for ann_path in self.annotations:
            image_id = ann_path['image_id']
            if image_id not in self.img_ids:
                self.img_ids[image_id] = order
                order += 1
        
        if split != "train":
            self._create_ground_truth_files()

    def _create_ground_truth_files(self):
        """Create COCO-style ground truth files for validation and test splits."""
        gt_file = os.path.join(self.root, f"{self.dataset}_{self.split}_gt.json")

        # Create COCO-style ground truth
        images = []
        for ann in self.annotations:
            image_id = ann['image_id']
            if {'id': image_id} not in images:
                images.append({'id': image_id})

        annotations = []
        for ann in self.annotations:
            image_id = ann['image_id']
            id = ann['id']
            caption = ann['caption']
            if self.tokenizer == 'bert-base-uncased':
                caption = unaccented_vn_pre_caption(ann.get('caption', ''), max_words=100)
            
            annotations.append({"image_id": image_id, "caption": caption, "id": id})

        gt_data = {
            "info": {"description": f"{self.dataset} dataset"},
            "images": images,
            "annotations": annotations
        }

        # Save to file
        with open(gt_file, 'w', encoding='utf-8') as f:
            json.dump(gt_data, f, ensure_ascii=False, indent=4)
        print(f"Ground truth file created: {gt_file}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        if self.image_dir:
            image_dir = self.image_dir
        else:
            image_dir = self.root

        image_path = os.path.join(image_dir, ann['image'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.split == 'train':
            if self.tokenizer == 'bert-base-uncased':
                caption = self.prompt + unaccented_vn_pre_caption(ann.get('caption', ''), self.max_words)
            else:
                caption = self.prompt + vn_pre_caption(ann.get('caption', ''), self.max_words)
            return image, caption, self.img_ids[ann['image_id']]

        else:
            return image, ann['image_id']

class CUSTOM_DATASET(Dataset):
    def __init__(
        self, 
        dataset, transform, 
        root, split='train',
        image_dir=None, ann_path=None,
        max_words=30, 
        tokenizer= 'bert-base-uncased', prompt='mot buc anh ve '
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.root = root
        self.split = split
        self.max_words = max_words
        self.prompt = prompt if prompt is not None else ''

        if image_dir is None or ann_path is None:
            return None
        
        self.image_dir = image_dir
        self.ann_path = ann_path

        ann_path = self.ann_path
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann_file = json.load(f)

        images = ann_file.get('images', []) 
        self.imgid2f = {}
        for image in images:
            fname = image['file_name']
            image_id = image['id']
            if fname is None or image_id is None:
                continue
            self.imgid2f[image_id] = fname

        annotations = ann_file.get('annotations', [])
        self.annotations = []
        for ann in annotations:
            image_id = ann['image_id']

            fname = self.imgid2f[image_id]
            caption = ann.get('caption', '')
            id = ann["id"]
            # if fname is None:
            #     # try building filename from id (zero-padded 12 by COCO style)
            #     try:
            #         fname = f"{image_id:012d}.jpg"
            #     except Exception:
            #         continue
            self.annotations.append({'image': fname, 'caption': caption, 'image_id': image_id, 'id': id})

        # build internal img index mapping (unique images)
        self.img_ids = {}
        order = 0
        for ann_path in self.annotations:
            image_id = ann_path['image_id']
            if image_id not in self.img_ids:
                self.img_ids[image_id] = order
                order += 1
        
        if split != "train":
            self._create_ground_truth_files()

    def _create_ground_truth_files(self):
        """Create COCO-style ground truth files for validation and test splits."""
        gt_file = os.path.join(self.root, f"{self.dataset}_{self.split}_gt.json")

        # Create COCO-style ground truth
        images = []
        for ann in self.annotations:
            image_id = ann['image_id']
            if {'id': image_id} not in images:
                images.append({'id': image_id})

        annotations = []
        for ann in self.annotations:
            image_id = ann['image_id']
            id = ann['id']
            caption = ann['caption']
            if self.tokenizer == 'bert-base-uncased':
                caption = unaccented_vn_pre_caption(ann.get('caption', ''), max_words=100)
            
            annotations.append({"image_id": image_id, "caption": caption, "id": id})

        gt_data = {
            "info": {"description": f"{self.dataset} dataset"},
            "images": images,
            "annotations": annotations
        }

        # Save to file
        with open(gt_file, 'w', encoding='utf-8') as f:
            json.dump(gt_data, f, ensure_ascii=False, indent=4)
        print(f"Ground truth file created: {gt_file}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        if self.image_dir:
            image_dir = self.image_dir
        else:
            image_dir = self.root

        image_path = os.path.join(image_dir, ann['image'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.split == 'train':
            if self.tokenizer == 'bert-base-uncased':
                caption = self.prompt + unaccented_vn_pre_caption(ann.get('caption', ''), self.max_words)
            else:
                caption = self.prompt + vn_pre_caption(ann.get('caption', ''), self.max_words)
            return image, caption, self.img_ids[ann['image_id']]

        else:
            return image, ann['image_id']
        

if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torchvision.transforms import InterpolationMode
    from transform.randaugment import RandomAugment

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(384 ,scale=(0.5, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        

    uitvic_dataset = UITVIC_DATASET(
        transform=transform_train,
        root=r"dataset/uitvic_dataset",
        split='train',
        image_dir=r"dataset\uitvic_dataset\coco_uitvic_test\coco_uitvic_test",
        ann_path=r"dataset\uitvic_dataset\uitvic_captions_test2017.json",
        prompt='mot buc anh ve ', tokenizer='bert-base-uncased'
    )
    print(f"UITVIC dataset size: {len(uitvic_dataset)}")

    ktvic_dataset = KTVIC_DATASET(
        dataset="ktvic",
        transform=transform_train,
        root=r"dataset/ktvic_dataset",
        split='train',
        image_dir=r"dataset\ktvic_dataset\public-test-images",
        ann_path=r"dataset\ktvic_dataset\test_data.json",
        prompt='mot buc anh ve ', tokenizer='bert-base-uncased'
    )
    print(f"KTVIC dataset size: {len(ktvic_dataset)}")

    merged_dataset = MERGE_Dataset([uitvic_dataset, ktvic_dataset])
    print(f"Total size of merged dataset: {len(merged_dataset)}")
    for i in range(5):
        img, cap, img_id = merged_dataset[i]
        print(f"Image shape: {img.shape}")
        print(f"Caption: {cap}")
        print(f"Image ID: {img_id}")
        
