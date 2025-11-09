from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def load_demo_image(image_path=None, image_size=384, device='cpu'):
    if image_path:
         raw_image = Image.open(image_path).convert('RGB')
    else:
        img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   

    w, h = raw_image.size
    raw_image.show()
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device) 
    print(f'Image shape: {image.shape}')  
    return image

def parse_args():
    parser = argparse.ArgumentParser(description="BLIP Inference Script")
    parser.add_argument('--image_path', type=str, default=None, help='Path to the input image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on (cuda or cpu)')
    parser.add_argument('--sample', action='store_true', help='Use nucleus sampling instead of beam search')
    return parser.parse_args()

def main(args):
    image = load_demo_image(image_path=args.image_path, image_size=384, device=args.device)  
    model = blip_decoder(pretrained=args.model_path, image_size=384, vit='base')
    model.eval()
    model = model.to(args.device)
    print('Model loaded.')

    with torch.no_grad():
        if args.sample:
            # nucleus sampling
            caption = model.generate(image, sample=args.sample, top_p=0.9, max_length=20, min_length=5) 
        else:
            # beam search
            caption = model.generate(image, sample=args.sample, num_beams=3, max_length=20, min_length=5) 
        print('caption: '+caption[0])

if __name__ == '__main__':
    args = parse_args()
    main(args)