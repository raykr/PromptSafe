import os
import torch
import lpips
import json
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
from PIL import Image

def load_lpips_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Replacing AlexNet with VGG network
    loss_fn = lpips.LPIPS(net='vgg').to(device)
    return loss_fn, device

def resize_image(image_path, target_size=(512, 512)):
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    return transform(img)

def compute_lpips_distance(loss_fn, device, img1_path, img2_path):
    # Check if the file exists
    if not os.path.exists(img1_path):
        raise FileNotFoundError(f"File not found: {img1_path}")
    if not os.path.exists(img2_path):
        raise FileNotFoundError(f"File not found: {img2_path}")
        
    # Load and resize the image
    img1 = resize_image(img1_path).unsqueeze(0).to(device)
    img2 = resize_image(img2_path).unsqueeze(0).to(device)
    
    # Calculate LPIPS distance
    with torch.no_grad():
        distance = loss_fn(img1, img2).item()
    
    return distance

def main():
    parser = argparse.ArgumentParser(description='Calculate the LPIPS distance between the generated image and the original image')
    parser.add_argument('--gen_dir', type=str, required=True, help='The directory path for generating images')
    parser.add_argument('--ori_dir', type=str, required=True, help='The directory path of the original image')
    parser.add_argument('--output', type=str, default='lpips_scores.json', help='Output JSON file path')
    args = parser.parse_args()
    
    # Check if the input directory exists
    if not os.path.exists(args.gen_dir):
        raise FileNotFoundError(f"The generated image directory does not exist: {args.gen_dir}")
    if not os.path.exists(args.ori_dir):
        raise FileNotFoundError(f"The original image directory does not exist: {args.ori_dir}")
    
    # loading model
    lpips_model, lpips_device = load_lpips_model()
    
    # Store results
    results = {}
    
    # Traverse all generated images
    for img_name in tqdm(os.listdir(args.gen_dir)):
        if not img_name.endswith('.png') and not img_name.endswith('.jpg'):
            continue
            
        gen_img_path = os.path.join(args.gen_dir, img_name)
        orig_img_path = os.path.join(args.ori_dir, img_name)
        
        try:
            # Calculate LPIPS distance
            lpips_score = compute_lpips_distance(
                lpips_model, lpips_device,
                gen_img_path, orig_img_path
            )
            
            # Store results
            results[img_name] = {
                "lpips_score": lpips_score
            }
        except Exception as e:
            print(f"Error processing image {img_name} : {str(e)}")
            continue
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"LPIPS scores saved to {args.output}")

if __name__ == "__main__":
    main() 