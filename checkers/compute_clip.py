import os
import torch
import clip
import pandas as pd
from PIL import Image
import json
from tqdm import tqdm
import argparse

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    return model, preprocess, device

def truncate_text(text, max_length=77):
    """Truncate text to adapt to CLIP's token restrictions"""
    tokens = clip.tokenize([text], truncate=True)[0]
    return text[:max_length]  # 简单截断，保留前77个字符

def parse_image_id(img_name: str) -> int:
    """Parse sample id from image filename (e.g. 0.png, 10_0.png for SD3)."""
    stem = img_name.rsplit(".", 1)[0]
    if "_" in stem:
        # SD3/Flux infer: {id}_{image_index}.png — do not use int(stem); int("10_0")==100 in Python
        return int(stem.rsplit("_", 1)[0])
    return int(stem)

def compute_clip_similarity(model, preprocess, device, image_path, text):
    # Load and preprocess images
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Truncate the text and encode it
    text = truncate_text(text)
    text = clip.tokenize([text]).to(device)
    
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        # Normalized features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = (image_features @ text_features.T).item()
    
    return similarity

def main():
    parser = argparse.ArgumentParser(description='Calculate the CLIP similarity between the generated image and the original prompt')
    parser.add_argument('--csv_path', type=str, required=True, help='CSV file path containing original prompt')
    parser.add_argument('--image_dir', type=str, required=True, help='The directory path for generating images')
    parser.add_argument('--output', type=str, default='clip_scores.json', help='Output JSON file path')
    args = parser.parse_args()
    
    # loading model
    clip_model, clip_preprocess, clip_device = load_clip_model()
    
    # read prompt
    df = pd.read_csv(args.csv_path)
    
    # Store results
    results = {}
    
    # Traverse all generated images
    for img_name in tqdm(os.listdir(args.image_dir)):
        if not img_name.endswith('.png') and not img_name.endswith('.jpg'):
            continue
            
        img_num = parse_image_id(img_name)
        img_path = os.path.join(args.image_dir, img_name)
        
        # Retrieve prompt by id column or row index (VEIL CSVs have no id column)
        if 'id' in df.columns:
            prompt = df[df['id'] == img_num]["prompt"].values[0]
        else:
            prompt = df.iloc[img_num]["prompt"]
        
        # Calculate CLIP similarity
        clip_score = compute_clip_similarity(
            clip_model, clip_preprocess, clip_device,
            img_path, prompt
        )
        
        # Store results
        results[img_name] = {
            "clip_score": clip_score * 100
        }
    
    # Save the Results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"CLIP scores saved to {args.output}")

if __name__ == "__main__":
    main() 