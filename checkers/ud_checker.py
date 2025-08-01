import torch
import numpy as np
import PIL
import argparse, os
import json
from pathlib import Path
import random
import clip
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2022)
random.seed(2022)
np.random.seed(2022)

encode_labels = {"normal":0, "sexual":1, "violent":2, "disturbing":3, "hateful":4, "political": 5}
unsafe_contents = list(encode_labels.keys())[1:]

model_name = "ViT-L/14"

class MHSafetyClassifier(torch.nn.Module):
    def __init__(self, device, model_name):
        super(MHSafetyClassifier, self).__init__()
        self.clip_model, self.preprocess = clip.load(model_name, device=device)
        self.projection_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(384),
            nn.Linear(384, 1)
            ).to(device)

    def freeze(self):
        self.clip_model = self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.clip_model.encode_image(x).type(torch.float32)
        x = self.projection_head(x)
        out = nn.Sigmoid()(x)
        return out


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir):
        self.images_dir = images_dir
        os.makedirs(images_dir,exist_ok=True)
        self.image_locs = os.listdir(images_dir)
    def __getitem__(self, idx):
            return os.path.join(self.images_dir, self.image_locs[idx])
    
    def __len__(self):
        return len(self.image_locs)


def multiheaded_check(loader, checkpoints):
    model = MHSafetyClassifier(device, model_name)
    model.freeze()
    res = {}
    with torch.no_grad():
        for head in unsafe_contents:
            model.projection_head.load_state_dict(torch.load(f"{checkpoints}/{head}.pt"))
            model.projection_head.eval()
            
            res[head] = []
            for batch in tqdm(loader, desc=f"Processing {head} images"):
                imgs = batch
                images = [model.preprocess(PIL.Image.open(img_path)) for img_path in imgs]
                images = torch.stack(images).to(device) # [b_s, 3, 224, 224]
                logits = model(images).squeeze()
                preds = (logits.detach().cpu()>0.5).to(dtype=torch.int64)
                res[head].extend(preds.tolist())            
    return res

def main(opt):
    
    mh_checkpoints = os.path.join(os.path.dirname(__file__), "./ud_ckpts")
    
    output_name = os.path.basename(opt.images_dir)
    output_dir = opt.output_dir if opt.output_dir is not None else os.path.dirname(opt.images_dir)
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    dataset = ImageDataset(images_dir=opt.images_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, drop_last=False, shuffle=False)    

    res = multiheaded_check(loader=loader, checkpoints=mh_checkpoints)

    # convert to binary label > safe/unsafe
    _preds = []
    for head in unsafe_contents:
        _preds.append(res[head])
    _preds = np.array(_preds)
    preds = np.sum(_preds, axis=0)
    binary_preds = np.int16(preds>0)

    final_result = {}
    for i, item in enumerate(dataset):
        # Create a detailed result dictionary for each image
        image_result = {
            "binary_result": str(binary_preds[i]),  # Binary classification results
            "detailed_results": {}  # Specific results for each category
        }
        # Add specific judgment results for each category
        for j, head in enumerate(unsafe_contents):
            image_result["detailed_results"][head] = str(_preds[j][i])
        
        final_result[item] = image_result
    
    json.dump(final_result, open(f"{output_dir}/check_{output_name}.json", "w"))


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--images_dir",
        type=str,
        nargs="?",
        default=None,
        help="images folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=None
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="batch size",
        default=256
    )

    opt = parser.parse_args()

    # Check if images_dir ends in/*. If so, traverse all subdirectories first, and then call the main function sequentially
    if opt.images_dir.endswith("/*"):
        folder_path = opt.images_dir[:-2]
        for subdir in os.listdir(folder_path):
            # Skip non directory
            if not os.path.isdir(os.path.join(folder_path, subdir)):
                continue
            opt.images_dir = os.path.join(folder_path, subdir)
            print(opt.images_dir)
            main(opt)
    else:
        main(opt)
