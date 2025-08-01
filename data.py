import pandas as pd
from torch.utils.data import Dataset
from utils.dataset_utils import prompt_class


class PromptPairDataset(Dataset):
    def __init__(self, csv_path, tokenizer, placeholder_token="<safety>", position="end", repeats=100, set="train"):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.position = position
        self.repeats = repeats
        self.num_prompts = len(self.data)
        self._length = self.num_prompts
        if set == "train":
            self._length *= repeats

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        prompt = self.data.iloc[idx % self.num_prompts]["prompt"]
        rewritten_prompt = self.data.iloc[idx % self.num_prompts]["rewritten_prompt"]
        label = self.data.iloc[idx % self.num_prompts]["label"]
        label = -1 if label == "toxic" else 1

        if self.position == "start":
            pseudo_prompt = f"{self.placeholder_token} {prompt}"
            pseudo_rewritten = f"{self.placeholder_token} {rewritten_prompt}"
        else:
            pseudo_prompt = f"{prompt} {self.placeholder_token}"
            pseudo_rewritten = f"{rewritten_prompt} {self.placeholder_token}"
        
        return {"prompt": str(prompt), "rewritten_prompt": str(rewritten_prompt), "pseudo_prompt": str(pseudo_prompt), "pseudo_rewritten": str(pseudo_rewritten), "label": label}


class GatedDataset(Dataset):
    def __init__(self, dataset,need_ids=True):
        self.dataset = dataset
        self.need_ids=need_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt,label,src,id=self.dataset[idx]
        if src in prompt_class:
            category_id=prompt_class[src]
        else:
          raise ValueError(f'The source {src} is not in prompt_class')

        if self.need_ids:
            return prompt,int(category_id),int(label),int(id)
        else:
            return prompt,int(category_id),int(label)