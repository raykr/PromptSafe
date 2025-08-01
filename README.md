# PromptSafe: Gated Prompt Tuning for Safe Text-to-Image Generation


## Prepare

```shell
conda create -n promptsafe python=3.12.0
pip install -r requirements.txt
```


## Training

Train soft prompt-guided embedding:
```shell
bash scripts/train.sh
```

Train gated network:
```shell
python train_gated.py
```

## Inference
```shell
bash scripts/eval.sh
```

