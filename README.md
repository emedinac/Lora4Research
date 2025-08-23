# Lora4Research

Fine-tuning and evaluation framework for offline LLMs using Low-Rank Adaptation (LoRA).

A fully offline system for research Q&A, enabling efficient fine-tuning of locally-hosted large language models and automatic metric evaluation.

| Model            | BLEU | ROUGE-L | BERTScore (F1) |
| -------------------- | ---- | ------- | --- |
| HF TinyLlama-1.1B (Baseline)  | 3.20 | 0.0980  | 0.8118 |
| LoRa-TinyLlama-1.1B (1 epoch) | 3.56 | 0.1127  | 0.8139 |
| LoRa-TinyLlama-1.1B (5 epoch) | 3.56 | 0.1120  | 0.8138 |

NOTE: For simplicity, all results were trained and tested on the same seed using only **0.1%** of the dataset.

- Train size: 3229
- Validation size: 130
- Test size: 130

## Key Features

LoRA Fine-tuning: Low-rank adaptation layers applied to base LLMs for efficient fine-tuning. Implemented via `PEFT` from Huggingface.

Computes automatically standard NLP metrics such as:

- BLEU (sacrebleu): evaluates n-gram overlap.
- ROUGE-L (rouge-score): evaluates longest common subsequence.
- BERTScore (bertscore): evaluates semantic similarity using contextual embeddings.

Specifically, Added:

- [X] LoRa implementation
- [X] metric evaluations (e.g. sacrebleu, rouge, bertscore)
- [ ] Preference Optimization (DTO for simplicity)

## Install

```bash
python3.10 -m venv prj4
source prj4/bin/activate
pip install --upgrade pip
pip install uv
uv init . 
uv sync --active
```

## Run

To fine-tune a model, run the following command, considering `lora_r` and `lora_alpha` as the main parameters. Follow the documentation, [PEFT](https://github.com/huggingface/peft/).

Command example for model training:

```bash
python3 src/train.py \
    --data_dir scillm/scientific_papers-archive \
    --output_dir lora/ \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --subset_fraction 0.001 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --num_processes 0 \
    --batch_preprocess 0 
```

Command example for model evaluation.

```bash
python3 src/test.py \
    --model_path lora/ \
    --data_dir scillm/scientific_papers-archive \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --max_new_tokens 512 \
    --subset_fraction 0.001 
```
