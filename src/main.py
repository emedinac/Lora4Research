import argparse
import os
import torch
from pathlib import Path
from datasets import load_dataset
import loaders
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    default_data_collator,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
# Inspired by https://github.com/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/lora_adapters.md


def format_sample(ex):
    return f"<problem>\n{ex['input']}\n</problem>\n<approach>\n{ex['output']}\n</approach>"


def build_label_mask(input_ids, tokenizer):
    labels = input_ids.clone()
    aid = tokenizer.convert_tokens_to_ids("<approach>")
    idx = (input_ids == aid).nonzero(as_tuple=True)
    if len(idx[0]):  # check if the token exists
        labels[: idx[0][0] + 1] = -100
    return labels


def preprocess(ex, tokenizer, max_seq_length=512):
    enc = tokenizer(format_sample(ex),
                    max_length=max_seq_length,
                    truncation=True,
                    padding="max_length")
    enc["labels"] = build_label_mask(torch.tensor([enc["input_ids"]]),
                                     tokenizer)[0].tolist()
    return enc


def preprocess_batch(examples, tokenizer, max_seq_length):
    texts = [f"<problem>\n{i}\n</problem>\n<approach>\n{o}\n</approach>"
             for i, o in zip(examples["input"], examples["output"])]
    enc = tokenizer(texts,
                    max_length=max_seq_length,
                    truncation=True,
                    padding="max_length")
    # build label masks
    labels = []
    for input_ids in enc["input_ids"]:
        labels.append(build_label_mask(
            torch.tensor([input_ids]), tokenizer)[0].tolist())
    enc["labels"] = labels
    return enc


def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens":
                                  ["<problem>",
                                   "</problem>",
                                   "<approach>",
                                   "</approach>"]
                                  }
                                 )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    config = AutoConfig.from_pretrained(args.model_name)
    max_seq_length = config.max_position_embeddings
    if args.max_seq_length > max_seq_length:
        args.max_seq_length = max_seq_length

    # Load and preprocess the dataset
    if not Path(f"data/processed-{args.max_seq_length}-{args.model_name}").exists():
        if args.ignore_ood_cases:
            db_skips = loaders.get_ood_ids(args.data_dir,
                                           max_tokens=args.max_seq_length)
        dataset = load_dataset(args.data_dir, "default", streaming=True)
        if args.num_processes == 0 and args.batch_preprocess == 0:
            dataset = dataset.map(lambda x: preprocess(x,
                                                       tokenizer,
                                                       max_seq_length
                                                       )
                                  )
        else:
            dataset = dataset.map(lambda x: preprocess_batch(x, tokenizer, max_seq_length),
                                  batched=True,
                                  batch_size=args.batch_preprocess,
                                  num_proc=args.num_processes,
                                  )
        dataset.save_to_disk(
            f"data/processed-{args.max_seq_length}-{args.model_name}")
    else:
        dataset = load_dataset(
            f"data/processed-{args.max_seq_length}-{args.model_name}")
    """
    Description to use in LoraConfig:
    q_proj: Query projection layer in the attention mechanism.
    k_proj: Key projection layer in the attention mechanism.
    v_proj: Value projection layer in the attention mechanism.
    o_proj: Output projection layer in the attention mechanism.
    gate_proj: Gate projection layer in the feed-forward network.
    up_proj: Up projection layer in the feed-forward network.
    down_proj: Down projection layer in the feed-forward network.
    embed_tokens: Embedding layer for token inputs.
    lm_head: Language modeling head for output generation.
    """
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    )
    model = get_peft_model(model, cfg)

    lora_training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=1,
        optim="adamw_torch",
        max_grad_norm=1.0,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=lora_training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    merged = Path(args.output_dir, "merged")
    model.merge_and_unload().save_pretrained(merged)
    print("finetunning done...", merged.resolve())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a model with LoRA on the Lora4Research dataset."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing the dataset files.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save the trained model.")
    parser.add_argument("--model_name", required=True,
                        default="meta-llama/Llama-2-7b-chat")
    # LoRa arguments
    parser.add_argument("--train_subset", type=float, default=1.0,
                        help="Fraction of the *train* split to use (0-1).")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="Rank for LoRA layers.")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Alpha for LoRA layers.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout rate for LoRA layers.")
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of steps to accumulate gradients before updating.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length for input samples.")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Total number of training steps to perform. Overrides num_train_epochs if set.")
    parser.add_argument("--ignore_ood_cases", action="store_true",
                        help="Ignore out-of-distribution cases during training.")
    parser.add_argument("--num_processes", type=int, default=4,
                        help="Number of processes to use for dataset preprocessing.")
    parser.add_argument("--batch_preprocess", type=int, default=4,
                        help="Batch size for dataset preprocessing.")
    args = parser.parse_args()
    main(args)
