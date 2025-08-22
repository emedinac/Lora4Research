import loaders
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
from pathlib import Path
import argparse


def main(args):
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<problem>", "</problem>", "<approach>", "</approach>"]
    })
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    model.resize_token_embeddings(len(tokenizer))

    # Load and preprocess dataset
    data_prec_path = f"data/processed-{args.max_seq_length}-{args.model_name.replace('/', '-')}-{args.subset_fraction}"
    dataset = loaders.get_dataset(
        data_prec_path, args, tokenizer, args.max_seq_length)

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj",
                        "o_proj"],  # All attention modules
    )
    model = get_peft_model(model, lora_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=10,
        optim="adamw_torch",
        max_grad_norm=1.0,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        save_strategy="epoch",
        eval_strategy="epoch",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    merged_dir = Path(args.output_dir, "merged")
    model.merge_and_unload().save_pretrained(merged_dir)
    print(
        f"Fine-tuning completed. Merged model saved to {merged_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune TinyLlama with LoRA on a dataset."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Dataset name or path.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save the trained model.")
    parser.add_argument("--model_name", required=True,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # Train parameters
    parser.add_argument("--lora_r", type=int, default=8,
                        help="Rank for LoRA layers.")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Alpha for LoRA layers.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout rate for LoRA layers.")
    parser.add_argument("--subset_fraction", type=float, default=0.01,
                        help="Fraction of train split to use (0-1).")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float,
                        default=5e-5, help="Learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length.")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Total training steps.")
    parser.add_argument("--ignore_ood_cases", action="store_true", default=False,
                        help="Ignore out-of-distribution cases.")
    parser.add_argument("--num_processes", type=int, default=0,
                        help="Number of processes for preprocessing.")
    parser.add_argument("--batch_preprocess", type=int,
                        default=16, help="Batch size for preprocessing.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()
    main(args)
