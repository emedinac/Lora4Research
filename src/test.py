import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
import numpy as np
import loaders
import torch
from pathlib import Path
import argparse


def main(args):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True
                                              )
    model = AutoModelForCausalLM.from_pretrained(Path(args.model_path).joinpath("merged"),
                                                 torch_dtype=torch.float16,
                                                 device_map="auto" if torch.cuda.is_available() else "cpu",
                                                 )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    # Load metrics
    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    # Load and preprocess dataset
    data_prec_path = f"data/processed-{args.max_new_tokens}-{args.model_name.replace('/', '-')}-{args.subset_fraction}"
    dataset = loaders.get_dataset(data_prec_path,
                                  args,
                                  tokenizer,
                                  args.max_new_tokens,
                                  "test")
    preds, gts = [], []
    for sample in tqdm.tqdm(dataset, desc="Processing"):
        prompt = f"<problem>\n{sample['problem']}\n</problem>\n<approach>\n"
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**input_ids,
                                 temperature=0.7,
                                 top_p=0.9,
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=True,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.pad_token_id,
                                 )
        out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        approach = out.split(
            "</approach>")[-1] if "</approach>" in out else out
        preds.append(approach.strip())
        gts.append(sample["approach"].strip())

    bleu_res = bleu.compute(predictions=preds, references=[[gt] for gt in gts])
    rouge_res = rouge.compute(predictions=preds, references=gts)
    bert_res = bertscore.compute(predictions=preds, references=gts, lang="en")

    print("\n\n=== Evaluation Metrics ===")
    print(f"BLEU:       {bleu_res['score']:.2f}")
    print(f"ROUGE-L:    {rouge_res['rougeL']:.4f}")
    print(f"BERTScore (F1): {np.mean(bert_res['f1']):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes the metrics: BLEU, ROUGE-L and BERTScore on the testset for a Fine-Tune model (LoRa)."
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to the merged saved model checkpoint.")
    parser.add_argument("--model_name", required=True,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()
    main(args)
