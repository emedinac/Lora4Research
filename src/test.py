import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
import numpy as np
import loaders
import torch
from pathlib import Path
import argparse


def main(args):
    data_prec_path = f"data/processed-{args.max_new_tokens}-{args.model_name.replace('/', '-')}-{args.subset_fraction}"
    if Path(data_prec_path).exists() is False:
        AssertionError("Please insert a valid preprocessed data path")
    hf_model_enabled = args.model_path == "" or args.model_path is None or Path(args.model_name).exists() is False
    if hf_model_enabled:
        print("Please insert a valid model path, but we will use the model name to load the HF model")
        model_name=args.model_name
        tokenizer_name=args.model_name
    else:
        model_name=Path(args.model_path).joinpath("merged")
        tokenizer_name=args.model_path

    # Load model and tokenizer
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_name,
                                              trust_remote_code=True
                                              )
    model=AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.float16,
                                                 device_map="auto" if torch.cuda.is_available() else "cpu",
                                                 )
    # in case tokenizer size changed, but for standard models it should be the same is the same :)
    if hf_model_enabled:
        tokenizer.add_special_tokens({
            "additional_special_tokens": ["<problem>", "</problem>", "<approach>", "</approach>"]
        })
        tokenizer.pad_token=tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    # Load and preprocess dataset
    dataset=loaders.get_dataset(data_prec_path,
                                  args,
                                  tokenizer,
                                  args.max_new_tokens,
                                  "test")
    # Load metrics
    bleu=evaluate.load("sacrebleu")
    rouge=evaluate.load("rouge")
    bertscore=evaluate.load("bertscore")

    preds, gts=[], []
    for sample in tqdm.tqdm(dataset, desc="Processing"):
        prompt=f"<problem>\n{sample['input']}\n</problem>\n<approach>\n"
        input_ids=tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs=model.generate(**input_ids,
                                 temperature=0.7,
                                 top_p=0.9,
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=True,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.pad_token_id,
                                 )
        out=tokenizer.decode(outputs[0], skip_special_tokens=True)
        approach=out.split(
            "</approach>")[-1] if "</approach>" in out else out
        preds.append(approach.strip())
        gts.append(sample["output"].strip())

    bleu_res=bleu.compute(predictions=preds, references=[[gt] for gt in gts])
    rouge_res=rouge.compute(predictions=preds, references=gts)
    bert_res=bertscore.compute(predictions=preds, references=gts, lang="en")

    print("\n\n=== Evaluation Metrics ===")
    print(f"BLEU:       {bleu_res['score']:.2f}")
    print(f"ROUGE-L:    {rouge_res['rougeL']:.4f}")
    print(f"BERTScore (F1): {np.mean(bert_res['f1']):.4f}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(
        description="Computes the metrics: BLEU, ROUGE-L and BERTScore on the testset for a Fine-Tune model (LoRa)."
    )
    parser.add_argument("--model_name", required=True,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_path",
                        help="Path to the merged saved model checkpoint.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--subset_fraction", type=float, default=0.01,
                        help="Fraction of train split to use (0-1).")
    parser.add_argument("--ignore_ood_cases", action="store_true", default=False,
                        help="Ignore out-of-distribution cases.")
    parser.add_argument("--num_processes", type=int, default=0,
                        help="Number of processes for preprocessing.")
    parser.add_argument("--batch_preprocess", type=int, default=0,
                        help="Batch size for preprocessing.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args=parser.parse_args()
    main(args)
