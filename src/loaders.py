from datasets import load_dataset
import tqdm
import re
from pathlib import Path
from datasets import load_dataset, DatasetDict, load_from_disk
import torch

MIN_SENTENCES = 1
TEST_MIN_APPROACH_TOKENS = 5


def custom_sent_tokenize(text):
    pattern = re.compile(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+(?=[A-Z0-9“\(])')
    return pattern.split(text.strip())


def split_abstract(abstract: str, max_tokens: int):
    sents = custom_sent_tokenize(abstract)
    if not sents or len(sents) <= MIN_SENTENCES:
        sents = [s.strip() for s in abstract.split("\n") if s.strip()]
    if not sents or len(sents) <= MIN_SENTENCES:
        return None, None, "too_few_sentences"

    problem = sents[0].strip()
    approach = " ".join(sents[1:]).strip()

    if len(problem.split()) > max_tokens:
        return None, None, "problem_too_long"

    if len(approach.split()) < TEST_MIN_APPROACH_TOKENS:
        return None, None, "approach_too_short"

    return problem, approach, None


def get_ood_ids(data_to_download, db="default", max_tokens=512):
    dataset = load_dataset(data_to_download, db)
    db_skips = {}
    for key, rows in dataset.items():
        skips = {"too_few_sentences": 0,
                 "problem_too_long": 0,
                 "approach_too_short": 0,
                 "other": 0,
                 "ids": [],
                 }
        db_skips[key] = skips
        for i, row in enumerate(tqdm.tqdm(rows, desc="Filtering")):
            abstract = row["input"]
            problem, approach, reason = split_abstract(abstract, max_tokens)
            if not (problem and approach):
                db_skips[key][reason or "other"] += 1
                db_skips[key]["ids"].append(row["id"])

    print("\nSummary Considerations:")
    for db, skips in db_skips.items():
        for reason, count in skips.items():
            if reason == "ids":
                print(f"{db} - {reason}: {len(count):}")
                continue
            print(f"{db} - {reason}: {count:,}")
    return db_skips


def format_sample(ex):
    return f"<problem>\n{ex['input']}\n</problem>\n<approach>\n{ex['output']}\n</approach>"


def build_label_mask(input_ids, tokenizer):
    labels = input_ids.clone()
    aid = tokenizer.convert_tokens_to_ids("<approach>")
    idx = (input_ids == aid).nonzero(as_tuple=True)
    if len(idx[0]):
        labels[: idx[0][0] + 1] = -100
    return labels


def data_preprocess(examples, tokenizer, max_seq_length=512):
    if isinstance(examples["input"], list):
        texts = [format_sample({"input": inp, "output": out})
                 for inp, out in zip(examples["input"], examples["output"])]
    else:
        texts = [format_sample(examples)]
    enc = tokenizer(
        texts,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    # Remove batch dimension for single examples
    if not isinstance(examples["input"], list):
        enc["input_ids"] = enc["input_ids"].squeeze(0)
        enc["attention_mask"] = enc["attention_mask"].squeeze(0)
    enc["labels"] = torch.stack([
        build_label_mask(input_ids, tokenizer)
        for input_ids in enc["input_ids"].split(1)
    ]).squeeze()
    # enc["attention_mask"] = enc["attention_mask"].bool()
    return enc


def get_dataset(data_prec_path, args, tokenizer, max_seq_length=512):
    if not Path(data_prec_path).exists():
        dataset = load_dataset(args.data_dir, "default")
        if args.ignore_ood_cases:
            db_skips = get_ood_ids(args.data_dir,
                                   max_tokens=args.max_seq_length)
            all_dataset = {}
            for split, _ in db_skips.items():
                all_indices = set(range(len(dataset[split])))
                keep_indices = list(all_indices - set(db_skips))
                all_dataset[split] = dataset[split].select(keep_indices)
            dataset = DatasetDict(all_dataset)
        if args.subset_fraction < 1.0:  # reduce it due to computational constraints
            dataset = DatasetDict({
                split: dataset[split].shuffle(seed=args.seed).select(
                    list(
                        range(int(args.subset_fraction * len(dataset[split]))))
                )
                for split in dataset.keys()
            })
        if args.num_processes == 0 and args.batch_preprocess == 0:
            dataset = dataset.map(lambda x: data_preprocess(x, tokenizer, max_seq_length)
                                  )
        else:
            dataset = dataset.map(lambda x: data_preprocess(x, tokenizer, max_seq_length),
                                  batched=True,
                                  batch_size=args.batch_preprocess,
                                  num_proc=args.num_processes,
                                  )
        dataset.save_to_disk(data_prec_path)
    else:
        dataset = load_from_disk(data_prec_path)

    print("\nDatabase status:")
    print("Dataset splits:", list(dataset.keys()))
    print("Train size:", len(dataset["train"]))
    print("Validation size:", len(dataset["validation"]))
    sample = dataset["train"][0]
    print("Sample input_ids shape:",
          torch.Tensor(sample["input_ids"]).shape)
    print("Sample attention_mask shape:",
          torch.Tensor(sample["attention_mask"]).shape)
    print("Sample labels shape:",
          torch.Tensor(sample["labels"]).shape)
    return dataset


if __name__ == "__main__":
    get_ood_ids("scillm/scientific_papers-archive",
                "default",
                max_tokens=512)  # arxiv and pubmed
    """
    Summary Considerations:
    train - too_few_sentences: 19,330
    train - problem_too_long: 20
    train - approach_too_short: 15,485
    train - other: 0
    train - ids: 34835
    validation - too_few_sentences: 353
    validation - problem_too_long: 0
    validation - approach_too_short: 295
    validation - other: 0
    validation - ids: 648
    test - too_few_sentences: 247
    test - problem_too_long: 0
    test - approach_too_short: 329
    test - other: 0
    test - ids: 576
    """
