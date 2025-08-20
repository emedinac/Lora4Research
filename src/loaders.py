from datasets import load_dataset
import tqdm
import re

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
