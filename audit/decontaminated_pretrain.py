"""Reruns the companion repo's (security-corpus-curation) token-budget-
matched continued-pretraining ablation with every corpus document this
audit implicated (exact 13-gram match or MinHash near-duplicate against
the 3 benchmarks) removed first, so the ablation's own measured
domain-accuracy gain can be reported both with and without contamination.
This is the one number this whole repo exists to report: how much of the
companion project's result was really curation signal, and how much was
the model having partly memorized the eval.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CURATION_REPO = ROOT.parent / "security-corpus-curation"
sys.path.insert(0, str(CURATION_REPO))
sys.path.insert(0, str(CURATION_REPO / "crawler"))
sys.path.insert(0, str(CURATION_REPO / "classifier"))

from pretrain.continued_pretrain import (  # noqa: E402
    BLOCK_SIZE, SEEDS, build_filters, eval_domain_accuracy, make_eval_set,
    tokenize_and_pack, train_one_run,
)
from transformers import AutoTokenizer  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def load_contaminated_doc_texts():
    audit = json.loads((DATA / "contamination_audit_results.json").read_text(encoding="utf-8"))
    contaminated_ids = set(audit["contaminated_corpus_doc_ids"])
    texts = set()
    for doc_id in contaminated_ids:
        path = CURATION_REPO / "data" / f"{doc_id}.json"
        if path.exists():
            texts.add(json.loads(path.read_text(encoding="utf-8"))["text"])
    return texts, contaminated_ids


def load_all_docs_excluding(excluded_texts):
    import glob
    docs = []
    for d in ["pages", "pages_attempt2"]:
        for p in glob.glob(str(CURATION_REPO / "data" / d / "*.json")):
            doc = json.loads(Path(p).read_text(encoding="utf-8"))
            if doc["text"] not in excluded_texts:
                docs.append(doc["text"])
    return docs


def run_ablation(docs, classifier_path, tag):
    import random
    rng = random.Random(42)
    rng.shuffle(docs)
    n_holdout = max(1, int(len(docs) * 0.15))
    holdout_docs = docs[:n_holdout]
    train_pool_docs = docs[n_holdout:]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    keyword_filtered, two_stage_filtered = build_filters(train_pool_docs, classifier_path)
    kw_blocks_probe = tokenize_and_pack(tokenizer, keyword_filtered, max_tokens=10 ** 9)
    two_blocks_probe = tokenize_and_pack(tokenizer, two_stage_filtered, max_tokens=10 ** 9)
    token_budget = min(kw_blocks_probe.numel(), two_blocks_probe.numel())
    kw_blocks = kw_blocks_probe[: token_budget // BLOCK_SIZE]
    two_blocks = two_blocks_probe[: token_budget // BLOCK_SIZE]

    eval_items = make_eval_set(holdout_docs, n_items=60)
    print(f"[{tag}] docs={len(docs)} keyword_filtered={len(keyword_filtered)} "
          f"two_stage_filtered={len(two_stage_filtered)} token_budget={token_budget} "
          f"eval_items={len(eval_items)}")

    deltas = []
    for seed in SEEDS:
        model_kw, tok = train_one_run(MODEL_NAME, kw_blocks, seed)
        acc_kw = eval_domain_accuracy(model_kw, tok, eval_items)
        del model_kw
        torch.cuda.empty_cache()

        model_two, tok = train_one_run(MODEL_NAME, two_blocks, seed)
        acc_two = eval_domain_accuracy(model_two, tok, eval_items)
        del model_two
        torch.cuda.empty_cache()

        delta = 100 * (acc_two - acc_kw)
        deltas.append(delta)
        print(f"[{tag}] seed={seed} acc_kw={acc_kw:.4f} acc_two={acc_two:.4f} delta={delta:.1f} points")

    return {"tag": tag, "n_docs": len(docs), "token_budget": token_budget,
            "deltas": deltas, "mean_delta_points": float(np.mean(deltas))}


def main():
    contaminated_texts, contaminated_ids = load_contaminated_doc_texts()
    print(f"removing {len(contaminated_texts)} contaminated documents "
          f"({len(contaminated_ids)} doc_ids implicated by the audit)")

    classifier_path = CURATION_REPO / "data" / "classifier_sklearn.pkl"

    docs_decontaminated = load_all_docs_excluding(contaminated_texts)
    result_decontaminated = run_ablation(docs_decontaminated, classifier_path, "decontaminated")

    pretrain_results_path = CURATION_REPO / "data" / "pretrain_ablation_results.json"
    original = json.loads(pretrain_results_path.read_text(encoding="utf-8")) if pretrain_results_path.exists() else None

    output = {
        "n_contaminated_docs_removed": len(contaminated_texts),
        "original_mean_delta_points": original["mean_delta_points"] if original else None,
        "decontaminated": result_decontaminated,
    }
    if original:
        output["gain_reduction_points"] = original["mean_delta_points"] - result_decontaminated["mean_delta_points"]
    (DATA / "decontaminated_pretrain_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nwrote", DATA / "decontaminated_pretrain_results.json")
    if original:
        print(f"original mean delta: {original['mean_delta_points']:.1f} points")
        print(f"decontaminated mean delta: {result_decontaminated['mean_delta_points']:.1f} points")


if __name__ == "__main__":
    main()
