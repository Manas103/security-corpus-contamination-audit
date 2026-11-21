"""Downloads 3 public security benchmarks directly from their Hugging Face
dataset repos (raw JSON/CSV/TSV files, not the `datasets` library loader,
which chokes on CyberMetric's nested-struct JSON schema) and normalizes each
to a flat list of {source, item_id, text} records, where `text` is the
natural-language content (question plus answer choices) that a contaminated
pretraining document would need to overlap with.
"""
import csv
import io
import json
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent.parent / "data"
HEADERS = {"User-Agent": "security-corpus-contamination-audit-research/0.1"}


def get(url):
    resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    return resp


def cybermetric_items():
    items = []
    url = "https://huggingface.co/datasets/tihanyin/CyberMetric/resolve/main/CyberMetric-500-v1.json"
    data = get(url).json()
    for i, row in enumerate(data["questions"]):
        q = row.get("question", "")
        answers = row.get("answers", {})
        choice_text = " ".join(f"{k}) {v}" for k, v in answers.items())
        text = f"{q} {choice_text}".strip()
        if text:
            items.append({"source": "CyberMetric-500", "item_id": f"cybermetric-{i}", "text": text})
    return items


def secqa_items():
    items = []
    for config in ["secqa_v1", "secqa_v2"]:
        for split in ["dev", "val", "test"]:
            url = f"https://huggingface.co/datasets/zefang-liu/secqa/resolve/main/data/{config}_{split}.csv"
            try:
                resp = get(url)
            except Exception:
                continue
            reader = csv.DictReader(io.StringIO(resp.text))
            for i, row in enumerate(reader):
                q = row.get("Question", "")
                choices = " ".join(f"{k}) {row[k]}" for k in ["A", "B", "C", "D"] if row.get(k))
                text = f"{q} {choices}".strip()
                if text:
                    items.append({"source": f"SecQA-{config}", "item_id": f"{config}-{split}-{i}", "text": text})
    return items


def ctibench_items():
    items = []
    url = "https://huggingface.co/datasets/AI4Sec/cti-bench/resolve/main/cti-mcq.tsv"
    resp = get(url)
    reader = csv.DictReader(io.StringIO(resp.text), delimiter="\t")
    for i, row in enumerate(reader):
        q = row.get("Question", "") or row.get("Prompt", "")
        choice_cols = [k for k in row.keys() if k and k.lower().startswith("option")]
        choices = " ".join(f"{k}) {row[k]}" for k in choice_cols if row.get(k))
        text = f"{q} {choices}".strip()
        if text:
            items.append({"source": "CTIBench-CTI-MCQ", "item_id": f"ctibench-{i}", "text": text})
    return items


def main():
    all_items = []
    for fn in [cybermetric_items, secqa_items, ctibench_items]:
        items = fn()
        print(f"{fn.__name__}: {len(items)} items")
        all_items.extend(items)

    out_path = DATA / "benchmark_items.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_items)} total benchmark items to {out_path}")

    by_source = {}
    for item in all_items:
        by_source[item["source"]] = by_source.get(item["source"], 0) + 1
    print("by source:", by_source)


if __name__ == "__main__":
    main()
