"""Audits the security-corpus-curation project's crawled corpus for
benchmark contamination against 3 public security benchmarks (CyberMetric,
SecQA, CTIBench-CTI-MCQ), using two independent detectors:

  1. 13-gram exact match (DuckDB): a benchmark item is flagged if any
     contiguous 13-word shingle of its text appears verbatim in any corpus
     document. This is the method published decontamination tools (Dolma,
     the Olmo 3 decon pipeline) use for "this exact passage was memorized."

  2. MinHash LSH at Jaccard 0.8 (datasketch): a benchmark item is flagged
     as a *near*-duplicate if a paragraph-level 5-gram MinHash signature of
     its text is within Jaccard 0.8 of some corpus paragraph's signature,
     among items the 13-gram check did NOT already catch.

A third check demonstrates why #2 matters: a paragraph-level Bloom-filter
dedup pass (the published corpus's own dedup method [2]) is simulated as
exact-paragraph-fingerprint membership, which by construction cannot catch
a near-duplicate that differs by even one token -- so every near-duplicate
#2 finds is, by definition, invisible to #1-style exact dedup.
"""
import json
import re
from pathlib import Path

import duckdb
from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS_DIRS = [
    Path(__file__).resolve().parent.parent.parent / "security-corpus-curation" / "data" / "pages",
    Path(__file__).resolve().parent.parent.parent / "security-corpus-curation" / "data" / "pages_attempt2",
]

NGRAM_N = 13
SHINGLE_N = 5
MINHASH_PERM = 128
JACCARD_THRESHOLD = 0.8


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(tokens, n):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def load_corpus_docs():
    docs = []
    for d in CORPUS_DIRS:
        for p in sorted(d.glob("*.json")):
            doc = json.loads(p.read_text(encoding="utf-8"))
            docs.append({"doc_id": f"{d.name}/{p.stem}", "text": doc["text"]})
    return docs


def load_benchmark_items():
    items = []
    with open(DATA / "benchmark_items.jsonl", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def exact_13gram_audit(con, corpus_docs, benchmark_items):
    con.execute("CREATE TABLE corpus_ngrams (doc_id VARCHAR, ngram VARCHAR)")
    rows = []
    for doc in corpus_docs:
        tokens = tokenize(doc["text"])
        for ng in set(ngrams(tokens, NGRAM_N)):
            rows.append((doc["doc_id"], ng))
    if rows:
        con.executemany("INSERT INTO corpus_ngrams VALUES (?, ?)", rows)
    con.execute("CREATE INDEX idx_ngram ON corpus_ngrams(ngram)")

    flagged = {}
    for item in benchmark_items:
        tokens = tokenize(item["text"])
        item_ngrams = list(set(ngrams(tokens, NGRAM_N)))
        if not item_ngrams:
            continue
        placeholders = ",".join(["?"] * len(item_ngrams))
        hit = con.execute(
            f"SELECT doc_id FROM corpus_ngrams WHERE ngram IN ({placeholders}) LIMIT 1", item_ngrams
        ).fetchone()
        if hit:
            flagged[item["item_id"]] = hit[0]
    return flagged


def paragraphs_of(text, min_words=20):
    paras = re.split(r"\n{2,}|(?<=[.!?])\s{2,}", text)
    out = []
    for p in paras:
        toks = tokenize(p)
        if len(toks) >= min_words:
            out.append(p)
    if not out and len(tokenize(text)) >= min_words:
        out = [text]
    return out


def minhash_of(tokens):
    mh = MinHash(num_perm=MINHASH_PERM)
    for sh in set(ngrams(tokens, SHINGLE_N)):
        mh.update(sh.encode("utf-8"))
    return mh


def minhash_lsh_audit(corpus_docs, benchmark_items, already_flagged):
    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=MINHASH_PERM)
    para_lookup = {}
    pid = 0
    for doc in corpus_docs:
        for para in paragraphs_of(doc["text"]):
            tokens = tokenize(para)
            if len(tokens) < SHINGLE_N:
                continue
            key = f"p{pid}"
            pid += 1
            mh = minhash_of(tokens)
            lsh.insert(key, mh)
            para_lookup[key] = doc["doc_id"]

    near_dup = {}
    for item in benchmark_items:
        if item["item_id"] in already_flagged:
            continue
        tokens = tokenize(item["text"])
        if len(tokens) < SHINGLE_N:
            continue
        mh = minhash_of(tokens)
        matches = lsh.query(mh)
        if matches:
            near_dup[item["item_id"]] = [para_lookup[m] for m in matches[:3]]
    return near_dup


def bloom_style_exact_paragraph_check(corpus_docs, benchmark_items, near_dup_item_ids):
    """Exact-fingerprint (not fuzzy) paragraph membership, the mechanism a
    paragraph-level Bloom filter actually implements. Returns the subset of
    near_dup_item_ids whose *exact* paragraph text also appears in the
    corpus -- expected to be empty, since anything exactly matching would
    already have been caught by the 13-gram exact check (a strictly more
    sensitive exact detector)."""
    corpus_paragraphs = set()
    for doc in corpus_docs:
        for para in paragraphs_of(doc["text"]):
            corpus_paragraphs.add(para.strip())

    also_caught_by_bloom = []
    for item in benchmark_items:
        if item["item_id"] not in near_dup_item_ids:
            continue
        if item["text"].strip() in corpus_paragraphs:
            also_caught_by_bloom.append(item["item_id"])
    return also_caught_by_bloom


def main():
    corpus_docs = load_corpus_docs()
    benchmark_items = load_benchmark_items()
    print(f"corpus documents: {len(corpus_docs)}")
    print(f"benchmark items: {len(benchmark_items)} across "
          f"{len(set(i['source'] for i in benchmark_items))} benchmarks")

    con = duckdb.connect()
    exact_flagged = exact_13gram_audit(con, corpus_docs, benchmark_items)
    pct_exact = 100 * len(exact_flagged) / len(benchmark_items)
    print(f"\n13-gram exact match: {len(exact_flagged)}/{len(benchmark_items)} items ({pct_exact:.1f}%) "
          f"found verbatim in the crawl")

    near_dup = minhash_lsh_audit(corpus_docs, benchmark_items, set(exact_flagged))
    pct_near = 100 * len(near_dup) / len(benchmark_items)
    print(f"MinHash LSH (Jaccard {JACCARD_THRESHOLD}), excluding exact matches: "
          f"{len(near_dup)}/{len(benchmark_items)} items ({pct_near:.1f}%) found as near-duplicates")

    also_bloom = bloom_style_exact_paragraph_check(corpus_docs, benchmark_items, set(near_dup))
    print(f"of those near-duplicates, {len(also_bloom)}/{len(near_dup)} are also caught by a "
          f"paragraph-level exact (Bloom-filter-style) dedup check")

    contaminated_doc_ids = set(exact_flagged.values())
    for matches in near_dup.values():
        contaminated_doc_ids.update(matches)
    print(f"\ncorpus documents implicated by either detector: {len(contaminated_doc_ids)} "
          f"of {len(corpus_docs)} ({100 * len(contaminated_doc_ids) / len(corpus_docs):.1f}%)")

    result = {
        "n_corpus_docs": len(corpus_docs),
        "n_benchmark_items": len(benchmark_items),
        "benchmark_sources": sorted(set(i["source"] for i in benchmark_items)),
        "exact_13gram_matches": len(exact_flagged),
        "exact_13gram_pct": pct_exact,
        "near_duplicates_minhash": len(near_dup),
        "near_duplicates_pct": pct_near,
        "near_dup_also_caught_by_bloom_style_exact_dedup": len(also_bloom),
        "contaminated_corpus_doc_ids": sorted(contaminated_doc_ids),
        "contaminated_item_ids_exact": exact_flagged,
        "contaminated_item_ids_near_dup": near_dup,
    }
    (DATA / "contamination_audit_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nwrote", DATA / "contamination_audit_results.json")


if __name__ == "__main__":
    main()
