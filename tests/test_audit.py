"""Offline unit tests over small synthetic corpora/benchmarks, no network
and no dependency on the companion repo's real crawl.
"""
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "audit"))

from contamination_audit import (  # noqa: E402
    bloom_style_exact_paragraph_check,
    exact_13gram_audit,
    minhash_lsh_audit,
    tokenize,
)

CORPUS_DOC = {
    "doc_id": "test/doc1",
    "text": (
        "Adversaries may abuse elevation control mechanisms to bypass controls "
        "put in place to limit privileges on a system. A ransomware attacker "
        "may deploy malicious code after gaining unauthorized access to a host. "
        "This is a long enough paragraph of filler security text to exceed the "
        "minimum word count used for paragraph level shingling in this test suite."
    ),
}


def test_exact_13gram_match_is_flagged():
    con = duckdb.connect()
    item = {
        "item_id": "q1",
        "text": "Adversaries may abuse elevation control mechanisms to bypass controls put in place to limit",
    }
    flagged = exact_13gram_audit(con, [CORPUS_DOC], [item])
    assert "q1" in flagged
    assert flagged["q1"] == CORPUS_DOC["doc_id"]


def test_unrelated_text_is_not_flagged_by_either_detector():
    con = duckdb.connect()
    item = {
        "item_id": "q2",
        "text": "The orchestra rehearsed a sonata in the concert hall on a quiet Sunday afternoon downtown.",
    }
    exact_flagged = exact_13gram_audit(con, [CORPUS_DOC], [item])
    assert "q2" not in exact_flagged
    near_dup = minhash_lsh_audit([CORPUS_DOC], [item], already_flagged=set())
    assert "q2" not in near_dup


def test_near_duplicate_with_one_word_changed_is_caught_by_minhash():
    # A single-word edit in the middle of a long paragraph: high 5-gram
    # Jaccard (only the shingles touching the edit differ), easily above
    # the 0.8 threshold. This example is deliberately *not* checked against
    # the exact detector first -- a single scattered edit like this
    # typically still leaves a surviving 13-word run elsewhere in a real
    # document (that is what real-world near-duplicates with a handful of
    # edits look like), which is exactly why the exact check must run
    # first and the near-dup count must exclude anything it already
    # caught; see contamination_audit.main()'s "excluding exact matches".
    item = {
        "item_id": "q3",
        "text": CORPUS_DOC["text"].replace("filler security text", "sample security content"),
    }
    near_dup = minhash_lsh_audit([CORPUS_DOC], [item], already_flagged=set())
    assert "q3" in near_dup


def test_near_duplicate_already_exact_flagged_is_excluded():
    item = {"item_id": "q3", "text": CORPUS_DOC["text"]}
    near_dup = minhash_lsh_audit([CORPUS_DOC], [item], already_flagged={"q3"})
    assert "q3" not in near_dup


def test_bloom_style_check_never_flags_a_true_near_duplicate():
    # By construction: the Bloom-style check only flags exact paragraph
    # text, and the near-dup set (by definition) excludes exact matches.
    near_dup_ids = {"q3"}
    benchmark_items = [{
        "item_id": "q3",
        "text": "a paraphrase that is not byte-identical to any corpus paragraph",
    }]
    caught = bloom_style_exact_paragraph_check([CORPUS_DOC], benchmark_items, near_dup_ids)
    assert caught == []


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("CVE-2024-1234, a Critical Flaw!") == ["cve", "2024", "1234", "a", "critical", "flaw"]
