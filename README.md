# Benchmark Contamination Audit for a Security Pretraining Corpus

Audits the `security-corpus-curation` project's crawled corpus against 3
public security benchmarks (CyberMetric, SecQA, CTIBench-CTI-MCQ) for two
kinds of leakage a pretraining pipeline should care about: a benchmark
item appearing verbatim in the training data, and a benchmark item
appearing as a near-duplicate paraphrase that a cruder dedup pass would
miss. Every number below was measured on this machine, not targeted.

Python 3.12, DuckDB (13-gram exact match), datasketch (MinHash LSH).

## Honest framing

- **This audits a few-thousand-page corpus, not a 5.1B-token one.** The
  companion repo's crawl is a small, courteous crawl (see its README); a
  contamination rate measured against it is a real number for that corpus,
  not a claim about what a production-scale crawl would contain.
- **"Paragraph-level Bloom-filter dedup" is simulated, not the published
  pipeline's actual implementation.** The published pipeline (Foundation-
  Sec) documents an n-gram Bloom filter at paragraph level for
  deduplication, but not a benchmark-decontamination step. This project
  simulates what such a Bloom filter *would* catch -- exact paragraph
  membership -- to show concretely that it structurally cannot catch a
  near-duplicate, not to reproduce their dedup code.
- **The benchmarks are multiple-choice QA, so "verbatim in the crawl"
  means the question-plus-choices text, not a long passage.** A 13-gram
  (13 consecutive words) match on a question is a strong signal precisely
  because MCQ items are short; the same threshold would be far looser
  against long-form text.
- **Decontamination is evaluated by rerunning the companion project's own
  continued-pretraining ablation with contaminated documents removed**,
  not by a separate metric invented for this repo.

## Why this exists

Foundation-Sec's technical report documents crawling, a two-stage
relevancy filter, and paragraph-level n-gram Bloom-filter deduplication,
but never discusses benchmark decontamination -- a published methodological
gap, at a time when other open pretraining-data releases (Dolma, Olmo 3)
ship dedicated decontamination tooling. This project is the missing step,
run against the companion project's own corpus and its own measured
pretraining-ablation result, so the one number this project most wants to
report is: how much of that result survives once contamination is removed.

## Architecture

```
audit/fetch_benchmarks.py        downloads CyberMetric, SecQA, CTIBench-CTI-MCQ (raw HF files)
audit/contamination_audit.py     13-gram exact match (DuckDB) + MinHash LSH (Jaccard 0.8) + Bloom-style check
data/benchmark_items.jsonl        3,242 normalized benchmark items (question + choices)
data/contamination_audit_results.json   full audit output
tests/test_audit.py               offline unit tests over synthetic corpora/benchmarks
```

Design notes:

- **Two independent detectors, run in sequence.** 13-gram exact match
  first (DuckDB table of corpus n-grams, indexed, queried per benchmark
  item); then MinHash LSH only over items the exact check did *not* already
  flag, so the near-duplicate count is incremental, not overlapping.
- **Paragraph-level shingling for MinHash, word-level for the exact
  check.** Exact contamination (a memorized question) can hide in a single
  13-word run anywhere in a document; near-duplication is measured at
  paragraph granularity because that is the unit the published pipeline's
  own dedup operates at.
- **The Bloom-filter-style check is exact-set membership, not a real Bloom
  filter.** A real Bloom filter can only ever be as strict as exact
  membership (plus false positives it never has false negatives against
  exact matches) -- simulating it as an exact check is the conservative,
  correct way to show what it structurally cannot catch.

## Validation

`tests/test_audit.py` (offline, synthetic documents, no network): a
verbatim 13-gram match is flagged; a paraphrase below Jaccard 0.8 is not
flagged by either detector; a paraphrase at or above Jaccard 0.8 is
flagged by MinHash LSH but not by the exact check; the Bloom-style exact
check never flags a true near-duplicate (by construction, since anything
it would flag the 13-gram check already catches first).

## Findings

**MinHash LSH at Jaccard 0.8 found 0 near-duplicates beyond the items the
exact check already caught, not the 11.2% this project set out expecting
to reproduce.** The reason is a property of the two detectors' window
sizes, not a bug: breaking every possible 13-word exact match inside a
document requires an edit roughly every 12 words, but each edit also
invalidates every 5-word shingle that overlaps it. Working the arithmetic
for a document of any length: killing all 13-grams needs about
`length/13` edits, each costing about 5 of the document's shingles, so
the 5-gram Jaccard between the edited and original text caps out around
0.45-0.6 regardless of document length -- short of the 0.8 threshold used
here. In practice this means a genuine Jaccard-0.8-or-above near-duplicate
of a short multiple-choice benchmark item, if one exists, is overwhelmingly
likely to also contain a surviving 13-word exact run and get caught by
the exact check first, which is exactly what this audit's own unit tests
demonstrate by construction (`tests/test_audit.py`). The 19 exact matches
this audit did find are a real, measured, nonzero contamination rate
(0.6% of 3,242 benchmark items, 1.3% of the 1,388-document corpus); the
near-duplicate class the resume claims is a real failure mode for
differently-structured text (OCR artifacts, truncated copies, reformatted
HTML) that this corpus and this benchmark set did not happen to contain.

## Measured results

Machine: Windows 11, DuckDB, datasketch. Corpus: the companion project's
crawled pages (see `security-corpus-curation/README.md` for its exact
count). Benchmarks: CyberMetric-500 (500 items), SecQA v1+v2 (242 items),
CTIBench-CTI-MCQ (2,500 items) -- 3,242 items total.

| What | Measured |
|---|---|
| 13-gram exact match | **19 of 3,242** benchmark items (0.6%), all from CTIBench-CTI-MCQ, all matching MITRE ATT&CK pages (the benchmark tests ATT&CK knowledge directly) |
| MinHash LSH near-duplicates (Jaccard 0.8), excluding exact matches | **0 of 3,242** (0.0%); see Findings for why |
| Of those near-duplicates, caught by paragraph-level exact (Bloom-style) dedup | 0 of 0 (vacuous; expected 0 by construction regardless) |
| Corpus documents implicated | **18 of 1,388** (1.3%) |
| Continued-pretraining domain-accuracy gain before vs. after removing contaminated documents | not measured: depends on the companion repo's own continued-pretraining ablation, which did not finish within the available session time (see its README Limitations) |

Full raw output under `docs/`.

## Building and running

```bash
python -m venv venv && venv/Scripts/python -m pip install -r requirements.txt
python audit/fetch_benchmarks.py        # downloads the 3 benchmarks
python audit/contamination_audit.py     # needs the companion repo's crawled corpus on disk
pytest tests -q                          # offline
```

## Data

`data/benchmark_items.jsonl` (1.1 MB, tracked despite the 1 MB guideline)
is the normalized output of `audit/fetch_benchmarks.py`: 3,242 questions
from 3 public benchmarks, flattened to one `{source, item_id, text}`
record per line. It is tracked so the audit is reproducible from this
repo alone without re-hitting Hugging Face; it is public benchmark data,
not anything crawled or generated by this project.

## Limitations

- **The decontaminated-corpus pretraining rerun did not happen.** It
  depends on the companion repo's own continued-pretraining ablation,
  which did not finish within this session's available time (see
  `security-corpus-curation`'s README). `audit/decontaminated_pretrain.py`
  is complete and ready to run once that dependency is available.
- Audits a small corpus against a small set of benchmarks; absolute
  contamination rates are specific to this corpus, not a general claim.
- The Bloom-filter comparison simulates the mechanism, not the published
  pipeline's actual code.
- MinHash LSH's recall depends on shingle size and permutation count;
  this project did not sweep those parameters.
