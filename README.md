# Provenance Lab

A small, reproducible Wikipedia-vs-generated-text experiment. The live browser demo uses a fine-tuned BERT-Tiny model, runs locally with WebAssembly, and publishes its false positives rather than offering an authorship verdict.

## Actual results, seed37

| Slice | Accuracy | AUROC | Human false positives | Generated false negatives |
|---|---:|---:|---:|---:|
| Deployed INT8 BERT, held-out GPT-4o |82.0%|0.904|14.4%|21.7%|
| Deployed INT8 BERT, held-out Mistral generator |73.6%|0.838|14.4%|39.0%|
| Full-precision BERT, held-out GPT-4o |82.0%|0.903|14.1%|22.0%|
| TF-IDF/logistic baseline, held-out GPT-4o |85.0%|0.938|15.9%|14.1%|

The lexical baseline is stronger on this small split. This is reported, not hidden. One seed, one task/subset and short token windows are not evidence of general detector quality. These are this experiment's results, not a reproduction of the benchmark paper's full system or its reported aggregate numbers.

## Data provenance and split

TSM-Bench by Gerrit Quaremba, Denny Vrandecic, Elizabeth Black and Elena Simperl, ICLR2026. Dataset revision `3f9cdae5a4a4715f14133872fd5db16ebfead474`, English paragraph-writing files `paras/paras_en_first_gpt4o.jsonl` and `paras/paras_en_first_mistral.jsonl`. Wikipedia `trgt` vs generator `mgt` labels are inherited from the dataset, not independently authenticated authorship claims. Source says Wikipedia collected before ChatGPT's2022release; it does not imply generated examples predate their models.

Page-title SHA256 with prefix `provenance37:` assigns70/15/15 partitions. Paired human/generated text for a page stays together. Texts below40words are excluded, whitespace normalized, identical texts appearing across partitions removed, exact texts deduplicated within each partition. Train3604examples/1569pages, validation749/320pages, test779/335pages. Manifest shards include rowid/page/label/hash, not raw dataset text. This is a custom grouped split, not the original paper's full evaluation.

Train only on GPT-4o pairs. Mistral alternatives are held out by generator and use only test-page rowids, with the same human comparison set. Same dataset/domain/task, so this is generator-shift, not a broad OOD test. No test threshold/epoch tuning. Fullprecision epoch3 selected by validationAUROC, threshold0.68 by validationaccuracy. Baseline threshold independently selected on validation, not reused from BERT.

## Model and reproduction

Google BERT miniature `google/bert_uncased_L-2_H-128_A-2`, exact revision `30b0a37ccaaa32f332884b96992754e246e48c5f`, pretrained base4.43Mparameters, Apache2.0. Fine-tuned all parameters with two-class head,3epochs,batch32,AdamW5e-5,weightdecay.01,gradientclip1,seed37,CPU2threads. 128tokens includingCLS/SEP; input is truncated to126WordPiece contenttokens. No model API or paid compute.

`research/train.py` records train/validation/test metrics, exact revisions/splits and export. `research/evaluate_quantized.py` evaluates the deployed INT8ONNX model one example at a time, matching browser batchsize. Dynamic quantization is batch-sensitive, so batch32 metrics are not substituted for deployed batch1 metrics. `docs/metrics.json`, `docs/deployed_metrics.json`, `docs/quantization.json` retain exact values. INT8 modelSHA256: `4b821a8a7ca53a447446b85a7e864cfd665cb5dc3d2fb59e3f1a69424cbba760`.

Training expects the two revision-pinned JSONL files under `/tmp/textdata/`; download sources are named above. The model, vocabulary and JS modules are hosted with the site. The 11.9 MB WASM engine downloads from a pinned jsDelivr version and must pass a hard-coded SHA256 integrity check before execution. jsDelivr receives ordinary download metadata, never entered text. No text uploads. A blocked CDN fails explicitly. WordPiece tokenization was checked against HF tokenizer fixtures and300real dataset paragraphs. ONNX Runtime Web1.23.2, MIT, bundled license in`docs/vendor`. No external JavaScript CDN module; the integrity-checked WASM binary is the only external runtime asset.

## Limits and safe use

This is not proof of authorship, originality, plagiarism or intent. Scores are uncalibrated dataset-class scores, not real-world author probabilities. Falsepositives on human text are substantial. Do not use for grading, accusations, hiring, enforcement or punitive decisions. Non-native English, editing, quotes, mixedauthorship, newgenerators, otherdomains and shorttext can invalidate patterns. Samepage heldout does not prevent topic overlap/pretraining leakage, and Wikipedia may contain historic automation or editorial artifacts. Training labels and dataset curation were not independently audited.

Runtime bounded to10000characters/40+words/126contenttokens, worker90stimeout, clear terminates worker, model download failure is explicit, no fake score. Approx17MB firstload, including the external integrity-checked runtime. Browser/device may lack requiredWASMfeatures. No account/tracker/textstorage, model runs in worker.

## Licensing and attribution

Dataset card: https://huggingface.co/datasets/GerritQ/tsm_bench , CC BY-NC-SA4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/ . Data-derived trained weights and manifest provided under that noncommercial/sharealike license, retaining source attribution. No claim this grant supersedes upstream Wikipedia/source rights. Base model Apache2.0: https://huggingface.co/google/bert_uncased_L-2_H-128_A-2 . ONNX Runtime MIT: https://github.com/microsoft/onnxruntime .

Paper: https://openreview.net/forum?id=zimuL7ZmIi . Prior WETBench: https://aclanthology.org/2025.wikinlp-1.6/ . These papers warn that realistic task-specific detection generalizes poorly. GPT-wiki-intro was considered but not selected because its card's generic`cc` label is less specific than this dataset's explicitlicense, and it uses an older singleCuriegenerator.


## Re-run

Use Python 3.11 with `research/requirements-lock.txt`. Run `python research/download_data.py`, then `python research/train.py`, then `python research/evaluate_quantized.py`. Training writes the same grouped split as a combined manifest and the shard files used by tests. A new run writes its exported model to the shipped `2-model-int8.onnx` path. CPU results may vary with library/platform changes. `node tests/test_manifest.mjs` checks split isolation; `node tests/test_tokenizer.mjs` checks WordPiece fixtures.

One unreferenced two-byte model upload is retained in repository history/tree rather than deleted. It is not a runnable model and the worker only loads `2-model-int8.onnx`; that file is SHA256 verified.
