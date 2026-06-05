"""
assistant_intent
================
A personal voice-assistant intent classifier, built on the embedding model from
`customer_intent_search` (reused as-is, never edited, so the upstream tutorial
stays cleanly mergeable).

Pipeline:
    intents.py    - label schema + annotation guideline   (SOURCE OF TRUTH)
    seeds.py      - hand-written seed phrasings per intent
    data_gen/     - seeds -> template expansion -> train/test split
                    (run: python -m assistant_intent.data_gen)
    data.py       - json I/O, pair/batch shaping, the load_eval_data helper
    cis.py        - borrows the embedder + tokenizer from customer_intent_search
    shared.py     - encoder-agnostic classifier kernel (prototypes, proto_recall1)
    train.py      - contrastive training, adapted to these intents
    evaluate.py   - Recall@1 / MRR + honest none-threshold (val-tuned)
    experiments.py- multi-seed ablation harness
"""
