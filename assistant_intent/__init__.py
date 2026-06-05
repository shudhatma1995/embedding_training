"""
assistant_intent
================
A personal voice-assistant intent classifier, built on the embedding model from
`customer_intent_search` (reused as-is, never edited, so the upstream tutorial
stays cleanly mergeable).

Pipeline (built one stage at a time):
    intents.py    - the label schema + annotation guideline  (SOURCE OF TRUTH)
    data_gen/     - seeds -> template expansion -> train/test split   [next]
                    (run: python -m assistant_intent.data_gen)
    train.py      - contrastive training, adapted to these intents    [later]
    evaluate.py   - Recall@1 / MRR on a held-out set                  [later]
"""
