"""
bench.py  -  reusable multi-seed experiment harness + JSON store + HTML report.
================================================================================
One place to define and run encoder experiments going forward (supersedes the ad-hoc
/tmp A/B scripts). An experiment = a set of ARMS (encoder builders) evaluated across
SEEDS on the standard eval sets (in-dist / wild / MASSIVE) via the shared evaluate_model.
Results are persisted to results/<name>/<ts>.json and rendered to an HTML page that
compares this run to the previous one (report.py).

Add a new experiment arm = add one builder to ARMS (e.g. a fine-tuned-fastText variant).

Run:  python bench.py                                  # word + random + fastText, 5 seeds
      python bench.py --arms word fasttext --seeds 0 1 2
      python bench.py --name my-idea --arms fasttext
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import random
import shutil
from collections import defaultdict
from datetime import UTC, datetime

import encoder as enc
import numpy as np
import report as report_mod
import torch
from data import DATA_DIR, load_by_intent
from intents import NONE_ID, REAL_INTENTS
from shared import make_encoder

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(HERE, "..", "results"))
EVAL_FILES = {"in-dist": "test.json", "wild": "test_wild.json", "MASSIVE": "test_massive.json"}

# canonical training recipe (config B) — held fixed so arms are comparable
EPOCHS, PAIRS, TEMP, WARMUP, NONE_K, SEQ = 25, 30, 0.05, 30, 8, 64
MAX_MISCLASSIFIED = 12  # concrete failure examples kept per arm/set (from the first seed)

_MODS = {}


def _mod(name, filename):
    """Lazily load a sibling module by path (train.py / evaluate.py collide with the
    customer_intent_search namesakes, so we can't `import train`). Lazy so importing
    bench for its pure helpers/tests doesn't exec the training code."""
    if name not in _MODS:
        spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _MODS[name] = m
    return _MODS[name]


# ── arm builders: (seed, ctx) -> (encode, label, n_params) ───────────────────
def _train_frozen(matrix, seed, ctx):
    """Train a FastTextEncoder over a FROZEN embedding `matrix` with the config-B recipe."""
    tr = _mod("ai_train", "train.py")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = enc.FastTextEncoder(matrix, max_seq_len=SEQ).to(ctx["device"])
    tok = enc.make_ft_tokenizer(ctx["ft_vocab"], max_seq_len=SEQ)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01
    )
    sched = tr.make_scheduler(opt, WARMUP, PAIRS * EPOCHS)
    for _ in range(EPOCHS):
        tr.run_epoch(
            model,
            tok,
            ctx["train_groups"],
            ctx["none_pool"],
            opt,
            sched,
            PAIRS,
            TEMP,
            NONE_K,
            rng,
            ctx["device"],
        )
    return make_encoder(model, tok, ctx["device"]), model.n_parameters()


def build_word(seed, ctx):
    tr = _mod("ai_train", "train.py")
    args = tr.build_parser().parse_args([])
    args.seed, args.tokenizer, args.max_seq_len = seed, "word", SEQ
    args.none_in_vocab, args.none_neg_k, args.epochs = True, NONE_K, EPOCHS
    args.output_dir = ctx["scratch"]
    with contextlib.redirect_stdout(io.StringIO()):
        model, tok = tr.train(args)
    return make_encoder(model, tok, ctx["device"]), "word (from-scratch)", model.n_parameters()


def build_fasttext(seed, ctx):
    encode, p = _train_frozen(ctx["ft_matrix"], seed, ctx)
    return encode, "fastText-frozen", p


def build_random(seed, ctx):
    encode, p = _train_frozen(enc.random_matrix_like(ctx["ft_matrix"], seed), seed, ctx)
    return encode, "random-frozen (control)", p


ARMS = {"word": build_word, "random": build_random, "fasttext": build_fasttext}
_NEEDS_FT = {"random", "fasttext"}


# ── run ──────────────────────────────────────────────────────────────────────
def load_eval_sets(real_intents):
    """{set_name: (test_groups, none_texts)} for whichever standard files exist."""
    sets = {}
    for name, fn in EVAL_FILES.items():
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path):
            continue
        sets[name] = (load_by_intent(path, real_intents), load_by_intent(path, [NONE_ID])[NONE_ID])
    return sets


def _ms(xs):
    a = np.array(xs, dtype=float)
    return float(a.mean()), float(a.std())


def run_experiment(name, arm_names, seeds, real_intents=None, device="cpu"):
    ev = _mod("ai_evaluate", "evaluate.py")
    real_intents = real_intents or list(REAL_INTENTS)
    train_groups = load_by_intent(os.path.join(DATA_DIR, "train.json"), real_intents)
    none_pool = load_by_intent(os.path.join(DATA_DIR, "train.json"), [NONE_ID])[NONE_ID]
    eval_sets = load_eval_sets(real_intents)

    ctx = {
        "device": device,
        "train_groups": train_groups,
        "none_pool": none_pool,
        "scratch": os.path.join(HERE, "models", "_bench_tmp"),
    }
    arm_names = list(arm_names)
    if any(a in _NEEDS_FT for a in arm_names):
        cache = os.path.join(enc.FT_CACHE, "matrix.npy")
        if os.path.exists(cache):
            ctx["ft_matrix"], ctx["ft_vocab"] = enc.load_ft_assets()
        else:
            dropped = [a for a in arm_names if a in _NEEDS_FT]
            print(
                f"  ! fastText cache missing ({enc.FT_CACHE}); run build_fasttext.py. Skipping arms: {dropped}"
            )
            arm_names = [a for a in arm_names if a not in _NEEDS_FT]

    raw = {a: {s: defaultdict(list) for s in eval_sets} for a in arm_names}
    per_n = {a: {s: {} for s in eval_sets} for a in arm_names}
    mis = {a: {} for a in arm_names}
    labels, params = {}, {}

    for si, seed in enumerate(seeds):
        for arm in arm_names:
            encode, label, p = ARMS[arm](seed, ctx)
            labels[arm], params[arm] = label, p
            for sname, (tg, none_t) in eval_sets.items():
                res = ev.evaluate_model(encode, train_groups, tg, none_t)
                r = raw[arm][sname]
                r["recall1"].append(res["micro"]["recall@1"])
                r["honest"].append(res["honest"]["overall"])
                r["none_recall"].append(res["honest"]["none_recall"])
                r["real_sim"].append(res["real_sim"])
                r["none_sim"].append(res["none_sim"])
                for it, m in res["per_intent"].items():
                    r[f"pi:{it}"].append(m["recall@1"])
                    per_n[arm][sname][it] = m["n"]
                if si == 0:
                    mis[arm][sname] = [
                        {"text": t, "true": tru, "pred": pr, "sim": s}
                        for (t, tru, pr, s) in res["misclassified"][:MAX_MISCLASSIFIED]
                    ]
            print(
                f"  seed {seed} {arm:<9} | "
                + "  ".join(f"{s} {raw[arm][s]['recall1'][-1] * 100:5.1f}%" for s in eval_sets),
                flush=True,
            )

    shutil.rmtree(ctx["scratch"], ignore_errors=True)

    arms_out = {}
    for arm in arm_names:
        sets_out = {}
        for sname in eval_sets:
            r = raw[arm][sname]
            r1m, r1s = _ms(r["recall1"])
            per_intent = {
                it: {"recall1_mean": float(np.mean(r[f"pi:{it}"])), "n": per_n[arm][sname][it]}
                for it in per_n[arm][sname]
            }
            sets_out[sname] = {
                "recall1_mean": r1m,
                "recall1_std": r1s,
                "honest_overall_mean": float(np.mean(r["honest"])),
                "none_recall_mean": float(np.mean(r["none_recall"])),
                "real_sim_mean": float(np.mean(r["real_sim"])),
                "none_sim_mean": float(np.mean(r["none_sim"])),
                "per_intent": per_intent,
                "misclassified": mis[arm][sname],
            }
        arms_out[arm] = {"label": labels[arm], "params": params[arm], "sets": sets_out}

    return {
        "name": name,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "seeds": list(seeds),
        "eval_sets": list(eval_sets.keys()),
        "intents": list(train_groups.keys()),
        "arms": arms_out,
    }


# ── persistence ────────────────────────────────────────────────────────────────
def _stamp(ts):
    return ts.replace(":", "").replace("-", "").replace("+", "_")


def save_results(results, root=RESULTS_DIR):
    d = os.path.join(root, results["name"])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{_stamp(results['timestamp'])}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def load_previous(name, root=RESULTS_DIR):
    """The most recent prior run JSON for this experiment (chronological = lexicographic
    on the timestamp filenames), or None. Call BEFORE save_results so it excludes current."""
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    if not files:
        return None
    with open(os.path.join(d, files[-1])) as f:
        return json.load(f)


def write_report(results, previous, root=RESULTS_DIR):
    d = os.path.join(root, results["name"])
    os.makedirs(d, exist_ok=True)
    html_str = report_mod.render_html(results, previous)
    path = os.path.join(d, f"{_stamp(results['timestamp'])}.html")
    for p in (path, os.path.join(d, "latest.html")):
        with open(p, "w") as f:
            f.write(html_str)
    return path


def main():
    ap = argparse.ArgumentParser(description="Run a multi-seed encoder experiment + HTML report")
    ap.add_argument("--name", default="encoder-arms")
    ap.add_argument("--arms", nargs="+", default=["word", "random", "fasttext"], choices=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"experiment '{args.name}' | arms={args.arms} | seeds={args.seeds} | device={device}")
    previous = load_previous(args.name)  # before we save the current run
    results = run_experiment(args.name, args.arms, args.seeds, device=device)
    jpath = save_results(results)
    hpath = write_report(results, previous)
    print(f"\n  results JSON → {jpath}")
    print(f"  HTML report  → {hpath}")
    if previous:
        regs = report_mod.regressions(results, previous)
        print(f"  vs previous ({previous['timestamp']}): {len(regs)} regression(s)")
    else:
        print("  (first run — baseline; future runs will diff against it)")


if __name__ == "__main__":
    main()
