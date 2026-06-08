"""
report.py  -  render an experiment run as a self-contained HTML page (lever-C infra).
================================================================================
Pure rendering + diffing: given a CURRENT run dict and (optionally) the PREVIOUS run
of the same experiment, produce one HTML file that keeps us grounded in real failures.
The page LEADS with what went wrong — regressions vs last time, weak intents, and
concrete misclassified queries — then shows the R@1 matrix (with color-coded deltas),
per-intent breakdowns, and the none/threshold metrics.

Run dict schema (produced by bench.py; numbers are fractions in [0,1]):
  {
    "name": str, "timestamp": iso str, "seeds": [int],
    "eval_sets": [str], "intents": [str],
    "arms": { arm: {
      "label": str, "params": int,
      "sets": { set_name: {
        "recall1_mean": float, "recall1_std": float,
        "honest_overall_mean": float, "none_recall_mean": float,
        "real_sim_mean": float, "none_sim_mean": float,
        "per_intent": { intent: {"recall1_mean": float, "n": int} },
        "misclassified": [ {"text":.., "true":.., "pred":.., "sim":..}, ... ],
      }}}}
  }

The render functions are pure (no file IO, no training) so they are unit-tested; the
CLI at the bottom re-renders any saved run JSON to HTML (handy after tweaking styling,
with no re-training). bench.py also calls render_html directly at run time.
"""

import argparse
import html
import json
import os

REGRESSION_THR = 0.02  # a drop of ≥2 pp vs the previous run is flagged as a regression
WEAK_INTENT_THR = 0.70  # per-intent R@1 below this is flagged as a weak spot


def _set(run, arm, name):
    """Convenience: the per-set dict for (arm, eval-set), or {} if absent."""
    return run["arms"].get(arm, {}).get("sets", {}).get(name, {})


def regressions(current, previous, thr=REGRESSION_THR):
    """List of (arm, set, cur, prev, delta) where R@1 dropped ≥ thr vs previous."""
    out = []
    if not previous:
        return out
    for arm in current["arms"]:
        for name in current["eval_sets"]:
            cur = _set(current, arm, name).get("recall1_mean")
            prev = _set(previous, arm, name).get("recall1_mean")
            if cur is None or prev is None:
                continue
            if cur - prev <= -thr:
                out.append((arm, name, cur, prev, cur - prev))
    return out


def weak_intents(current, thr=WEAK_INTENT_THR):
    """List of (arm, set, intent, r1) where a per-intent R@1 is below thr."""
    out = []
    for arm in current["arms"]:
        for name in current["eval_sets"]:
            per = _set(current, arm, name).get("per_intent", {})
            for it, m in per.items():
                r1 = m.get("recall1_mean")
                if r1 is not None and r1 < thr:
                    out.append((arm, name, it, r1))
    return out


def _pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def _delta_html(cur, prev):
    """A colored Δ vs previous, e.g. '▲ +2.3' / '▼ -1.8' / '~ 0.0' / '(new)'."""
    if cur is None:
        return ""
    if prev is None:
        return '<span class="flat">(new)</span>'
    d = (cur - prev) * 100
    if d <= -REGRESSION_THR * 100:
        return f'<span class="down">▼ {d:+.1f}</span>'
    if d >= REGRESSION_THR * 100:
        return f'<span class="up">▲ {d:+.1f}</span>'
    return f'<span class="flat">~ {d:+.1f}</span>'


_CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a;max-width:1100px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px;border-bottom:2px solid #eee;padding-bottom:4px}
.meta{color:#666;font-size:13px;margin-bottom:8px}
table{border-collapse:collapse;margin:8px 0 4px;font-variant-numeric:tabular-nums}
th,td{border:1px solid #e2e2e2;padding:5px 9px;text-align:right} th{background:#f7f7f7;text-align:center}
td.lbl,th.lbl{text-align:left}
.up{color:#0a7d28;font-weight:600} .down{color:#c0271a;font-weight:600} .flat{color:#999}
.weak{background:#fff3cd} .ok{color:#0a7d28}
.callout{background:#fff;border:1px solid #e2e2e2;border-left:4px solid #c0271a;padding:10px 14px;margin:8px 0;border-radius:4px}
.callout.good{border-left-color:#0a7d28}
.q{text-align:left;font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
.small{color:#666;font-size:12px}
"""


def _summary_table(current, previous):
    sets = current["eval_sets"]
    head = "".join(f"<th>{html.escape(s)}</th>" for s in sets)
    rows = []
    for arm, a in current["arms"].items():
        cells = [
            f'<td class="lbl">{html.escape(a["label"])}<br><span class="small">{a["params"]:,} params</span></td>'
        ]
        for name in sets:
            cur = _set(current, arm, name).get("recall1_mean")
            std = _set(current, arm, name).get("recall1_std")
            prev = _set(previous, arm, name).get("recall1_mean") if previous else None
            std_s = f"±{std * 100:.1f}" if std is not None else ""
            cells.append(
                f"<td><b>{_pct(cur)}</b> <span class='small'>{std_s}</span><br>{_delta_html(cur, prev)}</td>"
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<table><tr><th class="lbl">arm</th>{head}</tr>{"".join(rows)}</table>'


def _per_intent_tables(current):
    out = []
    intents = current["intents"]
    head = "".join(f"<th>{html.escape(it)}</th>" for it in intents)
    for name in current["eval_sets"]:
        rows = []
        for arm, a in current["arms"].items():
            per = _set(current, arm, name).get("per_intent", {})
            cells = [f'<td class="lbl">{html.escape(a["label"])}</td>']
            for it in intents:
                r1 = per.get(it, {}).get("recall1_mean")
                cls = ' class="weak"' if (r1 is not None and r1 < WEAK_INTENT_THR) else ""
                cells.append(f"<td{cls}>{_pct(r1)}</td>")
            rows.append(f"<tr>{''.join(cells)}</tr>")
        out.append(
            f"<h3>{html.escape(name)} — per-intent R@1</h3>"
            f'<table><tr><th class="lbl">arm</th>{head}</tr>{"".join(rows)}</table>'
        )
    return "".join(out)


def _none_table(current):
    sets = current["eval_sets"]
    rows = []
    for arm, a in current["arms"].items():
        cells = [f'<td class="lbl">{html.escape(a["label"])}</td>']
        for name in sets:
            s = _set(current, arm, name)
            cells.append(
                f"<td>{_pct(s.get('honest_overall_mean'))} / {_pct(s.get('none_recall_mean'))}"
                f"<br><span class='small'>sim {s.get('real_sim_mean', float('nan')):.2f}/"
                f"{s.get('none_sim_mean', float('nan')):.2f}</span></td>"
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")
    head = "".join(f"<th>{html.escape(s)}</th>" for s in sets)
    return (
        f'<table><tr><th class="lbl">arm</th>{head}</tr>{"".join(rows)}</table>'
        '<p class="small">cells: honest overall / none-recall · sim = realSim/noneSim</p>'
    )


def _failures(current):
    """Concrete misclassified queries per arm per set — the 'stay grounded' section."""
    out = []
    for name in current["eval_sets"]:
        blocks = []
        for arm, a in current["arms"].items():
            mis = _set(current, arm, name).get("misclassified", [])
            if not mis:
                continue
            items = "".join(
                f'<tr><td class="q">{html.escape(m["text"])}</td>'
                f'<td class="lbl">{html.escape(m["true"])} → <b>{html.escape(m["pred"])}</b></td>'
                f"<td>{m['sim']:.2f}</td></tr>"
                for m in mis
            )
            blocks.append(
                f'<h4>{html.escape(a["label"])} <span class="small">({len(mis)} shown)</span></h4>'
                f'<table><tr><th class="q">query</th><th class="lbl">true → predicted</th><th>sim</th></tr>'
                f"{items}</table>"
            )
        if blocks:
            out.append(f"<h3>{html.escape(name)} — misclassified examples</h3>{''.join(blocks)}")
    return "".join(out) or "<p>No misclassified examples recorded.</p>"


def render_html(current, previous=None):
    """Render the full self-contained HTML page for one run (vs the previous run)."""
    regs = regressions(current, previous)
    weak = weak_intents(current)

    if not previous:
        reg_box = '<div class="callout good">First run for this experiment — no previous to compare against. This becomes the baseline.</div>'
    elif regs:
        items = "".join(
            f"<li><b>{html.escape(arm)}</b> on <b>{html.escape(s)}</b>: "
            f"{_pct(prev)} → {_pct(cur)} <span class='down'>({d * 100:+.1f} pp)</span></li>"
            for arm, s, cur, prev, d in regs
        )
        reg_box = f'<div class="callout"><b>⚠ {len(regs)} regression(s) vs previous run:</b><ul>{items}</ul></div>'
    else:
        reg_box = '<div class="callout good"><b>✓ No regressions vs previous run</b> (no R@1 drop ≥ 2 pp).</div>'

    if weak:
        witems = "".join(
            f"<li><b>{html.escape(arm)}</b> · {html.escape(s)} · <b>{html.escape(it)}</b>: {_pct(r1)}</li>"
            for arm, s, it, r1 in weak
        )
        weak_box = f'<div class="callout"><b>Weak intents (R@1 &lt; {WEAK_INTENT_THR * 100:.0f}%):</b><ul>{witems}</ul></div>'
    else:
        weak_box = f'<div class="callout good"><b>✓ No weak intents</b> (all per-intent R@1 ≥ {WEAK_INTENT_THR * 100:.0f}%).</div>'

    prev_note = f" · vs previous {html.escape(previous['timestamp'])}" if previous else ""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(current["name"])} — {html.escape(current["timestamp"])}</title>
<style>{_CSS}</style></head><body>
<h1>{html.escape(current["name"])}</h1>
<div class="meta">{html.escape(current["timestamp"])}{prev_note} · seeds {current["seeds"]} · eval sets: {", ".join(map(html.escape, current["eval_sets"]))}</div>

<h2>Failures first</h2>
{reg_box}
{weak_box}

<h2>R@1 (mean ± std), Δ vs previous</h2>
{_summary_table(current, previous)}

<h2>none / threshold metrics</h2>
{_none_table(current)}

<h2>Per-intent R@1</h2>
{_per_intent_tables(current)}

<h2>Misclassified examples</h2>
{_failures(current)}
</body></html>"""


def _previous_sibling(json_path):
    """The run JSON immediately before `json_path` in its folder (chronological =
    lexicographic on the timestamp filenames), or None — for the CLI's auto-diff."""
    d = os.path.dirname(os.path.abspath(json_path))
    base = os.path.basename(json_path)
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    if base not in files:
        return None
    i = files.index(base)
    return os.path.join(d, files[i - 1]) if i > 0 else None


def main():  # pragma: no cover - CLI re-render of a saved run
    ap = argparse.ArgumentParser(
        description="Re-render a saved bench run JSON to HTML (no re-training)"
    )
    ap.add_argument("run", help="path to results/<name>/<ts>.json")
    ap.add_argument(
        "--previous", help="explicit previous-run JSON (default: the run just before it)"
    )
    ap.add_argument("--out", help="output HTML path (default: alongside the JSON)")
    args = ap.parse_args()

    with open(args.run) as f:
        current = json.load(f)
    prev_path = args.previous or _previous_sibling(args.run)
    previous = None
    if prev_path and os.path.exists(prev_path):
        with open(prev_path) as f:
            previous = json.load(f)

    out = args.out or os.path.splitext(args.run)[0] + ".html"
    with open(out, "w") as f:
        f.write(render_html(current, previous))
    note = f"vs {os.path.basename(prev_path)}" if previous else "no previous"
    print(f"rendered → {out}  ({note})")


if __name__ == "__main__":
    main()
