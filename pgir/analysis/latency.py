"""Per-request / per-item latency summaries (user request 2026-07-04).

- PGIR traces: serial_latency_s per item (concurrency-independent), plus
  per-arm gen_s / score_s / scorer_server_s breakdowns and retrieval times.
- Baselines: gen_call_log.jsonl -> per-request mean (elapsed / batch size).

Run: cd <package_root> && python -m pgir.analysis.latency <trace_or_run_dir>...
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def summarize_trace(path: Path) -> dict:
    items = [json.loads(l) for l in open(path)]
    serial = [it["diagnostics"].get("serial_latency_s", 0.0) for it in items]
    gen, score, server, retr = [], [], [], []
    for it in items:
        for a in it["pool"]:
            t = a.get("timing", {})
            if t:
                gen.append(t.get("gen_s", 0.0))
                score.append(t.get("score_s", 0.0))
                server.append(t.get("scorer_server_s", 0.0))
        retr.extend(it["diagnostics"].get("retr_times_s", []))
    out = {"file": str(path), "n_items": len(items),
           "serial_latency_s": {"mean": round(st.mean(serial), 3) if serial else None,
                                "p50": round(_pct(serial, 0.5), 3),
                                "p95": round(_pct(serial, 0.95), 3)}}
    for name, xs in (("gen_s", gen), ("score_s", score),
                     ("scorer_server_s", server), ("retr_s", retr)):
        if xs:
            out[name] = {"n": len(xs), "mean": round(st.mean(xs), 3),
                         "p50": round(_pct(xs, 0.5), 3), "p95": round(_pct(xs, 0.95), 3)}
    return out


def summarize_gen_log(path: Path) -> dict:
    calls = [json.loads(l) for l in open(path) if l.strip()]
    per_req = [c["elapsed_s"] / max(1, c["n_prompts"]) for c in calls]
    return {"file": str(path), "n_calls": len(calls),
            "total_requests": sum(c["n_prompts"] for c in calls),
            "per_request_mean_s": round(st.mean(per_req), 3) if per_req else None,
            "per_call_mean_s": round(st.mean(c["elapsed_s"] for c in calls), 3)
            if calls else None}


def main():
    for arg in sys.argv[1:]:
        p = Path(arg)
        targets = ([p] if p.is_file() else
                   sorted(p.rglob("pgir_trace_*.jsonl")) + sorted(p.rglob("gen_call_log.jsonl")))
        for t in targets:
            fn = summarize_gen_log if t.name == "gen_call_log.jsonl" else summarize_trace
            print(json.dumps(fn(t), indent=2))


if __name__ == "__main__":
    main()
