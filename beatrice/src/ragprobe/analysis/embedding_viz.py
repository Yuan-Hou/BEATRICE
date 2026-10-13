from __future__ import annotations

import concurrent.futures as futures
import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from ragprobe.llm.openai_client import OpenAICompatibleLLM
from ragprobe.utils.io import write_jsonl


@dataclass(frozen=True)
class TextItem:
    item_id: str
    dataset: str
    split: str
    question: str
    context: str
    source_path: str
    source_record_index: int
    answers: list[Any]
    context_count: int


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update((part or "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:20]


def _iter_jsonl(path: str | Path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _extract_contexts(example: dict[str, Any]) -> list[str]:
    contexts = example.get("contexts") or []
    out = []
    for ctx in contexts:
        if isinstance(ctx, str):
            text = ctx.strip()
            title = ""
        else:
            title = str(ctx.get("title") or "").strip()
            text = str(ctx.get("text") or ctx.get("paragraph_text") or "").strip()
        if not text:
            continue
        out.append(f"{title}\n{text}".strip() if title else text)
    return out


def _load_generation_records(config: dict[str, Any]) -> list[TextItem]:
    items: list[TextItem] = []
    data_cfg = config.get("input", {})
    split_filter = set(data_cfg.get("split_filter") or [])
    max_context_chars = data_cfg.get("max_context_chars")
    max_context_chars = int(max_context_chars) if max_context_chars is not None else None
    per_dataset_limit = data_cfg.get("max_samples_per_dataset")
    per_dataset_limit = int(per_dataset_limit) if per_dataset_limit is not None else None
    seed = int(config.get("seed", 13))
    rng = np.random.default_rng(seed)

    by_dataset: dict[str, list[TextItem]] = {}
    for spec in data_cfg.get("datasets", []):
        path = str(spec["path"])
        default_dataset = spec.get("dataset")
        for source_record_index, record in enumerate(_iter_jsonl(path)):
            example = record.get("example", record)
            split = str(example.get("split") or record.get("split") or "")
            if split_filter and split not in split_filter:
                continue
            dataset = str(example.get("dataset") or default_dataset or Path(path).stem)
            question = str(example.get("question") or record.get("question") or "").strip()
            if not question:
                continue
            contexts = _extract_contexts(example)
            if not contexts:
                continue
            context = "\n\n".join(contexts).strip()
            answers = example.get("answers") or record.get("answers") or []
            if max_context_chars is not None and max_context_chars > 0 and len(context) > max_context_chars:
                context = context[:max_context_chars].rstrip()
            example_id = str(example.get("id") or record.get("example_id") or _stable_id(dataset, split, question))
            item = TextItem(
                item_id=example_id,
                dataset=dataset,
                split=split,
                question=question,
                context=context,
                source_path=path,
                source_record_index=source_record_index,
                answers=answers if isinstance(answers, list) else [answers],
                context_count=len(contexts),
            )
            by_dataset.setdefault(dataset, []).append(item)

    for dataset, rows in sorted(by_dataset.items()):
        if per_dataset_limit is not None and len(rows) > per_dataset_limit:
            idx = rng.choice(len(rows), size=per_dataset_limit, replace=False)
            rows = [rows[int(i)] for i in idx]
        items.extend(rows)
    return items


def _question_rewrite_prompt(question: str) -> str:
    return f"""Rewrite the question into a normalized standalone information-seeking query.

Rules:
- Preserve the exact meaning and all entities.
- Do not answer the question.
- Do not add facts not present in the question.
- Make implicit references explicit when possible.
- Output exactly one line in this format:
Normalized Question: <rewritten question>

Question: {question}"""


def _context_rewrite_prompt(question: str, context: str) -> str:
    return f"""Rewrite the context into a compact evidence-style passage aligned to the question.

Rules:
- Use only information from the context.
- Preserve named entities, dates, numbers, and relations that may help answer the question.
- Remove irrelevant wording and formatting noise.
- Do not answer the question directly unless the context states the answer.
- Output exactly this format:
Normalized Context: <compact rewritten context>

Question: {question}

Context:
{context}"""


def _strip_prefix(text: str, prefixes: tuple[str, ...]) -> str:
    text = (text or "").strip()
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix):].strip()
    return text


class TextCache:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=60.0)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute("pragma journal_mode=WAL")
            self.conn.execute("create table if not exists text_cache (key text primary key, value text not null)")
            self.conn.commit()

    def get(self, key: str) -> str | None:
        with self.lock:
            row = self.conn.execute("select value from text_cache where key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set(self, key: str, value: str) -> None:
        with self.lock:
            self.conn.execute("insert or replace into text_cache(key, value) values (?, ?)", (key, value))
            self.conn.commit()


def _rewrite_items(items: list[TextItem], config: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    rewrite_cfg = config.get("rewrite", {})
    enabled = bool(rewrite_cfg.get("enabled", True))
    cache_path = output_dir / "rewrite_cache.sqlite3"
    cache = TextCache(cache_path)
    save_every = int(rewrite_cfg.get("save_every", 20))
    workers = int(rewrite_cfg.get("concurrency", 32))

    print(f"rewrite enabled={enabled} concurrency={workers} cache={cache_path}", flush=True)

    llm_cfg = rewrite_cfg.get("model", {})
    llm = OpenAICompatibleLLM(
        model_name=llm_cfg.get("name", "Qwen/Qwen3.5-27B"),
        api_key=llm_cfg.get("api_key", "EMPTY"),
        base_url=llm_cfg.get("base_url", "http://localhost:2334/v1"),
        enable_thinking=bool(llm_cfg.get("enable_thinking", False)),
        max_retries=int(llm_cfg.get("max_retries", 3)),
        retry_sleep=float(llm_cfg.get("retry_sleep", 1.0)),
        timeout=float(llm_cfg.get("timeout", 600)),
        max_tokens=llm_cfg.get("max_tokens"),
        max_total_tokens=llm_cfg.get("max_total_tokens"),
        context_token_buffer=llm_cfg.get("context_token_buffer", 256),
        temperature=float(llm_cfg.get("temperature", 0.0)),
    )

    def one(item: TextItem) -> dict[str, Any]:
        base = {
            "example_id": item.item_id,
            "dataset": item.dataset,
            "split": item.split,
            "question": item.question,
            "context": item.context,
            "source_path": item.source_path,
            "source_record_index": item.source_record_index,
            "answers": item.answers,
            "context_count": item.context_count,
        }
        if not enabled:
            return {**base, "rewritten_question": item.question, "rewritten_context": item.context, "rewrite_status": "disabled"}
        q_key = _stable_id("question", item.question)
        c_key = _stable_id("context", item.question, item.context)
        try:
            q_value = cache.get(q_key)
            if q_value is None:
                q_value = llm.generate(_question_rewrite_prompt(item.question))
                cache.set(q_key, q_value)
            c_value = cache.get(c_key)
            if c_value is None:
                c_value = llm.generate(_context_rewrite_prompt(item.question, item.context))
                cache.set(c_key, c_value)
            return {
                **base,
                "rewritten_question": _strip_prefix(q_value, ("Normalized Question:",)),
                "rewritten_context": _strip_prefix(c_value, ("Normalized Context:",)),
                "rewrite_status": "success",
            }
        except Exception as exc:
            return {**base, "rewritten_question": item.question, "rewritten_context": item.context, "rewrite_status": "failed", "error": str(exc)}

    results: list[dict[str, Any]] = []
    if workers <= 1:
        for i, item in enumerate(items, 1):
            results.append(one(item))
            if i % save_every == 0:
                print(f"rewrite progress {i}/{len(items)}", flush=True)
    else:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(one, item) for item in items]
            for i, fut in enumerate(futures.as_completed(futs), 1):
                results.append(fut.result())
                if i % save_every == 0:
                    print(f"rewrite progress {i}/{len(items)}", flush=True)
    results.sort(key=lambda r: (r["dataset"], r["example_id"]))
    return results


class EmbeddingClient:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.get("base_url", "http://localhost:2335/v1"), api_key=cfg.get("api_key", "EMPTY"), timeout=float(cfg.get("timeout", 600)))
        self.model = cfg.get("name", "Qwen/Qwen3-Embedding-8B")
        self.batch_size = int(cfg.get("batch_size", 32))
        print(f"embedding batch_size={self.batch_size} cache={cfg.get('cache_path')}", flush=True)
        self.max_retries = int(cfg.get("max_retries", 4))
        self.retry_sleep = float(cfg.get("retry_sleep", 1.0))
        cache_path = cfg.get("cache_path")
        self.conn = None
        if cache_path:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(cache_path), timeout=60.0)
            self.conn.execute("pragma journal_mode=WAL")
            self.conn.execute(
                "create table if not exists embeddings (key text primary key, dim integer not null, vector blob not null)"
            )
            self.conn.commit()

    def _embedding_key(self, text: str) -> str:
        h = hashlib.sha256()
        h.update(self.model.encode("utf-8"))
        h.update(b"\0")
        h.update((text or "").encode("utf-8"))
        return h.hexdigest()

    def _get_cached(self, key: str) -> np.ndarray | None:
        if self.conn is None:
            return None
        row = self.conn.execute("select dim, vector from embeddings where key = ?", (key,)).fetchone()
        if row is None:
            return None
        dim, blob = row
        return np.frombuffer(blob, dtype=np.float32, count=int(dim)).copy()

    def _put_cached_many(self, rows: list[tuple[str, np.ndarray]]) -> None:
        if self.conn is None or not rows:
            return
        self.conn.executemany(
            "insert or replace into embeddings(key, dim, vector) values (?, ?, ?)",
            [(key, int(vec.shape[0]), vec.astype(np.float32).tobytes()) for key, vec in rows],
        )
        self.conn.commit()

    def embed_texts(self, texts: list[str], *, view_name: str = "") -> np.ndarray:
        keys = [self._embedding_key(text) for text in texts]
        vectors: list[np.ndarray | None] = [None] * len(texts)
        pending: list[tuple[int, str, str]] = []
        for i, (key, text) in enumerate(zip(keys, texts)):
            cached = self._get_cached(key)
            if cached is None:
                pending.append((i, key, text))
            else:
                vectors[i] = cached
        if pending:
            print(f"embedding {view_name}: cache_hit={len(texts) - len(pending)} pending={len(pending)}", flush=True)
        else:
            print(f"embedding {view_name}: cache_hit={len(texts)} pending=0", flush=True)

        done = len(texts) - len(pending)
        for start in range(0, len(pending), self.batch_size):
            batch_items = pending[start:start + self.batch_size]
            batch = [item[2] for item in batch_items]
            last_error = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.client.embeddings.create(model=self.model, input=batch, timeout=float(self.cfg.get("timeout", 600)))
                    data = sorted(response.data, key=lambda x: x.index)
                    cache_rows = []
                    for item, datum in zip(batch_items, data):
                        i, key, _ = item
                        vec = np.asarray(datum.embedding, dtype=np.float32)
                        vectors[i] = vec
                        cache_rows.append((key, vec))
                    self._put_cached_many(cache_rows)
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(self.retry_sleep * attempt)
            else:
                raise RuntimeError(f"Embedding failed after retries: {last_error}")
            done += len(batch_items)
            print(f"embedded {view_name}: {done}/{len(texts)}", flush=True)

        arr = np.stack([v for v in vectors if v is not None]).astype(np.float32)
        if arr.shape[0] != len(texts):
            raise RuntimeError(f"Embedding count mismatch: got {arr.shape[0]}, expected {len(texts)}")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)


def _reduce_embeddings(vectors: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, str]:
    method = str(cfg.get("method", "auto")).lower()
    n = vectors.shape[0]
    if n < 3:
        return np.pad(vectors[:, :1], ((0, 0), (0, 1)), mode="constant"), "identity"
    if method in {"auto", "umap"}:
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(int(cfg.get("n_neighbors", 20)), max(n - 1, 2)),
                min_dist=float(cfg.get("min_dist", 0.1)),
                metric=str(cfg.get("metric", "cosine")),
                random_state=int(cfg.get("random_state", 13)),
            )
            return reducer.fit_transform(vectors), "umap"
        except Exception as exc:
            if method == "umap":
                raise
            print(f"UMAP unavailable, falling back to PCA: {exc}", flush=True)
    try:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=int(cfg.get("random_state", 13))).fit_transform(vectors), "pca"
    except Exception:
        centered = vectors - vectors.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:2].T, "svd_pca"


def _make_projection(records: list[dict[str, Any]], embeddings: dict[str, np.ndarray], reduce_cfg: dict[str, Any]) -> dict[str, Any]:
    projections: dict[str, Any] = {"views": {}, "method": None}
    fields = {
        "question_raw": "question",
        "context_raw": "context",
        "question_rewritten": "rewritten_question",
        "context_rewritten": "rewritten_context",
    }
    for view, field in fields.items():
        coords, method = _reduce_embeddings(embeddings[view], reduce_cfg)
        projections["method"] = method
        projections["views"][view] = [
            {
                "example_id": r["example_id"],
                "dataset": r["dataset"],
                "split": r["split"],
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "text": str(r.get(field) or "")[:500],
                "question": str(r.get("question") or "")[:300],
                "source_path": r.get("source_path"),
                "source_record_index": r.get("source_record_index"),
            }
            for i, r in enumerate(records)
        ]
    return projections


def _sample_points(points: list[dict[str, Any]], max_points: int | None, seed: int) -> list[dict[str, Any]]:
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return [points[int(i)] for i in idx]


def _write_html(projections: dict[str, Any], path: Path, title: str, report_cfg: dict[str, Any] | None = None) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    report_cfg = report_cfg or {}
    max_per_dataset_view = report_cfg.get("max_points_per_dataset_per_view", 2500)
    max_per_dataset_view = int(max_per_dataset_view) if max_per_dataset_view is not None else None
    hover_text_chars = int(report_cfg.get("hover_text_chars", 180))
    seed = int(report_cfg.get("seed", 13))

    views = [
        ("question_raw", "Raw Questions"),
        ("context_raw", "Raw Contexts"),
        ("question_rewritten", "Rewritten Questions"),
        ("context_rewritten", "Rewritten Contexts"),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[v[1] for v in views])
    datasets = sorted({p["dataset"] for rows in projections["views"].values() for p in rows})
    palette = report_cfg.get("dataset_colors") or [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"
    ]
    color_map = {dataset: palette[i % len(palette)] for i, dataset in enumerate(datasets)}
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    sampled_total = 0
    full_total = 0
    for (view, _), (row, col) in zip(views, positions):
        rows = projections["views"].get(view, [])
        for dataset in datasets:
            pts_full = [p for p in rows if p["dataset"] == dataset]
            if not pts_full:
                continue
            full_total += len(pts_full)
            pts = _sample_points(pts_full, max_per_dataset_view, seed + hash((view, dataset)) % 100000)
            sampled_total += len(pts)
            fig.add_trace(
                go.Scattergl(
                    x=[p["x"] for p in pts],
                    y=[p["y"] for p in pts],
                    mode="markers",
                    name=dataset,
                    legendgroup=dataset,
                    showlegend=(view == "question_raw"),
                    marker={"size": 5, "opacity": 0.62, "color": color_map.get(dataset)},
                    text=[str(p.get("text", ""))[:hover_text_chars] for p in pts],
                    customdata=[[p["example_id"], p["split"], str(p.get("question", ""))[:hover_text_chars], p.get("source_record_index")] for p in pts],
                    hovertemplate="dataset=%{fullData.name}<br>id=%{customdata[0]}<br>split=%{customdata[1]}<br>row=%{customdata[3]}<br>x=%{x:.3f} y=%{y:.3f}<br>%{text}<extra></extra>",
                ),
                row=row,
                col=col,
            )
    display_total = int(projections.get("full_total", full_total) or full_total)
    subtitle = f"sampled {sampled_total:,} / {display_total:,} points; full coordinates are in projection_points.jsonl"
    fig.update_layout(title=f"{title} ({projections.get('method')})<br><sup>{subtitle}</sup>", height=950, template="plotly_white")

    stats_html = ""
    stats = projections.get("stats") or {}
    if stats:
        stage_color = {"raw": "#7f7f7f", "rewritten": "#00a676"}
        stats_fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Question: within-dataset variance",
                "Context: within-dataset variance",
                "Mean centroid distance between datasets",
                "Mean pairwise energy distance between datasets",
            ],
        )
        variance_rows = stats.get("variance", [])
        summary_rows = stats.get("summary", [])
        for kind, col in [("question", 1), ("context", 2)]:
            for stage in ["raw", "rewritten"]:
                rows = [r for r in variance_rows if r.get("kind") == kind and r.get("stage") == stage]
                rows = sorted(rows, key=lambda r: r.get("dataset", ""))
                if not rows:
                    continue
                stats_fig.add_trace(
                    go.Bar(
                        x=[r["dataset"] for r in rows],
                        y=[r["variance"] for r in rows],
                        name=f"{kind} {stage}",
                        legendgroup=f"{kind}-{stage}",
                        marker={"color": stage_color[stage]},
                        opacity=0.72 if stage == "raw" else 0.95,
                    ),
                    row=1,
                    col=col,
                )
        for metric, col, title_suffix in [
            ("mean_centroid_distance", 1, "centroid"),
            ("mean_energy_distance", 2, "energy"),
        ]:
            for kind in ["question", "context"]:
                rows = [r for r in summary_rows if r.get("kind") == kind]
                rows = sorted(rows, key=lambda r: 0 if r.get("stage") == "raw" else 1)
                if not rows:
                    continue
                stats_fig.add_trace(
                    go.Bar(
                        x=[f"{kind} {r['stage']}" for r in rows],
                        y=[r.get(metric, 0.0) for r in rows],
                        name=f"{kind} {title_suffix}",
                    ),
                    row=2,
                    col=col,
                )
        stats_fig.update_layout(
            title="Rewrite distribution diagnostics<br><sup>Lower centroid/energy distance means dataset distributions are closer in the 2D embedding projection. Lower variance means tighter within-dataset spread.</sup>",
            height=850,
            template="plotly_white",
            barmode="group",
        )
        stats_html = stats_fig.to_html(full_html=False, include_plotlyjs=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
        "<style>body{font-family:Arial,sans-serif;margin:18px;} .note{color:#555;max-width:1100px;line-height:1.4}</style>"
        f"</head><body><div class='note'>Full coordinates and statistics are preserved outside this sampled HTML report.</div>"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}{stats_html}</body></html>"
    )
    path.write_text(html, encoding="utf-8")


def run_embedding_rewrite_viz(config: dict[str, Any]) -> dict[str, Any]:
    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("dir", "outputs/analysis/embedding_rewrite_viz"))
    output_dir.mkdir(parents=True, exist_ok=True)

    items = _load_generation_records(config)
    print(f"loaded items: {len(items)}", flush=True)
    if not items:
        raise ValueError("No input items loaded. Check input datasets and split_filter.")

    rows = _rewrite_items(items, config, output_dir)
    write_jsonl(rows, output_dir / "rewrite_records.jsonl")

    fields = {
        "question_raw": [r["question"] for r in rows],
        "context_raw": [r["context"] for r in rows],
        "question_rewritten": [r.get("rewritten_question") or r["question"] for r in rows],
        "context_rewritten": [r.get("rewritten_context") or r["context"] for r in rows],
    }
    embedding_cfg = dict(config.get("embedding", {}))
    embedding_cfg.setdefault("cache_path", str(output_dir / "embedding_cache.sqlite3"))
    embedder = EmbeddingClient(embedding_cfg)
    embeddings = {name: embedder.embed_texts(texts, view_name=name) for name, texts in fields.items()}
    np.savez_compressed(output_dir / "embeddings.npz", **embeddings)

    projections = _make_projection(rows, embeddings, config.get("reduction", {}))
    (output_dir / "projection.json").write_text(json.dumps(projections, ensure_ascii=False), encoding="utf-8")
    projection_rows = []
    for view_name, view_rows in projections.get("views", {}).items():
        for point in view_rows:
            projection_rows.append({"view": view_name, **point})
    write_jsonl(projection_rows, output_dir / "projection_points.jsonl")
    html_path = Path(output_cfg.get("html_path", output_dir / "embedding_rewrite_viz.html"))
    _write_html(projections, html_path, config.get("experiment_name", "embedding_rewrite_viz"), config.get("report", {}))

    summary = {
        "num_examples": len(rows),
        "datasets": {d: sum(1 for r in rows if r["dataset"] == d) for d in sorted({r["dataset"] for r in rows})},
        "rewrite_failed": sum(1 for r in rows if r.get("rewrite_status") == "failed"),
        "embedding_shapes": {k: list(v.shape) for k, v in embeddings.items()},
        "reduction_method": projections.get("method"),
        "html_path": str(html_path),
        "records_path": str(output_dir / "rewrite_records.jsonl"),
        "projection_path": str(output_dir / "projection.json"),
        "projection_points_path": str(output_dir / "projection_points.jsonl"),
    }
    metrics_path = Path(output_cfg.get("summary_path", output_dir / "summary.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
