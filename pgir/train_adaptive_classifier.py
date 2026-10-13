"""Train the Adaptive-RAG query-complexity classifier (t5-large, official arch).

Seq2seq: input = question text, target = single label letter A/B/C — exactly
what FlashRAG's AdaptiveJudger expects (it generates 1 step and softmaxes over
the A/B/C token logits).

Run (pin a free GPU):
  CUDA_VISIBLE_DEVICES=4 PYTORCH_NVML_BASED_CUDA_CHECK=0 \
      python -m pgir.train_adaptive_classifier
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ASSETS = Path(__file__).resolve().parent / "assets"
OUT = ASSETS / "adaptive_classifier_t5large"
BASE = "t5-large"


class QDataset(Dataset):
    def __init__(self, path, tok):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = self.tok(r["question"], truncation=True, max_length=128)
        y = self.tok(r["label"]).input_ids
        return {"input_ids": x.input_ids, "attention_mask": x.attention_mask, "labels": y}


def main():
    import argparse
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-suffix", default="",
                    help="optional label-file suffix: reads adaptive_labels_<suffix>*.jsonl "
                         "and appends the suffix to the output dir")
    a = ap.parse_args()
    global OUT
    if a.labels_suffix:
        OUT = ASSETS / f"adaptive_classifier_t5large_{a.labels_suffix.rstrip('_')}"

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
    train_ds = QDataset(ASSETS / f"adaptive_labels_{a.labels_suffix}train.jsonl", tok)
    val_ds = QDataset(ASSETS / f"adaptive_labels_{a.labels_suffix}val.jsonl", tok)

    args = TrainingArguments(
        output_dir=str(OUT / "ckpts"), per_device_train_batch_size=64,
        per_device_eval_batch_size=128, learning_rate=3e-4, num_train_epochs=2,
        logging_steps=50, eval_strategy="epoch", save_strategy="no",
        bf16=True, report_to=[], seed=17,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, model=model))
    trainer.train()

    # eval: label accuracy on silver val
    model.eval()
    ids = {lab: tok(lab).input_ids[0] for lab in "ABC"}
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(val_ds.rows), 128):
            batch = val_ds.rows[i:i + 128]
            enc = tok([r["question"] for r in batch], truncation=True, max_length=128,
                      padding=True, return_tensors="pt").to(model.device)
            scores = model.generate(**enc, max_length=2, return_dict_in_generate=True,
                                    output_scores=True).scores[0]
            sub = scores[:, [ids["A"], ids["B"], ids["C"]]]
            pred = sub.argmax(-1).tolist()
            for r, p in zip(batch, pred):
                correct += ("ABC"[p] == r["label"])
                total += 1
    print(f"silver-val label accuracy: {correct/total:.4f} ({total})")
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
