"""Per-model fine-tuning driver.

Usage (via the run.py CLI):

    train_one(
        model_cfg=...,
        train_df=..., val_df=..., test_df=...,
        rank=0, out_dir=...,
        epochs=3, lr=2e-5, batch_size=8, max_seq_len=256,
    )

Returns a dict of metrics + a path to the (LoRA) adapter checkpoint.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from fasturduguard.preprocess.pipeline import preprocess_iter
from fasturduguard.train.profile import StopWatch, cpu_info, gpu_info, write_profile

log = logging.getLogger("fug.train.trainer")


# ---------------------------------------------------------------------------
# Quantization / PEFT
# ---------------------------------------------------------------------------
def _maybe_quant_config(quant: str):
    if quant != "nf4":
        return None
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    except Exception as e:
        log.warning("bitsandbytes/NF4 unavailable (%s) -- falling back to FP16", e)
        return None


def _maybe_lora(model, mcfg: dict):
    if mcfg.get("peft") != "lora":
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:
        log.warning("peft missing (%s); training without LoRA", e)
        return model
    lc = LoraConfig(
        r=mcfg["lora_r"],
        lora_alpha=mcfg["lora_alpha"],
        lora_dropout=mcfg["lora_dropout"],
        target_modules=mcfg["lora_target"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    try:
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()
    except Exception as e:
        log.warning("LoRA injection failed for target=%s (%s); using full FT",
                    mcfg["lora_target"], e)
    return model


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
class TokDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict[str, Any], labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
        return item


def _build_dataset(tok, texts: list[str], labels: list[int], max_len: int) -> TokDataset:
    enc = tok(texts, padding=False, truncation=True, max_length=max_len)
    return TokDataset({k: v for k, v in enc.items()}, np.array(labels, dtype=np.int64))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def train_one(
    *,
    model_cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    rank: int,
    label_col: str = "label_4",
    text_col: str = "text",
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 8,
    grad_accum: int = 2,
    max_seq_len: int = 256,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    fp16: bool = True,
    seed: int = 42,
    n_workers_pre: int = 4,
    preprocess: bool = True,
) -> dict[str, Any]:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = model_cfg["name"]
    hf_id = model_cfg["hf_id"]
    placement = model_cfg.get("placement", "gpu")
    log.info("[rank=%d] Training '%s' (%s)  placement=%s  rows train/val/test=%d/%d/%d",
             rank, name, hf_id, placement, len(train_df), len(val_df), len(test_df))

    if placement == "cpu":
        device_str = "cpu"
        fp16 = False
        torch.set_num_threads(max(1, n_workers_pre))
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "cpu":
            log.warning("placement=gpu but CUDA not available; falling back to CPU")
            fp16 = False

    if preprocess:
        train_texts = preprocess_iter(train_df[text_col].tolist(), n_workers=n_workers_pre)
        val_texts = preprocess_iter(val_df[text_col].tolist(), n_workers=n_workers_pre)
        test_texts = preprocess_iter(test_df[text_col].tolist(), n_workers=n_workers_pre)
    else:
        train_texts = train_df[text_col].tolist()
        val_texts = val_df[text_col].tolist()
        test_texts = test_df[text_col].tolist()

    n_labels = int(max(
        train_df[label_col].max(),
        val_df[label_col].max(),
        test_df[label_col].max(),
    )) + 1

    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

    train_ds = _build_dataset(tok, train_texts, train_df[label_col].tolist(), max_seq_len)
    val_ds   = _build_dataset(tok, val_texts,   val_df[label_col].tolist(),   max_seq_len)
    test_ds  = _build_dataset(tok, test_texts,  test_df[label_col].tolist(),  max_seq_len)

    quant_cfg = _maybe_quant_config(model_cfg.get("quant", "fp16"))
    config = AutoConfig.from_pretrained(hf_id, num_labels=n_labels)

    model_kwargs: dict[str, Any] = {"config": config}
    if quant_cfg is not None and device_str == "cuda":
        model_kwargs["quantization_config"] = quant_cfg
        model_kwargs["device_map"] = {"": 0}
    elif fp16 and device_str == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(hf_id, **model_kwargs)

    if quant_cfg is not None:
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
        except Exception:
            pass

    model = _maybe_lora(model, model_cfg)

    ckpt = out_dir / "checkpoints" / name
    ckpt.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(ckpt),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(batch_size * 2, 16),
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=25,
        report_to=[],
        fp16=fp16 and device_str == "cuda",
        seed=seed,
        dataloader_num_workers=0,        # process pool already runs in preprocess
        use_cpu=(device_str == "cpu"),
        remove_unused_columns=False,
    )

    coll = DataCollatorWithPadding(tokenizer=tok, padding="longest")

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=coll,
        compute_metrics=_compute_metrics,
    )
    # transformers >=5 renamed `tokenizer` -> `processing_class`
    import inspect as _ins
    _params = _ins.signature(Trainer.__init__).parameters
    if "processing_class" in _params:
        trainer_kwargs["processing_class"] = tok
    elif "tokenizer" in _params:
        trainer_kwargs["tokenizer"] = tok
    trainer = Trainer(**trainer_kwargs)

    sw = StopWatch()
    cpu_before = cpu_info()
    gpu_before = gpu_info()
    with sw:
        train_out = trainer.train()
    cpu_after = cpu_info()
    gpu_after = gpu_info()

    val_metrics = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds)
    test_preds = trainer.predict(test_ds)
    cm = confusion_matrix(test_preds.label_ids, np.argmax(test_preds.predictions, axis=-1)).tolist()
    cls_report = classification_report(
        test_preds.label_ids, np.argmax(test_preds.predictions, axis=-1),
        digits=4, zero_division=0, output_dict=True,
    )

    # Save adapter (LoRA) if any, else full model
    try:
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(ckpt))
        if hasattr(model, "peft_config"):
            (ckpt / "is_lora").write_text("1", encoding="utf-8")
    except Exception as e:
        log.warning("Save failed: %s", e)
    tok.save_pretrained(str(ckpt))

    metrics = {
        "model": name,
        "hf_id": hf_id,
        "rank": rank,
        "placement": placement,
        "device_used": device_str,
        "rows_train": len(train_df),
        "rows_val": len(val_df),
        "rows_test": len(test_df),
        "val": val_metrics,
        "test": test_metrics,
        "training_runtime_s": sw.elapsed_s,
        "samples_per_sec": (
            float(getattr(train_out.metrics, "get", lambda *_: None)("train_samples_per_second"))
            if hasattr(train_out.metrics, "get") else
            float(train_out.metrics.get("train_samples_per_second", 0.0))
        ),
        "peak_vram_mb": sw.peak_vram_mb,
        "cpu_before": cpu_before,
        "cpu_after": cpu_after,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "checkpoint_dir": str(ckpt.resolve()),
    }

    write_profile(out_dir / "metrics" / f"{name}.json", metrics)
    return metrics
