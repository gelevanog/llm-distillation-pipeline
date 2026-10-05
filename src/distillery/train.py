"""Stage 5, train: LoRA supervised fine-tuning of the student on the chat-format dataset.

Plain `transformers.Trainer` + `peft`: the loss is computed on the assistant answer only (prompt
tokens are masked with -100), so the student learns to produce the JSON, not to repeat the
instructions. Runs on CPU (tiny model, measured in the README) or a single GPU (bigger model).
"""

from __future__ import annotations

import math
import platform
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from distillery.config import StudentConfig, TrainConfig
from distillery.io import iter_jsonl, write_json
from distillery.logging_config import get_logger
from distillery.student import resolve_device

log = get_logger(__name__)


def tokenize_example(tokenizer: Any, messages: Sequence[dict[str, str]], max_seq_len: int) -> dict[str, list[int]]:
    """input_ids + labels where everything before the assistant answer is masked out."""
    prompt = tokenizer.apply_chat_template(list(messages[:-1]), tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(list(messages), tokenize=False)
    prompt_ids: list[int] = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids: list[int] = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        # Templates that re-render earlier turns differently: fall back to prompt + answer + EOS.
        answer_ids: list[int] = tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"]
        full_ids = [*prompt_ids, *answer_ids, tokenizer.eos_token_id]
    full_ids = full_ids[:max_seq_len]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "labels": labels}


class PadCollator:
    """Right-pads input_ids with the pad token and labels with -100."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Sequence[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention = [], [], []
        for item in batch:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention),
        }


def load_chat_records(path: Path) -> list[list[dict[str, str]]]:
    return [record["messages"] for record in iter_jsonl(path)]


def train_student(
    train_path: Path,
    val_path: Path | None,
    adapter_dir: Path,
    *,
    student: StudentConfig,
    train: TrainConfig,
    metrics_path: Path | None = None,
    merged_dir: Path | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(train.seed)
    if student.torch_threads:
        torch.set_num_threads(student.torch_threads)
    device = resolve_device(student.device)
    tokenizer = AutoTokenizer.from_pretrained(student.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = [
        tokenize_example(tokenizer, messages, train.max_seq_len) for messages in load_chat_records(train_path)
    ]
    val_rows = (
        [tokenize_example(tokenizer, messages, train.max_seq_len) for messages in load_chat_records(val_path)]
        if val_path and val_path.exists()
        else []
    )
    if not train_rows:
        raise ValueError(f"no training examples in {train_path}")
    lengths = [len(row["input_ids"]) for row in train_rows]
    answer_tokens = sum(sum(1 for label in row["labels"] if label != -100) for row in train_rows)

    dtype = torch.bfloat16 if train.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(student.base_model, dtype=dtype)
    if train.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=train.lora_r,
        lora_alpha=train.lora_alpha,
        lora_dropout=train.lora_dropout,
        target_modules=train.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    log.info(
        "train.start",
        base_model=student.base_model,
        device=device,
        examples=len(train_rows),
        trainable_params=trainable,
        total_params=total,
    )

    steps_per_epoch = math.ceil(len(train_rows) / (train.batch_size * train.gradient_accumulation_steps))
    total_steps = max_steps or max(1, math.ceil(steps_per_epoch * train.epochs))
    args = TrainingArguments(
        output_dir=str(adapter_dir.parent / "checkpoints"),
        num_train_epochs=train.epochs,
        max_steps=max_steps or -1,
        per_device_train_batch_size=train.batch_size,
        per_device_eval_batch_size=train.batch_size,
        gradient_accumulation_steps=train.gradient_accumulation_steps,
        learning_rate=train.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(0, round(total_steps * train.warmup_ratio)),
        logging_steps=max(1, min(10, total_steps // 10 or 1)),
        eval_strategy="epoch" if val_rows and not max_steps else "no",
        save_strategy="no",
        report_to="none",
        seed=train.seed,
        bf16=train.bf16,
        use_cpu=device == "cpu",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        eval_dataset=val_rows or None,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    started = time.time()
    result = trainer.train()
    wall = time.time() - started

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    if merged_dir is not None:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

    history = [entry for entry in trainer.state.log_history if "loss" in entry or "eval_loss" in entry]
    metrics: dict[str, Any] = {
        "base_model": student.base_model,
        "device": device,
        "torch_threads": torch.get_num_threads(),
        "cpu": platform.processor() or platform.machine(),
        "examples": len(train_rows),
        "val_examples": len(val_rows),
        "epochs": train.epochs,
        "steps": trainer.state.global_step,
        "trainable_params": trainable,
        "total_params": total,
        "lora": {
            "r": train.lora_r,
            "alpha": train.lora_alpha,
            "dropout": train.lora_dropout,
            "targets": train.lora_target_modules,
        },
        "learning_rate": train.learning_rate,
        "batch_size": train.batch_size * train.gradient_accumulation_steps,
        "seq_len": {"mean": round(sum(lengths) / len(lengths), 1), "max": max(lengths)},
        "answer_tokens": answer_tokens,
        "train_loss": round(float(result.training_loss), 4),
        "wall_seconds": round(wall, 1),
        "seconds_per_example_epoch": round(wall / (len(train_rows) * train.epochs), 3) if not max_steps else None,
        "adapter_mb": round(sum(file.stat().st_size for file in adapter_dir.glob("*.safetensors")) / 1e6, 2),
        "history": history,
    }
    eval_losses = [entry["eval_loss"] for entry in history if "eval_loss" in entry]
    if eval_losses:
        metrics["final_eval_loss"] = round(float(eval_losses[-1]), 4)
    if metrics_path is not None:
        write_json(metrics_path, metrics)
    log.info("train.done", **{key: value for key, value in metrics.items() if key != "history"})
    return metrics
