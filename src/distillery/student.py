"""The student at inference time: a small causal LM (+ LoRA adapter) that emits the triage JSON.

Confidence is computed from the model's own token probabilities: for each structured field we take
the probability of the first token of the value the model chose (the moment it "decides" between
e.g. `refund_request` and `return_exchange`), and the answer's confidence is the minimum over
fields. The router compares it with a calibrated threshold.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

from distillery.build import chat_messages
from distillery.heuristics import heuristic_triage
from distillery.prompts import STUDENT_SYSTEM_PROMPT
from distillery.schema import STRUCTURED_FIELDS, TicketTriage, parse_triage

if TYPE_CHECKING:
    import torch


class StudentPrediction(BaseModel):
    raw: str
    triage: TicketTriage | None = None
    json_valid: bool = False
    error: str | None = None
    confidence: float = 0.0
    field_confidence: dict[str, float] = Field(default_factory=dict)
    latency_s: float = 0.0


class Student(Protocol):
    @property
    def label(self) -> str: ...

    def predict_batch(self, texts: Sequence[str]) -> list[StudentPrediction]: ...


class FakeStudent:
    """Keyword triage with a keyword-count confidence: the offline stand-in for a trained student."""

    @property
    def label(self) -> str:
        return "fake-student"

    def predict_batch(self, texts: Sequence[str]) -> list[StudentPrediction]:
        predictions = []
        for text in texts:
            started = time.perf_counter()
            result = heuristic_triage(text)
            predictions.append(
                StudentPrediction(
                    raw=result.triage.to_json(),
                    triage=result.triage,
                    json_valid=True,
                    confidence=result.confidence,
                    field_confidence=dict.fromkeys(STRUCTURED_FIELDS, result.confidence),
                    latency_s=time.perf_counter() - started,
                )
            )
        return predictions


_VALUE_START = {field: re.compile(rf'"{field}"\s*:\s*"?') for field in STRUCTURED_FIELDS}


def field_confidences(text: str, token_ends: Sequence[int], token_probs: Sequence[float]) -> dict[str, float]:
    """Probability of the first token of each field's value.

    `token_ends[i]` is the character offset in `text` where generated token i ends.
    """
    confidences: dict[str, float] = {}
    for field, pattern in _VALUE_START.items():
        match = pattern.search(text)
        if not match:
            continue
        value_start = match.end()
        for end, prob in zip(token_ends, token_probs, strict=True):
            if end > value_start:
                confidences[field] = round(prob, 4)
                break
    return confidences


def answer_confidence(field_conf: dict[str, float]) -> float:
    if not field_conf or any(field not in field_conf for field in STRUCTURED_FIELDS):
        return 0.0
    return min(field_conf.values())


def resolve_device(preference: str) -> str:
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFStudent:
    """Hugging Face causal LM with an optional PEFT/LoRA adapter (merged at load for faster inference)."""

    def __init__(
        self,
        base_model: str,
        adapter_path: Path | None = None,
        *,
        system_prompt: str = STUDENT_SYSTEM_PROMPT,
        max_new_tokens: int = 120,
        device: str = "auto",
        torch_threads: int | None = None,
        name: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch_threads:
            torch.set_num_threads(torch_threads)
        self.device = resolve_device(device)
        source = str(adapter_path) if adapter_path and (adapter_path / "tokenizer_config.json").exists() else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path)).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self._label = name or (f"{base_model}+lora" if adapter_path else base_model)

    @property
    def label(self) -> str:
        return self._label

    def _prompt(self, text: str) -> str:
        rendered = self.tokenizer.apply_chat_template(
            chat_messages(text, self.system_prompt), tokenize=False, add_generation_prompt=True
        )
        return str(rendered)

    def predict_batch(self, texts: Sequence[str]) -> list[StudentPrediction]:
        import torch

        if not texts:
            return []
        started = time.perf_counter()
        encoded = self.tokenizer(
            [self._prompt(text) for text in texts], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        scores = self.model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
        elapsed = time.perf_counter() - started
        prompt_length = encoded["input_ids"].shape[1]
        predictions = []
        for row in range(len(texts)):
            generated = output.sequences[row, prompt_length:]
            predictions.append(self._decode(generated, scores[row], elapsed / len(texts)))
        return predictions

    def _decode(self, generated: torch.Tensor, logprobs: torch.Tensor, latency: float) -> StudentPrediction:
        stop_ids = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        ids: list[int] = []
        probs: list[float] = []
        for token_id, logprob in zip(generated.tolist(), logprobs.tolist(), strict=True):
            if token_id in stop_ids:
                break
            ids.append(int(token_id))
            probs.append(math.exp(float(logprob)))
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        token_ends = [
            len(self.tokenizer.decode(ids[: index + 1], skip_special_tokens=True)) for index in range(len(ids))
        ]
        outcome = parse_triage(text)
        field_conf = field_confidences(text, token_ends, probs)
        return StudentPrediction(
            raw=text,
            triage=outcome.triage,
            json_valid=outcome.json_valid,
            error=outcome.error,
            confidence=answer_confidence(field_conf) if outcome.ok else 0.0,
            field_confidence=field_conf,
            latency_s=latency,
        )


def predict_all(student: Student, texts: Sequence[str], batch_size: int) -> list[StudentPrediction]:
    predictions: list[StudentPrediction] = []
    for start in range(0, len(texts), batch_size):
        predictions.extend(student.predict_batch(texts[start : start + batch_size]))
    return predictions


def load_student(
    backend: str,
    *,
    base_model: str,
    adapter_path: Path | None,
    max_new_tokens: int = 120,
    device: str = "auto",
    torch_threads: int | None = None,
    system_prompt: str = STUDENT_SYSTEM_PROMPT,
    name: str | None = None,
) -> Student:
    if backend == "fake":
        return FakeStudent()
    return HFStudent(
        base_model,
        adapter_path,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        torch_threads=torch_threads,
        name=name,
    )
