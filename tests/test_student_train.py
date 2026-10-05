from __future__ import annotations

from pathlib import Path

import pytest

from distillery.student import FakeStudent, answer_confidence, field_confidences

ROOT = Path(__file__).resolve().parent.parent


def test_field_confidences_take_first_token_of_each_value() -> None:
    text = (
        '{"intent": "refund_request", "urgency": "low", "sentiment": "neutral", '
        '"product_area": "camera", "order_id": null, "summary": "x"}'
    )
    # Fake tokenisation: one token per 5 characters.
    ends = list(range(5, len(text) + 5, 5))
    probs = [0.99] * len(ends)
    intent_value_start = text.index("refund_request")
    probs[intent_value_start // 5] = 0.42
    confidences = field_confidences(text, ends, probs)
    assert confidences["intent"] == 0.42
    assert set(confidences) == {"intent", "urgency", "sentiment", "product_area", "order_id"}
    assert answer_confidence(confidences) == 0.42
    assert answer_confidence({"intent": 0.9}) == 0.0  # missing fields -> no confidence


def test_fake_student() -> None:
    [prediction] = FakeStudent().predict_batch(["Where is my order BL-123456? Tracking has not moved."])
    assert prediction.triage is not None and prediction.triage.intent.value == "order_status"
    assert 0 < prediction.confidence <= 1


def _tiny_model(tmp_path: Path) -> Path:
    """A randomly initialised 2-layer Qwen2-style model with a tiny BPE tokenizer and a ChatML template."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    corpus = (ROOT / "data/gold/gold.jsonl").read_text().splitlines()
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(corpus, trainer)
    template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>", eos_token="<|im_end|>", chat_template=template
    )
    model_dir = tmp_path / "tiny"
    fast.save_pretrained(model_dir)
    config = Qwen2Config(
        vocab_size=len(fast),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1024,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
    )
    Qwen2ForCausalLM(config).save_pretrained(model_dir)
    return model_dir


def test_tokenize_example_masks_the_prompt(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from transformers import AutoTokenizer

    from distillery.build import to_chat_record
    from distillery.train import tokenize_example
    from tests.conftest import make_example

    tokenizer = AutoTokenizer.from_pretrained(_tiny_model(tmp_path))
    record = to_chat_record(make_example("x", "refund_request"))
    row = tokenize_example(tokenizer, record["messages"], max_seq_len=4096)
    answer = [token for token, label in zip(row["input_ids"], row["labels"], strict=True) if label != -100]
    decoded = tokenizer.decode(answer)
    assert decoded.startswith('{"intent": "refund_request"')
    assert row["labels"][0] == -100 and len(row["labels"]) == len(row["input_ids"])


@pytest.mark.slow
def test_lora_training_and_inference_on_a_tiny_model(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from distillery.build import to_chat_record
    from distillery.config import StudentConfig, TrainConfig
    from distillery.io import write_jsonl
    from distillery.student import HFStudent
    from distillery.train import train_student
    from tests.conftest import make_example

    model_dir = _tiny_model(tmp_path)
    examples = [make_example(f"e{i}", intent) for i, intent in enumerate(["order_status", "refund_request"] * 4)]
    write_jsonl(tmp_path / "train.jsonl", [to_chat_record(example) for example in examples])
    write_jsonl(tmp_path / "val.jsonl", [to_chat_record(example) for example in examples[:2]])
    metrics = train_student(
        tmp_path / "train.jsonl",
        tmp_path / "val.jsonl",
        tmp_path / "adapter",
        student=StudentConfig(base_model=str(model_dir), device="cpu"),
        train=TrainConfig(epochs=1, batch_size=4, lora_r=4, lora_alpha=8, lora_target_modules=["q_proj", "v_proj"]),
        metrics_path=tmp_path / "metrics.json",
        max_steps=2,
    )
    assert metrics["steps"] == 2 and metrics["trainable_params"] > 0
    assert (tmp_path / "adapter/adapter_config.json").exists()
    assert (tmp_path / "metrics.json").exists()

    student = HFStudent(str(model_dir), tmp_path / "adapter", max_new_tokens=8, device="cpu")
    predictions = student.predict_batch(["Where is my order?", "Refund please"])
    assert len(predictions) == 2
    # A random 2-layer model does not produce valid JSON: the router must see zero confidence.
    assert all(prediction.triage is None and prediction.confidence == 0.0 for prediction in predictions)
