"""Model and task contracts tested without downloading a checkpoint."""

from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from gradiend import (
    PrePruneConfig,
    Signal,
    SignalScope,
    TrainingArguments,
    VisionLanguagePredictionConfig,
    VisionLanguagePredictionTrainer,
)
from gradiend.trainer.core.signals import ActivationSignalExtractor, GradientSignalExtractor
from gradiend.trainer.vision_language.data import VisionLanguageExample
from gradiend.trainer.vision_language.model import VisionLanguageModelWithGradiend


class TinyProcessor:
    tokenizer = SimpleNamespace(all_special_ids=[1, 5])

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages[0]["content"][0]["type"] == "image"
        assert not tokenize and add_generation_prompt
        return "assistant: "

    def __call__(self, *, text, images, add_special_tokens, return_tensors, max_pixels):
        assert not add_special_tokens and return_tensors == "pt" and max_pixels > 0
        answer = text.removeprefix("assistant: ")
        ids = [1, 5, 5, 2]
        if answer == "unalignable":
            ids[-1] = 0
        ids += [{"glasses": 3, "no": 4, "special": 5, "unalignable": 4}[word] for word in answer.split()]
        red = images.getpixel((0, 0))[0] / 255.0
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "pixel_values": torch.tensor([[red, 1 - red, 0.3, 0.4]]),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }

    def save_pretrained(self, path):
        Path(path).mkdir()
        (Path(path) / "processor_config.json").write_text("{}", encoding="utf-8")


class TinyQwen(nn.Module):
    name_or_path = "test/vlm"
    config = SimpleNamespace(model_type="qwen2_5_vl", _commit_hash="locked-revision")

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Linear(4, 4, bias=False)
        self.model.visual = nn.Module()
        self.model.visual.backbone = nn.Linear(4, 4, bias=False)
        self.model.visual.merger = nn.Linear(4, 4, bias=False)
        self.embedding = nn.Embedding(6, 4)
        self.lm_head = nn.Linear(4, 6, bias=False)
        self.positions = None

    def forward(
        self, input_ids, attention_mask, pixel_values, image_grid_thw,
        *, logits_to_keep=0, use_cache=False, return_dict=True, labels=None,
    ):
        assert not use_cache and return_dict
        if isinstance(logits_to_keep, int):
            positions = slice(-logits_to_keep, None)
        else:
            positions = logits_to_keep
            self.positions = positions.tolist()
        visual = self.model.visual
        visual_state = visual.merger(visual.backbone(pixel_values))
        states = self.embedding(input_ids) + visual_state.unsqueeze(1)
        hidden = self.model.language_model(states)
        logits = self.lm_head(hidden[:, positions, :])
        loss = None if labels is None else F.cross_entropy(logits[0, :-1].float(), labels[0, 1:])
        return SimpleNamespace(logits=logits, loss=loss)


@pytest.fixture
def images(tmp_path):
    image_api = pytest.importorskip("PIL.Image")
    paths = [tmp_path / "red.png", tmp_path / "blue.png"]
    for path, color in zip(paths, ("red", "blue")):
        image_api.new("RGB", (2, 2), color=color).save(path)
    return paths


def make_adapter(scope, params, **kwargs):
    return VisionLanguageModelWithGradiend.from_pretrained(
        TinyQwen(), processor=TinyProcessor(), parameter_scope=scope, params=params, device="cpu", **kwargs
    )


@pytest.mark.parametrize(
    ("scope", "parameter"),
    [
        ("language", "model.language_model.weight"),
        ("vision", "model.visual.backbone.weight"),
        ("connector", "model.visual.merger.weight"),
    ],
)
def test_gradient_scopes_and_edits(images, scope, parameter):
    adapter = make_adapter(scope, [parameter])
    adapter.train()
    base = adapter.base_model
    assert adapter.training and not base.training
    assert set(adapter.gradiend.param_map) == {parameter}
    assert {name for name, value in base.named_parameters() if value.requires_grad} == {parameter}

    factual = VisionLanguageExample(images[0], "Are there glasses?", "glasses")
    image_swap = VisionLanguageExample(images[1], factual.prompt, factual.answer)
    answer_swap = VisionLanguageExample(factual.image, factual.prompt, "no glasses")
    gradient = adapter.create_gradients(factual)
    assert gradient.shape == (16,) and torch.isfinite(gradient).all()
    assert not torch.allclose(adapter.create_gradients(image_swap), gradient)
    assert not torch.allclose(adapter.create_gradients(answer_swap), gradient)
    assert all(value.grad is None for value in base.parameters())
    assert isinstance(adapter.encode(factual, return_float=True), float)

    with torch.no_grad():
        for value in adapter.gradiend.decoder.parameters():
            value.fill_(1.0)
    original = {name: value.detach().clone() for name, value in base.named_parameters()}
    edited = adapter.rewrite_base_model(learning_rate=0.1, feature_factor=1.0)
    assert {name for name, value in edited.named_parameters() if not torch.equal(value, original[name])} == {parameter}
    assert all(torch.equal(value, original[name]) for name, value in base.named_parameters())


def test_answer_scoring_uses_both_causally_shifted_tokens(images):
    adapter = make_adapter("language", ["model.language_model.weight"])
    example = VisionLanguageExample(images[0], "Identify the accessory", "no glasses")
    prepared = adapter.prepare_example(example)
    score, count = adapter.score_example(example)
    assert count == 2
    assert adapter.base_model.positions == [3, 4]

    inputs = prepared.model_inputs
    with torch.no_grad():
        logits = adapter.base_model(
            **inputs, logits_to_keep=torch.arange(6), use_cache=False, return_dict=True
        ).logits
        expected = logits[0].log_softmax(-1)[[3, 4], [4, 3]].sum().item()
    assert score == pytest.approx(expected)


def test_unreadable_or_unalignable_examples_include_row_information(images):
    adapter = make_adapter("language", ["model.language_model.weight"])
    with pytest.raises(ValueError, match="row 'broken-image'"):
        adapter.prepare_example(VisionLanguageExample(images[0].with_suffix(".missing"), "Question", "glasses"), row_id="broken-image")
    with pytest.raises(ValueError, match="align answer tokens.*row 'broken-answer'"):
        adapter.prepare_example(VisionLanguageExample(images[0], "Question", "unalignable"), row_id="broken-answer")
    with pytest.raises(ValueError, match="special token.*row 'special-answer'"):
        adapter.prepare_example(VisionLanguageExample(images[0], "Question", "special"), row_id="special-answer")
    short = make_adapter("language", ["model.language_model.weight"], max_length=5)
    with pytest.raises(ValueError, match="row 'long-answer'.*exceeding max_length=5"):
        short.prepare_example(VisionLanguageExample(images[0], "Question", "no glasses"), row_id="long-answer")


def test_parameter_filters_must_match_within_the_scope():
    with pytest.raises(ValueError, match="matches nothing in the selected scope"):
        make_adapter("vision", ["model.visual.*", "model.visual.merger.weight"])
    with pytest.raises(ValueError, match="Specify TrainingArguments.signal_scope"):
        make_adapter("language", None)


def test_adapter_dtype_change_preserves_base_model_precision():
    base = TinyQwen().to(dtype=torch.bfloat16)
    adapter = VisionLanguageModelWithGradiend.from_pretrained(
        base,
        processor=TinyProcessor(),
        params=["model.language_model.weight"],
        base_model_dtype="bfloat16",
        device="cpu",
    )
    adapter.to(torch_dtype=torch.float64)
    assert adapter.gradiend.torch_dtype == torch.float64
    assert next(base.parameters()).dtype == torch.bfloat16


@pytest.mark.parametrize("sparse", [False, True])
def test_scoped_bfloat16_edit_restores_weights_after_exception(sparse):
    base = TinyQwen().to(dtype=torch.bfloat16)
    parameter = "model.visual.merger.weight"
    adapter = VisionLanguageModelWithGradiend.from_pretrained(
        base, processor=TinyProcessor(), params=[parameter], parameter_scope="connector", device="cpu"
    )
    if sparse:
        adapter.prune_gradiend(keep_idx=torch.tensor([1, 5]))
    with torch.no_grad():
        for value in adapter.gradiend.decoder.parameters():
            value.fill_(1.0)
    originals = {name: value.detach().clone() for name, value in base.named_parameters()}

    with pytest.raises(RuntimeError, match="failed scoring"):
        with adapter.intervene(value=0.3, feature_factor=1.0):
            model = adapter.base_model
            touched = {
                name for name, value in model.named_parameters()
                if not torch.equal(value, originals[name])
            }
            assert touched == {parameter}
            changed_positions = (model.model.visual.merger.weight != originals[parameter]).flatten().nonzero().flatten()
            assert changed_positions.tolist() == ([1, 5] if sparse else list(range(16)))
            raise RuntimeError("failed scoring")

    assert all(torch.equal(value, originals[name]) for name, value in base.named_parameters())


def test_signal_payload_native_gradient_matches_continuation_loss(images):
    adapter = make_adapter("connector", ["model.visual.merger.weight"])
    example = VisionLanguageExample(images[0], "Identify the accessory", "no glasses")

    inputs = adapter.prepare_signal_inputs(example, include_answer=True)
    assert inputs["input_ids"].tolist() == [[1, 5, 5, 2, 4, 3]]
    assert inputs["labels"].tolist() == [[-100, -100, -100, -100, 4, 3]]
    assert inputs["prediction_mask"].nonzero().tolist() == [[0, 3]]
    assert inputs["prediction_mask_is_generating"] is True
    assert inputs["image_grid_thw"].tolist() == [[1, 2, 2]]
    captured = GradientSignalExtractor(adapter.gradient_creator)(inputs, requires_alternative=False)
    torch.testing.assert_close(captured.factual, adapter.create_gradients(example))
    assert all(parameter.grad is None for parameter in adapter.base_model.parameters())

    # The connector weight map has width 16, while the shared model's language
    # activation probe has width 4.
    scope = SignalScope.from_values(activation_sites=["model.language_model"])
    signal = Signal.activation_gradient(token_selector="prediction")
    activation_gradient = ActivationSignalExtractor(adapter, signal=signal, scope=scope)(
        inputs, requires_alternative=False
    )
    assert activation_gradient.factual.shape == (4,)
    assert torch.count_nonzero(activation_gradient.factual)


def test_native_activation_and_activation_gradient_use_generating_position(images):
    base = TinyQwen()
    scope = SignalScope.from_values(activation_sites=["model.language_model"])
    activation = Signal.activation(token_selector="prediction")
    adapter = VisionLanguageModelWithGradiend.from_pretrained(
        base, processor=TinyProcessor(), device="cpu",
        training_args=TrainingArguments(signal=activation, signal_scope=scope),
    )
    factual = VisionLanguageExample(images[0], "Identify the accessory", "no glasses")
    swapped = VisionLanguageExample(images[0], factual.prompt, "glasses")
    prompt = adapter.prepare_signal_inputs(factual, include_answer=False)
    prompt_swap = adapter.prepare_signal_inputs(swapped, include_answer=False)
    assert "labels" not in prompt
    assert prompt["input_ids"].tolist() == [[1, 5, 5, 2]]
    assert prompt["prediction_mask"].nonzero().tolist() == [[0, 3]]
    result = ActivationSignalExtractor(adapter, signal=activation, scope=scope)(prompt, prompt_swap)
    torch.testing.assert_close(result.factual, result.alternative)

    gradient = Signal.activation_gradient(token_selector="prediction")
    inputs = adapter.prepare_signal_inputs(factual, include_answer=True)
    swap_inputs = adapter.prepare_signal_inputs(swapped, include_answer=True)
    result = ActivationSignalExtractor(adapter, signal=gradient, scope=scope)(inputs, swap_inputs)
    assert result.factual.shape == result.alternative.shape == (4,)
    assert not torch.allclose(result.factual, result.alternative)
    assert all(parameter.grad is None for parameter in base.parameters())

    from gradiend.model.modified import register_activation_steering_hooks

    score, _ = adapter.score_example(factual)
    with adapter.intervene(
        value=0.2, token_selector="prediction", update_vector=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    ):
        edited, _ = adapter.score_example(factual)
    assert edited != pytest.approx(score)
    assert adapter.score_example(factual)[0] == pytest.approx(score)

    hooks = register_activation_steering_hooks(
        base,
        interventions=[{
            "module": "model.language_model",
            "tensor_key": "delta",
            "application": {"axis": "last_dim", "token_selector": "prediction"},
        }],
        tensors={"delta": torch.tensor([1.0, 0.0, 0.0, 0.0])},
    )
    try:
        external, _ = adapter.score_example(factual, mask_prediction_position=True)
    finally:
        for hook in hooks:
            hook.remove()
    assert external == pytest.approx(edited)


def test_checkpoint_load_restores_processor_and_scope(tmp_path, monkeypatch):
    from transformers import AutoProcessor

    parameter = "model.visual.merger.weight"
    adapter = make_adapter("connector", [parameter])
    checkpoint = tmp_path / "adapter"
    adapter.save_pretrained(str(checkpoint))

    def load_processor(path, *, revision):
        assert revision is None
        assert (path / "processor_config.json").is_file()
        return TinyProcessor()

    monkeypatch.setattr(AutoProcessor, "from_pretrained", load_processor)
    loaded = VisionLanguageModelWithGradiend.from_pretrained(
        str(checkpoint), base_model=adapter.base_model, device="cpu", require_gradiend_model=True
    )
    assert loaded.parameter_scope == "connector"
    assert loaded.gradiend.param_map == adapter.gradiend.param_map
    with pytest.raises(ValueError, match="parameter_scope=.*conflicts"):
        VisionLanguageModelWithGradiend.from_pretrained(
            str(checkpoint), base_model=adapter.base_model, device="cpu", parameter_scope="language"
        )


def test_trainer_constructs_and_pre_prunes_a_matched_manifest(images, monkeypatch):
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

    base = TinyQwen()
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: base.config)
    monkeypatch.setattr(Qwen2_5_VLForConditionalGeneration, "from_pretrained", lambda *args, **kwargs: base)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: TinyProcessor())
    prepared_rows = set()
    prepare_example = VisionLanguageModelWithGradiend.prepare_example

    def record_prepared_row(self, example, *, row_id=None):
        prepared_rows.add(row_id)
        return prepare_example(self, example, row_id=row_id)

    monkeypatch.setattr(VisionLanguageModelWithGradiend, "prepare_example", record_prepared_row)
    matched = pd.DataFrame([{
        "group_id": f"portrait-{index}",
        "split": "train",
        "prompt": f"Identify the accessory in portrait {index}",
        "image_a": str(images[0]),
        "answer_a": "no glasses",
        "image_b": str(images[1]),
        "answer_b": "glasses",
    } for index in range(4)])
    trainer = VisionLanguagePredictionTrainer(
        base.name_or_path,
        data=matched,
        config=VisionLanguagePredictionConfig(
            target_classes=("absent", "present"), parameter_scope="connector", base_model_dtype="float32",
        ),
        args=TrainingArguments(
            source="factual", target="diff", base_gradient_batch_size=1, do_eval=False,
            signal_scope=SignalScope.from_values(params=["model.visual.merger.weight"]),
        ),
    )
    assert trainer._get_decoder_eval_targets() == {"absent": ["no glasses"], "present": ["glasses"]}
    pre_prune = PrePruneConfig(n_samples=2, topk=3, use_streaming=False, seed=17)
    assert trainer.pre_prune(pre_prune) is trainer
    assert len(prepared_rows) == pre_prune.n_samples
    assert trainer.get_model().gradiend.input_dim == 3
    assert trainer.get_model().feature_class_encoding_direction == {"absent": -1, "present": 1}


def test_decoder_grid_can_reverse_full_edit_and_scores_only_selected_edit_on_test():
    scored = []

    @contextmanager
    def temporary_edit(*, value, feature_factor):
        adapter.base_model = value
        try:
            yield
        finally:
            adapter.base_model = 0.0

    def evaluate_base_model(model, *, split, **kwargs):
        scored.append((split, model))
        return {
            "grounding": 1.0 - model,
            "negative_image_margin": 0.0,
            "positive_image_margin": 1.0 - model,
            "neutral_mean_log_probability": -1.0,
        }

    adapter = SimpleNamespace(
        base_model=0.0, intervene=temporary_edit, exclusive_base_gradient_access=nullcontext,
    )
    trainer = SimpleNamespace(
        config=SimpleNamespace(min_neutral_likelihood_ratio=0.99),
        _resolve_artifact_use_cache=lambda value: False,
        _default_from_training_args=lambda value, name, fallback: value,
        _decoder_rows=lambda *args: None,
        get_model=lambda: adapter,
        evaluate_base_model=evaluate_base_model,
    )
    result = VisionLanguagePredictionTrainer.evaluate_decoder(trainer, lrs=[0.01, -0.01], feature_factors=[1.0])

    assert result["selected"]["learning_rate"] == -0.01
    assert result["test"]["edited"]["grounding_improvement"] == pytest.approx(0.01)
    assert scored == [
        ("validation", 0.0), ("validation", -0.01), ("validation", 0.01), ("test", 0.0), ("test", -0.01)
    ]
