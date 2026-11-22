"""Validate Janus generation alignment and VQ codes with a small causal stand-in."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from gradiend.trainer.vision_language import JanusGenerationTask


class TinyJanusProcessor:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return messages[0]["content"][0]["text"]

    def __call__(self, *, text, generation_mode, return_tensors):
        assert generation_mode == "image" and return_tensors == "pt"
        input_ids = torch.tensor([[0, len(text) % 4, 4]])
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def image_processor(self, *, images, return_tensors):
        assert images.mode == "RGB" and return_tensors == "pt"
        red, green, blue = images.getpixel((0, 0))
        pixel = torch.tensor([red, green, blue]) / 127.5 - 1
        return {"pixel_values": pixel.reshape(1, 3, 1, 1).expand(1, 3, 2, 2)}

    def postprocess(self, images, *, return_tensors):
        from PIL import Image

        assert return_tensors == "PIL.Image.Image"
        image = images[0]
        return {"pixel_values": [Image.new("RGB", (image.shape[1], image.shape[0]))]}


class TinyCausalBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)

    def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict):
        assert not use_cache and return_dict
        assert attention_mask.shape == inputs_embeds.shape[:2]
        return SimpleNamespace(last_hidden_state=self.linear(inputs_embeds.cumsum(dim=1)))


class TinyJanus(nn.Module):
    config = SimpleNamespace(
        model_type="janus",
        vision_config=SimpleNamespace(num_image_tokens=4),
        text_config=SimpleNamespace(max_position_embeddings=20),
    )
    generation_config = SimpleNamespace(generation_kwargs={"boi_token_id": 4})

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = TinyCausalBackbone()
        self.model.generation_embeddings = nn.Embedding(3, 3)
        self.model.generation_head = nn.Linear(3, 3)
        self.text_embeddings = nn.Embedding(5, 3)
        vq = self.model.vqmodel = nn.Module()
        vq.config = SimpleNamespace(num_embeddings=3, num_patches=2)
        vq.encoder = nn.Conv2d(3, 2, 1, bias=False)
        vq.quant_conv = nn.Identity()
        vq.quantize = nn.Module()
        vq.quantize.embedding = nn.Embedding(3, 2)
        with torch.no_grad():
            vq.encoder.weight.copy_(torch.tensor([[[[1]], [[0]], [[0]]], [[[0]], [[1]], [[0]]]]))
            vq.quantize.embedding.weight.copy_(torch.tensor([[100.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))

    def get_input_embeddings(self):
        return self.text_embeddings

    def prepare_embeddings_for_image_generation(self, input_ids):
        return self.model.generation_embeddings(input_ids)

    def decode_image_tokens(self, input_ids):
        assert input_ids.shape == (1, 4)
        return torch.zeros((1, 2, 2, 3), device=input_ids.device)


def test_teacher_forced_logits_match_one_code_at_a_time():
    model = TinyJanus().eval()
    task = JanusGenerationTask(model, TinyJanusProcessor())
    targets = torch.tensor([[0, 2, 1, 0]])
    prepared = task.prepare("a portrait", targets)
    assert prepared.input_ids.tolist() == [[0, 2, 4]]
    assert prepared.target_ids.tolist() == [[0, 2, 1, 0]]
    assert task.prepare_prompt("a portrait")["input_ids"].equal(prepared.input_ids)

    prefix = model.get_input_embeddings()(prepared.input_ids)
    expected = []
    for code in targets[0]:
        hidden = model.model.language_model(
            inputs_embeds=prefix,
            attention_mask=torch.ones(prefix.shape[:2], dtype=torch.long),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        expected.append(model.model.generation_head(hidden[:, -1]))
        prefix = torch.cat((prefix, model.prepare_embeddings_for_image_generation(code.reshape(1, 1))), dim=1)
    expected = torch.stack(expected, dim=1)

    logits = task.image_logits(prepared)
    torch.testing.assert_close(logits, expected)
    loss = task.loss(prepared)
    torch.testing.assert_close(loss, F.cross_entropy(expected[0], targets[0]))
    total, count = task.score(prepared)
    assert count == 4 and total == pytest.approx(-loss.item() * count)

    loss.backward()
    assert torch.count_nonzero(model.model.language_model.linear.weight.grad)
    assert torch.count_nonzero(model.model.generation_head.weight.grad)
    assert all(parameter.grad is None for parameter in model.model.vqmodel.parameters())


def test_reference_quantization_uses_native_janus_normalized_codebook(tmp_path):
    images = pytest.importorskip("PIL.Image")
    path = tmp_path / "portrait.png"
    images.new("RGB", (2, 2), (190, 138, 128)).save(path)
    model = TinyJanus().eval()
    task = JanusGenerationTask(model, TinyJanusProcessor())

    grad_modes = []
    handle = model.model.vqmodel.encoder.register_forward_hook(
        lambda *args: grad_modes.append(torch.is_grad_enabled())
    )
    try:
        codes = task.encode_image(path)
    finally:
        handle.remove()
    # Raw Euclidean distance would prefer code 1 because code 0 has a much
    # larger norm. The original Janus cosine assignment selects code 0.
    assert codes.tolist() == [[0, 0, 0, 0]]
    assert grad_modes == [False]
    assert codes.device.type == "cpu" and codes.dtype == torch.long
    assert task.decode_image(codes).size == (2, 2)


def test_invalid_janus_image_codes_and_prompt_fail_fast():
    task = JanusGenerationTask(TinyJanus().eval(), TinyJanusProcessor())
    with pytest.raises(ValueError, match="non-empty text prompt"):
        task.prepare_prompt(" ")
    with pytest.raises(ValueError, match="shape"):
        task.prepare("a portrait", torch.tensor([0, 1, 2]))
    with pytest.raises(ValueError, match="outside the checkpoint"):
        task.prepare("a portrait", torch.tensor([0, 1, 2, 3]))
