"""Teacher-forced image generation with the native Transformers Janus model."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any, TYPE_CHECKING

import torch
from torch.nn import functional as F

from gradiend.model.model_with_gradiend import ModelWithGradiend
from gradiend.trainer.core.callbacks import EvaluationInputError

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True)
class PreparedJanusGeneration:
    """One unpadded text prefix and one image code sequence, held on CPU."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_ids: torch.Tensor


class JanusGenerationTask:
    """Score reference image codes under Janus-Pro's conditional generation head.

    Pass a native Transformers JanusForConditionalGeneration model in evaluation
    mode and its matching JanusProcessor. The task does not update the base
    weights. Unlike native sampled generation, the teacher-forced objective
    computes the conditional logits before classifier-free guidance.
    """

    def __init__(self, model: Any, processor: Any):
        if isinstance(model, ModelWithGradiend):
            model = model.base_model
        if getattr(getattr(model, "config", None), "model_type", None) != "janus":
            raise TypeError("JanusGenerationTask requires the native Transformers Janus model.")
        self.model = model
        self.processor = processor
        self.vq = model.model.vqmodel
        self.num_image_tokens = int(model.config.vision_config.num_image_tokens)
        self.codebook_size = int(self.vq.config.num_embeddings)
        if self.num_image_tokens != self.vq.config.num_patches ** 2:
            raise ValueError("Janus generation length does not agree with the VQ code grid.")
        self._begin_image_token = int(model.generation_config.generation_kwargs["boi_token_id"])

    @torch.no_grad()
    def encode_image(self, path: str | PathLike) -> torch.Tensor:
        """Encode one reference image as a reusable CPU visual-token target."""
        from PIL import Image, ImageOps

        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, Image.DecompressionBombError) as exc:
            raise EvaluationInputError(f"Could not read Janus reference image {path}: {exc}") from exc

        pixels = self.processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
        reference = next(self.vq.encoder.parameters())
        pixels = pixels.to(device=reference.device, dtype=reference.dtype)
        features = self.vq.quant_conv(self.vq.encoder(pixels))
        # The original DeepSeek Janus VQ-16 uses L2-normalized encoder
        # vectors and codebook entries. Transformers 4.57.6's VQ encoder
        # omits this normalization; the image decoder still normalizes.
        patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).float()
        patches = F.normalize(patches, dim=-1)
        codebook = F.normalize(self.vq.quantize.embedding.weight.float(), dim=-1)
        # Both vectors have unit norm, so maximal cosine similarity is the
        # nearest entry under the original normalized Euclidean distance.
        target_ids = (patches @ codebook.T).argmax(dim=-1)
        return self._target_ids_cpu(target_ids)

    def prepare_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        """Build the same image-generation prefix used by the native sampler."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise EvaluationInputError("Janus generation requires a non-empty text prompt.")
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        formatted = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=formatted, generation_mode="image", return_tensors="pt")
        input_ids = inputs["input_ids"].detach().cpu()
        attention_mask = inputs["attention_mask"].detach().cpu()
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.shape[1]:
            raise EvaluationInputError("Janus generation requires one non-empty prompt.")
        if attention_mask.shape != input_ids.shape or not bool(attention_mask.all()):
            raise EvaluationInputError("Janus generation requires an unpadded single-example prompt.")
        if int(input_ids[0, -1]) != self._begin_image_token:
            raise EvaluationInputError("Janus generation prompt is missing the beginning-of-image token.")
        limit = self.model.config.text_config.max_position_embeddings
        if input_ids.shape[1] + self.num_image_tokens - 1 > limit:
            raise EvaluationInputError(f"Janus prompt and image codes exceed the {limit}-token context limit.")
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _target_ids_cpu(self, target_ids: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(target_ids) or target_ids.dtype != torch.long:
            raise TypeError("Janus image codes must be a torch.long tensor.")
        if target_ids.ndim == 1:
            target_ids = target_ids.unsqueeze(0)
        if target_ids.shape != (1, self.num_image_tokens):
            raise ValueError(f"Janus image codes must have shape (1, {self.num_image_tokens}).")
        target_ids = target_ids.detach().cpu()
        if bool(((target_ids < 0) | (target_ids >= self.codebook_size)).any()):
            raise ValueError("Janus image code is outside the checkpoint's VQ codebook.")
        return target_ids

    def prepare(self, prompt: str, target_ids: torch.Tensor) -> PreparedJanusGeneration:
        """Prepare one scored image continuation. `input_ids` contains only the prompt."""
        return PreparedJanusGeneration(**self.prepare_prompt(prompt), target_ids=self._target_ids_cpu(target_ids))

    def image_logits(self, prepared: PreparedJanusGeneration) -> torch.Tensor:
        """Return raw conditional logits at every image-code prediction position."""
        embeddings = self.model.get_input_embeddings()
        device = embeddings.weight.device
        prefix_ids = prepared.input_ids.to(device)
        target_ids = prepared.target_ids.to(device)
        prefix = embeddings(prefix_ids)
        image_context = self.model.prepare_embeddings_for_image_generation(target_ids[:, :-1])
        inputs_embeds = torch.cat((prefix, image_context), dim=1)
        prefix_mask = prepared.attention_mask.to(device)
        attention_mask = torch.cat((prefix_mask, prefix_mask.new_ones(image_context.shape[:2])), dim=1)
        hidden = self.model.model.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False, return_dict=True,
        ).last_hidden_state
        # BOI predicts code 0. Each subsequent input code predicts the next
        # code, including the final code which was held out from the input.
        return self.model.model.generation_head(hidden[:, prefix.shape[1] - 1 :, :])

    def _token_losses(self, prepared: PreparedJanusGeneration) -> torch.Tensor:
        logits = self.image_logits(prepared)[0]
        target = prepared.target_ids[0].to(logits.device)
        token_losses = F.cross_entropy(logits.float(), target, reduction="none")
        if not bool(torch.isfinite(token_losses).all()):
            raise EvaluationInputError("Janus image continuation has non-finite token loss.")
        return token_losses

    def loss(self, prepared: PreparedJanusGeneration) -> torch.Tensor:
        """Differentiable mean cross-entropy over the reference image codes."""
        return self._token_losses(prepared).mean()

    @torch.no_grad()
    def score(self, prepared: PreparedJanusGeneration) -> tuple[float, int]:
        """Return summed conditional log-probability and the image-code count."""
        token_losses = self._token_losses(prepared)
        return -token_losses.sum().item(), token_losses.numel()

    @torch.no_grad()
    def decode_image(self, target_ids: torch.Tensor) -> Image:
        """Decode a single target or generated code sequence using native Janus."""
        ids = self._target_ids_cpu(target_ids)
        device = self.vq.quantize.embedding.weight.device
        pixels = self.model.decode_image_tokens(ids.to(device))
        images = self.processor.postprocess(
            list(pixels.float().cpu()), return_tensors="PIL.Image.Image",
        )["pixel_values"]
        return images[0]
