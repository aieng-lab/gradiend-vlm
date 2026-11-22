"""Image-conditioned assistant continuation for the Qwen2.5-VL adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from gradiend.trainer.core.callbacks import EvaluationInputError

from .data import VisionLanguageExample

DEFAULT_MAX_LENGTH = 1024
DEFAULT_MAX_PIXELS = 256 * 28 * 28


@dataclass(frozen=True)
class PreparedContinuation:
    model_inputs: dict[str, torch.Tensor]
    answer_start: int
    context: str

    def prediction_mask(self) -> torch.Tensor:
        mask = torch.zeros_like(self.model_inputs["input_ids"], dtype=torch.bool)
        mask[:, self.answer_start - 1] = True
        return mask


class VisionLanguageContinuationTask:
    """Prepare one image and text prompt, then score the assistant's answer tokens."""

    def __init__(self, processor: Any, *, max_length: int, max_pixels: int):
        self.processor = processor
        self.max_length = max_length
        self.max_pixels = max_pixels

    def prepare_example(self, example: VisionLanguageExample, *, row_id: Any = None) -> PreparedContinuation:
        return self._prepare_continuation(example, include_answer=True, row_id=row_id)

    def prepare_signal_inputs(
        self, example: VisionLanguageExample, *, include_answer: bool, row_id: Any = None,
    ) -> dict[str, Any]:
        prepared = self._prepare_continuation(example, include_answer=include_answer, row_id=row_id)
        inputs = dict(prepared.model_inputs)
        tokens = inputs["input_ids"]
        inputs.update(
            prediction_mask=prepared.prediction_mask(),
            prediction_mask_is_generating=True,
            use_cache=False,
            return_dict=True,
        )
        if include_answer:
            labels = torch.full_like(tokens, -100)
            labels[:, prepared.answer_start:] = tokens[:, prepared.answer_start:]
            inputs["labels"] = labels
        else:
            inputs["logits_to_keep"] = 1
        return inputs

    def _prepare_continuation(
        self, example: VisionLanguageExample, *, include_answer: bool, row_id: Any,
    ) -> PreparedContinuation:
        from PIL import Image, ImageOps

        context = f"image {example.image}" if row_id is None else f"row {row_id!r}, image {example.image}"
        try:
            with Image.open(example.image) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, Image.DecompressionBombError) as exc:
            raise EvaluationInputError(f"Could not read VLM {context}: {exc}") from exc

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": example.prompt}],
            }
        ]
        try:
            prefix = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Run both strings through the same processor to locate the boundary
            # after the processor has expanded the image placeholder into patches.
            processor_kwargs = {
                "images": image,
                "add_special_tokens": False,
                "return_tensors": "pt",
                "max_pixels": self.max_pixels,
            }
            context_inputs = dict(self.processor(text=prefix, **processor_kwargs))
            model_inputs = (
                dict(self.processor(text=prefix + example.answer, **processor_kwargs))
                if include_answer else context_inputs
            )
        except (TypeError, ValueError) as exc:
            raise EvaluationInputError(f"Could not process VLM {context}: {exc}") from exc

        input_ids = model_inputs["input_ids"]
        prefix_ids = context_inputs["input_ids"]
        answer_start = prefix_ids.shape[1]
        length = input_ids.shape[1]
        if length > self.max_length:
            raise EvaluationInputError(f"VLM {context} has {length} tokens, exceeding max_length={self.max_length}.")
        if not answer_start or (
            include_answer and (answer_start >= length or not torch.equal(input_ids[:, :answer_start], prefix_ids))
        ):
            raise EvaluationInputError(f"Could not align answer tokens for VLM {context} after the assistant prompt.")
        if model_inputs["image_grid_thw"].shape != (1, 3) or not model_inputs["pixel_values"].numel():
            raise EvaluationInputError(f"Expected one processed image for VLM {context}.")
        if not bool(model_inputs["attention_mask"].all()):
            raise EvaluationInputError(f"Unexpected padding in the single-example VLM input for {context}.")

        if include_answer:
            answer_ids = input_ids[:, answer_start:]
            special_ids = input_ids.new_tensor(self.processor.tokenizer.all_special_ids)
            if bool(torch.isin(answer_ids, special_ids).any()):
                raise EvaluationInputError(f"VLM answer contains a special token for {context}.")
        return PreparedContinuation(model_inputs, answer_start, context)

    def token_log_probabilities(
        self,
        model: Any,
        prepared: PreparedContinuation,
        *,
        device: torch.device,
        mask_prediction_position: bool = False,
    ) -> torch.Tensor:
        """One log probability per answer token, preserving the computation graph."""
        model_inputs = {name: tensor.to(device) for name, tensor in prepared.model_inputs.items()}
        input_ids = model_inputs["input_ids"]
        answer_ids = input_ids[0, prepared.answer_start:]
        if mask_prediction_position:
            # The installed activation steering pre-hook consumes this field.
            model_inputs["prediction_mask"] = prepared.prediction_mask().to(device)
        # The token at position p is predicted by the logits at p - 1.
        generating_positions = torch.arange(prepared.answer_start - 1, input_ids.shape[1] - 1, device=device)
        outputs = model(**model_inputs, logits_to_keep=generating_positions, use_cache=False, return_dict=True)
        log_probabilities = -F.cross_entropy(outputs.logits[0].float(), answer_ids, reduction="none")
        if not bool(torch.isfinite(log_probabilities).all()):
            raise EvaluationInputError(f"Non-finite answer token probabilities for VLM {prepared.context}.")
        return log_probabilities
