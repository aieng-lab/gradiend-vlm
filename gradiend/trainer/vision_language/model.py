"""Parameter-gradient GRADIEND adapter for image-conditioned prediction."""

from __future__ import annotations

from collections.abc import Mapping
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from gradiend.model.core.backbone import _filter_params_by_include
from gradiend.model.model_with_gradiend import ModelWithGradiend
from gradiend.signal_space import resolve_signal_training_plan, scope_params
from gradiend.trainer.core.signals import SignalScope
from gradiend.util import unwrap_model

from .data import VisionLanguageExample
from .task import DEFAULT_MAX_LENGTH, DEFAULT_MAX_PIXELS, PreparedContinuation, VisionLanguageContinuationTask

PARAMETER_SCOPES = ("language", "vision", "connector")
_BASE_MODEL_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
_CHECKPOINT_SETTINGS = "vision_language"
_PROCESSOR_DIR = "processor"


def _validate_maximum(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class _ModelSettings:
    parameter_scope: str
    max_length: int
    max_pixels: int
    revision: Optional[str]
    resolved_revision: Optional[str]
    base_model_dtype: str

    def __post_init__(self) -> None:
        if self.parameter_scope not in PARAMETER_SCOPES:
            raise ValueError(f"parameter_scope must be one of {PARAMETER_SCOPES}; got {self.parameter_scope!r}.")
        _validate_maximum("max_length", self.max_length)
        _validate_maximum("max_pixels", self.max_pixels)
        if self.base_model_dtype not in _BASE_MODEL_DTYPES:
            raise ValueError(f"base_model_dtype must be one of {tuple(_BASE_MODEL_DTYPES)}.")


def _parameter_ids_for_scope(model: nn.Module, parameter_scope: str) -> set[int]:
    """Resolve Qwen's disjoint language, vision and visual-merger parameter scopes."""
    model = unwrap_model(model)
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen2_5_vl":
        raise ValueError("Vision-language prediction supports Qwen2.5-VL conditional-generation models.")
    if parameter_scope not in PARAMETER_SCOPES:
        raise ValueError(f"parameter_scope must be one of {PARAMETER_SCOPES}; got {parameter_scope!r}.")

    try:
        language = model.model.language_model
        vision = model.model.visual
        connector = vision.merger
        head = model.lm_head
    except AttributeError as exc:
        raise ValueError("Expected a full Qwen2.5-VL conditional-generation model with its language head.") from exc

    connector_ids = {id(parameter) for parameter in connector.parameters()}
    if parameter_scope == "connector":
        return connector_ids
    if parameter_scope == "vision":
        return {id(parameter) for parameter in vision.parameters()} - connector_ids
    return {id(parameter) for parameter in language.parameters()} - {id(parameter) for parameter in head.parameters()}


class VisionLanguageModelWithGradiend(ModelWithGradiend):
    """Attach GRADIEND to scoped parameters of a full vision-language model."""

    def __init__(self, base_model, gradiend, processor, settings: _ModelSettings, **kwargs: Any):
        self._settings = settings
        eligible = _parameter_ids_for_scope(base_model, settings.parameter_scope)
        eligible_names = {name for name, parameter in base_model.named_parameters() if id(parameter) in eligible}
        if gradiend.signal_kind == "gradient":
            out_of_scope = set(gradiend.param_map) - eligible_names
            if out_of_scope:
                raise ValueError(
                    f"VLM checkpoint map contains parameters outside {settings.parameter_scope}: {sorted(out_of_scope)}"
                )
        super().__init__(base_model, gradiend, **kwargs)
        self.processor = processor
        self._task = VisionLanguageContinuationTask(
            processor, max_length=settings.max_length, max_pixels=settings.max_pixels
        )
        self.base_model.eval()

    @property
    def parameter_scope(self) -> str:
        return self._settings.parameter_scope

    @property
    def gradient_creator(self):
        creator = super().gradient_creator
        return self.forward if creator is self else creator

    def train(self, mode: bool = True):
        super().train(mode)
        # Gradients remain enabled while the frozen VLM uses deterministic inference behavior.
        self.base_model.eval()
        return self

    def to(self, device=None, *, torch_dtype=None):
        """Move the adapter while keeping base-model and GRADIEND precision independent."""
        if device is not None:
            super().to(device)
        if torch_dtype is not None:
            self.gradiend.to(torch_dtype=torch_dtype)
        return self

    def prepare_example(self, example: VisionLanguageExample, *, row_id: Any = None) -> PreparedContinuation:
        with self.exclusive_base_gradient_access():
            return self._task.prepare_example(example, row_id=row_id)

    def prepare_signal_inputs(
        self, example: VisionLanguageExample, *, include_answer: bool, row_id: Any = None,
    ) -> dict[str, Any]:
        """Prepare an input for native signal extractors at the causal prediction position."""
        with self.exclusive_base_gradient_access():
            return self._task.prepare_signal_inputs(example, include_answer=include_answer, row_id=row_id)

    def create_gradients(
        self,
        example: VisionLanguageExample,
        return_dict: bool = False,
        *,
        target_device=None,
        row_id: Any = None,
    ):
        with self.exclusive_base_gradient_access():
            inputs = self.prepare_example(example, row_id=row_id)
            return self.forward(inputs, target_device=target_device, return_dict=return_dict)

    def forward(self, inputs, *, target_device=None, return_dict: bool = False):
        if not self.uses_gradients:
            raise ValueError("Use ActivationSignalExtractor for an activation-space VLM adapter.")
        with self.exclusive_base_gradient_access(), torch.enable_grad():
            model = self._get_base_forward_model()
            self._zero_base_grad()
            try:
                if isinstance(inputs, Mapping):
                    model_inputs = {
                        key: value for key, value in inputs.items()
                        if key not in {"prediction_mask", "prediction_mask_is_generating"}
                    }
                    if "labels" not in model_inputs:
                        raise ValueError("VLM parameter gradients require assistant answer labels.")
                    model_inputs["return_dict"] = True
                    loss = model(**self._place_inputs_for_base_forward(model_inputs)).loss
                else:
                    scores = self._task.token_log_probabilities(model, inputs, device=self._get_base_forward_device())
                    loss = -scores.mean()
                if loss is None or not bool(torch.isfinite(loss)):
                    raise ValueError("VLM parameter gradients require a finite assistant continuation loss.")
                loss.backward()
                return self.gradiend.extract_gradients(
                    model, return_dict=return_dict,
                    target_device=self.gradiend.device_encoder if target_device is None else target_device,
                )
            finally:
                self._zero_base_grad()

    def encode(self, input, label=None, return_float: bool = False):
        if isinstance(input, VisionLanguageExample):
            input = self.create_gradients(input)
        return super().encode(input, label=label, return_float=return_float)

    @torch.no_grad()
    def score_example(
        self, example: VisionLanguageExample, *, base_model=None, row_id: Any = None,
        mask_prediction_position: bool = False,
    ) -> tuple[float, int]:
        """Return summed answer log probability and the supervised token count.

        Set ``mask_prediction_position`` when an external activation steering
        pre-hook consumes the explicit VLM prediction mask.
        """
        with self.exclusive_base_gradient_access():
            model = self._get_base_forward_model() if base_model is None else base_model
            inputs = self.prepare_example(example, row_id=row_id)
            device = next(model.parameters()).device
            was_training = model.training
            model.eval()
            try:
                intervention = getattr(self, "active_intervention_metadata", None) or {}
                scopes_prediction = (
                    intervention.get("signal") == "activation"
                    and intervention.get("token_selector") == "prediction"
                    and intervention.get("active")
                )
                scores = self._task.token_log_probabilities(
                    model,
                    inputs,
                    device=device,
                    mask_prediction_position=mask_prediction_position or bool(scopes_prediction),
                )
            finally:
                model.train(was_training)
            return scores.sum().item(), scores.numel()

    def _save_model(self, save_directory, **kwargs):
        self.processor.save_pretrained(Path(save_directory) / _PROCESSOR_DIR)
        self.gradiend.kwargs[_CHECKPOINT_SETTINGS] = asdict(self._settings)

    @classmethod
    def from_pretrained(cls, load_directory, **kwargs):
        args = kwargs.get("training_args")
        argument_map = args.get("param_map") if isinstance(args, dict) else getattr(args, "param_map", None)
        if kwargs.get("param_map") or argument_map:
            raise ValueError("Use TrainingArguments.signal_scope for VLM parameter selection.")
        definition = kwargs.get("feature_definition")
        if definition is None:
            definition = kwargs.get("definition")
        model_settings = getattr(definition, "vision_language_model_kwargs", {})
        for name, value in model_settings.items():
            kwargs.setdefault(name, value)
        return super().from_pretrained(load_directory, **kwargs)

    @classmethod
    def _load_model(
        cls,
        load_directory,
        base_model_id=None,
        gradiend_kwargs=None,
        base_model=None,
        *,
        processor=None,
        parameter_scope=None,
        max_length=None,
        max_pixels=None,
        revision=None,
        base_model_dtype=None,
        base_model_device_map=None,
        trust_remote_code=False,
        **kwargs,
    ) -> tuple:
        if base_model_device_map not in (None, False):
            raise ValueError("The VLM adapter supports a single base-model device.")
        if trust_remote_code:
            raise ValueError("The Qwen2.5-VL adapter uses the native Transformers implementation.")
        if base_model_dtype is not None and base_model_dtype not in _BASE_MODEL_DTYPES:
            raise ValueError(f"base_model_dtype must be one of {tuple(_BASE_MODEL_DTYPES)}.")

        settings = None
        if base_model_id is not None:
            try:
                settings = _ModelSettings(**gradiend_kwargs[_CHECKPOINT_SETTINGS])
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Checkpoint {load_directory!r} lacks valid vision-language settings.") from exc
            for name, value in (
                ("parameter_scope", parameter_scope), ("max_length", max_length), ("max_pixels", max_pixels),
                ("base_model_dtype", base_model_dtype),
            ):
                if value is not None and value != getattr(settings, name):
                    raise ValueError(f"Requested {name}={value!r} conflicts with the VLM checkpoint.")
            if revision is not None and revision not in (settings.revision, settings.resolved_revision):
                raise ValueError(f"Requested revision {revision!r} conflicts with the VLM checkpoint.")
            if processor is not None:
                raise ValueError("The VLM checkpoint already contains its processor.")
            processor_path = Path(load_directory) / _PROCESSOR_DIR
            if not processor_path.is_dir():
                raise FileNotFoundError(f"VLM checkpoint processor is missing at {processor_path}.")
            processor_revision = None
            model_revision = settings.resolved_revision
        else:
            if parameter_scope is not None and parameter_scope not in PARAMETER_SCOPES:
                raise ValueError(f"parameter_scope must be one of {PARAMETER_SCOPES}; got {parameter_scope!r}.")
            if max_length is not None:
                _validate_maximum("max_length", max_length)
            if max_pixels is not None:
                _validate_maximum("max_pixels", max_pixels)
            processor_path = load_directory if isinstance(load_directory, str) else load_directory.name_or_path
            processor_revision = revision
            model_revision = revision

        base_model_path = base_model_id if base_model_id is not None else processor_path
        if base_model is None and not isinstance(load_directory, str):
            base_model = load_directory
        if base_model is None:
            from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

            requested_dtype = settings.base_model_dtype if settings is not None else (base_model_dtype or "bfloat16")
            config = AutoConfig.from_pretrained(base_model_path, revision=model_revision)
            if config.model_type != "qwen2_5_vl":
                raise ValueError(f"Expected a Qwen2.5-VL checkpoint, got model type {config.model_type!r}.")
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_path, config=config, revision=model_revision, dtype=_BASE_MODEL_DTYPES[requested_dtype]
            )
        elif base_model_id is not None:
            if base_model.name_or_path != base_model_id:
                raise ValueError(f"VLM checkpoint expects base model {base_model_id!r}.")
            observed_revision = getattr(base_model.config, "_commit_hash", None)
            if model_revision is not None and observed_revision != model_revision:
                raise ValueError(f"VLM checkpoint expects base model revision {model_revision!r}.")
        if getattr(base_model, "hf_device_map", None):
            raise ValueError("The VLM adapter supports a single base-model device.")
        observed_dtype = str(next(base_model.parameters()).dtype).removeprefix("torch.")
        expected_dtype = settings.base_model_dtype if settings is not None else base_model_dtype
        if expected_dtype is not None and observed_dtype != expected_dtype:
            raise ValueError(f"VLM base model has dtype {observed_dtype!r}; expected {expected_dtype!r}.")
        if settings is None:
            resolved_revision = getattr(base_model.config, "_commit_hash", None) or revision
            settings = _ModelSettings(
                "language" if parameter_scope is None else parameter_scope,
                DEFAULT_MAX_LENGTH if max_length is None else max_length,
                DEFAULT_MAX_PIXELS if max_pixels is None else max_pixels,
                revision,
                resolved_revision,
                observed_dtype,
            )
            processor_revision = resolved_revision
        _parameter_ids_for_scope(base_model, settings.parameter_scope)
        if processor is None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(processor_path, revision=processor_revision)
        return base_model, processor, settings

    @classmethod
    def _create_gradiend(
        cls, base_model, load_directory, *, signal_plan=None, parameter_scope=None, params=None, **kwargs,
    ):
        if signal_plan is None:
            signal_plan = resolve_signal_training_plan(base_model)
        space = signal_plan.single_space
        if space.kind == "gradient":
            patterns = scope_params(space.scope, base_model=base_model)
            if patterns is not None and params is not None and tuple(params) != patterns:
                raise ValueError("params conflicts with signal_scope.params.")
            patterns = patterns if patterns is not None else params
            if not patterns:
                raise ValueError("Specify TrainingArguments.signal_scope with params for VLM parameter gradients.")
            eligible = _parameter_ids_for_scope(base_model, "language" if parameter_scope is None else parameter_scope)
            parameters = OrderedDict(
                (name, parameter) for name, parameter in base_model.named_parameters() if id(parameter) in eligible
            )
            selected = set()
            for pattern in patterns:
                matches = _filter_params_by_include(parameters, [pattern])
                if not matches:
                    raise ValueError(f"Parameter filter {pattern!r} matches nothing in the selected scope.")
                selected.update(matches)
            weight_scope = SignalScope.from_values(params=[name for name in parameters if name in selected], mode="full")
            signal_plan = replace(signal_plan, spaces=(replace(space, scope=weight_scope),), scope=weight_scope)

        return super()._create_gradiend(
            base_model, load_directory, signal_plan=signal_plan,
            **{
                name: kwargs[name]
                for name in (
                    "device_encoder", "device_decoder", "torch_dtype", "latent_dim",
                    "activation_encoder", "activation_decoder", "bias_encoder", "bias_decoder",
                    "init_fan_in_floor", "gradiend_split", "pre_prune_config",
                )
                if name in kwargs
            },
        )
