"""GRADIEND training on paired image–text continuation examples."""

from collections import defaultdict
from math import isfinite
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.dataset import GradientTrainingDataset
from gradiend.trainer.core.signals import require_single_gradient_signal
from gradiend.trainer.trainer import Trainer
from gradiend.util import normalize_split_name

from .config import VisionLanguagePredictionConfig
from .data import expand_pairs, load_matched_examples, load_neutral_examples
from .evaluation import crossed_examples, neutral_mean_log_probability, paired_scores, score_shift, select_edit
from .model import VisionLanguageModelWithGradiend


class _PairDataset:
    batch_size = 1

    def __init__(self, pairs, model, target_classes, *, factual_only=False):
        self.pairs = pairs
        self.model = model
        self.classes = {-1: target_classes[0], 1: target_classes[1]}
        self.factual_only = factual_only
        labels = [self.classes[pair.label] for pair in pairs]
        # Pre-pruning can stratify the metadata before opening any images.
        self.data = pd.DataFrame({"factual_id": labels, "feature_class_id": labels})

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        row_id = f"{pair.group_id}/{pair.axis}/{pair.label:+d}"
        item = {
            "factual": self.model.prepare_example(pair.factual, row_id=row_id),
            "label": pair.label,
            "factual_id": self.classes[pair.label],
            "alternative_id": self.classes[-pair.label],
            "feature_class_id": self.classes[pair.label],
            "data_split": pair.split,
            "counterfactual_axis": pair.axis,
            "group_id": pair.group_id,
        }
        if not self.factual_only:
            item["alternative"] = self.model.prepare_example(pair.alternative, row_id=row_id)
        return item


class VisionLanguagePredictionTrainer(Trainer):
    """Learn a feature from image swaps, answer swaps, or both within matched pairs."""

    def __init__(
        self,
        model,
        *,
        data,
        config: VisionLanguagePredictionConfig,
        args: Optional[TrainingArguments] = None,
        eval_neutral_data=None,
    ):
        if not isinstance(config, VisionLanguagePredictionConfig):
            raise TypeError("config must be a VisionLanguagePredictionConfig")
        if args is not None and not isinstance(args, TrainingArguments):
            raise TypeError("args must be a TrainingArguments instance")
        if args is None:
            args = TrainingArguments(source="factual", target="diff", base_gradient_batch_size=1)
        # The base trainer orders binary classes positive-first; the manifest is A-negative.
        super().__init__(model, target_classes=list(reversed(config.target_classes)), args=args)
        self.config = config
        self._matches = load_matched_examples(data)
        self._neutral = [] if eval_neutral_data is None else load_neutral_examples(eval_neutral_data)
        match_images = {example.image for match in self._matches for example in (match.negative, match.positive)}
        match_groups = {match.group_id for match in self._matches}
        for row in self._neutral:
            if row.group_id in match_groups or row.example.image in match_images:
                raise ValueError(f"Neutral group {row.group_id!r} overlaps the matched manifest")

    @property
    def default_model_with_gradiend_cls(self):
        return VisionLanguageModelWithGradiend

    @property
    def vision_language_model_kwargs(self):
        return {
            "parameter_scope": self.config.parameter_scope,
            "max_length": self.config.max_length,
            "max_pixels": self.config.max_pixels,
            "revision": self.config.revision,
            "base_model_dtype": self.config.base_model_dtype,
        }

    def _configure_model(self, model):
        if not isinstance(model, VisionLanguageModelWithGradiend):
            raise TypeError("VisionLanguagePredictionTrainer requires a VisionLanguageModelWithGradiend")
        if model.parameter_scope != self.config.parameter_scope:
            raise ValueError("The model parameter scope differs from the trainer configuration")
        classes = self.config.target_classes
        recorded = model.gradiend.kwargs.get("vision_language_target_classes")
        if recorded is not None and tuple(recorded) != classes:
            raise ValueError("The manifest class order differs from the VLM checkpoint")
        model.gradiend.kwargs["vision_language_target_classes"] = list(classes)
        model.set_feature_class_encoding_direction({classes[0]: -1, classes[1]: 1})

    def get_target_feature_class_ids(self):
        return self.target_classes

    def _resolve_training_args(self, overrides):
        args = super()._resolve_training_args(overrides)
        require_single_gradient_signal(signal=args.signal, signals=args.signals, context=type(self).__name__)
        if args.input_signal is not None:
            raise ValueError("Vision–language feature training currently uses one parameter-gradient signal")
        if args.source != "factual" or args.target != "diff":
            raise ValueError("Vision–language prediction requires source='factual' and target='diff'")
        if args.base_gradient_batch_size != 1:
            raise ValueError("Vision–language prediction requires base_gradient_batch_size=1")
        if args.torch_dtype != torch.float32:
            raise ValueError("Vision–language feature training requires torch_dtype=torch.float32")
        if args.use_cache or args.use_cached_gradients:
            raise ValueError("Vision–language prediction requires use_cache=False and use_cached_gradients=False")
        if args.do_eval and args.eval_steps > 0 and not any(row.split == "validation" for row in self._matches):
            raise ValueError("In-training evaluation requires matched rows in the validation split")
        return args

    def _resolve_artifact_use_cache(self, value, *, fallback=False):
        if super()._resolve_artifact_use_cache(value, fallback=fallback):
            raise ValueError("Vision–language evaluation requires use_cache=False")
        return False

    def _matched_rows(self, split, max_size):
        if isinstance(split, str):
            split = [split]
        wanted = {normalize_split_name(name) for name in split}
        if "all" in wanted:
            wanted = {row.split for row in self._matches}
        available = {row.split for row in self._matches}
        if not wanted or wanted - available:
            raise ValueError(f"Requested matched splits {sorted(wanted)} are unavailable; found {sorted(available)}")
        if max_size is not None and (type(max_size) is not int or max_size < 0):
            raise ValueError("max_size must be a non-negative integer")
        matches = [row for row in self._matches if row.split in wanted]
        return matches[:max_size]

    def create_training_data(
        self, model_with_gradiend, *, split="train", batch_size=1, max_size=None, factual_only=False, **kwargs,
    ):
        self._configure_model(model_with_gradiend)
        if (model_with_gradiend.source, model_with_gradiend.target) != ("factual", "diff"):
            raise ValueError("The VLM adapter must use factual source and difference target")
        if batch_size != 1:
            raise ValueError("Vision–language prediction prepares one example per gradient calculation")
        axis = "image" if factual_only else self.config.counterfactual_axis
        pairs = expand_pairs(self._matched_rows(split, max_size), axis)
        return _PairDataset(pairs, model_with_gradiend, self.config.target_classes, factual_only=factual_only)

    def create_eval_data(self, model_with_gradiend, *args, **kwargs):
        # The fitted axis does not change the matched examples used for monitoring.
        return super().create_eval_data(model_with_gradiend, *args, factual_only=True, **kwargs)

    def create_gradient_training_dataset(
        self,
        training_data,
        model_with_gradiend,
        *,
        source="factual",
        target="diff",
        use_cached_gradients=False,
        cache_dir=None,
        **kwargs,
    ):
        if cache_dir is not None or use_cached_gradients:
            raise ValueError("Vision–language prediction requires gradient caching to be disabled")
        if source != "factual" or target not in ("diff", None):
            raise ValueError("Vision–language gradients require factual input and an optional difference target")
        return GradientTrainingDataset(
            training_data,
            model_with_gradiend.gradient_creator,
            source=source,
            target=target,
            use_cached_gradients=False,
            **{
                key: kwargs[key]
                for key in ("dtype", "device", "timing_steps", "timing_label", "signal", "signals")
                if key in kwargs
            },
        )

    def _analyze_encoder(self, model_with_gradiend, split="test", max_size=None, **kwargs):
        if kwargs.get("source") not in ("factual", None):
            raise ValueError("Vision–language encoder evaluation uses factual gradients")
        if kwargs.get("pre_load_gradients"):
            raise ValueError("Vision–language encoder evaluation requires pre_load_gradients=False")
        if kwargs.get("neutral_data_df") is not None:
            raise ValueError("Supply the neutral image–text manifest as eval_neutral_data at initialization")
        matches = self._matched_rows(split, max_size)
        rows = []
        negative, positive = self.config.target_classes

        def encode(example, group_id):
            inputs = model_with_gradiend.prepare_example(example, row_id=group_id)
            gradient = model_with_gradiend.gradient_creator(inputs)
            return model_with_gradiend.encode(gradient, return_float=True)

        for match in matches:
            a_a, a_b, b_a, b_b = (encode(example, match.group_id) for example in crossed_examples(match))
            for encoded, image_swap, text_swap, label, source_id, target_id, example in (
                (a_a, b_a, a_b, -1, negative, positive, match.negative),
                (b_b, a_b, b_a, 1, positive, negative, match.positive),
            ):
                rows.append({
                    "encoded": encoded,
                    "encoded_image_swap": image_swap,
                    "encoded_text_swap": text_swap,
                    "label": label,
                    "source_id": source_id,
                    "target_id": target_id,
                    "feature_class_id": source_id,
                    "type": "training",
                    "data_split": match.split,
                    "group_id": match.group_id,
                    "image": str(example.image),
                    "prompt": example.prompt,
                    "answer": example.answer,
                })
        selected_splits = {match.split for match in matches}
        neutral_rows = [row for row in self._neutral if row.split in selected_splits][:max_size]
        for row in neutral_rows:
            rows.append({
                "encoded": encode(row.example, row.group_id),
                "label": 0,
                "source_id": "neutral",
                "target_id": "neutral",
                "feature_class_id": "neutral",
                "type": "neutral_dataset",
                "data_split": row.split,
                "group_id": row.group_id,
                "image": str(row.example.image),
                "prompt": row.example.prompt,
                "answer": row.example.answer,
            })
        return pd.DataFrame.from_records(rows)

    def evaluate_encoder(self, *args, return_df=False, **kwargs):
        """Return standard encoder metrics and effects of the two controlled swaps."""
        result = super().evaluate_encoder(*args, return_df=True, **kwargs)
        frame = result["encoder_df"]
        if all(f"encoded_{axis}_swap" in frame.columns for axis in ("image", "text")):
            effects = defaultdict(dict)
            for name, rows in frame[frame["type"] == "training"].groupby("source_id"):
                for axis in ("image", "text"):
                    effects[axis][name] = float((rows[f"encoded_{axis}_swap"] - rows["encoded"]).mean())
            result["mean_swap_shift_by_factual_class"] = dict(effects)
        if not return_df:
            del result["encoder_df"]
        return result

    def _decoder_rows(self, split, max_size, max_size_neutral):
        matches = self._matched_rows(split, max_size)
        neutral = [row for row in self._neutral if row.split == split][:max_size_neutral]
        if not matches or not neutral:
            raise ValueError(f"Decoder evaluation requires non-empty matched and neutral examples in {split!r}")
        return matches, neutral

    def _get_decoder_eval_dataframe(self, *args, **kwargs):
        """The text decoder evaluator's frame contract is specific to its scorer."""
        raise NotImplementedError("Use evaluate_decoder() for paired image–text scoring and preservation evaluation.")

    def _get_decoder_eval_targets(self):
        """Return observed assistant answers for each concept class."""
        negative, positive = self.config.target_classes
        return {
            negative: sorted({match.negative.answer for match in self._matches}),
            positive: sorted({match.positive.answer for match in self._matches}),
        }

    def evaluate_base_model(
        self, model, tokenizer=None, use_cache=None, *, split="test", max_size=None, max_size_neutral=None,
    ):
        """Score the four conditions and neutral continuations for one base model."""
        self._resolve_artifact_use_cache(use_cache)
        matches, neutral = self._decoder_rows(normalize_split_name(split), max_size, max_size_neutral)
        adapter = self.get_model()

        def score(example, group_id):
            return adapter.score_example(example, base_model=model, row_id=group_id)

        def candidate_score(example, group_id):
            log_probability, tokens = score(example, group_id)
            return log_probability / tokens

        return {
            **paired_scores(matches, candidate_score),
            "neutral_mean_log_probability": neutral_mean_log_probability(neutral, score),
        }

    def evaluate_decoder(
        self,
        lrs=None,
        feature_factors=None,
        *,
        max_size=None,
        max_size_neutral=None,
        use_cache=None,
    ):
        """Select an edit on validation data and report its held-out test effect.

        Signed ``lrs`` scale the full decoder output, including its bias.
        ``feature_factors`` choose inputs to the decoder.
        """
        self._resolve_artifact_use_cache(use_cache)
        lrs = self._default_from_training_args(
            lrs, "decoder_eval_lrs", fallback=(-0.1, -0.01, -0.001, 0.001, 0.01, 0.1)
        )
        feature_factors = self._default_from_training_args(
            feature_factors, "decoder_eval_feature_factors", fallback=(-1.0, 1.0)
        )
        lrs, feature_factors = tuple(lrs), tuple(feature_factors)
        if not lrs or any(not isfinite(lr) or lr == 0 for lr in lrs):
            raise ValueError("lrs must contain finite nonzero numbers")
        if not feature_factors or any(not isfinite(factor) for factor in feature_factors):
            raise ValueError("feature_factors must contain finite numbers")
        for cap in (max_size, max_size_neutral):
            if cap is not None and (type(cap) is not int or cap < 1):
                raise ValueError("Decoder evaluation size caps must be positive integers")
        for split in ("validation", "test"):
            self._decoder_rows(split, max_size, max_size_neutral)
        adapter = self.get_model()
        score_kwargs = {"max_size": max_size, "max_size_neutral": max_size_neutral, "use_cache": False}
        baseline = self.evaluate_base_model(adapter.base_model, split="validation", **score_kwargs)
        grid = []
        for lr in sorted(set(lrs)):
            for factor in sorted(set(feature_factors)):
                with adapter.exclusive_base_gradient_access(), adapter.intervene(value=lr, feature_factor=factor):
                    scores = self.evaluate_base_model(adapter.base_model, split="validation", **score_kwargs)
                grid.append({"learning_rate": lr, "feature_factor": factor, **scores, **score_shift(baseline, scores)})
        selected = select_edit(grid, self.config.min_neutral_likelihood_ratio)
        test_baseline = self.evaluate_base_model(adapter.base_model, split="test", **score_kwargs)
        test_edited = None
        if selected is not None:
            with adapter.exclusive_base_gradient_access(), adapter.intervene(
                value=selected["learning_rate"], feature_factor=selected["feature_factor"]
            ):
                scores = self.evaluate_base_model(adapter.base_model, split="test", **score_kwargs)
            test_edited = {**scores, **score_shift(test_baseline, scores)}
        return {
            "baseline": baseline,
            "grid": grid,
            "selected": selected,
            "test": {"baseline": test_baseline, "edited": test_edited},
        }

    def evaluate(self, *, kwargs_encoder=None, kwargs_decoder=None, **kwargs):
        """Run the paired encoder and decoder evaluations together."""
        return {
            "encoder": self.evaluate_encoder(**{**(kwargs_encoder or {}), **kwargs}),
            "decoder": self.evaluate_decoder(**{**(kwargs_decoder or {}), **kwargs}),
        }

    def rewrite_base_model(self, decoder_results, *, output_dir=None):
        """Apply the validation-selected edit and optionally export it with its processor."""
        selected = decoder_results["selected"]
        if selected is None:
            raise ValueError("No validation edit qualified for rewriting")
        adapter = self.get_model()
        edit = {"learning_rate": selected["learning_rate"], "feature_factor": selected["feature_factor"]}
        if output_dir is None:
            return adapter.rewrite_base_model(**edit)
        if isinstance(output_dir, str) and not output_dir.strip():
            raise ValueError("output_dir must be a non-empty path")
        directory = Path(output_dir)
        with adapter.exclusive_base_gradient_access(), adapter.intervene(
            value=edit["learning_rate"], feature_factor=edit["feature_factor"]
        ):
            adapter.base_model.save_pretrained(directory)
            adapter.processor.save_pretrained(directory)
        return str(directory)
