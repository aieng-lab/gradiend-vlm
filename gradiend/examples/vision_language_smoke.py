"""Check image and text counterfactuals on the first pair in your own manifest.

From a source checkout with the ``vision`` extra installed, run:

    python -m gradiend.examples.vision_language_smoke pairs.json
    python -m gradiend.examples.vision_language_smoke pairs.json --task generation

Both tasks report all four combinations, using one scoped parameter gradient
per combination. Generation scores reference images as Janus visual codes.
Use ``--parse-only`` to validate the manifest before loading a model.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import isfinite
from pathlib import Path


_QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
_QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
_QWEN_PARAMETER = "model.visual.merger.mlp.2.bias"
_JANUS_MODEL = "deepseek-community/Janus-Pro-1B"
_JANUS_REVISION = "1655280bb75959cc1cb85529a2a8b26e7016072e"
_JANUS_PARAMETER = "model.language_model.norm.weight"
_CELLS = (
    "factual_image_factual_text",
    "factual_image_counterfactual_text",
    "counterfactual_image_factual_text",
    "counterfactual_image_counterfactual_text",
)


def _audit(cells, score, gradient):
    scores, gradients = {}, {}
    for key, item in cells.items():
        log_probability, count = score(item)
        if not isfinite(log_probability) or count < 1:
            raise ValueError(f"Invalid token score for {key}")
        vector = gradient(item).detach().float().cpu().flatten()
        if not vector.numel() or not bool(vector.isfinite().all()):
            raise ValueError(f"Invalid parameter gradient for {key}")
        scores[key] = {"mean_log_probability": log_probability / count, "tokens": count}
        gradients[key] = vector

    aa, ab, ba, bb = (gradients[key] for key in _CELLS)
    return {
        "scores": scores,
        "gradient_difference_l2": {
            "factual_anchor": {"image_swap": (aa - ba).norm().item(), "text_swap": (aa - ab).norm().item()},
            "counterfactual_anchor": {"image_swap": (bb - ab).norm().item(), "text_swap": (bb - ba).norm().item()},
        },
    }


def _prediction(match, device):
    from gradiend import SignalScope, TrainingArguments
    from gradiend.trainer.vision_language.evaluation import crossed_examples
    from gradiend.trainer.vision_language.model import VisionLanguageModelWithGradiend

    scope = SignalScope.from_values(params=[_QWEN_PARAMETER], mode="full")
    model = VisionLanguageModelWithGradiend.from_pretrained(
        _QWEN_MODEL, revision=_QWEN_REVISION, parameter_scope="connector",
        training_args=TrainingArguments(source="factual", target="diff", signal_scope=scope),
        device=device,
    )
    cells = dict(zip(_CELLS, reversed(crossed_examples(match))))
    return {
        "model": _QWEN_MODEL,
        "parameter": _QWEN_PARAMETER,
        "text_field": "answer",
        **_audit(cells, model.score_example, lambda example: model.create_gradients(example, target_device="cpu")),
    }


def _generation(pair, device):
    import torch
    from transformers import JanusForConditionalGeneration, JanusProcessor

    from gradiend.trainer.vision_language import JanusGenerationTask

    processor = JanusProcessor.from_pretrained(_JANUS_MODEL, revision=_JANUS_REVISION)
    model = JanusForConditionalGeneration.from_pretrained(
        _JANUS_MODEL, revision=_JANUS_REVISION, dtype=torch.bfloat16,
    ).to(device).eval()
    model.requires_grad_(False)
    parameter = model.get_parameter(_JANUS_PARAMETER).requires_grad_(True)
    task = JanusGenerationTask(model, processor)
    a, b = task.encode_image(pair.factual_image), task.encode_image(pair.counterfactual_image)
    p, q = pair.factual_prompt, pair.counterfactual_prompt
    cells = dict(zip(_CELLS, (task.prepare(p, a), task.prepare(q, a),
                             task.prepare(p, b), task.prepare(q, b))))

    def gradient(item):
        return torch.autograd.grad(task.loss(item), parameter)[0]

    return {
        "model": _JANUS_MODEL,
        "parameter": _JANUS_PARAMETER,
        "text_field": "prompt",
        **_audit(cells, task.score, gradient),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to your matched-pair manifest")
    parser.add_argument("--task", choices=("prediction", "generation"), default="prediction")
    parser.add_argument("--device", default="cuda", help="Device for the model, such as cuda or cuda:1")
    parser.add_argument("--parse-only", action="store_true", help="Validate the manifest without loading weights")
    args = parser.parse_args()

    if args.task == "prediction":
        from gradiend.trainer.vision_language.data import load_matched_examples

        pairs = load_matched_examples(args.manifest)
        measure = _prediction
    else:
        from gradiend.trainer.vision_language.data import load_generation_pairs

        pairs = load_generation_pairs(args.manifest)
        measure = _generation

    first = pairs[0]
    result = {
        "task": args.task,
        "pair_count": len(pairs),
        "pairs_by_split": dict(Counter(pair.split for pair in pairs)),
        "first_group_id": first.group_id,
    }
    if not args.parse_only:
        result.update(measure(first, args.device))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
