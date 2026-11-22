# Vision–language experiments

The [README](../../README.md) shows a complete one-pair JSON manifest and smoke commands. Use photos with and without an attribute, preferably of the same subject or scene and under similar capture conditions. Put the attribute-present image in `factual_image` and its matched reference in `counterfactual_image`. Both matching examples serve as factual anchors during training.

## Data fields

- Each row in `pairs` has `factual_image`, `counterfactual_image`, `group_id`, and `split`. Paths resolve relative to the JSON file. Splits are `train`, `validation`, and `test`. Use a shared group ID for repeated photos of the same person or scene. The loader rejects a group or image appearing in multiple splits.
- For **image understanding**, use `prompt`, `factual_answer`, and `counterfactual_answer`. Both photos receive the same question. The text swap changes the answer being scored.
- For **image generation**, use `factual_prompt` and `counterfactual_prompt`. Each prompt can score either target photo through Janus's visual-token loss. The text swap changes the prompt while keeping the target photo fixed.
- A manifest may include either text setup or both. Define the shared fields for each mode you use. Individual rows can override those defaults. Keep the wording consistent with the meaning of each photo.

Existing understanding datasets can still use CSV or a pandas DataFrame with `group_id`, `split`, `prompt`, `image_a`, `image_b`, `answer_a`, and `answer_b`. In this format side A is the negative class and side B the positive class.

## Fit a Qwen2.5-VL feature

Use your own subject-disjoint training, validation, and test photos. The trainer takes the JSON path as `data`. Order `target_classes` from the counterfactual image's class to the factual image's class.

```python
from gradiend import (
    PrePruneConfig,
    SignalScope,
    TrainingArguments,
    VisionLanguagePredictionConfig,
    VisionLanguagePredictionTrainer,
)

trainer = VisionLanguagePredictionTrainer(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    data="pairs.json",
    config=VisionLanguagePredictionConfig(
        target_classes=("no_glasses", "glasses"),
        counterfactual_axis="both",
        parameter_scope="connector",
        max_pixels=224 * 224,
        max_length=256,
    ),
    args=TrainingArguments(
        source="factual",
        target="diff",
        base_gradient_batch_size=1,
        gradiend_batch_size=8,
        signal_scope=SignalScope.from_values(
            params=["model.visual.merger.mlp.2.weight", "model.visual.merger.mlp.2.bias"],
        ),
        pre_prune_config=PrePruneConfig(n_samples=16, topk=65_536, source="diff", batch_size=1),
        learning_rate=0.003,
        num_train_epochs=4,
        eval_steps=50,
        experiment_dir="runs/vlm_glasses",
    ),
)
trainer.train()
print(trainer.evaluate_encoder(split="test"))
```

Choose `counterfactual_axis="image"` or `"text"` to train on one type of swap. Encoder and decoder effects are measured separately. For decoder evaluation, pass a separate `eval_neutral_data` CSV to the constructor with `group_id`, `split`, `image`, `prompt`, and `answer` columns. Include unrelated validation and test examples, with group IDs and image paths distinct from the paired data. The decoder evaluator chooses a signed edit on validation data and measures grounding and neutral continuation likelihood on test data.

## Score generation pairs with Janus

Import `load_generation_pairs` and `JanusGenerationTask` from `gradiend.trainer.vision_language`. The loader returns each pair's two resolved image paths and prompts. The task encodes each reference photo once with the frozen visual tokenizer. Use `task.prepare(prompt, codes)` and `task.loss(prepared)` for a differentiable mean cross-entropy over image tokens, or `task.score(prepared)` for summed conditional log probability and token count. The [smoke example](../../gradiend/examples/vision_language_smoke.py) shows how to obtain all four losses and compute both counterfactual gradient contrasts.

For prompt-only activation measurements, the activation at image start depends on the prompt. Image-target gradients can vary at that same position because the loss depends on the reference image tokens. Evaluate any generation edit on actual sampled images with held-out prompts, matched seeds, and controls.
