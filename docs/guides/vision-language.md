# Vision–language experiments

The [README](../../README.md) shows a complete one-pair JSON manifest and smoke commands. Use photos with and without an attribute, preferably of the same subject or scene and under similar capture conditions. Put the attribute-present image in `factual_image` and its matched reference in `counterfactual_image`. Both matching examples serve as factual anchors during training.

## Data fields

- Each row in `pairs` has `factual_image`, `counterfactual_image`, `group_id`, and `split`. Paths resolve relative to the JSON file. Splits are `train`, `validation`, and `test`. Use a shared group ID for repeated photos of the same person or scene. The loader rejects a group or image appearing in multiple splits.
- For **image understanding**, use `prompt`, `factual_answer`, and `counterfactual_answer`. Both photos receive the same question. The text swap changes the answer being scored.
- For **image generation**, use `factual_prompt` and `counterfactual_prompt`. Each prompt can score either target photo through Janus's visual-token loss. The text swap changes the prompt while keeping the target photo fixed.
- A manifest may include either text setup or both. Define the shared fields for each mode you use. Individual rows can override those defaults. Keep the wording consistent with the meaning of each photo.

Existing understanding datasets can still use CSV or a pandas DataFrame with `group_id`, `split`, `prompt`, `image_a`, `image_b`, `answer_a`, and `answer_b`. In this format side A is the negative class and side B the positive class.

## Fit a Qwen2.5-VL feature

Use your own subject-disjoint training, validation, and test photos. The trainer takes the JSON path as `data`. For the README example, `target_classes=("no_glasses", "glasses")` means

- Index 0, `no_glasses`, labels the matching `counterfactual_image` and `counterfactual_answer` (`"no"`). This is the negative pole, with label −1.
- Index 1, `glasses`, labels the matching `factual_image` and `factual_answer` (`"yes"`). This is the positive pole, with label +1.

The class names are report and checkpoint identifiers. They can be named independently of the model's answer strings. The association comes from tuple order and the JSON field roles. With the glasses photo and answer `"yes"` as the factual anchor, the image swap keeps `"yes"` and uses the bare photo. The text swap keeps the glasses photo and scores `"no"`. The bare photo and answer `"no"` also serve as an anchor, with the swaps reversed.

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
        pre_prune_config=PrePruneConfig(n_samples=16, topk=0.05, source="diff", batch_size=1),
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

### Choose a pre-pruning budget

`SignalScope` sets which parameters are eligible. In this example it selects the final connector projection's weight and bias, which contain **10,487,808** coordinates in Qwen2.5-VL-3B-Instruct. A float `topk=0.05` keeps **5% of those eligible coordinates**, rounded up to **524,391**. An integer `topk=65_536` keeps that absolute count. Use `topk=1.0` or omit the pre-pruning configuration to retain the entire selected scope.

The pilots used a fixed **65,536-coordinate** budget to limit the width of the fitted feature and the size of stored gradients. That kept **0.625%** of Qwen's selected projection and **1.5625%** of Janus-Pro-1B's selected shared attention output projection. Relative to all tensor elements stored in each checkpoint, the budgets were approximately **0.00175%** and **0.00316%**, respectively. The full VLM still processes each example. The pilots did not establish an optimal budget. The 5% example shows how to set a comparable *fraction* when the scope or model changes.

## Score generation pairs with Janus

Import `load_generation_pairs` and `JanusGenerationTask` from `gradiend.trainer.vision_language`. The loader returns each pair's two resolved image paths and prompts. The task encodes each reference photo once with the frozen visual tokenizer. Use `task.prepare(prompt, codes)` and `task.loss(prepared)` for a differentiable mean cross-entropy over image tokens, or `task.score(prepared)` for summed conditional log probability and token count. The [smoke example](../../gradiend/examples/vision_language_smoke.py) shows how to obtain all four losses and compute both counterfactual gradient contrasts.

For prompt-only activation measurements, the activation at image start depends on the prompt. Image-target gradients can vary at that same position because the loss depends on the reference image tokens. Evaluate any generation edit on actual sampled images with held-out prompts, matched seeds, and controls.

## End-to-end Gros–Straub pipeline

`vlm_prepare_gros_straub.py` downloads and pairs matched `vanilla` vs. `glasses` or `hat` portraits from the public Mendeley Data archives (`bpy24trj6s`, `ycjd7mdsbs`, `55wmmr8j3g`, `zsv3bgp7t4`) into a `pairs.json` manifest with subject-disjoint `train`, `validation`, and `test` splits:

```bash
python -m gradiend.examples.vlm_prepare_gros_straub --concept glasses --output data/gros_straub_glasses
python -m gradiend.examples.vlm_prepare_gros_straub --concept hat --output data/gros_straub_hats
```

You can then pass `data/gros_straub_glasses/pairs.json` directly to `VisionLanguagePredictionTrainer` (replace `data="pairs.json"` in the Qwen2.5-VL snippet above) or run the full Janus-Pro gradient extraction, GRADIEND training, and causal steering sweep:

```bash
python -m gradiend.examples.vlm_janus_steering data/gros_straub_glasses/pairs.json --output runs/janus_glasses --strengths 1.0 1.5
```

This writes `steering_panel.png` and individual `baseline`, `more`, `less`, and `random` images under `runs/janus_glasses/`.
