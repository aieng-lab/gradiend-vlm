"""End-to-end Janus-Pro GRADIEND training and causal image-generation steering.

Reads any ``pairs.json`` manifest with ``factual_prompt`` and
``counterfactual_prompt``, preprunes ``model.language_model.layers.11.self_attn.o_proj.weight``,
extracts teacher-forced visual-token gradients, trains a 1-D GRADIEND model on
the ``image`` and/or ``text`` counterfactual axis, renders matched-seed
``baseline`` / ``more`` / ``less`` / ``random`` images, and writes a contact
sheet ``steering_panel.png``.

Example usage (after ``python -m gradiend.examples.vlm_prepare_gros_straub``):

    python -m gradiend.examples.vlm_janus_steering data/gros_straub_glasses/pairs.json --output runs/janus_glasses
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import (
    ClassifierFreeGuidanceLogitsProcessor,
    JanusForConditionalGeneration,
    JanusProcessor,
    LogitsProcessorList,
    TopKLogitsWarper,
)

from gradiend import TrainingArguments, set_seed
from gradiend.model.model_with_gradiend import ModelWithGradiend
from gradiend.model.param_mapped import ParamMappedGradiendModel
from gradiend.trainer.core.training import train
from gradiend.trainer.vision_language.data import load_generation_pairs
from gradiend.trainer.vision_language.janus import JanusGenerationTask


JANUS_MODEL_ID = "deepseek-community/Janus-Pro-1B"
JANUS_REVISION = "1655280bb75959cc1cb85529a2a8b26e7016072e"
PARAMETER = "model.language_model.layers.11.self_attn.o_proj.weight"
AXES = {"image": (0, 2, 1, 3), "text": (0, 1, 2, 3)}
FACTUAL = (0, 3)
ALTERNATIVE = {"image": (1, 2), "text": (2, 1)}

DEFAULT_NEUTRAL_PROMPTS = (
    "A realistic head-and-shoulders portrait of an adult against a plain studio backdrop.",
    "A clear chest-up photograph of a person standing outdoors beside a brick wall.",
)


class _MappedJanus(ModelWithGradiend):
    """Wrap Janus with GRADIEND's indexed parameter-rewrite and restoration hooks."""

    def __init__(self, task: JanusGenerationTask, indices: np.ndarray | torch.Tensor, gradiend=None):
        self.task = task
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        if gradiend is None:
            parameter = task.model.get_parameter(PARAMETER)
            gradiend = ParamMappedGradiendModel(
                input_dim=len(indices),
                latent_dim=1,
                param_map={PARAMETER: {"shape": tuple(parameter.shape), "repr": "indices", "indices": self.indices}},
                device=torch.device("cpu"),
            )
        super().__init__(task.model, gradiend, base_model_device=next(task.model.parameters()).device)

    def create_gradients(self, prepared, **kwargs):
        parameter = self.base_model.get_parameter(PARAMETER)
        gradient, = torch.autograd.grad(self.task.loss(prepared), parameter)
        return gradient.detach().reshape(-1)[self.indices.to(gradient.device)].float()

    def _save_model(self, save_directory, **kwargs):
        self.task.processor.save_pretrained(save_directory)

    @classmethod
    def _load_model(cls, load_directory, **kwargs):
        raise NotImplementedError("Reload via JanusForConditionalGeneration and saved param_map.")


class _PairedGradients(torch.utils.data.Dataset):
    def __init__(self, values: np.ndarray, axis: str, scale: float):
        self.values = values
        self.alternative = ALTERNATIVE[axis]
        self.scale = scale

    def __len__(self) -> int:
        return 2 * len(self.values)

    def __getitem__(self, index: int) -> dict:
        row, pole = divmod(index, 2)
        fact = self.values[row, FACTUAL[pole]] / self.scale
        alternative = self.values[row, self.alternative[pole]] / self.scale
        return {
            "source": torch.from_numpy(fact),
            "target": torch.from_numpy(fact - alternative),
            "label": 1.0 if pole == 0 else -1.0,
        }


class _SeededImageSampler:
    def __init__(self, seeds, device, *, guidance_scale: float, temperature: float, top_k: int | None):
        self.generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        self.guide = ClassifierFreeGuidanceLogitsProcessor(guidance_scale) if guidance_scale > 1 else None
        self.filter = TopKLogitsWarper(top_k) if top_k else None
        self.temperature = temperature

    def __call__(self, input_ids, scores):
        batch_size = len(self.generators)
        logits = self.guide(input_ids, scores) if self.guide is not None else scores[:batch_size]
        logits = logits.float() / self.temperature
        if self.filter is not None:
            logits = self.filter(input_ids, logits)
        probabilities = torch.softmax(logits, dim=-1)
        choices = torch.stack([
            torch.multinomial(row, num_samples=1, generator=generator)
            for row, generator in zip(probabilities, self.generators)
        ])
        return torch.full_like(logits, -torch.inf).scatter_(1, choices, 0.0)


@torch.inference_mode()
def _generate_images(
    model,
    processor,
    prompts: list[str],
    seeds: list[int],
    *,
    guidance_scale: float = 5.0,
    temperature: float = 1.0,
) -> list[Image.Image]:
    formatted = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    device = model.get_input_embeddings().weight.device
    batch = processor(
        text=formatted, generation_mode="image", padding=True, padding_side="left", return_tensors="pt"
    ).to(device)
    top_k = model.generation_config.top_k
    sampler = _SeededImageSampler(
        seeds, device, guidance_scale=guidance_scale, temperature=temperature, top_k=top_k
    )
    codes = model.generate(
        **batch,
        generation_mode="image",
        guidance_scale=1.0,
        do_sample=False,
        num_return_sequences=1,
        return_dict_in_generate=False,
        logits_processor=LogitsProcessorList([sampler]),
    )
    decoded = model.decode_image_tokens(codes).float().cpu()
    if decoded.shape[1] == 3:
        decoded = decoded.permute(0, 2, 3, 1)
    arrays = (decoded.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    return [Image.fromarray(array) for array in arrays]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Direction vector has zero or non-finite norm")
    return (vector / norm).astype(np.float32)


def _pair_examples(task: JanusGenerationTask, pair, token_cache: Path):
    key = hashlib.sha256(f"{pair.factual_image}|{pair.counterfactual_image}".encode()).hexdigest()
    cache_path = token_cache / f"{key}.npy"
    if cache_path.is_file():
        targets = torch.from_numpy(np.load(cache_path, allow_pickle=False)).long()
    else:
        targets = torch.cat([task.encode_image(pair.factual_image), task.encode_image(pair.counterfactual_image)])
        tmp = cache_path.with_suffix(".part")
        with tmp.open("wb") as stream:
            np.save(stream, targets.numpy().astype(np.int16), allow_pickle=False)
        tmp.replace(cache_path)
    for text in (pair.factual_prompt, pair.counterfactual_prompt):
        for tokens in targets:
            yield task.prepare(text, tokens.unsqueeze(0))


def _extract_gradients(task: JanusGenerationTask, pairs_by_split: dict, output: Path, *, width: int) -> np.ndarray:
    token_cache = output / "tokens"
    token_cache.mkdir(parents=True, exist_ok=True)
    parameter = task.model.get_parameter(PARAMETER)
    parameter.requires_grad_(True)

    indices_path = output / "indices.npy"
    if indices_path.is_file():
        indices = np.load(indices_path, allow_pickle=False)
    else:
        train_pairs = pairs_by_split["train"]
        sample_size = min(16, len(train_pairs))
        chosen = np.random.default_rng(0).choice(len(train_pairs), size=sample_size, replace=False)
        importance = torch.zeros(parameter.numel(), device="cuda", dtype=torch.float32)
        for idx in chosen:
            grads = []
            for prepared in _pair_examples(task, train_pairs[int(idx)], token_cache):
                grad, = torch.autograd.grad(task.loss(prepared), parameter)
                grads.append(grad.detach().reshape(-1).float())
            pp, pn, np_, nn = grads
            importance += (pp + np_ - pn - nn).abs() + (pp + pn - np_ - nn).abs()
        k = min(width, parameter.numel())
        indices = torch.topk(importance, k).indices.sort().values.cpu().numpy()
        np.save(indices_path, indices, allow_pickle=False)

    idx_tensor = torch.from_numpy(indices).to("cuda")
    for split in ("train", "validation"):
        split_pairs = pairs_by_split[split]
        grad_path = output / f"{split}-gradient.npy"
        done_path = output / f"{split}-progress.json"
        if done_path.is_file() and json.loads(done_path.read_text())["completed"] == len(split_pairs) and grad_path.is_file():
            continue
        values = np.lib.format.open_memmap(grad_path, mode="w+", dtype=np.float32, shape=(len(split_pairs), 4, len(indices)))
        for row_idx, pair in enumerate(split_pairs):
            for cell, prepared in enumerate(_pair_examples(task, pair, token_cache)):
                grad, = torch.autograd.grad(task.loss(prepared), parameter)
                values[row_idx, cell] = grad.detach().reshape(-1).index_select(0, idx_tensor).float().cpu().numpy()
            if (row_idx + 1) % 100 == 0 or row_idx + 1 == len(split_pairs):
                values.flush()
                print(f"Extracted {split}: {row_idx + 1}/{len(split_pairs)} pairs", flush=True)
        done_path.write_text(json.dumps({"completed": len(split_pairs)}) + "\n", encoding="utf-8")

    parameter.requires_grad_(False)
    return indices


@torch.no_grad()
def _encode(gradiend, values: np.ndarray, scale: float, batch_size: int = 128) -> np.ndarray:
    out = []
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size]
        flat = torch.from_numpy(batch.reshape(-1, batch.shape[-1]) / scale).to(gradiend.device_encoder)
        out.append(gradiend.forward_encoder(flat).reshape(len(batch), -1).cpu().numpy())
    return np.concatenate(out)


def _train_gradiend(
    task: JanusGenerationTask,
    indices: np.ndarray,
    output: Path,
    axes: tuple[str, ...],
    *,
    lr: float = 0.003,
    epochs: int = 6,
) -> dict[str, tuple[np.ndarray, int]]:
    train_grads = np.load(output / "train-gradient.npy", mmap_mode="r", allow_pickle=False)
    val_grads = np.load(output / "validation-gradient.npy", mmap_mode="r", allow_pickle=False)
    scale = float(np.sqrt(np.mean(np.square(train_grads[:, FACTUAL], dtype=np.float64))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Non-finite or zero training gradient scale")

    directions = {}
    for axis in axes:
        dir_path = output / f"gradiend-{axis}.npy"
        meta_path = output / f"gradiend-{axis}.json"
        if dir_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            directions[axis] = (np.load(dir_path, allow_pickle=False), int(meta["orientation"]))
            continue

        best = None
        for seed in (0, 1):
            set_seed(seed)
            wrapper = _MappedJanus(task, indices)
            dataset = _PairedGradients(train_grads, axis, scale)
            run_dir = output / "learned" / axis / f"seed-{seed}"
            args = TrainingArguments(
                source="factual",
                target="diff",
                learning_rate=lr,
                num_train_epochs=epochs,
                train_batch_size=32,
                eval_steps=max(1, len(dataset) // 32),
                output_dir=str(run_dir),
                experiment_dir=str(run_dir),
                use_cache=False,
                fail_on_non_convergence=False,
            )
            train(wrapper, dataset, args)
            scores = _encode(wrapper.gradiend, val_grads, scale)
            signed = scores[:, [0, 3]].ravel()
            labels = np.tile([1.0, -1.0], len(scores))
            corr = float(np.corrcoef(signed, labels)[0, 1])
            orientation = 1 if corr >= 0 else -1
            vector = _unit(wrapper.gradiend.decoder[0].weight.detach().float().cpu().numpy().ravel())
            if best is None or abs(corr) > best["abs_corr"]:
                best = {"abs_corr": abs(corr), "corr": corr, "orientation": orientation, "vector": vector, "seed": seed}

        np.save(dir_path, best["vector"], allow_pickle=False)
        meta_path.write_text(
            json.dumps({"axis": axis, "seed": best["seed"], "val_correlation": best["corr"], "orientation": best["orientation"]}, indent=2) + "\n",
            encoding="utf-8",
        )
        directions[axis] = (best["vector"], best["orientation"])
        print(f"Trained gradiend-{axis}: val_corr={best['corr']:+.3f} (orientation={best['orientation']})", flush=True)
    return directions


def _build_contact_sheet(rows: list[tuple[str, list[tuple[str, Image.Image]]]], out_path: Path) -> None:
    thumb, pad, header, left = 224, 8, 28, 220
    cols = [label for label, _ in rows[0][1]]
    canvas = Image.new("RGB", (left + len(cols) * (thumb + pad) + pad, header + len(rows) * (thumb + pad) + pad), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for ci, title in enumerate(cols):
        draw.text((left + ci * (thumb + pad) + 4, 6), title, fill=(0, 0, 0))
    for ri, (prompt_label, images) in enumerate(rows):
        y = header + ri * (thumb + pad)
        draw.text((6, y + 60), prompt_label, fill=(0, 0, 0))
        for ci, (_, image) in enumerate(images):
            canvas.paste(image.resize((thumb, thumb)), (left + ci * (thumb + pad), y))
    canvas.save(out_path)
    print(f"Saved contact sheet to {out_path}", flush=True)


def run_pipeline(
    data: Path,
    output: Path,
    *,
    axes: tuple[str, ...],
    strengths: tuple[float, ...],
    width: int = 65536,
    local_files_only: bool = False,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    pairs = load_generation_pairs(data)
    by_split = {split: [p for p in pairs if p.split == split] for split in ("train", "validation", "test")}
    if not by_split["train"] or not by_split["validation"] or not by_split["test"]:
        raise ValueError("Manifest must include train, validation, and test pairs.")

    processor = JanusProcessor.from_pretrained(JANUS_MODEL_ID, revision=JANUS_REVISION, local_files_only=local_files_only)
    model = JanusForConditionalGeneration.from_pretrained(
        JANUS_MODEL_ID, revision=JANUS_REVISION, local_files_only=local_files_only, dtype=torch.bfloat16
    ).to("cuda").eval()
    model.requires_grad_(False)
    task = JanusGenerationTask(model, processor)

    indices = _extract_gradients(task, by_split, output, width=width)
    directions = _train_gradiend(task, indices, output, axes)

    wrapper = _MappedJanus(task, indices)
    parameter = model.get_parameter(PARAMETER)
    weight_norm = parameter.detach().reshape(-1)[wrapper.indices.to(parameter.device)].float().norm().item()

    # Select distinct test prompts + neutral prompts with fixed seeds.
    test_prompts = list(dict.fromkeys(
        [p.factual_prompt for p in by_split["test"][:1]]
        + [p.counterfactual_prompt for p in by_split["test"][:1]]
        + list(DEFAULT_NEUTRAL_PROMPTS)
    ))
    render_items = [
        (prompt, int.from_bytes(hashlib.sha256(f"{idx}:{rep}".encode()).digest()[:4], "big"))
        for idx, prompt in enumerate(test_prompts)
        for rep in range(2)
    ]

    variants = [("baseline", None, 0.0, False)]
    for axis, (vector, orientation) in directions.items():
        for strength in strengths:
            signed = strength * orientation
            variants.extend([
                (f"gradiend-{axis}/more@{strength:.2f}", vector, signed, False),
                (f"gradiend-{axis}/less@{strength:.2f}", vector, -signed, False),
                (f"gradiend-{axis}/rand@{strength:.2f}", vector, signed, True),
            ])

    rendered_dir = output / "images"
    rendered_dir.mkdir(exist_ok=True)
    rendered = {}
    prompts_batch = [item[0] for item in render_items]
    seeds_batch = [item[1] for item in render_items]
    for name, vector, signed, randomized in variants:
        if vector is None:
            ctx = nullcontext()
        else:
            vec = np.random.default_rng(481).permutation(vector) if randomized else vector
            delta = torch.from_numpy(vec) * (signed * weight_norm)
            ctx = wrapper.intervene(signed, signal="gradient", update_vector=delta)
        with ctx:
            imgs = _generate_images(model, processor, prompts_batch, seeds_batch)
        rendered[name] = imgs
        for idx, img in enumerate(imgs):
            safe = name.replace("/", "_").replace("@", "_")
            img.save(rendered_dir / f"{safe}_row{idx}.png")

    panel_rows = []
    for idx, (prompt, seed) in enumerate(render_items):
        short_prompt = "\n".join(prompt[i:i + 28] for i in range(0, min(len(prompt), 84), 28))
        label = f"{short_prompt}\nseed={seed}"
        panel_rows.append((label, [(name, rendered[name][idx]) for name, _, _, _ in variants]))
    _build_contact_sheet(panel_rows, output / "steering_panel.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Path to pairs.json manifest")
    parser.add_argument("--output", type=Path, default=Path("runs/janus_steering"))
    parser.add_argument("--axis", choices=("image", "text", "both"), default="both")
    parser.add_argument("--strengths", type=float, nargs="+", default=[1.0, 1.5])
    parser.add_argument("--width", type=int, default=65536)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    axes = ("image", "text") if args.axis == "both" else (args.axis,)
    run_pipeline(
        args.data,
        args.output,
        axes=axes,
        strengths=tuple(args.strengths),
        width=args.width,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()
