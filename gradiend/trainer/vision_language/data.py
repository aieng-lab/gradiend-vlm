"""Paired manifests for learning features from image and text counterfactuals."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import Iterator, Literal, Sequence, Union

import pandas as pd

from gradiend.util import normalize_split_name


Manifest = Union[pd.DataFrame, str, PathLike]
CounterfactualAxis = Literal["image", "text", "both"]
PairAxis = Literal["image", "text"]

_MATCH_COLUMNS = ("group_id", "split", "prompt", "image_a", "image_b", "answer_a", "answer_b")
_NEUTRAL_COLUMNS = ("group_id", "split", "image", "prompt", "answer")
_SPLITS = {"train", "validation", "test"}
_JSON_PAIR_FIELDS = ("group_id", "split", "factual_image", "counterfactual_image")
_UNDERSTANDING_TEXT = ("prompt", "factual_answer", "counterfactual_answer")
_GENERATION_TEXT = ("factual_prompt", "counterfactual_prompt")
_JSON_TEXT_FIELDS = {*_UNDERSTANDING_TEXT, *_GENERATION_TEXT}


@dataclass(frozen=True)
class VisionLanguageExample:
    image: Path
    prompt: str
    answer: str


@dataclass(frozen=True)
class VisionLanguageMatch:
    group_id: str
    split: str
    negative: VisionLanguageExample
    positive: VisionLanguageExample


@dataclass(frozen=True)
class VisionLanguageGenerationPair:
    """Matched factual and counterfactual images with text-to-image prompts."""

    group_id: str
    split: str
    factual_image: Path
    counterfactual_image: Path
    factual_prompt: str
    counterfactual_prompt: str


@dataclass(frozen=True)
class VisionLanguageNeutral:
    group_id: str
    split: str
    example: VisionLanguageExample


@dataclass(frozen=True)
class VisionLanguagePair:
    group_id: str
    split: str
    axis: PairAxis
    label: int
    factual: VisionLanguageExample
    alternative: VisionLanguageExample


def _manifest_rows(
    data: Manifest, columns: Sequence[str]
) -> Iterator[tuple[str, Path, tuple]]:
    if isinstance(data, pd.DataFrame):
        source = "VLM manifest"
        image_base = Path.cwd()
        _validate_columns(list(data.columns), columns, source)
        if data.empty:
            raise ValueError(f"{source} contains no rows.")
        values = data.loc[:, list(columns)].itertuples(index=False, name=None)
        for row_number, row_values in enumerate(values, 1):
            yield f"{source} row {row_number}", image_base, row_values
    elif isinstance(data, (str, PathLike)):
        path = Path(data).expanduser().resolve()
        source = f"VLM manifest {str(path)!r}"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            try:
                header = next(reader, [])
                _validate_columns(header, columns, source)
                positions = [header.index(column) for column in columns]
                found_rows = False
                for values in reader:
                    if not values:
                        continue
                    found_rows = True
                    row = f"{source} line {reader.line_num}"
                    if len(values) != len(header):
                        raise ValueError(f"{row}: expected {len(header)} fields, got {len(values)}.")
                    yield row, path.parent, tuple(values[position] for position in positions)
            except csv.Error as exc:
                raise ValueError(f"{source} line {reader.line_num}: invalid CSV data.") from exc
            if not found_rows:
                raise ValueError(f"{source} contains no rows.")
    else:
        raise TypeError("VLM data must be a pandas DataFrame or a CSV path.")


def _validate_columns(actual: Sequence[object], required: Sequence[str], source: str) -> None:
    if len(set(actual)) != len(actual):
        raise ValueError(f"{source} contains duplicate column names.")
    missing = [column for column in required if column not in actual]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}.")


def _required_text(value: object, field: str, row: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{row}: {field} must be a non-empty string.")
    return value


def _group_and_split(group_value: object, split_value: object, row: str) -> tuple[str, str]:
    if isinstance(group_value, Integral) and not isinstance(group_value, bool):
        group_value = str(group_value)
    group_id = _required_text(group_value, "group_id", row).strip()
    split = normalize_split_name(_required_text(split_value, "split", row).strip())
    if split not in _SPLITS:
        raise ValueError(f"{row}: unsupported split {split!r}. Expected train, validation, or test.")
    return group_id, split


def _image_path(value: object, field: str, base: Path, row: str) -> Path:
    if isinstance(value, PathLike):
        value = str(value)
    path = Path(_required_text(value, field, row))
    try:
        path = (base / path.expanduser()).resolve(strict=True)
        if path.is_file():
            return path
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{row}: {field} must reference an existing local image file: {value!r}.") from exc
    raise ValueError(f"{row}: {field} must reference an existing local image file: {value!r}.")


def _remember_split(seen: dict, key: object, field: str, split: str, row: str) -> None:
    previous = seen.setdefault(key, split)
    if previous != split:
        raise ValueError(f"{row}: {field} {str(key)!r} appears in both {previous!r} and {split!r} splits.")


def _unique_json_fields(items: list[tuple]) -> dict:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate field {key!r}.")
        result[key] = value
    return result


def _validate_json_contrasts(text: dict[str, str], row: str) -> None:
    for factual, counterfactual in (("factual_answer", "counterfactual_answer"), _GENERATION_TEXT):
        if factual in text and text[factual].strip() == text[counterfactual].strip():
            raise ValueError(f"{row}: {factual} and {counterfactual} must differ.")


def _json_pairs(
    data: Union[str, PathLike], required_text: tuple[str, ...]
) -> Iterator[tuple[str, str, str, Path, Path, dict[str, str]]]:
    path = Path(data).expanduser().resolve()
    source = f"VLM JSON manifest {str(path)!r}"
    with path.open(encoding="utf-8-sig") as stream:
        try:
            manifest = json.load(stream, object_pairs_hook=_unique_json_fields)
        except ValueError as exc:
            raise ValueError(f"{source}: invalid JSON. {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{source} must contain a JSON object with shared text and a pairs array.")
    unknown = manifest.keys() - {"pairs", *_JSON_TEXT_FIELDS}
    if unknown:
        raise ValueError(f"{source} has unsupported fields: {', '.join(sorted(unknown))}.")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{source} requires a non-empty pairs array.")

    shared = {}
    for fields in (_UNDERSTANDING_TEXT, _GENERATION_TEXT):
        if fields == required_text or any(field in manifest for field in fields):
            missing = [field for field in fields if field not in manifest]
            if missing:
                raise ValueError(f"{source} is missing shared text fields: {', '.join(missing)}.")
            for field in fields:
                shared[field] = _required_text(manifest[field], field, source)
    _validate_json_contrasts(shared, source)

    groups, images = {}, {}
    for index, pair in enumerate(pairs, 1):
        row = f"{source} pair {index}"
        if not isinstance(pair, dict):
            raise ValueError(f"{row} must be a JSON object.")
        missing = [field for field in _JSON_PAIR_FIELDS if field not in pair]
        if missing:
            raise ValueError(f"{row} is missing required fields: {', '.join(missing)}.")
        unknown = pair.keys() - {*_JSON_PAIR_FIELDS, *shared}
        if unknown:
            raise ValueError(f"{row} has unsupported fields without shared settings: {', '.join(sorted(unknown))}.")
        group_id, split = _group_and_split(pair["group_id"], pair["split"], row)
        factual = _image_path(pair["factual_image"], "factual_image", path.parent, row)
        counterfactual = _image_path(pair["counterfactual_image"], "counterfactual_image", path.parent, row)
        if factual == counterfactual:
            raise ValueError(f"{row}: factual_image and counterfactual_image must be different images.")
        text = {
            field: _required_text(pair[field], field, row) if field in pair else value
            for field, value in shared.items()
        }
        _validate_json_contrasts(text, row)
        _remember_split(groups, group_id, "group_id", split, row)
        for image in (factual, counterfactual):
            _remember_split(images, image, "image", split, row)
        yield row, group_id, split, factual, counterfactual, text


def load_generation_pairs(data: Union[str, PathLike]) -> list[VisionLanguageGenerationPair]:
    """Read paired image-generation prompts from a JSON manifest.

    The manifest sets ``factual_prompt`` and ``counterfactual_prompt`` once.
    Each entry in ``pairs`` supplies ``group_id``, ``split``, ``factual_image``
    and ``counterfactual_image``. An entry may override either shared prompt.
    Relative images are resolved beside the manifest.
    """
    pairs = []
    seen = set()
    for row, group_id, split, factual, counterfactual, text in _json_pairs(data, _GENERATION_TEXT):
        examples = (factual, counterfactual, text["factual_prompt"], text["counterfactual_prompt"])
        if examples in seen:
            raise ValueError(f"{row}: duplicate generation pair.")
        seen.add(examples)
        pairs.append(VisionLanguageGenerationPair(group_id, split, *examples))
    return pairs


def _json_matches(data: Union[str, PathLike]) -> list[VisionLanguageMatch]:
    matches = []
    seen = set()
    for row, group_id, split, factual, counterfactual, text in _json_pairs(data, _UNDERSTANDING_TEXT):
        negative = VisionLanguageExample(counterfactual, text["prompt"], text["counterfactual_answer"])
        positive = VisionLanguageExample(factual, text["prompt"], text["factual_answer"])
        examples = (negative, positive)
        if examples in seen:
            raise ValueError(f"{row}: duplicate matched example.")
        seen.add(examples)
        matches.append(VisionLanguageMatch(group_id, split, negative, positive))
    return matches


def load_matched_examples(data: Manifest) -> list[VisionLanguageMatch]:
    """Read aligned image and answer pairs, with side A negative and side B positive.

    JSON manifests name the positive and negative images ``factual_image`` and
    ``counterfactual_image`` and supply a shared ``prompt``, ``factual_answer``
    and ``counterfactual_answer``. Pairs may override those text fields.
    Relative images are resolved beside a CSV or JSON manifest, or relative to
    the working directory for a DataFrame. Groups and images stay in one split.
    """
    if isinstance(data, (str, PathLike)) and Path(data).suffix.lower() == ".json":
        return _json_matches(data)
    matches = []
    groups = {}
    images = {}
    seen_matches = set()
    for row, base, values in _manifest_rows(data, _MATCH_COLUMNS):
        group_value, split_value, prompt_value, image_a, image_b, answer_a, answer_b = values
        group_id, split = _group_and_split(group_value, split_value, row)
        prompt = _required_text(prompt_value, "prompt", row)
        negative = VisionLanguageExample(
            _image_path(image_a, "image_a", base, row), prompt, _required_text(answer_a, "answer_a", row)
        )
        positive = VisionLanguageExample(
            _image_path(image_b, "image_b", base, row), prompt, _required_text(answer_b, "answer_b", row)
        )
        if negative.image == positive.image:
            raise ValueError(f"{row}: image_a and image_b must be different images.")
        if negative.answer == positive.answer:
            raise ValueError(f"{row}: answer_a and answer_b must be different answers.")

        _remember_split(groups, group_id, "group_id", split, row)
        for image in (negative.image, positive.image):
            _remember_split(images, image, "image", split, row)
        examples = (negative, positive)
        if examples in seen_matches:
            raise ValueError(f"{row}: duplicate matched example.")
        seen_matches.add(examples)
        matches.append(VisionLanguageMatch(group_id, split, negative, positive))
    return matches


def load_neutral_examples(data: Manifest) -> list[VisionLanguageNeutral]:
    """Read unrelated image–prompt–answer examples for preservation evaluation."""
    neutral = []
    groups = {}
    images = {}
    seen_examples = set()
    for row, base, values in _manifest_rows(data, _NEUTRAL_COLUMNS):
        group_value, split_value, image_value, prompt_value, answer_value = values
        group_id, split = _group_and_split(group_value, split_value, row)
        image = _image_path(image_value, "image", base, row)
        prompt = _required_text(prompt_value, "prompt", row)
        answer = _required_text(answer_value, "answer", row)
        _remember_split(groups, group_id, "group_id", split, row)
        _remember_split(images, image, "image", split, row)
        example = VisionLanguageExample(image, prompt, answer)
        if example in seen_examples:
            raise ValueError(f"{row}: duplicate neutral example.")
        seen_examples.add(example)
        neutral.append(VisionLanguageNeutral(group_id, split, example))
    return neutral


def expand_pairs(matches: Sequence[VisionLanguageMatch], axis: CounterfactualAxis) -> list[VisionLanguagePair]:
    """Expand each matched example into directed pairs, balanced by axis and class."""
    if axis not in ("image", "text", "both"):
        raise ValueError(f"Unsupported counterfactual axis {axis!r}. Expected image, text, or both.")

    pairs = []
    for match in matches:
        for label, factual, opposite in (
            (-1, match.negative, match.positive),
            (1, match.positive, match.negative),
        ):
            if axis in ("image", "both"):
                alternative = replace(factual, image=opposite.image)
                pairs.append(VisionLanguagePair(match.group_id, match.split, "image", label, factual, alternative))
            if axis in ("text", "both"):
                alternative = replace(factual, answer=opposite.answer)
                pairs.append(VisionLanguagePair(match.group_id, match.split, "text", label, factual, alternative))
    return pairs
