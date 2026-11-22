"""Manifest and paired-example tests with paths standing in for image files."""

import json

import pandas as pd
import pytest

from gradiend.trainer.vision_language.data import (
    VisionLanguageExample,
    VisionLanguageGenerationPair,
    expand_pairs,
    load_generation_pairs,
    load_matched_examples,
    load_neutral_examples,
)


@pytest.fixture
def matched_row(tmp_path):
    for filename in ("without.png", "with.png", "unrelated.png"):
        (tmp_path / filename).touch()
    return {
        "group_id": "subject-1",
        "split": "Train",
        "prompt": "What is on the face?",
        "image_a": tmp_path / "without.png",
        "image_b": tmp_path / "with.png",
        "answer_a": "no glasses",
        "answer_b": "glasses",
    }


def test_balanced_pairs_change_exactly_one_field(matched_row):
    rows = load_matched_examples(pd.DataFrame([matched_row]))
    pairs = expand_pairs(rows, "both")

    assert [(pair.label, pair.axis) for pair in pairs] == [(-1, "image"), (-1, "text"), (1, "image"), (1, "text")]
    assert all(pair.group_id == "subject-1" and pair.split == "train" for pair in pairs)
    assert pairs[0].factual == pairs[1].factual == rows[0].negative
    assert pairs[2].factual == pairs[3].factual == rows[0].positive
    assert pairs[0].alternative == pairs[3].alternative == VisionLanguageExample(
        matched_row["image_b"], matched_row["prompt"], "no glasses"
    )
    assert pairs[1].alternative == pairs[2].alternative == VisionLanguageExample(
        matched_row["image_a"], matched_row["prompt"], "glasses"
    )
    assert expand_pairs(rows, "image") == [pairs[0], pairs[2]]
    assert expand_pairs(rows, "text") == [pairs[1], pairs[3]]
    with pytest.raises(ValueError, match="Unsupported counterfactual axis"):
        expand_pairs(rows, "audio")


def test_csv_paths_are_relative_to_the_manifest_and_text_is_preserved(tmp_path, matched_row):
    row = {
        **matched_row,
        "group_id": "001",
        "split": " Val ",
        "image_a": "without.png",
        "image_b": "with.png",
        "prompt": 'Look at the face, then answer "briefly".\n',
        "answer_a": "NA",
    }
    path = tmp_path / "matched.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    match, = load_matched_examples(path)

    assert match.group_id == "001"
    assert match.split == "validation"
    assert match.negative == VisionLanguageExample(tmp_path / "without.png", row["prompt"], "NA")


def test_dataframe_paths_are_relative_to_working_directory(tmp_path, matched_row, monkeypatch):
    monkeypatch.chdir(tmp_path)
    row = {**matched_row, "group_id": 5, "image_a": "without.png"}

    match, = load_matched_examples(pd.DataFrame([row]))

    assert match.group_id == "5"
    assert match.negative.image == tmp_path / "without.png"


def test_match_allows_repeated_subject_with_different_prompt_in_same_split(matched_row):
    second = {**matched_row, "prompt": "Does the subject have glasses?"}
    matches = load_matched_examples(pd.DataFrame([matched_row, second]))
    assert len(matches) == 2


@pytest.mark.parametrize(
    ("group_id", "message"),
    [("subject-1", ".*group_id.*splits"), ("subject-2", ".*image.*splits")],
)
def test_rejects_subject_or_image_reuse_across_splits(matched_row, group_id, message):
    second = {**matched_row, "group_id": group_id, "split": "test"}
    with pytest.raises(ValueError, match=message):
        load_matched_examples(pd.DataFrame([matched_row, second]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prompt", " ", ".*prompt.*non-empty string"),
        ("answer_a", None, ".*answer_a.*non-empty string"),
        ("answer_a", "glasses", ".*different answers"),
        ("split", "dev", ".*unsupported split"),
        ("image_a", "missing.png", ".*image_a.*existing local image file"),
    ],
)
def test_rejects_invalid_manifest_row(matched_row, field, value, message):
    with pytest.raises(ValueError, match=message):
        load_matched_examples(pd.DataFrame([{**matched_row, field: value}]))


def test_rejects_identical_images_and_duplicate_pairs(tmp_path, matched_row):
    with pytest.raises(ValueError, match="different images"):
        row = {**matched_row, "image_a": tmp_path / "unused" / ".." / "with.png"}
        (tmp_path / "unused").mkdir()
        load_matched_examples(pd.DataFrame([row]))
    with pytest.raises(ValueError, match="duplicate matched example"):
        load_matched_examples(pd.DataFrame([matched_row, matched_row]))


def test_rejects_malformed_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("group_id,split,prompt,image_a,image_b,answer_a,answer_b\na,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2: expected 7 fields, got 2"):
        load_matched_examples(path)

    path.write_text("group_id,split,prompt,prompt,image_a,image_b,answer_a,answer_b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate column names"):
        load_matched_examples(path)


def test_reads_neutral_examples_and_rejects_split_leakage(tmp_path):
    image = tmp_path / "unrelated.png"
    image.touch()
    row = {"group_id": "neutral-1", "split": "val", "image": image, "prompt": "What color?", "answer": "blue"}

    neutral, = load_neutral_examples(pd.DataFrame([row]))

    assert neutral.group_id == "neutral-1"
    assert neutral.split == "validation"
    assert neutral.example == VisionLanguageExample(image, "What color?", "blue")
    with pytest.raises(ValueError, match="image.*appears in both"):
        second = {**row, "group_id": "neutral-2", "split": "test"}
        load_neutral_examples(pd.DataFrame([row, second]))


@pytest.fixture
def json_pair_data(matched_row):
    return {
        "prompt": "What is on the face?",
        "factual_answer": "glasses",
        "counterfactual_answer": "no glasses",
        "factual_prompt": "A portrait of a person wearing glasses.",
        "counterfactual_prompt": "A portrait of a person with uncovered eyes.",
        "pairs": [{
            "group_id": "subject-1",
            "split": " Val ",
            "factual_image": matched_row["image_b"].name,
            "counterfactual_image": matched_row["image_a"].name,
        }],
    }


def write_json_manifest(tmp_path, data):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_json_supports_understanding_and_generation_with_the_same_images(tmp_path, json_pair_data):
    path = write_json_manifest(tmp_path, json_pair_data)

    match, = load_matched_examples(str(path))
    generation, = load_generation_pairs(path)

    assert match.group_id == "subject-1"
    assert match.split == "validation"
    assert match.positive == VisionLanguageExample(tmp_path / "with.png", json_pair_data["prompt"], "glasses")
    assert match.negative == VisionLanguageExample(tmp_path / "without.png", json_pair_data["prompt"], "no glasses")
    assert generation == VisionLanguageGenerationPair(
        "subject-1", "validation", match.positive.image, match.negative.image,
        json_pair_data["factual_prompt"], json_pair_data["counterfactual_prompt"],
    )
    positive_pairs = [pair for pair in expand_pairs([match], "both") if pair.label == 1]
    assert [p.alternative.image for p in positive_pairs] == [match.negative.image, match.positive.image]
    assert [p.alternative.answer for p in positive_pairs] == [match.positive.answer, match.negative.answer]


def test_json_pair_text_overrides_inherit_the_remaining_shared_fields(tmp_path, json_pair_data):
    [pair] = json_pair_data["pairs"]
    pair.update(prompt="  Describe the eyewear.\n", factual_answer="spectacles", counterfactual_prompt="A bare face.")
    path = write_json_manifest(tmp_path, json_pair_data)

    match, = load_matched_examples(path)
    generation, = load_generation_pairs(path)

    assert match.positive.prompt == match.negative.prompt == "  Describe the eyewear.\n"
    assert match.positive.answer == "spectacles"
    assert match.negative.answer == json_pair_data["counterfactual_answer"]
    assert generation.factual_prompt == json_pair_data["factual_prompt"]
    assert generation.counterfactual_prompt == "A bare face."


def test_json_each_mode_can_be_written_independently(tmp_path, json_pair_data):
    understanding = {field: json_pair_data[field] for field in ("prompt", "factual_answer", "counterfactual_answer", "pairs")}
    path = write_json_manifest(tmp_path, understanding)
    assert len(load_matched_examples(path)) == 1
    with pytest.raises(ValueError, match="missing shared text fields: factual_prompt, counterfactual_prompt"):
        load_generation_pairs(path)

    generation = {field: json_pair_data[field] for field in ("factual_prompt", "counterfactual_prompt", "pairs")}
    path = write_json_manifest(tmp_path, generation)
    assert len(load_generation_pairs(path)) == 1
    with pytest.raises(ValueError, match="missing shared text fields: prompt, factual_answer, counterfactual_answer"):
        load_matched_examples(path)


@pytest.mark.parametrize("load", [load_matched_examples, load_generation_pairs])
@pytest.mark.parametrize(("group", "message"), [("subject-1", "group_id.*splits"), ("subject-2", "image.*splits")])
def test_json_rejects_split_leakage(tmp_path, json_pair_data, load, group, message):
    json_pair_data["pairs"].append({**json_pair_data["pairs"][0], "group_id": group, "split": "test"})
    path = write_json_manifest(tmp_path, json_pair_data)
    with pytest.raises(ValueError, match=message):
        load(path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("factual_image", "missing.jpg", "factual_image.*existing local image file"),
        ("factual_image", "without.png", "must be different images"),
        ("counterfactual_answer", " glasses ", "factual_answer and counterfactual_answer must differ"),
        ("factual_prompt", None, "factual_prompt.*non-empty string"),
        ("counterfactual_prompt", "A portrait of a person wearing glasses.", "factual_prompt and counterfactual_prompt must differ"),
        ("typo", "value", "unsupported fields.*typo"),
    ],
)
def test_json_rejects_invalid_pair_values(tmp_path, json_pair_data, field, value, error):
    json_pair_data["pairs"][0][field] = value
    path = write_json_manifest(tmp_path, json_pair_data)
    with pytest.raises(ValueError, match=error):
        load_matched_examples(path)


def test_json_rejects_duplicate_fields_and_empty_pairs(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text('{"pairs": [], "pairs": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate field 'pairs'"):
        load_matched_examples(path)
    path.write_text('{"pairs": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty pairs array"):
        load_generation_pairs(path)
