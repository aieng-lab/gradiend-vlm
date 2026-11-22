"""Counterfactual scoring and validation selection without a model."""

from math import log
from pathlib import Path

import pytest

from gradiend.trainer.vision_language.data import VisionLanguageExample, VisionLanguageMatch, VisionLanguageNeutral
from gradiend.trainer.vision_language.evaluation import neutral_mean_log_probability, paired_scores, score_shift, select_edit


def test_grounding_separates_the_image_effect_from_global_answer_preference():
    negative = VisionLanguageExample(Path("without.png"), "Eyeglasses?", "no glasses")
    positive = VisionLanguageExample(Path("with.png"), negative.prompt, "glasses")
    pair = VisionLanguageMatch("subject", "validation", negative, positive)
    log_probability = {
        (negative.image, negative.answer): -0.1,
        (negative.image, positive.answer): -1.1,
        (positive.image, negative.answer): -2.1,
        (positive.image, positive.answer): -0.1,
    }

    def score(example, group_id):
        assert group_id == "subject"
        return log_probability[example.image, example.answer]

    baseline = paired_scores([pair], score)
    shifted = paired_scores([pair], lambda example, group_id: score(example, group_id) - (example.answer == negative.answer))
    assert baseline["negative_image_margin"] == pytest.approx(-1.0)
    assert baseline["positive_image_margin"] == pytest.approx(2.0)
    assert baseline["grounding"] == pytest.approx(3.0)
    assert shifted["grounding"] == pytest.approx(baseline["grounding"])
    assert shifted["global_answer_preference"] - baseline["global_answer_preference"] == pytest.approx(1.0)


def test_preservation_is_weighted_by_token_count_and_gates_validation_selection():
    neutral = [
        VisionLanguageNeutral(answer, "validation", VisionLanguageExample(Path(f"{answer}.png"), "Color?", answer))
        for answer in ("blue", "dark blue")
    ]
    scores = {"blue": (-1.0, 1), "dark blue": (-5.0, 2)}
    mean_log_probability = neutral_mean_log_probability(neutral, lambda example, _: scores[example.answer])
    assert mean_log_probability == -2.0
    baseline = {
        "grounding": 1.0, "negative_image_margin": 0.0, "positive_image_margin": 1.0,
        "neutral_mean_log_probability": mean_log_probability,
    }
    acceptable = score_shift(baseline, {**baseline, "grounding": 1.2})
    rejected = score_shift(baseline, {
        **baseline, "grounding": 2.0, "neutral_mean_log_probability": mean_log_probability + log(0.98),
    })
    assert rejected["neutral_likelihood_ratio"] == pytest.approx(0.98)
    assert select_edit([rejected, acceptable], 0.99) is acceptable
    assert select_edit([rejected, score_shift(baseline, baseline)], 0.99) is None
