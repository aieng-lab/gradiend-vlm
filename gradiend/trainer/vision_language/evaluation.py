"""Controlled evaluation for matched image–text continuation pairs."""

from dataclasses import replace
import math
from statistics import mean


def crossed_examples(match):
    """Return rows in image order, then in answer order within each image."""
    negative, positive = match.negative, match.positive
    return (
        negative,
        replace(negative, answer=positive.answer),
        replace(positive, answer=negative.answer),
        positive,
    )


def paired_scores(matches, score):
    """Measure answer preference by image and its dependence on the image."""
    negative_margins = []
    positive_margins = []
    for match in matches:
        a_a, a_b, b_a, b_b = (score(example, match.group_id) for example in crossed_examples(match))
        negative_margins.append(a_b - a_a)
        positive_margins.append(b_b - b_a)
    negative = mean(negative_margins)
    positive = mean(positive_margins)
    return {
        "negative_image_margin": negative,
        "positive_image_margin": positive,
        "grounding": positive - negative,
        "global_answer_preference": (positive + negative) / 2,
        "n_matched_pairs": len(matches),
    }


def neutral_mean_log_probability(rows, score):
    total_log_probability = 0.0
    total_tokens = 0
    for row in rows:
        log_probability, tokens = score(row.example, row.group_id)
        total_log_probability += log_probability
        total_tokens += tokens
    if total_tokens <= 0:
        raise ValueError("Neutral evaluation requires at least one answer token")
    return total_log_probability / total_tokens


def score_shift(baseline, edited):
    return {
        "grounding_improvement": edited["grounding"] - baseline["grounding"],
        "neutral_likelihood_ratio": math.exp(
            edited["neutral_mean_log_probability"] - baseline["neutral_mean_log_probability"]
        ),
        "negative_image_margin_shift": edited["negative_image_margin"] - baseline["negative_image_margin"],
        "positive_image_margin_shift": edited["positive_image_margin"] - baseline["positive_image_margin"],
    }


def select_edit(grid, minimum_neutral_ratio):
    """Select the best validation improvement that preserves neutral likelihood."""
    eligible = (
        row for row in grid
        if row["neutral_likelihood_ratio"] >= minimum_neutral_ratio and row["grounding_improvement"] > 0
    )
    return max(eligible, key=lambda row: row["grounding_improvement"], default=None)
