"""
FSRS v4 spaced repetition scheduler — updated for 1-10 grade scale.

User-facing grades (1-10):
  1-3  → Poor      (FSRS internal grade 1: Blackout — review tomorrow)
  4-6  → Average   (FSRS internal grade 2: Hard — short interval)
  7-9  → Good      (FSRS internal grade 3: Good — normal growth)
  10   → Perfect   (FSRS internal grade 4: Perfect — large boost)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

_INITIAL_STABILITY = {1: 0.4, 2: 1.2, 3: 3.1, 4: 15.5}
_STABILITY_MULTIPLIER = {1: 0.2, 2: 1.3, 3: 2.0, 4: 2.5}
_DIFFICULTY_DELTA = {1: 2.0, 2: 0.5, 3: -0.1, 4: -0.5}
_MAX_INTERVAL_DAYS = 365

def map_grade(grade_1_10: int) -> int:
    """Maps user-facing 1-10 grade to internal FSRS 1-4 grade."""
    if grade_1_10 <= 3:
        return 1
    elif grade_1_10 <= 6:
        return 2
    elif grade_1_10 <= 9:
        return 3
    else:
        return 4

@dataclass
class FSRSResult:
    new_stability: float
    new_difficulty: float
    new_retrievability: float
    next_review_date: date
    interval_days: int

def calculate_retrievability(stability: float, elapsed_days: int) -> float:
    if stability <= 0:
        return 0.0
    return math.exp(-elapsed_days / max(stability, 0.01))

def schedule(
    current_stability: float,
    current_difficulty: float,
    grade: int,
    total_reviews: int,
    days_since_last_review: int = 1,
) -> FSRSResult:
    """
    Computes new FSRS state after a review.
    Accepts grade as 1-10 (user-facing) or 1-4 (internal).
    Values > 4 are automatically mapped via map_grade().
    """
    if grade > 4:
        grade = map_grade(grade)

    if grade not in (1, 2, 3, 4):
        raise ValueError(f"Grade must be 1-4 or 1-10, got {grade}")

    if total_reviews == 0:
        new_stability = _INITIAL_STABILITY[grade]
    else:
        if grade == 1:
            new_stability = min(current_stability * 0.2, 0.5)
        else:
            r = calculate_retrievability(current_stability, days_since_last_review)
            d_factor = max(0.1, (10.0 - current_difficulty) / 9.0)
            new_stability = current_stability * _STABILITY_MULTIPLIER[grade] * d_factor
            if r > 0.8 and grade >= 3:
                new_stability *= 1.1

    new_stability = max(0.1, round(new_stability, 2))
    new_difficulty = max(1.0, min(10.0, round(current_difficulty + _DIFFICULTY_DELTA[grade], 2)))

    if grade == 1:
        interval_days = 1
    elif grade == 2:
        interval_days = max(1, round(new_stability * 0.8))
    else:
        interval_days = max(1, round(new_stability))

    interval_days = min(interval_days, _MAX_INTERVAL_DAYS)
    next_review_date = date.today() + timedelta(days=interval_days)
    new_retrievability = min(1.0, grade / 4.0)

    return FSRSResult(
        new_stability=new_stability,
        new_difficulty=new_difficulty,
        new_retrievability=new_retrievability,
        next_review_date=next_review_date,
        interval_days=interval_days,
    )