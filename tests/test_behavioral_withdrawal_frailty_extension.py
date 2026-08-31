import numpy as np
import pandas as pd
from run_behavioral_withdrawal_frailty_extension import (
    _bootstrap_metrics,
    _full_rank_base,
    _support,
)

from behavioral_withdrawal_frailty_core import (
    absolute_fried_weakness,
    add_direction_categories,
    apply_grip_completion_reason,
    delayed_outcome,
    first_eligible_mask,
    fried_scores,
    stable_person_fold,
    wave_stratified_highest_fifth,
)


def test_fried_complete_scoring_requires_all_items():
    components = pd.DataFrame(
        {
            "shrinking": [0, 1, 1],
            "exhaustion": [0, 1, 1],
            "weakness": [0, 0, np.nan],
            "slowness": [0, 1, 1],
            "low_activity": [0, 0, 1],
        }
    )
    scored = fried_scores(components)
    assert scored.loc[0, "fried_category"] == "robust"
    assert scored.loc[1, "fried_category"] == "frail"
    assert pd.isna(scored.loc[2, "fried5_t1"])
    assert scored.loc[2, "fried4_no_activity_t1"] != scored.loc[2, "fried4_no_activity_t1"]


def test_health_inability_is_deficit_but_refusal_is_missing():
    base = pd.Series([np.nan, np.nan, 0.0])
    grip = pd.Series([np.nan, np.nan, 25.0])
    result = apply_grip_completion_reason(
        base,
        grip,
        health=pd.Series([1.0, 0.0, 0.0]),
        unable=pd.Series([0.0, 0.0, 0.0]),
        refusal=pd.Series([0.0, 1.0, 0.0]),
        equipment=pd.Series([0.0, 0.0, 0.0]),
    )
    assert result.iloc[0] == 1.0
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 0.0


def test_absolute_fried_grip_cutoffs_follow_sex_and_bmi():
    sex = pd.Series([2, 2, 1, 1])
    bmi = pd.Series([22, 31, 23, 30])
    grip = pd.Series([17, 22, 28, 33])
    assert absolute_fried_weakness(sex, bmi, grip).tolist() == [1.0, 0.0, 1.0, 0.0]


def test_walk_seconds_highest_fifth_is_slow():
    value = pd.Series(np.arange(1, 101, dtype=float))
    wave = pd.Series([2] * 100)
    sex = pd.Series([1] * 100)
    height = pd.Series([1.7] * 100)
    result = wave_stratified_highest_fifth(value, wave, sex, height)
    assert result.iloc[0] == 0.0
    assert result.iloc[-1] == 1.0
    assert 19 <= int(result.sum()) <= 21


def test_gain_loss_direction_categories_are_mutually_exclusive():
    behavior = pd.DataFrame(
        {
            "core_valid": [True] * 5,
            "loss_count_core": [0, 0, 1, 1, 2],
            "alcohol_gain": [0, 1, 1, 0, 0],
            "activity_gain": [0, 0, 0, 0, 0],
            "work_gain": [0, 0, 0, 0, 0],
            "alcohol_baseline": [0, 0, 1, 1, 1],
            "activity_baseline": [0, 0, 1, 1, 1],
            "work_baseline": [0, 0, 0, 0, 1],
            "alcohol_loss": [0, 0, 0, 1, 1],
            "activity_loss": [0, 0, 1, 0, 1],
            "work_loss": [0, 0, 0, 0, 0],
        }
    )
    result = add_direction_categories(behavior)
    assert result["transition_direction"].tolist() == [
        "stable", "expansion", "mixed", "contraction_1", "contraction_2_plus"
    ]


def test_delayed_outcome_requires_disease_free_intermediate_visit():
    episodes = pd.DataFrame(
        {
            "t1__diabe": [0, 0, 1, 0],
            "t2__diabe": [0, 1, 0, 0],
            "t3__diabe": [1, 1, 1, np.nan],
            "delayed_comparable_window": [True, True, True, True],
        }
    )
    result = delayed_outcome(episodes, "incident_diabetes")
    assert result.iloc[0] == 1.0
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])
    assert pd.isna(result.iloc[3])


def test_first_eligible_interval_and_person_fold_are_stable():
    episodes = pd.DataFrame(
        {"person_id": ["a", "a", "b", "b"], "t1": [3, 2, 2, 3]}
    )
    mask = first_eligible_mask(episodes, pd.Series([True, True, False, True]))
    assert mask.tolist() == [False, True, False, True]
    assert stable_person_fold("hrs", "a", 5) == stable_person_fold("hrs", "a", 5)


def test_category_support_is_gated_independently_of_binary_withdrawal():
    n = 240
    data = pd.DataFrame(
        {
            "person_id": [f"p{i}" for i in range(n)],
            "any_withdrawal": [0] * 120 + [1] * 120,
            "loss_1": [0] * 120 + [1] * 100 + [0] * 20,
            "loss_2plus": [0] * 220 + [1] * 20,
        }
    )
    y = pd.Series([0, 1] * 120, dtype=float)
    X_any = pd.DataFrame({"intercept": 1.0, "any_withdrawal": data["any_withdrawal"]})
    X_score = pd.DataFrame(
        {"intercept": 1.0, "loss_1": data["loss_1"], "loss_2plus": data["loss_2plus"]}
    )
    extension = {
        "minimum_clusters": 30,
        "events_per_parameter": 10,
        "minimum_exposed_events": 20,
        "minimum_exposed_nonevents": 20,
    }
    binary_status, _ = _support(
        data, y, X_any, ["any_withdrawal"], extension
    )
    score_status, _ = _support(
        data, y, X_score, ["loss_1", "loss_2plus"], extension
    )
    assert binary_status == "ESTIMABLE"
    assert score_status == "NOT_EVALUABLE_LOSS_2PLUS_EVENTS"


def test_redundant_routine_predictor_is_removed_before_exposure_terms():
    design = pd.DataFrame(
        {"intercept": [1.0] * 5, "x": [0, 1, 2, 3, 4], "twice_x": [0, 2, 4, 6, 8]}
    )
    result = _full_rank_base(design)
    assert result.columns.tolist() == ["intercept", "x"]
    assert result.attrs["dropped_redundant_base_columns"] == ["twice_x"]


def test_empty_design_reaches_not_evaluable_gate_without_rank_failure():
    design = pd.DataFrame(columns=["intercept", "x"], dtype=float)
    result = _full_rank_base(design)
    assert result.empty
    assert result.columns.tolist() == ["intercept", "x"]


def test_grouped_bootstrap_reports_paired_performance_differences():
    y = np.array([0, 1] * 20, dtype=float)
    people = np.array([f"p{i // 2}" for i in range(40)])
    predictions = {
        "m0": np.repeat([0.2, 0.8], 20),
        "m1": np.linspace(0.1, 0.9, 40),
        "m2": np.linspace(0.15, 0.85, 40),
        "m3": np.linspace(0.05, 0.95, 40),
    }
    rows = _bootstrap_metrics(y, predictions, people, replicates=20, seed=7)
    delta = [row for row in rows if row["metric"] == "delta_c_statistic"]
    assert len(delta) == 4
    assert all(row["bootstrap_replicates"] == 20 for row in delta)
    assert all(row["ci_low"] <= row["ci_high"] for row in delta)
