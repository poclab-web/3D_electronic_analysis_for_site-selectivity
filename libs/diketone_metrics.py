"""Semiquantitative eight-selectivity evaluation for diketone networks."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


GAS_CONSTANT = 1.987e-3
GROUPS = ("a", "b", "c", "d", "e", "f")
ENTRY_ORDER = ("1", "2", "3", "4", "13", "14", "23", "24", "31", "32", "41", "42")
PRODUCT_ORDER = ("1-3", "1-4", "2-3", "2-4")
TEMPERATURE_BY_GROUP = {
    "a": 273.15,
    "b": 298.15,
    "c": 298.15,
    "d": 298.15,
    "e": 298.15,
    "f": 298.15,
}

INITIAL_IDENTITY_TARGETS = {
    "a": "2",
    "b": "1",
    "c": "2",
    "d": "1",
    "e": "2",
    "f": "1",
}
FINAL_IDENTITY_TARGETS = {"a": "2-4", "e": "2-3"}


MAX_TARGETS = {
    # observed_family_percent and observed_dr_percent are approximate values
    # read from Scheme 3. Lower-bound entries use the reported bound.
    "a": {"family": ("1", "2"), "major": "2", "family_percent": 88.0, "dr_percent": 87.0},
    "b": {"family": ("1", "2"), "major": "1", "family_percent": 76.3, "dr_percent": 100.0},
    "c": {"family": ("1", "2"), "major": "2", "family_percent": 88.0, "dr_percent": 99.0},
    "d": {"family": ("1", "2"), "major": "1", "family_percent": 77.0, "dr_percent": 95.0},
    "e": {"family": ("1", "2"), "major": "2", "family_percent": 76.0, "dr_percent": 100.0},
    "f": {"family": ("1", "2"), "major": "1", "family_percent": 89.0, "dr_percent": 100.0},
}

FINAL_TARGETS = {
    "a": {
        "top": "2-4",
        "first_axis_major": "2",
        "first_axis_percent": 89.0,
        "second_axis_major": "4",
        "second_axis_percent": 90.0,
    },
    "e": {
        "top": "2-3",
        "first_axis_major": "2",
        "first_axis_percent": 100.0,
        "second_axis_major": "3",
        "second_axis_percent": 100.0,
    },
}


def rate(delta_g: float, temperature: float) -> float:
    """Convert an activation free energy in kcal/mol to a relative rate."""
    return math.exp(-delta_g / (GAS_CONSTANT * temperature))


def _intermediate_concentration(
    initial_path_rate: float,
    followup_rate: float,
    initial_total_rate: float,
    time_value: float,
) -> float:
    """Return a monoalcohol concentration, including the equal-rate limit."""
    if np.isclose(followup_rate, initial_total_rate, rtol=1.0e-12, atol=1.0e-15):
        return (
            initial_path_rate
            * time_value
            * math.exp(-initial_total_rate * time_value)
        )
    return initial_path_rate / (followup_rate - initial_total_rate) * (
        math.exp(-initial_total_rate * time_value)
        - math.exp(-followup_rate * time_value)
    )


def _final_product_concentration(
    initial_path_rate: float,
    product_path_rate: float,
    followup_rate: float,
    initial_total_rate: float,
    time_value: float,
) -> float:
    """Return a diol concentration, including the equal-rate limit."""
    if np.isclose(followup_rate, initial_total_rate, rtol=1.0e-12, atol=1.0e-15):
        scaled_time = initial_total_rate * time_value
        return initial_path_rate * product_path_rate / initial_total_rate**2 * (
            1.0 - math.exp(-scaled_time) * (1.0 + scaled_time)
        )
    first_integral = -math.expm1(-initial_total_rate * time_value) / initial_total_rate
    second_integral = -math.expm1(-followup_rate * time_value) / followup_rate
    return initial_path_rate * product_path_rate / (
        followup_rate - initial_total_rate
    ) * (first_integral - second_integral)


def simulate_barrier_network(
    barriers: np.ndarray,
    temperature: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Simulate one 12-barrier diketone network with the canonical time grid.

    ``barriers`` follows :data:`ENTRY_ORDER` and is expressed in kcal/mol;
    ``temperature`` is in kelvin. All rates are divided by their largest value;
    this prevents overflow and leaves every concentration and selectivity
    unchanged because the time grid is scaled by the same common factor.

    Returns the absolute peak monoalcohol percentages and normalized endpoint
    diol percentages, in :data:`ENTRY_ORDER` and :data:`PRODUCT_ORDER` order.
    """
    barriers = np.asarray(barriers, dtype=float)
    if barriers.shape != (len(ENTRY_ORDER),):
        raise ValueError(
            f"Expected {len(ENTRY_ORDER)} barriers in ENTRY_ORDER, got {barriers.shape}."
        )
    if not np.isfinite(barriers).all() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Diketone barriers and temperature must be finite; temperature must be positive.")
    log_rates = -barriers / (GAS_CONSTANT * temperature)
    normalized_rates = np.exp(np.clip(log_rates - np.max(log_rates), -745.0, 0.0))
    (
        k1,
        k2,
        k3,
        k4,
        k13p,
        k14p,
        k23p,
        k24p,
        k31p,
        k32p,
        k41p,
        k42p,
    ) = normalized_rates

    k1p_sum = k13p + k14p
    k2p_sum = k23p + k24p
    k3p_sum = k31p + k32p
    k4p_sum = k41p + k42p
    ka = k1 + k2 + k3 + k4
    max_rate = max(normalized_rates)
    time_points = np.logspace(-10, 10, 1000) / max_rate

    p1 = [_intermediate_concentration(k1, k1p_sum, ka, t) for t in time_points]
    p2 = [_intermediate_concentration(k2, k2p_sum, ka, t) for t in time_points]
    p3 = [_intermediate_concentration(k3, k3p_sum, ka, t) for t in time_points]
    p4 = [_intermediate_concentration(k4, k4p_sum, ka, t) for t in time_points]
    intermediate_total = [sum(values) for values in zip(p1, p2, p3, p4)]
    max_index = int(np.argmax(intermediate_total))
    intermediate_abs = {
        "1": p1[max_index] * 100,
        "2": p2[max_index] * 100,
        "3": p3[max_index] * 100,
        "4": p4[max_index] * 100,
    }

    final_abs = {
        "1-3": (
            _final_product_concentration(k1, k13p, k1p_sum, ka, time_points[-1])
            + _final_product_concentration(k3, k31p, k3p_sum, ka, time_points[-1])
        )
        * 100,
        "1-4": (
            _final_product_concentration(k1, k14p, k1p_sum, ka, time_points[-1])
            + _final_product_concentration(k4, k41p, k4p_sum, ka, time_points[-1])
        )
        * 100,
        "2-3": (
            _final_product_concentration(k2, k23p, k2p_sum, ka, time_points[-1])
            + _final_product_concentration(k3, k32p, k3p_sum, ka, time_points[-1])
        )
        * 100,
        "2-4": (
            _final_product_concentration(k2, k24p, k2p_sum, ka, time_points[-1])
            + _final_product_concentration(k4, k42p, k4p_sum, ka, time_points[-1])
        )
        * 100,
    }
    final_total = sum(final_abs.values())
    final_frac = {label: value / final_total * 100 for label, value in final_abs.items()}
    return intermediate_abs, final_frac


def simulate_full(pred_by_entry: dict[str, float], group: str) -> dict[str, object]:
    """Simulate peak monoalcohols and endpoint diols for a named network.

    ``pred_by_entry`` maps the twelve ``<group><suffix>`` labels to barriers in
    kcal/mol. The temperature is selected from :data:`TEMPERATURE_BY_GROUP`.
    """
    barriers = np.asarray(
        [pred_by_entry[f"{group}{suffix}"] for suffix in ENTRY_ORDER], dtype=float
    )
    intermediate_abs, final_frac = simulate_barrier_network(
        barriers,
        TEMPERATURE_BY_GROUP[group],
    )
    return {"intermediate_abs": intermediate_abs, "final_frac": final_frac}


def save_selectivity_identity_summary(
    frame: pd.DataFrame,
    save_path: str | Path,
) -> pd.DataFrame:
    """Save the eight reported diketone identity checks.

    Six checks compare the major monoalcohol at its maximum concentration;
    two checks compare the major endpoint diol. This function preserves the
    identity-only summary used by the accepted-model runner, while
    :func:`evaluate_predictions` provides the semiquantitative evaluation.
    """
    pred_by_entry = {
        str(entry): float(value)
        for entry, value in zip(frame["entry"].astype(str), frame["prediction"])
        if str(entry)[:1] in set("abcdef") and pd.notna(value)
    }
    rows: list[dict[str, object]] = []
    for group, expected in INITIAL_IDENTITY_TARGETS.items():
        values = simulate_full(pred_by_entry, group)["intermediate_abs"]
        predicted = max(values, key=values.get)
        rows.append(
            {
                "group": group,
                "stage": "initial",
                "expected": expected,
                "predicted": predicted,
                "ok": predicted == expected,
                "target_percent": values[expected],
            }
        )
    for group, expected in FINAL_IDENTITY_TARGETS.items():
        values = simulate_full(pred_by_entry, group)["final_frac"]
        predicted = max(values, key=values.get)
        rows.append(
            {
                "group": group,
                "stage": "final",
                "expected": expected,
                "predicted": predicted,
                "ok": predicted == expected,
                "target_percent": values[expected],
            }
        )
    summary = pd.DataFrame(rows)
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def max_metrics(sim: dict[str, object], group: str) -> list[dict[str, object]]:
    """Compare peak monoalcohol family and diastereomer ratios with experiment."""
    target = MAX_TARGETS[group]
    values = sim["intermediate_abs"]
    family_total = sum(values[label] for label in target["family"])
    major_value = values[target["major"]]
    predicted_dr = major_value / family_total * 100 if family_total else 0.0
    predicted_top = max(values, key=values.get)
    return [
        {
            "target": f"{group}:max_family",
            "kind": "max_family_percent",
            "expected_label": "+".join(target["family"]),
            "predicted_label": "+".join(target["family"]),
            "top_match": predicted_top == target["major"],
            "predicted_percent": family_total,
            "observed_percent": target["family_percent"],
            "abs_error_percent": abs(family_total - target["family_percent"]),
        },
        {
            "target": f"{group}:max_dr",
            "kind": "max_dr_percent",
            "expected_label": target["major"],
            "predicted_label": predicted_top,
            "top_match": predicted_top == target["major"],
            "predicted_percent": predicted_dr,
            "observed_percent": target["dr_percent"],
            "abs_error_percent": abs(predicted_dr - target["dr_percent"]),
        },
    ]


def final_axis_percent(final_frac: dict[str, float], major: str) -> float:
    """Marginalize four diol fractions along one stereochemical axis."""
    if major == "1":
        return final_frac["1-3"] + final_frac["1-4"]
    if major == "2":
        return final_frac["2-3"] + final_frac["2-4"]
    if major == "3":
        return final_frac["1-3"] + final_frac["2-3"]
    if major == "4":
        return final_frac["1-4"] + final_frac["2-4"]
    raise ValueError(major)


def final_metrics(sim: dict[str, object], group: str) -> list[dict[str, object]]:
    """Compare endpoint diol identity and marginal ratios with experiment."""
    target = FINAL_TARGETS[group]
    final_frac = sim["final_frac"]
    predicted_top = max(final_frac, key=final_frac.get)
    first_percent = final_axis_percent(final_frac, target["first_axis_major"])
    second_percent = final_axis_percent(final_frac, target["second_axis_major"])
    return [
        {
            "target": f"{group}:final_top",
            "kind": "final_top_product",
            "expected_label": target["top"],
            "predicted_label": predicted_top,
            "top_match": predicted_top == target["top"],
            "predicted_percent": final_frac[target["top"]],
            "observed_percent": np.nan,
            "abs_error_percent": np.nan,
        },
        {
            "target": f"{group}:final_axis_{target['first_axis_major']}",
            "kind": "final_dr_percent",
            "expected_label": target["first_axis_major"],
            "predicted_label": target["first_axis_major"],
            "top_match": first_percent >= 50.0,
            "predicted_percent": first_percent,
            "observed_percent": target["first_axis_percent"],
            "abs_error_percent": abs(first_percent - target["first_axis_percent"]),
        },
        {
            "target": f"{group}:final_axis_{target['second_axis_major']}",
            "kind": "final_dr_percent",
            "expected_label": target["second_axis_major"],
            "predicted_label": target["second_axis_major"],
            "top_match": second_percent >= 50.0,
            "predicted_percent": second_percent,
            "observed_percent": target["second_axis_percent"],
            "abs_error_percent": abs(second_percent - target["second_axis_percent"]),
        },
    ]


def evaluate_predictions(
    label: str,
    pred_by_entry: dict[str, float],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate all reported diketone checks and return summary/detail rows."""
    rows = []
    for group in GROUPS:
        sim = simulate_full(pred_by_entry, group)
        rows.extend(max_metrics(sim, group))
        if group in FINAL_TARGETS:
            rows.extend(final_metrics(sim, group))
    detail = [{"condition": label, **row} for row in rows]
    quantified = [row for row in rows if not pd.isna(row["observed_percent"])]
    top_checks = [row for row in rows if pd.notna(row["top_match"])]
    summary = {
        "condition": label,
        "top_checks_passed": sum(bool(row["top_match"]) for row in top_checks),
        "top_checks_total": len(top_checks),
        "semiquant_metric_n": len(quantified),
        "semiquant_mae_percent": float(np.mean([row["abs_error_percent"] for row in quantified])),
        "semiquant_max_error_percent": float(np.max([row["abs_error_percent"] for row in quantified])),
    }
    return summary, detail
