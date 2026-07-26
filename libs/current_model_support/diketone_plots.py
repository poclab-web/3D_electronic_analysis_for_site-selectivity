"""Concentration-profile figures for the accepted diketone network model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def reaction_concentration_plot_complex(
    barriers,
    temperature: float = 298.15,
    initial_concentration: float = 100,
    save_path: str | Path = "simulation_complex.png",
    **legacy_kwargs,
) -> None:
    """Plot the eight-species concentration profile for a diketone network.

    ``barriers`` contains twelve activation free energies in the order
    1, 2, 3, 4, 13, 14, 23, 24, 31, 32, 41, and 42. The plotting calculation
    uses relative Eyring rates, so removing the common prefactor changes only
    the time scale and not concentrations or selectivities.

    ``T`` and ``a0`` are accepted as compatibility aliases for the historical
    plotting interface.
    """
    if "T" in legacy_kwargs:
        temperature = legacy_kwargs.pop("T")
    if "a0" in legacy_kwargs:
        initial_concentration = legacy_kwargs.pop("a0")
    if legacy_kwargs:
        unexpected = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    gas_constant = 1.987e-3
    delta_g = np.asarray(barriers, dtype=float)
    if delta_g.size != 12:
        raise ValueError("barriers must contain 12 activation free energies.")
    if not np.all(np.isfinite(delta_g)):
        raise ValueError("barriers contains non-finite values.")

    log_rates = -delta_g / (gas_constant * temperature)
    log_rates = log_rates - np.max(log_rates)
    rates = np.exp(np.clip(log_rates, -745.0, 0.0))
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
    ) = rates

    k1p_sum = k13p + k14p
    k2p_sum = k23p + k24p
    k3p_sum = k31p + k32p
    k4p_sum = k41p + k42p
    initial_total_rate = k1 + k2 + k3 + k4

    positive_rates = rates[rates > 0]
    if positive_rates.size == 0 or initial_total_rate <= 0:
        raise ValueError("All kinetic rates are zero after scaling.")

    base_times = np.logspace(-8, 8, 900, base=10)
    time_parts = [np.array([0.0])]
    for rate in positive_rates:
        time_parts.append(base_times / rate)
    time = np.unique(np.concatenate(time_parts))
    time = time[np.isfinite(time)]
    time.sort()

    def exp_decay(rate):
        if rate <= 0:
            return np.ones_like(time)
        return np.exp(-np.clip(rate * time, 0.0, 745.0))

    def one_minus_exp_over_rate(rate):
        if rate <= 0:
            return time.copy()
        return -np.expm1(-np.clip(rate * time, 0.0, 745.0)) / rate

    def safe_species(values):
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=initial_concentration,
            neginf=0.0,
        )
        return np.clip(values, 0.0, initial_concentration)

    initial = initial_concentration * exp_decay(initial_total_rate)

    def intermediate(path_rate, followup_rate):
        if path_rate <= 0:
            return np.zeros_like(time)
        scale = max(abs(followup_rate), abs(initial_total_rate), 1.0)
        if abs(followup_rate - initial_total_rate) <= 1e-10 * scale:
            return safe_species(
                path_rate
                * initial_concentration
                * time
                * exp_decay(initial_total_rate)
            )
        return safe_species(
            (path_rate * initial_concentration / (followup_rate - initial_total_rate))
            * (exp_decay(initial_total_rate) - exp_decay(followup_rate))
        )

    p1 = intermediate(k1, k1p_sum)
    p2 = intermediate(k2, k2p_sum)
    p3 = intermediate(k3, k3p_sum)
    p4 = intermediate(k4, k4p_sum)
    intermediate_total = p1 + p2 + p3 + p4
    max_index = int(np.nanargmax(intermediate_total))

    def final_product(path_rate, product_rate, followup_rate):
        if path_rate <= 0 or product_rate <= 0:
            return np.zeros_like(time)
        scale = max(abs(followup_rate), abs(initial_total_rate), 1.0)
        if abs(followup_rate - initial_total_rate) <= 1e-10 * scale:
            if initial_total_rate <= 0:
                return np.zeros_like(time)
            integral = (
                1.0
                - exp_decay(initial_total_rate)
                * (1.0 + initial_total_rate * time)
            ) / (initial_total_rate * initial_total_rate)
            return safe_species(
                path_rate * product_rate * initial_concentration * integral
            )
        first_term = one_minus_exp_over_rate(initial_total_rate)
        second_term = one_minus_exp_over_rate(followup_rate)
        return safe_species(
            (
                path_rate
                * product_rate
                * initial_concentration
                / (followup_rate - initial_total_rate)
            )
            * (first_term - second_term)
        )

    p13 = final_product(k1, k13p, k1p_sum) + final_product(k3, k31p, k3p_sum)
    p14 = final_product(k1, k14p, k1p_sum) + final_product(k4, k41p, k4p_sum)
    p23 = final_product(k2, k23p, k2p_sum) + final_product(k3, k32p, k3p_sum)
    p24 = final_product(k2, k24p, k2p_sum) + final_product(k4, k42p, k4p_sum)
    final_total = p13 + p14 + p23 + p24

    progress = p1 / 2 + p2 / 2 + p3 / 2 + p4 / 2 + final_total
    progress = np.nan_to_num(
        progress,
        nan=0.0,
        posinf=initial_concentration,
        neginf=0.0,
    )
    progress = np.maximum.accumulate(progress)
    if initial_concentration > 0:
        progress = np.clip(progress / initial_concentration, 0.0, 1.0)
        p1, p2, p3, p4, p13, p14, p23, p24 = [
            np.clip(values / initial_concentration, 0.0, 1.0)
            for values in (p1, p2, p3, p4, p13, p14, p23, p24)
        ]
        intermediate_total = np.clip(p1 + p2 + p3 + p4, 0.0, 1.0)
        final_total = np.clip(p13 + p14 + p23 + p24, 0.0, 1.0)
        initial = np.clip(initial / initial_concentration, 0.0, 1.0)

    maximum_total = intermediate_total[max_index]
    if maximum_total > 0:
        peak_percentages = [
            p1[max_index] * 100.0,
            p2[max_index] * 100.0,
            p3[max_index] * 100.0,
            p4[max_index] * 100.0,
        ]
    else:
        peak_percentages = [0.0, 0.0, 0.0, 0.0]

    figure, axis = plt.subplots(figsize=(3.5, 2.5))
    figure.patch.set_alpha(0.0)
    intermediate_colors = ["red", "tab:pink", "blue", "tab:blue"]
    product_facecolors = ["red", "red", "tab:pink", "tab:pink"]
    product_edgecolors = ["blue", "tab:blue", "blue", "tab:blue"]
    hatches = ["///", "\\\\", "xx", ".."]
    labels = [
        rf"$\bf{{1}}$ {peak_percentages[0]:4.1f}%",
        rf"$\bf{{2}}$ {peak_percentages[1]:4.1f}%",
        rf"$\bf{{3}}$ {peak_percentages[2]:4.1f}%",
        rf"$\bf{{4}}$ {peak_percentages[3]:4.1f}%",
        rf"$\bf{{1-3}}$ {p13[-1] * 100:4.1f}%",
        rf"$\bf{{1-4}}$ {p14[-1] * 100:4.1f}%",
        rf"$\bf{{2-3}}$ {p23[-1] * 100:4.1f}%",
        rf"$\bf{{2-4}}$ {p24[-1] * 100:4.1f}%",
    ]
    polygons = axis.stackplot(
        progress,
        p1,
        p2,
        p3,
        p4,
        p13,
        p14,
        p23,
        p24,
        colors=intermediate_colors + product_facecolors,
        labels=labels,
    )
    for polygon in polygons[:4]:
        polygon.set_alpha(0.6)
    for index, polygon in enumerate(polygons[4:]):
        polygon.set_alpha(0.5)
        polygon.set_hatch(hatches[index])
        polygon.set_edgecolor(product_edgecolors[index])
        polygon.set_linewidth(0.6)

    axis.plot(progress, intermediate_total, color="gray", linestyle="-")
    maximum_progress = progress[max_index]
    maximum_concentration = intermediate_total[max_index]
    axis.plot(
        [maximum_progress, maximum_progress],
        [0, maximum_concentration],
        color="green",
        linestyle="--",
        linewidth=1.0,
    )
    axis.plot([1, 1], [0, 1], color="purple", linestyle="--", linewidth=1.0)
    axis.plot(
        progress,
        intermediate_total + final_total,
        color="tab:gray",
        linestyle="-",
    )
    axis.set_xlabel("reaction progress [-]")
    axis.set_ylabel("concentration [-]")
    axis.set_xticks([0, 0.5, 1])
    axis.set_yticks([0, 0.5, 1])
    axis.set_ylim(-0.02, 1.01)
    axis.set_xlim(-0.01, 1.01)

    handles, legend_labels = axis.get_legend_handles_labels()
    first_legend = axis.legend(
        handles[:4],
        legend_labels[:4],
        loc="upper left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fontsize=9,
        borderpad=0.2,
        labelspacing=0.2,
        handlelength=1.0,
        handletextpad=0.3,
        borderaxespad=0.2,
        frameon=False,
        framealpha=0.8,
        title="Max point",
        title_fontsize=9,
    )
    second_legend = axis.legend(
        handles[4:],
        legend_labels[4:],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.02),
        ncol=1,
        fontsize=9,
        borderpad=0.2,
        labelspacing=0.2,
        handlelength=1.0,
        handletextpad=0.3,
        borderaxespad=0.2,
        frameon=False,
        framealpha=0.8,
        title="Final point",
        title_fontsize=9,
    )
    axis.add_artist(first_legend)
    first_legend.get_title().set_color("green")
    second_legend.get_title().set_color("purple")
    for text in first_legend.get_texts():
        text.set_color("green")
    for text in second_legend.get_texts():
        text.set_color("purple")
    figure.tight_layout()
    figure.subplots_adjust(right=0.7)
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=500, transparent=False)
    plt.close(figure)
