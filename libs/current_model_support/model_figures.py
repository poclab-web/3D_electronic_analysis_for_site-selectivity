"""Publication figures that are part of the accepted-model workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import Bbox


def _ensure_output_dir(save_path: str | Path) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_component_contribution_series(
    df: pd.DataFrame,
    component: str,
    entries: list[str],
    reference_entry: str,
    save_path: str | Path,
    title: str,
    *,
    contribution_column: str | None = None,
    target_column: str = "ΔΔG.expt.",
    highlighted_names: tuple[str, ...] = (),
    highlighted_labels: dict[str, str] | None = None,
    series_label: str = "Series",
    highlight_label: str = "Highlighted substrates",
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_grid: bool = True,
    avoid_label_overlap: bool = False,
    reference_label: str | None = None,
    regression_excluded_entries: tuple[str, ...] = (),
    excluded_label: str = "Excluded from linear fit",
    label_fontsize: float = 7.5,
    figure_size: tuple[float, float] = (5.3, 4.2),
    regression_line_color: str = "0.2",
    square_axes: bool = False,
    equal_axis_scale: bool = False,
    x_tick_step: float | None = None,
    y_tick_step: float | None = None,
    swap_axes: bool = False,
    label_fontweight: str = "normal",
    ax=None,
) -> pd.DataFrame:
    """Plot experimental energy changes against one contribution component.

    Entries are compared with ``reference_entry``. Named substrates can be
    overlaid with a distinct marker. Entries in
    ``regression_excluded_entries`` remain visible but are omitted from the
    fitted line and correlation. The returned rows include centered values and
    a ``used_for_linear_fit`` flag.
    """
    contribution_column = contribution_column or f"{component}_contribution"
    required = {"entry", "name", contribution_column, target_column}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}")

    highlighted_labels = highlighted_labels or {}
    selected = (
        df["entry"].astype(str).isin(entries)
        | df["name"].isin(highlighted_names)
        | df["entry"].astype(str).eq(reference_entry)
    )
    subset = df.loc[selected].copy()
    reference_rows = subset.loc[subset["entry"].astype(str).eq(reference_entry)]
    if reference_rows.empty:
        raise ValueError(
            f"Reference entry '{reference_entry}' was not found in the selected data."
        )

    reference = reference_rows.iloc[0]
    subset["contribution_change"] = (
        subset[contribution_column] - float(reference[contribution_column])
    )
    subset["experimental_change"] = (
        subset[target_column] - float(reference[target_column])
    )
    if swap_axes:
        x_column, y_column = "experimental_change", "contribution_change"
    else:
        x_column, y_column = "contribution_change", "experimental_change"
    points = subset.loc[subset["entry"].astype(str).ne(reference_entry)].copy()
    excluded_from_fit = points["entry"].astype(str).isin(regression_excluded_entries)
    fit_points = points.loc[~excluded_from_fit].copy()
    if len(fit_points) < 3:
        raise ValueError(
            "At least three non-reference points are required for a component series plot."
        )

    slope, intercept = np.polyfit(fit_points[x_column], fit_points[y_column], 1)
    correlation = float(
        np.corrcoef(fit_points[x_column], fit_points[y_column])[0, 1]
    )
    highlighted = points["name"].isin(highlighted_names) & ~excluded_from_fit
    base_points = points.loc[~highlighted & ~excluded_from_fit]
    highlighted_points = points.loc[highlighted]
    excluded_points = points.loc[excluded_from_fit]

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figure_size)
    else:
        fig = ax.figure
    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.axvline(0, color="0.7", linewidth=0.8)
    ax.scatter(
        base_points[x_column],
        base_points[y_column],
        color="#4e79a7",
        s=48,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
        label=series_label,
    )
    if not highlighted_points.empty:
        ax.scatter(
            highlighted_points[x_column],
            highlighted_points[y_column],
            color="#f28e2b",
            marker="D",
            s=62,
            edgecolors="black",
            linewidths=0.55,
            zorder=4,
            label=highlight_label,
        )
    if not excluded_points.empty:
        ax.scatter(
            excluded_points[x_column],
            excluded_points[y_column],
            color="#d9534f",
            marker="X",
            s=72,
            edgecolors="black",
            linewidths=0.55,
            zorder=4,
            label=excluded_label,
        )
    ax.scatter([0], [0], marker="*", color="black", s=82, zorder=5)

    x_line = np.linspace(
        float(fit_points[x_column].min()) - 0.08,
        float(fit_points[x_column].max()) + 0.08,
        100,
    )
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color=regression_line_color,
        linewidth=1.1,
        zorder=0,
    )
    ax.text(
        0.04,
        0.96,
        rf"$R^2$ = {correlation ** 2:.2f}, N = {len(fit_points)}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
    )
    ax.set(
        title=title,
        xlabel=xlabel
        or f"{component.capitalize()} contribution change vs {reference_entry} [kcal/mol]",
        ylabel=ylabel
        or rf"Experimental $\Delta\Delta G^\ddagger$ change vs {reference_entry} [kcal/mol]",
    )
    if x_tick_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))
    if y_tick_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_step))
    if equal_axis_scale:
        ax.set_aspect("equal", adjustable="box")
    elif square_axes:
        ax.set_box_aspect(1)
    labels = [
        (
            float(row[x_column]),
            float(row[y_column]),
            highlighted_labels.get(str(row["name"]), str(row["entry"])),
        )
        for _, row in points.iterrows()
    ]
    if reference_label:
        labels.append((0.0, 0.0, reference_label))
    if avoid_label_overlap:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        occupied = []
        for x_value, y_value, _ in labels:
            display_x, display_y = ax.transData.transform((x_value, y_value))
            occupied.append(Bbox.from_bounds(display_x - 6, display_y - 6, 12, 12))
        offsets = (
            (5, 5),
            (5, -12),
            (-20, 5),
            (-20, -12),
            (6, 15),
            (-22, 15),
            (6, -22),
            (-22, -22),
            (24, 3),
            (-38, 3),
            (24, -13),
            (-38, -13),
            (24, 16),
            (-38, 16),
            (24, -25),
            (-38, -25),
        )
        for x_value, y_value, label in sorted(
            labels, key=lambda item: (item[1], item[0])
        ):
            best_offset = offsets[0]
            best_score = float("inf")
            for offset in offsets:
                trial = ax.annotate(
                    label,
                    (x_value, y_value),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=label_fontsize,
                    fontweight=label_fontweight,
                )
                fig.canvas.draw()
                box = trial.get_window_extent(renderer=renderer).expanded(1.08, 1.18)
                overlap = sum(
                    max(0.0, min(box.x1, other.x1) - max(box.x0, other.x0))
                    * max(0.0, min(box.y1, other.y1) - max(box.y0, other.y0))
                    for other in occupied
                )
                outside = (
                    max(0.0, ax.bbox.x0 - box.x0)
                    + max(0.0, box.x1 - ax.bbox.x1)
                    + max(0.0, ax.bbox.y0 - box.y0)
                    + max(0.0, box.y1 - ax.bbox.y1)
                )
                score = overlap + 1000.0 * outside + 0.01 * (
                    offset[0] ** 2 + offset[1] ** 2
                )
                trial.remove()
                if score < best_score:
                    best_score = score
                    best_offset = offset
            annotation = ax.annotate(
                label,
                (x_value, y_value),
                xytext=best_offset,
                textcoords="offset points",
                fontsize=label_fontsize,
                fontweight=label_fontweight,
                arrowprops={"arrowstyle": "-", "color": "0.55", "lw": 0.45},
            )
            fig.canvas.draw()
            occupied.append(
                annotation.get_window_extent(renderer=renderer).expanded(1.08, 1.18)
            )
    else:
        for x_value, y_value, label in labels:
            ax.annotate(
                label,
                (x_value, y_value),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=label_fontsize,
                fontweight=label_fontweight,
            )
    if not highlighted_points.empty or not excluded_points.empty:
        ax.legend(
            frameon=False,
            fontsize=9,
            loc="upper left",
            bbox_to_anchor=(0.025, 0.885),
            borderaxespad=0,
        )
    ax.title.set_fontsize(13)
    ax.xaxis.label.set_fontsize(11.5)
    ax.yaxis.label.set_fontsize(11.5)
    ax.tick_params(axis="both", labelsize=10.5)
    if show_grid:
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.4)
    else:
        ax.grid(False)
    if owns_figure:
        fig.tight_layout()
        fig.savefig(_ensure_output_dir(save_path), dpi=500)
        plt.close(fig)
    subset["used_for_linear_fit"] = (
        subset["entry"].astype(str).ne(reference_entry)
        & ~subset["entry"].astype(str).isin(regression_excluded_entries)
    )
    return subset
