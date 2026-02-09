from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np

def plot_origin_regression_series(
    series_data: dict[str, np.ndarray],
    save_path: str,
    colors: list[str] | None = None,
    text_positions: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Plot multiple origin-through regressions and return ΔΔG values.

    Parameters
    ----------
    series_data : dict[str, np.ndarray]
        Dict of label -> array shape (N, 2).
        Each array column:
            [:, 0] -> x values
            [:, 1] -> y values
    save_path : str
        Output image file path, e.g. "figures/series_fit.png".
        Intermediate folders are created if they do not exist.
    colors : list[str] or None, optional
        List of colors for each series in the order of series_data.keys().
        If None, default colors are used.
    text_positions : dict[str, tuple[float, float]] or None, optional
        Explicit positions for regression text, keyed by series label.
        Example: {"Series A": (1.0, 2.0), "Series B": (0.5, 1.5)}
        If a label is not present in this dict, an automatic position
        (offset from the regression line) is used.

    Returns
    -------
    dict[str, float]
        Dictionary {label: ΔΔG} for each series, in kcal/mol.
    """
    # --- ensure output folder exists ---
    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # default colors if not provided
    if colors is None:
        colors = ["darkorange", "blue", "green"]

    if text_positions is None:
        text_positions = {}

    fig, ax = plt.subplots(figsize=(3, 3))
    delta_delta_g_values: dict[str, float] = {}

    # gather all x,y for axis range
    all_x = np.concatenate([arr[:, 0] for arr in series_data.values()])
    all_y = np.concatenate([arr[:, 1] for arr in series_data.values()])

    x_min_data = float(np.nanmin(all_x))
    x_max_data = float(np.nanmax(all_x))
    y_min_data = float(np.nanmin(all_y))
    y_max_data = float(np.nanmax(all_y))

    x_span = max(x_max_data - x_min_data, 1.0)
    y_span = max(y_max_data - y_min_data, 1.0)
    x_min = x_min_data - 0.1 * x_span
    x_max = x_max_data + 0.1 * x_span
    y_min = y_min_data - 0.1 * y_span
    y_max = y_max_data + 0.1 * y_span

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    n_series = len(series_data)

    # --- main loop over each series ---
    for i, (label, arr) in enumerate(series_data.items()):
        x = arr[:, 0]
        y = arr[:, 1]

        # origin-through regression: y = slope * x
        slope = np.sum(x * y) / np.sum(x ** 2)
        y_pred = slope * x

        # R^2 for origin-through regression
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot

        # ΔΔG = -1.99 * 0.273 * ln(slope)
        delta_delta_g = -1.99 * 0.273 * np.log(slope)
        delta_delta_g_values[label] = float(round(delta_delta_g, 2))

        # plotting of line and points
        color = colors[i % len(colors)]
        x_fit = np.linspace(0.0, x.max(), 100)
        y_fit = slope * x_fit

        ax.plot(
            x_fit,
            y_fit,
            color=color,
            linestyle="-",
            linewidth=1.5,
            zorder=0,
        )
        ax.scatter(
            x,
            y,
            facecolors="white",
            edgecolors=color,
            s=50,
            linewidth=1.5,
            zorder=1,
            label=label,
        )

        # ===== text position =====
        if label in text_positions:
            # 明示的に座標が指定されている場合はそれを使う
            x_text, y_text = text_positions[label]
        else:
            # 自動配置（直線に直交する方向にオフセット）
            x0 = float(np.max(x))
            y0 = slope * x0

            nx = -slope
            ny = 1.0
            n_norm = np.hypot(nx, ny)
            if n_norm == 0.0:
                nx, ny = 0.0, 1.0
                n_norm = 1.0
            nx /= n_norm
            ny /= n_norm

            base_d = 0.18 * max(x_span, y_span)
            d = base_d * (1.0 + 0.1 * (i - (n_series - 1) / 2.0))

            margin_y = 0.05 * y_span
            margin_x = 0.05 * x_span

            def candidate_position(sign: float) -> tuple[float, float]:
                xt = x0 + sign * d * nx
                yt = y0 + sign * d * ny
                return xt, yt

            candidates = [1.0, -1.0]
            x_text, y_text = x0, y0
            chosen = False
            for sgn in candidates:
                xt, yt = candidate_position(sgn)
                if (x_min + margin_x <= xt <= x_max - margin_x) and (
                    y_min + margin_y <= yt <= y_max - margin_y
                ):
                    x_text, y_text = xt, yt
                    chosen = True
                    break

            if not chosen:
                xt, yt = candidate_position(1.0)
                xt = min(max(xt, x_min + margin_x), x_max - margin_x)
                yt = min(max(yt, y_min + margin_y), y_max - margin_y)
                x_text, y_text = xt, yt

        ax.text(
            x_text,
            y_text,
            f"y = {slope:.2f}x\n$R^2$ = {r_squared:.3f}",
            fontsize=9,
            color=color,
            va="center",
            ha="center",
            zorder=2,
        )

    # origin point
    ax.scatter(0, 0, color="black", s=10, zorder=2)

    ax.set_xlabel(r"$-\log([\mathrm{A}_1]_f/[\mathrm{A}_1]_0)$")
    ax.set_ylabel(r"$-\log([\mathrm{A}_2]_f/[\mathrm{A}_2]_0)$")
    fig.patch.set_alpha(0.0)

    ax.set_xticks(np.arange(0, 4))
    ax.set_yticks(np.arange(0, 4))

    plt.tight_layout()
    fig.savefig(
        save_path,
        dpi=500,
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)

    return delta_delta_g_values
def plot_dft_bar_dual(save_path: str) -> None:
    """Plot DFT ΔG‡ (left panel) and experimental ΔΔG‡ (right panel) and save the figure.

    The function reproduces the two-panel bar chart:
      - Left: ΔG‡DFT for four reducing agents and seven substrates.
      - Right: ΔΔG‡expt. for five substrates.

    Parameters
    ----------
    save_path : str
        Output image file path (e.g., "data/DFT.png").
        If intermediate folders in the path do not exist, they are created.
    """
    # --- ensure output folder exists ---
    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Labels
    reducing_agents = [
        r"NaBH$_4$",
        r"Na(MeO)BH$_3$",
        r"Na(MeO)$_2$BH$_2$",
        r"Na(MeO)$_3$BH",
    ]
    substrates_full = [
        r"Methanolysis",
        r"Ph$_2$CO",
        r"$\mathit{eq}$-CyO",
        r"$\mathit{ax}$-CyO",
        r"$\mathit{p}$-NO$_2$ PhMeCO",
        r"PhMeCO",
        r"$\mathit{p}$-MeO PhMeCO",
    ]

    substrates_delta = [
        r"$\mathit{eq}$-CyO",
        r"$\mathit{ax}$-CyO",
        r"$\mathit{p}$-NO$_2$ PhMeCO",
        r"PhMeCO",
        r"$\mathit{p}$-MeO PhMeCO",
    ]

    # ΔG‡ data
    data_full = np.array([
        [21.8, 26.5, 26.5, 29.0, 25.0, 27.4, 30.6],
        [23.0, 14.8, 13.8, 15.9, 14.2, 15.4, 18.4],
        [26.3, 15.6, 15.4, 17.3, 14.0, 16.2, 17.6],
        [28.4, 19.2, 16.2, 18.7, 15.6, 19.0, 20.1],
    ])

    # ΔΔG‡ data
    data_delta = np.array([
        [-2.56, -1.72, -2.02, -0.25, 0.56],
    ])

    # Colors for 7 substrates
    colors = [
        "#1f77b4",  # MeOH
        "#7f7f7f",  # Ph₂CO
        "#ff7f0e",  # eq-CyO
        "#ffbb78",  # ax-CyO
        "#2ca02c",  # 4-NO2
        "#98df8a",  # PhMeCO
        "#c7e9c0",  # 4-MeO
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(8, 3),
        gridspec_kw={"width_ratios": [6.0, 1.0]},
    )

    # ===== Left panel: ΔG‡ =====
    x1 = np.arange(len(reducing_agents)) * 1.1
    width = 0.15

    for i, (label, color) in enumerate(zip(substrates_full, colors)):
        offset = (i - 3) * width
        bars = ax1.bar(x1 + offset, data_full[:, i], width, label=label, color=color)
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.25,
                f"{h:.1f}",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax1.set_xticks(x1)
    ax1.set_xticklabels(reducing_agents)
    ax1.margins(x=0.01)
    ax1.set_ylabel(r"$\Delta G^{\ddagger}_{\mathrm{DFT}}$ [kcal/mol]")
    ax1.set_ylim(10, 34)
    ax1.set_yticks(np.arange(10, 33, 5))
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    # ===== Right panel: ΔΔG‡ =====
    x2 = np.arange(1) * 1.2
    for i, (label, color) in enumerate(zip(substrates_delta, colors[2:])):  # substrates 3–7
        offset = (i - 2) * width
        bars = ax2.bar(x2 + offset, data_delta[:, i], width, label=label, color=color)
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + np.sign(h) * 0.42,
                f"{h:.2f}".replace("-", "−"),
                rotation=90,
                ha="center",
                va="center",
                fontsize=10,
            )

    ax2.margins(y=0.25)
    ax2.set_xticks([])
    ax2.set_xticklabels([])
    ax2.set_ylabel(r"$\Delta\Delta G^{\ddagger}_{\mathrm{expt}}$ [kcal/mol]")
    ax2.set_yticks(np.arange(-3, 2, 1))
    ax2.set_ylim(-3.5, 1.5)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    # ===== Common legend =====
    fig.legend(
        substrates_full,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=7,
        frameon=True,
        borderpad=0.2,
        handletextpad=0.3,
        columnspacing=0.5,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, dpi=500, transparent=False, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_diketone_reduction_stackplot(from_file_path: str, save_path: str) -> pd.DataFrame:
    """Load diketone reduction data from Excel and create a stacked area plot.

    Parameters
    ----------
    from_file_path : str
        Path to the input Excel file.
        The file is expected to contain a sheet named
        "diketone_reduction_results" with a header row in the second line
        (the first row is skipped).

    save_path : str
        Path where the output image file (stacked area plot) will be saved.
        If the folder part does not exist, it is created automatically.

    Returns
    -------
    pandas.DataFrame
        The loaded DataFrame used for plotting.
    """
    # --- ensure output folder exists ---
    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_excel(
        from_file_path,
        sheet_name="diketone_reduction_results",
        skiprows=1,
    )  # .iloc[:150]
    print(df.columns)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    # Colors for each intermediate/product
    c1 = "red"
    c2 = "tab:pink"
    c3 = "blue"
    c4 = "tab:blue"
    colors = [c1, c2, c3, c4]

    # Labels for the stacked components
    labels = [
        r"$\bf{1}$",
        r"$\bf{2}$",
        r"$\bf{3}$",
        r"$\bf{4}$",
    ]

    # Stacked area plot (normalized to unity)
    polys = ax.stackplot(
        df["reaction rate [%]"] / 100,
        df[1] / 200,
        df[2] / 200,
        df[3] / 200,
        df[4] / 200,
        colors=colors,
        labels=labels,
    )

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="upper left",   # upper left of the plotting area
        ncol=1,
        fontsize=8,         # keep font size
        borderpad=0.2,      # padding inside legend frame
        labelspacing=0.2,   # spacing between labels
        handlelength=1.0,   # line length in legend
        handletextpad=0.3,  # distance between line and text
        borderaxespad=0.2,  # distance from axes to legend
        frameon=True,
        framealpha=0.8,
    )

    # Axis formatting
    # ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("reaction progress [-]")
    ax.set_ylabel("concentration [-]")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(-0.01, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=500)

    return df



if __name__ == "__main__":
    series_data = {
    'Series A': np.array([[2.18E-02, 2.56E-01],
                          [2.47E-02, 6.25E-01],
                          [5.35E-02, 1.50E+00]]),
    'Series B': np.array([[2.78E-01, 9.95E-03],
                          [4.78E-01, 1.98E-02],
                          [7.42E-01, 2.96E-02]]),
    'Series C': np.array([[3.55E-01, 3.78E-01],
                          [8.25E-01, 8.37E-01],
                          [2.04E+00, 2.14E+00]]),
    }
    text_pos = {
        "Series A": (0.7, 1.6),
        "Series B": (1.2, 0.3),
        "Series C": (2.4, 2.4)
    }
    ddg = plot_origin_regression_series(series_data, "data/eda/comparison_A.png",text_positions=text_pos,)
    print(ddg)
    series_data = {
        'Series A': np.array([[1.33E-01, 2.39E-01],
                            [1.76E-01, 2.93E-01],
                            [2.23E-01, 3.44E-01]]),
        'Series B': np.array([[6.98E-03, 1.12E-01],
                            [9.53E-02, 1.50E+00],
                            [1.99E-01, 2.94E+00]]),
        'Series C': np.array([[2.78E-01, 9.95E-03],
                            [4.78E-01, 1.98E-02],
                            [7.42E-01, 2.96E-02]])
    }
    text_pos = {
        "Series A": (0.7, .8),
        "Series C": (1.2, 0.3),
        "Series B": (.8, 2.7)
    }
    ddg = plot_origin_regression_series(series_data, "data/eda/comparison_B.png", text_positions=text_pos)
    print(ddg)
    series_data = {
        'Series A': np.array([[0.055710607, 0.683096845],
                            [0.085297581, 1.124929597],
                            [0.185114625, 2.424802726]]),
        'Series B': np.array([[0.493475985, 0.108451708],
                            [0.524728529, 0.116916450],
                            [1.042394609, 0.237800709]]),
        'Series C': np.array([[0.316998127, 0.099210565],
                            [0.515215972, 0.152275946],
                            [0.667829373, 0.213188769]])
    }
    text_pos = {
        "Series A": (0.8, 2.3),
        "Series C": (.7, 0.5),
        "Series B": (1.7, .4)
    }
    ddg = plot_origin_regression_series(series_data, "data/eda/comparison_C.png", text_positions=text_pos)
    print(ddg)
    plot_dft_bar_dual("data/eda/DFT.png")
    plot_diketone_reduction_stackplot(
        "data/all_experimental_data.xlsx",
        "data/eda/reaction_rate_stackplot.png",
    )
