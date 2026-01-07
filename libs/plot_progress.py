from matplotlib import pyplot as plt
import pandas as pd


def common(from_file_path: str, to_file_path: str) -> pd.DataFrame:
    """Load diketone reduction data from Excel and create a stacked area plot.

    Parameters
    ----------
    from_file_path : str
        Path to the input Excel file.
        The file is expected to contain a sheet named
        "diketone_reduction_results" with a header row in the second line
        (the first row is skipped).

    to_file_path : str
        Path where the output image file (stacked area plot) will be saved.

    Returns
    -------
    pandas.DataFrame
        The loaded DataFrame used for plotting.
    """
    # Load data
    df = pd.read_excel(
        from_file_path,
        sheet_name="diketone_reduction_results",
        skiprows=1,
    )  # .iloc[:150]
    print(df.columns)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(3, 3))

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
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("reaction progress [-]")
    ax.set_ylabel("concentration [-]")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(-0.01, 1.0)

    plt.tight_layout()
    plt.savefig(to_file_path, dpi=500)  # , bbox_inches="tight")#, transparent=True)

    return df


if __name__ == "__main__":
    common(
        "data/all_experimental_data.xlsx",
        "data/reaction_rate_stackplot.png",
    )
