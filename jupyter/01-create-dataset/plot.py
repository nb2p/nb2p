from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def setup_matplotlib():
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    sns.set_theme(
        style="ticks",
        font_scale=1.65,  # type: ignore
        rc={
            "axes.facecolor": "white",
            # "axes.grid": False,
            "grid.color": ".8",
            "font.sans-serif": ["Linux Libertine O"],
        },
    )


def draw_figure(
    ax,
    df: pd.DataFrame,
    xaxis,
    xlabel,
    ylim=None,
    exclude_outliers=False,
    outlier_q: Optional[float] = None,
    hue: Optional[str] = None,
    bins: Optional[Union[int, str]] = None,
):
    if exclude_outliers:
        if not outlier_q:
            df = df.copy()
            outlier_q = df[xaxis].quantile(0.99)
        print(f"INFO  Exclude outliers (X >= {outlier_q})")
        df = df[(df[xaxis] < outlier_q)]

    df[xaxis] = df[xaxis].apply(lambda x: int(x))

    # fig, ax = plt.subplots(1, 1, figsize=(5, 4.2))
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    if not bins:
        bins = "auto"

    sns.histplot(
        data=df,
        x=xaxis,
        stat="probability",
        ax=ax,
        #  bins=100,
        kde=True,
        binwidth=1 if xaxis == "num_code_cells" else None,
        hue=hue,
        common_norm=False,
        bins=bins,
        # palette=["#2d6e8e", "#51c468"],
        palette="tab10",
    )
    # ax.set_xlim(None, 315)
    ax.set_ylim(0, ylim)  # type: ignore
    ax.set(xlabel=xlabel, ylabel="Probability")


def draw_merged_figure(
    ax,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    xaxis,
    xlabel,
    ylim=None,
    exclude_outliers=False,
    outlier_q: Optional[float] = None,
    bins: Optional[int] = None,
):
    df1 = df1.copy()
    df2 = df2.copy()

    df1["Type"] = "Prompted"
    df2["Type"] = "Unprompted"

    total_df = pd.concat([df1, df2])

    draw_figure(
        ax, total_df, xaxis, xlabel, ylim, exclude_outliers, outlier_q, "Type", bins
    )


def draw_merged_figure_one_df(
    ax,
    df: pd.DataFrame,
    xaxis,
    xlabel,
    ylim=None,
    exclude_outliers=False,
    outlier_q: Optional[float] = None,
    bins: Optional[int] = None,
):
    total_df = df.copy()

    draw_figure(
        ax, total_df, xaxis, xlabel, ylim, exclude_outliers, outlier_q, "prompted", bins
    )
