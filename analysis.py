"""논문의 표와 그림을 한 번에 생성한다."""

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"


def read_figure1_data():
    path = DATA / "figure1_ingredient_counts.csv"
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = [
            (
                int(row["ingredient_count"]),
                int(row["food_count_before_matching"]),
                int(row["herbal_count"]),
            )
            for row in csv.DictReader(file)
        ]

    if [row[0] for row in rows] != list(range(2, 20)):
        raise ValueError("ingredient_count는 2부터 19까지 한 번씩 있어야 합니다.")
    if sum(row[1] for row in rows) != 762996:
        raise ValueError("Food 조성 수의 합계가 762,996이 아닙니다.")
    if sum(row[2] for row in rows) != 2009:
        raise ValueError("Herbal 조성 수의 합계가 2,009가 아닙니다.")
    return rows


def make_figure1():
    rows = read_figure1_data()
    lengths = [row[0] for row in rows]
    food_counts = [row[1] for row in rows]
    herbal_counts = [row[2] for row in rows]
    food = [count / sum(food_counts) * 100 for count in food_counts]
    herbal = [count / sum(herbal_counts) * 100 for count in herbal_counts]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.9), facecolor="white")
    width = 0.38
    ax.bar(
        [x - width / 2 for x in lengths],
        food,
        width=width,
        color="#bdbdbd",
        edgecolor="#555555",
        linewidth=0.5,
        label="Food before matching (n=762,996)",
    )
    ax.bar(
        [x + width / 2 for x in lengths],
        herbal,
        width=width,
        color="#222222",
        edgecolor="#222222",
        linewidth=0.5,
        label="Herbal (n=2,009)",
    )

    ax.set_xlabel("Number of ingredients")
    ax.set_ylabel("Unique compositions (%)")
    ax.set_xticks(lengths)
    ax.set_ylim(0, max(food + herbal) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=8.2)

    FIGURES.mkdir(exist_ok=True)
    svg = FIGURES / "Figure1_dataset_matching.svg"
    png = FIGURES / "Figure1_dataset_matching.png"
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Figure 1: PASS")
    print(svg)
    print(png)


def main():
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    make_figure1()
    print("\nAll analyses completed.")


if __name__ == "__main__":
    main()
