"""Draw the ingredient-count distributions used in Figure 1."""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def make_figure1(herbal_counts, food_counts):
    sizes = list(range(2, 20))
    food = [food_counts[size] for size in sizes]
    herbal = [herbal_counts[size] for size in sizes]

    print("Ingredients | Food before matching | Herbal")
    for size, food_count, herbal_count in zip(sizes, food, herbal):
        print(f"{size:>11} | {food_count:>20,} | {herbal_count:>6,}")
    print(f"Totals: food={sum(food):,}, herbal={sum(herbal):,}")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    width = 0.38
    ax.bar(
        [size - width / 2 for size in sizes], food, width,
        color="#bdbdbd", edgecolor="#555555", linewidth=0.5,
        label="Food before matching (n=762,996)",
    )
    ax.bar(
        [size + width / 2 for size in sizes], herbal, width,
        color="#222222", edgecolor="#222222", linewidth=0.5,
        label="Herbal (n=2,009)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Number of ingredients")
    ax.set_ylabel("Unique compositions (log scale)")
    ax.set_xticks(sizes)
    ax.legend(frameon=False)

    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    svg = output / "Figure1_dataset_matching.svg"
    png = output / "Figure1_dataset_matching.png"
    fig.savefig(svg, bbox_inches="tight", facecolor="white", metadata={"Date": None})
    clean = "\n".join(
        line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()
    )
    svg.write_text(clean + "\n", encoding="utf-8")
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", svg.relative_to(ROOT))
    print("Saved:", png.relative_to(ROOT))
