"""원자료 구조와 전처리 수를 출력한 뒤 논문의 Figure 1을 만든다."""

import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from raw_pipeline.herbal import open_csv, preprocess


ROOT = Path(__file__).resolve().parent
HERBAL_RAW = ROOT / "data" / "raw" / "herbal"
FOOD_EXAMPLE = ROOT / "data" / "raw" / "food_example" / "recipe1m_record_excerpt.json"
FOOD_PROCESSED = ROOT / "data" / "processed" / "food"
HERBAL_PROCESSED = ROOT / "data" / "processed" / "herbal"
FIGURES = ROOT / "figures"


def heading(number, title):
    print("\n" + "=" * 72)
    print(f"{number}. {title}")
    print("=" * 72)


def show_raw_herbal_example():
    files = sorted(HERBAL_RAW.glob("*.csv"))
    if len(files) != 5:
        raise FileNotFoundError("data/raw/herbal에 CSV 5개가 있어야 합니다.")
    print("[Herbal raw files]")
    for path in files:
        print(f"  - {path.name}: {path.stat().st_size:,} bytes")

    handle, reader, encoding = open_csv(files[0])
    try:
        first = next(reader)
        fields = reader.fieldnames or []
    finally:
        handle.close()
    print(f"\n[Actual CSV header: {files[0].name}, {encoding}]")
    print(" | ".join(fields))
    print("\n[Actual first row: selected fields]")
    for field in ("처방아이디", "처방한글명", "출전", "출처", "약재한글명", "용량", "단위"):
        print(f"  {field}: {first.get(field, '')}")


def make_herbal_counts():
    metadata, collapsed, _, counts = preprocess(HERBAL_RAW, HERBAL_PROCESSED)
    expected = {
        "source_formulas": 3078,
        "unique_compositions": 2082,
        "selected_unique_compositions": 2009,
    }
    for key, value in expected.items():
        if metadata[key] != value:
            raise ValueError(f"Herbal {key}: {metadata[key]:,} != {value:,}")

    print("\n[Herbal processing counts]")
    print(f"  CSV rows: {metadata['source_rows']:,}")
    print(f"  Formulas identified by file + formula ID: {metadata['source_formulas']:,}")
    print(f"  Unique complete compositions: {metadata['unique_compositions']:,}")
    print(f"  Unique compositions with 2–19 herbs: {metadata['selected_unique_compositions']:,}")

    example = next(row for row in collapsed if "육미지황" in row["formula_names"] and row["weight"] == 19)
    print("\n[Actual duplicate-composition example]")
    print(f"  Formula names: {example['formula_names']}")
    print(f"  Complete herb set: {example['herbs']}")
    print(f"  Training weight: {example['weight']}")
    return counts


def show_food_provenance_and_counts():
    example = json.loads(FOOD_EXAMPLE.read_text(encoding="utf-8"))
    metadata = json.loads((FOOD_PROCESSED / "metadata.json").read_text(encoding="utf-8"))
    print("[Official acquisition]")
    print("  Recipe1M registration: https://im2recipe.csail.mit.edu/dataset/download")
    print("  Inverse Cooking code: https://github.com/facebookresearch/inversecooking")
    print("  Required files: det_ingrs.json + layer1.json")
    print("  Full Recipe1M raw files are not redistributed in this repository.")

    print("\n[Actual paired JSON record excerpt]")
    print(f"  Shared ID: {example['id']}")
    print(f"  layer1 fields: {', '.join(example['layer1']['fields'])}")
    print(f"  det_ingrs fields: {', '.join(example['det_ingrs']['fields'])}")
    print(f"  Title: {example['layer1']['title']}")
    print("  Raw ingredient excerpt:")
    for item in example["layer1"]["ingredient_excerpt"]:
        print(f"    - {item['text']}")
    print("  Standardized detection excerpt:")
    for item in example["det_ingrs"]["ingredient_excerpt"]:
        print(f"    - {item['text']}")

    print("\n[Food processing counts from the verified full-data run]")
    print(f"  Paired Recipe1M records: {metadata['source_records']:,}")
    print(f"  Eligible before vocabulary mapping: {metadata['first_pass_eligible_recipes']:,}")
    print(f"  Included after standardized mapping: {metadata['included_recipes_after_canonical_mapping']:,}")
    print(f"  Standardized ingredients: {metadata['canonical_ingredients']:,}")
    print(f"  Ingredient mapping coverage: {metadata['mapping_coverage']:.4%}")
    print(f"  Unique complete compositions: {metadata['unique_compositions']:,}")
    print(f"  Recipes merged into duplicate compositions: {metadata['duplicate_recipe_compositions']:,}")
    print("  To recalculate these values from registered raw files: python raw_pipeline/food.py")

    counts = {}
    with (FOOD_PROCESSED / "composition_size_distribution.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            counts[int(row["ingredient_count"])] = int(row["unique_compositions"])
    if sum(counts.values()) != 762996:
        raise ValueError("Food 2–19 unique-composition count is not 762,996")
    return counts


def show_figure_input(food_counts, herbal_counts):
    print("Length | Food count | Food % | Herbal count | Herbal %")
    food_total = sum(food_counts.values())
    herbal_total = sum(herbal_counts.values())
    for length in range(2, 20):
        food = food_counts[length]
        herbal = herbal_counts[length]
        print(
            f"{length:>6} | {food:>10,} | {food / food_total * 100:>6.2f} | "
            f"{herbal:>12,} | {herbal / herbal_total * 100:>8.2f}"
        )


def make_figure1(food_counts, herbal_counts):
    lengths = list(range(2, 20))
    food_total = sum(food_counts.values())
    herbal_total = sum(herbal_counts.values())
    food = [food_counts[length] / food_total * 100 for length in lengths]
    herbal = [herbal_counts[length] / herbal_total * 100 for length in lengths]

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(7.2, 3.9), facecolor="white")
    width = 0.38
    ax.bar(
        [x - width / 2 for x in lengths], food, width=width,
        color="#bdbdbd", edgecolor="#555555", linewidth=0.5,
        label="Food before matching (n=762,996)",
    )
    ax.bar(
        [x + width / 2 for x in lengths], herbal, width=width,
        color="#222222", edgecolor="#222222", linewidth=0.5,
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
    print(f"Saved: {svg}")
    print(f"Saved: {png}")


def main():
    heading(0, "Reproduction mode")
    print("Herbal preprocessing is rerun from the five public raw CSV files.")
    print("Food counts are read from the verified full Recipe1M run because raw access requires registration.")
    print("DATA.md gives the official URLs, checksums, schema, and full raw-data command.")

    heading(1, "Raw herbal data")
    show_raw_herbal_example()

    heading(2, "Herbal preprocessing: raw rows to unique compositions")
    herbal_counts = make_herbal_counts()

    heading(3, "Food data provenance, raw structure, and preprocessing record")
    food_counts = show_food_provenance_and_counts()

    heading(4, "Exact values used in Figure 1")
    show_figure_input(food_counts, herbal_counts)

    heading(5, "Create Figure 1")
    make_figure1(food_counts, herbal_counts)
    print("\nPASS: all checks completed.")


if __name__ == "__main__":
    main()
