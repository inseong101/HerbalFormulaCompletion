"""Build Figure 1 from the textbook CSV files and Recipe1M files."""

import csv
import subprocess
import sys
from pathlib import Path

from src.figure import make_figure1
from src.food import paired_records
from src.herbs import open_csv, preprocess


ROOT = Path(__file__).resolve().parent


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}", flush=True)


def herbal_source():
    files = sorted((ROOT / "data/herbal").glob("*.csv"))
    if len(files) != 5:
        raise SystemExit("Five textbook CSV files are required in data/herbal/.")
    handle, reader, encoding = open_csv(files[0])
    try:
        first = next(reader)
        print("Files:", ", ".join(path.name for path in files))
        print("Header:", ", ".join(reader.fieldnames or []))
        print("First row:", {key: first[key] for key in
              ("처방아이디", "처방한글명", "출전", "약재한글명", "용량", "단위")})
        print("Encoding:", encoding)
    finally:
        handle.close()


def recipe1m_source():
    folder = ROOT / "data/recipe1m"
    archive = folder / "recipe1M_layers.tar.gz"
    layer1 = folder / "layer1.json"
    detections = folder / "det_ingrs.json"
    layers = archive if archive.exists() else layer1
    if not layers.exists() or not detections.exists():
        raise SystemExit(
            "Recipe1M files are missing. Place det_ingrs.json and "
            "recipe1M_layers.tar.gz in data/recipe1m/."
        )
    layer, detected = next(paired_records(layers, detections))
    print("Files: layer1.json, det_ingrs.json")
    print("Shared ID:", layer["id"])
    print("Title:", layer["title"])
    print("layer1.json fields:", ", ".join(layer))
    print("det_ingrs.json fields:", ", ".join(detected))
    print("Raw ingredients:", [item["text"] for item in layer["ingredients"][:5]])
    print("Detected ingredients:", [item["text"] for item in detected["ingredients"][:5]])
    return layers, detections


def read_counts(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            int(row["ingredient_count"]): int(row["unique_compositions"])
            for row in csv.DictReader(handle)
        }


def main():
    heading("Herbal source data")
    herbal_source()

    heading("Herbal preprocessing")
    metadata, collapsed, _, herbal_counts = preprocess(
        ROOT / "data/herbal", ROOT / "work/herbal"
    )
    print("Formulas:", f"{metadata['source_formulas']:,}")
    print("Unique compositions:", f"{metadata['unique_compositions']:,}")
    print("Compositions with 2–19 herbs:",
          f"{metadata['selected_unique_compositions']:,}")
    example = next(
        row for row in collapsed
        if "육미지황" in row["formula_names"] and row["weight"] == 19
    )
    print("Example names:", example["formula_names"])
    print("Example composition:", example["herbs"])
    print("Weight:", example["weight"])

    heading("Food source data")
    layers, detections = recipe1m_source()

    heading("Food preprocessing")
    subprocess.run(
        [sys.executable, "-m", "src.food", "--layers", str(layers),
         "--detections", str(detections)],
        cwd=ROOT,
        check=True,
    )
    food_counts = read_counts(ROOT / "work/food/composition_size_distribution.csv")

    heading("Figure 1")
    make_figure1(herbal_counts, food_counts)


if __name__ == "__main__":
    main()
