#!/usr/bin/env python3
"""Preprocess the two datasets and draw Figure 1."""

import csv
import hashlib
import io
import json
import re
import tarfile
import unicodedata
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


BASE_WORDS = [
    "peppers", "tomato", "spinach_leaves", "turkey_breast", "lettuce_leaf",
    "chicken_thighs", "milk_powder", "bread_crumbs", "onion_flakes",
    "red_pepper", "pepper_flakes", "juice_concentrate", "cracker_crumbs",
    "hot_chili", "seasoning_mix", "dill_weed", "pepper_sauce", "sprouts",
    "cooking_spray", "cheese_blend", "basil_leaves", "pineapple_chunks",
    "marshmallow", "chile_powder", "corn_kernels", "tomato_sauce", "chickens",
    "cracker_crust", "lemonade_concentrate", "red_chili", "mushroom_caps",
    "mushroom_cap", "breaded_chicken", "frozen_pineapple", "seaweed",
    "bouillon_granules", "stuffing_mix", "parsley_flakes", "chicken_breast",
    "baguettes", "green_tea", "peanut_butter", "green_onion", "fresh_cilantro",
    "hot_pepper", "dried_lavender", "white_chocolate", "cake_mix",
    "cheese_spread", "chucken_thighs", "mandarin_orange", "laurel",
    "cabbage_head", "pistachio", "cheese_dip", "thyme_leave", "boneless_pork",
    "onion_dip", "skinless_chicken", "dark_chocolate", "canned_corn", "muffin",
    "frozen_broccoli", "philadelphia",
]

ROOT = Path(__file__).resolve().parent


@contextmanager
def open_json(path: Path):
    if path.name.endswith(".tar.gz"):
        archive = tarfile.open(path, "r:gz")
        binary = archive.extractfile("layer1.json")
        if binary is None:
            archive.close()
            raise FileNotFoundError(f"layer1.json이 압축파일에 없습니다: {path}")
        try:
            with io.TextIOWrapper(binary, encoding="utf-8") as handle:
                yield handle
        finally:
            archive.close()
    else:
        with path.open(encoding="utf-8") as handle:
            yield handle


def stream_json_array(path: Path, chunk_size=1 << 20):
    """큰 JSON 배열을 메모리에 모두 올리지 않고 한 레코드씩 읽는다."""
    decoder = json.JSONDecoder()
    with open_json(path) as handle:
        buffer = ""
        position = 0
        started = False
        ended = False
        while True:
            if position >= len(buffer) and not ended:
                buffer = handle.read(chunk_size)
                position = 0
                ended = not buffer
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    raise ValueError(f"빈 JSON 파일입니다: {path}")
                if buffer[position] != "[":
                    raise ValueError(f"최상위 JSON이 배열이 아닙니다: {path}")
                position += 1
                started = True
            while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                item, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if ended:
                    raise
                remainder = buffer[position:]
                addition = handle.read(chunk_size)
                buffer = remainder + addition
                position = 0
                ended = not addition
                continue
            yield item
            position = end
            if position > chunk_size:
                buffer = buffer[position:]
                position = 0


def paired_records(layer_path: Path, detection_path: Path):
    """두 파일을 ID 순서로 함께 읽고 어긋난 레코드를 즉시 중단한다."""
    layers = stream_json_array(layer_path)
    detections = stream_json_array(detection_path)
    index = 0
    while True:
        try:
            layer = next(layers)
        except StopIteration:
            try:
                next(detections)
            except StopIteration:
                return
            raise ValueError("det_ingrs.json의 레코드가 더 많습니다")
        try:
            detection = next(detections)
        except StopIteration:
            raise ValueError("layer1.json의 레코드가 더 많습니다")
        index += 1
        if layer["id"] != detection["id"]:
            raise ValueError(f"{index}번째 ID 불일치: {layer['id']} != {detection['id']}")
        yield layer, detection


def clean_ingredient(text: str) -> str:
    value = text.lower()
    value = "".join(character for character in value if not character.isdigit())
    value = value.replace("&", "and").replace("'n", "and")
    for character in ("%", ",", ".", "#", "[", "]", "!", "?"):
        value = value.replace(character, "")
    return value.strip().replace(" ", "_")


def clean_instruction(text: str) -> str:
    value = text.lower().replace("&", "and").replace("'n", "and")
    value = value.replace("#", "").replace("[", "").replace("]", "").strip()
    return "" if value and value[0].isdigit() else value


def valid_raw_ingredients(detection):
    return [
        clean_ingredient(item["text"])
        for item, valid in zip(detection["ingredients"], detection["valid"])
        if item and valid
    ]


def valid_instructions(layer):
    return [
        value for item in layer["instructions"]
        if (value := clean_instruction(item["text"]))
    ]


def eligible(ingredients, instructions):
    return (
        2 <= len(ingredients) < 20
        and 2 <= len(instructions) < 20
        and sum(map(len, instructions)) >= 20
    )


def cluster_ingredients(counts):
    clustered_counts = {}
    clusters = {}
    for ingredient, count in counts.items():
        parts = ingredient.split("_")
        candidates = [parts[-1], parts[0]]
        if len(parts) > 1:
            candidates = [parts[-1], parts[0], "_".join(parts[-2:]), "_".join(parts[:2])]
        representative = None
        for candidate in candidates:
            if candidate in counts:
                candidate_parts = candidate.split("_")
                if candidate_parts[0] in counts:
                    candidate = candidate_parts[0]
                elif len(candidate_parts) > 1 and candidate_parts[1] in counts:
                    candidate = candidate_parts[1]
                representative = candidate
                break
        representative = representative or ingredient
        clustered_counts[representative] = clustered_counts.get(representative, 0) + count
        clusters.setdefault(representative, []).append(ingredient)
    return clustered_counts, clusters


def remove_plurals(counts, clusters):
    deletions = []
    for ingredient, count in list(counts.items()):
        if not ingredient:
            deletions.append(ingredient)
        elif ingredient.endswith("es") and ingredient[:-2] in counts:
            singular = ingredient[:-2]
            counts[singular] += count
            clusters[singular].extend(clusters[ingredient])
            deletions.append(ingredient)
        elif ingredient.endswith("s") and ingredient[:-1] in counts:
            singular = ingredient[:-1]
            counts[singular] += count
            clusters[singular].extend(clusters[ingredient])
            deletions.append(ingredient)
    for ingredient in deletions:
        del counts[ingredient]
        del clusters[ingredient]
    return counts, clusters


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def preprocess_food(layers, detections, output_dir, ingredient_threshold=10):
    for path in (detections, layers):
        if not path.exists():
            raise FileNotFoundError(f"Recipe1M 등록 후 파일을 이 경로에 두세요: {path}")

    counts = Counter()
    total_records = 0
    first_pass_eligible = 0
    print("[Food pass 1/2] train split으로 standardized vocabulary 재구성")
    for layer, detection in tqdm(paired_records(layers, detections), unit="recipe"):
        total_records += 1
        raw = valid_raw_ingredients(detection)
        instructions = valid_instructions(layer)
        if eligible(raw, instructions):
            first_pass_eligible += 1
            if layer["partition"] == "train":
                counts.update(raw)

    for ingredient in BASE_WORDS:
        counts.setdefault(ingredient, 1)
    canonical_counts, clusters = cluster_ingredients(counts)
    canonical_counts, clusters = remove_plurals(canonical_counts, clusters)
    canonical_counts = {
        name: count for name, count in canonical_counts.items()
        if count >= ingredient_threshold
    }
    alias_to_canonical = {
        alias: canonical
        for canonical in canonical_counts
        for alias in clusters[canonical]
    }

    composition_counts = Counter()
    examples = {}
    example_titles = defaultdict(list)
    included = 0
    dropped = 0
    raw_occurrences = 0
    mapped_occurrences = 0
    print("[Food pass 2/2] standardized ingredient set과 동일 조성 weight 생성")
    for layer, detection in tqdm(
        paired_records(layers, detections), total=total_records, unit="recipe"
    ):
        raw = valid_raw_ingredients(detection)
        instructions = valid_instructions(layer)
        if not eligible(raw, instructions):
            continue
        raw_occurrences += len(raw)
        mapped_occurrences += sum(item in alias_to_canonical for item in raw)
        composition = tuple(sorted({
            alias_to_canonical[item] for item in raw if item in alias_to_canonical
        }))
        if not eligible(composition, instructions):
            dropped += 1
            continue
        included += 1
        composition_counts[composition] += 1
        examples.setdefault(composition, (layer["id"], layer["title"], layer["partition"]))
        if layer["title"] not in example_titles[composition] and len(example_titles[composition]) < 5:
            example_titles[composition].append(layer["title"])

    composition_rows = []
    for composition, weight in composition_counts.items():
        signature = "|".join(composition)
        recipe_id, title, partition = examples[composition]
        composition_rows.append({
            "composition_id": hashlib.sha256(signature.encode()).hexdigest()[:16],
            "weight": weight,
            "ingredient_count": len(composition),
            "ingredients": signature,
            "example_recipe_id": recipe_id,
            "example_title": title,
            "example_partition": partition,
        })
    composition_rows.sort(key=lambda row: row["composition_id"])
    size_counts = Counter(row["ingredient_count"] for row in composition_rows)

    mapping_rows = [
        {
            "canonical_ingredient": canonical,
            "training_occurrence_count": canonical_counts[canonical],
            "alias": alias,
        }
        for canonical in canonical_counts
        for alias in sorted(clusters[canonical])
    ]
    mapping_rows.sort(key=lambda row: (row["canonical_ingredient"], row["alias"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "canonical_ingredient_mapping.csv",
        ("canonical_ingredient", "training_occurrence_count", "alias"),
        mapping_rows,
    )
    write_csv(
        output_dir / "unique_compositions.csv",
        ("composition_id", "weight", "ingredient_count", "ingredients",
         "example_recipe_id", "example_title", "example_partition"),
        composition_rows,
    )
    write_csv(
        output_dir / "composition_size_distribution.csv",
        ("ingredient_count", "unique_compositions"),
        ({"ingredient_count": size, "unique_compositions": size_counts[size]}
         for size in sorted(size_counts)),
    )

    metadata = {
        "source": "Recipe1M det_ingrs.json + layer1.json",
        "method": "Inverse Cooking build_vocab.py-compatible preprocessing",
        "source_records": total_records,
        "first_pass_eligible_recipes": first_pass_eligible,
        "included_recipes_after_canonical_mapping": included,
        "dropped_after_canonical_mapping": dropped,
        "canonical_ingredients": len(canonical_counts),
        "aliases": len(alias_to_canonical),
        "raw_valid_ingredient_occurrences_in_eligible_recipes": raw_occurrences,
        "mapped_ingredient_occurrences": mapped_occurrences,
        "mapping_coverage": mapped_occurrences / raw_occurrences,
        "unique_compositions": len(composition_rows),
        "duplicate_recipe_compositions": included - len(composition_rows),
        "ingredient_frequency_threshold_train_only": ingredient_threshold,
        "duplicate_policy": "merge identical complete standardized ingredient sets and preserve source count as weight",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Recipes:", f"{total_records:,}")
    print("Recipes with 2–19 standardized ingredients:", f"{included:,}")
    print("Unique compositions:", f"{len(composition_rows):,}")
    target = ("cheese", "garlic", "oil", "paprika", "pepper", "potato", "salt")
    if composition_counts[target] == 19:
        print("Example titles:", " | ".join(example_titles[target]))
        print("Example composition:", "|".join(target))
        print("Weight:", composition_counts[target])
    print("Saved: work/food/unique_compositions.csv")
    print("Saved: work/food/composition_size_distribution.csv")
    return metadata, size_counts


def clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip()


def open_textbook_csv(path):
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        handle = path.open(encoding=encoding, newline="")
        try:
            reader = csv.DictReader(handle)
            reader.fieldnames = [clean_text(name) for name in (reader.fieldnames or [])]
            if reader.fieldnames:
                return handle, reader, encoding
        except UnicodeDecodeError:
            pass
        handle.close()
    raise ValueError(f"CSV 인코딩을 읽지 못했습니다: {path}")


def find_column(fields, candidates):
    for candidate in candidates:
        if candidate in fields:
            return candidate
    raise ValueError(f"필요한 열이 없습니다: {candidates}")


def preprocess_herbal(input_dir, output_dir):
    prescriptions = {}
    names = {}
    source_rows = 0
    files = sorted(input_dir.glob("*.csv"))
    if len(files) != 5:
        raise FileNotFoundError(f"교과서 CSV 5개가 필요합니다: {input_dir}")

    for path in files:
        handle, reader, encoding = open_textbook_csv(path)
        try:
            fields = reader.fieldnames or []
            id_column = find_column(fields, ("처방아이디", "처방ID", "처방id"))
            herb_column = find_column(fields, ("약재한글명", "약재명"))
            name_column = find_column(fields, ("처방한글명", "처방명"))
            for row_number, row in enumerate(reader, 2):
                source_rows += 1
                formula_id = clean_text(row.get(id_column))
                if not formula_id:
                    raise ValueError(f"처방아이디가 비었습니다: {path}:{row_number}")
                key = (path.name, formula_id)
                name = clean_text(row.get(name_column))
                herb = clean_text(row.get(herb_column))
                if name:
                    names[key] = name
                if herb:
                    prescriptions.setdefault(key, set()).add(herb)
        finally:
            handle.close()
        count = sum(key[0] == path.name for key in prescriptions)
        print(f"  {path.name}: {encoding}, {count:,} formulas")

    grouped = defaultdict(list)
    for key, herbs in prescriptions.items():
        if herbs:
            grouped[tuple(sorted(herbs))].append(key)

    rows = []
    for herbs, sources in grouped.items():
        signature = "|".join(herbs)
        rows.append({
            "composition_id": hashlib.sha256(signature.encode()).hexdigest()[:16],
            "weight": len(sources),
            "herb_count": len(herbs),
            "herbs": signature,
            "formula_names": "|".join(sorted({names[key] for key in sources if key in names})),
        })
    rows.sort(key=lambda row: row["composition_id"])
    selected = [row for row in rows if 2 <= row["herb_count"] <= 19]
    size_counts = Counter(row["herb_count"] for row in selected)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "unique_compositions.csv",
        ("composition_id", "weight", "herb_count", "herbs", "formula_names"),
        rows,
    )
    write_csv(
        output_dir / "composition_size_distribution.csv",
        ("ingredient_count", "unique_compositions"),
        ({"ingredient_count": size, "unique_compositions": size_counts[size]}
         for size in range(2, 20)),
    )
    metadata = {
        "source_rows": source_rows,
        "source_formulas": len(prescriptions),
        "unique_compositions": len(rows),
        "selected_unique_compositions": len(selected),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata, rows, size_counts


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
        "svg.hashsalt": "HerbalFormulaCompletion",
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


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}", flush=True)


def main():
    herbal_dir = ROOT / "data/herbal"
    herbal_files = sorted(herbal_dir.glob("*.csv"))
    if len(herbal_files) != 5:
        raise SystemExit("Five textbook CSV files are required in data/herbal/.")

    heading("Herbal source data")
    handle, reader, encoding = open_textbook_csv(herbal_files[0])
    try:
        first = next(reader)
        print("Files:", ", ".join(path.name for path in herbal_files))
        print("Header:", ", ".join(reader.fieldnames or []))
        print("First row:", {key: first[key] for key in
              ("처방아이디", "처방한글명", "출전", "약재한글명", "용량", "단위")})
        print("Encoding:", encoding)
    finally:
        handle.close()

    heading("Herbal preprocessing")
    herbal_meta, herbal_rows, herbal_counts = preprocess_herbal(
        herbal_dir, ROOT / "work/herbal"
    )
    print("Formulas:", f"{herbal_meta['source_formulas']:,}")
    print("Unique compositions:", f"{herbal_meta['unique_compositions']:,}")
    print("Compositions with 2–19 herbs:",
          f"{herbal_meta['selected_unique_compositions']:,}")
    example = next(
        row for row in herbal_rows
        if "육미지황" in row["formula_names"] and row["weight"] == 19
    )
    print("Example names:", example["formula_names"])
    print("Example composition:", example["herbs"])
    print("Weight:", example["weight"])

    recipe_dir = ROOT / "data/recipe1m"
    archive = recipe_dir / "recipe1M_layers.tar.gz"
    layer1 = recipe_dir / "layer1.json"
    layers = archive if archive.exists() else layer1
    detections = recipe_dir / "det_ingrs.json"
    if not layers.exists() or not detections.exists():
        raise SystemExit("Recipe1M files are required in data/recipe1m/.")

    heading("Food source data")
    records = paired_records(layers, detections)
    layer, detected = next(records)
    records.close()
    print("Files: layer1.json, det_ingrs.json")
    print("Example ID:", layer["id"])
    print("Recipe:", layer["title"])
    print("Original ingredient -> detected ingredient")
    for original, result in zip(layer["ingredients"][:5], detected["ingredients"][:5]):
        print(f"  {original['text']} -> {result['text']}")

    heading("Food preprocessing")
    _, food_counts = preprocess_food(layers, detections, ROOT / "work/food")

    heading("Figure 1")
    make_figure1(herbal_counts, food_counts)


if __name__ == "__main__":
    main()
