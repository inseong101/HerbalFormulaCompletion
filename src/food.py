#!/usr/bin/env python3
"""Recipe1M 원자료를 Inverse Cooking 기준의 재료 집합으로 정리한다."""

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, default=Path("data/recipe1m/det_ingrs.json"))
    parser.add_argument(
        "--layers", type=Path, default=Path("data/recipe1m/recipe1M_layers.tar.gz")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("work/food"))
    parser.add_argument("--ingredient-threshold", type=int, default=10)
    args = parser.parse_args()

    for path in (args.detections, args.layers):
        if not path.exists():
            raise FileNotFoundError(f"Recipe1M 등록 후 파일을 이 경로에 두세요: {path}")

    counts = Counter()
    total_records = 0
    first_pass_eligible = 0
    print("[Food pass 1/2] train split으로 standardized vocabulary 재구성")
    for layer, detection in tqdm(paired_records(args.layers, args.detections), unit="recipe"):
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
        if count >= args.ingredient_threshold
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
        paired_records(args.layers, args.detections), total=total_records, unit="recipe"
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "canonical_ingredient_mapping.csv",
        ("canonical_ingredient", "training_occurrence_count", "alias"),
        mapping_rows,
    )
    write_csv(
        args.output_dir / "unique_compositions.csv",
        ("composition_id", "weight", "ingredient_count", "ingredients",
         "example_recipe_id", "example_title", "example_partition"),
        composition_rows,
    )
    write_csv(
        args.output_dir / "composition_size_distribution.csv",
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
        "ingredient_frequency_threshold_train_only": args.ingredient_threshold,
        "duplicate_policy": "merge identical complete standardized ingredient sets and preserve source count as weight",
    }
    (args.output_dir / "metadata.json").write_text(
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


if __name__ == "__main__":
    main()
