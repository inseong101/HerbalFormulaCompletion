#!/usr/bin/env python3
"""Reproduce preprocessing, recommendation evaluation, and fixed-margin structure analysis.

Local setup: python -m pip install "matplotlib>=3.9,<4" "tqdm>=4.66,<5" "numba>=0.61,<0.68" "networkx>=3.4,<4"
Run: python run.py
Generated tables and counts are written to work/.
"""

# MIT License
#
# Copyright (c) 2026 inseong101
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import csv
import argparse
import gzip
import math
import hashlib
import io
import json
import re
import random
import tarfile
import unicodedata
from collections import Counter, defaultdict
from contextlib import contextmanager
from itertools import combinations
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
    raw_size_eligible = 0
    first_pass_eligible = 0
    print("[Food pass 1/2] train split으로 standardized vocabulary 재구성")
    for layer, detection in tqdm(paired_records(layers, detections), unit="recipe"):
        total_records += 1
        raw = valid_raw_ingredients(detection)
        instructions = valid_instructions(layer)
        raw_size_eligible += 2 <= len(raw) < 20
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
        "raw_ingredient_count_eligible_recipes": raw_size_eligible,
        "excluded_by_raw_ingredient_count": total_records - raw_size_eligible,
        "excluded_by_instruction_criteria": raw_size_eligible - first_pass_eligible,
        "first_pass_eligible_recipes": first_pass_eligible,
        "included_recipes_after_canonical_mapping": included,
        "excluded_by_final_eligibility": dropped,
        "canonical_ingredients": len(canonical_counts),
        "aliases": len(alias_to_canonical),
        "raw_valid_ingredient_occurrences_in_source_recipes": raw_occurrences,
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
    print("Recipes with 2–19 valid raw ingredients:", f"{raw_size_eligible:,}")
    print("Recipes also meeting instruction criteria:", f"{first_pass_eligible:,}")
    print("Eligible recipes with 2–19 standardized ingredients:", f"{included:,}")
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

    selected_prescriptions = {
        key: herbs for key, herbs in prescriptions.items() if 2 <= len(herbs) <= 19
    }
    grouped = defaultdict(list)
    for key, herbs in selected_prescriptions.items():
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
    size_counts = Counter(row["herb_count"] for row in rows)

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
        "included_formulas_with_2_to_19_herbs": len(selected_prescriptions),
        "excluded_by_herb_count": len(prescriptions) - len(selected_prescriptions),
        "unique_compositions": len(rows),
        "selected_unique_compositions": len(rows),
        "duplicate_formula_compositions": len(selected_prescriptions) - len(rows),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata, rows, size_counts


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def matched_food_samples(herbal_path, food_path, output_dir, replicates=100, seed=20260823):
    """Sample unique compositions without replacement within each independent draw."""
    if replicates < 1:
        raise ValueError("At least one food sample is required")
    herbs = read_rows(herbal_path)
    foods = read_rows(food_path)
    needed = Counter(int(row["herb_count"]) for row in herbs)
    if not needed or not all(2 <= size <= 19 for size in needed):
        raise ValueError("Herbal compositions must contain 2–19 herbs")
    by_size = defaultdict(list)
    source_ids = set()
    for row in sorted(foods, key=lambda row: row["composition_id"]):
        if row["composition_id"] in source_ids:
            raise ValueError("Duplicate food composition IDs")
        source_ids.add(row["composition_id"])
        by_size[int(row["ingredient_count"])].append(row)
    if len({row["composition_id"] for row in herbs}) != len(herbs):
        raise ValueError("Duplicate herbal composition IDs")
    for size, number in needed.items():
        if len(by_size[size]) < number:
            raise ValueError(f"Insufficient food compositions at size {size}")
    membership, checks, summaries = [], [], []
    for replicate in range(1, replicates + 1):
        rng = random.Random(seed + replicate - 1)
        sample = [row for size in sorted(needed)
                  for row in rng.sample(by_size[size], needed[size])]
        counts = Counter(int(row["ingredient_count"]) for row in sample)
        ids = [row["composition_id"] for row in sample]
        if len(ids) != len(set(ids)) or counts != needed:
            raise AssertionError("Food matching validation failed")
        for row in sample:
            membership.append({"replicate": replicate,
                               "composition_id": row["composition_id"],
                               "ingredient_count": int(row["ingredient_count"]),
                               "weight": int(row["weight"])})
        for size in range(2, 20):
            checks.append({"replicate": replicate, "ingredient_count": size,
                           "herbal_count": needed[size], "food_count": counts[size],
                           "matches": counts[size] == needed[size]})
        summaries.append({"replicate": replicate, "seed": seed + replicate - 1,
                          "unique_compositions": len(ids),
                          "source_record_weight_sum": sum(int(row["weight"]) for row in sample),
                          "sorted_id_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()})
    for name, rows in (("membership.csv", membership), ("size_checks.csv", checks),
                       ("sample_summary.csv", summaries)):
        write_csv(output_dir / name, tuple(rows[0]), rows)
    metadata = {"replicates": replicates, "seed": seed,
                "seed_rule": "seed + replicate - 1",
                "food_population_unique": len(foods), "compositions_per_sample": len(herbs),
                "within_sample": "without replacement, uniform within each ingredient-count stratum",
                "between_samples": "independent draws; compositions may recur across samples",
                "weight_policy": "source multiplicity retained for later training; not used as sampling probability",
                "all_size_checks_passed": True,
                "fold_assignment_and_model_evaluation": "not performed in this step"}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return summaries, checks


def audit_duplicate_examples(herbal_dir, layers, detections, work_dir):
    """Trace the manuscript's two example compositions to every original record."""
    herbal_target = tuple(sorted(("목단피", "복령", "산수유", "산약", "숙지황", "택사")))
    food_target = ("cheese", "garlic", "oil", "paprika", "pepper", "potato", "salt")
    source_formulas = {}
    for path in sorted(herbal_dir.glob("*.csv")):
        handle, reader, _ = open_textbook_csv(path)
        try:
            fields = reader.fieldnames
            id_col = find_column(fields, ("처방아이디", "처방ID", "처방id"))
            herb_col = find_column(fields, ("약재한글명", "약재명"))
            name_col = find_column(fields, ("처방한글명", "처방명"))
            for row in reader:
                key = (path.name, clean_text(row.get(id_col)))
                record = source_formulas.setdefault(key, {"herbs": set(), "names": set(), "pages": set()})
                for dest, col in (("herbs", herb_col), ("names", name_col), ("pages", "페이지")):
                    value = clean_text(row.get(col))
                    if value:
                        record[dest].add(value)
        finally:
            handle.close()
    herbal_sources = [
        {"source_file": file, "formula_id": ident,
         "formula_names": "|".join(sorted(row["names"])),
         "pages": "|".join(sorted(row["pages"])), "herbs": "|".join(herbal_target)}
        for (file, ident), row in sorted(source_formulas.items())
        if tuple(sorted(row["herbs"])) == herbal_target
    ]
    mapping = {row["alias"]: row["canonical_ingredient"]
               for row in read_rows(work_dir / "food/canonical_ingredient_mapping.csv")}
    food_sources = []
    for layer, detection in tqdm(paired_records(layers, detections), unit="recipe"):
        raw = valid_raw_ingredients(detection)
        composition = tuple(sorted({mapping[item] for item in raw if item in mapping}))
        if composition == food_target and eligible(composition, valid_instructions(layer)):
            food_sources.append({"recipe_id": layer["id"], "title": layer["title"],
                                 "partition": layer["partition"], "ingredients": "|".join(composition),
                                 "original_ingredients": json.dumps(layer["ingredients"], ensure_ascii=False),
                                 "cleaned_to_standardized": json.dumps(
                                     [{"cleaned": item, "standardized": mapping.get(item)} for item in raw],
                                     ensure_ascii=False)})
    summary = {}
    for domain, target, field, sources, key in (
        ("herbal", herbal_target, "herbs", herbal_sources, "formula_names"),
        ("food", food_target, "ingredients", food_sources, "title"),
    ):
        row = next(row for row in read_rows(work_dir / domain / "unique_compositions.csv")
                   if row[field] == "|".join(target))
        if len(sources) != int(row["weight"]):
            raise AssertionError(f"{domain} example source count does not match weight")
        summary[domain] = {"composition_id": row["composition_id"], "composition": list(target),
                           "source_records": len(sources), "weight": int(row["weight"]),
                           "names_or_titles": dict(sorted(Counter(r[key] for r in sources).items()))}
        write_csv(work_dir / "examples" / f"{domain}_sources.csv", tuple(sources[0]), sources)
    (work_dir / "examples/summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def assign_composition_folds(rows, number=5, seed=20260812, ingredient_field="herbs"):
    """Deterministic, length-load-balanced folds of unique compositions."""
    ordered = sorted(rows, key=lambda row: row["composition_id"])
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda row: len(row[ingredient_field].split("|")), reverse=True)
    buckets = [[] for _ in range(number)]
    loads = [0] * number
    for row in ordered:
        index = min(range(number), key=lambda i: (loads[i], len(buckets[i]), i))
        buckets[index].append(row)
        loads[index] += len(row[ingredient_field].split("|"))
    return buckets


def training_statistics(rows, ingredient_field="herbs"):
    """Count ingredients and unordered pairs using source multiplicity."""
    counts, pairs = Counter(), Counter()
    for row in rows:
        items = sorted(set(row[ingredient_field].split("|")))
        weight = int(row["weight"])
        counts.update({item: weight for item in items})
        pairs.update({pair: weight for pair in combinations(items, 2)})
    return counts, pairs


def recommendation_scores(context, counts, pairs):
    """Return every training-vocabulary candidate with all three scores."""
    context = tuple(sorted(set(context)))
    if not context:
        raise ValueError("At least one input ingredient is required")
    rows = []
    for candidate in sorted(set(counts) - set(context)):
        conditional, jaccard = 0.0, 0.0
        for observed in context:
            both = pairs[tuple(sorted((observed, candidate)))]
            conditional += both / counts[observed] if counts[observed] else 0.0
            union = counts[observed] + counts[candidate] - both
            jaccard += both / union if union else 0.0
        rows.append({"candidate": candidate, "popularity": counts[candidate],
                     "mean_conditional": conditional / len(context),
                     "mean_jaccard": jaccard / len(context)})
    return rows


def recommendation_example(herbal_path, output_dir, fold_seed=20260812):
    """Compute the manuscript example using only its held-out fold's training data."""
    rows = read_rows(herbal_path)
    target = set(("목단피", "복령", "산수유", "산약", "숙지황", "택사"))
    context = ("목단피", "산수유")
    hidden = target - set(context)
    folds = assign_composition_folds(rows, seed=fold_seed)
    fold_index = next(i for i, fold in enumerate(folds)
                      if any(set(row["herbs"].split("|")) == target for row in fold))
    test = folds[fold_index]
    train = [row for i, fold in enumerate(folds) if i != fold_index for row in fold]
    target_row = next(row for row in test if set(row["herbs"].split("|")) == target)
    if any(set(row["herbs"].split("|")) == target for row in train):
        raise AssertionError("Example composition unexpectedly included in training")
    counts, pairs = training_statistics(train)
    scores = recommendation_scores(context, counts, pairs)
    top_rows = []
    for method in ("popularity", "mean_conditional", "mean_jaccard"):
        ranked = sorted(scores, key=lambda row: (-row[method], row["candidate"]))[:10]
        for rank, row in enumerate(ranked, 1):
            top_rows.append({"method": method, "rank": rank, "candidate": row["candidate"],
                             "score": row[method], "withheld_herb": row["candidate"] in hidden})
    support = []
    for observed in context:
        for candidate in sorted({row["candidate"] for row in top_rows} | hidden):
            both = pairs[tuple(sorted((observed, candidate)))]
            union = counts[observed] + counts[candidate] - both
            support.append({"input": observed, "candidate": candidate,
                            "input_weighted_count": counts[observed],
                            "candidate_weighted_count": counts[candidate],
                            "both_weighted_count": both, "either_weighted_count": union,
                            "conditional": both / counts[observed] if counts[observed] else 0.0,
                            "jaccard": both / union if union else 0.0})
    assignments = [{"composition_id": row["composition_id"], "fold": i + 1,
                    "example_role": "test" if i == fold_index else "train",
                    "weight": int(row["weight"])}
                   for i, fold in enumerate(folds) for row in fold]
    for name, records in (("top10.csv", top_rows), ("all_candidate_scores.csv", scores),
                          ("pair_support.csv", support), ("fold_assignments.csv", assignments)):
        write_csv(output_dir / name, tuple(records[0]), records)
    metadata = {"fold_seed": fold_seed, "number_of_folds": len(folds),
                "fold_sizes": [len(fold) for fold in folds], "example_test_fold": fold_index + 1,
                "training_unique_compositions": len(train), "test_unique_compositions": len(test),
                "training_source_weight_sum": sum(int(row["weight"]) for row in train),
                "example_composition_id": target_row["composition_id"],
                "excluded_example_source_weight": int(target_row["weight"]),
                "input": list(context), "withheld": sorted(hidden),
                "herbal_source_sha256": hashlib.sha256(herbal_path.read_bytes()).hexdigest(),
                "fold_policy": "Sort IDs, seeded shuffle, stable descending length sort, then assign to lowest total length; break ties by fold size and index",
                "scores": {"popularity": "weighted occurrence count",
                           "mean_conditional": "mean conditional proportion (0–1)",
                           "mean_jaccard": "mean pairwise Jaccard similarity (0–1)"},
                "ranking": "descending score, then ascending ingredient name (Unicode)",
                "scope": "one illustrative herbal test case, not aggregate performance evaluation"}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata, top_rows, support


EVALUATION_MODELS = ("popularity", "mean_conditional", "mean_jaccard")
EVALUATION_CONDITIONS = ("2", "3", "50%", "75%", "N-1")


def evaluation_contexts(row, condition, ingredient_field, seed=20260823, maximum=5):
    items = tuple(sorted(row[ingredient_field].split("|")))
    n = len(items)
    if condition == "N-1":
        size = n - 1
    elif condition.endswith("%"):
        size = min(n - 1, max(1, math.floor(n * int(condition[:-1]) / 100 + 0.5)))
    else:
        size = int(condition)
    if not 1 <= size < n:
        return []
    if condition == "N-1" or math.comb(n, size) <= maximum:
        return list(combinations(items, size))
    digest = hashlib.sha256((row["composition_id"] + condition).encode()).hexdigest()[:16]
    rng = random.Random(int(digest, 16) + seed)
    result = set()
    while len(result) < maximum:
        result.add(tuple(sorted(rng.sample(items, size))))
    return sorted(result)


def strict_top_indices(scores, k=10):
    """Exact Top-k; vocabulary indices are already in Unicode name order."""
    import numpy as np
    available = np.flatnonzero(np.isfinite(scores))
    if len(available) < k:
        raise ValueError("Fewer than 10 eligible training-vocabulary candidates")
    threshold = np.partition(scores[available], -k)[-k]
    above = np.flatnonzero(scores > threshold)
    ties = np.flatnonzero(scores == threshold)
    selected = np.concatenate((above, ties[:k - len(above)]))
    return selected[np.lexsort((selected, -scores[selected]))]


def evaluate_cohort(task):
    """One cohort: five folds, three methods, five input conditions."""
    import numpy as np
    domain, replicate, rows, ingredient_field, output_dir, fold_seed, context_seed = task
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets = assign_composition_folds(rows, seed=fold_seed, ingredient_field=ingredient_field)
    fold_rows, assignments = [], []
    reference_checks = 0
    evaluated_contexts = 0
    composition_fields = ("fold", "composition_id", "condition", "method", "contexts",
                          "recovered_total", "hidden_total", "performance", "target_coverage",
                          "mean_candidates")
    with gzip.open(output_dir / "composition_results.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=composition_fields)
        writer.writeheader()
        for fold_index, test in enumerate(buckets, 1):
            train = [row for i, fold in enumerate(buckets, 1) if i != fold_index for row in fold]
            assignments.extend({"composition_id": row["composition_id"], "test_fold": fold_index}
                               for row in test)
            train_sets = {tuple(sorted(row[ingredient_field].split("|"))) for row in train}
            if any(tuple(sorted(row[ingredient_field].split("|"))) in train_sets for row in test):
                raise AssertionError("Overlapping training/test compositions")
            counts, pairs = training_statistics(train, ingredient_field)
            vocabulary = sorted(counts)
            index = {name: i for i, name in enumerate(vocabulary)}
            c = np.array([counts[name] for name in vocabulary], dtype=np.float64)
            p = np.zeros((len(c), len(c)), dtype=np.float64)
            for (a, b), count in pairs.items():
                i, j = index[a], index[b]
                p[i, j] = p[j, i] = count
            conditional = p / c[:, None]
            jaccard = p / (c[:, None] + c[None, :] - p)
            accum = defaultdict(list)
            reference_conditions = set()
            for row in test:
                full = set(row[ingredient_field].split("|"))
                for condition in EVALUATION_CONDITIONS:
                    contexts = evaluation_contexts(row, condition, ingredient_field, context_seed)
                    if not contexts:
                        continue
                    hits = Counter()
                    covered, hidden_total, candidate_total = 0, 0, 0
                    for context in contexts:
                        evaluated_contexts += 1
                        hidden = full - set(context)
                        seen = [index[x] for x in context if x in index]
                        targets = {index[x] for x in hidden if x in index}
                        covered += len(targets)
                        hidden_total += len(hidden)
                        candidate_total += len(vocabulary) - len(seen)
                        score_arrays = [c.copy(), np.zeros(len(c)), np.zeros(len(c))]
                        for x in context:
                            if x in index:
                                score_arrays[1] += conditional[index[x]]
                                score_arrays[2] += jaccard[index[x]]
                        score_arrays[1] /= len(context)
                        score_arrays[2] /= len(context)
                        reference = None
                        if condition not in reference_conditions:
                            reference = recommendation_scores(context, counts, pairs)
                            reference_conditions.add(condition)
                        for model, scores in zip(EVALUATION_MODELS, score_arrays):
                            scores[seen] = -np.inf
                            top = strict_top_indices(scores)
                            if len(top) != 10 or len(set(top)) != 10 or set(top) & set(seen):
                                raise AssertionError("Invalid Top-10 list")
                            if reference is not None:
                                expected = sorted(reference, key=lambda r: (-r[model], r["candidate"]))[:10]
                                if [vocabulary[i] for i in top] != [r["candidate"] for r in expected]:
                                    raise AssertionError("Optimized ranking differs from the example scoring code")
                                reference_checks += 1
                            hits[model] += len(targets & set(top))
                    for model in EVALUATION_MODELS:
                        performance = hits[model] / hidden_total
                        item = {"fold": fold_index, "composition_id": row["composition_id"],
                                "condition": condition, "method": model, "contexts": len(contexts),
                                "recovered_total": hits[model], "hidden_total": hidden_total,
                                "performance": performance, "target_coverage": covered / hidden_total,
                                "mean_candidates": candidate_total / len(contexts)}
                        writer.writerow(item)
                        accum[condition, model].append(item)
            for condition in EVALUATION_CONDITIONS:
                for model in EVALUATION_MODELS:
                    values = accum[condition, model]
                    fold_rows.append({"domain": domain, "replicate": replicate, "fold": fold_index,
                        "condition": condition, "method": model,
                        "metric": "Hit@10" if condition == "N-1" else "Recall@10",
                        "train_unique": len(train), "test_unique": len(test),
                        "eligible_test_unique": len(values), "excluded_test_unique": len(test) - len(values),
                        "contexts": sum(v["contexts"] for v in values),
                        "performance": sum(v["performance"] for v in values) / len(values),
                        "target_coverage": sum(v["target_coverage"] for v in values) / len(values),
                        "mean_candidates": sum(v["mean_candidates"] for v in values) / len(values)})
    write_csv(output_dir / "fold_results.csv", tuple(fold_rows[0]), fold_rows)
    write_csv(output_dir / "fold_assignments.csv", tuple(assignments[0]), assignments)
    audit = {"domain": domain, "replicate": replicate, "unique_compositions": len(rows),
             "fold_sizes": [len(f) for f in buckets], "evaluated_contexts": evaluated_contexts,
             "reference_ranking_checks": reference_checks, "strict_top10_checks_passed": True,
             "training_test_composition_overlap": 0}
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return fold_rows, audit


def evaluate_all(work_dir, workers=2):
    """Evaluate the herbal cohort and every saved matched food sample."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import numpy as np
    work_dir = Path(work_dir)
    out = work_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    herbs = read_rows(work_dir / "herbal/unique_compositions.csv")
    memberships = read_rows(work_dir / "matching/membership.csv")
    wanted = {r["composition_id"] for r in memberships}
    with (work_dir / "food/unique_compositions.csv").open(encoding="utf-8-sig", newline="") as handle:
        food = {r["composition_id"]: r for r in csv.DictReader(handle) if r["composition_id"] in wanted}
    cohorts = defaultdict(list)
    for r in memberships:
        original = food[r["composition_id"]]
        if any(int(r[k]) != int(original[k]) for k in ("weight", "ingredient_count")):
            raise AssertionError("Membership differs from source composition")
        cohorts[int(r["replicate"])].append(original)
    if set(cohorts) != set(range(1, 101)):
        raise ValueError("Exactly 100 matched food samples are required")
    needed = Counter(int(r["herb_count"]) for r in herbs)
    for sample in cohorts.values():
        if len({r["composition_id"] for r in sample}) != len(herbs) or Counter(int(r["ingredient_count"]) for r in sample) != needed:
            raise AssertionError("Matched food sample invalid")
    input_files = ("herbal/unique_compositions.csv", "food/unique_compositions.csv", "matching/membership.csv")
    hashes = {}
    for name in input_files:
        with (work_dir / name).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = {"status": "running", "models": list(EVALUATION_MODELS),
                "conditions": list(EVALUATION_CONDITIONS), "folds": 5, "fold_seed": 20260812,
                "context_seed": 20260823, "maximum_contexts_per_composition_condition": {"N-1": "all N leave-one-out cases", "other_conditions": 5},
                "input_hashes": hashes,
                "training_weight": "source-record multiplicity",
                "evaluation_weight": "mean within each composition, then equal weight to each eligible composition across all test folds",
                "percentage_input_rounding": "nearest integer, halves rounded up; at least 1, at most N-1",
                "unseen_hidden_ingredients": "count as misses; retained in metric denominator",
                "ranking": "strict Top-10; descending score then Unicode ingredient name",
                "food_summary": "mean, sample SD, and empirical 2.5th–97.5th percentiles across 100 samples; percentiles are not confidence intervals",
                "scope": "main recommendation performance, excluding structure and learning-curve analyses"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Evaluating herbal cohort...", flush=True)
    fold_rows, audit = evaluate_cohort(("herbal", 0, herbs, "herbs", out / "cohorts/herbal_000", 20260812, 20260823))
    audits = [audit]
    print("Herbal cohort complete. Evaluating 100 food samples...", flush=True)
    tasks = [("food", rep, rows, "ingredients", out / f"cohorts/food_{rep:03d}", 20260812, 20260823)
             for rep, rows in sorted(cohorts.items())]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate_cohort, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            values, check = future.result()
            fold_rows.extend(values)
            audits.append(check)
            if completed == 1 or completed % 10 == 0:
                print(f"Food samples complete: {completed}/100", flush=True)
    fold_rows.sort(key=lambda r: (r["domain"], r["replicate"], r["condition"], r["method"], r["fold"]))
    grouped = defaultdict(list)
    for row in fold_rows:
        grouped[row["domain"], row["replicate"], row["condition"], row["method"]].append(row)
    sample_rows = []
    for (domain, rep, condition, model), values in sorted(grouped.items()):
        n = sum(r["eligible_test_unique"] for r in values)
        item = {"domain": domain, "replicate": rep, "condition": condition, "method": model,
                "metric": values[0]["metric"], "eligible_unique": n,
                "excluded_unique": sum(r["excluded_test_unique"] for r in values),
                "contexts": sum(r["contexts"] for r in values)}
        for key in ("performance", "target_coverage", "mean_candidates"):
            item[key] = sum(r[key] * r["eligible_test_unique"] for r in values) / n
        sample_rows.append(item)
    summary = []
    for condition in EVALUATION_CONDITIONS:
        for model in EVALUATION_MODELS:
            h = next(r for r in sample_rows if r["domain"] == "herbal" and r["condition"] == condition and r["method"] == model)
            f = [r for r in sample_rows if r["domain"] == "food" and r["condition"] == condition and r["method"] == model]
            a = np.array([r["performance"] for r in f])
            summary.append({"condition": condition, "method": model, "metric": h["metric"],
                "eligible_unique_per_cohort": h["eligible_unique"], "excluded_unique_per_cohort": h["excluded_unique"],
                "herbal_performance": h["performance"], "food_mean": float(a.mean()),
                "food_sd": float(a.std(ddof=1)), "food_p2_5": float(np.percentile(a, 2.5)),
                "food_p97_5": float(np.percentile(a, 97.5)),
                "herbal_target_coverage": h["target_coverage"],
                "food_mean_target_coverage": sum(r["target_coverage"] for r in f) / len(f)})
    for name, records in (("fold_results.csv", fold_rows), ("sample_results.csv", sample_rows), ("summary.csv", summary)):
        write_csv(out / name, tuple(records[0]), records)
    audits.sort(key=lambda r: (r["domain"], r["replicate"]))
    (out / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    metadata.update(status="complete", evaluated_cohorts=len(audits), fold_result_rows=len(fold_rows),
                    sample_result_rows=len(sample_rows), reference_ranking_checks=sum(r["reference_ranking_checks"] for r in audits))
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_evaluation_figure(summary, ROOT / "figures")
    print("Complete: work/evaluation/summary.csv and figures/Figure2_recommendation_performance.png", flush=True)
    return summary


def make_evaluation_figure(summary, output_dir):
    """Primary leave-one-out evaluation: every ingredient is withheld once."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = ("popularity", "mean_conditional", "mean_jaccard")
    rows = [next(r for r in summary if r["condition"] == "N-1" and r["method"] == m) for m in models]
    herbal = [100 * float(r["herbal_performance"]) for r in rows]
    food = [100 * float(r["food_mean"]) for r in rows]
    errors = [[v - 100 * float(r["food_p2_5"]) for v,r in zip(food,rows)],
              [100 * float(r["food_p97_5"]) - v for v,r in zip(food,rows)]]
    with mpl.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 9, "axes.linewidth": .8, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "HerbalFormulaCompletion"}):
        fig, ax = plt.subplots(figsize=(7.2, 3.9))
        width = .32
        ax.bar([i-width/2 for i in range(3)], herbal, width, color="#222222", edgecolor="#222222", linewidth=.5, label="Herbal")
        ax.bar([i+width/2 for i in range(3)], food, width, color="#bdbdbd", edgecolor="#555555", linewidth=.5,
               label="Food", yerr=errors, error_kw={"elinewidth": .7, "capsize": 3, "capthick": .7, "ecolor": "#555555"})
        ax.set_xticks(range(3), ["Popularity", "Mean conditional probability", "Mean pairwise Jaccard"])
        ax.set_ylabel("Hit@10 (%)")
        ax.set_ylim(0, 65)
        ax.set_yticks(range(0, 61, 10))
        ax.legend(frameon=False, loc="upper left")
        for ext in ("png", "svg", "pdf"):
            metadata = {"Date": None} if ext == "svg" else ({"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
            fig.savefig(output_dir / f"Figure2_recommendation_performance.{ext}", dpi=600, bbox_inches="tight", facecolor="white", metadata=metadata)
        plt.close(fig)
    caption = ("Fig. 2. Recovery of a single withheld ingredient. Each ingredient in every composition was withheld once, "
               "with all remaining ingredients provided as input. Hit@10 was averaged first within each composition and then equally across all 2,009 compositions. "
               "Black bars show herbal performance; gray bars show mean performance across 100 matched food samples. "
               "Food error bars represent empirical 2.5th–97.5th percentiles across samples, not confidence intervals.")
    (output_dir / "Figure2_caption.txt").write_text(caption + "\n", encoding="utf-8")
    make_supplementary_evaluation_figures(summary, output_dir)
    for path in output_dir.glob("Figure*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def make_supplementary_evaluation_figures(summary, output_dir):
    """Separate herbal and food plots, following the Figure 1 style."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_models = ("popularity", "mean_conditional", "mean_jaccard")
    labels = ("Popularity", "Mean conditional probability", "Mean pairwise Jaccard")
    colors = ("#eeeeee", "#bdbdbd", "#222222")
    with mpl.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 9, "axes.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "HerbalFormulaCompletion"}):
        for domain, title in (("herbal", "Herbal"), ("food", "Food")):
            fig, ax = plt.subplots(figsize=(7.2, 3.9))
            width = .24
            for method_index, (model, label, color) in enumerate(zip(plot_models, labels, colors)):
                rows = [next(r for r in summary if r["condition"] == c and r["method"] == model)
                        for c in EVALUATION_CONDITIONS]
                key = "herbal_performance" if domain == "herbal" else "food_mean"
                values = [100 * float(r[key]) for r in rows]
                errors = None
                if domain == "food":
                    errors = [[v - 100 * float(r["food_p2_5"]) for v, r in zip(values, rows)],
                              [100 * float(r["food_p97_5"]) - v for v, r in zip(values, rows)]]
                ax.bar([i + (method_index - 1) * width for i in range(5)], values, width,
                       color=color, edgecolor="#555555", linewidth=.5, label=label,
                       yerr=errors, error_kw={"elinewidth": .7, "capsize": 2, "capthick": .7,
                                             "ecolor": "#555555"})
            ax.set_title(title, fontsize=10)
            ax.set_xticks(range(5), ["2", "3", "50%", "75%", "N−1"])
            ax.set_xlabel("Input ingredients retained")
            ax.set_ylabel("Recall@10 / Hit@10 (%)")
            ax.set_ylim(0, 65)
            ax.set_yticks(range(0, 61, 10))
            ax.legend(frameon=False, loc="upper left", fontsize=8)
            for ext in ("png", "svg", "pdf"):
                metadata = {"Date": None} if ext == "svg" else ({"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
                fig.savefig(output_dir / f"FigureS1_{domain}_performance.{ext}", dpi=600,
                            bbox_inches="tight", facecolor="white", metadata=metadata)
            plt.close(fig)
    caption = ("Fig. 2. Ingredient recommendation performance in herbal formulas and matched food samples. "
               "Separate plots show herbal performance and mean performance across 100 matched food samples. "
               "Bars represent Popularity, Mean conditional probability, and Mean pairwise Jaccard. "
               "Food error bars show the empirical 2.5th–97.5th percentiles across samples, not confidence intervals. "
               "Input conditions retain 2 or 3 ingredients, 50% or 75% of ingredients, or all but one ingredient (N−1). "
               "Performance is Recall@10, equivalent to Hit@10 for N−1. Scores are averaged within each composition and then equally across eligible compositions. "
               "Each cohort includes 1,899 eligible compositions for the 2-ingredient condition, 1,753 for the 3-ingredient condition, and 2,009 for each remaining condition.")
    (output_dir / "FigureS1_caption.txt").write_text(caption.replace("Fig. 2.", "Fig. S1.") + "\n", encoding="utf-8")


def make_figure1(herbal_counts, food_counts):
    sizes = list(range(2, 20))
    food = [food_counts[size] for size in sizes]
    herbal = [herbal_counts[size] for size in sizes]
    food_total = sum(food)
    herbal_total = sum(herbal)
    food_percent = [100 * count / food_total for count in food]
    herbal_percent = [100 * count / herbal_total for count in herbal]

    print("Ingredients | Food n (%)          | Herbal n (%)")
    for size, food_count, food_pct, herbal_count, herbal_pct in zip(
        sizes, food, food_percent, herbal, herbal_percent
    ):
        print(
            f"{size:>11} | {food_count:>7,} ({food_pct:>5.2f}%) | "
            f"{herbal_count:>5,} ({herbal_pct:>5.2f}%)"
        )
    print(f"Totals: food={food_total:,}, herbal={herbal_total:,}")

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
        [size - width / 2 for size in sizes], food_percent, width,
        color="#bdbdbd", edgecolor="#555555", linewidth=0.5,
        label=f"Food before matching (n={food_total:,})",
    )
    ax.bar(
        [size + width / 2 for size in sizes], herbal_percent, width,
        color="#222222", edgecolor="#222222", linewidth=0.5,
        label=f"Herbal (n={herbal_total:,})",
    )
    ax.set_xlabel("Number of ingredients")
    ax.set_ylabel("Compositions (%)")
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


import numpy as np
from numba import njit

@njit(cache=True)
def curveball_trade(matrix, lengths):
    n = len(lengths)
    a = np.random.randint(n)
    b = np.random.randint(n - 1)
    if b >= a: b += 1
    la, lb = lengths[a], lengths[b]
    shared = np.empty(19, np.int64)
    pool = np.empty(38, np.int64)
    nc, na, total = 0, 0, 0
    for i in range(la):
        x = matrix[a, i]
        found = False
        for j in range(lb):
            if matrix[b, j] == x: found = True; break
        if found:
            shared[nc] = x; nc += 1
        else:
            pool[total] = x; total += 1; na += 1
    for j in range(lb):
        x = matrix[b, j]
        found = False
        for i in range(la):
            if matrix[a, i] == x: found = True; break
        if not found: pool[total] = x; total += 1
    if na == 0 or total == na: return
    for k in range(total - 1, 0, -1):
        j = np.random.randint(k + 1)
        pool[k], pool[j] = pool[j], pool[k]
    for j in range(nc):
        matrix[a,j] = shared[j]; matrix[b,j] = shared[j]
    for j in range(na): matrix[a,nc+j] = pool[j]
    for j in range(na,total): matrix[b,nc+j-na] = pool[j]

@njit(cache=True)
def pair_vector(matrix, lengths, lookup, pair_a, pair_b, k):
    counts = np.zeros((k,k), np.int32)
    for r in range(len(lengths)):
        for u in range(lengths[r]):
            a = lookup[matrix[r,u]]
            if a < 0: continue
            for v in range(u+1,lengths[r]):
                b = lookup[matrix[r,v]]
                if b < 0: continue
                if a < b: counts[a,b] += 1
                else: counts[b,a] += 1
    result = np.empty(len(pair_a),np.int32)
    for p in range(len(pair_a)): result[p] = counts[pair_a[p],pair_b[p]]
    return result

@njit(cache=True)
def curveball_chain(original, lengths, lookup, pair_a, pair_b, k, draws, seed, burn, spacing):
    np.random.seed(seed)
    matrix = original.copy()
    values = np.empty((draws,len(pair_a)),np.int16)
    overlap = np.empty(draws,np.float64)
    for _ in range(burn): curveball_trade(matrix,lengths)
    expected_columns = np.zeros(len(lookup),np.int64)
    for r in range(len(lengths)):
        for u in range(lengths[r]): expected_columns[original[r,u]] += 1
    for draw in range(draws):
        for _ in range(spacing): curveball_trade(matrix,lengths)
        columns = np.zeros(len(lookup),np.int64)
        common = 0
        for r in range(len(lengths)):
            for u in range(lengths[r]):
                a = matrix[r,u]
                columns[a] += 1
                for v in range(u):
                    if matrix[r,v] == a: raise ValueError('Duplicate ingredient after trade')
                for v in range(lengths[r]):
                    if original[r,v] == a: common += 1; break
        if not np.array_equal(columns,expected_columns): raise ValueError('Column margins changed')
        overlap[draw] = common / lengths.sum()
        values[draw] = pair_vector(matrix,lengths,lookup,pair_a,pair_b,k)
    return values,overlap,matrix


# Section 2.5: fixed-margin randomization, after Strona et al. (2014),
# doi:10.1038/ncomms5114. Binary unique compositions, NOT source-weighted margins.
# Its finite Curveball chains are checked empirically; exact independent sampling
# is not claimed. Ahn et al. (2011), doi:10.1038/srep00196, motivates a
# frequency-controlled food comparison, not this exact co-occurrence protocol.

STRUCTURE_SEED = 20260928
STRUCTURE_MIN_OCCURRENCES = 10


def structure_bh(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind="stable")
    q = np.empty(len(pvalues))
    q[order] = np.minimum(1, np.minimum.accumulate(
        (pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1))[::-1])[::-1])
    return q


def structure_prepare(rows, field, weighted=False):
    vocabulary = sorted({x for r in rows for x in r[field].split("|")})
    index = {x:i for i,x in enumerate(vocabulary)}
    expanded = [r for r in rows for _ in range(int(r["weight"]) if weighted else 1)]
    matrix = np.full((len(expanded), 19), -1, dtype=np.int64)
    lengths = np.array([len(r[field].split("|")) for r in expanded],dtype=np.int64)
    for i,r in enumerate(expanded):
        matrix[i,:lengths[i]] = sorted(index[x] for x in r[field].split("|"))
    frequency = np.bincount(matrix[matrix >= 0], minlength=len(vocabulary))
    retained = np.flatnonzero(frequency >= STRUCTURE_MIN_OCCURRENCES)
    lookup = np.full(len(vocabulary), -1, dtype=np.int64)
    lookup[retained] = np.arange(len(retained))
    pair_a,pair_b = np.triu_indices(len(retained), 1)
    return vocabulary,matrix,lengths,frequency,retained,lookup,pair_a,pair_b


def structure_diagnostics(chains, overlaps):
    def diagnose(traces):
        m = min(len(t) for t in traces)
        x = np.array([t[:m] for t in traces],dtype=float)
        within = float(np.mean(np.var(x,axis=1,ddof=1)))
        between = float(m*np.var(x.mean(axis=1),ddof=1))
        rhat = math.sqrt(((m-1)/m*within+between/m)/within) if within else 1.0
        correlations = [float(np.corrcoef(t[:-1],t[1:])[0,1]) if np.std(t)>0 else 0.0 for t in traces]
        return {"rhat":rhat,"lag1":correlations}
    concentration = [(np.einsum("ij,ij->i",c,c,dtype=np.int64)-c.sum(axis=1))/2 for c in chains]
    return {"incidence_overlap":diagnose(overlaps),"repeated_pair_count":diagnose(concentration)}, concentration


def structure_algorithm_hash():
    import inspect
    import numba
    functions = (curveball_trade.py_func, pair_vector.py_func, curveball_chain.py_func,
                 structure_prepare, structure_bh, structure_diagnostics, structure_cohort)
    material = "\n".join(inspect.getsource(fn) for fn in functions)
    material += f"\n{STRUCTURE_MIN_OCCURRENCES}|{np.__version__}|{numba.__version__}"
    return hashlib.sha256(material.encode()).hexdigest()


def structure_cohort(task):
    domain,replicate,rows,field,out,draws,weighted,spacing_override = task
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    vocabulary,matrix,lengths,frequency,retained,lookup,pa,pb = structure_prepare(rows,field,weighted)
    if len(matrix)>=32767: raise ValueError("Null count storage requires fewer than 32767 rows")
    if len(pa)==0: raise ValueError("No eligible ingredient pairs")
    seed = STRUCTURE_SEED + replicate*100 + (50000 if weighted else 0)
    algorithm_hash = structure_algorithm_hash()
    previous=out/'metadata.json'
    if previous.exists():
        saved=json.loads(previous.read_text())
        summary=saved.get('summary',{})
        if (saved.get('status')=='complete' and saved.get('algorithm_sha256')==algorithm_hash
            and saved.get('seed')==seed
            and saved.get('chains')==2 and summary.get('null_draws')==draws
            and summary.get('basis')==('source_records' if weighted else 'unique_compositions')
            and summary.get('tested_pairs')==len(pa)
            and summary.get('trade_spacing_per_row',0)>=(spacing_override or 5)):
            with np.load(out/'margin_check_states.npz') as state:
                reusable=np.array_equal(state['original'],matrix) and np.array_equal(state['vocabulary'],np.array(vocabulary))
            if reusable and all((out/name).exists() for name in ('pair_results.csv.gz','top_pairs.csv','mixing_trace.csv','illustrative_nulls.npz')): return summary
    previous.write_text(json.dumps({'status':'running'})+'\n')
    original = pair_vector(matrix,lengths,lookup,pa,pb,len(retained))
    seed = STRUCTURE_SEED + replicate*100 + (50000 if weighted else 0)
    spacing = spacing_override or 5
    for attempt in range(3):
        chains=[];overlaps=[];final_states=[]
        for chain,n in enumerate((draws//2,draws-draws//2)):
            values,overlap,last = curveball_chain(matrix,lengths,lookup,pa,pb,len(retained),n,
                seed+chain,20*spacing//5*len(rows if not weighted else matrix),spacing*len(matrix))
            chains.append(values);overlaps.append(overlap);final_states.append(last)
        diagnostic,traces=structure_diagnostics(chains,overlaps)
        passed=all(d["rhat"]<1.05 and max(abs(x) for x in d["lag1"])<.2 for d in diagnostic.values())
        if passed: break
        spacing *= 2
    if not passed: raise RuntimeError(f"Mixing checks failed for {domain} {replicate}: {diagnostic}")
    null=np.concatenate(chains)
    mean=null.mean(axis=0)
    second=np.einsum("ij,ij->j",null,null,dtype=np.float64)
    sd=np.sqrt(np.maximum(0,(second-len(null)*mean*mean)/(len(null)-1)))
    p=(1+np.sum(null>=original,axis=0))/(len(null)+1)
    q=structure_bh(p)
    excess=original-mean
    ratio=np.divide(original,mean,out=np.full_like(mean,np.nan),where=mean>0)
    z=np.divide(excess,sd,out=np.zeros_like(mean),where=sd>0)
    records=[]
    for j,(a,b) in enumerate(zip(pa,pb)):
        ia,ib=retained[a],retained[b]
        records.append({"ingredient_a":vocabulary[ia],"ingredient_b":vocabulary[ib],
            "occurrence_a":int(frequency[ia]),"occurrence_b":int(frequency[ib]),
            "observed":int(original[j]),"null_mean":float(mean[j]),"null_sd":float(sd[j]),
            "excess":float(excess[j]),"observed_expected_ratio":float(ratio[j]) if mean[j]>0 else "",
            "standardized_excess":float(z[j]),"p_upper":float(p[j]),"q_bh":float(q[j]),
            "enriched_q05":bool(q[j]<=.05 and excess[j]>0),
            "network_edge":bool(q[j]<=.05 and excess[j]>0 and original[j]>=10)})
    with gzip.open(out/'pair_results.csv.gz','wt',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=tuple(records[0]));writer.writeheader();writer.writerows(records)
    ranked=sorted(range(len(records)),key=lambda i:(-excess[i],records[i]['ingredient_a'],records[i]['ingredient_b']))
    write_csv(out/'top_pairs.csv',tuple(records[0]),[records[i] for i in ranked[:100]])
    chosen=ranked[:4]
    if domain=='herbal':
        for wanted in ({'황금','황련'},{'목단피','산수유'}):
            chosen += [i for i,r in enumerate(records) if {r['ingredient_a'],r['ingredient_b']}==wanted]
    chosen=list(dict.fromkeys(chosen))
    np.savez_compressed(out/'illustrative_nulls.npz',counts=null[:,chosen],
        labels=np.array([records[i]['ingredient_a']+' / '+records[i]['ingredient_b'] for i in chosen]),
        observed=original[chosen])
    write_csv(out/'mixing_trace.csv',('chain','draw','incidence_overlap','repeated_pair_count'),
        ({'chain':c+1,'draw':j+1,'incidence_overlap':float(overlaps[c][j]),'repeated_pair_count':float(traces[c][j])}
         for c in range(2) for j in range(len(overlaps[c]))))
    np.savez_compressed(out/'margin_check_states.npz',original=matrix,lengths=lengths,
        chain1=final_states[0],chain2=final_states[1],vocabulary=np.array(vocabulary))
    global_null=np.concatenate(traces)
    global_original=int(np.sum(original.astype(np.int64)*(original-1)//2))
    null_lag=diagnostic['repeated_pair_count']['lag1']
    summary={'domain':domain,'replicate':replicate,'basis':'source_records' if weighted else 'unique_compositions',
        'compositions':len(matrix),'vocabulary':len(vocabulary),'eligible_ingredients':len(retained),
        'tested_pairs':len(pa),'enriched_pairs_q05':sum(r['enriched_q05'] for r in records),
        'network_edges_q05_support10':sum(r['network_edge'] for r in records),
        'repeated_pair_observed':global_original,'repeated_pair_null_mean':float(global_null.mean()),
        'repeated_pair_observed_expected':float(global_original/global_null.mean()),
        'repeated_pair_p_upper':float((1+np.sum(global_null>=global_original))/(draws+1)),
        'null_draws':draws,'trade_spacing_per_row':spacing,
        'mixing_rhat':diagnostic['repeated_pair_count']['rhat'],
        'maximum_absolute_lag1':max(abs(x) for d in diagnostic.values() for x in d['lag1'])}
    metadata={'status':'complete','algorithm_sha256':algorithm_hash,'summary':summary,'seed':seed,'chains':2,'diagnostics':diagnostic,
        'burn_in_trades_per_chain':20*spacing//5*len(matrix),'between_draw_trades':spacing*len(matrix),
        'row_and_column_margins_checked_each_draw':True,'duplicate_ingredients_checked_each_draw':True,
        'null_duplicate_rows':'allowed; do not re-merge randomized compositions',
        'multiplicity_family':'all unordered pairs of ingredients occurring in at least 10 rows, including observed-zero pairs',
        'p_values':'one-sided enrichment, (1 + null counts >= observed)/(B + 1); finite-chain Monte Carlo estimates',
        'adjustment':'Benjamini-Hochberg separately per cohort, q<=0.05; exploratory associations, not clinical rules',
        'network_display':'q<=0.05 and observed>=10; top edges by observed-minus-null mean; selection is descriptive'}
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    return summary


def structure_frequency_tables(work_dir,out):
    full=[];cohorts={};concentration=[]
    memberships=read_rows(work_dir/'matching/membership.csv')
    wanted={r['composition_id'] for r in memberships}
    food_selected={}
    herbs=read_rows(work_dir/'herbal/unique_compositions.csv')
    for domain,field,path in [('herbal','herbs',work_dir/'herbal/unique_compositions.csv'),
                               ('food','ingredients',work_dir/'food/unique_compositions.csv')]:
        weighted=Counter();unique=Counter();n=0;weight_sum=0
        with path.open(encoding='utf-8-sig',newline='') as h:
            for row in csv.DictReader(h):
                xs=row[field].split('|');w=int(row['weight']);n+=1;weight_sum+=w
                unique.update(xs);weighted.update({x:w for x in xs})
                if domain=='food' and row['composition_id'] in wanted: food_selected[row['composition_id']]=row
        for basis,counts,denominator in [('source_records',weighted,weight_sum),('unique_compositions',unique,n)]:
            ordered=sorted(counts,key=lambda x:(-counts[x],x))
            for rank,name in enumerate(ordered,1):
                full.append({'domain':domain,'basis':basis,'rank':rank,'ingredient':name,'count':counts[name],
                             'records':denominator,'prevalence':counts[name]/denominator})
            concentration.append({'domain':domain,'basis':basis,'records':denominator,'vocabulary':len(counts),
                'top10_incidence_share':sum(counts[x] for x in ordered[:10])/sum(counts.values()),
                'top_ingredient':ordered[0],'top_ingredient_prevalence':counts[ordered[0]]/denominator})
    write_csv(out/'ingredient_frequencies.csv',tuple(full[0]),full)
    write_csv(out/'frequency_summary.csv',tuple(concentration[0]),concentration)
    for m in memberships:
        row=food_selected[m['composition_id']]
        assert int(row['weight'])==int(m['weight']) and int(row['ingredient_count'])==int(m['ingredient_count'])
        cohorts.setdefault(int(m['replicate']),[]).append(row)
    assert set(cohorts)==set(range(1,101))
    expected=Counter(int(r['herb_count']) for r in herbs)
    for rows in cohorts.values():
        assert len(rows)==len({r['composition_id'] for r in rows})==2009
        assert Counter(int(r['ingredient_count']) for r in rows)==expected
    return herbs,cohorts


def structure_read_pairs(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as h:
        rows=list(csv.DictReader(h))
    for r in rows:
        for key in ('occurrence_a','occurrence_b','observed'): r[key]=int(r[key])
        for key in ('null_mean','null_sd','excess','standardized_excess','p_upper','q_bh'):r[key]=float(r[key])
        r['network_edge']=r['network_edge']=='True'
    return rows


def structure_font():
    from matplotlib import font_manager
    for path in font_manager.findSystemFonts():
        if 'Nanum' in path:
            try: font_manager.fontManager.addfont(path)
            except Exception: pass
    for name in ('AppleGothic','NanumGothic','Noto Sans CJK KR','Malgun Gothic'):
        try:
            font_manager.findfont(name,fallback_to_default=False)
            return name
        except ValueError: pass
    raise RuntimeError('A Korean font is required: install fonts-nanum in Colab/Linux.')


def structure_save(fig,out,stem):
    for ext in ('png','svg','pdf'):
        metadata={'Date':None} if ext=='svg' else ({'CreationDate':None,'ModDate':None} if ext=='pdf' else None)
        fig.savefig(out/f'{stem}.{ext}',dpi=400,bbox_inches='tight',facecolor='white',metadata=metadata)
    plt.close(fig)
    path=out/f'{stem}.svg';path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')


def make_structure_figures(work_dir,figures):
    import networkx as nx
    out=work_dir/'structure';figures.mkdir(parents=True,exist_ok=True)
    font=structure_font()
    with mpl.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':9,'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','svg.hashsalt':'HerbalFormulaCompletion-structure'}):
        frequencies=read_rows(out/'ingredient_frequencies.csv')
        fig,axs=plt.subplots(1,2,figsize=(9,3.8))
        for ax,domain,title in zip(axs,('herbal','food'),('A  Herbal','B  Food')):
            for basis,color,style,label in [('source_records','#222222','-','Source records'),('unique_compositions','#888888','--','Unique compositions')]:
                rows=[r for r in frequencies if r['domain']==domain and r['basis']==basis]
                ax.plot([int(r['rank']) for r in rows],[100*float(r['prevalence']) for r in rows],
                    color=color,ls=style,lw=1.2,label=label)
            inset=ax.inset_axes([.45,.32,.5,.43])
            for basis,color,style in [('source_records','#222222','-'),('unique_compositions','#888888','--')]:
                sub=[r for r in frequencies if r['domain']==domain and r['basis']==basis]
                inset.plot([int(r['rank']) for r in sub],[100*float(r['prevalence']) for r in sub],color=color,ls=style,lw=.8)
            inset.set_xscale('log');inset.set_yscale('log');inset.set_xlim(1,2000);inset.set_ylim(.0005,100)
            inset.tick_params(labelsize=6);inset.set_title('Log–log view',fontsize=7,loc='left')
            ax.set_xlim(0,1500);ax.set_ylim(0,60);ax.set_xlabel('Ingredient frequency rank')
            ax.set_ylabel('Records containing ingredient (%)');ax.set_title(title,loc='left',fontsize=10)
            ax.legend(frameon=False,fontsize=8)
        fig.tight_layout();structure_save(fig,figures,'Figure3_ingredient_frequency')

        fig,axs=plt.subplots(1,2,figsize=(9,3.6))
        before=np.array([[1,1,1,0,0,0],[0,0,0,1,1,1]])
        after=np.array([[1,0,1,1,0,0],[0,1,0,0,1,1]])
        labels=['황금','황련','감초','당귀','천궁','작약']
        for ax,data,title in zip(axs,(before,after),('Before trade','After trade')):
            ax.imshow(data,cmap='Greys',vmin=0,vmax=1,aspect='equal')
            ax.set_xticks(range(6),labels,fontfamily=font,fontsize=10)
            ax.set_yticks([0,1],['Formula A','Formula B'])
            for (i,j),value in np.ndenumerate(data):
                ax.text(j,i,str(value),ha='center',va='center',color='white' if value else '#333333',fontsize=10)
            for i in range(2):ax.text(6,i,f'{data[i].sum()} herbs',va='center',fontsize=9)
            ax.text(-.6,2,'Counts:',ha='right',fontsize=8)
            for j in range(6):ax.text(j,2,str(data[:,j].sum()),ha='center',fontsize=9)
            ax.set_title(title,fontsize=10);ax.set_xlim(-.5,7.1);ax.set_ylim(2.6,-.7)
            ax.tick_params(length=0);ax.spines[['left','bottom']].set_visible(False)
        fig.text(.5,.02,'Illustrative binary compositions: row sizes and ingredient counts are unchanged.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.08,1,1]);structure_save(fig,figures,'Figure4_fixed_margin_trade')

        pairs={d:structure_read_pairs(out/f'cohorts/{d}_{r:03d}/pair_results.csv.gz')
               for d,r in [('herbal',0),('food',1)]}
        fig,axs=plt.subplots(1,2,figsize=(9,4))
        for ax,domain,title in zip(axs,('herbal','food'),('A  Herbal','B  Food sample 1')):
            rows=pairs[domain]
            for significant,color,label in [(False,'#bdbdbd','Other tested pairs'),(True,'#222222','Enriched (BH q ≤ 0.05)')]:
                subset=[r for r in rows if (r['q_bh']<=.05 and r['excess']>0)==significant]
                ax.scatter([r['null_mean'] for r in subset],[r['observed'] for r in subset],s=5,color=color,alpha=.5,label=label,rasterized=True)
            limit=max(max(r['null_mean'],r['observed']) for r in rows)*1.07
            ax.plot([0,limit],[0,limit],color='#888888',lw=.8,ls='--')
            ax.set_xlim(0,limit);ax.set_ylim(0,limit);ax.set_aspect('equal')
            ax.set_xlabel('Mean co-occurrence in randomized data');ax.set_ylabel('Observed co-occurrence')
            ax.set_title(title,loc='left',fontsize=10);ax.legend(frameon=False,fontsize=7,loc='upper left')
            marked = sorted(rows, key=lambda r: (-r['excess'], r['ingredient_a'], r['ingredient_b']))[:3]
            for j, r in enumerate(marked):
                ax.annotate(r['ingredient_a']+'–'+r['ingredient_b'], (r['null_mean'],r['observed']),
                    xytext=(.62,.43-j*.08), textcoords='axes fraction', fontsize=7,
                    bbox={'facecolor':'white','edgecolor':'none','pad':1},
                    fontfamily=font if domain=='herbal' else 'DejaVu Sans',
                    arrowprops={'arrowstyle':'-', 'color':'#777777','lw':.5})
        fig.tight_layout();structure_save(fig,figures,'Figure5_observed_vs_randomized')

        herbal=pairs['herbal']
        all_edges=sorted([r for r in herbal if r['network_edge']],key=lambda r:(-r['excess'],r['ingredient_a'],r['ingredient_b']))
        display_edges=all_edges[:40]
        graph=nx.Graph()
        for r in display_edges:graph.add_edge(r['ingredient_a'],r['ingredient_b'],weight=r['excess'])
        fig,ax=plt.subplots(figsize=(9,6))
        if graph:
            components=sorted(nx.connected_components(graph),key=lambda c:(-len(c),sorted(c)))
            pos={}
            for ci,component in enumerate(components):
                sub=graph.subgraph(sorted(component)).copy()
                local=nx.spring_layout(sub,seed=STRUCTURE_SEED,weight=None,iterations=500,k=1.3/math.sqrt(len(sub)))
                if ci==0:
                    for x,v in local.items():pos[x]=np.array([1.6*v[0],1.4*v[1]])
                else:
                    cy=1.2-(ci-1)*1.2
                    for x,v in local.items():pos[x]=np.array([2.8+.45*v[0],cy+.3*v[1]])
            # Resolve close labels without treating layout coordinates as measured distances.
            main_nodes=sorted(components[0])
            for _ in range(100):
                for i,a in enumerate(main_nodes):
                    for b in main_nodes[i+1:]:
                        delta=pos[b]-pos[a];distance=float(np.linalg.norm(delta))
                        if 0<distance<.38:
                            shift=.5*(.38-distance)*delta/distance
                            pos[a]-=shift;pos[b]+=shift
            freq={r['ingredient']:int(r['count']) for r in frequencies if r['domain']=='herbal' and r['basis']=='unique_compositions'}
            nx.draw_networkx_edges(graph,pos,ax=ax,width=[.5+3*graph[a][b]['weight']/max(r['excess'] for r in display_edges) for a,b in graph.edges],edge_color='#888888',alpha=.75)
            nx.draw_networkx_nodes(graph,pos,ax=ax,node_size=[150+900*freq[x]/2009 for x in graph],node_color='white',edgecolors='#555555',linewidths=.7)
            nx.draw_networkx_labels(graph,pos,ax=ax,font_family=font,font_size=9,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':.5})
            position_rows=[{'ingredient':x,'x':float(pos[x][0]),'y':float(pos[x][1]),'unique_occurrences':freq[x]} for x in sorted(graph)]
            write_csv(out/'network_nodes.csv',tuple(position_rows[0]),position_rows)
        else: ax.text(.5,.5,'No pairs met the prespecified network criteria.',ha='center',transform=ax.transAxes)
        ax.axis('off');ax.margins(.15)
        ax.set_title('Herbal co-occurrence network',fontsize=11)
        fig.text(.5,.025,'Top 40 edges by excess co-occurrence; BH q ≤ 0.05 and observed count ≥ 10.',ha='center',fontsize=8)
        structure_save(fig,figures,'Figure6_herbal_relationships')
        if display_edges:write_csv(out/'network_edges.csv',tuple(display_edges[0]),display_edges)
        herbs=read_rows(work_dir/'herbal/unique_compositions.csv')
        examples=[]
        for r in display_edges:
            target={r['ingredient_a'],r['ingredient_b']}
            supporting=[x for x in herbs if target<=set(x['herbs'].split('|'))]
            for x in sorted(supporting,key=lambda x:(-int(x['weight']),x['composition_id']))[:3]:
                examples.append({'ingredient_a':r['ingredient_a'],'ingredient_b':r['ingredient_b'],
                    'composition_id':x['composition_id'],'formula_names':x['formula_names'],
                    'herbs':x['herbs'],'source_weight':x['weight']})
        if examples:write_csv(out/'network_formula_examples.csv',tuple(examples[0]),examples)

        # Draw full null distributions for descriptive, explicitly selected pairs.
        with np.load(out/'cohorts/herbal_000/illustrative_nulls.npz') as data:
            fig,axs=plt.subplots(2,3,figsize=(10,5.5))
            for j,ax in enumerate(axs.flat):
                if j>=len(data['labels']):ax.axis('off');continue
                values=data['counts'][:,j];obs=int(data['observed'][j])
                ax.hist(values,bins=np.arange(values.min()-.5,values.max()+1.5),color='#bdbdbd',edgecolor='white')
                ax.axvline(obs,color='#222222',lw=1.2,label=f'Observed: {obs}')
                ax.set_title(str(data['labels'][j]),fontfamily=font,fontsize=10)
                ax.set_xlabel('Co-occurrence count');ax.set_ylabel('Randomized matrices');ax.legend(frameon=False,fontsize=7)
            fig.tight_layout();structure_save(fig,figures,'FigureS2_pair_null_distributions')
        summaries=read_rows(out/'cohort_summary.csv')
        fig,axs=plt.subplots(1,2,figsize=(9,3.7))
        food=[r for r in summaries if r['domain']=='food' and r['basis']=='unique_compositions']
        herb=next(r for r in summaries if r['domain']=='herbal' and r['basis']=='unique_compositions')
        axs[0].hist([float(r['repeated_pair_observed_expected']) for r in food],bins=15,color='#bdbdbd',edgecolor='white',label='100 food samples')
        axs[0].axvline(float(herb['repeated_pair_observed_expected']),color='#222222',label='Herbal')
        axs[0].axvline(1,color='#888888',ls='--',lw=.8)
        axs[0].set_xlabel('Repeated pair count: observed / null mean');axs[0].set_ylabel('Food samples');axs[0].legend(frameon=False,fontsize=8)
        traces=read_rows(out/'cohorts/herbal_000/mixing_trace.csv')
        for chain,color in [('1','#222222'),('2','#aaaaaa')]:
            vals=[r for r in traces if r['chain']==chain]
            axs[1].plot([int(r['draw']) for r in vals],[float(r['incidence_overlap']) for r in vals],color=color,lw=.5,label='Chain '+chain)
        axs[1].set_xlabel('Retained null draw');axs[1].set_ylabel('Original incidence retained');axs[1].legend(frameon=False,fontsize=8)
        fig.tight_layout();structure_save(fig,figures,'FigureS3_null_summary_and_mixing')
    captions={
      'Figure3':'Ingredient rank–frequency distributions. Frequencies are proportions of eligible preprocessed source records (weights retained) or unique compositions. Food uses all 770,945 unique compositions, representing 921,927 eligible source records; herbal uses 2,009 unique compositions, representing 2,992 eligible source records. Main panels use the same linear axes; insets use the same logarithmic axes to show the tails. No power-law fit is claimed.',
      'Figure4':'Illustration of a valid Curveball trade. Shared ingredients are retained; ingredients unique to the two selected compositions are randomly redistributed while keeping each row size. This simple example exchanges Hwangryeon and Danggwi. It is schematic, not two named recorded prescriptions. Repeated trades preserve every ingredient column total and every composition row total.',
      'Figure5':'Observed and fixed-margin null co-occurrence counts. Each point is an unordered pair of ingredients occurring in at least 10 unique compositions; observed-zero pairs are included in the testing family. Black points indicate positive enrichment with Benjamini–Hochberg q≤0.05 from one-sided Monte Carlo tests. The diagonal denotes observed equals null mean. The three pairs with the greatest positive excess in each panel are labeled descriptively. Food sample 1 was selected by its predefined replicate number, not by its result; all 100 food samples were analyzed and saved.',
      'Figure6':'Descriptive herbal co-occurrence network. The 40 largest positive excesses (observed minus null mean) among pairs with BH q≤0.05 and observed co-occurrence≥10 are displayed. Node area reflects ingredient occurrence in unique compositions; edge width reflects excess co-occurrence. Each displayed connected component uses a fixed-seed force layout; components are placed separately for readability. Positions are not a calibrated distance or evidence of therapeutic synergy. Apparent disconnected groups may result from displaying only 40 edges. All eligible pairs and supporting recorded compositions are provided in CSV files.',
      'FigureS2':'Null distributions for four herbal pairs with the greatest observed-minus-null excess, plus Hwanggeum–Hwangryeon and Mokdanpi–Sansuyu when eligible. Selection is illustrative and is explicitly data-dependent for the first four panels; inferential adjustment covers all eligible pairs.',
      'FigureS3':'Repeated co-occurrence concentration and mixing diagnostics. For each eligible ingredient pair, count the unordered pairs of compositions that both contain that ingredient pair, then sum across ingredient pairs. The observed total is divided by its null mean. The food histogram uses all 100 matched samples. The herbal trace shows retained-incidence overlap with the initial matrix in two independent seeded chains; these checks do not prove exact independent sampling.'}
    (figures/'Structure_figure_captions.txt').write_text('\n\n'.join(k+'. '+v for k,v in captions.items())+'\n')


def analyze_structure(work_dir,workers=2,draws=1999):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import platform
    import numba,networkx
    work_dir=Path(work_dir);out=work_dir/'structure';out.mkdir(parents=True,exist_ok=True)
    hashes={name:hashlib.sha256((work_dir/name).read_bytes()).hexdigest() for name in (
        'herbal/unique_compositions.csv','food/unique_compositions.csv','matching/membership.csv')}
    manifest={'status':'running','seed':STRUCTURE_SEED,'null_draws_per_cohort':draws,'chains':2,
        'minimum_occurrences':STRUCTURE_MIN_OCCURRENCES,'primary_basis':'unique compositions, source multiplicities not used',
        'frequency_plot_basis':'eligible preprocessed source records and unique compositions, both shown',
        'null_model':'binary fixed row and column margins via Curveball trades; row duplicates allowed in null matrices',
        'references':{'food_null_model_motivation':'10.1038/srep00196','curveball_algorithm':'10.1038/ncomms5114'},
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_hashes':hashes,'algorithm_sha256':structure_algorithm_hash(),'software':{'python':platform.python_version(),'numpy':np.__version__,'numba':numba.__version__,'networkx':networkx.__version__},
        'not_performed':'No clinical efficacy, no formula-family holdout, no retraining of recommendation models on randomized data.'}
    (out/'metadata.json').write_text(json.dumps(manifest,indent=2)+'\n')
    herbs,foods=structure_frequency_tables(work_dir,out)
    tasks=[('herbal',0,herbs,'herbs',out/'cohorts/herbal_000',draws,False,0)]
    tasks += [('food',rep,rows,'ingredients',out/f'cohorts/food_{rep:03d}',draws,False,0) for rep,rows in sorted(foods.items())]
    print('Structure: herbal fixed-margin null model',flush=True)
    summaries=[structure_cohort(tasks[0])]
    print('Structure: herbal complete; evaluating all 100 food samples',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(structure_cohort,t) for t in tasks[1:]]
        for done,f in enumerate(as_completed(futures),1):
            summaries.append(f.result())
            if done==1 or done%10==0:print(f'Structure food samples complete: {done}/100',flush=True)
    print('Structure: source-record sensitivity and longer-trade sensitivity',flush=True)
    weighted=structure_cohort(('herbal',0,herbs,'herbs',out/'sensitivity/herbal_source_records',draws,True,0))
    longer=structure_cohort(('herbal',0,herbs,'herbs',out/'sensitivity/herbal_longer_trades',draws,False,10))
    summaries.sort(key=lambda r:(r['domain'],r['replicate']))
    write_csv(out/'cohort_summary.csv',tuple(summaries[0]),summaries)
    write_csv(out/'sensitivity_summary.csv',tuple(weighted),[weighted,longer])
    primary=structure_read_pairs(out/'cohorts/herbal_000/pair_results.csv.gz')
    sensitivity=[]
    for label,folder in [('source_records','herbal_source_records'),('longer_trades','herbal_longer_trades')]:
        other=structure_read_pairs(out/f'sensitivity/{folder}/pair_results.csv.gz')
        lookup={(r['ingredient_a'],r['ingredient_b']):r for r in other}
        edges=[r for r in primary if r['network_edge']]
        both=sum(lookup.get((r['ingredient_a'],r['ingredient_b']),{}).get('network_edge',False) for r in edges)
        sensitivity.append({'comparison':label,'primary_edges':len(edges),'also_selected':both,
                            'retained_fraction':both/len(edges) if edges else ''})
    write_csv(out/'network_sensitivity.csv',tuple(sensitivity[0]),sensitivity)
    make_structure_figures(work_dir,ROOT/'figures')
    manifest.update(status='complete',primary_cohorts=len(summaries),sensitivity_runs=2,
        all_saved_draw_margins_verified=True,total_null_draws=draws*(len(summaries)+2))
    (out/'metadata.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Complete: work/structure and Figures 3–6 (plus S2–S3)',flush=True)
    return summaries



def heading(text):
    print(f"\n{text}\n{'-' * len(text)}", flush=True)


def main(workers=2, null_draws=1999):
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
    print("Formulas with 2–19 herbs:",
          f"{herbal_meta['included_formulas_with_2_to_19_herbs']:,}")
    print("Unique compositions:", f"{herbal_meta['unique_compositions']:,}")
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
    food_meta, food_counts = preprocess_food(layers, detections, ROOT / "work/food")

    flow = [
        {"dataset": "food", "source_records": food_meta["source_records"],
         "eligible_records_before_merging": food_meta["included_recipes_after_canonical_mapping"],
         "unique_compositions": food_meta["unique_compositions"]},
        {"dataset": "herbal", "source_records": herbal_meta["source_formulas"],
         "eligible_records_before_merging": herbal_meta["included_formulas_with_2_to_19_herbs"],
         "unique_compositions": herbal_meta["unique_compositions"]},
    ]
    write_csv(ROOT / "work/dataset_flow.csv", tuple(flow[0]), flow)
    heading("Dataset flow: source -> eligibility -> identical-composition merging")
    for row in flow:
        print(f"{row['dataset']}: {row['source_records']:,} -> "
              f"{row['eligible_records_before_merging']:,} -> {row['unique_compositions']:,}")

    heading("Duplicate examples: source-record audit")
    audit_duplicate_examples(herbal_dir, layers, detections, ROOT / "work")
    heading("100 matched food samples")
    matched_food_samples(ROOT / "work/herbal/unique_compositions.csv",
                         ROOT / "work/food/unique_compositions.csv", ROOT / "work/matching")

    heading("Recommendation methods: held-out herbal example")
    recommendation_example(ROOT / "work/herbal/unique_compositions.csv",
                           ROOT / "work/recommendation_example")

    heading("Figure 1")
    make_figure1(herbal_counts, food_counts)
    heading("Full recommendation evaluation")
    evaluate_all(ROOT / "work", workers=workers)
    analyze_structure(ROOT / "work", workers=workers, draws=null_draws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--null-draws", type=int, default=1999)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.evaluate_only and args.structure_only:
        parser.error("Choose only one of --evaluate-only and --structure-only")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.null_draws < 99:
        parser.error("--null-draws must be at least 99")
    if args.structure_only:
        analyze_structure(ROOT / "work", workers=args.workers, draws=args.null_draws)
    elif args.evaluate_only:
        evaluate_all(ROOT / "work", workers=args.workers)
    else:
        main(workers=args.workers, null_draws=args.null_draws)
