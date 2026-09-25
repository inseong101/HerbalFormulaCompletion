#!/usr/bin/env python3
"""Reproduce preprocessing, Figure 1, examples, matching, and recommendation evaluation.

Local setup: python -m pip install "matplotlib>=3.9,<4" "tqdm>=4.66,<5"
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
    if math.comb(n, size) <= maximum:
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
                "context_seed": 20260823, "maximum_contexts_per_composition_condition": 5,
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
    """Plot all methods and conditions; food intervals describe sample variation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ("Popularity", "Mean conditional probability", "Mean pairwise Jaccard")
    with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "hfc-evaluation"}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
        for panel, (ax, model, title) in enumerate(zip(axes, EVALUATION_MODELS, labels)):
            rows = [next(r for r in summary if r["condition"] == c and r["method"] == model)
                    for c in EVALUATION_CONDITIONS]
            herbal = [100 * float(r["herbal_performance"]) for r in rows]
            food = [100 * float(r["food_mean"]) for r in rows]
            lower = [100 * float(r["food_p2_5"]) for r in rows]
            upper = [100 * float(r["food_p97_5"]) for r in rows]
            ax.scatter([i - .12 for i in range(5)], herbal, color="#222222", marker="o",
                       s=45, label="Herbal", zorder=3)
            ax.errorbar([i + .12 for i in range(5)], food,
                        yerr=[[m-l for m,l in zip(food,lower)], [u-m for u,m in zip(upper,food)]],
                        fmt="s", color="#888888", markersize=5.5, capsize=4,
                        linewidth=1.5, label="Food (100-sample mean)", zorder=3)
            ax.set_title(f"{chr(65+panel)}  {title}", loc="left", fontsize=11, pad=12)
            ax.set_xticks(range(5), ["2", "3", "50%", "75%", "N−1"])
            ax.set_xlabel("Input ingredients retained", labelpad=9)
            ax.set_ylim(0, 60)
            ax.set_yticks(range(0, 61, 10))
            ax.set_xlim(-.5, 4.5)
            ax.grid(axis="y", color="#e5e5e5", linewidth=.7)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Recall@10 / Hit@10 (%)")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.5, .065),
                   frameon=False, ncol=2)
        fig.text(.5, .025, "Food error bars: empirical 2.5–97.5 percentiles across 100 samples (not confidence intervals).",
                 ha="center", fontsize=9, color="#444444")
        fig.subplots_adjust(left=.065, right=.985, top=.88, bottom=.27, wspace=.12)
        for ext in ("png", "svg", "pdf"):
            metadata = {"Date": None} if ext == "svg" else ({"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
            fig.savefig(output_dir / f"Figure2_recommendation_performance.{ext}", dpi=300,
                        facecolor="white", metadata=metadata)
        plt.close(fig)
    caption = ("Fig. 2. Ingredient recommendation performance in herbal formulas and matched food samples. "
               "Panels A–C show Popularity, Mean conditional probability, and Mean pairwise Jaccard, respectively. "
               "Black circles indicate herbal performance; gray squares indicate mean performance across 100 matched food samples. "
               "Error bars show the empirical 2.5th–97.5th percentiles of food-sample performance, not confidence intervals. "
               "Input conditions retain 2 or 3 ingredients, 50% or 75% of ingredients, or all but one ingredient (N−1). "
               "Performance is Recall@10, equivalent to Hit@10 for N−1. Scores are averaged within each composition and then equally across eligible compositions. "
               "Each cohort includes 1,899 eligible compositions for the 2-ingredient condition, 1,753 for the 3-ingredient condition, and 2,009 for each remaining condition.")
    (output_dir / "Figure2_caption.txt").write_text(caption + "\n", encoding="utf-8")


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


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}", flush=True)


def main(workers=2):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.evaluate_only:
        evaluate_all(ROOT / "work", workers=args.workers)
    else:
        main(workers=args.workers)
