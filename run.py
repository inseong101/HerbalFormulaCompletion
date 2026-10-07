#!/usr/bin/env python3
"""Reproduce preprocessing and a shared training randomization for prediction and pair maps.

Local setup: python -m pip install "matplotlib>=3.9,<4" "tqdm>=4.66,<5" "numba>=0.61,<0.68" "networkx>=3.4,<4"
Run: python run.py
Training-only control after evaluation: python run.py --predictive-null-only --predictive-draws 20
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
EVALUATION_CONDITIONS = ("N-1",)


def evaluation_contexts(row, condition, ingredient_field):
    """Exhaustive leave-one-ingredient-out inputs for the manuscript evaluation."""
    if condition != "N-1":
        raise ValueError("Only N-1 evaluation is supported")
    items = tuple(sorted(row[ingredient_field].split("|")))
    return list(combinations(items, len(items) - 1))


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
    """One cohort: five folds, three methods, exhaustive N-1 inputs."""
    import numpy as np
    domain, replicate, rows, ingredient_field, output_dir, fold_seed = task
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
                    contexts = evaluation_contexts(row, condition, ingredient_field)
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
                        "metric": "Hit@10",
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
                "contexts_per_composition": "all N leave-one-out cases",
                "input_hashes": hashes,
                "training_weight": "source-record multiplicity",
                "evaluation_weight": "mean within each composition, then equal weight to each eligible composition across all test folds",
                "unseen_hidden_ingredients": "count as misses; retained in metric denominator",
                "ranking": "strict Top-10; descending score then Unicode ingredient name",
                "food_summary": "mean, sample SD, and empirical 2.5th–97.5th percentiles across 100 samples; percentiles are not confidence intervals",
                "scope": "main recommendation performance, excluding structure and learning-curve analyses"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Evaluating herbal cohort...", flush=True)
    fold_rows, audit = evaluate_cohort(("herbal", 0, herbs, "herbs", out / "cohorts/herbal_000", 20260812))
    audits = [audit]
    print("Herbal cohort complete. Evaluating 100 food samples...", flush=True)
    tasks = [("food", rep, rows, "ingredients", out / f"cohorts/food_{rep:03d}", 20260812)
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
    with mpl.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 9, "axes.linewidth": .8, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "HerbalFormulaCompletion"}):
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        width = .32
        ax.bar([i-width/2 for i in range(3)], herbal, width, color="#222222", edgecolor="#222222", linewidth=.5, label="Herbal")
        ax.bar([i+width/2 for i in range(3)], food, width, color="#bdbdbd", edgecolor="#555555", linewidth=.5,
               label="Food", yerr=errors, error_kw={"elinewidth": .7, "capsize": 3, "capthick": .7, "ecolor": "#555555"})
        ax.set_xticks(range(3), ["Popularity", "Mean conditional\nprobability", "Mean pairwise\nJaccard"])
        ax.set_ylabel("Hit@10 (%)")
        ax.set_ylim(0, 65)
        ax.set_yticks(range(0, 61, 10))
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        structure_save(fig, output_dir, "Figure2_recommendation_performance")
    caption = ("Fig. 2. Ingredient recommendation performance in herbal formulas and matched food samples. Each ingredient in every composition was withheld once, "
               "with all remaining ingredients provided as input. Hit@10 was averaged first within each composition and then equally across all 2,009 compositions. "
               "Black bars show herbal performance; gray bars show mean performance across 100 matched food samples. "
               "Food error bars represent empirical 2.5th–97.5th percentiles across samples, not confidence intervals.")
    (output_dir / "Figure2_caption.txt").write_text(caption + "\n", encoding="utf-8")
    for path in output_dir.glob("Figure*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


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
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.hashsalt": "HerbalFormulaCompletion",
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = 0.38
    ax.bar(
        [size + width / 2 for size in sizes], food_percent, width,
        color="#bdbdbd", edgecolor="#555555", linewidth=0.5,
        label="Food",
    )
    ax.bar(
        [size - width / 2 for size in sizes], herbal_percent, width,
        color="#222222", edgecolor="#222222", linewidth=0.5,
        label="Herbal",
    )
    ax.set_xlabel("Number of ingredients")
    ax.set_ylabel("Compositions (%)")
    ax.set_xticks(sizes)
    ax.legend(*[list(reversed(items)) for items in ax.get_legend_handles_labels()], frameon=False)

    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    fig.tight_layout()
    structure_save(fig, output, "Figure1_dataset_matching")
    print("Saved: figures/Figure1_dataset_matching (EPS, TIFF, PDF, SVG, PNG)")



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




# Section 2.5: fixed-margin randomization, after Strona et al. (2014),
# doi:10.1038/ncomms5114. Binary unique compositions, NOT source-weighted margins.
# Its finite Curveball chains are checked empirically; exact independent sampling
# is not claimed. Ahn et al. (2011), doi:10.1038/srep00196, motivates a
# frequency-controlled food comparison, not this exact co-occurrence protocol.

STRUCTURE_MIN_OCCURRENCES = 10




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












# Display names apply only to manuscript ingredients; statistical identifiers remain unchanged.
HERB_DISPLAY_NAMES = {row['source_name']: row['display_name']
                      for row in read_rows(ROOT / 'data/herbal_display_names.csv')}


def ingredient_display_name(name, wrap=False):
    label = HERB_DISPLAY_NAMES.get(name, name.replace('_', ' '))
    if wrap:
        import textwrap
        return '\n'.join(textwrap.wrap(label, width=19, break_long_words=False, break_on_hyphens=False))
    return label


def structure_save(fig, out, stem):
    """IMR submission: embedded-font EPS and 1000-dpi TIFF; other files are previews."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        for ext in ('eps', 'tiff', 'pdf', 'svg', 'png'):
            metadata = {'Date': None} if ext == 'svg' else ({'CreationDate': None, 'ModDate': None} if ext == 'pdf' else None)
            options = {'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tiff' else {}
            fig.savefig(out / f'{stem}.{ext}', dpi=1000 if ext == 'tiff' else 300,
                        bbox_inches='tight', pad_inches=.06, facecolor='white', metadata=metadata, **options)
    plt.close(fig)
    path = out / f'{stem}.svg'
    path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines()) + '\n')








# Predictive control: keep the observed test problems fixed, randomize training only.
@njit(cache=True)
def predictive_pair_counts(matrix, lengths, k):
    p = np.zeros((k, k), np.float64)
    for r in range(len(lengths)):
        for i in range(lengths[r]):
            a = matrix[r, i]
            for j in range(i + 1, lengths[r]):
                b = matrix[r, j]
                p[a, b] += 1
                p[b, a] += 1
    return p


@njit(cache=True)
def predictive_nminus1_hits(test, lengths, counts, pairs):
    """Exact Top-10 target membership; same arithmetic/order/ties as evaluate_cohort."""
    k = len(counts)
    conditional = pairs / counts.reshape((k, 1))
    jaccard = pairs / (counts.reshape((k, 1)) + counts.reshape((1, k)) - pairs)
    hits = np.zeros((int(lengths.sum()), 3), np.uint8)
    case = 0
    for r in range(len(lengths)):
        n = lengths[r]
        for omitted in range(n - 1, -1, -1):
            target = test[r, omitted]
            if target >= 0:
                excluded = np.zeros(k, np.uint8)
                conditional_scores = np.zeros(k)
                jaccard_scores = np.zeros(k)
                for j in range(n):
                    if j == omitted: continue
                    observed = test[r, j]
                    if observed >= 0:
                        excluded[observed] = 1
                        for v in range(k):
                            conditional_scores[v] += conditional[observed, v]
                            jaccard_scores[v] += jaccard[observed, v]
                # Division is retained to reproduce existing floating-point ranking exactly.
                conditional_scores /= n - 1
                jaccard_scores /= n - 1
                ranks = np.ones(3, np.int64)
                for v in range(k):
                    if excluded[v] or v == target: continue
                    if counts[v] > counts[target] or (counts[v] == counts[target] and v < target): ranks[0] += 1
                    if conditional_scores[v] > conditional_scores[target] or (conditional_scores[v] == conditional_scores[target] and v < target): ranks[1] += 1
                    if jaccard_scores[v] > jaccard_scores[target] or (jaccard_scores[v] == jaccard_scores[target] and v < target): ranks[2] += 1
                for m in range(3): hits[case, m] = ranks[m] <= 10
            case += 1
    return hits


@njit(cache=True)
def predictive_shuffle(original, lengths, seed, trades_per_row):
    np.random.seed(seed)
    shuffled = original.copy()
    for _ in range(trades_per_row * len(lengths)):
        curveball_trade(shuffled, lengths)
    return shuffled


def relationship_pair_candidates(rows, field, pair_keys=None):
    """Display support is unique compositions; all reported counts retain source weights."""
    unique_support = Counter()
    weighted_support = Counter()
    for row in rows:
        pairs = list(combinations(sorted(row[field].split('|')), 2))
        unique_support.update(pairs)
        weighted_support.update({pair: int(row['weight']) for pair in pairs})
    keys = list(pair_keys) if pair_keys is not None else sorted(pair for pair, count in unique_support.items() if count >= 10)
    support = [(unique_support[pair], weighted_support[pair]) for pair in keys]
    return keys, support


def food_relationship_candidates(cohorts):
    counts=Counter()
    for rows in cohorts.values():
        for r in rows: counts.update(combinations(sorted(r['ingredients'].split('|')),2))
    # Same support density as herbal: at least ten unique compositions per sample on average.
    return sorted(pair for pair,n in counts.items() if n >= 10*len(cohorts))


def pooled_food_pairs(root):
    groups=defaultdict(list)
    for rep in range(1,101):
        for r in read_training_pairs(root/f'cohorts/food_{rep:03d}/training_pairs.csv'):
            groups[r['ingredient_a'],r['ingredient_b']].append(r)
    result=[]
    for (a,b),rows in sorted(groups.items()):
        if len(rows)!=100:raise ValueError('Pair counts must cover all 100 samples; rerun the pipeline in a fresh work directory')
        before=sum(r['observed_training_mean'] for r in rows)/100
        after=sum(r['randomized_training_mean'] for r in rows)/100
        result.append({'ingredient_a':a,'ingredient_b':b,
            'unique_composition_support':sum(r['unique_composition_support'] for r in rows),
            'source_record_support':sum(r['source_record_support'] for r in rows),
            'observed_training_mean':before,'randomized_training_mean':after,
            'ratio':before/after if after else None,
            'observed_fold_sum':sum(r['observed_fold_sum'] for r in rows),
            'randomized_fold_mean_sum':sum(r['randomized_fold_mean_sum'] for r in rows),
            'positive_excess':before>after})
    return result


def save_training_pair_results(output, keys, support, observed, randomized, audits):
    # Each original composition is in four training folds. This is an accounting
    # check, not four independent observations of that composition.
    weights = np.array([w for u, w in support], dtype=np.int64)
    if not np.array_equal(observed.sum(axis=0), 4 * weights):
        raise AssertionError('Original fold pair counts disagree with source weights')
    seeds = np.array([r['seed'] for r in audits], dtype=np.int64).reshape(randomized.shape[:2])
    np.savez_compressed(output/'training_pair_counts.npz',
        ingredient_a=np.array([a for a,b in keys]), ingredient_b=np.array([b for a,b in keys]),
        unique_support=np.array([u for u,w in support]), source_support=weights,
        observed=observed, randomized=randomized, seeds=seeds)
    rows=[]
    for j,(a,b) in enumerate(keys):
        before=float(observed[:,j].mean())
        after=float(randomized[:,:,j].mean())
        rows.append({'ingredient_a':a,'ingredient_b':b,
            'unique_composition_support':support[j][0], 'source_record_support':support[j][1],
            'observed_training_mean':before,'randomized_training_mean':after,
            'ratio':before/after if after>0 else '',
            'observed_fold_sum':int(observed[:,j].sum()),
            'randomized_fold_mean_sum':float(randomized[:,:,j].mean(axis=1).sum()),
            'positive_excess':before>after})
    write_csv(output/'training_pairs.csv',
        ('ingredient_a','ingredient_b','unique_composition_support','source_record_support',
         'observed_training_mean','randomized_training_mean','ratio','observed_fold_sum',
         'randomized_fold_mean_sum','positive_excess'),rows)


def read_training_pairs(path):
    rows=read_rows(path)
    for r in rows:
        for field in ('unique_composition_support','source_record_support','observed_fold_sum'):
            r[field]=int(r[field])
        for field in ('observed_training_mean','randomized_training_mean','randomized_fold_mean_sum'):
            r[field]=float(r[field])
        r['ratio']=float(r['ratio']) if r['ratio'] else None
        r['positive_excess']=r['positive_excess']=='True'
    return rows


def make_frequency_figure(work_dir, figures):
    work_dir,figures=Path(work_dir),Path(figures)
    out=work_dir/'frequency';out.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    frequencies=[]
    for domain,field in [('herbal','herbs'),('food','ingredients')]:
        counts=Counter();total=0
        with (work_dir/domain/'unique_compositions.csv').open(encoding='utf-8-sig',newline='') as h:
            for row in csv.DictReader(h):
                weight=int(row['weight']);total+=weight
                counts.update({x:weight for x in row[field].split('|')})
        frequencies.extend({'domain':domain,'rank':i,'ingredient':x,'source_count':counts[x],
            'source_records':total,'prevalence':counts[x]/total}
            for i,x in enumerate(sorted(counts,key=lambda x:(-counts[x],x)),1))
    write_csv(out/'ingredient_frequencies.csv',tuple(frequencies[0]),frequencies)
    with mpl.rc_context({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
            'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
            'svg.fonttype':'none','svg.hashsalt':'HFC-unified'}):
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),sharey=True)
        for ax,domain,title in zip(axes,('herbal','food'),('(A) Herbal','(B) Food')):
            rows=[r for r in frequencies if r['domain']==domain]
            ax.plot([r['rank'] for r in rows],[100*r['prevalence'] for r in rows],color='#333333',lw=1.2)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(1,2000); ax.set_ylim(.0001,100)
            ax.set_xticks([1,10,100,1000], ['1','10','100','1000'])
            ax.set_yticks([.0001,.001,.01,.1,1,10,100], ['0.0001','0.001','0.01','0.1','1','10','100'])
            ax.minorticks_off()
            ax.set_xlabel('Ingredient frequency rank'); ax.set_title(title,loc='left')
        axes[0].set_ylabel('Source records (%)')
        fig.tight_layout();structure_save(fig,figures,'Figure3_ingredient_frequency')
    caption3=('Fig. 3. Ingredient rank–frequency distributions in herbal formulas and food recipes. (A) Herbal formulas. (B) Food recipes. Both axes use logarithmic scales. '
        'Each domain is shown by one line retaining source-record multiplicities. '
        'The denominator is 2,992 herbal source records or 921,927 food source records, respectively. '
        'The full food source pool is used here, not the matched samples.')
    (figures/'Figure3_caption.txt').write_text(caption3+'\n')
    print('Frequency distribution complete',flush=True)


def predictive_null_code_hash():
    import inspect
    import numba
    functions = (curveball_trade.py_func, predictive_pair_counts.py_func,
                 predictive_nminus1_hits.py_func, predictive_shuffle.py_func,
                 predictive_null_cohort, assign_composition_folds, structure_prepare,
                 relationship_pair_candidates, food_relationship_candidates, save_training_pair_results)
    return hashlib.sha256((''.join(inspect.getsource(f) for f in functions)
                           + np.__version__ + numba.__version__).encode()).hexdigest()


def predictive_null_cohort(task):
    domain, replicate, rows, field, output, draws, trades_per_row, seed, baseline_dir, pair_keys = task
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = {'pair_keys_sha256': hashlib.sha256(json.dumps(pair_keys, ensure_ascii=False).encode()).hexdigest(),
                 'domain': domain, 'replicate': replicate, 'draws': draws,
                 'trades_per_row': trades_per_row, 'seed': seed,
                 'algorithm_sha256': predictive_null_code_hash(),
                 'input_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}
    meta_path = output / 'metadata.json'
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get('status') == 'complete' and all(old.get(k) == v for k, v in signature.items()) and all((output / x).exists() for x in ('draw_results.csv', 'case_hits.npz', 'ingredient_results.csv', 'case_results.csv.gz', 'training_pair_counts.npz', 'training_pairs.csv')):
            return read_rows(output / 'draw_results.csv')
    meta_path.write_text(json.dumps(dict(signature, status='running'), indent=2))
    folds = assign_composition_folds(rows, ingredient_field=field)
    cases, composition_results, audits, fold_assignments = [], [], [], []
    observed_parts, null_parts = [], []
    existing = {}
    baseline_path = Path(baseline_dir) / 'composition_results.csv.gz'
    if not baseline_path.exists():
        raise FileNotFoundError('Run --evaluate-only first; the control must verify the original N-1 results.')
    with gzip.open(baseline_path, 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['condition'] == 'N-1': existing[r['composition_id'], r['method']] = float(r['performance'])
    pair_keys, pair_support = relationship_pair_candidates(rows, field, pair_keys)
    original_pairs = np.zeros((5, len(pair_keys)), dtype=np.int64)
    shuffled_pairs = np.zeros((5, draws, len(pair_keys)), dtype=np.int64)
    baseline_checks = 0
    for fold_index, test_rows in enumerate(folds):
        train = [r for j, fold in enumerate(folds) if j != fold_index for r in fold]
        vocab, original, lengths, frequency, *_ = structure_prepare(train, field, weighted=True)
        index = {v: i for i, v in enumerate(vocab)}
        counts = frequency.astype(np.float64)
        test = np.full((len(test_rows), 19), -1, np.int64)
        test_lengths = np.array([len(r[field].split('|')) for r in test_rows], np.int64)
        original_sets = {tuple(sorted(r[field].split('|'))) for r in train}
        for j, r in enumerate(test_rows):
            items = sorted(r[field].split('|'))
            assert tuple(items) not in original_sets
            test[j, :len(items)] = [index.get(x, -1) for x in items]
            fold_assignments.append({'composition_id': r['composition_id'], 'test_fold': fold_index + 1})
            for target in reversed(items):
                cases.append({'fold': fold_index + 1, 'composition_id': r['composition_id'],
                              'ingredient_count': len(items), 'target': target,
                              'input': '|'.join(x for x in items if x != target),
                              'training_source_count': int(frequency[index[target]]) if target in index else 0,
                              'training_source_rows': len(original)})
        p = predictive_pair_counts(original, lengths, len(vocab))
        pair_a = np.array([index.get(a, -1) for a, b in pair_keys], dtype=np.int64)
        pair_b = np.array([index.get(b, -1) for a, b in pair_keys], dtype=np.int64)
        present = (pair_a >= 0) & (pair_b >= 0)
        original_pairs[fold_index, present] = p[pair_a[present], pair_b[present]]
        observed = predictive_nminus1_hits(test, test_lengths, counts, p)
        null = np.empty((draws, len(observed), 3), np.uint8)
        # A shuffled row can match a test composition by chance; record but do not reject it.
        # Rejection would condition the null on the held-out answers.
        test_keys = {tuple(sorted(vocab[x] for x in row[:n]))
                     for row, n in zip(test, test_lengths) if np.all(row[:n] >= 0)}
        for draw in range(draws):
            draw_seed = seed + replicate * 100000 + fold_index * 1000 + draw
            shuffled = predictive_shuffle(original, lengths, draw_seed, trades_per_row)
            if not np.array_equal(np.bincount(shuffled[shuffled >= 0], minlength=len(vocab)), frequency):
                raise AssertionError('Weighted ingredient frequencies changed')
            if not np.array_equal((shuffled >= 0).sum(axis=1), lengths):
                raise AssertionError('Source-record lengths changed')
            accidental = 0
            overlap = 0
            for row, before, n in zip(shuffled, original, lengths):
                if len(set(row[:n])) != n: raise AssertionError('Duplicate ingredient in shuffled row')
                overlap += len(set(row[:n]) & set(before[:n]))
                accidental += tuple(sorted(vocab[x] for x in row[:n])) in test_keys
            shuffled_counts = predictive_pair_counts(shuffled, lengths, len(vocab))
            shuffled_pairs[fold_index, draw, present] = shuffled_counts[pair_a[present], pair_b[present]]
            null[draw] = predictive_nminus1_hits(test, test_lengths, counts, shuffled_counts)
            if not np.array_equal(observed[:, 0], null[draw, :, 0]):
                raise AssertionError('Popularity changed despite preserved frequencies')
            audits.append({'fold': fold_index + 1, 'draw': draw + 1, 'seed': draw_seed,
                           'training_source_rows': len(original), 'test_unique': len(test_rows),
                           'margin_checks_passed': True, 'popularity_identical': True,
                           'source_slot_overlap': overlap / int(lengths.sum()),
                           'chance_test_matches_in_randomized_rows': accidental})
        offset = 0
        for row, n in zip(test_rows, test_lengths):
            for m, method in enumerate(EVALUATION_MODELS):
                value = float(observed[offset:offset + n, m].mean())
                if not math.isclose(value, existing[row['composition_id'], method], abs_tol=1e-14):
                    raise AssertionError('Observed baseline differs from existing evaluation')
                baseline_checks += 1
                composition_results.append({'fold': fold_index + 1, 'composition_id': row['composition_id'],
                    'method': method, 'observed': value,
                    'randomized_mean': float(null[:, offset:offset+n, m].mean()),
                    'difference': value - float(null[:, offset:offset+n, m].mean())})
            offset += n
        observed_parts.append(observed)
        null_parts.append(null)
    observed = np.concatenate(observed_parts)
    randomized = np.concatenate(null_parts, axis=1)
    # Every composition contributes equally, independently of its ingredient count.
    case_weights = np.array([1 / int(r['ingredient_count']) / len(rows) for r in cases])
    observed_scores = (observed * case_weights[:, None]).sum(axis=0)
    null_scores = (randomized * case_weights[None, :, None]).sum(axis=1)
    results = []
    for d in range(draws):
        for m, method in enumerate(EVALUATION_MODELS):
            results.append({'domain': domain, 'replicate': replicate, 'draw': d+1, 'method': method,
                            'observed': float(observed_scores[m]), 'randomized': float(null_scores[d, m]),
                            'difference': float(observed_scores[m] - null_scores[d, m])})
    np.savez_compressed(output / 'case_hits.npz', observed=observed, randomized=randomized,
                        composition_id=np.array([r['composition_id'] for r in cases]),
                        target=np.array([r['target'] for r in cases]), case_weights=case_weights)
    records, ingredient_accum = [], defaultdict(list)
    random_mean = randomized.mean(axis=0)
    for j, case in enumerate(cases):
        for m, method in enumerate(EVALUATION_MODELS):
            item = dict(case, method=method, observed_hit=int(observed[j,m]),
                        randomized_hit_rate=float(random_mean[j,m]),
                        difference=float(observed[j,m]-random_mean[j,m]))
            records.append(item)
            ingredient_accum[case['target'], method].append(item)
    with gzip.open(output / 'case_results.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=tuple(records[0])); writer.writeheader(); writer.writerows(records)
    ingredients = []
    for (target, method), values in sorted(ingredient_accum.items()):
        ingredients.append({'ingredient': target, 'method': method, 'test_cases': len(values),
            'observed_hit_rate': sum(r['observed_hit'] for r in values)/len(values),
            'randomized_hit_rate': sum(r['randomized_hit_rate'] for r in values)/len(values),
            'difference': sum(r['difference'] for r in values)/len(values),
            'mean_training_source_count': sum(r['training_source_count'] for r in values)/len(values)})
    for filename, values in [('draw_results.csv', results), ('fold_audit.csv', audits),
                             ('composition_results.csv', composition_results),
                             ('ingredient_results.csv', ingredients), ('fold_assignments.csv', fold_assignments)]:
        write_csv(output / filename, tuple(values[0]), values)
    save_training_pair_results(output, pair_keys, pair_support, original_pairs, shuffled_pairs, audits)
    metadata = dict(signature, status='complete', compositions=len(rows), cases=len(cases),
                    shared_pair_counts=True, relationship_pairs=len(pair_keys),
                    baseline_composition_checks=baseline_checks, all_draw_margin_checks_passed=True,
                    popularity_identical_in_every_case=True,
                    training='Source multiplicities expanded after fold assignment; each source row has weight 1.',
                    test='Observed test compositions and all N-1 problems remain fixed.',
                    source_slot_overlap_mean=float(np.mean([r['source_slot_overlap'] for r in audits])),
                    chance_test_matches_total=sum(r['chance_test_matches_in_randomized_rows'] for r in audits))
    meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return results


def make_predictive_null_figure(output, figures):
    output, figures = Path(output), Path(figures)
    summary = read_rows(output / 'summary.csv')
    figures.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 9, 'axes.linewidth': .8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'HerbalFormulaCompletion'}):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
        for ax, domain, title in zip(axes, ('herbal', 'food'), ('(A) Herbal', '(B) Food')):
            selected = [next(r for r in summary if r['domain']==domain and r['method']==m) for m in EVALUATION_MODELS]
            original = [100*float(r['observed']) for r in selected]
            shuffled = [100*float(r['randomized_mean']) for r in selected]
            x = np.arange(3); width=.32
            ax.bar(x-width/2, original, width, color='#222222', label='Original')
            ax.bar(x+width/2, shuffled, width, color='#bdbdbd', edgecolor='#555555', linewidth=.5,
                   label='Randomized')
            ax.set_title(title, loc='left')
            ax.set_xticks(x, ['Popularity','Mean conditional\nprobability','Mean pairwise\nJaccard'], fontsize=7.5)
            ax.set_ylim(0,65);ax.set_yticks(np.arange(0,61,10))
        axes[0].set_ylabel('Hit@10 (%)')
        axes[0].legend(frameon=False, loc='upper left', fontsize=8)
        fig.tight_layout()
        structure_save(fig, figures, 'Figure4_training_randomization')
    draws = int(json.loads((output/'metadata.json').read_text())['draws'])
    caption = ('Fig. 4. Recommendation performance with original and randomized training data. '
        '(A) Herbal formulas. (B) Matched food samples. Observed test compositions, five-fold assignments, and all leave-one-ingredient-out problems were held fixed. '
        'Within each training fold, source-record multiplicities were expanded before Curveball randomization, '
        'preserving each source-record length and the weighted occurrence count of every ingredient. '
        f'Gray bars average {draws} independently seeded finite randomization runs per cohort; '
        'food bars additionally average the 100 matched food samples. '
        'Performance was averaged within each composition and then equally across compositions. '
        'Popularity predictions were identical in every original and randomized test case. '
        'This is a descriptive training-data ablation; the repeated runs are not independent datasets or a confidence interval. '
        'All run-level values and finite-randomization sensitivity results are provided with the code.')
    (figures/'Figure4_caption.txt').write_text(caption+'\n')




def evaluate_predictive_null(work_dir, workers=2, draws=20, trades_per_row=50):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    work_dir = Path(work_dir); out = work_dir/'predictive_null'; out.mkdir(parents=True, exist_ok=True)
    herbs = read_rows(work_dir/'herbal/unique_compositions.csv')
    members = read_rows(work_dir/'matching/membership.csv')
    wanted = {r['composition_id'] for r in members}
    with (work_dir/'food/unique_compositions.csv').open(encoding='utf-8-sig',newline='') as f:
        food = {r['composition_id']: r for r in csv.DictReader(f) if r['composition_id'] in wanted}
    cohorts = defaultdict(list)
    for r in members: cohorts[int(r['replicate'])].append(food[r['composition_id']])
    assert set(cohorts)==set(range(1,101))
    hashes = {}
    for name in ('herbal/unique_compositions.csv','food/unique_compositions.csv','matching/membership.csv'):
        with (work_dir/name).open('rb') as f: hashes[name]=hashlib.file_digest(f,'sha256').hexdigest()
    metadata = {'status':'running','draws':draws,'trades_per_row':trades_per_row,'seed':20261001,
                'fold_seed':20260812,'input_hashes':hashes,'algorithm_sha256':predictive_null_code_hash(),
                'scope':'Descriptive training-only randomization ablation, not a clinical validation or a causal test.',
                'weighted_margin':'Expand training source multiplicities before shuffling; source-row lengths and weighted ingredient counts fixed.',
                'held_fixed':'Original folds, observed test compositions, training vocabulary, weighted ingredient counts, N-1 contexts, Top-10 tie rule.',
                'changed':'Training co-occurrences and source composition identities; unique-row count need not be preserved.',
                'duplicates':'Randomized duplicate rows retained; chance matches with fixed test compositions audited, not rejected.',
                'aggregation':'Mean within composition, then equal composition weight. Food summaries first average draws within each sample.',
                'randomization_uncertainty':'Independent seeds with a fresh start from observed training data and finite trades; no claim of exact uniform independent draws.',
                'inference':f'{draws} runs are exploratory stability repeats, not biological replicates. No Monte Carlo p-values or causal attribution are reported.'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    global_food_keys=food_relationship_candidates(cohorts)
    def task(domain,rep,rows,field,folder,trades):
        return (domain,rep,rows,field,folder,draws,trades,20261001,work_dir/f'evaluation/cohorts/{domain}_{rep:03d}',global_food_keys if domain=='food' else None)
    print('Training-only randomization: herbal cohort',flush=True)
    results = predictive_null_cohort(task('herbal',0,herbs,'herbs',out/'cohorts/herbal_000',trades_per_row))
    tasks = [task('food',rep,rows,'ingredients',out/f'cohorts/food_{rep:03d}',trades_per_row) for rep,rows in sorted(cohorts.items())]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(predictive_null_cohort,t) for t in tasks]
        for done,future in enumerate(as_completed(futures),1):
            results.extend(future.result())
            if done==1 or done%10==0: print(f'Training-only randomized food cohorts: {done}/100',flush=True)
    print('Longer-randomization sensitivity: herbal and food sample 1',flush=True)
    sensitivity=[]
    for domain,rep,rows,field in [('herbal',0,herbs,'herbs'),('food',1,cohorts[1],'ingredients')]:
        sensitivity.extend(predictive_null_cohort(task(domain,rep,rows,field,out/f'sensitivity/{domain}_{rep:03d}',trades_per_row*2)))
    results.sort(key=lambda r:(r['domain'],int(r['replicate']),int(r['draw']),r['method']))
    grouped=defaultdict(list)
    for r in results: grouped[r['domain'],int(r['replicate']),r['method']].append(r)
    sample=[]
    for (domain,rep,method),values in sorted(grouped.items()):
        a=np.array([float(r['randomized']) for r in values]);observed=float(values[0]['observed'])
        sample.append({'domain':domain,'replicate':rep,'method':method,'observed':observed,
                       'randomized_mean':float(a.mean()),'randomized_sd':float(a.std(ddof=1)),
                       'randomized_min':float(a.min()),'randomized_max':float(a.max()),
                       'difference':observed-float(a.mean())})
    summary=[]
    for domain in ('herbal','food'):
        for method in EVALUATION_MODELS:
            values=[r for r in sample if r['domain']==domain and r['method']==method]
            summary.append({'domain':domain,'method':method,'cohorts':len(values),'draws_per_cohort':draws,
                'observed':float(np.mean([r['observed'] for r in values])),
                'randomized_mean':float(np.mean([r['randomized_mean'] for r in values])),
                'difference':float(np.mean([r['difference'] for r in values])),
                'sample_difference_p2_5':float(np.percentile([r['difference'] for r in values],2.5)),
                'sample_difference_p97_5':float(np.percentile([r['difference'] for r in values],97.5))})
    stability=[]
    for domain,rep in [('herbal',0),('food',1)]:
        for method in EVALUATION_MODELS:
            short=next(r for r in sample if r['domain']==domain and r['replicate']==rep and r['method']==method)
            long=np.array([float(r['randomized']) for r in sensitivity if r['domain']==domain and r['method']==method])
            stability.append({'domain':domain,'replicate':rep,'method':method,'primary_trades_per_row':trades_per_row,
                              'longer_trades_per_row':2*trades_per_row,'primary_randomized_mean':short['randomized_mean'],
                              'longer_randomized_mean':float(long.mean()),'longer_minus_primary':float(long.mean()-short['randomized_mean'])})
    for name,values in [('draw_results.csv',results),('sample_results.csv',sample),('summary.csv',summary),('sensitivity_summary.csv',stability)]:
        write_csv(out/name,tuple(values[0]),values)
    metadata.update(status='complete',primary_cohorts=101,sensitivity_cohorts=2,
                    randomized_training_matrices=103*5*draws,
                    baseline_composition_checks=103*len(herbs)*3)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    make_predictive_null_figure(out,ROOT/'figures')
    print('Complete: work/predictive_null and Figure4_training_randomization',flush=True)
    return summary


def relationship_positions(edges, wide=False):
    """Pack displayed connected components for legibility; no community inference."""
    import networkx as nx
    graph=nx.Graph()
    for r in edges: graph.add_edge(r['ingredient_a'],r['ingredient_b'])
    components=sorted(nx.connected_components(graph),key=lambda c:(-len(c),sorted(c)))
    if not wide:
        # Explicit component geometry separates labels from every herb-pair edge.
        layouts=[
            {'석곡':(0,1),'육종용':(1,1),'녹용':(0,0),'토사자':(1,2),'파극천':(2,1)},
            {'가자':(0,0),'육두구':(1,0),'보골지':(1,2),'회향':(2,2)},
            {'검인':(0,0),'연자육':(1,1),'백편두':(0,2),'의이인':(2,2)},
            {'금은화':(0,0),'연교':(1,1),'우방자':(2,0),'죽엽':(1,2)},
            {'백자인':(0,0),'복신':(0,2),'산조인':(2,0),'원지':(2,2)},
            {'구맥':(0,1),'목통':(1,0),'차전자':(2,1)},
            {'귀판':(0,0),'우슬':(1,1),'두중':(2,2)},
            {'당귀미':(0,0),'홍화':(1,1),'도인':(2,2)},
            {'산사':(0,2),'산사육':(0,0),'맥아':(2,1)},
        ]
        positions={};heights=[0.,0.,0.];pairs=[]
        for component in components:
            if len(component)==2:
                pairs.append(sorted(component));continue
            col=int(np.argmin(heights));names=sorted(component)
            local=next((v for v in layouts if set(v)==component),None)
            if local is None:local={names[0]:(0,1),names[1]:(2,1)}
            height=3.4 if len(component)>2 else 1.55
            for name,(x,y) in local.items():
                positions[name]=np.array([col*5.2+.75+1.65*x,-heights[col]-1.0-(2-y)*.95])
            heights[col]+=height+.65
        base=max(heights)
        for i,names in enumerate(pairs):
            col=i%2;row=i//2
            for j,name in enumerate(names):
                positions[name]=np.array([.75+col*7.8+j*4.7,-base-1.-row*2.1])
        return graph,positions,base+1.9+2.1*((len(pairs)-1)//2)
    positions={};heights=[0.,0.]
    for component in components:
        names=sorted(component);col=int(heights[1]<heights[0]);n=len(names)
        full_width=wide and n>=7
        if full_width:heights=[max(heights)]*2;col=0
        height=3.5 if full_width else (.85 if n==2 else max(1.5,.55*math.sqrt(n)+.7))
        if n==2:
            local={names[0]:np.array([-.9,0]),names[1]:np.array([.9,0])}
        else:
            sub=nx.Graph();sub.add_nodes_from(names)
            sub.add_edges_from(sorted((min(a,b),max(a,b)) for a,b in graph.edges if a in component and b in component))
            local=nx.spring_layout(sub,seed=20260928,weight=None,k=1.4/math.sqrt(n),iterations=1000)
            for name in names:local[name]=np.array([float(local[name][0]),float(local[name][1])])
        if full_width:
            # Fill both dimensions of the allotted space instead of retaining a narrow spring layout.
            xy=np.array([local[x] for x in names]);low=xy.min(axis=0);span=np.maximum(xy.max(axis=0)-low,1e-8)
            local={x:2*(local[x]-low)/span-1 for x in names}
            # Presentation-only anchors separate the two spice triangles and their branches.
            anchors={'turmeric':(-.15,.45),'coriander':(.7,.15),'cumin':(-.15,-.2),'chili':(.85,-.55),
                     'ginger':(-.55,.65),'soy_sauce':(-.85,.85),'cornstarch':(-1,1),
                     'cilantro':(-.45,-.6),'lime':(-1,-.8),'avocado':(-.7,-1),'jalapeno':(.25,-1)}
            if {'turmeric','coriander','cumin','chili'} <= component:
                for name in names:
                    if name in anchors:local[name]=np.array(anchors[name])
        # Fixed display anchors keep short edges visible around long labels.
        if {'baking_soda','baking_powder','buttermilk'} <= component:
            anchors={'baking_soda':(-.25,.1),'baking_powder':(.25,.8),'buttermilk':(-.65,.85),
                     'oats':(-1,.25),'cocoa':(-.75,-.3),'extract':(-1,-.8),
                     'applesauce':(.2,-.45),'cinnamon':(.65,-.1),'nutmeg':(1,.5),
                     'raisins':(1,-.45),'allspice':(.5,-1)}
            local.update({x:np.array(v) for x,v in anchors.items() if x in component})
        if {'산사','산사육','맥아'} == component:
            local={'산사':np.array([-.8,.85]),'맥아':np.array([.6,0]),'산사육':np.array([-.8,-.85])}
        if {'ketchup','mustard','worcestershire_sauce'} == component:
            local={'mustard':np.array([-.85,.8]),'ketchup':np.array([.8,0]),
                   'worcestershire_sauce':np.array([-.85,-.8])}
        for name,xy in local.items():
            center=3.1 if full_width else col*3.2+1.45
            radius=2.45 if full_width else 1.05
            positions[name]=np.array([center+radius*xy[0],-(heights[col]+height/2)+.35*height*xy[1]])
        # Repel overlapping label rectangles within each displayed component.
        half_width={x:.045*max(len(part) for part in ingredient_display_name(x, wrap=True).split('\n'))+.10 for x in names}
        half_height={x:.20+.14*ingredient_display_name(x, wrap=True).count('\n') for x in names}
        for _ in range(300):
            moved=False
            for i,a in enumerate(names):
                for b in names[i+1:]:
                    delta=positions[b]-positions[a]
                    overlap_x=half_width[a]+half_width[b]-abs(delta[0])
                    overlap_y=half_height[a]+half_height[b]-abs(delta[1])
                    if overlap_x>0 and overlap_y>0:
                        axis=0 if overlap_x<overlap_y else 1
                        shift=.51*(overlap_x if axis==0 else overlap_y)
                        sign=1 if delta[axis]>=0 else -1
                        positions[a][axis]-=shift*sign;positions[b][axis]+=shift*sign;moved=True
            if not moved:break
        heights[col]+=height+.35
        if full_width:heights[1]=heights[0]
    return graph,positions,max(heights)


def draw_relationship_map(displayed, cohorts, figures):
    """Draw existing pair scores without recalculating the analysis."""
    import networkx as nx
    positions=[];max_ratio=max(r['ratio'] for rows in displayed.values() for r in rows)
    with mpl.rc_context({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
            'font.size':8,'svg.fonttype':'none','svg.hashsalt':'HFC-ratio-atlas'}):
        fig,axes=plt.subplots(1,2,figsize=(10.4,7.2), gridspec_kw={'width_ratios':[1.55,1]})
        panel_geometry=[]
        for ax,domain,title in zip(axes,('herbal','food'),('(A) Herbal','(B) Food')):
            rows=displayed[domain];graph,pos,height=relationship_positions(rows, wide=domain=='food')
            freq=Counter(x for r in cohorts[domain] for x in r['herbs' if domain=='herbal' else 'ingredients'].split('|'))
            ratio_lookup={frozenset((r['ingredient_a'],r['ingredient_b'])):r['ratio'] for r in rows}
            nx.draw_networkx_edges(graph,pos,ax=ax,edge_color='#686868',width=[.5+3*ratio_lookup[frozenset((a,b))]/max_ratio for a,b in graph.edges])
            nx.draw_networkx_nodes(graph,pos,ax=ax,node_size=25,node_color='white',edgecolors='#444444',linewidths=.7)
            import textwrap
            labels={x:'\n'.join(textwrap.wrap(ingredient_display_name(x), width=14 if domain=='herbal' else 19,
                         break_long_words=False, break_on_hyphens=False)) for x in graph}
            if domain=='food':labels['baking_soda']='baking\nsoda'
            texts={name:ax.annotate(labels[name],pos[name],xytext=(0,0),
                      textcoords='offset points',ha='center',va='center',fontsize=8,
                      annotation_clip=False) for name in graph}
            ax.set_xlim(-.65,15.9) if domain=='herbal' else ax.set_xlim(-.9,7.15)
            ax.set_ylim(-height-.2,.8);ax.axis('off');ax.set_title(title,loc='left',fontsize=11)
            panel_geometry.append((ax,graph,pos,texts,ratio_lookup))
            positions.extend({'domain':domain,'ingredient':x,'x':float(v[0]),'y':float(v[1]),'unique_occurrences':freq[x]} for x,v in sorted(pos.items()))
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([0],[0],color='#686868',lw=.5+3*r/max_ratio,label=f'{r:g}×') for r in (5,20,40)],
                   title='Co-occurrence ratio',loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.005),fontsize=8,title_fontsize=8)
        fig.subplots_adjust(left=.015,right=.985,top=.925,bottom=.11,wspace=.07)
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        from matplotlib.transforms import Bbox
        def line_hits_box(a,b,box):
            lo,hi=0.,1.
            for k,lower,upper in ((0,box.x0,box.x1),(1,box.y0,box.y1)):
                delta=b[k]-a[k]
                if abs(delta)<1e-12:
                    if not lower<=a[k]<=upper:return False
                else:
                    u,v=(lower-a[k])/delta,(upper-a[k])/delta
                    lo=max(lo,min(u,v));hi=min(hi,max(u,v))
                    if lo>hi:return False
            return True
        audit=[]
        for ax,graph,pos,texts,ratios in panel_geometry:
            points={n:ax.transData.transform(pos[n]) for n in graph}
            segments=[(points[a],points[b],(.5+3*ratios[frozenset((a,b))]/max_ratio)*fig.dpi/144)
                      for a,b in graph.edges]
            node_radius=4.2*fig.dpi/72
            placed=[]
            # Place constrained hub labels first; each candidate must clear every edge and node.
            order=sorted(graph,key=lambda n:(-graph.degree[n],-len(texts[n].get_text()),n))
            for name in order:
                text=texts[name];box=text.get_window_extent(renderer)
                width,height=box.width+3,box.height+3
                origin=points[name];candidates=[]
                for gap in (7,11,16,23,32,44,58,75):
                    for angle in np.arange(0,360,15):
                        rad=math.radians(angle);ux,uy=math.cos(rad),math.sin(rad)
                        dx=ux*(width/2+gap*fig.dpi/72)
                        dy=uy*(height/2+gap*fig.dpi/72)
                        center=origin+np.array([dx,dy])
                        rect=Bbox.from_bounds(center[0]-width/2,center[1]-height/2,width,height)
                        if rect.x0<ax.bbox.x0 or rect.x1>ax.bbox.x1 or rect.y0<ax.bbox.y0 or rect.y1>ax.bbox.y1:continue
                        def box_distance(point):
                            return math.hypot(max(rect.x0-point[0],0,point[0]-rect.x1),
                                              max(rect.y0-point[1],0,point[1]-rect.y1))
                        own_distance=box_distance(origin)
                        if own_distance>26*fig.dpi/72:continue
                        if any(box_distance(point)<own_distance+1 for other,point in points.items() if other!=name):continue
                        if any(rect.overlaps(other) for other in placed):continue
                        if any(rect.padded(node_radius).contains(*v) for v in points.values()):continue
                        if any(line_hits_box(a,b,rect.padded(stroke+2)) for a,b,stroke in segments):continue
                        distance=math.hypot(dx,dy)
                        # Prefer vertical labels when equally close to their node.
                        candidates.append((distance+abs(ux)*3,dx,dy,rect))
                    if candidates:break
                if not candidates:raise ValueError('No unobstructed label position: '+name)
                _,dx,dy,rect=min(candidates,key=lambda v:v[0])
                text.set_position((dx*72/fig.dpi,dy*72/fig.dpi));placed.append(rect)
            fig.canvas.draw()
            boxes=[t.get_window_extent(renderer).padded(1) for t in texts.values()]
            label_collisions=sum(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:])
            node_collisions=sum(box.padded(node_radius).contains(*v) for box in boxes for v in points.values())
            edge_collisions=sum(line_hits_box(a,b,box.padded(stroke+1)) for box in boxes for a,b,stroke in segments)
            if label_collisions or node_collisions or edge_collisions:
                raise AssertionError('Relationship map label collision')
            audit.append({'panel':ax.get_title(loc='left'),'nodes':len(graph),'edges':len(graph.edges),
                          'label_label_overlaps':int(label_collisions),'label_node_overlaps':int(node_collisions),
                          'label_edge_overlaps':int(edge_collisions)})
        Path(figures).mkdir(parents=True,exist_ok=True)
        (Path(figures)/'Figure5_layout_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
        structure_save(fig,figures,'Figure5_ingredient_relationships')
    return positions


def make_relationship_atlas(work_dir, figures):
    """Reuse the exact training-pair counts used in the prediction comparison."""
    import networkx as nx
    work_dir,figures=Path(work_dir),Path(figures)
    out=work_dir/'relationship_atlas';out.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    structure=work_dir/'predictive_null'
    source_meta=json.loads((structure/'metadata.json').read_text())
    if source_meta['status']!='complete':raise ValueError('Run --predictive-null-only first')
    for name,wanted in source_meta['input_hashes'].items():
        with (work_dir/name).open('rb') as h:
            if hashlib.file_digest(h,'sha256').hexdigest()!=wanted:raise ValueError('Structure and composition inputs differ: '+name)
    herbs=read_rows(work_dir/'herbal/unique_compositions.csv')
    members=read_rows(work_dir/'matching/membership.csv')
    wanted={r['composition_id'] for r in members}
    membership_count=Counter(r['composition_id'] for r in members)
    with (work_dir/'food/unique_compositions.csv').open(encoding='utf-8-sig',newline='') as h:
        foods=[r for r in csv.DictReader(h) if r['composition_id'] in wanted]
    assert len(herbs)==2009 and len(members)==100*2009
    cohorts={'herbal':herbs,'food':foods};ranked={};displayed={};all_rows=[];examples=[]
    pair_input_hashes={}
    for domain,rep,field in [('herbal',0,'herbs'),('food',1,'ingredients')]:
        path=structure/f'cohorts/{domain}_{rep:03d}/training_pairs.csv'
        with path.open('rb') as h:pair_input_hashes[domain]=hashlib.file_digest(h,'sha256').hexdigest()
        pairs=pooled_food_pairs(structure) if domain=='food' else read_training_pairs(path)
        if domain=='food':
            pair_input_hashes[domain]={}
            for sample in range(1,101):
                q=structure/f'cohorts/food_{sample:03d}/training_pairs.csv'
                with q.open('rb') as h:pair_input_hashes[domain][str(sample)]=hashlib.file_digest(h,'sha256').hexdigest()
        eligible=[r for r in pairs if r['ratio'] is not None and r['ratio']>1]
        eligible.sort(key=lambda r:(-r['ratio'],-r['observed_training_mean'],r['ingredient_a'],r['ingredient_b']))
        selected_keys={(r['ingredient_a'],r['ingredient_b']) for r in eligible}
        support_lookup=defaultdict(list)
        for r in cohorts[domain]:
            for key in combinations(sorted(r[field].split('|')),2):
                if key in selected_keys:support_lookup[key].append(r)
        ranked[domain]=[]
        for i,pair in enumerate(eligible,1):
            item=dict(pair,domain=domain,replicate="pooled" if domain=="food" else rep,rank=i,ratio=pair['ratio'],displayed=i<=30)
            supporting=support_lookup[pair['ingredient_a'],pair['ingredient_b']]
            if sum(membership_count[r['composition_id']] if domain=='food' else 1 for r in supporting)!=pair['unique_composition_support'] or sum(int(r['weight'])*(membership_count[r['composition_id']] if domain=='food' else 1) for r in supporting)!=pair['source_record_support']:raise AssertionError('Pair support does not match unique compositions')
            ranked[domain].append(item);all_rows.append(item)
            if i<=30:
                for r in sorted(supporting,key=lambda r:r['composition_id'])[:3]:
                    examples.append({'domain':domain,'replicate':'pooled' if domain=='food' else rep,'pair_rank':i,
                        'ingredient_a':pair['ingredient_a'],'ingredient_b':pair['ingredient_b'],
                        'composition_id':r['composition_id'],'composition':r[field],
                        'source_weight':int(r['weight']),
                        'names_or_title':r['formula_names'] if domain=='herbal' else r['example_title'],
                        'source_id':'' if domain=='herbal' else r['example_recipe_id'],
                        'source_file':'','source_pages':'','source_verified':False})
        displayed[domain]=ranked[domain][:30]
    # Trace herbal examples to an original textbook record, not just a composition label.
    textbook_records={}
    for path in sorted((ROOT/'data/herbal').glob('*.csv')):
        handle,reader,_=open_textbook_csv(path)
        try:
            id_col=find_column(reader.fieldnames,('처방아이디','처방ID','처방id'))
            herb_col=find_column(reader.fieldnames,('약재한글명','약재명'))
            name_col=find_column(reader.fieldnames,('처방한글명','처방명'))
            for row in reader:
                key=(path.name,clean_text(row.get(id_col)))
                r=textbook_records.setdefault(key,{'herbs':set(),'names':set(),'pages':set()})
                for dest,col in [('herbs',herb_col),('names',name_col),('pages','페이지')]:
                    value=clean_text(row.get(col))
                    if value:r[dest].add(value)
        finally:handle.close()
    by_composition=defaultdict(list)
    for (file,ident),r in sorted(textbook_records.items()):by_composition['|'.join(sorted(r['herbs']))].append((file,ident,r))
    for r in examples:
        if r['domain']=='herbal':
            sources=by_composition[r['composition']]
            if len(sources)!=r['source_weight']:raise AssertionError('Herbal source multiplicity mismatch')
            file,ident,record=sources[0]
            r.update(source_file=file,source_id=ident,source_pages='|'.join(sorted(record['pages'])),
                     names_or_title='|'.join(sorted(record['names'])),source_verified=True)
    # Independently check each chosen food record against Recipe1M and the saved vocabulary.
    food_ids={r['source_id'] for r in examples if r['domain']=='food'};verified={}
    mapping={r['alias']:r['canonical_ingredient'] for r in read_rows(work_dir/'food/canonical_ingredient_mapping.csv')}
    for layer,detection in paired_records(ROOT/'data/recipe1m/recipe1M_layers.tar.gz',ROOT/'data/recipe1m/det_ingrs.json'):
        if layer['id'] not in food_ids:continue
        composition=tuple(sorted({mapping[x] for x in valid_raw_ingredients(detection) if x in mapping}))
        if not eligible_recipe_for_atlas(composition,layer):raise AssertionError('Ineligible source recipe')
        verified[layer['id']]={'composition':'|'.join(composition),'title':layer['title']}
        if len(verified)==len(food_ids):break
    for r in examples:
        if r['domain']=='food':
            v=verified[r['source_id']]
            if r['composition']!=v['composition'] or r['names_or_title']!=v['title']:raise AssertionError('Food source mismatch')
            r.update(source_file='recipe1M_layers.tar.gz:layer1.json + det_ingrs.json',source_verified=True)
    # Summarize displayed food pairs across the matched samples.
    targets={(r['ingredient_a'],r['ingredient_b']):[] for r in displayed['food']}
    for rep in range(1,101):
        for r in read_training_pairs(structure/f'cohorts/food_{rep:03d}/training_pairs.csv'):
            key=(r['ingredient_a'],r['ingredient_b'])
            if key in targets:targets[key].append(r)
    stability=[]
    for key,rows in targets.items():
        ratios=[r['ratio'] for r in rows if r['ratio'] is not None]
        stability.append({'ingredient_a':key[0],'ingredient_b':key[1],
            'support10_samples':sum(r['unique_composition_support']>=10 for r in rows),'samples_with_pair':sum(r['unique_composition_support']>0 for r in rows),'ratio_above1_samples':sum(r['ratio'] is not None and r['ratio']>1 for r in rows),
            'ratio_median':float(np.median(ratios)) if ratios else ''})
    for filename,rows in [('ranked_pairs.csv',all_rows),('displayed_pairs.csv',displayed['herbal']+displayed['food']),
                          ('source_examples.csv',examples),('food_pair_stability.csv',stability)]:
        write_csv(out/filename,tuple(rows[0]),rows)
    positions=draw_relationship_map(displayed,cohorts,figures)
    write_csv(out/'network_positions.csv',tuple(positions[0]),positions)
    caption=('Fig. 5. Ingredient relationships relative to frequency-preserving randomization. Counts use the same training data as Fig. 4. '
        '(A) Herbal formulas. (B) Food recipes aggregated across 100 matched samples. Pharmacopoeial names are used for display. Source-record weights are retained. '
        'For each pair, original counts are averaged over five training folds; randomized counts are averaged over the same folds and all randomization runs. '
        'For food, the numerator and denominator are additionally averaged across all 100 samples, including samples with zero observed co-occurrence; sample ratios are not averaged. Line width represents the ratio on a common scale. '
        'Each panel shows the 30 largest finite ratios above one among pairs occurring in at least 10 unique compositions per sample on average (10 for herbal; summed sample support of at least 1,000 for food). Repeated compositions across food samples are retained as sampled memberships, not independent records. '
        'Pairs with zero randomized mean have no finite ratio and are retained in the downloadable counts but omitted from ranking. '
        'The map is descriptive; no p-values or significance filtering are used. '
        'Positions aid readability and are not measured similarities or inferred communities. '
        'Repeated appearances across training folds are not independent observations.')
    (figures/'Figure5_caption.txt').write_text(caption+'\n')
    metadata={'status':'complete','basis':'source-weighted training records, identical to prediction comparison',
        'selection':'Top 30 finite ratios >1; average unique support per sample >=10; no significance testing',
        'food_samples':100,'draws_per_fold':source_meta['draws'],'folds':5,
        'input_hashes':source_meta['input_hashes'],'pair_result_sha256':pair_input_hashes,
        'verified_source_examples':len(examples),'unique_food_source_records_verified':len(food_ids),
        'all_pair_counts_independently_recounted':True,
        'aggregation':'ratio of counts averaged over folds and all samples; shuffled denominator also averaged over draws',
        'shared_randomizations':True}
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print('Relationship atlas complete: 60 displayed pairs and '+str(len(examples))+' verified source examples.',flush=True)
    return metadata


def eligible_recipe_for_atlas(composition,layer):
    return eligible(composition,valid_instructions(layer))


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}", flush=True)


def main(workers=2, predictive_draws=20):
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
    make_frequency_figure(ROOT / "work", ROOT / "figures")
    evaluate_predictive_null(ROOT / "work", workers=workers, draws=predictive_draws)
    make_relationship_atlas(ROOT / "work", ROOT / "figures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--frequency-only", action="store_true")
    parser.add_argument("--predictive-null-only", action="store_true")
    parser.add_argument("--relationships-only", action="store_true")
    parser.add_argument("--predictive-draws", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if sum((args.evaluate_only, args.frequency_only, args.predictive_null_only, args.relationships_only)) > 1:
        parser.error("Choose only one analysis-only mode")
    if args.predictive_draws < 2:
        parser.error("--predictive-draws must be at least 2")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.relationships_only:
        make_relationship_atlas(ROOT / "work", ROOT / "figures")
    elif args.predictive_null_only:
        evaluate_predictive_null(ROOT / "work", workers=args.workers, draws=args.predictive_draws)
    elif args.frequency_only:
        make_frequency_figure(ROOT / "work", ROOT / "figures")
    elif args.evaluate_only:
        evaluate_all(ROOT / "work", workers=args.workers)
    else:
        main(workers=args.workers, predictive_draws=args.predictive_draws)
