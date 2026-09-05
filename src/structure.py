#!/usr/bin/env python3
"""표본 수와 조성 길이를 맞춰 한약·음식 재료쌍 구조를 비교한다."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from tqdm import tqdm


def read_compositions(path, item_field, size_field):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{"items": tuple(row[item_field].split("|")),
                 "size": int(row[size_field]), "weight": int(row["weight"])}
                for row in csv.DictReader(handle)]


def unique_by_size(rows, minimum, maximum):
    grouped = defaultdict(list)
    for row in rows:
        if minimum <= row["size"] <= maximum:
            grouped[row["size"]].append(row["items"])
    return grouped


def gini(values):
    ordered = sorted(values)
    total = sum(ordered)
    if not ordered or total == 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2 * weighted) / (len(ordered) * total) - (len(ordered) + 1) / len(ordered)


def structure_metrics(source_compositions):
    item_counts, pair_counts = Counter(), Counter()
    unique_sets = set(source_compositions)
    for items in source_compositions:
        item_counts.update(items)
        pair_counts.update(combinations(items, 2))
    pair_total = sum(pair_counts.values())
    item_total = sum(item_counts.values())
    pair_values = list(pair_counts.values())
    item_values = list(item_counts.values())
    top_pair_n = max(1, math.ceil(len(pair_values) * 0.01))
    top_item_n = max(1, math.ceil(len(item_values) * 0.01))
    return {
        "source_compositions": len(source_compositions),
        "unique_compositions": len(unique_sets),
        "unique_items": len(item_counts),
        "item_incidences": item_total,
        "distinct_pairs": len(pair_counts),
        "pair_incidences": pair_total,
        "item_frequency_gini": gini(item_values),
        "pair_frequency_gini": gini(pair_values),
        "item_hhi": sum((value/item_total)**2 for value in item_values),
        "pair_hhi": sum((value/pair_total)**2 for value in pair_values),
        "top_1pct_item_share": sum(sorted(item_values, reverse=True)[:top_item_n])/item_total,
        "top_1pct_pair_share": sum(sorted(pair_values, reverse=True)[:top_pair_n])/pair_total,
        "max_item_prevalence": max(item_values)/len(source_compositions),
        "max_pair_prevalence": max(pair_values)/len(source_compositions),
        "singleton_pair_share": sum(value == 1 for value in pair_values)/len(pair_values),
    }


def percentile(values, probability):
    values = sorted(values)
    position = (len(values)-1)*probability
    low = math.floor(position); high = math.ceil(position)
    if low == high:
        return values[low]
    return values[low]*(high-position) + values[high]*(position-low)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def summarize_replicates(herb_metrics, replicate_rows, excluded=()):
    metric_names = [key for key in herb_metrics if key not in {"source_compositions", *excluded}]
    comparisons = []
    for metric in metric_names:
        food_values = [row[metric] for row in replicate_rows]
        mean = sum(food_values)/len(food_values)
        sd = math.sqrt(sum((value-mean)**2 for value in food_values)/(len(food_values)-1))
        herb_value = herb_metrics[metric]
        comparisons.append({
            "design": "size_matched", "metric": metric, "herb": herb_value, "food_mean": mean,
            "food_sd": sd, "food_p2_5": percentile(food_values, 0.025),
            "food_p97_5": percentile(food_values, 0.975),
            "herb_minus_food_mean": herb_value-mean,
            "herb_percentile_among_food_replicates": sum(value <= herb_value for value in food_values)/len(food_values),
        })
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--herbs", type=Path, default=Path("work/herbal/unique_compositions.csv"))
    parser.add_argument("--food", type=Path, default=Path("work/food/unique_compositions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/structure"))
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--minimum-size", type=int, default=2)
    parser.add_argument("--maximum-size", type=int, default=19)
    args = parser.parse_args()

    herbs = read_compositions(args.herbs, "herbs", "herb_count")
    foods = read_compositions(args.food, "ingredients", "ingredient_count")
    herb_unique = unique_by_size(herbs, args.minimum_size, args.maximum_size)
    food_unique = unique_by_size(foods, args.minimum_size, args.maximum_size)
    required = {size: len(rows) for size, rows in herb_unique.items()}
    shortages = {size: (count, len(food_unique[size])) for size, count in required.items()
                 if len(food_unique[size]) < count}
    if shortages:
        raise ValueError(f"길이별 음식 조성이 부족합니다: {shortages}")

    herb_population = [items for size in sorted(herb_unique) for items in herb_unique[size]]
    herb_metrics = structure_metrics(herb_population)
    size_matched_rows = []
    for replicate in tqdm(range(1, args.replicates + 1), desc="Matched food structure samples"):
        rng = random.Random(args.seed + replicate - 1)
        sample = [items for size in sorted(required)
                  for items in rng.sample(food_unique[size], required[size])]
        size_matched_rows.append({"replicate": replicate, **structure_metrics(sample)})

    comparisons = summarize_replicates(
        herb_metrics, size_matched_rows,
        excluded={"pair_incidences", "item_incidences"},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir/"food_replicate_metrics.csv", list(size_matched_rows[0]), size_matched_rows)
    write_csv(args.output_dir/"observed_comparison.csv", list(comparisons[0]), comparisons)
    write_csv(args.output_dir/"matched_size_distribution.csv", ["composition_size", "unique_compositions"],
              ({"composition_size": size, "unique_compositions": required[size]} for size in sorted(required)))
    metadata = {
        "question": "한약 처방과 음식 레시피의 재료쌍 집중도가 비슷한가?",
        "design": "각 음식 표본의 고유조성 수와 조성 길이별 개수를 한약과 동일하게 맞춤",
        "common_size_support": [args.minimum_size, args.maximum_size],
        "herbal_unique_compositions": len(herb_population),
        "food_replicates": args.replicates,
        "seed_rule": "seed + replicate - 1",
        "seed": args.seed,
        "sampling": f"Recipe1M 고유조성에서 반복 안에서는 비복원 추출하며 {args.replicates}회 반복",
        "network_definition": "재료를 노드로 보고 두 재료를 함께 포함한 고유조성마다 무방향 재료쌍에 1을 더함",
        "interpretation": f"한약 고정 cohort를 {args.replicates}개의 독립적인 길이·표본수 매칭 음식 표본 분포와 비교",
        "weight_policy": "주요 구조 비교에서는 두 영역 모두 원자료 반복 가중치를 사용하지 않음",
        "herb_metrics": herb_metrics,
    }
    (args.output_dir/"metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    for row in comparisons:
        print(row)


if __name__ == "__main__":
    main()
