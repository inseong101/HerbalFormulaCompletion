#!/usr/bin/env python3
"""원자료 학습 레코드 수에 따른 Mean conditional probability 성능을 계산한다."""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .core import (
        assign_folds, read_compositions, sample_contexts, stable_context_seed,
        strict_top_k, train_statistics,
    )
except ImportError:
    from core import (
        assign_folds, read_compositions, sample_contexts, stable_context_seed,
        strict_top_k, train_statistics,
    )


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/learning"
HERBAL_SCALES = (0.1, 0.25, 0.5, 0.75, 1.0)
FOOD_SCALES = HERBAL_SCALES + (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def shuffled_by_size(rows, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["size"]].append(row)
    for size in grouped:
        random.Random(seed + size).shuffle(grouped[size])
    return grouped


def prefix_by_weight(rows, target_weight):
    selected = []
    total = 0
    for row in rows:
        if total >= target_weight:
            break
        selected.append(row)
        total += row["weight"]
    return selected


def nested_training_by_weight(base_train, population, test_ids, scale, seed):
    """같은 순서의 prefix를 써서 scale이 커질수록 학습자료가 포함되게 한다."""
    base = shuffled_by_size(base_train, seed)
    base_ids = {row["id"] for row in base_train}
    extras = shuffled_by_size(
        [row for row in population if row["id"] not in test_ids and row["id"] not in base_ids],
        seed + 1000,
    )
    selected = []
    shortfall = 0
    for size in sorted(base):
        base_weight = sum(row["weight"] for row in base[size])
        target = max(1, round(base_weight * scale))
        if scale <= 1:
            selected.extend(prefix_by_weight(base[size], target))
            continue
        selected.extend(base[size])
        remaining = target - base_weight
        added = prefix_by_weight(extras[size], remaining)
        selected.extend(added)
        shortfall += max(0, remaining - sum(row["weight"] for row in added))

    if shortfall:
        used = {row["id"] for row in selected}
        fallback = [
            row for size in sorted(extras) for row in extras[size]
            if row["id"] not in used
        ]
        added = prefix_by_weight(fallback, shortfall)
        selected.extend(added)
        if sum(row["weight"] for row in added) < shortfall:
            raise ValueError(f"{scale}배 분석에 사용할 원자료 레코드가 부족합니다")
    return selected


def evaluate_n_minus_1(test, statistics, maximum_contexts=5, sampling_seed=20260823):
    vocabulary = statistics["vocabulary"]
    composition_hits = []
    composition_coverage = []
    for row in test:
        items = row["items"]
        contexts = sample_contexts(
            items,
            len(items) - 1,
            maximum_contexts,
            stable_context_seed(row["id"], "N-1", sampling_seed),
        )
        hits = []
        coverages = []
        for context in contexts:
            hidden = next(iter(set(items) - set(context)))
            top10 = strict_top_k("mean_conditional", context, statistics, 10)
            hits.append(float(hidden in top10))
            coverages.append(float(hidden in vocabulary))
        composition_hits.append(sum(hits) / len(hits))
        composition_coverage.append(sum(coverages) / len(coverages))
    return (
        sum(composition_hits) / len(composition_hits),
        sum(composition_coverage) / len(composition_coverage),
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    herbs = read_compositions(
        ROOT / "work/herbal/unique_compositions.csv", "herbs", "herb_count"
    )
    foods = read_compositions(
        ROOT / "work/food/unique_compositions.csv",
        "ingredients",
        "ingredient_count",
    )

    food_by_size = defaultdict(list)
    for row in foods:
        food_by_size[row["size"]].append(row)
    needed = Counter(row["size"] for row in herbs)
    rng = random.Random(20260823)
    food_base = [
        row for size in sorted(needed)
        for row in rng.sample(food_by_size[size], needed[size])
    ]

    fold_rows = []
    for domain, base, population, scales in (
        ("herbal", herbs, herbs, HERBAL_SCALES),
        ("food", food_base, foods, FOOD_SCALES),
    ):
        buckets = assign_folds(base, 5, 20260812)
        for fold_index, test in enumerate(buckets, 1):
            base_train = [
                row for index, bucket in enumerate(buckets, 1)
                if index != fold_index for row in bucket
            ]
            test_ids = {row["id"] for row in test}
            for scale in scales:
                train = nested_training_by_weight(
                    base_train,
                    population,
                    test_ids,
                    scale,
                    20260823 + fold_index,
                )
                statistics = train_statistics(train)
                hit, coverage = evaluate_n_minus_1(test, statistics)
                row = {
                    "domain": domain,
                    "model": "mean_conditional",
                    "condition": "N-1",
                    "metric": "Hit@10",
                    "training_source_record_scale": scale,
                    "fold": fold_index,
                    "train_unique_compositions": len(train),
                    "train_source_records": sum(value["weight"] for value in train),
                    "test_unique_compositions": len(test),
                    "hit@10": hit,
                    "target_coverage": coverage,
                }
                fold_rows.append(row)
                print(
                    domain, scale, fold_index, len(train), row["train_source_records"],
                    round(hit, 4), flush=True,
                )

    grouped = defaultdict(list)
    for row in fold_rows:
        grouped[row["domain"], row["training_source_record_scale"]].append(row)
    summary = []
    for (domain, scale), values in sorted(grouped.items()):
        hits = [row["hit@10"] for row in values]
        mean = sum(hits) / len(hits)
        summary.append({
            "domain": domain,
            "model": "mean_conditional",
            "condition": "N-1",
            "metric": "Hit@10",
            "training_source_record_scale": scale,
            "folds": len(values),
            "mean_train_unique_compositions": sum(
                row["train_unique_compositions"] for row in values
            ) / len(values),
            "mean_train_source_records": sum(
                row["train_source_records"] for row in values
            ) / len(values),
            "mean_hit@10": mean,
            "sd_hit@10": math.sqrt(
                sum((value - mean) ** 2 for value in hits) / (len(hits) - 1)
            ),
            "mean_target_coverage": sum(row["target_coverage"] for row in values) / len(values),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "fold_results.csv", fold_rows)
    write_csv(OUT / "summary.csv", summary)
    metadata = {
        "analysis_role": "secondary learning-curve analysis",
        "model": "Mean conditional probability",
        "condition": "N-1",
        "metric": "strict Top-10 Hit@10",
        "training_weight": "원자료 반복수(source multiplicity)",
        "evaluation_weight": "각 고유 시험조성을 같은 비중으로 평가",
        "test_policy": "영역 안에서 모든 학습규모에 같은 5-fold 시험자료 사용",
        "training_scale": "각 fold의 기본 학습자료가 대표하는 원자료 레코드 수에 대한 비율",
        "training_sampling": "조성 길이별 고정 순서의 비복원 prefix; 규모가 커질수록 포함관계 유지",
        "large_food_length_matching": "20배까지 길이별로 확장하고, 50·100배의 부족분은 다른 길이의 적격 음식 조성으로 보충",
        "contexts": "조성별 숨김 대상을 최대 5개 고정 추출",
        "tie_policy": "점수 내림차순 뒤 항목명 Unicode 오름차순으로 정확히 10개 선택",
        "folds": 5,
        "fold_seed": 20260812,
        "sampling_seed": 20260823,
        "herbal_scales": HERBAL_SCALES,
        "food_scales": FOOD_SCALES,
    }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
