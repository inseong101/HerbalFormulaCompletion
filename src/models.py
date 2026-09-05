#!/usr/bin/env python3
"""세 가지 공출현 추천법을 strict Top-10 기준으로 비교한다."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

try:
    from .core import (
        CONDITIONS, MODELS, assign_folds, context_size, performance_name, percentile,
        read_compositions, sample_contexts, stable_context_seed, strict_top_k, train_statistics,
    )
except ImportError:
    from core import (
        CONDITIONS, MODELS, assign_folds, context_size, performance_name, percentile,
        read_compositions, sample_contexts, stable_context_seed, strict_top_k, train_statistics,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "work/models"


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_dataset(domain: str, rows, replicate: int, folds: int,
                     fold_seed: int, sampling_seed: int, maximum_contexts: int):
    """한 cohort를 평가하고 fold별 결과를 반환한다."""
    buckets = assign_folds(rows, folds, fold_seed)
    fold_rows = []
    for fold_index, test in enumerate(buckets, 1):
        train = [row for index, bucket in enumerate(buckets, 1)
                 if index != fold_index for row in bucket]
        statistics = train_statistics(train)
        vocabulary = statistics["vocabulary"]
        composition_values = defaultdict(list)

        for composition in test:
            items = composition["items"]
            for condition in CONDITIONS:
                observed_count = context_size(condition, len(items))
                if not 1 <= observed_count < len(items):
                    continue
                contexts = sample_contexts(
                    items,
                    observed_count,
                    maximum_contexts,
                    stable_context_seed(composition["id"], condition, sampling_seed),
                )
                values = defaultdict(list)
                coverages = []
                candidate_counts = []
                for context in contexts:
                    hidden = set(items) - set(context)
                    available = hidden & vocabulary
                    coverages.append(len(available) / len(hidden))
                    candidate_counts.append(len(vocabulary - set(context)))
                    for model in MODELS:
                        top10 = strict_top_k(model, context, statistics, 10)
                        hits = len(hidden & set(top10))
                        values[model].append(hits / len(hidden))

                for model in MODELS:
                    composition_values[condition, model].append({
                        "performance": sum(values[model]) / len(values[model]),
                        "target_coverage": sum(coverages) / len(coverages),
                        "candidate_count": sum(candidate_counts) / len(candidate_counts),
                    })

        for condition in CONDITIONS:
            for model in MODELS:
                values = composition_values[condition, model]
                fold_rows.append({
                    "domain": domain,
                    "food_replicate": replicate if domain == "food" else "",
                    "fold": fold_index,
                    "condition": condition,
                    "model": model,
                    "metric": performance_name(condition),
                    "test_unique_compositions": len(values),
                    "performance": sum(row["performance"] for row in values) / len(values),
                    "target_coverage": sum(row["target_coverage"] for row in values) / len(values),
                    "mean_candidates": sum(row["candidate_count"] for row in values) / len(values),
                })
    return fold_rows


def sample_food(food_by_size, needed, replicate: int, seed: int):
    """반복별 독립 seed로 길이 분포가 같은 음식 표본을 만든다."""
    rng = random.Random(seed + replicate - 1)
    return [
        row
        for size in sorted(needed)
        for row in rng.sample(food_by_size[size], needed[size])
    ]


def evaluate_food_replicate(arguments):
    replicate, rows, folds, fold_seed, sampling_seed, maximum_contexts = arguments
    return replicate, evaluate_dataset(
        "food", rows, replicate, folds, fold_seed, sampling_seed, maximum_contexts
    )


def summarize(herbal_fold_rows, food_fold_rows):
    summary = []

    grouped = defaultdict(list)
    for row in herbal_fold_rows:
        grouped[row["condition"], row["model"]].append(row)
    for (condition, model), values in sorted(grouped.items()):
        performances = [row["performance"] for row in values]
        mean = sum(performances) / len(performances)
        summary.append({
            "domain": "herbal",
            "condition": condition,
            "model": model,
            "metric": performance_name(condition),
            "summary_unit": "fold",
            "summary_units": len(values),
            "mean_performance": mean,
            "sd_performance": math.sqrt(
                sum((value - mean) ** 2 for value in performances) / (len(performances) - 1)
            ),
            "p2_5_performance": "",
            "p97_5_performance": "",
            "mean_target_coverage": sum(row["target_coverage"] for row in values) / len(values),
            "mean_candidates": sum(row["mean_candidates"] for row in values) / len(values),
        })

    replicate_rows = []
    grouped = defaultdict(list)
    for row in food_fold_rows:
        grouped[row["food_replicate"], row["condition"], row["model"]].append(row)
    for (replicate, condition, model), values in sorted(grouped.items()):
        replicate_rows.append({
            "food_replicate": replicate,
            "condition": condition,
            "model": model,
            "metric": performance_name(condition),
            "folds": len(values),
            "performance": sum(row["performance"] for row in values) / len(values),
            "target_coverage": sum(row["target_coverage"] for row in values) / len(values),
            "mean_candidates": sum(row["mean_candidates"] for row in values) / len(values),
        })

    grouped = defaultdict(list)
    for row in replicate_rows:
        grouped[row["condition"], row["model"]].append(row)
    for (condition, model), values in sorted(grouped.items()):
        performances = [row["performance"] for row in values]
        mean = sum(performances) / len(performances)
        summary.append({
            "domain": "food",
            "condition": condition,
            "model": model,
            "metric": performance_name(condition),
            "summary_unit": "matched_food_sample",
            "summary_units": len(values),
            "mean_performance": mean,
            "sd_performance": math.sqrt(
                sum((value - mean) ** 2 for value in performances) / (len(performances) - 1)
            ),
            "p2_5_performance": percentile(performances, 0.025),
            "p97_5_performance": percentile(performances, 0.975),
            "mean_target_coverage": sum(row["target_coverage"] for row in values) / len(values),
            "mean_candidates": sum(row["mean_candidates"] for row in values) / len(values),
        })
    return summary, replicate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--food-replicates", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260812)
    parser.add_argument("--food-seed", type=int, default=20260823)
    parser.add_argument("--sampling-seed", type=int, default=20260823)
    parser.add_argument("--maximum-contexts", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.food_replicates < 2:
        raise ValueError("음식 sampling variability를 계산하려면 food-replicates가 2 이상이어야 합니다")

    herbs = read_compositions(
        ROOT / "work/herbal/unique_compositions.csv", "herbs", "herb_count"
    )
    foods = read_compositions(
        ROOT / "work/food/unique_compositions.csv",
        "ingredients", "ingredient_count",
    )
    food_by_size = defaultdict(list)
    for row in foods:
        food_by_size[row["size"]].append(row)
    needed = Counter(row["size"] for row in herbs)
    shortages = {size: (needed[size], len(food_by_size[size])) for size in needed
                 if len(food_by_size[size]) < needed[size]}
    if shortages:
        raise ValueError(f"길이별 음식 고유조성이 부족합니다: {shortages}")

    print("한약 고정 cohort 평가", flush=True)
    herbal_fold_rows = evaluate_dataset(
        "herbal", herbs, 0, args.folds, args.fold_seed,
        args.sampling_seed, args.maximum_contexts,
    )

    tasks = []
    for replicate in range(1, args.food_replicates + 1):
        sample = sample_food(food_by_size, needed, replicate, args.food_seed)
        tasks.append((replicate, sample, args.folds, args.fold_seed,
                      args.sampling_seed, args.maximum_contexts))

    food_fold_rows = []
    if args.workers == 1:
        for task in tqdm(tasks, desc="Matched food samples"):
            _, values = evaluate_food_replicate(task)
            food_fold_rows.extend(values)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_food_replicate, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Matched food samples"):
                _, values = future.result()
                food_fold_rows.extend(values)

    herbal_fold_rows.sort(key=lambda row: (row["fold"], row["condition"], row["model"]))
    food_fold_rows.sort(key=lambda row: (
        int(row["food_replicate"]), row["fold"], row["condition"], row["model"]
    ))
    summary, food_replicate_rows = summarize(herbal_fold_rows, food_fold_rows)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fold_fields = list(herbal_fold_rows[0])
    write_csv(out / "herbal_fold_results.csv", fold_fields, herbal_fold_rows)
    write_csv(out / "food_fold_results.csv", fold_fields, food_fold_rows)
    write_csv(out / "food_replicate_results.csv", list(food_replicate_rows[0]), food_replicate_rows)
    write_csv(out / "summary.csv", list(summary[0]), summary)

    metadata = {
        "models": {
            "popularity": "학습자료의 weighted item count",
            "mean_conditional": "입력 항목별 weighted conditional probability의 0 포함 평균",
            "mean_jaccard": "입력 항목과 후보 간 weighted pairwise Jaccard의 0 포함 평균",
        },
        "training_weight": "동일 조성의 원자료 반복수를 세 모델 모두에 사용",
        "evaluation_weight": "각 고유 시험 조성을 같은 비중으로 평가",
        "conditions": list(CONDITIONS),
        "metrics": {condition: performance_name(condition) for condition in CONDITIONS},
        "top10_policy": "점수 내림차순 뒤 항목명 Unicode 오름차순으로 정확히 10개 선택",
        "masking": f"조성·조건별 가능한 입력 부분집합을 최대 {args.maximum_contexts}개 고정 추출",
        "folds": args.folds,
        "fold_seed": args.fold_seed,
        "sampling_seed": args.sampling_seed,
        "food_sampling": {
            "replicates": args.food_replicates,
            "seed_rule": "food_seed + replicate - 1",
            "food_seed": args.food_seed,
            "matching": "각 반복에서 한약과 고유조성 수 및 조성 길이별 개수를 동일하게 맞춤",
            "sampling": "반복 안에서는 비복원, 반복 사이에는 동일 조성이 다시 선택될 수 있음",
            "summary_unit": "각 food sample의 5-fold 평균",
        },
        "herbal_summary_unit": "고정 한약 cohort의 5개 fold",
        "leakage_control": "동일 조성 병합 후 fold를 나누고 모든 count는 training fold에서만 계산",
        "workers": args.workers,
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
