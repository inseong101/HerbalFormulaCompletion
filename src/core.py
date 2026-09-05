"""공출현 기반 구성 추천에 공통으로 사용하는 함수."""
from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


MODELS = ("popularity", "mean_conditional", "mean_jaccard")
CONDITIONS = ("2", "3", "50%", "75%", "N-1")


def read_compositions(path: Path, item_field: str, size_field: str,
                      minimum_size: int = 2, maximum_size: int = 19):
    """고유조성 CSV를 읽고 공통 형식으로 바꾼다."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            size = int(row[size_field])
            if minimum_size <= size <= maximum_size:
                rows.append({
                    "id": row["composition_id"],
                    "items": tuple(row[item_field].split("|")),
                    "size": size,
                    "weight": int(row["weight"]),
                })
        return rows


def context_size(condition: str, composition_size: int) -> int:
    if condition == "N-1":
        return composition_size - 1
    if condition.endswith("%"):
        percentage = int(condition[:-1])
        return min(composition_size - 1,
                   max(1, math.floor(composition_size * percentage / 100 + 0.5)))
    return int(condition)


def sample_contexts(items, size: int, maximum: int, seed: int):
    """가능한 문맥이 많으면 고정 seed로 일부만 추출한다."""
    total = math.comb(len(items), size)
    if total <= maximum:
        return list(combinations(items, size))
    rng = random.Random(seed)
    result = set()
    while len(result) < maximum:
        result.add(tuple(sorted(rng.sample(items, size))))
    return sorted(result)


def stable_context_seed(composition_id: str, condition: str, sampling_seed: int) -> int:
    digest = hashlib.sha256((composition_id + condition).encode()).hexdigest()[:16]
    return int(digest, 16) + sampling_seed


def assign_folds(rows, number: int, seed: int):
    """조성 길이 합이 비슷하도록 고유조성을 fold에 배정한다."""
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda row: len(row["items"]), reverse=True)
    buckets = [[] for _ in range(number)]
    loads = [0] * number
    for row in shuffled:
        index = min(range(number), key=lambda value: (loads[value], len(buckets[value]), value))
        buckets[index].append(row)
        loads[index] += len(row["items"])
    return buckets


def train_statistics(rows):
    """원자료 반복 weight를 사용해 출현·공출현 통계를 계산한다."""
    item_counts = Counter()
    pair_counts = Counter()
    for row in rows:
        weight = row["weight"]
        item_counts.update({item: weight for item in row["items"]})
        pair_counts.update({pair: weight for pair in combinations(row["items"], 2)})

    conditional = defaultdict(dict)
    jaccard = defaultdict(dict)
    for (first, second), count in pair_counts.items():
        conditional[first][second] = count / item_counts[first]
        conditional[second][first] = count / item_counts[second]
        similarity = count / (item_counts[first] + item_counts[second] - count)
        jaccard[first][second] = similarity
        jaccard[second][first] = similarity

    vocabulary = set(item_counts)
    popularity_order = sorted(vocabulary, key=lambda item: (-item_counts[item], item))
    alphabetical_order = sorted(vocabulary)
    return {
        "vocabulary": vocabulary,
        "item_counts": item_counts,
        "conditional": conditional,
        "jaccard": jaccard,
        "popularity_order": popularity_order,
        "alphabetical_order": alphabetical_order,
    }


def strict_top_k(model: str, context, statistics, k: int = 10):
    """점수 내림차순, 항목명 오름차순으로 정확히 k개까지 반환한다."""
    if model not in MODELS:
        raise ValueError(f"지원하지 않는 모델입니다: {model}")
    if k < 1:
        raise ValueError("k는 1 이상이어야 합니다")
    context_set = set(context)
    if model == "popularity":
        return [item for item in statistics["popularity_order"] if item not in context_set][:k]

    source = statistics["conditional"] if model == "mean_conditional" else statistics["jaccard"]
    scores = defaultdict(float)
    for observed in context:
        for candidate, value in source.get(observed, {}).items():
            if candidate not in context_set:
                scores[candidate] += value

    positive = sorted(scores, key=lambda item: (-scores[item], item))[:k]
    if len(positive) == k:
        return positive
    selected = set(positive)
    for item in statistics["alphabetical_order"]:
        if item not in context_set and item not in selected:
            positive.append(item)
            if len(positive) == k:
                break
    return positive


def performance_name(condition: str) -> str:
    return "Hit@10" if condition == "N-1" else "Recall@10"


def percentile(values, probability: float):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
