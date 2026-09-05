#!/usr/bin/env python3
"""교과서 CSV를 약재 집합으로 만들고 동일한 전체 조성을 병합한다."""

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def clean(value) -> str:
    """약재명에 NFC와 공백 정리만 적용한다."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip()


def find_column(fields, candidates) -> str:
    for candidate in candidates:
        if candidate in fields:
            return candidate
    raise ValueError(f"필요한 열이 없습니다: {candidates}")


def open_csv(path: Path):
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        handle = path.open(encoding=encoding, newline="")
        try:
            reader = csv.DictReader(handle)
            reader.fieldnames = [clean(name) for name in (reader.fieldnames or [])]
            if reader.fieldnames:
                return handle, reader, encoding
        except UnicodeDecodeError:
            pass
        handle.close()
    raise ValueError(f"CSV 인코딩을 읽지 못했습니다: {path}")


def load_source_compositions(input_dir: Path):
    """파일명과 처방아이디를 원처방 식별자로 사용한다."""
    prescriptions = {}
    names = {}
    row_counts = Counter()
    files = sorted(input_dir.glob("*.csv"))
    if len(files) != 5:
        raise FileNotFoundError(f"교과서 CSV 5개가 필요합니다: {input_dir}")

    for path in files:
        handle, reader, encoding = open_csv(path)
        try:
            fields = reader.fieldnames or []
            id_column = find_column(fields, ("처방아이디", "처방ID", "처방id"))
            herb_column = find_column(fields, ("약재한글명", "약재명"))
            name_column = find_column(fields, ("처방한글명", "처방명"))
            for row_number, row in enumerate(reader, 2):
                prescription_id = clean(row.get(id_column))
                if not prescription_id:
                    raise ValueError(f"처방아이디가 비었습니다: {path}:{row_number}")
                key = (path.name, prescription_id)
                row_counts[key] += 1
                name = clean(row.get(name_column))
                herb = clean(row.get(herb_column))
                if name:
                    names[key] = name
                if herb:
                    prescriptions.setdefault(key, set()).add(herb)
        finally:
            handle.close()
        count = len({key for key in prescriptions if key[0] == path.name})
        print(f"  {path.name}: {encoding}, {count:,} formulas")

    return prescriptions, names, row_counts, files


def collapse_compositions(prescriptions, names):
    grouped = defaultdict(list)
    for key, herbs in prescriptions.items():
        if herbs:
            grouped[tuple(sorted(herbs))].append(key)

    rows = []
    for herbs, sources in grouped.items():
        signature = "|".join(herbs)
        rows.append(
            {
                "composition_id": hashlib.sha256(signature.encode()).hexdigest()[:16],
                "weight": len(sources),
                "herb_count": len(herbs),
                "herbs": signature,
                "formula_names": "|".join(
                    sorted({names.get(key, "") for key in sources if names.get(key, "")})
                ),
            }
        )
    return sorted(rows, key=lambda row: row["composition_id"])


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("composition_id", "weight", "herb_count", "herbs", "formula_names")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def preprocess(input_dir: Path, output_dir: Path, minimum_size=2, maximum_size=19):
    prescriptions, names, row_counts, files = load_source_compositions(input_dir)
    collapsed = collapse_compositions(prescriptions, names)
    selected = [
        row for row in collapsed
        if minimum_size <= row["herb_count"] <= maximum_size
    ]
    size_counts = Counter(row["herb_count"] for row in selected)

    metadata = {
        "input_files": [path.name for path in files],
        "source_identity": "CSV filename + formula ID",
        "herb_identifier": "Korean herb name",
        "name_processing": "NFC and whitespace only; no alias merging",
        "within_formula": "repeated Korean herb names counted once",
        "duplicate_policy": "merge identical complete sorted herb sets",
        "weight": "number of original formula IDs represented by a composition",
        "source_rows": sum(row_counts.values()),
        "source_formulas": sum(row["weight"] for row in collapsed),
        "unique_compositions": len(collapsed),
        "selected_size_range": [minimum_size, maximum_size],
        "selected_unique_compositions": len(selected),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "unique_compositions.csv", collapsed)
    with (output_dir / "composition_size_distribution.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("ingredient_count", "unique_compositions"))
        for size in range(minimum_size, maximum_size + 1):
            writer.writerow((size, size_counts[size]))
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata, collapsed, selected, size_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/herbal"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/herbal"))
    args = parser.parse_args()
    metadata, collapsed, _, _ = preprocess(args.input_dir, args.output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("\n[동일 조성 병합 예시]")
    for row in sorted(collapsed, key=lambda value: (-value["weight"], value["composition_id"]))[:3]:
        print(f"이름: {row['formula_names']}")
        print(f"조성({row['herb_count']}개): {row['herbs']}")
        print(f"학습 weight: {row['weight']}\n")


if __name__ == "__main__":
    main()
