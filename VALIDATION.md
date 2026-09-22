# 원본 대조 및 수정 결과

Inverse Cooking의 공식 build_vocab.py에 있는 문자열 정리·재료 통합 함수를 직접 사용하고, 원본의 기본값 및 최종 선정 조건으로 전체 Recipe1M 자료를 독립 집계했다. 현재 구현의 모든 재료명 정리 결과, 어휘 매핑, 최종 조성과 중복 횟수가 일치했다. 이미지 처리와 문장 토큰화 결과는 이 검증 범위가 아니다.

| 자료 | 전체 | 전처리/개수 조건 적용 | 중복 병합 |
|---|---:|---:|---:|
| Recipe1M | 1,029,720 | 921,927 | 770,945 |
| 한약 처방 | 3,078 | 2,992 | 2,009 |

기존 구현은 최종 선정 단계에서 원본에 없는 표준화 전 재료 수 제한을 추가로 적용했다. 이를 제거하여 8,247개 레시피가 복원되었다. 기존 음식 수치 913,680 및 762,996은 원본 규칙에 따른 결과로 사용하지 않는다. 음식의 조리 지시문 조건은 원본대로 유지했다. 원자료 개수 제한은 어휘 구축 단계에 적용되며 최종 자료 선정의 순차 제외 단계가 아니다.

수정 코드, 집계 CSV, Figure 1을 재생성했다. 한약의 최종 조성은 변경되지 않았다. 원본 대조 증거는 work/original_comparison.json, 실행 로그는 work/run.log에 있다. verify_original.py는 work/reference_build_vocab.py의 공식 원본 스냅샷으로 검증을 재실행한다.

## 원고에 사용할 문장

Food recipes were obtained from Recipe1M, which contained 1,029,720 recipes. [13] Following preprocessing using the Inverse Cooking procedure, 921,927 recipes containing 2–19 standardized ingredients remained. [14] Merging recipes with identical ingredient compositions yielded 770,945 unique compositions for comparison.

Herbal formulas were collected from five KM internal medicine textbooks covering the liver, heart, spleen, lung, and kidney systems, yielding 3,078 formulas. [15–19] Restricting the dataset to formulas containing 2–19 herbs retained 2,992 formulas. Merging formulas with identical herb compositions yielded 2,009 unique compositions for comparison.

## GitHub에서 확인할 파일

- `run.py`: 수정 코드. `preprocess_food()`의 두 번째 순회에서는 표준화 후 재료 수로 선정한다.
- `validation/dataset_flow.csv`: 두 자료의 최종 집계.
- `validation/original_comparison.json`: 원본 스냅샷 SHA-256과 전체 대조 결과.
- `validation/food_metadata.json`, `validation/herbal_metadata.json`: 상세 집계.
- `validation/run.log`: 전체 실행 출력.
- `figures/Figure1_dataset_matching.png`: 수정된 Figure 1.

**범위:** 전처리와 Figure 1만 재실행했다. 음식 자료에 의존하는 후속 분석은 재실행해야 하며, 아직 검증 완료로 간주하지 않는다.

## 재현

원자료를 준비한 뒤 `python run.py`를 실행한다. 이어 공식 원본을 내려받아 독립 대조한다. 검증 스크립트는 SHA-256이 검증 당시 스냅샷과 다르면 중단한다.

```bash
curl -L --fail https://raw.githubusercontent.com/facebookresearch/inversecooking/master/src/build_vocab.py -o work/reference_build_vocab.py
python verify_original.py --source work/reference_build_vocab.py
```
