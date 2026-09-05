# Herbal Formula Completion

Reproducible code and data provenance for comparing ingredient composition in food recipes and Korean medicine herbal formulas.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inseong101/HerbalFormulaCompletion/blob/main/Run_analysis_in_Colab.ipynb)
[![Run analysis](https://github.com/inseong101/HerbalFormulaCompletion/actions/workflows/run-analysis.yml/badge.svg)](https://github.com/inseong101/HerbalFormulaCompletion/actions/workflows/run-analysis.yml)

## What the run shows

`analysis.py` does not jump directly to a finished figure. It prints:

1. the five raw herbal filenames, the actual CSV header, and an actual first row;
2. the number of formulas found in each textbook CSV;
3. the change from 3,078 formulas to 2,082 unique complete compositions and then 2,009 compositions with 2–19 herbs;
4. an actual duplicate-composition example and its training weight;
5. the official Recipe1M source, required raw filenames, actual JSON fields, and a paired record excerpt;
6. every food preprocessing count from the verified full-data run;
7. all length-specific counts and percentages used in Figure 1;
8. the final SVG and 600-dpi PNG paths.

## One-click run

Click **Open in Colab**, then choose **Runtime → Run all**. No local installation is needed.

For a local run:

```bash
pip install -r requirements.txt
python analysis.py
```

## Data availability

- The five complete herbal CSV files are included because the Korean Public Data Portal lists their permission scope as unrestricted (`이용허락범위 제한 없음`).
- Recipe1M requires registration. Its full JSON files are not redistributed here. The repository provides the official download page, papers, verified file hashes, actual schema example, exact preprocessing code, and validated aggregate output.
- A registered user can place `det_ingrs.json` and `layer1.json` in `data/raw/recipe1m/` and run `python raw_pipeline/food.py` to repeat the full food preprocessing.

See [DATA.md](DATA.md) for exact URLs, citations, file structure, checksums, and preprocessing details.

## Minimal repository structure

```text
HerbalFormulaCompletion/
├── Run_analysis_in_Colab.ipynb
├── analysis.py
├── DATA.md
├── data/
│   ├── raw/
│   │   ├── herbal/              # five complete public CSV files
│   │   └── food_example/         # verified Recipe1M schema excerpt
│   └── processed/
│       └── food/                 # validated output needed for Figure 1
├── raw_pipeline/
│   ├── herbal.py
│   └── food.py
└── figures/
```

Generated herbal intermediates are written to `data/processed/herbal/`. Figure 1 is saved to `figures/Figure1_dataset_matching.svg` and `figures/Figure1_dataset_matching.png`.

## Suggested Code Availability statement

> The analysis code, public raw herbal-formula data, data-provenance records, and derived inputs required to reproduce Figure 1 are available at <https://github.com/inseong101/HerbalFormulaCompletion>. The Recipe1M raw files are not redistributed in the repository because official access requires registration; they can be obtained from <https://im2recipe.csail.mit.edu/dataset/download> and processed with the supplied script.
