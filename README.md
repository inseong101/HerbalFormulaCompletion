# Food Recipes and Herbal Formulas

Code for preprocessing the two datasets and producing Figure 1 of the manuscript.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inseong101/HerbalFormulaCompletion/blob/main/Colab.ipynb)

## Verified correction

See [VALIDATION.md](VALIDATION.md) for the original-code comparison, corrected
counts, and manuscript wording. Food counts are now **921,927 eligible recipes**
and **770,945 unique compositions**; earlier counts of 913,680 and 762,996 are
superseded. [validation/](validation/) contains the executed counts and log.
Only preprocessing and Figure 1 have been rerun; downstream food-dependent
analyses must be rerun before their results are used.

## Run

```bash
pip install -r requirements.txt
python run.py
```

The terminal shows the raw fields, one raw record, preprocessing counts, one duplicate-composition example from each domain, the ingredient-count table, and the saved Figure 1 paths.

Both datasets follow the same reporting and processing flow: source records →
eligible records containing 2–19 ingredients/herbs → unique compositions after
merging identical sets. The summary is saved to `work/dataset_flow.csv`.
Food eligibility also retains the Inverse Cooking instruction criteria (2–19
cleaned instructions and at least 20 characters in total), and checks ingredient
count before vocabulary mapping when building the vocabulary and after mapping
when selecting final recipes, as in the original implementation. Food metadata reports these stages
separately; the final eligible count is not an ingredient-count-only filter.
Herbal formulas are now filtered before merging; `work/herbal/unique_compositions.csv`
contains only the eligible 2–19-herb compositions. Source multiplicities remain
available as `weight` in both datasets.

## Files

```text
Colab.ipynb       Colab run
run.py            Complete analysis
data/herbal/      Five textbook CSV files
data/recipe1m/    Recipe1M files
work/             Generated tables
figures/          Figure 1
```

The generated `work/` directory can be deleted and recreated from the raw files. Data sources and required file locations are described in [data/README.md](data/README.md).

Analysis code is released under the MIT License. External datasets remain subject to their source terms.
