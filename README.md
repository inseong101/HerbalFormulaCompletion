# Food Recipes and Herbal Formulas

Code for preprocessing the two datasets and producing Figure 1 of the manuscript.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inseong101/HerbalFormulaCompletion/blob/main/Colab.ipynb)

## Run

```bash
pip install -r requirements.txt
python run.py
```

The terminal shows the raw fields, one raw record, preprocessing counts, one duplicate-composition example from each domain, the ingredient-count table, and the saved Figure 1 paths.

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
