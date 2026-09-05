# Herbal Formula Completion

Code and data for comparing ingredient combinations and Top-10 ingredient recommendation in food recipes and Korean medicine herbal formulas.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inseong101/HerbalFormulaCompletion/blob/main/Colab.ipynb)

## Run

The notebook prints the raw-data fields, preprocessing counts, duplicate examples, model settings, and output paths while the analysis runs.

```bash
pip install -r requirements.txt
python run.py --full
```

The full run performs Recipe1M preprocessing, exact ingredient-count matching, 100 food samples, five-fold recommendation evaluation, structural comparison, and the learning-curve analysis. Progress for the 100 samples is shown with `tqdm`.

Use `python run.py` to rebuild the manuscript figures quickly from `data/results.xlsx` without repeating the full Recipe1M analysis.

## Data

The five herbal-formula CSV files are in `data/herbal/`. They were obtained from the [Korean Public Data Portal](https://www.data.go.kr/) by searching for `교과서 처방`.

Recipe1M is available from the [Recipe1M homepage](https://im2recipe.csail.mit.edu/) after registration. The analysis uses `det_ingrs.json` and `layer1.json`; the latter is stored inside `recipe1M_layers.tar.gz` in the working repository. See [data/README.md](data/README.md) for the input structure and checksums.

## Files

```text
Colab.ipynb        Colab run
run.py             analysis entry point
src/               preprocessing, evaluation, and figures
data/herbal/       textbook CSV files
data/recipe1m/     Recipe1M text files
data/results.xlsx  manuscript results
figures/           manuscript figures
```

Analysis code is released under the MIT License. The external datasets remain subject to their source terms.
