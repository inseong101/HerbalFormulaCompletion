# Data

## Herbal formulas

The five CSV files in `herbal/` were provided by the Korea Institute of Oriental Medicine through the Korean Public Data Portal. Each dataset contains formulas from a Korean medicine internal medicine textbook.

| Repository file | Official source and download page |
|---|---|
| `herbal/liver.csv` | [Liver system (간계내과학)](https://www.data.go.kr/data/15075920/fileData.do) |
| `herbal/heart.csv` | [Heart system (심계내과학)](https://www.data.go.kr/data/15076000/fileData.do) |
| `herbal/spleen.csv` | [Spleen system (비계내과학)](https://www.data.go.kr/data/15076001/fileData.do) |
| `herbal/lung.csv` | [Lung system (폐계내과학)](https://www.data.go.kr/data/15076002/fileData.do) |
| `herbal/kidney.csv` | [Kidney system (신계내과학)](https://www.data.go.kr/data/15076003/fileData.do) |

To obtain the files from the provider, open each page, choose the CSV download (`다운로드`), and save it under the corresponding filename above in `data/herbal/`. The repository includes the inputs used in this study; future provider updates may differ.

The analysis uses the formula ID (`처방아이디`), Korean formula name (`처방한글명`), and Korean herb name (`약재한글명`). Dose and unit are not used.

## Food recipes

Recipe1M is available from the [Recipe1M homepage](https://im2recipe.csail.mit.edu/) after registration. The expected layout is also documented by [Inverse Cooking](https://github.com/facebookresearch/inversecooking#data).

The analysis reads:

```text
data/recipe1m/det_ingrs.json
data/recipe1m/recipe1M_layers.tar.gz
```

`recipe1M_layers.tar.gz` contains `layer1.json`. Images and `layer2.json` are not used. The two JSON arrays are joined by recipe ID. Ingredient names marked valid in `det_ingrs.json` are standardized using the Inverse Cooking preprocessing procedure.

| File | Size | SHA-256 |
|---|---:|---|
| `det_ingrs.json` | 361,085,654 bytes | `e1399c338b004f83f3dd9d85a8479dd04f120666e35f4d67c4b6947fae07aa50` |
| `recipe1M_layers.tar.gz` | 399,115,593 bytes | `e180260dcd438be96e63c1c68b5f09a6415d1d1c51b1f2c2bc0b08079cb5d6c3` |

## Generated files

Running `python run.py` reproduces preprocessing, Table 1's recommendation example, exhaustive N−1 evaluation, frequency distributions, and the shared training randomizations for recommendation comparison and relationship maps. `Colab.ipynb` runs the same pipeline step by step from the raw inputs.

Generated audit tables and scores are saved under `work/`. The manuscript figures are:

1. `Figure1_dataset_matching`: ingredient-count distributions before matching.
2. `Figure2_recommendation_performance`: N−1 Hit@10.
3. `Figure3_ingredient_frequency`: source-record frequency distributions.
4. `Figure4_training_randomization`: original versus randomized training performance.
5. `Figure5_ingredient_relationships`: herbal and pooled-food relationship maps.

Table 1 scores are saved in `work/recommendation_example/top10.csv`. These generated files are outputs, not raw data.

## Display names

`herbal_display_names.csv` maps the 62 herbs shown in the manuscript to pharmacopoeial display names, with sources and author-confirmed qualifiers. Source data and statistical identifiers remain unchanged.
