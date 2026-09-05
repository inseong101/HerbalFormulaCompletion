# Data

## Herbal formulas

Source: [Korean Public Data Portal](https://www.data.go.kr/)

Search for `교과서 처방`. The five files in `herbal/` are the liver, heart, spleen, lung, and kidney internal-medicine textbook datasets supplied by the Korea Institute of Oriental Medicine.

Each row is one herb in one formula. The analysis uses the formula ID, Korean formula name, and Korean herb name. Dose and unit are not used.

```text
처방아이디, 처방한글명, 출전, ..., 약재한글명, ..., 용량, 단위
FO00700097, 도담탕, 得效, ..., 반하, ..., 7.5, g
```

## Food recipes

Source: [Recipe1M homepage](https://im2recipe.csail.mit.edu/) — registration required

The analysis uses:

- `det_ingrs.json`
- `layer1.json`, supplied in `recipe1M_layers.tar.gz`

Their expected layout is also documented by [Inverse Cooking](https://github.com/facebookresearch/inversecooking#data). Images and `layer2.json` are not used in this study.

`layer1.json` contains recipe titles, ingredient text, instructions, and dataset partitions. `det_ingrs.json` contains detected ingredient names and validity flags for the same recipe IDs. A checked pair of records is shown in `example.json`.

The two files are joined by recipe ID. Ingredient detections marked invalid are discarded. Ingredient names are standardized with the Inverse Cooking preprocessing procedure, repeated ingredients within a recipe are counted once, and recipes with identical complete ingredient sets are merged. The number of recipes represented by each unique set becomes its training weight.

| File | Size | SHA-256 |
|---|---:|---|
| `det_ingrs.json` | 361,085,654 bytes | `e1399c338b004f83f3dd9d85a8479dd04f120666e35f4d67c4b6947fae07aa50` |
| `recipe1M_layers.tar.gz` | 399,115,593 bytes | `e180260dcd438be96e63c1c68b5f09a6415d1d1c51b1f2c2bc0b08079cb5d6c3` |

For the public Code Availability release, obtain these files from the Recipe1M homepage and place them in `data/recipe1m/` before running `python run.py --full`.

## Results

`results.xlsx` contains the final manuscript values: dataset construction, exact length matching, structural comparison and its 100 food samples, three recommendation methods, five herbal folds, 100 matched food samples, learning curves, ingredient prevalence, matrix density, and duplicate examples.

## References

1. Salvador A, Hynes N, Aytar Y, et al. Learning Cross-Modal Embeddings for Cooking Recipes and Food Images. CVPR. 2017. doi: [10.1109/CVPR.2017.327](https://doi.org/10.1109/CVPR.2017.327).
2. Salvador A, Drozdzal M, Giro-i-Nieto X, Romero A. Inverse Cooking: Recipe Generation From Food Images. CVPR. 2019:10453–10462.
