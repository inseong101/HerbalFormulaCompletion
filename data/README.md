# Data

## Herbal formulas

The five CSV files in `herbal/` were obtained from the [Korean Public Data Portal](https://www.data.go.kr/). Search for `교과서 처방`.

Each row is one herb in one formula. The analysis uses the formula ID, Korean formula name, and Korean herb name. Dose and unit are not used.

```text
처방아이디,처방한글명,출전,...,약재한글명,...,용량,단위
FO00700097,도담탕,濟生方,...,반하,...,7.5,g
```

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

Running `python run.py` creates `work/herbal/`, `work/food/`, and the two Figure 1 files in `figures/`. These are outputs, not raw data.
