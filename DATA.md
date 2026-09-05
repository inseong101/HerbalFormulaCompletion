# Data provenance

This file separates data that can be redistributed from data that must be obtained by each user.

## Food recipes

### Source and access

The food data are from **Recipe1M**:

- Official registration and download page: <https://im2recipe.csail.mit.edu/dataset/download>
- Dataset paper: Salvador A, Hynes N, Aytar Y, Marin J, Ofli F, Weber I, Torralba A. Learning Cross-Modal Embeddings for Cooking Recipes and Food Images. *Proceedings of CVPR*. 2017:3020–3028. DOI: [10.1109/CVPR.2017.327](https://doi.org/10.1109/CVPR.2017.327)
- Paper page: <https://openaccess.thecvf.com/content_cvpr_2017/html/Salvador_Learning_Cross-Modal_Embeddings_CVPR_2017_paper.html>

Recipe1M requires registration. For that reason, its full JSON files are **not redistributed** in this repository. The repository includes their checked schema, a short record excerpt, the complete preprocessing code, and the exact aggregate output used by Figure 1.

After registration, download these two items from the Recipe1M page:

1. **Layers**: the downloaded archive is `recipe1M_layers.tar.gz`; this study uses `layer1.json` inside it.
2. **Ingredient detections**: `det_ingrs.json`.

Images, `layer2.json`, and nutritional data are not used.

The downloaded files used for the manuscript were checked as follows:

| File | Bytes | SHA-256 |
|---|---:|---|
| `det_ingrs.json` | 361,085,654 | `e1399c338b004f83f3dd9d85a8479dd04f120666e35f4d67c4b6947fae07aa50` |
| `recipe1M_layers.tar.gz` | 399,115,593 | `e180260dcd438be96e63c1c68b5f09a6415d1d1c51b1f2c2bc0b08079cb5d6c3` |
| `layer1.json` after extraction | 1,405,719,644 | — |

Different official releases should be documented if their hashes differ.

### Actual JSON structure

The two JSON files are arrays in the same recipe-ID order. The preprocessor checks the ID at every position before joining them.

`layer1.json` contains recipe information:

```json
{
  "ingredients": [{"text": "6 ounces penne"}],
  "url": "http://www.epicurious.com/...",
  "partition": "train",
  "title": "Worlds Best Mac and Cheese",
  "id": "000018c8a5",
  "instructions": [{"text": "..."}]
}
```

`det_ingrs.json` contains detected ingredient names and a validity flag for each position:

```json
{
  "valid": [true, true, true],
  "id": "000018c8a5",
  "ingredients": [
    {"text": "penne"},
    {"text": "cheese sauce"},
    {"text": "cheddar cheese"}
  ]
}
```

The actual first paired record is excerpted in [`data/raw/food_example/recipe1m_record_excerpt.json`](data/raw/food_example/recipe1m_record_excerpt.json). Instruction text was deliberately omitted from the public excerpt.

### Ingredient standardization

Ingredient processing follows the vocabulary-building procedure released with **Inverse Cooking**:

- Official code: <https://github.com/facebookresearch/inversecooking>
- Official ingredient vocabulary: <https://dl.fbaipublicfiles.com/inversecooking/ingr_vocab.pkl>
- Paper: Salvador A, Drozdzal M, Giro-i-Nieto X, Romero A. Inverse Cooking: Recipe Generation From Food Images. *Proceedings of CVPR*. 2019:10453–10462.
- Paper page: <https://openaccess.thecvf.com/content_CVPR_2019/html/Salvador_Inverse_Cooking_Recipe_Generation_From_Food_Images_CVPR_2019_paper.html>

The manuscript run produced this auditable sequence:

```text
1,029,720 paired Recipe1M records
  914,277 recipes eligible before canonical mapping
  913,680 recipes after canonical ingredient mapping
    1,486 standardized ingredients
  762,996 unique complete ingredient compositions with 2–19 ingredients
```

The mapping covered 7,870,299 of 7,879,020 valid ingredient occurrences (99.8893%). Recipes with an identical complete standardized ingredient set were collapsed, while the number of source recipes was retained as the training weight.

The code is [`raw_pipeline/food.py`](raw_pipeline/food.py). To reproduce the food preprocessing from registered files:

```bash
mkdir -p data/raw/recipe1m
tar -xzf recipe1M_layers.tar.gz -C data/raw/recipe1m layer1.json
cp det_ingrs.json data/raw/recipe1m/det_ingrs.json
curl -L https://dl.fbaipublicfiles.com/inversecooking/ingr_vocab.pkl \
  -o data/raw/recipe1m/ingr_vocab.pkl
python raw_pipeline/food.py
```

This writes the full standardized ingredient mapping, unique compositions with source multiplicity, size distribution, and metadata to `data/processed/food/`. Full preprocessing takes substantially longer than the default Figure 1 reproduction.

### What the public one-click run does

The default Colab run does not pretend to reconstruct Recipe1M without the registered JSON files. It:

1. prints the official source and required filenames;
2. prints the actual paired JSON schema and record excerpt;
3. prints the stored counts and validation metadata from the completed full-data run;
4. checks that the public food length counts sum to 762,996;
5. independently rebuilds the herbal counts from the public raw CSV files;
6. draws Figure 1 from those checked counts.

## Herbal formulas

The five complete CSV files in `data/raw/herbal/` were downloaded from the Korean Public Data Portal. The provider is the Korea Institute of Oriental Medicine, the update date is 2024-11-26, and the portal lists the permission scope as unrestricted (`이용허락범위 제한 없음`).

| File in this repository | Official dataset page |
|---|---|
| `liver.csv` | <https://www.data.go.kr/data/15075920/fileData.do> |
| `heart.csv` | <https://www.data.go.kr/data/15076000/fileData.do> |
| `spleen.csv` | <https://www.data.go.kr/data/15076001/fileData.do> |
| `lung.csv` | <https://www.data.go.kr/data/15076002/fileData.do> |
| `kidney.csv` | <https://www.data.go.kr/data/15076003/fileData.do> |

The default run processes these CSV files from the original rows rather than reading a precomputed herbal count table.
