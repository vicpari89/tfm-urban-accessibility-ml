# Urban Accessibility ML

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project aims to design and validate a Machine Learning model to classify urban street accessibility using field data from two Spanish cities (Albacete and Algeciras).

## Project Generation & Configuration

This repository structure was generated using [Cookiecutter Data Science v2 (CCDS)](https://cookiecutter-data-science.drivendata.org/) via `uvx`.

### Generation Command

`uvx --from cookiecutter-data-science ccds`

### Cookiecutter Options & Architectural Choices

* **project_name:** `Urban Accessibility ML` — Human-readable title of the thesis project.
* **repository_name:** `urban_accessibility_ml` — Root repository directory name.
* **module_name:** `urban_accessibility_ml` — Python package name located inside `urban_accessibility_ml/`.
* **author_name:** `Víctor Pariente González` — Author and maintainer.
* **environment_manager:** `uv` — Modern, Rust-based, fast package and virtual environment manager.
* **dependency_file:** `pyproject.toml` — Standard single-file project metadata and dependency specification.
* **pydata_packages:** `none` — Clean slate; dependencies added modularly via `uv add`.
* **testing_framework:** `pytest` — Standard Python testing framework using native `assert` statements and fixtures.
* **linting_and_formatting:** `ruff` — Ultra-fast linter and formatter replacing `flake8`, `black`, and `isort`.
* **open_source_license:** `MIT` — Permissive open-source license.
* **docs:** `mkdocs` — Markdown-based documentation framework with `mkdocs-material` support.
* **include_code_scaffold:** `Yes` — Includes initial boilerplates for modules (`dataset.py`, `features.py`, etc.) and basic test structures.

## Reproducing the unified dataset

The raw data (ACCESIA, Albacete and Algeciras) is not in the repository: download it from [Zenodo](https://zenodo.org/records/23168998) and place it in `data/raw/albacete/` and `data/raw/algeciras/`.

```
data/raw ──(make data: data/dataset.py)──> data/interim ──(data/pipeline.py)──> data/processed
```

| Command | What it does |
|---|---|
| `make data` | Checks the raw files, cleans, unifies and validates them (`DatasetPipeline` in `urban_accessibility_ml/data/pipeline.py`, launched by `urban_accessibility_ml/data/dataset.py`). |
| `make unified` | Same result, running the notebooks `0.01` to `0.06` in order (they run on temporary copies, so the notebooks in the repository are not modified). |
| `make test` | Unit tests, and quality checks of `data/processed/streets_unified.csv`. |

Both routes give identical files. They write:

* `data/interim/{algeciras,albacete}/*_streets_cleaned.csv`: the cleaned streets of each city.
* `data/processed/streets_unified.csv`: one row per street section (`ciudad` + `id_tramo`) with the same columns for both cities: `barrio`, street data, scores scaled to 0-1 (total, per dimension and per profile: functional, sensory and cognitive) and the classes A, B and C (total and per profile).
* `data/processed/albacete_profiles.csv`: the five original profiles of Albacete (wheelchair, reduced mobility, visual, hearing and cognitive), joined by `id_tramo`.

The pipeline also writes the streets with geometry to `data/interim/<city>/<city>_streets_geo.geojson` (Algeciras needs the cached OSM file `data/external/algeciras/osm_streets_by_name.gpkg`, or internet to download it). The maps of notebooks `0.03` and `0.04` are not part of the pipeline. Run the commands with `uv run make ...` if the virtual environment is not activated.

## Models, hyperparameter tuning and experiments

The models, the tuning and the experiment matrix are in `urban_accessibility_ml/modeling/`, and the 17 train/test partitions (grouped by street) in `urban_accessibility_ml/preprocessing/split.py`. They need `data/processed/streets_unified.csv` (see above).

| Command | What it does |
|---|---|
| `make tune` | Tunes the hyperparameters of logistic regression, random forest, XGBoost and LightGBM in each of the 17 partitions and each feature set (`calle` and `combinado` share it), using only their training rows. It saves each result when it ends and continues a stopped run. Writes `reports/results/best_params.json` and `reports/results/tuning-candidates.csv`. |
| `make tune-variants` | Repeats the tuning of the three tree models with another seed and with 100 candidates, and runs the experiment matrix for each, to see how much the tuning depends on chance (about an hour). Continues a stopped run. Writes `best_params_*.json`, `tuning-candidates_*.csv` and `experiment-matrix_*.csv` in `reports/results/variants/`. |
| `make experiments` | Runs the experiment matrix (5 models × 4 feature sets × 4 profiles × 17 partitions) twice, with the default and with the tuned hyperparameters (about 8 minutes). Writes `reports/results/experiment-matrix-default.csv` and `reports/results/experiment-matrix-tuned.csv`. |

`make experiments` stops before running if its result files already exist; to write over them, run the module with `--overwrite`, for example `uv run python -m urban_accessibility_ml.modeling.experiments_run --overwrite`. `make tune` skips what is already in `best_params.json`; run `uv run python -m urban_accessibility_ml.modeling.tuning.tuning_run` with `--overwrite` to start from scratch, `--n-jobs` to set the parallel fits (half of the cores by default) or `--dry-run` to only estimate the time left. `make tune-variants` accepts the same options, and `--variant` to run only one of them. The results are committed, so the notebooks below can read them without running anything long.

| Notebook | Content |
|---|---|
| `0.07` | The train/test partitions. |
| `0.08` | Baseline models. |
| `0.09` | Random forest, XGBoost and LightGBM. |
| `0.10` | Experiment matrix with the default hyperparameters. |
| `0.11` | Hyperparameter tuning: search spaces, procedure and chosen values. It reads the results of `make tune`. |
| `0.12` | Default against tuned hyperparameters in the test set. It reads the results of `make experiments`. |
| `0.13` | How much the tuning of the tree models depends on the seed and on the number of candidates. It reads the results of `make tune-variants`. |

## Documentation

The notebooks are published as a web page (mkdocs-material + mkdocs-jupyter, config in `docs/mkdocs.yml`).

| Command | What it does |
|---|---|
| `make docs` | Runs the notebooks on temporary copies and builds the site in `docs/site/` (ignored by git). It needs `data/processed/streets_unified.csv` (`python -m urban_accessibility_ml.data.dataset`, see above), and internet for notebook `0.03`. The notebooks in `notebooks/` are not modified. |
| `EXECUTE=0 make docs` | Same, without running the notebooks (uses the outputs they already have). |

It is published automatically in GitHub Pages by `.github/workflows/docs.yml` when a tag is pushed (`git tag v0.1 && git push origin v0.1`), or by hand in the Actions tab. The workflow downloads the raw data from Zenodo (`scripts/download_raw.py`), builds the dataset, runs the notebooks and builds the site. **The executed notebooks are not saved in the repository**: the notebooks are committed without outputs (`nbstripout`).

One-time setup:

* In every clone, run `uv run nbstripout --install` (the git filter that removes the outputs when committing; your working copy keeps them). The `*.ipynb` rule is in `.gitattributes`.
* Zenodo: create a personal access token (Settings → Applications) and save it in the repository as the secret `ZENODO_TOKEN` (Settings → Secrets and variables → Actions).
* GitHub: Settings → Pages → Source: **GitHub Actions**. (If you restrict the deployment branches and tags of the `github-pages` environment, allow the tags `*`.)

To preview it in local: `make docs` (runs the notebooks on temporary copies; needs `data/processed/streets_unified.csv`), or `EXECUTE=0 make docs` (uses the outputs of your working copy, without running them).

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks, numbered in order (`0.01` to `0.13`) with a short
│                         `-` delimited description, e.g. `0.11-hyperparameter-tuning`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         urban_accessibility_ml and configuration for tools like ruff
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── results        <- Results of the models: tuning (`best_params.json`, `tuning-candidates.csv`),
│                         the matrices of `make experiments` (`experiment-matrix-*.csv`), the variants of the tuning
│                         (`variants/`) and the CSV
│                         that notebooks 0.08 to 0.10 write
│
├── scripts            <- `build_docs.sh` (builds the documentation) and `download_raw.py` (raw data from Zenodo)
│
├── tests              <- Unit tests, with one folder per package: `data`, `preprocessing` and `modeling`
│
├── uv.lock            <- Locked dependency versions for reproducible environments
│
└── urban_accessibility_ml   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes urban_accessibility_ml a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── data                    <- Raw data -> `data/processed/streets_unified.csv`
    │   ├── dataset.py              <- Orchestrator: runs `DatasetPipeline`, or notebooks 0.01-0.06 with `--notebooks`
    │   ├── pipeline.py             <- `DatasetPipeline`: checks, cleans and unifies the data
    │   ├── pipeline_algeciras.py   <- Algeciras-specific cleaning, geometry (KML + OSM) and unification
    │   ├── pipeline_albacete.py    <- Albacete-specific cleaning, geometry and unification
    │   └── pipeline_common.py      <- Helpers shared by both cities
    │
    ├── preprocessing           <- From the unified data to the model inputs
    │   ├── features.py             <- Feature sets and profiles (`get_xy`)
    │   └── split.py                <- The 17 train/test partitions, grouped by street (`build_all_splits`)
    │
    ├── modeling                <- Models, experiments and tuning
    │   ├── models                  <- One builder per model (`model_*.py`) and `build_model`
    │   ├── train.py                <- `run_one`: train and evaluate one model in one partition
    │   ├── evaluate.py             <- Metrics (F1 macro, balanced accuracy, weighted kappa)
    │   ├── experiments.py          <- `run_experiments`: the whole matrix, and `save_results`
    │   ├── experiments_run.py      <- `make experiments`: the matrix with default and tuned hyperparameters
    │   ├── predict.py              <- Code to run model inference with trained models
    │   └── tuning                  <- Hyperparameter tuning, launched with `make tune` (`tuning_run.py`) and its variants (`tuning_variants.py`, `make tune-variants`)
    │
    └── plotting                <- Visual system of the thesis
        ├── palettes.py             <- Colour palettes
        ├── plots.py                <- Code to create visualizations (`use_style`, `save_figure`)
        └── tfm.mplstyle            <- matplotlib style
```

--------

## 📊 Data Access & Reproducibility

The raw datasets for Albacete (`.sql` scripts) and Algeciras (`.xlsx` files) are archived on Zenodo under Restricted Access:

* **Zenodo Record:** [https://zenodo.org/records/23168998](https://zenodo.org/records/23168998)
* **Access Rights:** Restricted to academic evaluation for UOC Master's Thesis.

### Local Setup Instructions

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Request access to the Zenodo record ([https://zenodo.org/records/23168998](https://zenodo.org/records/23168998)) and download the files manually.
3. Place them in the following directories:
   * Albacete SQL scripts (`albacete_1.sql`, `zz_gis_e.sql`, `zz_gis_v.sql`) -> `data/raw/albacete/`
   * Algeciras Excel files (`260518_Tabla_resultados_Edificios_Algeciras.xlsx`, `260522_Tabla_resultados_Viario_Algeciras.xlsx`) -> `data/raw/algeciras/`
4. Verify that all files are in place:
   ```bash
   uv run python -m urban_accessibility_ml.data.dataset
   ```

--------