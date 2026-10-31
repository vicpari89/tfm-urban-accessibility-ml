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
data/raw ──(make data: dataset.py)──> data/interim ──(pipeline.py)──> data/processed
```

| Command | What it does |
|---|---|
| `make data` | Checks the raw files, cleans, unifies and validates them (`DatasetPipeline` in `urban_accessibility_ml/pipeline.py`, launched by `urban_accessibility_ml/dataset.py`). |
| `make unified` | Same result, running the notebooks `0.01` to `0.05` in order (they run on temporary copies, so the notebooks in the repository are not modified). |
| `make test` | Unit tests, and quality checks of `data/processed/streets_unified.csv`. |

Both routes give identical files. They write:

* `data/interim/{algeciras,albacete}/*_streets_cleaned.csv`: the cleaned streets of each city.
* `data/processed/streets_unified.csv`: one row per street section (`ciudad` + `id_tramo`) with the same columns for both cities: `barrio`, street data, scores scaled to 0-1 (total, per dimension and per profile: functional, sensory and cognitive) and the classes A, B and C (total and per profile).
* `data/processed/albacete_profiles.csv`: the five original profiles of Albacete (wheelchair, reduced mobility, visual, hearing and cognitive), joined by `id_tramo`.

The pipeline also writes the streets with geometry to `data/interim/<city>/<city>_streets_geo.geojson` (Algeciras needs the cached OSM file `data/external/algeciras/osm_streets_by_name.gpkg`, or internet to download it). The maps of notebooks `0.03` and `0.04` are not part of the pipeline. Run the commands with `uv run make ...` if the virtual environment is not activated.

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
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         urban_accessibility_ml and configuration for tools like ruff
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── uv.lock            <- Locked dependency versions for reproducible environments
│
└── urban_accessibility_ml   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes urban_accessibility_ml a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Orchestrator: runs `DatasetPipeline`, or notebooks 0.01-0.05 with `--notebooks`
    │
    ├── pipeline.py             <- `DatasetPipeline`: checks, cleans and unifies the data into `data/processed/`
    ├── pipeline_algeciras.py   <- Algeciras-specific cleaning, geometry (KML + OSM) and unification
    ├── pipeline_albacete.py    <- Albacete-specific cleaning, geometry and unification
    ├── pipeline_common.py      <- Helpers shared by both cities
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
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
   uv run python -m urban_accessibility_ml.dataset
   ```

--------