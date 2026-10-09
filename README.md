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
    ├── dataset.py              <- Verifies the raw data is present in `data/raw/`
    │
    ├── features.py             <- Code to create features for modeling
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