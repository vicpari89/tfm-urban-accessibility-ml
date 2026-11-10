#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = temp_proj
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset: clean data/raw into data/interim and unify it into data/processed
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m urban_accessibility_ml.data.dataset


## Same as `make data`, but running the notebooks 0.01-0.06 in order
.PHONY: unified
unified: requirements
	$(PYTHON_INTERPRETER) -m urban_accessibility_ml.data.dataset --notebooks


## Tune the hyperparameters in the 17 partitions and the feature sets (resumable) and write reports/results/best_params.json
.PHONY: tune
tune: requirements
	$(PYTHON_INTERPRETER) -m urban_accessibility_ml.modeling.tuning.tuning_run


## Tune the tree models with another seed and with 100 candidates (about an hour) and run the matrix for each; results in reports/results/variants/
.PHONY: tune-variants
tune-variants: requirements
	$(PYTHON_INTERPRETER) -m urban_accessibility_ml.modeling.tuning.tuning_variants


## Run the experiment matrix with the default and the tuned hyperparameters and write the 2 CSV in reports/results/
.PHONY: experiments
experiments: requirements
	$(PYTHON_INTERPRETER) -m urban_accessibility_ml.modeling.experiments_run


## Run the matrix for the feature sets of every viability threshold (sensitivity analysis) and write reports/results/sensitivity.csv
.PHONY: sensitivity
sensitivity: requirements
	$(PYTHON_INTERPRETER) -m urban_accessibility_ml.modeling.sensitivity_run

## Build the documentation site in docs/site/ (runs the notebooks; EXECUTE=0 skips it)
.PHONY: docs
docs:
	bash scripts/build_docs.sh


#################################################################################
# Self Documenting Commands                                                    #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
