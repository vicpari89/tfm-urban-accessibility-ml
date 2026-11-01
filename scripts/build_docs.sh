#!/usr/bin/env bash
# Build the documentation site in docs/site/ from TEMPORARY copies of the notebooks.
# The notebooks in notebooks/ are not modified, and nothing is committed.
#   scripts/build_docs.sh            run the notebooks (needs data/processed/streets_unified.csv) and build
#   EXECUTE=0 scripts/build_docs.sh  build with the outputs the notebooks already have
set -euo pipefail
cd "$(dirname "$0")/.."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp -r docs/docs "$tmp/docs"
cp docs/mkdocs.yml "$tmp/mkdocs.yml"
mkdir "$tmp/docs/notebooks"

if [ "${EXECUTE:-1}" = 1 ]; then
    for nb in notebooks/0.*.ipynb; do
        echo ">>> Executing $nb"
        uv run jupyter nbconvert --to notebook --execute --output-dir "$tmp/docs/notebooks" "$nb"
    done
else
    cp notebooks/0.*.ipynb "$tmp/docs/notebooks/"
fi

uv run mkdocs build -f "$tmp/mkdocs.yml" -d "$PWD/docs/site"
echo ">>> Site in docs/site/ (open docs/site/index.html)"
