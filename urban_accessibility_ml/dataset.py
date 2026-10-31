"""Orchestrator: build the dataset from data/raw/, either with the code or running the notebooks 0.01-0.05 in order."""

import glob
from pathlib import Path
import subprocess
import sys
import tempfile

from loguru import logger
import typer

from urban_accessibility_ml.pipeline import DatasetPipeline

app = typer.Typer(help=__doc__)

NOTEBOOKS = sorted(glob.glob(str(Path(__file__).parents[1] / "notebooks" / "0.0[1-5]-*.ipynb")))


def run_notebooks() -> None:
    """Execute the notebooks in order on temporary copies, so the repository ones are not modified."""
    with tempfile.TemporaryDirectory() as out:
        for nb in NOTEBOOKS:
            logger.info(nb)
            subprocess.run(
                [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                 "--output-dir", out, nb],
                check=True,
            )  # fmt: skip


@app.command()
def main(
    notebooks: bool = typer.Option(False, help="Run the notebooks instead of the code."),
) -> None:
    """Build data/interim/ and data/processed/."""
    try:
        run_notebooks() if notebooks else DatasetPipeline().run()
    except FileNotFoundError as e:
        logger.error(e)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
