"""Verify that the raw data is present in data/raw/."""

from loguru import logger
import typer

from urban_accessibility_ml.config import RAW_DATA_DIR

app = typer.Typer(help=__doc__)

ZENODO_RECORD_URL = "https://zenodo.org/records/23168998"

REQUIRED_FILES = (
    RAW_DATA_DIR / "albacete" / "albacete_1.sql",
    RAW_DATA_DIR / "albacete" / "zz_gis_e.sql",
    RAW_DATA_DIR / "albacete" / "zz_gis_v.sql",
    RAW_DATA_DIR / "algeciras" / "260518_Tabla_resultados_Edificios_Algeciras.xlsx",
    RAW_DATA_DIR / "algeciras" / "260522_Tabla_resultados_Viario_Algeciras.xlsx",
)


@app.command()
def main() -> None:
    """Check the raw files exist; otherwise point to the Zenodo record."""
    if missing := [f for f in REQUIRED_FILES if not f.exists()]:
        logger.error(f"Missing raw files: {[str(f.relative_to(RAW_DATA_DIR)) for f in missing]}")
        logger.info(f"Download them from {ZENODO_RECORD_URL} and place them in {RAW_DATA_DIR}")
        raise typer.Exit(code=1)
    logger.success("Raw data present in data/raw/. Ready to continue.")


if __name__ == "__main__":
    app()
