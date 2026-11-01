"""Pipeline that builds the dataset without notebooks: raw -> interim (clean) -> processed (unify)."""

from loguru import logger
import numpy as np
import pandas as pd

from urban_accessibility_ml.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from urban_accessibility_ml.pipeline_albacete import (
    ALBACETE_GEOMETRY_FILE,
    ALBACETE_STREETS_FILE,
    GEO_ALBACETE_FILE,
    INTERIM_ALBACETE_FILE,
    add_geometry_albacete,
    albacete_profiles,
    clean_albacete,
    unify_albacete,
)
from urban_accessibility_ml.pipeline_algeciras import (
    ALGECIRAS_STREETS_FILE,
    GEO_ALGECIRAS_FILE,
    INTERIM_ALGECIRAS_FILE,
    KML_FILE,
    add_geometry_algeciras,
    clean_algeciras,
    unify_algeciras,
)

ZENODO_RECORD_URL = "https://zenodo.org/records/23168998"
REQUIRED_FILES = (
    RAW_DATA_DIR / "albacete" / "albacete_1.sql",
    RAW_DATA_DIR / "albacete" / "zz_gis_e.sql",
    ALBACETE_STREETS_FILE,
    RAW_DATA_DIR / "algeciras" / "260518_Tabla_resultados_Edificios_Algeciras.xlsx",
    ALGECIRAS_STREETS_FILE,
    ALBACETE_GEOMETRY_FILE,
    KML_FILE,
)
UNIFIED_FILE = PROCESSED_DATA_DIR / "streets_unified.csv"
ALBACETE_PROFILES_FILE = PROCESSED_DATA_DIR / "albacete_profiles.csv"


def validate(unified: pd.DataFrame, profiles: pd.DataFrame) -> None:
    """Quality checks of the files to export; raise ValueError on the first one that fails."""
    checks = {
        "unique columns": unified.columns.is_unique,
        "naming rules (snake_case, ASCII, < 40 characters)": unified.columns.str.fullmatch(
            r"[a-z][a-z0-9_]{0,38}"
        ).all(),
        "no transport column": not any("transporte" in col for col in unified.columns),
        "id_tramo filled and unique per city": not unified["id_tramo"].isnull().any()
        and not unified.duplicated(["ciudad", "id_tramo"]).any(),
        "key_tramo is ciudad + id_tramo, filled and unique": unified["key_tramo"].is_unique
        and not unified["key_tramo"].isnull().any()
        and (unified["key_tramo"] == unified["ciudad"] + "_" + unified["id_tramo"]).all(),
        "scores between 0 and 1": unified.filter(like="puntuacion_").stack().between(0, 1).all(),
        "classes A, B or C": unified.filter(like="clase_").isin(["A", "B", "C"]).all().all(),
        "profiles are the streets of Albacete": profiles["id_tramo"].is_unique
        and set(profiles["id_tramo"])
        == set(unified.loc[unified["ciudad"] == "Albacete", "id_tramo"]),
        "funcional is the mean of silla_ruedas and movilidad_reducida": np.allclose(
            unified.loc[unified["ciudad"] == "Albacete"].set_index("id_tramo")[
                "puntuacion_total_funcional"
            ],
            profiles.set_index("id_tramo")[
                ["puntuacion_total_silla_ruedas", "puntuacion_total_movilidad_reducida"]
            ].mean(axis=1),
        ),
    }
    if failed := [name for name, ok in checks.items() if not ok]:
        raise ValueError(f"Quality checks failed: {failed}")


class DatasetPipeline:
    """Check the raw files, clean them into data/interim/ and unify them into data/processed/."""

    def check_raw(self) -> None:
        """Raise FileNotFoundError if any raw file is missing."""
        if missing := [f for f in REQUIRED_FILES if not f.exists()]:
            names = [str(f.relative_to(RAW_DATA_DIR)) for f in missing]
            raise FileNotFoundError(
                f"Missing raw files: {names}. Download them from {ZENODO_RECORD_URL} "
                f"and place them in {RAW_DATA_DIR}"
            )
        logger.success("Raw data present in data/raw/.")

    def clean(self) -> None:
        """Write the cleaned streets of both cities in data/interim/."""
        for clean, file in (
            (clean_algeciras, INTERIM_ALGECIRAS_FILE),
            (clean_albacete, INTERIM_ALBACETE_FILE),
        ):
            df = clean()
            file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(file, index=False)
            logger.success(
                f"{file.relative_to(INTERIM_DATA_DIR)}: {df.shape[0]} streets, {df.shape[1]} columns"
            )

    def geometry(self) -> None:
        """Write the streets of both cities with their geometry (GeoJSON) in data/interim/."""
        for add_geometry, source, file in (
            (add_geometry_algeciras, INTERIM_ALGECIRAS_FILE, GEO_ALGECIRAS_FILE),
            (add_geometry_albacete, INTERIM_ALBACETE_FILE, GEO_ALBACETE_FILE),
        ):
            df = pd.read_csv(source, dtype={"REFERENCIA": str})
            gdf = add_geometry(df)
            gdf.to_file(file, driver="GeoJSON")
            logger.success(
                f"{file.name}: {gdf.geometry.notna().sum()} of {len(gdf)} streets with geometry"
            )

    def unify(self) -> None:
        """Build and validate streets_unified.csv and albacete_profiles.csv from data/interim/."""
        algeciras = pd.read_csv(INTERIM_ALGECIRAS_FILE, dtype={"REFERENCIA": str})
        albacete = pd.read_csv(INTERIM_ALBACETE_FILE)

        unified = pd.concat(
            [unify_algeciras(algeciras), unify_albacete(albacete)], ignore_index=True
        )
        # id_tramo is only unique per city: key_tramo is unique in the whole dataset
        unified.insert(2, "key_tramo", unified["ciudad"] + "_" + unified["id_tramo"].astype(str))
        profiles = albacete_profiles(albacete)
        validate(unified, profiles)

        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        unified.to_csv(UNIFIED_FILE, index=False)
        profiles.to_csv(ALBACETE_PROFILES_FILE, index=False)
        for file, df in ((UNIFIED_FILE, unified), (ALBACETE_PROFILES_FILE, profiles)):
            logger.success(f"{file.name}: {df.shape[0]} streets, {df.shape[1]} columns")

    def run(self) -> None:
        self.check_raw()
        self.clean()
        self.geometry()
        self.unify()
