"""Albacete: clean the raw streets (notebooks 0.02 and 0.04) and unify them (notebook 0.05)."""

import re

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

from urban_accessibility_ml.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from urban_accessibility_ml.pipeline_common import (
    UNIFIED_DIMENSIONS,
    UNIFIED_DIVERSITY_TYPES,
    base_name,
    class_by_thirds,
    normalize_name,
    proposal_kind,
    street_type,
)

ALBACETE_STREETS_FILE = RAW_DATA_DIR / "albacete" / "zz_gis_v.sql"
ALBACETE_GEOMETRY_FILE = RAW_DATA_DIR / "albacete" / "albacete_viario_accesibilidad.json"
GEO_ALBACETE_FILE = INTERIM_DATA_DIR / "albacete" / "albacete_streets_geo.geojson"
INTERIM_ALBACETE_FILE = INTERIM_DATA_DIR / "albacete" / "albacete_streets_cleaned.csv"

# Albacete scores: dimension -> its GLOBAL<DIMENSION> column. Each one has a score per diversity type
ALBACETE_DIMENSIONS = {
    "PENDIENTE": "GLOBALPEND",
    "ANCHURA": "GLOBALANCH",
    "PAVIMENTO": "GLOBALPAVIM",
    "CRUCES": "GLOBALCRUCES",
    "SEÑALIZ": "GLOBALSEÑAL",
    "ILUMIN": "GLOBALILUM",
    "MOBILIAR": "GLOBALMOBI",
    "TRANSP": "GLOBALTRANSP",
}
ALBACETE_DIVERSITY_SUFFIXES = ["SR", "MR", "DV", "HA", "DI"]
# The letters of REFER identify the zone of the street, and each zone is a barrio (X and Y have none)
ZONA_TO_BARRIO = {
    "CA": "Carretas-Huerta Marzo", "CÑ": "Cañicas-Canal de María Cristina", "ES": "La Estrella",
    "FE": "Feria", "FR": "Franciscanos", "FT": "Fátima", "HF": "Hermanos Falcó", "HS": "Hospital",
    "ID": "Industria", "LL": "Los Llanos", "MD": "Medicina", "MI": "La Milagrosa", "PA": "La Pajarita",
    "PL": "El Pilar", "PM": "Pedro Lamata", "PQ": "Parque Sur", "PS": "Polígono de San Antón",
    "SA": "San Antonio Abad", "SE": "Sepulcro-Bolera", "SM": "San Pedro-Mortero", "SP": "San Pablo",
    "ST": "Santa Teresa", "UV": "Universidad", "VC": "Villacerrada-Centro", "VE": "Vereda",
}  # fmt: skip
ALBACETE_MAX_SCORE = 100
ALBACETE_PROFILES = {
    "SR": "silla_ruedas",
    "MR": "movilidad_reducida",
    "DV": "visual",
    "HA": "hipoacusia",
    "DI": "cognitiva",
}


def _mysql_dump_to_duckdb(text: str) -> str:
    """Convert a MySQL/MariaDB dump into SQL that DuckDB can run (CREATE TABLE and INSERT only)."""
    text = text.replace("`", '"')
    text = re.sub(r"\bint\(\d+\)", "INTEGER", text)
    text = re.sub(r"\bvarchar\(\d+\)", "VARCHAR", text)
    text = re.sub(r"\)\s*ENGINE=[^;]*;", ");", text)
    # Drop comments and session/transaction statements
    return re.sub(
        r"^(--.*|/\*!.*|SET .*|START TRANSACTION;|COMMIT;)$", "", text, flags=re.MULTILINE
    )


def clean_albacete(path=ALBACETE_STREETS_FILE) -> pd.DataFrame:
    """Streets of Albacete: no photo links, scores capped to 100 with the GLOBAL means recalculated, and barrio."""
    con = duckdb.connect()  # in-memory
    con.execute('CREATE SCHEMA "zz_gis_v"')
    con.execute("SET schema = 'zz_gis_v'")
    con.execute(_mysql_dump_to_duckdb(path.read_text(encoding="utf-8")))
    df = con.sql("SELECT * FROM zz_gis_v").df()

    # The columns starting with F are photo and Google Drive links; all the streets are from Albacete
    df = df.drop(columns=[col for col in df.columns if col.startswith("F")] + ["MUNICIPIO"])
    df["ZONA"] = df["REFER"].str.extract(r"^([A-ZÑ]+)")[0]
    df["NOMBRE_BASE"] = df["NOMBRE"].map(base_name)
    df["TRAMO"] = df["NOMBRE"].map(normalize_name).str.extract(r"\s(\d+)$")[0].astype("Int64")

    # Scores above 100 (5 values) are capped, and the means checked in notebook 0.02 are recalculated
    score_columns = [f"{d}{s}" for d in ALBACETE_DIMENSIONS for s in ALBACETE_DIVERSITY_SUFFIXES]
    df[score_columns] = df[score_columns].clip(upper=100)
    for dimension, global_column in ALBACETE_DIMENSIONS.items():
        columns = [f"{dimension}{s}" for s in ALBACETE_DIVERSITY_SUFFIXES]
        df[global_column] = df[columns].mean(axis=1).round().astype(int)
    for suffix in ALBACETE_DIVERSITY_SUFFIXES:
        scores = df[[f"{d}{suffix}" for d in ALBACETE_DIMENSIONS]]
        df[f"GLOBAL{suffix}"] = (
            scores.where(scores > 0).mean(axis=1).round().astype(int)
        )  # TRANSP 0 = no data
    df["ACCGLOBAL"] = (
        df[[f"GLOBAL{s}" for s in ALBACETE_DIVERSITY_SUFFIXES]].mean(axis=1).round().astype(int)
    )

    df["BARRIO"] = df["ZONA"].map(ZONA_TO_BARRIO).fillna(df["ZONA"])  # X and Y keep their letter
    return df


def scale_albacete(df: pd.DataFrame) -> pd.DataFrame:
    """Albacete scores scaled to 0-1; the TRANSP zeros mean there is no data, so they become nulls."""
    df = df.copy()
    score_columns = ["ACCGLOBAL"] + [col for col in df.columns if col.startswith("GLOBAL")]
    for _, prefix, _ in UNIFIED_DIMENSIONS.values():
        score_columns += [f"{prefix}{s}" for s in ALBACETE_PROFILES]
    score_columns = list(dict.fromkeys(score_columns))
    df[score_columns] = df[score_columns] / ALBACETE_MAX_SCORE
    transport_columns = [
        col for col in df.columns if col.startswith("TRANSP") or col == "GLOBALTRANSP"
    ]
    df[transport_columns] = df[transport_columns].replace(0, np.nan)
    return df


def unify_albacete(df: pd.DataFrame) -> pd.DataFrame:
    """Unified columns of Albacete, with the scores scaled to 0-1."""
    scaled = scale_albacete(df)
    data = {
        "ciudad": "Albacete",
        "id_tramo": scaled["REFER"],
        "nombre_calle": scaled["NOMBRE"],
        "nombre_calle_base": scaled["NOMBRE_BASE"],
        "barrio": scaled["BARRIO"],
        "tipologia_via": street_type(scaled["TIPOLACT"]),
        "propuesta_actuacion": proposal_kind(scaled["PROPACT"]),
        "presupuesto_euros": scaled["PRESUP"],
        "puntuacion_total": scaled["ACCGLOBAL"],
    }
    for diversity, (_, suffixes, _) in UNIFIED_DIVERSITY_TYPES.items():
        data[f"puntuacion_total_{diversity}"] = scaled[[f"GLOBAL{s}" for s in suffixes]].mean(
            axis=1
        )
    for dimension, (code, prefix, global_column) in UNIFIED_DIMENSIONS.items():
        if code is None:
            continue
        data[f"puntuacion_{dimension}_global"] = scaled[global_column]
        for diversity, (_, suffixes, _) in UNIFIED_DIVERSITY_TYPES.items():
            data[f"puntuacion_{dimension}_{diversity}"] = scaled[
                [f"{prefix}{s}" for s in suffixes]
            ].mean(axis=1)
    data["clase_total"] = class_by_thirds(df["ACCGLOBAL"], ALBACETE_MAX_SCORE)
    for diversity in UNIFIED_DIVERSITY_TYPES:
        data[f"clase_{diversity}"] = class_by_thirds(data[f"puntuacion_total_{diversity}"], 1)
    return pd.DataFrame(data)


def albacete_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Original five profiles of Albacete (scores scaled to 0-1), with the unified naming."""
    scaled = scale_albacete(df)
    data = {"id_tramo": scaled["REFER"]}
    for suffix, profile in ALBACETE_PROFILES.items():
        data[f"puntuacion_total_{profile}"] = scaled[f"GLOBAL{suffix}"]
        for dimension, (_, prefix, _) in UNIFIED_DIMENSIONS.items():
            data[f"puntuacion_{dimension}_{profile}"] = scaled[f"{prefix}{suffix}"]
        data[f"clase_{profile}"] = class_by_thirds(scaled[f"GLOBAL{suffix}"], 1)
    return pd.DataFrame(data)


def add_geometry_albacete(df: pd.DataFrame, path=ALBACETE_GEOMETRY_FILE) -> gpd.GeoDataFrame:
    """Streets of Albacete with their geometry (notebook 0.04); streets without it keep a null geometry."""
    raw = gpd.read_file(path)
    # One geometry per street: the lines that share the same reference are merged (MultiLineString)
    geometry = raw.dissolve(by="referencia")[["geometry"]].reset_index()
    merged = df.merge(geometry, left_on="REFER", right_on="referencia", how="left")
    return gpd.GeoDataFrame(merged.drop(columns="referencia"), geometry="geometry", crs=raw.crs)
