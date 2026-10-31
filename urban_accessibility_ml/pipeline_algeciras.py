"""Algeciras: clean the raw streets (notebook 0.01) and unify them (notebook 0.05)."""

import re
import unicodedata

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from rapidfuzz import fuzz, process

from urban_accessibility_ml.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR
from urban_accessibility_ml.pipeline_common import (
    UNIFIED_DIMENSIONS,
    UNIFIED_DIVERSITY_TYPES,
    base_name,
    normalize_name,
    proposal_kind,
    street_type,
)

ALGECIRAS_STREETS_FILE = (
    RAW_DATA_DIR / "algeciras" / "260522_Tabla_resultados_Viario_Algeciras.xlsx"
)
KML_FILE = RAW_DATA_DIR / "algeciras" / "251113 PAI Algeciras.kml"
OSM_STREETS_FILE = EXTERNAL_DATA_DIR / "algeciras" / "osm_streets_by_name.gpkg"
GEO_ALGECIRAS_FILE = INTERIM_DATA_DIR / "algeciras" / "algeciras_streets_geo.geojson"
INTERIM_ALGECIRAS_FILE = INTERIM_DATA_DIR / "algeciras" / "algeciras_streets_cleaned.csv"

# Algeciras scores go from 0 to 5 and Albacete ones from 0 to 100; both are scaled to 0-1
ALGECIRAS_MAX_SCORE = 5


def section_number(name: str, reference: str) -> object:
    """Section (tramo) of an Algeciras street: the letter at the end of REFERENCIA (A=1, B=2...) or the mark of the name."""
    if letter := re.search(r"^\d+([A-Z])$", str(reference)):
        return ord(letter.group(1)) - 64
    if mark := re.search(r"\s(\d+|[A-C])$", normalize_name(name)):
        return int(mark.group(1)) if mark.group(1).isdigit() else ord(mark.group(1)) - 64
    return pd.NA


def clean_algeciras(path=ALGECIRAS_STREETS_FILE) -> pd.DataFrame:
    """Streets of Algeciras: no totals rows, numeric scores, name base and section."""
    df = pd.read_excel(path)
    # The rows without REFERENCIA are totals or empty rows, and the columns left empty only had data in them
    df = df.dropna(subset=["REFERENCIA"]).dropna(axis=1, how="all").reset_index(drop=True)
    numeric_columns = [col for col in df.columns if col.startswith("PUNT_")] + ["PRESUPUESTO"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="raise")
    df["NOMBRE"] = df["NOMBRE"].astype(str)
    df["REFERENCIA"] = df["REFERENCIA"].astype(str)
    df["NOMBRE_BASE"] = df["NOMBRE"].map(lambda name: base_name(name, r"\d+|[A-C]"))
    df["TRAMO"] = pd.Series(
        [section_number(n, r) for n, r in zip(df["NOMBRE"], df["REFERENCIA"])],
        index=df.index,
        dtype="Int64",
    )
    return df


def unify_algeciras(df: pd.DataFrame) -> pd.DataFrame:
    """Unified columns of Algeciras, with the scores scaled to 0-1."""
    df = df.copy()
    score_columns = [col for col in df.columns if col.startswith("PUNT_")]
    df[score_columns] = df[score_columns] / ALGECIRAS_MAX_SCORE

    data = {
        "ciudad": "Algeciras",
        "id_tramo": df["REFERENCIA"],
        "nombre_calle": df["NOMBRE"],
        "nombre_calle_base": df["NOMBRE_BASE"],
        "barrio": df["Barriada"],
        "tipologia_via": street_type(df["TIPOLOGÍA"]),
        "propuesta_actuacion": proposal_kind(df["PROPUESTA"]),
        "presupuesto_euros": df["PRESUPUESTO"],
        "puntuacion_total": df["PUNT_TOTAL"],
    }
    for diversity, (suffix, _, _) in UNIFIED_DIVERSITY_TYPES.items():
        data[f"puntuacion_total_{diversity}"] = df[f"PUNT_TOTAL_{suffix}"]
    for dimension, (code, _, _) in UNIFIED_DIMENSIONS.items():
        if code is None:
            continue
        data[f"puntuacion_{dimension}_global"] = df[f"PUNT_{code}_GB"]
        for diversity, (suffix, _, _) in UNIFIED_DIVERSITY_TYPES.items():
            data[f"puntuacion_{dimension}_{diversity}"] = df[f"PUNT_{code}_{suffix}"]
    data["clase_total"] = df["PIC_TOTAL"]
    for diversity, (_, _, pic) in UNIFIED_DIVERSITY_TYPES.items():
        data[f"clase_{diversity}"] = df[pic]
    return pd.DataFrame(data)


CRS = 25830  # ETRS89 / UTM 30N, in meters
# The threshold depends on the search area: lower inside the zone of the barriada, higher in the whole city
OSM_THRESHOLD_ZONE = 80
OSM_THRESHOLD_CITY = 90


def _normalize_street(name: str) -> str:
    """Lowercase, no accents, no section mark (A/B/C at the end) and no street type, to compare names."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    name = re.sub(r"\b([abc])$", "", name.strip())
    name = re.sub(r"\b(calle|avenida|avda|plaza|paseo|barriada|urbanizacion)\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _osm_streets_by_name(path=OSM_STREETS_FILE) -> gpd.GeoDataFrame:
    """OSM streets of Algeciras, one geometry per name. Downloaded once and cached in data/external."""
    if not path.exists():  # the Overpass server sometimes refuses the connection, hence the cache
        osm = ox.features_from_place("Algeciras, Cádiz, España", tags={"highway": True})
        osm = osm[osm.geom_type.isin(["LineString", "MultiLineString"])]
        osm = osm[osm["name"].notna()].to_crs(CRS)
        path.parent.mkdir(parents=True, exist_ok=True)
        osm.dissolve(by="name").reset_index()[["name", "geometry"]].to_file(path)
    return gpd.read_file(path)


def add_geometry_algeciras(
    df: pd.DataFrame, kml=KML_FILE, osm_file=OSM_STREETS_FILE
) -> gpd.GeoDataFrame:
    """Streets of Algeciras with geometry from the KML and, for the missing ones, from OSM (notebook 0.03).

    The OSM street is the closest name (fuzzy match) among the ones that touch the zone of the barriada.
    """
    # One geometry per reference; CALLES 1 has priority and CALLES 2 only fills the missing references
    calles1 = gpd.read_file(kml, layer="CALLES 1")
    calles2 = gpd.read_file(kml, layer="CALLES 2")
    geom = gpd.GeoSeries(
        calles1.set_index("REFERENCIA").geometry.combine_first(calles2.set_index("Name").geometry),
        crs=calles1.crs,
    ).force_2d().to_crs(CRS)  # fmt: skip
    geom.name = "geometry"
    streets = gpd.GeoDataFrame(
        df.merge(geom, left_on="REFERENCIA", right_index=True, how="left"),
        geometry="geometry",
        crs=CRS,
    )
    streets["GEOM_FUENTE"] = np.where(streets.geometry.notna(), "KML", None)

    # Approximate zone of each barriada, from its streets already drawn in the KML
    zones = (
        streets[streets.geometry.notna()]
        .dissolve(by="Barriada").buffer(150).convex_hull
        .rename("zone").reset_index()
    )  # fmt: skip
    osm = _osm_streets_by_name(osm_file)
    osm["key"] = osm["name"].map(_normalize_street)

    def find_osm_street(row: pd.Series) -> pd.Series:
        zone = zones.loc[zones["Barriada"] == row["Barriada"], "zone"]
        candidates = osm if zone.empty else osm[osm.intersects(zone.iloc[0])]
        match = process.extractOne(
            _normalize_street(row["NOMBRE"]), candidates["key"], scorer=fuzz.token_sort_ratio
        )
        return pd.Series(
            {
                "OSM_NOMBRE": osm.loc[match[2], "name"],
                "OSM_SCORE": match[1],
                "OSM_EN_ZONA": not zone.empty,
            }
        )

    missing = streets[streets.geometry.isna()]
    if not missing.empty:
        found = missing.apply(find_osm_street, axis=1)
        streets[["OSM_NOMBRE", "OSM_SCORE", "OSM_EN_ZONA"]] = found
        threshold = np.where(found["OSM_EN_ZONA"], OSM_THRESHOLD_ZONE, OSM_THRESHOLD_CITY)
        accepted = found[found["OSM_SCORE"] >= threshold]
        streets.loc[accepted.index, "geometry"] = gpd.GeoSeries(
            osm.set_index("name").geometry.loc[accepted["OSM_NOMBRE"]].values,
            index=accepted.index,
            crs=CRS,
        )
        streets.loc[accepted.index, "GEOM_FUENTE"] = "OSM"
    return streets
