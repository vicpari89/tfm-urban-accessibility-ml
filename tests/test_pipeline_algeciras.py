import pandas as pd
import pytest

from urban_accessibility_ml import pipeline_algeciras as alg


@pytest.fixture(scope="module")
def interim():
    if not alg.INTERIM_ALGECIRAS_FILE.exists():
        pytest.skip("interim file not generated (run `make data`)")
    return pd.read_csv(alg.INTERIM_ALGECIRAS_FILE, dtype={"REFERENCIA": str})


def test_section_number():
    assert alg.section_number("Calle Ángel", "0004B") == 2  # the letter of the reference wins
    assert alg.section_number("Calle Ángel C", "0004") == 3
    assert alg.section_number("Calle Ángel 2", "0004") == 2
    assert alg.section_number("Calle Ángel", "0004") is pd.NA


def test_clean_algeciras_removes_totals_and_adds_columns(interim):
    assert interim["REFERENCIA"].notna().all()
    assert {"NOMBRE_BASE", "TRAMO"} <= set(interim.columns)
    assert len(interim) == 506


def test_unify_algeciras_scales_scores_and_keeps_city(interim):
    unified = alg.unify_algeciras(interim)
    assert (unified["ciudad"] == "Algeciras").all()
    assert len(unified) == len(interim)
    assert unified.filter(like="puntuacion_").stack().between(0, 1).all()
    assert not any("transporte" in col for col in unified.columns)


def test_add_geometry_algeciras_uses_kml_then_osm(interim):
    if not (alg.KML_FILE.exists() and alg.OSM_STREETS_FILE.exists()):
        pytest.skip("KML or cached OSM streets not available")
    gdf = alg.add_geometry_algeciras(interim)
    assert len(gdf) == len(interim)
    assert gdf["REFERENCIA"].is_unique
    assert gdf["GEOM_FUENTE"].value_counts().to_dict() == {"KML": 285, "OSM": 150}
    assert gdf.geometry.notna().sum() == 435
