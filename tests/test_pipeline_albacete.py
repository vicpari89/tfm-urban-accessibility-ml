import numpy as np
import pandas as pd
import pytest

from urban_accessibility_ml import pipeline_albacete as alb


@pytest.fixture(scope="module")
def interim():
    if not alb.INTERIM_ALBACETE_FILE.exists():
        pytest.skip("interim file not generated (run `make data`)")
    return pd.read_csv(alb.INTERIM_ALBACETE_FILE)


def test_mysql_dump_to_duckdb_drops_mysql_only_syntax():
    dump = "-- comment\nSET NAMES utf8;\nCREATE TABLE `t` (`a` int(11), `b` varchar(20)) ENGINE=InnoDB DEFAULT CHARSET=utf8;\nCOMMIT;"
    sql = alb._mysql_dump_to_duckdb(dump)
    assert 'CREATE TABLE "t" ("a" INTEGER, "b" VARCHAR);' in sql
    assert "SET" not in sql and "ENGINE" not in sql and "COMMIT" not in sql


def test_scale_albacete_scales_and_nulls_transport_zeros(interim):
    df = interim.head(3).copy()
    df["ACCGLOBAL"], df["GLOBALTRANSP"], df["TRANSPSR"] = 50, 0, 0
    scaled = alb.scale_albacete(df)
    assert (scaled["ACCGLOBAL"] == 0.5).all()
    assert scaled["GLOBALTRANSP"].isna().all() and scaled["TRANSPSR"].isna().all()


def test_clean_albacete_caps_scores_and_adds_barrio(interim):
    score_columns = [
        f"{d}{s}" for d in alb.ALBACETE_DIMENSIONS for s in alb.ALBACETE_DIVERSITY_SUFFIXES
    ]
    assert (interim[score_columns] <= 100).all().all()
    assert interim["BARRIO"].notna().all()
    assert len(interim) == 916


def test_unify_albacete_scales_scores_and_classes(interim):
    unified = alb.unify_albacete(interim)
    assert (unified["ciudad"] == "Albacete").all()
    assert unified.filter(like="puntuacion_").stack().between(0, 1).all()
    assert set(unified["clase_total"]) <= {"A", "B", "C"}


def test_albacete_profiles_has_five_profiles_per_street(interim):
    profiles = alb.albacete_profiles(interim)
    assert profiles["id_tramo"].is_unique and len(profiles) == len(interim)
    for profile in alb.ALBACETE_PROFILES.values():
        assert f"puntuacion_total_{profile}" in profiles.columns
        assert f"clase_{profile}" in profiles.columns


def test_add_geometry_albacete_keeps_every_street(interim):
    if not alb.ALBACETE_GEOMETRY_FILE.exists():
        pytest.skip("raw geometry file not available")
    gdf = alb.add_geometry_albacete(interim)
    assert len(gdf) == len(interim)
    assert gdf["REFER"].is_unique
    assert gdf.geometry.isna().sum() == 12  # streets without geometry in the raw file
