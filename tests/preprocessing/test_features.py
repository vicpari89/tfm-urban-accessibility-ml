import pandas as pd
import pytest

from urban_accessibility_ml.config import DIMENSIONS, FEATURE_SETS, PROFILES, TARGET_COLUMNS
from urban_accessibility_ml.preprocessing.features import (
    check_columns,
    dimensions_for,
    encode_class,
    get_xy,
)

EXPECTED_DIMENSIONS = {
    "cartografico_aereo": ["anchura", "pavimento", "cruces", "mobiliario"],
    "calle": ["pendiente", "anchura", "cruces", "senalizacion", "mobiliario"],
    "combinado": ["pendiente", "anchura", "pavimento", "cruces", "senalizacion", "mobiliario"],
    "completo": DIMENSIONS,
}


@pytest.fixture(autouse=True)
def threshold_40(monkeypatch):
    """EXPECTED_DIMENSIONS assumes a threshold of 40, whatever config.THRESHOLD says."""
    monkeypatch.setattr("urban_accessibility_ml.preprocessing.features.THRESHOLD", 40)


@pytest.fixture
def df():
    """Three segments with every score column, the total scores and every class column."""
    data = {f"puntuacion_{d}_{p}": [1.0, 2.0, 3.0] for d in DIMENSIONS for p in PROFILES}
    data |= {"puntuacion_total": [9.0] * 3, "puntuacion_total_funcional": [9.0] * 3}
    data |= {col: ["A", "C", "B"] for col in TARGET_COLUMNS.values()}
    return pd.DataFrame(data, index=pd.Index(["t1", "t2", "t3"], name="key_tramo"))


# --- check_columns -----------------------------------------------------------------------------


def test_check_columns_ok(df):
    check_columns(df, ["puntuacion_anchura_global"])


def test_check_columns_missing(df):
    with pytest.raises(ValueError, match="not in the dataframe"):
        check_columns(df, ["puntuacion_anchura_global", "puntuacion_inexistente_global"])


@pytest.mark.parametrize("column", ["puntuacion_total", "puntuacion_total_funcional"])
def test_check_columns_total_leak(df, column):
    with pytest.raises(ValueError, match="leakage"):
        check_columns(df, [column])


# --- dimensions_for ----------------------------------------------------------------------------


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_dimensions_for(feature_set):
    assert dimensions_for(feature_set) == EXPECTED_DIMENSIONS[feature_set]


def test_dimensions_for_covers_every_feature_set():
    assert set(EXPECTED_DIMENSIONS) == set(FEATURE_SETS)


def test_dimensions_for_combinado_includes_the_other_two():
    assert set(dimensions_for("cartografico_aereo")) | set(dimensions_for("calle")) == set(
        dimensions_for("combinado")
    )


def test_dimensions_for_unknown():
    with pytest.raises(ValueError, match="Unknown feature set"):
        dimensions_for("otro")


# --- encode_class ------------------------------------------------------------------------------


def test_encode_class_order_and_index():
    out = encode_class(pd.Series(["A", "C", "B"], index=[5, 6, 7]))
    assert out.tolist() == [2, 0, 1]
    assert out.index.tolist() == [5, 6, 7]
    assert out.dtype.kind == "i"


def test_encode_class_empty():
    assert encode_class(pd.Series([], dtype=object)).empty


@pytest.mark.parametrize("bad", ["X", None])
def test_encode_class_unknown_or_missing(bad):
    with pytest.raises(ValueError, match="Unknown classes"):
        encode_class(pd.Series(["A", bad]))


# --- get_xy ------------------------------------------------------------------------------------


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
@pytest.mark.parametrize("profile", PROFILES)
def test_get_xy_one_profile(df, feature_set, profile):
    X, y = get_xy(df, feature_set, profile)
    assert list(X.columns) == [
        f"puntuacion_{d}_{profile}" for d in EXPECTED_DIMENSIONS[feature_set]
    ]
    assert y.tolist() == [2, 0, 1]
    assert X.index.equals(y.index) and X.index.equals(df.index)


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_get_xy_all_profiles(df, feature_set):
    X, _ = get_xy(df, feature_set, "global", include_all_profiles=True)
    expected = [f"puntuacion_{d}_{p}" for d in EXPECTED_DIMENSIONS[feature_set] for p in PROFILES]
    assert list(X.columns) == expected


def test_get_xy_target_follows_profile(df):
    df["clase_funcional"] = ["C", "C", "C"]
    _, y = get_xy(df, "calle", "funcional")
    assert y.tolist() == [0, 0, 0]


def test_get_xy_never_uses_total_scores(df):
    X, _ = get_xy(df, "completo", "global", include_all_profiles=True)
    assert not any(c.startswith("puntuacion_total") for c in X.columns)


def test_get_xy_unknown_feature_set(df):
    with pytest.raises(ValueError):
        get_xy(df, "otro", "global")


def test_get_xy_unknown_profile(df):
    with pytest.raises(ValueError):
        get_xy(df, "calle", "otro")


def test_get_xy_missing_column(df):
    with pytest.raises(ValueError, match="not in the dataframe"):
        get_xy(df.drop(columns="puntuacion_anchura_global"), "calle", "global")


def test_get_xy_unknown_class_in_target(df):
    df["clase_total"] = ["A", "X", "B"]
    with pytest.raises(ValueError, match="Unknown classes"):
        get_xy(df, "calle", "global")


def test_dimensions_for_threshold_is_inclusive(monkeypatch):
    monkeypatch.setattr("urban_accessibility_ml.preprocessing.features.THRESHOLD", 50)
    # cruces and senalizacion are exactly 50 in "calle": accepted; pendiente (40) is not
    assert dimensions_for("calle") == ["anchura", "cruces", "senalizacion", "mobiliario"]


# --- feature sets with the REAL values of config.py (THRESHOLD, VIABILITY) -----------------------

CONFIG_DIMENSIONS = {
    "cartografico_aereo": ["cruces", "mobiliario"],
    "calle": ["anchura", "cruces", "senalizacion", "mobiliario"],
    "combinado": ["anchura", "cruces", "senalizacion", "mobiliario"],
}


class TestFeatureSetsWithConfigThreshold:

    @pytest.fixture(autouse=True)
    def threshold_40(self):
        """Overrides the module fixture that pins the threshold: use the config value."""

    def test_expected_sets_cover_all_but_completo(self):
        assert set(CONFIG_DIMENSIONS) == set(FEATURE_SETS) - {"completo"}

    @pytest.mark.parametrize("feature_set", CONFIG_DIMENSIONS)
    def test_dimensions_for_each_set(self, feature_set):
        assert dimensions_for(feature_set) == CONFIG_DIMENSIONS[feature_set]

    def test_threshold_is_inclusive(self):
        # cruces and senalizacion are exactly 50 in "calle" and must be accepted
        assert {"cruces", "senalizacion"} <= set(dimensions_for("calle"))

    def test_sets_are_nested(self):
        assert set(dimensions_for("cartografico_aereo")) <= set(dimensions_for("calle"))
        assert set(dimensions_for("calle")) <= set(dimensions_for("combinado"))

    def test_combinado_equals_calle(self):
        assert dimensions_for("combinado") == dimensions_for("calle")
