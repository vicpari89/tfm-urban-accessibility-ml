import pandas as pd
import pytest

from urban_accessibility_ml.config import PROCESSED_DATA_DIR

UNIFIED_FILE = PROCESSED_DATA_DIR / "streets_unified.csv"
CLASS_COLUMNS = {"clase_total": "puntuacion_total", "clase_funcional": "puntuacion_total_funcional",
                 "clase_sensorial": "puntuacion_total_sensorial", "clase_cognitiva": "puntuacion_total_cognitiva"}


@pytest.fixture(scope="module")
def df():
    if not UNIFIED_FILE.exists():
        pytest.skip("data/processed/streets_unified.csv not generated (run notebook 0.05)")
    return pd.read_csv(UNIFIED_FILE, dtype={"id_tramo": str})


def test_scores_between_0_and_1(df):
    scores = df.filter(like="puntuacion_").stack()  # stack() drops the nulls (transporte)
    assert scores.between(0, 1).all()


def test_street_and_class_counts(df):
    assert df["ciudad"].value_counts().to_dict() == {"Albacete": 916, "Algeciras": 506}
    assert not df.duplicated(["ciudad", "id_tramo"]).any()
    for class_column in CLASS_COLUMNS:
        assert set(df[class_column]) == {"A", "B", "C"}
        assert df[class_column].value_counts().sum() == len(df)
    # A, B and C are the thirds of the 0-1 score
    for class_column, score_column in CLASS_COLUMNS.items():
        assert (df.loc[df[class_column] == "A", score_column] >= 2 / 3).all()
        assert (df.loc[df[class_column] == "C", score_column] < 1 / 3).all()
