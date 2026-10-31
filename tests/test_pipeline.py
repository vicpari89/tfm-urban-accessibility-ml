import pandas as pd
import pytest
from typer.testing import CliRunner

from urban_accessibility_ml import dataset, pipeline, pipeline_common


def test_proposal_kind_and_street_type():
    kinds = pipeline_common.proposal_kind(
        pd.Series(["Mejora de la vía", "Cambio a plaza", "Sin propuesta"])
    )
    assert kinds.tolist() == ["mejora_via_existente", "cambio_tipologia", "sin_propuesta"]
    assert pipeline_common.street_type(pd.Series(["Peatonal", "Plaza"])).tolist() == [
        "peatonal",
        "plaza",
    ]
    with pytest.raises(ValueError):
        pipeline_common.proposal_kind(pd.Series(["???"]))
    with pytest.raises(ValueError):
        pipeline_common.street_type(pd.Series(["???"]))


def test_validate_on_processed_data():
    unified = pd.read_csv(pipeline.UNIFIED_FILE, dtype={"id_tramo": str})
    profiles = pd.read_csv(pipeline.ALBACETE_PROFILES_FILE, dtype={"id_tramo": str})
    pipeline.validate(unified, profiles)
    unified.loc[0, "puntuacion_total"] = 2.0
    with pytest.raises(ValueError, match="scores between 0 and 1"):
        pipeline.validate(unified, profiles)


def test_missing_raw_files(monkeypatch):
    monkeypatch.setattr(pipeline, "REQUIRED_FILES", (pipeline.RAW_DATA_DIR / "nope.xlsx",))
    with pytest.raises(FileNotFoundError, match="nope.xlsx"):
        pipeline.DatasetPipeline().check_raw()
    # the orchestrator turns it into exit code 1
    result = CliRunner().invoke(dataset.app, [])
    assert result.exit_code == 1


def test_orchestrator_runs_notebooks_flag(monkeypatch):
    called = []
    monkeypatch.setattr(dataset, "run_notebooks", lambda: called.append(1))
    assert CliRunner().invoke(dataset.app, ["--notebooks"]).exit_code == 0
    assert called == [1]
