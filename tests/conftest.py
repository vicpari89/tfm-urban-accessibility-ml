import numpy as np
import pandas as pd
import pytest

from urban_accessibility_ml.config import CITIES, DIMENSIONS, PROFILES, TARGET_COLUMNS

N_STREETS_PER_CITY = 20
N_TRAMOS_PER_STREET = 3


@pytest.fixture
def city_df():
    """Two cities with 20 streets of 3 tramos each (classes C, B, A), indexed by key_tramo."""
    rows = []
    for city in CITIES:
        for street in range(N_STREETS_PER_CITY):
            for tramo, cls in enumerate(["C", "B", "A"][:N_TRAMOS_PER_STREET]):
                rows.append({
                    "ciudad": city,
                    "nombre_calle_base": f"calle {street}",
                    "key_tramo": f"{city}_{street}_{tramo}",
                    "clase": cls,
                })
    base = pd.DataFrame(rows)
    scores = np.linspace(0, 1, len(base))
    data = {f"puntuacion_{d}_{p}": scores for d in DIMENSIONS for p in PROFILES}
    data |= {col: base["clase"] for col in TARGET_COLUMNS.values()}
    out = pd.concat([base.drop(columns="clase"), pd.DataFrame(data)], axis=1)
    return out.set_index("key_tramo", drop=False)
