"""Helpers shared by the pipelines of both cities."""

import re
import unicodedata

import numpy as np
import pandas as pd

# Unified name: (Algeciras code, Albacete prefix, Albacete global column)
UNIFIED_DIMENSIONS = {
    "pendiente": ("PEND", "PENDIENTE", "GLOBALPEND"),
    "anchura": ("ANCH", "ANCHURA", "GLOBALANCH"),
    "pavimento": ("PAVI", "PAVIMENTO", "GLOBALPAVIM"),
    "cruces": ("CRUC", "CRUCES", "GLOBALCRUCES"),
    "senalizacion": ("SEÑA", "SEÑALIZ", "GLOBALSEÑAL"),
    "iluminacion": ("ILUM", "ILUMIN", "GLOBALILUM"),
    "mobiliario": ("MOBI", "MOBILIAR", "GLOBALMOBI"),
    "transporte": (
        None,
        "TRANSP",
        "GLOBALTRANSP",
    ),  # Only in Albacete: left out of the unified file
}
# Unified name: (Algeciras suffix, Albacete suffixes that are averaged, Algeciras class column)
UNIFIED_DIVERSITY_TYPES = {
    "funcional": ("FU", ["SR", "MR"], "PIC_FUNC"),
    "sensorial": ("SE", ["DV", "HA"], "PIC_SENS"),
    "cognitiva": ("CG", ["DI"], "PIC_COGN"),
}
STREET_TYPES = {
    "tráfico diferenciado 1 carril": "trafico_diferenciado",
    "tráfico diferenciado 2 o más carriles": "trafico_diferenciado",
    "tráfico diferenciado": "trafico_diferenciado",
    "peatonal": "peatonal",
    "peatonal residentes / emergencias": "peatonal",
    "plataforma única": "plataforma",
    "plataforma": "plataforma",
    "plaza": "plaza",
    "bulevar": "bulevar",
    "en obras parcialmente": "en_obras",
    "en obras totalmente": "en_obras",
}


def normalize_name(name: str) -> str:
    """Uppercase, no accents and no repeated spaces."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().upper()
    return re.sub(r"\s+", " ", name).strip()


def base_name(name: str, section_mark: str = r"\d+") -> str:
    """Normalized name without the section mark at the end (Albacete: 1, 2...; Algeciras: also A-C)."""
    return re.sub(rf"\s+({section_mark})$", "", normalize_name(name))


def class_by_thirds(score: pd.Series, maximum: float) -> pd.Series:
    """Class of a score: C below 1/3 of the maximum score, B between 1/3 and 2/3, and A from 2/3."""
    return pd.Series(
        np.where(score < maximum / 3, "C", np.where(score < 2 * maximum / 3, "B", "A")),
        index=score.index,
    )


def proposal_kind(proposal: pd.Series) -> pd.Series:
    """Kind of proposal, without the type of street that the original text adds."""
    lower = proposal.str.lower()
    kind = pd.Series(
        np.select(
            [
                lower.str.startswith("mejora"),
                lower.str.startswith("cambio"),
                lower.str.startswith(("sin", "en obras")),
            ],
            ["mejora_via_existente", "cambio_tipologia", "sin_propuesta"],
            default="desconocida",
        ),
        index=proposal.index,
    )
    if (kind == "desconocida").any():
        raise ValueError(f"Unknown proposals: {proposal[kind == 'desconocida'].unique()}")
    return kind


def street_type(types: pd.Series) -> pd.Series:
    """Unified type of street."""
    unified = types.str.lower().map(STREET_TYPES)
    if unified.isnull().any():
        raise ValueError(f"Unknown street types: {types[unified.isnull()].unique()}")
    return unified
