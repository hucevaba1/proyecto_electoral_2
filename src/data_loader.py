from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from typing import Iterable

FILENAME_RE = re.compile(
    r"^datosabiertos_deceyec_conteoscensales(?P<year>\d{4})_(?P<state>[a-z]{2,5})\.csv$")


def load_data(
    base_dir: str | Path,
    years: tuple[int, ...] = (2009, 2012, 2015, 2018, 2021, 2024),
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Carga la información del directorio especificado.

    Agrega columnas:
      - year (int): derivado del nombre del archivo
      - state_code (str): derivado del nombre del archivo

    """
    base_dir = Path(base_dir)
    frames: list[pd.DataFrame] = []

    for y in years:
        year_dir = base_dir / "data" / str(y)
        if not year_dir.exists():
            raise FileNotFoundError(
                f"No existe el directorio esperado: {year_dir}")

        for fp in sorted(year_dir.glob("*.csv")):
            m = FILENAME_RE.match(fp.name)
            if not m:
                raise ValueError(f"Archivo con nombre inesperado: {fp.name}")

            year = int(m.group("year"))
            state = m.group("state")

            df = pd.read_csv(fp, encoding=encoding, low_memory=False)
            df["year"] = year
            df["state_code"] = state
            frames.append(df)

    if not frames:
        raise ValueError(
            "No se cargó ningún CSV. Revisa la ruta base y la estructura de carpetas.")

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df_all = load_data(base_dir=".")
    print(df_all.shape)
    print(df_all.columns)
    print(df_all.head())


def load_state_data(
        base_dir: str | Path,
        state_code: str,
        years: tuple[int, ...] = (2009, 2012, 2015, 2018, 2021, 2024),
        encoding: str = "utf-8",
) -> pd.DataFrame:
    """Carga la información de un estado específico del directorio especificado.

    Agrega columnas:
      - year (int): derivado del nombre del archivo
      - state_code (str): derivado del nombre del archivo

    """
    base_dir = Path(base_dir)
    state_code = state_code.lower()
    frames: list[pd.DataFrame] = []

    for y in years:
        fp = (
            base_dir
            / "data"
            / str(y)
            / f"datosabiertos_deceyec_conteoscensales{y}_{state_code}.csv"
        )

        if not fp.exists():
            raise FileNotFoundError(f"No existe el archivo esperado: {fp}")

        df = pd.read_csv(fp, encoding=encoding, low_memory=False)
        df["year"] = y
        df["state_code"] = state_code
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_year_data(
    base_dir: str | Path,
    year: int = 2024,
    encoding: str = "utf-8",
    states: Iterable[str] | None = None,
    strict_states: bool = True,
) -> pd.DataFrame:
    """
    Carga todos los estados para un año específico (default: 2024).

    Parameters
    ----------
    base_dir : ruta raíz del proyecto
    year : año a cargar
    states : iterable de state_code (ej. ["ags", "cdmx"])
             Si es None, carga todos.
    strict_states : si True, lanza error si falta algún estado solicitado.
    """

    base_dir = Path(base_dir)
    year_dir = base_dir / "data" / str(year)

    if not year_dir.exists():
        raise FileNotFoundError(
            f"No existe el directorio esperado: {year_dir}")

    wanted: set[str] | None = None
    if states is not None:
        wanted = {s.lower() for s in states}

    frames: list[pd.DataFrame] = []
    loaded_states: set[str] = set()

    for fp in sorted(year_dir.glob("*.csv")):
        m = FILENAME_RE.match(fp.name)
        if not m:
            raise ValueError(f"Archivo con nombre inesperado: {fp.name}")

        year_from_name = int(m.group("year"))
        if year_from_name != year:
            raise ValueError(
                f"Archivo con año inesperado: {fp.name}. "
                f"Se esperaba año {year}."
            )

        state_code = m.group("state").lower()

        if wanted is not None and state_code not in wanted:
            continue

        df = pd.read_csv(fp, encoding=encoding, low_memory=False)
        df["year"] = year_from_name
        df["state_code"] = state_code
        frames.append(df)
        loaded_states.add(state_code)

    if wanted is not None and strict_states:
        missing = wanted - loaded_states
        if missing:
            raise FileNotFoundError(
                f"Faltan archivos para estos estados en {year_dir}: "
                f"{sorted(missing)}"
            )

    if not frames:
        raise ValueError(f"No se cargó ningún CSV en {year_dir}")

    return pd.concat(frames, ignore_index=True)