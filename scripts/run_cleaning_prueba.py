from pathlib import Path
from src.data_loader import load_state_data
from src.data_cleaning import optimize_types
from src.transformations import standardize_dimensions
from src.aggregations import aggregate_participation
from src.views import get_view

base_dir = Path(".")

df_raw = load_state_data(
    # Acotado para ags
    base_dir=base_dir, state_code="ags", years=(2018, 2021, 2024))
df = optimize_types(df_raw)

df = standardize_dimensions(df)

# Ejemplos de agregación (uno por “modo”)
by_municipio = aggregate_participation(df, "MPIONOM")
by_sexo = aggregate_participation(df, "sexo")
by_edad = aggregate_participation(df, "generacion")
by_seccion = aggregate_participation(df, "tipo_seccion")

print(by_edad.head())

view_edad = get_view(df, "generacion", include_abstentions=True)

print(view_edad.head())