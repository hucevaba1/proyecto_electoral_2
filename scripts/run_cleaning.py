from pathlib import Path
from src.data_loader import load_state_data
from src.data_cleaning import optimize_types

base_dir = Path(".")

# Carga SOLO AGS (por ahora para pruebas) y optimiza tipos
df_raw = load_state_data(
    base_dir=base_dir,
    state_code="ags",
    years=(2018, 2021, 2024)
)

print("Memoria antes:",
      df_raw.memory_usage(deep=True).sum() / (1024**3), "GB")

df = optimize_types(df_raw)

print("Memoria después:",
      df.memory_usage(deep=True).sum() / (1024**3), "GB")

print(df.shape)
print(df.dtypes)
print(df.head())