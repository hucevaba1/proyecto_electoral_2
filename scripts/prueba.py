from pathlib import Path
import sys
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pipeline import load_processed_multi_year
from src.maps import (
    load_municipal_geometries,
    prepare_state_map_data,
    plot_state_choropleth,
)

YEAR = 2024
STATE_CODE_NUM = 1
STATE_NAME = "Aguascalientes"
SHP_PATH = PROJECT_ROOT / "data" / "geometrias_municipios" / "mun22cw.shp"

df = load_processed_multi_year(PROJECT_ROOT, (YEAR,))
gdf = load_municipal_geometries(SHP_PATH)

gdf_map = prepare_state_map_data(
    df=df,
    gdf=gdf,
    state_code_num=STATE_CODE_NUM,
)

fig, ax = plot_state_choropleth(
    gdf_map,
    column="sv_ratio",
    state_name="Aguascalientes",
    year=2024,
)

import matplotlib.pyplot as plt
plt.show()