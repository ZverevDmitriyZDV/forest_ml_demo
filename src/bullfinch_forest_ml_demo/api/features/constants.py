CAT_COLS_BASE = {
    "species",
    "location_zone",
    "forest_type",
    "soil_type",
    "sensor_status",
}

NUM_COLS_HINTS = (
    "wind_exposure",
    "planting_year",
    "sap_flow_rate",
    "moisture_level",
    "temperature",
    "humidity",
    "leaf_color_index",
    "trunk_deg",
    "lag_",
    "delta_",
    "roll_",
    "_mean_",
    "_std_",
)

DERIVED_FEATURES = {
    "trunk_deg_delta_1",
    "trunk_deg_delta_7",
    "trunk_deg_roll_mean_7",
    "trunk_deg_roll_std_7",
    "trunk_deg_roll_mean_14",
    "trunk_deg_roll_std_14",
}

