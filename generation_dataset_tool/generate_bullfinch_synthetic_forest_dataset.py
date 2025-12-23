#!/usr/bin/env python3
"""
Bullfinch Earth — Synthetic Forestry Dataset Generator (daily)

Generates:
- trees_level1.csv (static attributes)

- trees_daily_dataset.csv (daily readings + labels)

Inputs:
- Bullfinch_Synthetic_Forestry_Dataset_Spec.xlsx

Usage:

  python generate_bullfinch_synthetic_forest_dataset.py \

    --spec Bullfinch_Synthetic_Forestry_Dataset_Spec.xlsx \

    --out_dir .

"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spec",
        type=str,
        default=str(Path(__file__).resolve().parent / "Bullfinch_Synthetic_Forestry_Dataset_Spec.xlsx"),
    )

    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "raw"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = pd.read_excel(args.spec, sheet_name='schema')
    gparams = pd.read_excel(args.spec, sheet_name='global_params')
    species_params = pd.read_excel(args.spec, sheet_name='species_params')
    zone_params = pd.read_excel(args.spec, sheet_name='zone_params')
    soil_params = pd.read_excel(args.spec, sheet_name='soil_params')
    health_effects = pd.read_excel(args.spec, sheet_name='health_effects')

    gp = dict(zip(gparams['param'], gparams['value']))
    seed = int(gp['seed'])
    rng = np.random.default_rng(seed)

    n_trees = int(gp['n_trees'])
    start_date = pd.to_datetime(str(gp['start_date']))
    n_days = int(gp['n_days'])

    health_probs = np.array([gp['health_p_healthy'], gp['health_p_stressed'], gp['health_p_diseased']], dtype=float)
    health_probs = health_probs / health_probs.sum()
    health_states = ['healthy', 'stressed', 'diseased']

    sensor_probs = np.array([gp['sensor_p_ok'], gp['sensor_p_degraded'], gp['sensor_p_offline']], dtype=float)
    sensor_probs = sensor_probs / sensor_probs.sum()
    sensor_states = ['ok', 'degraded', 'offline']

    p_worsen = float(gp['p_health_transition_daily'])
    p_recover = float(gp['p_recovery_daily'])

    miss_ok = float(gp['missing_rate_ok'])
    miss_deg = float(gp['missing_rate_degraded'])
    miss_off = float(gp['missing_rate_offline'])

    risk_deg_thr = float(gp['risk_trunk_deg_threshold'])
    risk_delta_thr = float(gp['risk_daily_delta_threshold'])

    sp = species_params.set_index('species').to_dict(orient='index')
    zp = zone_params.set_index('location_zone').to_dict(orient='index')
    so = soil_params.set_index('soil_type').to_dict(orient='index')
    he = health_effects.set_index('health_status').to_dict(orient='index')

    species_list = list(sp.keys())
    zone_list = list(zp.keys())
    soil_list = list(so.keys())
    forest_types = ['boreal', 'temperate', 'mixed']

    tree_ids = [f"TR{str(i + 1).zfill(6)}" for i in range(n_trees)]
    trees = pd.DataFrame({
        'tree_id': tree_ids,
        'species': rng.choice(species_list, size=n_trees),
        'location_zone': rng.choice(zone_list, size=n_trees),
        'forest_type': rng.choice(forest_types, size=n_trees, p=[0.35, 0.35, 0.30]),
        'soil_type': rng.choice(soil_list, size=n_trees),
        'wind_exposure': np.clip(rng.normal(0.55, 0.18, size=n_trees), 0.0, 1.0),
    })

    age_init = np.clip(rng.gamma(shape=3.0, scale=6.0, size=n_trees) + 5, 5, 60)
    current_year = start_date.year
    trees['planting_year'] = (current_year - np.round(age_init)).astype(int)
    trees['health_status_init'] = rng.choice(health_states, size=n_trees, p=health_probs)

    records = []
    for _, row in trees.iterrows():
        tree_id = row['tree_id']
        species = row['species']
        zone = row['location_zone']
        soil = row['soil_type']
        wind = float(row['wind_exposure'])
        planting_year = int(row['planting_year'])
        health = row['health_status_init']

        trunk = float(sp[species]['trunk_deg_base']) + rng.normal(0, 0.4)
        prev_trunk = trunk

        for d in range(n_days):
            ts = start_date + pd.Timedelta(days=d)
            doy = ts.dayofyear
            z = zp[zone]
            temp = z['temp_mean'] + z['temp_amp'] * np.sin(2 * np.pi * (doy / 365.0)) + rng.normal(0, 1.8)
            hum = z['humidity_mean'] + z['humidity_amp'] * np.cos(2 * np.pi * (doy / 365.0)) + rng.normal(0, 3.5)
            hum = float(np.clip(hum, 20, 100))

            s = so[soil]
            moisture = s['moisture_base'] + s['moisture_humidity_sensitivity'] * (hum - 60) + s[
                'moisture_temp_sensitivity'] * (temp - 10) + rng.normal(0, 0.03)
            moisture = float(np.clip(moisture, 0.05, 0.98))

            sensor_status = rng.choice(sensor_states, p=sensor_probs)

            if rng.random() < p_worsen and health != 'diseased':
                health = health_states[health_states.index(health) + 1]
            if rng.random() < p_recover and health != 'healthy':
                health = health_states[health_states.index(health) - 1]

            h = he[health]
            noise_mult = float(h['noise_multiplier'])

            sap_base = sp[species]['sap_flow_base']
            sap_temp_sens = sp[species]['sap_temp_sensitivity']
            sap = sap_base + sap_temp_sens * temp + 0.6 * (moisture - 0.5) + 0.15 * ((hum - 60) / 40) + rng.normal(0,
                                                                                                                   0.12 * noise_mult)
            sap *= float(h['sap_multiplier'])
            sap = float(max(sap, 0.01))

            leaf_base = sp[species]['leaf_color_base']
            season_factor = 0.75 + 0.25 * np.sin(2 * np.pi * ((doy - 80) / 365.0))
            leaf = leaf_base * season_factor * float(h['leaf_multiplier']) + rng.normal(0, 0.03 * noise_mult)
            leaf = float(np.clip(leaf, 0.05, 0.99))

            drift = 0.004 * wind + float(h['trunk_trend_add']) + rng.normal(0, 0.03 * noise_mult)
            trunk = max(0.0, prev_trunk + drift)
            delta_trunk = trunk - prev_trunk

            est_age = (ts.year - planting_year) + (ts.dayofyear / 365.0) + rng.normal(0, 0.6)
            est_age = float(max(est_age, 0.5))

            biomass = sp[species]['biomass_base'] + sp[species]['biomass_age_coeff'] * est_age + 20 * (
                        moisture - 0.5) + rng.normal(0, 4.0 * noise_mult)
            biomass *= float(h['biomass_multiplier'])
            biomass = float(max(biomass, 1.0))

            risk_flag = (trunk >= risk_deg_thr) or (abs(delta_trunk) >= risk_delta_thr)

            rec = {
                'timestamp': ts,
                'tree_id': tree_id,
                'species': species,
                'location_zone': zone,
                'forest_type': row['forest_type'],
                'soil_type': soil,
                'planting_year': planting_year,
                'wind_exposure': wind,
                'sensor_status': sensor_status,
                'temperature': temp,
                'humidity': hum,
                'moisture_level': moisture,
                'sap_flow_rate': sap,
                'leaf_color_index': leaf,
                'trunk_deg': trunk,
                'health_status': health,
                'estimated_age': est_age,
                'biomass': biomass,
                'risk_flag': bool(risk_flag),
            }

            miss_rate = miss_ok if sensor_status == 'ok' else (miss_deg if sensor_status == 'degraded' else miss_off)
            for col in ['temperature', 'humidity', 'moisture_level', 'sap_flow_rate', 'leaf_color_index', 'trunk_deg']:
                if rng.random() < miss_rate:
                    rec[col] = np.nan

            records.append(rec)
            prev_trunk = trunk

    data = pd.DataFrame.from_records(records)
    trees_out = trees.drop(columns=['health_status_init']).copy()

    trees_out.to_csv(out_dir / 'trees_level1.csv', index=False)
    data.to_csv(out_dir / 'trees_daily_dataset.csv', index=False)

    print('Wrote:', out_dir / 'trees_level1.csv')
    print('Wrote:', out_dir / 'trees_daily_dataset.csv')


if __name__ == '__main__':
    main()
