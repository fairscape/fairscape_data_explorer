# generate_example_data.py
# Creates sample CSV data (2 hospitals, 2 sepsis statuses = 4 populations)
# and a corresponding JSON schema.

import pandas as pd
import numpy as np
import json
import uuid
import random
from scipy import stats

# --- Configuration ---
NUM_ROWS = 250  # Number of patients
DATA_FILENAME = "example_data_sepsis.csv"
SCHEMA_FILENAME = "example_schema_sepsis.json"
HOSPITALS = ["HOSP_A", "HOSP_B"]
SEED = 2024  # For reproducibility

# Set seed for numpy and random
np.random.seed(SEED)
random.seed(SEED)

# --- Define Schema Structure ---
schema_properties = {
    "patient_id": {
        "description": "Unique Identifier for the patient",
        "type": "string",
        "index": 0
    },
    "hospital_id": {
        "description": "Originating hospital identifier (HOSP_A or HOSP_B)",
        "type": "string",
        "index": 1
    },
    "sepsis": {
        "description": "Whether the patient has sepsis (binary outcome)",
        "type": "boolean",
        "index": 2
    },
    "HR_Mean": {
        "description": "Mean Heart Rate (beats per minute)",
        "type": "number",
        "index": 3
    },
    "HR_Std": {
        "description": "Standard deviation of Heart Rate",
        "type": "number",
        "index": 4
    },
    "HR_Skew": {
        "description": "Skewness of Heart Rate distribution",
        "type": "number",
        "index": 5
    },
    "SpO2_Mean": {
        "description": "Mean Oxygen Saturation (%)",
        "type": "number",
        "index": 6
    },
    "SpO2_Std": {
        "description": "Standard deviation of Oxygen Saturation",
        "type": "number",
        "index": 7
    },
    "SpO2_Skew": {
        "description": "Skewness of Oxygen Saturation distribution",
        "type": "number",
        "index": 8
    },
    "Age_years": {
        "description": "Age of the patient in years (can be fractional for <1 year)",
        "type": "number",
        "index": 9
    }
}

schema_structure = {
    "@id": "local:sepsis_schema-v1.0",
    "name": "Example Sepsis Schema (2 Hospitals, Binary Outcome)",
    "description": "Schema definition for population-level metrics with sepsis outcome and distinct hospital/sepsis populations.",
    "properties": schema_properties,
    # Define required fields
    "required": [
        "patient_id", "hospital_id", "sepsis", "HR_Mean", 
        "HR_Std", "SpO2_Mean", "SpO2_Std", "Age_years"
        # Skewness metrics aren't strictly required
    ],
    "additionalProperties": False
}

# --- Generate Sample Data ---

# 1. Create base patient attributes
patient_ids = [f"PAT_{uuid.uuid4().hex[:8].upper()}" for _ in range(NUM_ROWS)]
hospital_ids = np.random.choice(HOSPITALS, NUM_ROWS, p=[0.55, 0.45])  # Slightly uneven distribution
# Sepsis prevalence (e.g., 20%)
sepsis_status = np.random.choice([True, False], NUM_ROWS, p=[0.20, 0.80])
# Generate age, assuming infants/children
age_years = np.random.uniform(0.1, 5.0, NUM_ROWS).round(2)  # Age 0.1 to 5 years

df = pd.DataFrame({
    'patient_id': patient_ids,
    'hospital_id': hospital_ids,
    'sepsis': sepsis_status,
    'Age_years': age_years
})

# 2. Define population parameters based on hospital and sepsis status

# Heart Rate Mean: Determined by Hospital
# HOSP_A: Base HR 120, HOSP_B: Base HR 140
hr_base_mean = np.where(df['hospital_id'] == 'HOSP_A', 120, 140)

# Heart Rate Standard Deviation: Correlated with sepsis but with overlap
# Base ranges: Septic patients: ~10, Non-septic: ~20
# But with noise to create overlapping distributions
hr_std_base_value = np.where(df['sepsis'] == True, 10, 20)
# Add significant noise to create overlap between populations
hr_std_noise = np.random.normal(0, 5, NUM_ROWS)
hr_std_base = hr_std_base_value + hr_std_noise
# Ensure values stay within realistic limits
hr_std_base = np.clip(hr_std_base, 5, 30)

# SpO2 Mean: Random but clinically appropriate ranges
# Random across all patients regardless of sepsis status
spo2_mean = np.random.uniform(92, 99, NUM_ROWS)

# SpO2 Standard Deviation: Random but clinically appropriate
spo2_std = np.random.uniform(0.8, 2.5, NUM_ROWS)

# Skewness parameters - random across all populations, not correlated with sepsis
hr_skew_range = (-0.6, 0.6)  # Range for all HR skewness values
spo2_skew_range = (-0.8, 0.3)  # Range for all SpO2 skewness values

# 3. Generate the statistical metrics for each patient
df['HR_Mean'] = hr_base_mean + np.random.normal(0, 3, NUM_ROWS)  # Add some noise
df['HR_Std'] = hr_std_base
df['SpO2_Mean'] = spo2_mean
df['SpO2_Std'] = spo2_std

# 4. Generate skewness values randomly (not correlated with sepsis or hospital)
hr_min, hr_max = hr_skew_range
spo2_min, spo2_max = spo2_skew_range

hr_skew = np.random.uniform(hr_min, hr_max, NUM_ROWS)
spo2_skew = np.random.uniform(spo2_min, spo2_max, NUM_ROWS)

df['HR_Skew'] = hr_skew
df['SpO2_Skew'] = spo2_skew

# 5. Round values to appropriate decimal places
df['HR_Mean'] = df['HR_Mean'].round(1)
df['HR_Std'] = df['HR_Std'].round(1)
df['HR_Skew'] = df['HR_Skew'].round(2)
df['SpO2_Mean'] = df['SpO2_Mean'].round(1)
df['SpO2_Std'] = df['SpO2_Std'].round(1)
df['SpO2_Skew'] = df['SpO2_Skew'].round(2)

# 6. Reorder columns to match schema index property
schema_order = sorted(schema_properties.keys(), key=lambda k: schema_properties[k]['index'])
df = df[schema_order]

# --- Save Data and Schema ---

# Save Data
try:
    df.to_csv(DATA_FILENAME, index=False, quoting=1)  # quoting=1 corresponds to csv.QUOTE_ALL
    print(f"Saved example data ({NUM_ROWS} rows, 4 populations) to: {DATA_FILENAME}")
except Exception as e:
    print(f"Error saving data CSV: {e}")

# Save Schema
try:
    with open(SCHEMA_FILENAME, 'w') as f:
        json.dump(schema_structure, f, indent=4)
    print(f"Saved example schema to: {SCHEMA_FILENAME}")
except Exception as e:
    print(f"Error saving schema JSON: {e}")

# --- Display Sample Head and Population Summaries ---
print("\nSample Data Head:")
print(df.head())

# Quick population statistics summary
print("\nPopulation Statistics:")
for hosp in HOSPITALS:
    for sep in [True, False]:
        pop = df[(df['hospital_id'] == hosp) & (df['sepsis'] == sep)]
        print(f"\n{hosp}, Sepsis={sep} (n={len(pop)}):")
        print(f"  HR Mean: {pop['HR_Mean'].mean():.1f}, Std: {pop['HR_Std'].mean():.1f}, Skew: {pop['HR_Skew'].mean():.2f}")
        print(f"  SpO2 Mean: {pop['SpO2_Mean'].mean():.1f}, Std: {pop['SpO2_Std'].mean():.1f}, Skew: {pop['SpO2_Skew'].mean():.2f}")

# --- End of Script ---