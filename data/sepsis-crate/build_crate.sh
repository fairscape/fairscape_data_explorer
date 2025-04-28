#!/bin/bash

# Initialize the RO-Crate
ROCRATE_ID=$(fairscape-cli rocrate init \
  --name "Fake Sepsis Example" \
  --organization-name "University of Virginia" \
  --project-name "CHORUS" \
  --description "Much longer description of the dataset" \
  --keywords test \
  --keywords ro-crate \
  --author "Niestroy Justin" \
  --date-published "2025-01-28" \
  --version "1.0" \
  --associated-publication "https://www.biorxiv.org/content/10.1101/2024.11.03.621734v2" \
  --conditions-of-access "This dataset was created by investigators and staff of the Cell Maps for Artificial Intelligence project (CM4AI - https://cm4ai.org), a Data Generation Project of the NIH Bridge2AI program, and is copyright (c) 2024 by The Regents of the University of California and, for cellular imaging data, by The Board of Trustees of the Leland Stanford Junior University. It is licensed for reuse under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC-BY-NC-SA 4.0) license, whose terms are summarized here: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en. Proper attribution credit as required by the license includes citation of the copyright holders and of the attribution parties, which includes citation of the following article: Clark T, Schaffer L, Obernier K, Al Manir S, Churas CP, Dailamy A, Doctor Y, Forget A, Hansen JN, Hu M, Lenkiewicz J, Levinson MA, Marquez C, Mohan J, Nourreddine S, Niestroy J, Pratt D, Qian G, Thaker S, Belisle-Pipon J-C, Brandt C, Chen J, Ding Y, Fodeh S, Krogan N, Lundberg E, Mali P, Payne-Foster P, Ratcliffe S, Ravitsky V, Sali A, Schulz W, Ideker T. Cell Maps for Artificial Intelligence: AI-Ready Maps of Human Cell Architecture from Disease-Relevant Cell Lines. BioRXiv 2024." \
  --copyright-notice "Copyright (c) 2024 by The Regents of the University of California" \
  --custom-properties '{"packageType": "pipeline"}')

echo "RO-Crate created with ID: $ROCRATE_ID"

# Register sepsis schema
SCHEMA_ID=$(fairscape-cli rocrate register dataset . \
  --name "Example Sepsis Schema (2 Hospitals, Binary Outcome)" \
  --author "Niestroy Justin" \
  --version "1.0" \
  --description "Schema definition for population-level metrics with sepsis outcome and distinct hospital/sepsis populations." \
  --keywords test \
  --data-format "JSON" \
  --filepath "schema-sepsis-example.json" \
  --custom-properties '{
    "@type": "https://w3id.org/EVI#Schema",
    "properties": {
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
    },
    "required": [
      "patient_id",
      "hospital_id",
      "sepsis",
      "HR_Mean",
      "HR_Std",
      "SpO2_Mean",
      "SpO2_Std",
      "Age_years"
    ],
    "additionalProperties": false
  }')

echo "Schema registered with ID: $SCHEMA_ID"

# Register input dataset (raw patient data) without associating it with a file
INPUT_DATASET_ID=$(fairscape-cli rocrate register dataset . \
  --name "Raw Patient Data" \
  --author "Niestroy Justin" \
  --version "1.0" \
  --description "Raw patient data from hospital monitoring systems" \
  --keywords test \
  --keywords raw \
  --keywords patient \
  --data-format "CSV" \
  --filepath "input_data.csv" \
  --date-published "2025-01-28" \
  --url "")

echo "Input dataset registered with ID: $INPUT_DATASET_ID"

# Register GE Heart Rate Monitor software
SOFTWARE_ID=$(fairscape-cli rocrate register software . \
  --name "GE Heart Rate Monitor" \
  --author "GE Healthcare" \
  --version "3.2.1" \
  --description "Software that collects and processes heart rate data from GE monitoring equipment" \
  --keywords medical \
  --keywords monitoring \
  --keywords heartrate \
  --file-format "py" \
  --url "https://www.gehealthcare.com" \
  --date-modified "2024-11-15" \
  --filepath "ge_monitor.py")

echo "Software registered with ID: $SOFTWARE_ID"

# Register computation that uses input data and software
COMPUTATION_ID=$(fairscape-cli rocrate register computation . \
  --name "Sepsis Detection Computation" \
  --run-by "Niestroy Justin" \
  --command "python process_data.py --input input_data.csv --output sepsis_data.csv" \
  --date-created "2025-01-28T17:53:41Z" \
  --description "Process raw patient data using GE Heart Rate Monitor to extract relevant metrics for sepsis detection" \
  --keywords computation \
  --keywords sepsis \
  --keywords detection \
  --used-software "$SOFTWARE_ID" \
  --used-dataset "$INPUT_DATASET_ID")

echo "Computation registered with ID: $COMPUTATION_ID"

# Register output dataset that points to computation as generatedBy
OUTPUT_DATASET_ID=$(fairscape-cli rocrate register dataset . \
  --name "Sepsis Data" \
  --author "Niestroy Justin" \
  --version "1.0" \
  --description "Fake Sepsis Example Data" \
  --keywords test \
  --data-format "CSV" \
  --filepath "example_data_sepsis.csv" \
  --date-published "2025-01-28" \
  --schema "$SCHEMA_ID" \
  --generated-by "$COMPUTATION_ID")

echo "Output dataset registered with ID: $OUTPUT_DATASET_ID"

echo "RO-Crate creation complete."