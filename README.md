# perioperative-dashboard
Preoperative risk visualization and intervention simulation tool
# Perioperative Risk Assessment Dashboard

This repository contains the code for a multi-module Streamlit dashboard developed for the MSc Health Data Science project at the University of Exeter, in collaboration with Ultramed Ltd.

## Features

- **Module 1**: Patient-Level Risk Summary
- **Module 2**: Regional Risk Overview (Choropleth map)
- **Module 3**: Intervention-by-Intervention Simulation

## Files

- `Dashboard.py`: Main Streamlit app file
- `cleaned_data_extract.csv`: Anonymised patient data
- `uk_9cities_simplified.geojson`: UK region map file
- `requirements.txt`: Python dependencies
- `.gitignore`: File exclusions

## ▶How to run

```bash
pip install -r requirements.txt
streamlit run Dashboard.py
```

> **Note:** This is an academic prototype. Not for clinical use.  
>
> - `cleaned_data_extract.csv`: synthetic dataset with the same structure as the real data used during development.  
> - `uk_9cities_simplified.geojson`: contains regional map boundaries for visualisation purposes.


