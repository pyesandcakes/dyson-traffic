I have realized that this prediction model should only be a fallback in case real-time live data fails. Also possible use case is determining the base times per segment of street and additional times to add on top of base time for light activation from EM starting point to ETA to specific light pole.




## Project description

This project collects traffic data for the Eindhoven region using the TomTom Traffic API and combines it with weather information to train a machine-learning model that predicts traffic conditions.
The goal is to estimate how early drivers should be warned under different traffic and weather scenarios.

## Project structure

- `model_eindhoven.joblib`  
  Serialized (joblib) machine-learning model for Eindhoven traffic.  
  Intended to be loaded from Python (e.g. with `joblib.load`) to predict traffic conditions based on the collected features (traffic + weather).

- `data/`  
  Data directory.  
  Used to store:
  - raw and/or processed traffic data (e.g. TomTom responses)
  - possibly derived feature tables for training / evaluating the prediction model  
  Subpaths like `data/tomtom/eindhoven/` are intentionally ignored by git so that local data does not end up in the repository.

- `jupyter/`  
  Folder reserved for Jupyter notebooks (exploratory analysis, feature engineering and model training).  
 
- `ndw/`  
  Python package for working with NDW incident data (not gonna be used as only contains info for highways.  
  It provides:
  - `ndw.incidents.load_incidents()` — returns a DataFrame with all NDW incidents
  - `ndw.incidents.load_active_incidents()` — returns only currently active incidents with usable coordinates, suitable for mapping or joining with other data  

- `scripts/`  
  Utility scripts related to data collection and preprocessing (e.g. calling the TomTom API, transforming responses into tabular data, automating daily downloads).  


