# Data, units and reproducibility

[Back to project](../README.md)

## Existing assets

- `environments/Data/EMS/`: irradiance (W/m²), temperature (°C), electrical demand and constants. `solar_power()` converts irradiance and temperature to PV power (kW) with nominal power 90 kW.
- `environments/Data/WMS/`: weather tables, crop/soil configuration and historical source files.
- `predictive_models/`: PyTorch checkpoint metadata and weights.
- `logs/`: historical RL checkpoints, evaluations and monitor traces.
- `simu_results/`: saved research simulations and figures.

Assets retain their original paths because core modules and historical experiments reference them directly. New runs use ignored `outputs/`. Existing results are preserved; the reorganization does not rewrite numerical experiment outputs. Dataset provenance, measurement orientation and redistribution terms are not fully documented in this checkout; do not infer them from filenames.

## Units and assumptions

Energy–water dynamics use 600-second steps (144 per day). PV and demand are kW, energy residuals and battery state are kWh, pump flow is L/s and tank/irrigation volumes are m³. Daily crop controllers use irrigation depth in metres internally, with conversions in wrappers. The video labels PV power, not solar irradiance.

The simulation reuses/mixes stored daily profiles and is a research environment, not a calibrated digital twin or deployment-ready controller. Its rainfall look-ahead and forecasting validation assumptions are described in the source and forecasting guide. Do not infer causal field performance from stored comparisons.

## Reproducing work

`requirements.txt` records versions available during the local Python 3.12 validation. Seeds and CLI arguments are saved with new runs. An editable install (`python -m pip install -e .`) is optional when invoking modules outside the repository; keep the data checkout in place. A clean-environment installation and full research retraining are distinct from short workflow verification.

Historical scripts may use CUDA, LaTeX, interactive plotting and fixed file paths. Their original algorithm settings are preserved under `experiments/`. Read flags such as `train` and `run` before executing. The new single-agent training defaults differ from those scripts.

## Refactor changes

- Grouped historical scripts by research topic; kept core imports and model paths stable.
- Added explicit simulation, SB3 training, inference and comparison commands.
- Kept compatibility launchers for `animate_pv.py` and `predict_pv.py`; the latter now evaluates safely rather than constructing and saving an untrained model.
- Corrected pumping-policy constructors from `_init_` to `__init__` and initialized zero irrigation for a low tank in the rule-based controller.
- Added targeted tests and CI; excluded local caches and newly generated runs from Git.

No license is added on the author's behalf. The presence of source and data in GitHub is not a grant of unrestricted redistribution rights.
