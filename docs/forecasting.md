# Forecasts for predictive control

[Back to project](../README.md)

| Checkpoint | Variable | Sampling | Consumer |
|---|---|---|---|
| `predictive_models/pv_model.pt` | PV power (kW) | 10 min | Energy MPC |
| `predictive_models/pd_model.pt` | Electrical demand (kW) | Hourly; expanded by MPC | Energy MPC |
| `predictive_models/et_model.pt` | Reference evapotranspiration | Daily | Irrigation MPC |
| `predictive_models/soil_moisture/` | Soil and root-depth models | Model-specific | Exploratory soil experiments |

`Predictor` is an LSTMCell followed by a linear head. Checkpoints include mean/std and architecture metadata. `Forecaster` normalizes observations, recursively predicts the requested horizon and clips negative outputs to zero. Core code: [predict_utils.py](../environments/utils/predict_utils.py).

```bash
python -m rl_ems.forecasting.predict --target pv --day 64 --output outputs/pv-day64.csv
python -m rl_ems.forecasting.predict --target demand --day 140 --output outputs/demand-day140.csv
python -m rl_ems.forecasting.animate --start-day 64 --days 3 --seconds-per-day 8
```

The video uses real, unaugmented summer reference windows and 288 past samples. At boundary `k`, observed history ends at `k-1`, so the forecast covers `k..143`. Its first value is stored as the latest prediction for that sample; later updates never rewrite the solid historical trace. All arrays are exported in JSON beside the MP4. The figure uses sample indices rather than clock time because the loader applies the original six-hour offset.

The MPC policy appends the latest observed PV sample before requesting `N = 144 - timestep` predictions. The animation aligns the resulting next-sample forecasts explicitly to avoid using a sample to predict itself. The controller's existing disturbance-to-control timing is preserved by this refactor.

## Original training experiments

[PV training](../experiments/forecasting/predict_pv.py), [demand training](../experiments/forecasting/predict_residential.py), [soil models](../experiments/forecasting/soil_moisture_model.py), and [solar notebook](../predictive_models/solar.ipynb) preserve the original work. They are historical interactive scripts, not importable training APIs: several assume CUDA and perform training or checkpoint writes at top level.

The PV script forms 60-day training and 21-day validation segments per season, but computes normalization from all segments and augments validation through `PVDataSet`. The demand script defines split lengths but uses the same full series for training and validation. Consequently, daily inference outputs are demonstrations of the saved models, not a claim of leakage-free test performance. Correct those procedures before reporting new benchmark results; the refactor does not retrain or replace published checkpoints.
