# Simulating the microgrid

[Back to project](../README.md)

`EnergyWaterMG` models tanks, pumping, battery state and energy balance. `Cultivates` models soil and crops. `SimuEnv` couples daily irrigation decisions to 144 energy–water steps per day. `MicrogridEnv` and `CultivateEnv` expose learning environments through Gymnasium.

```bash
python -m rl_ems.simulation.run --days 2 --irrigation scheduled --energy rb --output outputs/baseline
python -m rl_ems.simulation.run --days 2 --irrigation rl --energy rl --output outputs/rl_rl
python -m rl_ems.simulation.run --days 2 --irrigation rl --energy mpc --output outputs/rl_mpc
```

Irrigation options: `scheduled`, `rl` (saved SAC), `mpc` (PSO with ET forecasts). Energy options: `rb`, `rl` (saved TD3), `mpc` (CasADi/IPOPT, shrinking daily horizon). MPC runs are substantially slower than rule-based simulation.

Defaults: seed 42, weather year 2018, planting day from the crop configuration. The CLI uses `logs/wms/weights_5/sac/best_model.zip` and `logs/ems/weights_6/td3/best_model.zip` for RL. These choices mirror the coupled research experiments. See the factory in `rl_ems/simulation/run.py` to select other checkpoints.

Each run saves `config.json`, `mg_data.pkl`, `crop_data.pkl` and `soil_data.pkl`. Use the comparison command to summarize them. The new CLI runs the selected configuration only; it does not execute the historical parameter sweep.

## Core files

- [Coupled simulator](../environments/SimuEnv.py)
- [Energy–water dynamics](../environments/EnergyWaterMG.py)
- [Crop and soil dynamics](../environments/Cultivates.py)
- [Energy and pumping policies](../environments/EMS_policies.py)
- [Irrigation policies](../environments/WMS_policies.py)

Data loaders in the original engine use repository-relative paths. The new simulation and training commands temporarily set the repository working directory and restore it afterwards. The checkout and data are required; this is not a self-contained wheel distribution.
