# Original research experiments

[Back to project](../README.md)

These files retain the interactive research workflows, including their existing settings and local edits. They are grouped by purpose and are not automatically imported by the new CLI. Run from the repository root with `python -m experiments.<group>.<module>` only after inspecting training flags, device assumptions and output paths.

| Area | Scripts | Purpose |
|---|---|---|
| [Simulation](simulation/) | `ems_test`, `wms_test`, `new_mpc`, `crop_simu_results` | Standalone environment studies, MPC exploration and crop figures |
| [Training](training/) | `ems_training`, `wms_training` | Original reward sweeps, RL training and learning curves |
| [Comparisons](comparisons/) | `ems_comparison`, `wms_comparison` | Energy and irrigation controller evaluation |
| [Comparisons](comparisons/) | `ewms_comparison`, `ewms_results` | Coupled RL/MPC case studies, saved results and figures |
| [Forecasting](forecasting/) | `predict_pv`, `predict_residential`, `soil_moisture_model` | Original neural forecasting/model-training cells |

Example path migration: root `ewms_comparison.py` → `experiments/comparisons/ewms_comparison.py`. For the supported report command use `python -m rl_ems.comparison.report` instead. Files named `*_test.py` here are experiments, not the automated test suite.
