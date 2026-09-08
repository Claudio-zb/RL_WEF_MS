# Training RL agents

[Back to project](../README.md)

Train a single controller with an explicit budget and output directory:

```bash
python -m rl_ems.training.train --system ems --algorithm sac --timesteps 10000 --output outputs/ems-sac
python -m rl_ems.training.train --system wms --algorithm td3 --timesteps 10000 --output outputs/wms-td3
```

Options include `--algorithm sac|td3|ppo`, `--weights W1 W2 W3`, `--seed`, `--device` and `--timesteps`. CPU is the default. Three weights are passed to the existing system-specific reward functions; their meanings differ between EMS and WMS. Read `default_rwd_fun` in `EMS_env.py` and `reward_function2` in `WMS_env.py` before interpreting a sweep.

Training saves configuration, monitor data, the final model and periodic evaluations (every 2,000 steps). The best checkpoint is created only after an evaluation. PPO collects full rollout batches, so actual steps can exceed the requested minimum. Small budgets are smoke runs, not trained research-quality agents.

The new command uses Stable-Baselines3 for all three algorithms. Historical WMS PPO experiments use the custom implementation in `RL_algorithms/PPO2.py`; their `.pth` checkpoints are not interchangeable with SB3 `.zip` models. The new command is a convenient fresh-training workflow, not an exact replay of the thesis hyperparameters.

The original weight sweeps and plotting code are preserved in [experiments/training](../experiments/training). They contain interactive cells, hard-coded paths and training flags. Inspect them before execution; they may write into historical `logs/` directories. Some original figures require an external LaTeX installation.
