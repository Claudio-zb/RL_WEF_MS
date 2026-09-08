"""Run a coupled water–energy–food simulation with explicit controllers."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from rl_ems.paths import ROOT, repository_directory


def simulate(days=2, irrigation='scheduled', energy='rb', seed=42, year=2018):
    from environments.Cultivates import Cultivates
    from environments.SimuEnv import SimuEnv
    from environments.WMS_policies import ScheduledIrrigationPolicy, RLIrrigationPolicy, MPCIrrigationPolicy
    from environments.EMS_policies import RBPumpingPolicy, RLPumpingPolicy, MPCPumpingPolicy
    from environments.utils.predict_utils import Forecaster, load_model
    from stable_baselines3 import SAC, TD3

    with repository_directory():
        if irrigation == 'scheduled':
            water = ScheduledIrrigationPolicy(1, frequency=3, irr_amount=5)
        elif irrigation == 'rl':
            water = RLIrrigationPolicy(1, SAC.load('logs/wms/weights_5/sac/best_model', device='cpu'), year=year, isNormalized=True)
        else:
            water = MPCIrrigationPolicy(1, Cultivates(), year)
        if energy == 'rb':
            power = RBPumpingPolicy(1)
        elif energy == 'rl':
            power = RLPumpingPolicy(1, TD3.load('logs/ems/weights_6/td3/best_model', device='cpu'), True)
        else:
            power = MPCPumpingPolicy(Forecaster(load_model('predictive_models/pv_model.pt')),
                                     Forecaster(load_model('predictive_models/pd_model.pt')), days_ahead=0)
        env = SimuEnv(water, power, year=year, seed=seed)
        if energy == 'mpc':
            power.init_buffer(np.concatenate((env.prev_pv_daily_profile, env.pv_daily_profile)),
                              env.pd_daily_profile[::6])
        env.run(init_doy=env.cultivate_env.crops[0].plantation_day, total_days=days)
        return env.get_simu_data()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=int, default=2)
    parser.add_argument('--irrigation', choices=['scheduled', 'rl', 'mpc'], default='scheduled')
    parser.add_argument('--energy', choices=['rb', 'rl', 'mpc'], default='rb')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--year', type=int, default=2018)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/simulation')
    args = parser.parse_args()
    if not 1 <= args.days <= 81:
        parser.error('Choose between 1 and 81 simulation days.')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty; choose a new --output to preserve results.')
    results = simulate(args.days, args.irrigation, args.energy, args.seed, args.year)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, result in zip(['mg_data', 'crop_data', 'soil_data'], results):
        with (args.output/f'{name}.pkl').open('wb') as f:
            pickle.dump(result, f)
    (args.output/'config.json').write_text(json.dumps(vars(args), default=str, indent=2))
    print(f'Saved simulation to {args.output}')


if __name__ == '__main__':
    main()
