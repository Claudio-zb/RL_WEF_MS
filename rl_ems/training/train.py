"""Train one SB3 agent without modifying the published research checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
from rl_ems.paths import ROOT, repository_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=['ems', 'wms'], required=True)
    parser.add_argument('--algorithm', choices=['sac', 'td3', 'ppo'], default='sac')
    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--weights', type=float, nargs=3, default=[1, 1, 1])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/training')
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error('--timesteps must be positive')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Choose an empty --output directory to preserve previous runs.')
    from stable_baselines3 import SAC, TD3, PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback
    from environments.EMS_env import MicrogridEnv, NormalizationWrapper
    from environments.WMS_env import CultivateEnv, NormalizedWMS, EvalWMS
    def make_env(evaluation=False):
        weights = np.array(args.weights, dtype=float)
        if args.system == 'ems':
            return NormalizationWrapper(MicrogridEnv(render=False, weights=weights))
        env = NormalizedWMS(CultivateEnv(), days_ahead=1, reward_weigths=weights)
        return EvalWMS(env) if evaluation else env
    with repository_directory():
        output.mkdir(parents=True, exist_ok=True)
        env = Monitor(make_env(), str(output/'monitor.csv'))
        eval_env = Monitor(make_env(evaluation=True))
        try:
            cls = {'sac': SAC, 'td3': TD3, 'ppo': PPO}[args.algorithm]
            kwargs = {'n_steps': 256, 'batch_size': 64} if args.algorithm == 'ppo' else {'batch_size': 256}
            model = cls('MlpPolicy', env, seed=args.seed, device=args.device, verbose=1, **kwargs)
            callback = EvalCallback(eval_env, best_model_save_path=str(output/'best'),
                                    log_path=str(output/'evaluation'), eval_freq=2000,
                                    n_eval_episodes=3, deterministic=True)
            (output/'config.json').write_text(json.dumps(vars(args), default=str, indent=2))
            model.learn(total_timesteps=args.timesteps, callback=callback)
            model.save(str(output/'final_model'))
        finally:
            env.close()
            eval_env.close()
    print(f'Saved training run to {output}')


if __name__ == '__main__':
    main()
