"""Summarize saved coupled simulations without rerunning controllers."""
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from rl_ems.paths import ROOT


def metrics(directory):
    with (directory/'mg_data.pkl').open('rb') as f:
        mg = pickle.load(f)
    with (directory/'crop_data.pkl').open('rb') as f:
        crop = pickle.load(f)
    # Row zero is the initial state, not a simulated interval.
    residual = np.asarray(mg['mg_obs'])[1:, 5]
    supplied = np.asarray(mg['end_of_day_samples'])[:, 1]
    required = np.asarray(crop['v_reqs'])
    if supplied.shape != required.shape:
        raise ValueError(f'Misaligned daily arrays in {directory}')
    return {'case': directory.name, 'days': len(required),
            'grid_import_kwh': float(-np.minimum(residual, 0).sum()),
            'net_residual_kwh': float(residual.sum()),
            'irrigation_m3': float(supplied.sum()),
            'irrigation_mae_m3': float(np.abs(supplied-required).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'simu_results/wef_ms')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/comparison')
    args = parser.parse_args()
    directories = sorted(p.parent for p in args.input.glob('*/mg_data.pkl'))
    if (args.input/'mg_data.pkl').exists():
        directories = [args.input]
    if not directories:
        parser.error('No saved simulations found in --input.')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose an empty --output directory.')
    table = pd.DataFrame([metrics(d) for d in directories])
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output/'metrics.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, label in zip(axes, ['grid_import_kwh', 'irrigation_m3', 'irrigation_mae_m3'],
                              ['Grid import (kWh)', 'Irrigation (m³)', 'Tracking MAE (m³)']):
        ax.bar(table['case'], table[key], color='#245B78')
        ax.set_ylabel(label)
        ax.tick_params(axis='x', rotation=35)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output/'comparison.png', dpi=180)
    plt.close(fig)
    print(table.to_string(index=False))
    if table['days'].nunique() > 1:
        print('Different simulation lengths: totals are not directly comparable.')


if __name__ == '__main__':
    main()
