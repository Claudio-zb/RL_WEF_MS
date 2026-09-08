"""Evaluate a saved PV or demand LSTM on a real, unaugmented daily window."""
import argparse
from pathlib import Path
import numpy as np
from rl_ems.paths import ROOT, repository_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['pv', 'demand'], default='pv')
    parser.add_argument('--day', type=int, default=64, help='One-based day in the selected series')
    parser.add_argument('--season', choices=['ver', 'inv'], default='ver')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/forecast.csv')
    args = parser.parse_args()
    from environments.utils.funcionesEMS import get_rad, get_temperatura, get_demand_2, solar_power
    from environments.utils.predict_utils import Forecaster, load_model
    with repository_directory():
        if args.target == 'pv':
            series = solar_power(get_rad(args.season), get_temperatura(args.season))
            steps, checkpoint = 144, 'pv_model.pt'
        else:
            series = get_demand_2()
            steps, checkpoint = 24, 'pd_model.pt'
        start = (args.day-1)*steps
        if args.day < 3 or start+steps > len(series):
            parser.error('Choose a day with two complete days of history and a full target day.')
        model = load_model(str(ROOT/'predictive_models'/checkpoint))
        model.eval()
        prediction = Forecaster(model).predict(series[start-2*steps:start], steps)
    if args.output.exists():
        parser.error('Output exists; choose another --output.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    truth = series[start:start+steps]
    np.savetxt(args.output, np.column_stack((np.arange(steps), truth, prediction)),
               delimiter=',', header='step,reference_kw,prediction_kw', comments='')
    print(f'MAE: {np.mean(np.abs(truth-prediction)):.3f} kW; saved {args.output}')


if __name__ == '__main__':
    main()
