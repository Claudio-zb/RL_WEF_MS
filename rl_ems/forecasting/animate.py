"""Render real daily LSTM forecasts as a square, silent H.264 video.

Run: python3 -m rl_ems.forecasting.animate
Requires: torch, numpy, pandas, matplotlib, imageio-ffmpeg.
The checkpoint from predict_pv.py predicts PV POWER, not irradiance.
"""
from pathlib import Path
import argparse
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'pv-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import imageio_ffmpeg
import numpy as np
import pandas as pd
import torch
from environments.utils.predict_utils import Forecaster, load_model
from environments.utils.funcionesEMS import solar_power

from rl_ems.paths import ROOT
BG, WHITE, MUTED = '#FFFFFF', '#20252A', '#697078'
GOLD, TEAL = '#B46526', '#245B78'


def forecasts(season, start_day, days):
    """Use consecutive validation days, no random dataset augmentation.

    Same six-hour data offset and power conversion as predict_pv.py.
    Last output of the input pass predicts the first future sample;
    taking the final 144 outputs preserves that alignment.
    """
    data = ROOT / 'environments/Data/EMS'
    rad = pd.read_csv(data / f'data_rad_{season}.csv').values.flatten()[36:][:81*144]
    temp = pd.read_csv(data / f'data_temp_{season}.csv').interpolate().values.flatten()[36:][:81*144]
    power = solar_power(rad, temp)
    model = load_model(str(ROOT / 'predictive_models/pv_model.pt'), device='cpu')
    model.eval()
    forecaster = Forecaster(model)
    results = []
    for day in range(start_day, start_day + days):
        begin = day * 144
        truth = power[begin:begin+144]
        if begin < 288 or len(truth) != 144:
            raise ValueError('Selected days do not have complete input/target windows.')
        if not np.isfinite(power[begin-288:begin+144]).all():
            raise ValueError('Non-finite source data in selected window.')
        # At boundary k, samples through k-1 have arrived. Predict k..143.
        # This is the MPC buffer update + shrinking horizon, aligned to the
        # next sample returned by Forecaster, without using future observations.
        remaining = [forecaster.predict(power[begin+k-288:begin+k], 144-k)
                     for k in range(144)]
        latest = np.array([prediction[0] for prediction in remaining])
        results.append((day+1, truth, remaining, latest))
        print(f'Computed 144 rolling forecasts for day {day+1}', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', choices=['ver', 'inv'], default='ver')
    parser.add_argument('--start-day', type=int, default=64, help='1-based season day, 63–81 (validation)')
    parser.add_argument('--days', type=int, default=3)
    parser.add_argument('--seconds-per-day', type=float, default=8)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--size', type=int, default=1080)
    parser.add_argument('--output', type=Path, default=ROOT / 'pv_animation/linkedin_solar_forecast.mp4')
    args = parser.parse_args()
    if not (63 <= args.start_day <= 81 and 1 <= args.days <= 82-args.start_day):
        parser.error('Choose complete validation windows between days 63 and 81.')
    if args.fps < 1 or args.seconds_per_day < 3 or args.size < 480 or args.size % 2:
        parser.error('Use positive fps, at least 3 seconds/day, and an even size >= 480.')
    results = forecasts(args.season, args.start_day-1, args.days)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.family': 'DejaVu Serif', 'text.color': WHITE,
                               'font.size': 13, 'axes.labelcolor': WHITE,
                               'xtick.color': MUTED, 'ytick.color': MUTED,
                               'animation.ffmpeg_path': imageio_ffmpeg.get_ffmpeg_exe()})
    fig = plt.figure(figsize=(10, 10), dpi=args.size/10, facecolor=BG)
    fig.text(.12, .925, 'Intraday solar power forecasting', fontsize=25)
    fig.text(.12, .883, 'LSTM · Receding horizon to the end of the day', fontsize=14, color=MUTED)
    day_label = fig.text(.12, .815, '', fontsize=14)
    status = fig.text(.92, .815, '', fontsize=12, ha='right', color=MUTED)
    ax = fig.add_axes([.12, .28, .80, .46], facecolor=BG)
    ymax = np.ceil(max(max(t.max(), max(p.max() for p in r)) for _, t, r, _ in results)/20)*20
    ax.set(xlim=(0, 144), ylim=(0, ymax*1.06),
           xlabel='Time step within the day (10 min)', ylabel='PV power (kW)')
    ax.set_xticks([0, 24, 48, 72, 96, 120, 144])
    ax.tick_params(direction='out', length=4, pad=8)
    for name, spine in ax.spines.items():
        spine.set_visible(name in ['bottom', 'left'])
        spine.set_color('#A9AFB5')
        spine.set_linewidth(.7)
    ax.grid(axis='y', color='#E6E8EA', linewidth=.6)
    x = np.arange(144)
    actual, = ax.plot([], [], color=TEAL, lw=1.8, label='Reference')
    predicted, = ax.plot([], [], color=GOLD, lw=2, label='Latest prediction (past)')
    future, = ax.plot([], [], color=GOLD, lw=1.8, linestyle=(0, (5, 3)), label='Remaining forecast')
    cursor = ax.axvline(0, color='#7B8187', lw=.8, linestyle=':')
    ax.legend(handles=[actual, predicted, future], loc='lower left',
              bbox_to_anchor=(-.01, 1.025), frameon=False, ncol=2, fontsize=11)
    fig.text(.12, .17, 'Dashed: updated forecast to step 144. Solid: last prediction before arrival.', fontsize=11)
    fig.text(.12, .135, '144 samples per day; rolling 48-hour history; update every 10 minutes.', fontsize=11, color=MUTED)
    fig.text(.12, .10, 'Reference PV power derived from irradiance and temperature.', fontsize=11, color=MUTED)
    frames = round(args.seconds_per_day * args.fps)
    if frames * .8 < 144:
        parser.error('Increase fps or seconds-per-day so all 144 updates can be displayed.')
    report = []
    writer = FFMpegWriter(fps=args.fps, codec='libx264', bitrate=-1,
                         extra_args=['-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    with writer.saving(fig, str(args.output), dpi=args.size/10):
        for idx, (day, truth, remaining, latest) in enumerate(results):
            report.append({'season_day': day, 'reference_kw': truth.tolist(),
                           'latest_prediction_kw': latest.tolist(),
                           'remaining_forecasts_kw': [p.tolist() for p in remaining]})
            day_label.set_text(f'{"Summer" if args.season == "ver" else "Winter"} · Day {day}')
            for frame in range(frames):
                # Pause at the initial forecast and at the completed day.
                fraction = np.clip((frame/frames-.06)/.80, 0, 1)
                k = min(144, int(fraction*144))
                actual.set_data(x[:k], truth[:k])
                predicted.set_data(x[:k], latest[:k])
                if k < 144:
                    # Join the dashed horizon to the last frozen prediction.
                    fx = x[k:] if k == 0 else x[k-1:]
                    fy = remaining[k] if k == 0 else np.r_[latest[k-1], remaining[k]]
                    future.set_data(fx, fy)
                else:
                    future.set_data([], [])
                cursor.set_xdata([k]*2)
                status.set_text(f'Step {k:03d} / 144   ·   {144-k} remaining')
                if idx == 0 and frame == int(frames*.46):
                    fig.savefig(args.output.with_suffix('.png'), dpi=args.size/10, facecolor=BG)
                writer.grab_frame(facecolor=BG)
            print(f'Rendered day {day}', flush=True)
    args.output.with_suffix('.json').write_text(json.dumps({'quantity': 'PV power (kW)',
        'history_samples': 288, 'sample_minutes': 10, 'season': args.season,
        'alignment': 'At boundary k history ends at k-1; forecast covers k through 143. Solid stores forecast[k][0].',
        'days': report}, indent=2))
    plt.close(fig)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
