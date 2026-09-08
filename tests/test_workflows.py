"""Regression tests for supported entry points and forecast alignment."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'rl-ems-test-mpl'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from rl_ems.paths import ROOT, repository_directory


class WorkflowTests(unittest.TestCase):
    def test_pumping_constructors_and_low_tank(self):
        from environments.EMS_policies import RBPumpingPolicy, MPCPumpingPolicy, MPCPumpingPolicy2
        rb = RBPumpingPolicy(n_crops=1)
        action = rb.get_action(np.array([1.]), np.array([.5, 0, 0, 0, 0, 0, 0]), np.zeros(2))
        np.testing.assert_array_equal(action, [0, 0])
        for cls in [MPCPumpingPolicy, MPCPumpingPolicy2]:
            policy = cls(pv_model=None, pd_model=None)
            self.assertEqual(policy.n_crops, 1)
            policy.init_buffer(np.zeros(288), np.zeros(24))
            self.assertEqual(len(policy.pv), 288)

    def test_forecast_prefix_independent_of_horizon(self):
        from environments.utils.predict_utils import Forecaster, load_model
        model = load_model(str(ROOT/'predictive_models/pv_model.pt'))
        forecaster = Forecaster(model)
        history = np.maximum(0, np.sin(np.arange(288)*np.pi/72))*50
        full = forecaster.predict(history, 144)
        for horizon in [1, 7, 72, 143]:
            np.testing.assert_allclose(forecaster.predict(history, horizon), full[:horizon], atol=1e-5)

    def test_animation_uses_only_past_samples(self):
        from rl_ems.forecasting.animate import forecasts
        import pandas as pd
        from environments.utils.funcionesEMS import solar_power
        data = ROOT/'environments/Data/EMS'
        rad = pd.read_csv(data/'data_rad_ver.csv').values.flatten()[36:]
        temp = pd.read_csv(data/'data_temp_ver.csv').interpolate().values.flatten()[36:]
        power = solar_power(rad, temp)
        calls = []
        def fake_predict(self, history, horizon):
            calls.append((history.copy(), horizon))
            return np.full(horizon, history[-1])
        with patch('rl_ems.forecasting.animate.Forecaster.predict', fake_predict):
            _, truth, remaining, latest = forecasts('ver', 63, 1)[0]
        for k, (history, horizon) in enumerate(calls):
            np.testing.assert_array_equal(history, power[63*144+k-288:63*144+k])
            self.assertEqual(horizon, 144-k)
            self.assertEqual(latest[k], remaining[k][0])
        np.testing.assert_array_equal(truth, power[63*144:64*144])

    def test_coupled_simulation(self):
        from rl_ems.simulation.run import simulate
        mg, crop, _ = simulate(days=1)
        self.assertEqual(mg['mg_obs'].shape, (145, 7))
        self.assertEqual(mg['mg_actions'].shape, (144, 2))
        self.assertEqual(len(crop['v_reqs']), 1)
        self.assertTrue(np.isfinite(mg['mg_obs']).all())

    def test_metrics_exclude_initial_state(self):
        import pickle
        from rl_ems.comparison.report import metrics
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            obs = np.zeros((4, 7)); obs[:, 5] = [-999, -2, 3, -4]
            daily = np.zeros((2, 7)); daily[:, 1] = [3, 5]
            with (directory/'mg_data.pkl').open('wb') as f:
                pickle.dump({'mg_obs': obs, 'end_of_day_samples': daily}, f)
            with (directory/'crop_data.pkl').open('wb') as f:
                pickle.dump({'v_reqs': np.array([2, 0])}, f)
            result = metrics(directory)
            self.assertEqual(result['grid_import_kwh'], 6)
            self.assertEqual(result['net_residual_kwh'], -3)
            self.assertEqual(result['irrigation_mae_m3'], 3)

    def test_repository_context_restores_directory_on_error(self):
        previous = Path.cwd()
        with self.assertRaises(RuntimeError), repository_directory():
            self.assertEqual(Path.cwd(), ROOT)
            raise RuntimeError('test')
        self.assertEqual(Path.cwd(), previous)


if __name__ == '__main__':
    unittest.main()
