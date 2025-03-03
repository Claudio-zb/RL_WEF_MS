layer_specs = {"depth": .17,
               "theta_fc": .30,
               "theta_wp": .13,
               "theta_sat": .429,
               "theta_res": .06,
               "alpha": 0.02,  # .02,
               "K0": 12.8,
               "n": 1.75,  # 1.41,
               "theta": 0.25}

layer_specs2 = {"depth": .17,
                "theta_fc": .30,
                "theta_wp": .13,
                "theta_sat": .429,
                "theta_res": .06,
                "alpha": 0.02,  # .02,
                "K0": 12.8,
                "n": 1.75,  # 1.41,
                "theta": 0.15}

evp_layer_specs = {"depth": 0.17,
                   "theta_fc": 0.3,
                   "theta_wp": 0.13,
                   "theta_sat": 0.429,
                   "theta_res": 0.06,
                   "alpha": 0.02,
                   "K0": 12.8,
                   "n": 1.75,
                   "theta": 0.15,
                   "rew": 0.005} # readily water available [m]

# parameters of cultives
tomato = {"crop_name": "tomato",
          "plantation_day": 295,  # [doy]
          "stages_duration": [30, 40, 40, 25],  # [days]
          "Kcb": [0.6, 1.15, 0.6],
          "Ky": [0.4, 1.1, 0.8, 0.4],
          "MAD": 0.5,
          "root_depth_init": 0.2 * 0.8,  # [m]
          "root_depth_max": 1.1 * 0.8,  # [m]
          "height_max": 0.6,  # [m]
          "production_max": 86910,  # [kg/ha]
          "price": 220,  # [$/kg]
          "f_c": [.1, .8, .2]
          }

#%%
def get_fc(silt: float, clay: float) -> float:
    """Computes an estimation of the soil field capacity based on the soil textural composition"""
    return .75 - 0.003 * clay + 0.014 * silt


def get_pwp(silt: float, clay: float) -> float:
    """Computes an estimation of the soil permanent wilting point based on the soil textural composition"""
    return 0.03 + 0.013 * clay + 0.006 * silt
