layer_specs = {"depth": .5,
               "theta_fc": .30,
               "theta_wp": .1,
               "theta_sat": .429,
               "theta_res": .107,
               "alpha": 0.078,  # .02,
               "K0": 10.8,
               "n": 1.75,  # 1.41,
               "theta": 0.3}

layer_specs2 = {"depth": .5,
                "theta_fc": .30,
                "theta_wp": .1,
                "theta_sat": .429,
                "theta_res": .107,
                "alpha": 0.078,  # .02,
                "K0": 10.8,
                "n": 1.75,  # 1.41,
                "theta": 0.25}

evp_layer_specs = {"depth": 0.1,
                   "theta_fc": 0.3,
                   "theta_wp": 0.11,
                   "theta_sat": 0.402,
                   "theta_res": 0.0902,
                   "alpha": 0.078,
                   "K0": 10.8,
                   "n": 1.75,
                   "theta": 0.3,
                   "rew": 0.2}

#%%
def get_fc(silt:float, clay:float) -> float:
    """Computes an estimation of the soil field capacity based on the soil textural composition"""
    return .75 - 0.003*clay + 0.014*silt

def get_pwp(silt:float, clay:float) -> float:
    """Computes an estimation of the soil permanent wilting point based on the soil textural composition"""
    return 0.03 + 0.013*clay + 0.006*silt