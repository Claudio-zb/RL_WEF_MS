# This file stores the constants used in the EMS environment

import numpy as np

# Constants and bounds
B_p = 1e5 
h_p_max = 1.0
h_p_const = 100   # 20 # [m]
        
# Tank Bounds
Vt_max = 5  # [m3]
Vt_min = 1  # [m3]

# Batteries Constants
Pbat_nom: float = 100  # [kW]
Pbat_max: float = Pbat_nom  # 100  # [Kw]
SoE_max: float = Pbat_nom  # [kWh]
SoE_min: float = 0.2 * Pbat_nom  # [kWh]

# Irrigation Constants

I_max = 1.0  # 1 / 1000  # 1L / s -> 0.001m3 / s
I_min = 0
d_I_bound = 1e-3  # I_max 
Q_p_max = 1.0  # (1 / 1000)  # 1L / s <= > 0.001m3 / s
d_Q_p_bound = 1e-3  # Q_p_max     

dt = 600  # [s] <=> 10 minutes

# Battery efficiency
n_c = 0.85
n_d = 1.15

# another stats

max_power_sun: float = 60.  # 17.577985943193
max_power_d: float = 10.  # 9.75726894105415

# observation transform

T_matrix = np.array([[1/4., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1/4., 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1/I_max, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1/Vt_max, 0, 0, 0, 0, 0, 0, 0], 
                     [0, 0, 0, 0, 1/Q_p_max, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
                     [0, 0, 0, 0, 0, 0, 1/SoE_max, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1/max_power_sun, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1/max_power_d, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1/143, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)

def gen_t_matrix(n_crops:int=1) -> np.ndarray:
    """Generates the observation transformation matrix"""
    # first normalize the rows
    t_matrix = np.zeros((4*n_crops+5, 4*n_crops+5), dtype=np.float32)
    for i in range(n_crops):
        #  v_refs and v_irrs
        #t_matrix
        t_matrix[i, i] = 1/4
        t_matrix[i+n_crops, i+n_crops] = 1/Vt_max
        t_matrix[i+2*n_crops, i+2*n_crops] = 1/4
        t_matrix[i+3*n_crops, i+3*n_crops] = 1/Vt_max
    # now the rest of the variables
    t_matrix[4*n_crops, 4*n_crops] = 1/max_power_sun
    t_matrix[4*n_crops+1, 4*n_crops+1] = 1/max_power_d
    t_matrix[4*n_crops+2, 4*n_crops+2] = 1/SoE_max
    t_matrix[4*n_crops+3, 4*n_crops+3] = 1
    t_matrix[4*n_crops+4, 4*n_crops+4] = 1/143
    return t_matrix


# aquifer constants 

T = 35.1062/(24*60*60)  # [m2/s]
S = 0.19  # [1/m]
r_wells = 0.1270  # [m]