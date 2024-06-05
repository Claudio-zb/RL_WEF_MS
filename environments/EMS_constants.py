# This file stores the constants used in the EMS environment

# Constants and bounds
B_p = 1e5 
h_p_const = 3*20 #20 
        
# Tank Bounds
Vt_max = 5  # [m3]
Vt_min = 1  # [m3]

# Batteries Constants
Pbat_nom = 75  # [kW]
Pbat_max = Pbat_nom  # 100  # [Kw]
SoE_max = Pbat_nom # [kWh]
SoE_min = 0.2 * Pbat_nom # [kWh]

# Irrigation Constants

I_max = 1  # 1 / 1000  # 1L / s -> 0.001m3 / s
I_min = 0
d_I_bound = 1e-3  # I_max 
Q_p_max = 1  # (1 / 1000)  # 1L / s <= > 0.001m3 / s
d_Q_p_bound = 1e-3  # Q_p_max     

dt = 600 # [s] <=> 10 minutes

# Battery efficiency
n_c = 0.85
n_d = 1.15