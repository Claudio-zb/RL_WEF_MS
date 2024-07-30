# Constants and bounds
B_p = 1000 * 10 #1e5
h_p_const = 20
        # Tank Constants
Vt_max = 5
Vt_min = 1

# Batteries Constants
Pbat_nom = 1  # 100
Pbat_max = Pbat_nom  # 100  # [Kw]
SoE_max = Pbat_nom
SoE_min = 0.2 * Pbat_nom

# Irrigation Constants

I_max = 100  # 1 / 1000  # 1L / s -> 0.001m3 / s
I_min = 0
d_I_bound = 1e-3  # I_max 
Q_p_max = 100  # (1 / 1000)  # 1L / s <= > 0.001m3 / s
d_Q_p_bound = 1e-3  # Q_p_max     

dt = 600
n_c = 0.85
n_d = 1.15