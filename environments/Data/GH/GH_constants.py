#Lets define some variables 

# Arquitectura del invernader

rho_r = 0.0001  # flujo de agua de riego [m3/s]
rho_rec = 0.5 / (25 * 60)  # flujo de agua de bomba [m3/s]

eta_char = 0.85  # eficiencia de carga batería
eta_desc = 0.85  # eficiencia de descarga batería
eta_inv = 0.90  # eficiencia inversor

T_std = 25  # temperatura estandar [C]
T_n = 25  # Temperatura nominal de operación [C]

alpha_fv = -0.39  # coeficiente de temperatura del panel [%/C]
I_std = 1000  # Irradiación estandar [W/m^2]
P_fv_nom = 320  # Potencia nominal del panel [W]
P_bomba = 745.699  # Potencia de la bomba [W]

# Modelo fenomenológico 

rho_air = 1.225  # air density [kg/m^3]
V_inv = 152.4  # Greenhouse Volume  [m^3]
c_pa = 1006.9254  # calor especifico aire [J/kgK]
c_pv = 1875.6864  # calor especifico vapor recalentado [J/kgK]
c_pg = 1700  # calor especifico tierra J/kgK
L_ss = 0.15  # [m]

# Efecto radiación solar 

z: float = 10.0  # [m] altura del invernadero
p_atm: float = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26  # atm pressure [kPA]
A_g: float = 60.0  # Area del suelo del invernadero [m^2]
rho_g: float = 1500.0  # kg/m^3

A_cu: float = 136.002  # Area de la cubierta [m^2]
alpha_inv: float = 0.08  # coeficiente de absorción invernadero
tau_inv: float = 0.78  # coeficiente de transmisión invernadero
epsilon_inv: float = 0.97

alpha_c = 0.81
alpha_g = 0.95
f_c = .8
f_n = 0.95
sigma = 5.67e-8  # constant stefan-boltzmann [W/m^2K^4]

# efecto conducción y convección
e_c = 0.004  #[m]
lambda_c = 0.19  #[W/m3K]

# Efecto intercambio calor con el suelo
z = 4  # m
z_w = 1.2  # m
A_w = 0.9  # m^2

# Demanda de riego 

K_c = 1.2  # coeficiente de cultivo
K_y = 1  # factor de respuesta del cultivo
C_n = 900  # [K mms3/ kg dia]
C_d = .34  # [s/m]
rho_c = 0.4  # [m3/m2]
Z_r = 1.1  # largo de la raíz [m]
theta_fc = 0.96  # capacidad de campo
eta_r = 0.9  # eficiencia de riego
p_t = 1  # porosidad del terreno
