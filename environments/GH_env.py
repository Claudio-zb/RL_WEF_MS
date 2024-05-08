import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from gymnasium import spaces
from torch import Tensor
from environments.custom_env import Custom_env
from environments.GH_constants import *

data_path = "./Data/GH/datosene2020.xls"

class GH_env(Custom_env):
    """ Green House environment class"""
    def __init__(self):
        super().__init__(-1, 1, continuous=False)

        ### micro climate variables 

        # T_inv, T_ss, X_inv, (RH_inv): state variables
        self.state = np.array([25, 25, 0, 0])

        # I, T_ext, RH_ext, w_speed: disturbances
        self.disturbances = np.array([0, 25, 0, 0])
        
        # W: manipulated variable
        self.action = np.array([0])

        data = pd.read_excel(data_path)
        self.T_ext_data = data["T_ext"].to_numpy()
        self.RH_ext_data = data["HR_ext"].to_numpy()
        self.w_speed_data = data["u_ext"].to_numpy()
        self.I_data = data["I_r"].to_numpy()

        #self.fig = Figure(figsize=(3,8), dpi = 300)
        #self.axs =
        self.fig, self.axs = plt.subplots(3,1)
        self.steps = 0
        self.time = 0
        
    def step(self, action):
        self.time = self.steps*600 # [segs]
        # I, T_ext, RH_ext, w_speed, RH_int : disturbances
        self.disturbances[0] = self.I_data[self.steps]
        self.disturbances[1] = self.T_ext_data[self.steps]
        self.disturbances[2] = self.RH_ext_data[self.steps]
        self.disturbances[3] = self.w_speed_data[self.steps]
        self.state = integrate_RK4(GH_climate_ode, 
                                   x0=self.state, 
                                   t0=self.time,
                                   tf=(self.steps+1)*600,
                                   d=self.disturbances,
                                   u=action, 
                                   N=600)
        self.steps += 1 # next timestep (144ts per day)
        return self.state
    
    def reset(self, seed=None, options=None):

        # reset timesteps counter 
        self.steps = 0

        # T_inv, T_ss, X_inv, (RH_inv): state variables
        self.state = np.array([40, 15, 41.0])

        # I, T_ext, RH_ext, w_speed : disturbances
        self.disturbances = np.array([self.I_data[self.steps], 
                                      self.T_ext_data[self.steps], 
                                      self.RH_ext_data[self.steps],
                                      self.w_speed_data[self.steps]])

        return self.state
    
    def render(self):
        pass

    def show_sample(self, policy, scaler):
        pass

    def map_action(self, policy_output: Tensor) -> np.ndarray:
        pass

    def load_initial_conditions(self, initial_conditions: dict) -> np.ndarray:
        pass

### Modeling a Green House environment

# defining the state variables  


### micro climate variables 

# T_inv, T_ss, X_inv, (RH_inv): state variables
# W: manipulated variable
# I, T_ext, RH_ext, w_speed: disturbances


# Solar Irradiation Effect

T_cu = lambda T_inv, T_ext: (T_inv+T_ext)/2

def R_n(I, T_inv, T_ext, RH_ext):
    """Calculates the radiation heat transfer [W] """

    T_atm = f_n*T_ext + 0.0552*(1-f_n)*(T_ext + 273.15)**1.5 
    eo_ext = 6.1078*np.exp(17.269*T_ext/(T_ext+237.3)) # presión parcial vapor [hPa]
    ea_ext = eo_ext*RH_ext/100  
    epsilon_atm = 1-np.exp(-10*ea_ext/T_ext)

    R_sol = A_g*I*(alpha_inv + tau_inv*(alpha_c*f_c+alpha_g*(1-f_c))) # [W]
    R_ter = A_cu*sigma*tau_inv*(epsilon_atm*T_atm**4 - epsilon_inv*((T_cu(T_inv, T_ext))+273.15)**4) # [W]
    return R_sol + R_ter

# Conduction - Convection effect

h_e = lambda w_speed: 7.2 + 3.85*w_speed

def h_i(T_inv, T_ext):
    t_cu = T_cu(T_inv, T_ext)
    outcome = 7.2 if t_cu-T_inv > 11.1 else 1.5 * np.abs(t_cu - T_inv)**0.3
    return outcome 

def U_cc(T_inv, T_ext, w_speed):
    h_e_ = h_e(w_speed)
    h_i_ = h_i(T_inv, T_ext)
    
    outcome = (1/h_i_ + e_c/lambda_c + 1/h_e_)**-1 if h_i_ > 0 else (e_c/lambda_c + 1/h_e_)**-1
    return outcome

def Q_cc(T_inv, T_ext, w_speed):
    # Calor por convección en unidades de potencia 
    return A_cu*U_cc(T_inv, T_ext, w_speed)*(T_inv - T_ext) # [W]

#%% ground heat exchange  

K_g = 2.3 #1 W/mK

def Q_g(T_inv, T_ss):
    # Calor del intercambio con el suelo en unidades de potencia 
    return K_g*A_g*(T_inv - T_ss)/L_ss


#%% Air renovation effect

R_inf = 1

lambda_0 = lambda T_inv: 2502535.259 - 2385.76424*T_inv 

X_ext = lambda RH_ext, T_ext: 0.6219*RH_ext*np.exp(17.269*T_ext/(T_ext+237.3))/(p_atm - (RH_ext/100)*np.exp(17.269*T_ext/(T_ext+237.3)))

w_speed_2 = lambda w_speed: w_speed*4.87/np.log(67.8*z-5.42) # velocidad del viento a la altura [m/s]
w_speed_w = lambda w_speed: w_speed_2(w_speed)*np.log(67.8*z_w - 5.42)/4.87 # velocidad del viento a la altura [m/s]
#G = lambda w_speed_w = C_d * np.sqrt(2*9.81*)
def Ren_air(w_speed, W):
    G = w_speed_w(w_speed)*A_w
    return 3600*G/V_inv if W > 0 else R_inf

def Q_ren(X_inv, RH_ext, T_inv, T_ext, w_speed, W):
    Ren = Ren_air(w_speed, W) #ojito
    X_ext_ = X_ext(RH_ext, T_ext) # humedad exterior absoluta m3/m3
    outcome = V_inv*Ren/3600*(rho_air*c_pa*(T_inv - T_ext) 
                              + lambda_0(T_inv)*(X_inv - X_ext_)
                              + c_pv*(X_inv*T_inv - X_ext_*T_ext))
    return outcome # de nuevo en W

def X_ren(X_inv, RH_ext, T_ext, X_ext, w_speed, W):
    Ren = Ren_air(w_speed, W)
    X_ext_ = X_ext(RH_ext, T_ext)
    return V_inv*Ren*(X_ext_ - X_inv)/3600

#%% Evapotranspiration Effect  

delta = lambda T_inv: 1000*4098*0.6107*np.exp(17.269*T_inv/(T_inv+237.3))/(T_inv+237.3)**2 # [Pa]
gamma = lambda T_inv, p: c_pa*p/(0.6219*lambda_0(T_inv))

K_s = 1.0

Q_soil = lambda T_inv, T_ss: K_s*A_cu*(T_inv - T_ss)/L_ss

DPV = lambda T_inv, RH_inv: 6.1078*np.exp(17.269*T_inv/(T_inv+237.3))*(1-RH_inv/100) 

def ET_0(I, T_inv, T_ext, T_ss, RH_inv, RH_ext, w_speed, W, p):
    delta_ = delta(T_inv)
    gamma_ = gamma(T_inv, p)
    R_n_ = np.maximum(0, R_n(I, T_inv, T_ext, RH_ext)*0.0036/A_g)
    DPV_ = DPV(T_inv, RH_inv)
    return (0.408*delta_*np.maximum(R_n_ - Q_g(T_inv, T_ss), 0) + 37*gamma_*w_speed*DPV_/T_inv)/(delta_ + gamma_*(1+0.34*w_speed))


ET_c = lambda ET_0_, k_c: ET_0_*k_c

Q_evp = lambda ET_c, T_inv: lambda_0(T_inv)*ET_c

A_c = A_g*0.8

X_evap = lambda ET_c: 1000*ET_c*A_c/V_inv/3600

X_inv_sat = lambda T_inv: 5.5638*np.exp(0.0572*T_inv)

def X_c(T_inv, T_ext,T_cu, X_inv):
    T_aux = (T_inv - T_cu(T_ext, T_inv))
    g_c = A_cu/A_g*(10**(-3))*T_aux**(1/3) if T_aux > 0 else 0
    X_inv_sat_ = X_inv_sat(T_inv)
    arg = 0.2522*np.exp(0.0485*T_inv)*(T_inv - T_ext) - (X_inv_sat_ - X_inv) 
    return A_g*g_c*arg/V_inv/3600

#%% Condensation Effect



def RK4(f, x, d, u, h):
    "Performs one step of the Runge-Kutta 4th order method"
    k1 = h * f(x, d, u)
    k2 = h * f(x + 0.5 * k1, d, u)
    k3 = h * f(x + 0.5 * k2,d, u)
    k4 = h * f(x + k3, d, u)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def Euler(f, x, d, u, h):
    """"""
    return x + h*f(x, d, u)

def Verlet(f, x, d, u, h):
    pass


def integrate_RK4(f, x0, t0, tf, d, u, N):
    "integrates the function f using the Runge-Kutta 4th order method"
    x = x0
    h = (tf - t0) / N
    for i in range(N):
        x = Euler(f, x, d, u, h)
        x[2] = np.maximum(x[2], 0)
    return x

#%%

def GH_climate_ode(x, d, u):
    "ODE for the Green House climate"
    # State variables 
    # T_inv [°C], T_ss [°C], X_in [g/m3]
    T_inv = np.clip(x[0], 5, 40)
    T_ss = x[1]
    X_inv = np.maximum(x[2],0)

    # Disturbances
    # I [w/m2], T_ext [°C], RH_ext [%], w_speed [m/s]: 
    I = d[0]
    T_ext = d[1]
    RH_ext = np.clip(d[2], 0, 100)
    w_speed = d[3]

    # Manipulated variable
    # Window aperture
    W = u[0]

    # Auxiliar variables
    X_inv_sat_ = np.maximum(X_inv_sat(T_inv), 0.001)
    RH_inv = 100*X_inv/X_inv_sat_
    RH_inv = np.clip(RH_inv, 0, 100)

    ET_0_ = ET_0(I, T_inv, T_ext, T_ss, RH_inv, RH_ext, w_speed, W, p_atm) 
    ET_pc_ = ET_c(ET_0_, K_c)

    # heat balance computation 
    Q_rad = R_n(I, T_inv, T_ext, RH_ext)
    Q_g = Q_soil(T_inv, T_ss)
    Q_evp_ = Q_evp(ET_pc_, T_inv)
    Q_cc_ = Q_cc(T_inv, T_ext, w_speed) 
    Q_ren_ = Q_ren(X_inv, RH_ext, T_inv, T_ext, w_speed, W)  
    
    Q_t = Q_rad - Q_cc_ - Q_ren_ - Q_g - Q_evp_

    d_T_inv = Q_t/((rho_air*c_pa + X_inv*c_pv/1000)*V_inv)/3600
    d_T_ss = Q_g/(A_g*L_ss*rho_g*c_pg)
    d_X_inv = (X_evap(ET_pc_) - X_ren(X_inv, RH_ext, T_ext, X_ext, w_speed, W) - X_c(T_inv, T_ext,T_cu, X_inv))/3600

    return np.array([d_T_inv, d_T_ss, d_X_inv])