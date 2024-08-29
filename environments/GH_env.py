import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from torch import Tensor
#from custom_env import DiscreteCustomEnv
#from GH_constants import *
from environments.custom_env import DiscreteCustomEnv
from environments.Data.GH.GH_constants import *


data_path = "./Data/GH/datosene2020.xls"

class GH_env(DiscreteCustomEnv):
    """ Green House environment class"""
    def __init__(self):
        super().__init__(np.array([0.0, 1.0]))

        ### micro climate variables 

        # T_inv, T_ss, X_inv, (RH_inv): state variables
        self.state:np.ndarray = np.array([25, 25, 0])

        # I, T_ext, RH_ext, w_speed: disturbances
        self.disturbances:np.ndarray = np.array([0, 25, 0, 0])
        
        # W: manipulated variable
        self.action:np.ndarray = np.array([0])

        data = pd.read_excel(data_path)
        self.T_ext_data:np.ndarray = data["T_ext"].to_numpy()
        self.RH_ext_data:np.ndarray = data["HR_ext"].to_numpy()
        self.w_speed_data:np.ndarray = data["u_ext"].to_numpy()
        self.I_data:np.ndarray = data["I_r"].to_numpy()

        #self.fig = Figure(figsize=(3,8), dpi = 300)
        #self.axs =
        self.fig:Figure = None
        self.axs:np.ndarray[plt.Axes] = None
        
        self.fig, self.axs = plt.subplots(3,1)
        
        self.steps:int = 0
        self.time = 0

        self.variables = {"T_inv": [],
                          "T_ss": [],
                          "X_inv": [],
                          "RH_inv": [],
                          "I": [],
                          "T_ext": [],
                          "RH_ext": [],
                          "w_speed": [],
                          "W": [],
                          "Q_rad": [],
                          "Q_g": [],
                          "Q_cc": [],
                          "Q_evp": [],
                          "Q_ren": [],
                          "ET_0": [],
                          "ET_pc": [],
                          "ET_c": [],
                          "Q_t": [], 
                          "R_n": [],
                          "R_n_c": [], 
                          "X_ext": [], 
                          "x_evap": [],
                          "x_c": [],
                          "x_vent": []
                          }
        
    def step(self, action):
        self.time = self.steps*600 # [segs]
        # I, T_ext, RH_ext, w_speed : disturbances
        self.disturbances[0] = self.I_data[self.steps] + 10*np.random.randn() if self.I_data[self.steps] > 100 else self.I_data[self.steps]
        self.disturbances[1] = self.T_ext_data[self.steps] + 0.1*np.random.randn() + 5.0
        self.disturbances[2] = self.RH_ext_data[self.steps] + 0.5*np.random.randn()
        self.disturbances[3] = self.w_speed_data[self.steps] + 0.1*np.random.randn()
        self.state = integrate(self.GH_climate_ode, 
                                x0=self.state, 
                                t0=self.time,
                                tf=(self.steps+1)*600,
                                d=self.disturbances,
                                u=action, 
                                N=600)
        self.steps += 1 # next timestep (144ts per day)
        self.steps = self.steps % 144
        return self.state
    
    def reset(self, seed=None, options=None):

        # reset timesteps counter 
        self.steps = 0

        # T_inv, T_ss, X_inv, (RH_inv): state variables
        self.state = np.array([12.5, 15, 80.5])

        # I, T_ext, RH_ext, w_speed : disturbances with timestep 10 min
        self.disturbances = np.array([self.I_data[self.steps], 
                                      self.T_ext_data[self.steps], 
                                      self.RH_ext_data[self.steps],
                                      self.w_speed_data[self.steps]])
        
        return self.state
    
    def get_figure(self):
        return self.fig

    def render(self):
        pass
    def sample_trajectory(self, policy = None, scaler = None):
        if policy is None:
            policy = lambda x, t: np.array([0.5])
        x0 = self.reset()
        N = 143*1
        states = np.zeros((N+1, 3))
        actions = np.zeros(N)
        states[0] = x0
        rewards = np.zeros(N)
        for i in range(N):
            action = policy(x0, i)
            x0 = self.step(action)
            states[i+1] = x0
            actions[i] = action
        return states, actions, rewards

    def show_sample(self, policy = None, scaler = None):
        states, actions, rewards = self.sample_trajectory(policy, scaler)

        for ax in self.axs:
            ax.clear()

        self.fig.suptitle("Green House Environment", fontweight = "bold")

        self.axs[0].plot(states[:,0], label = "T_inv")
        self.axs[0].set_ylabel("Temperature [°C]")
        self.axs[0].set_title("Temperature of the greenhouse")

        self.axs[1].plot(states[:,1], label = "T_ss")
        self.axs[1].set_ylabel("Temperature [°C]")
        self.axs[1].set_title("Temperature of the ground")

        self.axs[2].plot(states[:,2], label = "X_inv")
        self.axs[2].set_xlabel("Sample")
        self.axs[2].set_ylabel("Humidity [g/m3]")
        self.axs[2].set_title("Internal Absolute Humidity")

        self.fig.tight_layout()

        for ax in self.axs:
            ax.legend()

    def map_action(self, policy_output: Tensor) -> np.ndarray:
        pass

    def load_initial_conditions(self, initial_conditions: dict) -> np.ndarray:
        pass

    def GH_climate_ode(self, x, d, u, savedata = False):
        "ODE for the Green House climate"
        # State variables 
        # T_inv [°C], T_ss [°C], X_in [g/m3]
        T_inv = np.clip(x[0], 5, 40)
        T_ss = x[1]
        X_inv = np.maximum(x[2],0)

        # Disturbances
        # I [w/m2], T_ext [°C], RH_ext [%], w_speed [m/s]: 
        I_s = d[0]
        T_ext = d[1]
        RH_ext = np.clip(d[2], 0, 100)
        w_speed = d[3]

        # Manipulated variable
        # Window aperture [0-1]
        W = u[0]

        # Auxiliar variables
        x_inv_sat_ = x_inv_sat(T_inv)
        RH_inv = 100*X_inv/x_inv_sat_
        RH_inv = np.clip(RH_inv, 0, 100)

        ET_0_ = ET_0(I_s, T_inv, T_ext, T_ss, RH_inv, RH_ext, w_speed, p_atm) 
        ET_pc_ = ET_0_

        # heat balance computation 
        Ren_ = Ren_air(w_speed, W)
        Rn_sun, Rn_ter = R_n(I_s, T_inv, T_ext, RH_ext)
        Q_rad = Rn_sun + Rn_ter 
        Q_g = Q_soil(T_inv, T_ss)
        Q_evp_ = Q_evp(ET_pc_, T_inv)
        #Q_evp_ = Q_evp_tom(T_inv, T_ext, RH_inv, I_s, RH_ext, w_speed)
        Q_cc_ = Q_cc(T_inv, T_ext, w_speed) 
        Q_ren_ = Q_ren(X_inv, RH_ext, T_inv, T_ext, Ren_)  
        
        Q_t = Q_rad - Q_g - Q_cc_ - Q_evp_ - Q_ren_  

        d_T_inv = Q_t/((rho_air*c_pa + X_inv*c_pv/1000)*V_inv)
        d_T_ss = Q_g/(A_g*L_ss*rho_g*c_pg)
        phi_wind_ = phi_wind(w_speed, W)

        x_evap_ = (ET_pc_*1000/3600) # [g/s]  #x_evap(T_inv, T_ext, X_inv, I_s, phi_wind_)
        x_c_ = x_c(T_inv, T_ext, X_inv)
        x_vent_ = x_vent(T_ext, X_inv, RH_ext, Ren_)

        d_X_inv = (A_cu/V_inv)*(x_evap_ - x_c_ - x_vent_)

        if savedata:
            self.variables["T_inv"].append(T_inv)
            self.variables["T_ss"].append(T_ss)
            self.variables["X_inv"].append(X_inv)
            self.variables["RH_inv"].append(RH_inv)
            self.variables["I"].append(I_s)
            self.variables["T_ext"].append(T_ext)
            self.variables["RH_ext"].append(RH_ext)
            self.variables["w_speed"].append(w_speed)
            self.variables["W"].append(W)
            self.variables["Q_rad"].append(Q_rad)
            self.variables["Q_g"].append(Q_g)
            self.variables["Q_cc"].append(Q_cc_)
            self.variables["Q_evp"].append(Q_evp_)
            self.variables["Q_ren"].append(Q_ren_)
            self.variables["ET_0"].append(ET_0_)
            self.variables["ET_pc"].append(ET_pc_)   
            self.variables["R_n"].append(Rn_sun)
            self.variables["R_n_c"].append(Rn(I_s))  
            self.variables["X_ext"].append(X_ext(RH_ext, T_ext))  
            self.variables["x_evap"].append(x_evap_)
            self.variables["x_c"].append(x_c_)
            self.variables["x_vent"].append(x_vent_)

        return np.array([d_T_inv, d_T_ss, d_X_inv])


# ----------------------------------------------------------------------------- #
#---------------------- END OF CLASS DEFINITION --------------------------------#
# ----------------------------------------------------------------------------- #

### Modeling a Green House environment

# defining the state variables  


### micro climate variables 

# T_inv [°C], T_ss[°C], X_inv [g/m3], (RH_inv): state variables
# W: manipulated variable
# I, T_ext, RH_ext, w_speed: disturbances


# Solar Irradiation Effect

T_cu = lambda T_inv, T_ext: (T_inv+T_ext)/2

def R_n(I, T_inv, T_ext, RH_ext):
    """Calculates the radiation heat transfer [W] """

    T_atm = f_n*(T_ext + 273.15) + 0.0552*(1-f_n)*(T_ext + 273.15)**1.5 
    eo_ext = 6.1078*np.exp(17.269*T_ext/(T_ext+237.3)) # presión parcial vapor [hPa]
    ea_ext = eo_ext*RH_ext/100  
    epsilon_atm = 1-np.exp(-10*ea_ext/T_ext)

    R_sol = A_g*I*(alpha_inv + tau_inv*(alpha_c*f_c+alpha_g*(1-f_c))) # [W]
    R_ter = A_cu*sigma*tau_inv*(epsilon_atm*T_atm**4 - epsilon_inv*((T_cu(T_inv, T_ext))+273.15)**4) # [W]
    return R_sol, R_ter

# Conduction - Convection effects

def Q_cc(T_inv, T_ext, w_speed):
    """Calor por convección en unidades de potencia [W]"""
    
    t_cu = (T_inv + T_ext)/2 # [°C]
    h_e = 7.2 + 3.85*w_speed # [W/m2K]
    h_i = 7.2 if t_cu-T_inv > 11.1 else 1.95 * np.abs(t_cu - T_inv)**0.3 # [W/m2K]
    U_cc = (1/h_i + e_c/lambda_c + 1/h_e)**-1 if h_i > 0 else (e_c/lambda_c + 1/h_e)**-1 # [W/m2K]

    return A_cu*U_cc*(T_inv - T_ext) # [W]

#%% ground heat exchange  

K_g = 2.3 #1 [W/mK]

def Q_g(T_inv, T_ss):
    """Calculates the heat exchange with the ground [W]"""
    return K_g*A_g*(T_inv - T_ss)/L_ss


#%% Air renovation effect

R_inf = 3.0/3600 # [1/s]

def lambda_0(T_inv): 
    """Calculates the latent heat of vaporization [J/kg]"""
    return 2502535.259 - 2385.76424*T_inv

def X_ext(RH_ext, T_ext):
    """Calculates the absolute humidity of the external air [g/m3]"""
    e_s = 6.1078*np.exp(17.269*T_ext/(T_ext+237.3)) # presion saturación [hPa]
    numerator = 0.6219*(RH_ext/100)*e_s #presion de vapor real
    denominator = p_atm - (RH_ext/100)*e_s
    X_ext_ = numerator/denominator # [kg/kg]
    K_conversion = 1000*1.225 # [kg/kg] -> [g/m3]
    X_ext_ = K_conversion*X_ext_
    return X_ext_

#G = lambda w_speed_w = C_d * np.sqrt(2*9.81*)
def Ren_air(w_speed, W):
    """Calculates the air renovation rate [1/s]"""
    w_speed_w = w_speed*np.log(67.8*z_w - 5.42)/np.log(67.8*z-5.42)/10 # velocidad del viento a la altura de la ventana [m/s]
    G = w_speed_w*A_w*W # [m3/s]
    return G/V_inv + R_inf/3 if W > 0 else R_inf/3

def Q_ren(X_inv, RH_ext, T_inv, T_ext, Ren):
    """Calculates the air renovation effect in the Green House [W]"""
    X_ext_ = X_ext(RH_ext, T_ext) # humedad exterior absoluta [g/m3]
    outcome = V_inv*Ren*(rho_air*c_pa*(T_inv - T_ext) 
                              + 0.1*lambda_0(T_inv)*(X_inv - X_ext_)/1000 
                              + c_pv*(X_inv*T_inv - X_ext_*T_ext)/1000)
    return outcome # de nuevo en W

def X_ren(X_inv, RH_ext, T_ext, X_ext, w_speed, W):
    Ren = Ren_air(w_speed, W)
    X_ext_ = X_ext(RH_ext, T_ext)
    return V_inv*Ren*(X_ext_ - X_inv)/3600

#%% Evapotranspiration Effect  

K_s = 1.0

Q_soil = lambda T_inv, T_ss: K_s*A_cu*(T_inv - T_ss)/L_ss

DPV = lambda T_inv, RH_inv: 6.1078*np.exp(17.269*T_inv/(T_inv+237.3))*(1-RH_inv/100) 

def ET_0(I, T_inv, T_ext, T_ss, RH_inv, RH_ext, w_speed, p):
    """Hourly evapotranspiration [mm/h]
    I: Solar radiation [W/m2]
    T_inv: Internal temperature [°C]
    T_ext: External temperature [°C]
    T_ss: Soil temperature [°C]
    RH_inv: Internal relative humidity [%]
    RH_ext: External relative humidity [%]
    w_speed: Wind speed [m/s]
    W: Window aperture [m]
    p: Atmospheric pressure [kPa]"""

    delta_ = 1000*4098*0.6107*np.exp(17.269*T_inv/(T_inv+237.3))/(T_inv+237.3)**2 # [Pa]
    gamma_ = c_pa*p/(0.6219*lambda_0(T_inv))
    Rn_sun, Rn_ter = R_n(I, T_inv, T_ext, RH_ext) # [j/s]
    #R_n_ = np.maximum(0, Rn_sun*0.0036/A_g) # [MJ/m2h]
    R_n_ = Rn(I)*0.0036
    G = Q_g(T_inv, T_ss)*0.0036/A_g # [MJ/m2h]
    DPV_ = DPV(T_inv, RH_inv)
    return (0.408*delta_*np.maximum(R_n_ - G, 0) + 37*gamma_*w_speed*DPV_/T_inv)/(delta_ + gamma_*(1+0.34*w_speed))


def Q_evp(ET_c, T_inv): 
    """Calculates the evaporation effect in the Green House [W/m2]"""
    ET_c = ET_c/3600*A_c # [mm/h] -> [kg/s]
    Q_evp_ = lambda_0(T_inv)*ET_c # [W/m2] [J/kg * mm/h ]
    return Q_evp_

def Q_evp_tom(T_inv, T_ext, RH_inv, I, RH_ext, w_speed):
    Rn_sun, _ = R_n(I, T_inv, T_ext, RH_ext)
    DPV_ = DPV(T_inv, RH_inv) 
    Q_evp_ = A_c*(0.2*Rn_sun + 5.5*DPV_ + 5.3*w_speed)
    return Q_evp_

A_c = A_g*0.8

LAI = 1.0 # Leaf Area Index [m2/m2]

def phi_wind(w_speed, W):
    """Calculates the wind speed inside the green house [m/s]"""
    f_lsd = 0
    fr_window_lsd = 0.1
    f_wsd = 0
    fr_window_wsd = 0
    phi_wind_ = (f_lsd*fr_window_lsd + f_wsd*fr_window_wsd)*w_speed*W*A_w
    return phi_wind_

def r_b(T_inv, T_ext, phi_wind):
    d = 1
    return 1.174*d**.5/(d*np.abs((T_ext-T_inv)/2) + 207*phi_wind**2)**.25

def epsilon(T_inv):
    return 0.7584*np.exp(0.0518*T_inv)

def Rn(I):
    """Calculates the net radiation (chapingo)  [W/m2]"""
    return tau_inv*(1-np.exp(-.7*LAI))*I

k = 0.8 # coeficiente extinsión de la radiación [J/g]

def r_s(T_inv, I):
    Rn_ = Rn(I)
    return 82*(4.3/0.54)*np.exp(-k*Rn_/(2*LAI))*(1+ 0.023*(T_inv-20)**2)

def g_E(T_inv, T_ext, phi_wind, I):
    value = 2*LAI/((1+0.7584*np.exp(0.0518*T_inv))*r_b(T_inv, T_ext, phi_wind) + r_s(T_inv, I))
    return value

def x_crop(T_inv, T_ext, I, phi_wind):
    """Computes de humidity of the crops [g/m3]"""
    epsilon_ = epsilon(T_inv)
    r_b_ = r_b(T_inv, T_ext, phi_wind)
    Rn_ = Rn(I)
    lambda_0_ = lambda_0(T_inv)
    return x_inv_sat(T_inv) + epsilon_*r_b_*Rn_/(2*LAI*lambda_0_) # [g/m3]

def x_evap(T_inv, T_ext, x_inv, Is, phi_wind):
    """Calculates the evaporation effect in the Green House [g/(s m3)]"""
    x_crop_ = x_crop(T_inv, T_ext, Is, phi_wind)
    g_E_ = g_E(T_inv, T_ext, phi_wind, Is)
    return g_E_ * (x_crop_ - x_inv)

def x_inv_sat(T_inv): 
    """Calculates the saturation humidity in the Green House [g/m3]"""
    return 5.5638*np.exp(0.0572*T_inv)

def x_c(T_inv, T_ext, x_inv):
    """Calculates the condensation effect in the Green House [g/(s m3)]"""
    T_cu = (T_inv + T_ext)/2
    #g_c = np.maximum(0, 250.0*np.sign(T_inv-T_cu)*np.abs(T_inv-T_cu)**(1./3.))
    g_c = np.maximum(0, A_cu/A_g*1.64e-3*np.sign(T_inv-T_cu)*np.abs(T_inv-T_cu)**(1./3.))
    x_inv_sat_ = x_inv_sat(T_inv)
    return g_c*(0.2522*np.exp(0.0485*T_inv)*(T_inv - T_ext) - (x_inv_sat_ - x_inv))

def x_vent(T_ext, x_inv, RH_ext, Ren):
    """Calculates the ventilation effect in the Green House [g/(s m3)]"""
    #phi_lek = 0.000083 + 0.000035
    #phi_win = Ren #*V_inv # ventilation
    #gv = phi_lek + phi_win
    x_vent_ = Ren*(x_inv - X_ext(RH_ext, T_ext))
    return x_vent_



#%% Condensation Effect

def RK4(f, x, d, u, h, savedata = False):
    """Performs one step of the Runge-Kutta 4th order method
    f: function to integrate
    x: state variable
    d: disturbances
    u: manipulated variable
    h: step size
    """
    k1 = h * f(x, d, u, savedata = savedata)
    k2 = h * f(x + 0.5 * k1, d, u)
    k3 = h * f(x + 0.5 * k2,d, u)
    k4 = h * f(x + k3, d, u)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def Euler(f, x, d, u, h):
    """Performs one step of the Euler method
    f: function to integrate
    x: state variable
    d: disturbances
    u: manipulated variable
    h: step size
    """
    return x + h*f(x, d, u)

def Verlet(f, x, d, u, h):
    """Performs one step of the Verlet method
    f: function to integrate
    x: state variable
    d: disturbances
    u: manipulated variable
    """
    k = h*f(x, d, u)
    return x + h*f(x + 0.5*k, d, u)


def integrate(f, x0, t0, tf, d, u, N)->np.ndarray:
    "integrates the function f using the Runge-Kutta 4th order method"
    x = x0
    h = (tf - t0) / N
    for i in range(N):
        if i == 0:
            x = RK4(f, x, d, u, h, savedata = True)
        else:
            x = RK4(f, x, d, u, h)
        x[2] = np.maximum(x[2], 0)
    return x

if __name__ == "__main__":
    gh_env = GH_env()

# plot the sample trajectory
    gh_env.show_sample()
    gh_env.fig.set_size_inches(8, 7)
    gh_env.fig.savefig('figures/greenhouse_simu.png', dpi = 300)