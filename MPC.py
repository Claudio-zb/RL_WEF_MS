import matplotlib.pyplot as plt
import numpy as np
from utils_functions.funcionesEMS import get_demand, get_rad, get_temperatura, solar_power
from psopy import init_feasible, minimize


def manage_batteries(SoE: float,
                     P_fv: float,
                     P_demanded: float,
                     P_pump: float) -> tuple[float, float, float, float]:
        """
        Choose the power to re/discharge the batteries and computes the next SoE
        :param SoE:
        :param P_fv:
        :param P_demanded:
        :param P_pump:
        :return:
        """
        SoE_max = 80
        SoE_min = 20
        n_d = 1.15
        n_c = 0.85  
        Pbat_max = 30
        

        E_surplus = 0
        E_deficit = 0
        Pbat = P_fv - P_demanded - P_pump
        if not -Pbat_max <= Pbat <= Pbat_max:  # The surplus is under the power of power bounds of the battery
            P_not_used = Pbat - np.clip(Pbat, -Pbat_max,
                                        Pbat_max)  # positive for surplus, negative for deficit
            Pbat = np.clip(Pbat, -Pbat_max, Pbat_max)
        else:
            P_not_used = 0
        delta_SoE = np.max([Pbat, 0]) * n_c * (1/6) + np.min([Pbat, 0]) / n_d * (1/6)
        next_SoE = SoE + delta_SoE
        if SoE_min <= next_SoE <= SoE_max:  # The recharge is done immediately
            Pbat = Pbat
        else:  # The re/discharge is done but there is a surplus/deficit of energy
            E_surplus = next_SoE - SoE_max if next_SoE > SoE_max else 0
            E_deficit = next_SoE - SoE_min if next_SoE < SoE_min else 0

            next_SoE = np.clip(next_SoE, SoE_min, SoE_max)
            Pbat = np.max([next_SoE - SoE, 0]) / (1/6) / n_c + np.min(
                [next_SoE - SoE, 0]) * n_d / (1/6)

        E_surplus = E_surplus + P_not_used * n_c * (1/6) if P_not_used > 0 else E_surplus
        E_deficit = E_deficit - P_not_used / n_d * (1/6) if P_not_used < 0 else E_deficit

        return Pbat, next_SoE, E_surplus, E_deficit

def get_reward(E_surplus,
               E_deficit,
               Irr_prev,
               Irr,
               V_Irr,
               V_ref,
               Qp_prev, Qp) -> float:
    """
    Computes the reward for the current state of the system
    :param E_surplus: Energy surplus [kWh]
    :param E_deficit: Energy deficit [kWh]
    :param Irr_prev: Previous irrigation [%] in the first day
    :param Irr: Irrigation [%] in the first day
    :param V_Irr_prev: Previous water volume fulfilled the first day [m3]
    :param V_Irr: Water volume fulfilled the first day [m3]
    :param V_ref: Water volume demand the first day [m3]
    :param Qp_prev: Previous power of the pump [%]
    :param Qp: Power of the pump [%]
    :param Pbat_prev: Previous power of the battery [%]
    :param Pbat: Power of the battery [%]
    :return: reward r(t)
    """

    d_Irr = (Irr - Irr_prev)/100  # [l/s]
    d_Qp = (Qp - Qp_prev)/100  # [l/s]

    economic_component = 25 * E_surplus - 100 * E_deficit
    w_ns = np.max([0.0, V_ref - V_Irr])  # Pending demand
    w_ex = np.max([0.0, -(V_ref - V_Irr)])  # Excedent

    reward = (30*np.exp(-0.05 * (w_ns**2 + 5*w_ex**2 + d_Irr**2 + d_Qp**2))
              + 20*np.exp(-0.1 * (V_ref - V_Irr)**2)
              + economic_component/10)
    return reward

# state x: [v_irr, Vtank, SoE, I_rr_prev, Q_p_prev]
# action u: [Irr, Q_p]
# disturbances d: [V_ref, P_fv, P_demand]

def next_state(x, u, d):

    irrigation_penalty = 0

    if x[1] <= 1:  # If the tank is empty, there is no irrigation
        if u[0] > 0:
            u[0] = 0
            irrigation_penalty = 10  # -300

    amount_to_irrigate = 600 * (u[0] * 1e-6)

    extraction_penalty = 0
    Vt_to_fill = 5 - x[1] - amount_to_irrigate  # Amount of water that can be filled
    if Vt_to_fill <= 0 < u[1]:  # If the tank is full and the pump is feeding, the pump is turned off
        u[1] = 0
        extraction_penalty = 10  # -300

    amount_to_pump = 600 * (u[1] * 1e-6)

    V_tank_next = np.clip(x[1] + amount_to_pump - amount_to_irrigate, 1, 5)
    V_Irr_next = x[0] + amount_to_irrigate
    #print(amount_to_irrigate)
    
    # V_irr_next = x[0] + (d[0]*1e-6)*(600/3600)
    V_tank_next = x[1] - (u[0]*1e-6)*(600/3600) + (u[1]*1e-6)*(600/3600)
    P_pump = 600 * 1e4 * u[1] / 600 * 1 / 1e3
    Pbat, SoE_next, E_surplus, E_deficit = manage_batteries(x[2], d[1], d[2], P_pump)

    reward = get_reward(E_surplus, E_deficit, x[3], u[0], V_Irr_next, d[0], x[4], u[1])
    reward = reward - irrigation_penalty - extraction_penalty
    return np.array([V_Irr_next, V_tank_next, SoE_next, u[0], u[1]]), reward

next_state(np.array([0, 2, 40, 0, 0]), np.array([50, 0]), np.array([4.0, 0, 0]))

def traj_total_reward(initial_state, actions, horizon, set_of_disturbances):
    x = initial_state
    total_reward = 0
    num_states = initial_state.shape[0]
    set_of_states = np.zeros((num_states, horizon))
    set_of_rewards = np.zeros(horizon)
    for i in range(horizon):
        set_of_states[:,i], set_of_rewards[i] = next_state(x, 
                                                         np.array([actions[i], actions[horizon+i]]),
                                                         set_of_disturbances[:,i])
    
    return -np.sum(set_of_rewards)      


p_fv = solar_power(get_rad(), get_temperatura())
p_demanded = get_demand()
v_ref = np.ones(144)*3
dis = np.zeros((144, 3))
for i in range(144):
    dis[i, :] = np.array([v_ref[i], p_fv[i], p_demanded[i]])

initial_guess = 100*np.ones((1000, 144*2), dtype = float)
initial_condition = np.array([0, 2, 50, 0, 0])
horizon = 144
constraints = ({'type': 'ineq', 'fun': lambda x:  x},
               {'type': 'ineq', 'fun': lambda x: 100 - x})
solution = minimize(fun=lambda u: traj_total_reward(initial_condition, u, 144,dis),  
                    x0 = initial_guess, 
                    constraints = constraints)
solution
