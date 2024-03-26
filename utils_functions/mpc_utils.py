import do_mpc
import numpy as np
import matplotlib.pyplot as plt
from casadi import *

model_type = 'discrete' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

# States

x = model.set_variable(var_type='_x', var_name='x', shape=(1,1))
x_prev = model.set_variable(var_type='_x', var_name='x_prev', shape=(1,1))
u_prev = model.set_variable(var_type='_x', var_name='u_prev', shape=(1,1))
ref = model.set_variable(var_type='_x', var_name='ref', shape=(1,1))
du = model.set_variable(var_type='_u', var_name='du', shape=(1,1))

##

bound = 100
x_next = 1.1*x - 0.1*x_prev**2 + 0.8*(du + u_prev)**2
model.set_rhs('x_prev', x)
model.set_rhs('x', x_next.fmax(-bound).fmin(bound))
model.set_rhs('u_prev', u_prev + du)
model.set_rhs('ref', ref)
model.setup()

##

mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': 10,
    't_step': 1,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

tvp_template = mpc.get_tvp_template()

mterm = (x-ref)**2 
lterm = (x-ref)**2 


mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(du=0.7)


simulator = do_mpc.simulator.Simulator(model)
setup_simulator = {
    't_step': 1,
}

sim_tvp_template = simulator.get_tvp_template()


simulator.set_param(**setup_simulator)

simulator.setup()
mpc.setup()

    