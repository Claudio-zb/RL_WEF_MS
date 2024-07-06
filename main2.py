import torch
import numpy as np
from environments.EMS_env import DiscreteEMSEnv, ContinousEMSEnv, drawndown
from matplotlib import pyplot as plt
import glob
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Build Env

env = DiscreteEMSEnv()
max_e_steps = 288 #env._max_episode_steps
N = 60

file_paths = glob.glob('./models/*')
L = len(file_paths)
results = np.zeros((4, 50))

for idx, file_path in enumerate(file_paths):
    print(idx)
    if idx >=50:
        break
    print(file_path)
    policy = torch.load(file_path, map_location=device)

    def true_policy(state):
        if type(state) is np.ndarray:
            state = state.flatten()
            state = torch.tensor(state, dtype=torch.float32).to(device)
        return state #.cpu().data.numpy().flatten()
    
    hydric_request_error = np.zeros(N)
    E_buy = np.zeros(N)

    for i in range(N//2):
        states, actions, rewards = env.sample_trajectory(true_policy)
        error1, error2 = states[143,0]  - states[143,1], states[287,0] - states[287,1]
        hydric_request_error[2*i], hydric_request_error[2*i+1] = error1, error2 #first metric done
        E_buy = np.minimum(states[1:,-1], np.zeros_like(states[1:,-1])).sum()


    results[0, idx] = hydric_request_error.mean()
    results[1, idx] = hydric_request_error.std()
    results[2, idx] = E_buy

print("a)")

