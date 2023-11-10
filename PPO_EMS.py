import torch
from Networks.networks import ValueNN, ActorNN
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch import nn, distributions
import numpy as np
import pickle
from EMS_env import EMS_env
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from Funciones.train_utils import sample_trajectory


class PPO_EMS:
    def __init__(self, env):

        self._init_hyperparameters()
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.actor = ActorNN(self.obs_dim, self.action_dim).cuda()
        self.value = ValueNN(self.obs_dim, 1).cuda()

        # creating covariance matrix to use get_action() method
        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.65)
        self.cov_mat = torch.diag(self.cov_var).cuda()
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.value.parameters(), lr=self.lr)
        self.stats = {}
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        self.fig = fig
        self.axs = axs

        if torch.cuda.is_available():
            self.actor.cuda()
            self.value.cuda()
            self.cov_mat.cuda()

    def learn(self, max_iter):
        self.value.train()

        # if GPU is to be used
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        exploration_decay = 0.075 ** (1 / max_iter)
        k = 0
        means = np.zeros(max_iter)
        variances = np.zeros(max_iter)
        durations = np.zeros(max_iter)
        total_rewards = np.zeros(max_iter)
        self.stats = {'means': means
            , 'vars': variances
            , 'durations': durations
            , 'episode_reward': total_rewards}

        best_reward_mean = -np.inf
        best_reward_std = np.inf
        best_ep_reward = -np.inf
        while k < max_iter:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, cum_rews, batch_rews, batch_vals, batch_dones = self.rollout()

            # update the statistics
            traj_reward_mean = cum_rews.mean()
            traj_reward_var = cum_rews.var()
            self.stats['episode_reward'][k] = float(cum_rews.mean())  # np.array(batch_rews).mean()
            self.stats['means'][k] = traj_reward_mean.float()
            self.stats['vars'][k] = traj_reward_var.float()
            self.stats['durations'][k] = np.array(batch_lens).mean()

            if traj_reward_mean > best_reward_mean and traj_reward_var < best_reward_std and np.array(
                    batch_lens).mean() >= 200:
                best_reward_mean = traj_reward_mean
                traj_reward_var = traj_reward_var
                torch.save(self.actor, "models/policy_best_a.pt")
                print("best cumulative reward so far")

            if self.stats['episode_reward'][k] > best_ep_reward:
                best_ep_reward = self.stats['episode_reward'][k]
                torch.save(self.actor, "models/policy_best_ep.pt")
                print("best episode reward so far")

            print(traj_reward_mean.float(), f" n_iter: {k}")
            print(f"mean duration: {self.stats['durations'][k]}")
            print(f"mean reward: {self.stats['episode_reward'][k]}")
            print("--------------------------------------------------")

            if k > 0 and k % 10 == 0: # update the plots
                clear_output(wait=True)
                traj, action, y_ref = sample_trajectory(self.actor)
                self.axs[0].clear()
                self.axs[1].clear()
                self.axs[2].clear()
                self.axs[3].clear()

                self.axs[0].plot(traj, label='Trajectory')  # Follow the reference
                self.axs[0].hlines(y_ref, 0, len(traj), label='Reference')
                self.axs[1].plot(action, label='action u')
                display(self.fig)
                # clear_output(wait=True)
                plt.pause(0.2)

            if k >= 5:
                last_durations = self.stats['durations'][k - 10:k]
                if np.all(last_durations) >= 200:
                    torch.save(self.actor, "models/policy_stable_.pt")
                    break

            # we need now update the parameters of V(s, a) and pi(a|s)
            scheduled = False
            if self.stats['durations'][k] and not scheduled >= 200:
                self.actor_optim.param_groups[0]["lr"] = 0.0001
                scheduled = True

            V, _, _ = self.evaluate(batch_obs, batch_acts)

            # Calculate Advantage

            V = self.value(batch_obs).squeeze()
            A_k = batch_rtgs - V.detach()  # ALG STEP 5
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # La parte chida

            for j in range(self.n_epochs_policy):  # update policy
                _, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # calculate surrogate losses
                self.actor_optim.zero_grad()
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_loss = (-torch.min(surr1, surr2)).mean()
                entropy_loss = entropy.mean()
                actor_loss = actor_loss - self.ent_coef * entropy_loss

                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

            for j in range(self.n_epochs_critic):  # update critics

                # Calculate V_phi and pi_theta(a_t | s_t)
                self.critic_optim.zero_grad()
                V, curr_log_probs, _ = self.evaluate(batch_obs, batch_acts)

                critic_loss = nn.MSELoss()(V, batch_rtgs)
                # Calculate gradients and perform backward propagation for critic network

                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)  # Clip gradients
                self.critic_optim.step()
            frac = k / max_iter
            new_lr = self.lr * (1.0 - frac)
            new_lr = max(new_lr, 5e-5)
            self.actor_optim.param_groups[0]["lr"] = new_lr
            self.critic_optim.param_groups[0]["lr"] = new_lr

            self.cov_var = self.cov_var * exploration_decay
            self.cov_mat = torch.diag(self.cov_var).cuda()

            k += 1
        torch.save(self.actor, "models/policy_v2_a.pt")
        print("max iter reached")
        with open('weas/data.pkl', 'wb') as file:
            pickle.dump(self.stats, file)
        return self.stats

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1]) - ep_vals[t]
                else:
                    delta = torch.tensor(ep_rews[t]).cuda() - ep_vals[t]

                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float).cuda()

    def evaluate(self, batch_obs, batch_acts):
        # Calculate log probabilities using the most recent network
        mean = self.actor(batch_obs)
        # cov_matrix = torch.diag(std.squeeze(1)).cuda()
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts.squeeze(1)).cuda()
        self.value.eval()
        V = self.value(batch_obs).squeeze()

        return V, log_probs, dist.entropy()

    def get_action2(self, obs):

        # obtain the mean action value from the actor network
        mean, _ = self.actor(obs)
        # create a distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # sample an action
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().cpu().numpy(), log_prob.detach()



    def _init_hyperparameters(self):
        self.timesteps_per_batch = 3000
        self.max_timesteps_per_episodes = 200
        self.episodes_per_batch = 15
        self.gamma = 0.91
        self.n_epochs_critic = 10
        self.n_epochs_policy = 6
        self.clip = 0.2
        self.lr = 0.001
        self.ent_coef = 0.01
        self.max_grad_norm = 0.5
        self.lam = 0.95

    def rollout(self) -> dict:
        batch_results = {'batch_obs': [],
                        'batch_acts': [],
                        'batch_log_probs': [],
                        'batch_rtgs': [],
                        'batch_lens': [],
                        'cumulative_rewards': [],
                        'batch_rews': [],
                        'batch_vals': [],
                        'batch_dones': [],
                        'ep_dones': []}

        # collect a set of trajectories. This set is called batch

        i = 0
        env = self.env
        total_timesteps = 0
        while total_timesteps < self.timesteps_per_batch:  # while i < self.episodes_per_batch:
            # rewards from this trajectory
            traj_rews = []
            ep_vals = []
            ep_dones = []
            obs, info = env.reset()
            done = False
            aux_eps = 0
            for eps_t in range(self.max_timesteps_per_episodes):  # for eps_t in range(self.max_timesteps_per_episodes):
                ep_dones.append(done)
                aux_eps = eps_t
                # collect the current observation
                batch_results["batch_obs"].append(obs)

                # compute the action
                action, log_prob = self.get_action(obs)  # env.action_space.sample()
                self.value.eval()
                val = self.value(obs)
                # go to the next state
                obs, rew, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # collect the data into the batch (reward, action and log(prob) )
                traj_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_results["batch_acts"].append(action)
                batch_results["batch_log_probs"].append(log_prob)

                if done:
                    total_timesteps += aux_eps + 1
                    break

            batch_results["batch_lens"].append(aux_eps + 1)
            batch_results["batch_rews"].append(traj_rews)
            batch_results["batch_vals"].append(ep_vals)
            batch_results["batch_dones"].append(ep_dones)
            i += 1

        # reshape data as a tensor
        batch_results["batch_obs"] = torch.tensor(np.array(batch_results["batch_obs"]), dtype=torch.float32).cuda()
        batch_results["batch_acts"] = torch.tensor(np.array(batch_results["batch_acts"]), dtype=torch.float32).cuda()
        batch_results["batch_log_probs"] = torch.tensor(batch_results["batch_log_probs"], dtype=torch.float32).cuda()

        # for step 4, implement Rewards to-go
        batch_rtgs, cumulative_rewards, total_rews = self.compute_rgts(batch_results["batch_rews"])

        return batch_results

    def compute_rgts(self, batch_rews) -> (torch.tensor, torch.tensor):
        gamma = self.gamma
        batch_rgts = []
        cumulative_rewards = []
        total_rewards = []
        # we start with the last episode rewards and go backwards
        for traj_rews in reversed(batch_rews):
            total_rewards.append(np.sum(traj_rews))
            discounted_reward = 0
            # we start with the last reward of the episode and go backwards
            for rew in reversed(traj_rews):
                discounted_reward = rew + discounted_reward * gamma
                # we add the discounted reward to the list of rewards to go (cumulative rewards)
                batch_rgts.insert(0, discounted_reward)
            cumulative_rewards.insert(0, discounted_reward)
        batch_rgts = torch.tensor(batch_rgts, dtype=torch.float32).cuda()
        cumulative_rewards = torch.tensor(cumulative_rewards, dtype=torch.float32).cpu()
        return batch_rgts.squeeze(1), cumulative_rewards, total_rewards

    def get_set_of_reward(self):
        pass


env = EMS_env()
model = PPO_EMS(env)
model.learn(500)
