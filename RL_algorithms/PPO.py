import torch
from EMS_networks import ValueNN, ActorNN
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch import nn
import numpy as np
from environments.custom_env import Custom_env
from utils_functions.train_utils import get_action
from RL_algorithms.RL_algorithm import RL_algorithm
from utils_functions.ReplayMemory import PPOReplayMemory, PPOTransition


class PPO(RL_algorithm):
    """
    This class implements the PPO algorithm with Experience Replay. It can handle continuous action spaces.
    """

    def update_training_plots(self, i_episode: int):
        pass

    def __init__(self, env: Custom_env, options=None):
        self._init_hyperparameters(options)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        policy_low = env.action_low
        policy_high = env.action_high
        self.policy = ActorNN(self.obs_dim, self.action_dim, policy_low, policy_high).cuda()
        self.value = ValueNN(self.obs_dim).cuda()
        self.memory = PPOReplayMemory(20000)

        # creating covariance matrix to use get_action() method
        self.cov_var = torch.full(size=(self.action_dim,), fill_value=20.0)
        self.cov_mat = torch.diag(self.cov_var).cuda()
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.value.parameters(), lr=self.lr)
        self.stats = {}

        self.scheduled = False

        if torch.cuda.is_available():
            self.policy.cuda()
            self.value.cuda()
            self.cov_mat.cuda()

    def learn(self, max_iter: int) -> tuple[dict[str, np.ndarray], ActorNN]:
        """
        This method implements the PPO algorithm
        :param max_iter: number of iterations to run the algorithm
        :return statistics and the policy
        """

        # if GPU is to be used
        exploration_decay = 0.075 ** (1 / max_iter)
        k = 0
        means = np.zeros(max_iter)
        variances = np.zeros(max_iter)
        durations = np.zeros(max_iter)
        total_rewards = np.zeros(max_iter)
        self.stats = {'means': means,
                      'vars': variances,
                      'durations': durations,
                      'episode_reward': total_rewards}

        best_reward_mean = -np.inf
        best_reward_std = np.inf
        best_ep_reward = -np.inf
        while k < max_iter:

            if k % 10 == 0:
                self.show_trajectory(self.policy)

            batch_results = self.rollout()
            # update the statistics
            traj_reward_mean = batch_results["cumulative_rewards"].mean()
            traj_reward_var = batch_results["cumulative_rewards"].var()
            self.stats['episode_reward'][k] = float(
                batch_results["cumulative_rewards"].mean())  # np.array(batch_rews).mean()
            self.stats['means'][k] = traj_reward_mean.float()
            self.stats['vars'][k] = traj_reward_var.float()
            self.stats['durations'][k] = np.array(batch_results["batch_lens"]).mean()

            if traj_reward_mean > best_reward_mean and traj_reward_var < best_reward_std and np.array(
                    batch_results["batch_lens"]).mean() >= 200:
                best_reward_mean = traj_reward_mean
                torch.save(self.policy, "./models/policy_best_a.pt")
                print("best cumulative reward so far")

            if self.stats['episode_reward'][k] > best_ep_reward:
                best_ep_reward = self.stats['episode_reward'][k]
                torch.save(self.policy, "../RL_EMS/models/policy_best_ep.pt")
                torch.save(self.value, "../RL_EMS/models/value_best_ep.pt")
                print("best episode reward so far")

            print(traj_reward_mean.float(), f" n_iter: {k}")
            print(f"mean duration: {self.stats['durations'][k]}")
            print(f"mean reward: {self.stats['episode_reward'][k]}")
            print("--------------------------------------------------")

            if k >= 5:
                last_durations = self.stats['durations'][k - 10:k]
                if np.all(last_durations) >= 200:
                    torch.save(self.policy, "./models/policy_stable_.pt")
                    break

            # we need now update the parameters of V(s, a) and pi(a|s)
            if self.stats['durations'][k] and not self.scheduled >= 200:
                self.policy_optim.param_groups[0]["lr"] = 0.0001
                self.scheduled = True

            # we need to sample the data from the replay memory

            transitions = self.memory.sample(self.timesteps_per_batch)

            batch = PPOTransition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                               if s is not None])

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action).to(self.device)
            reward_batch = torch.cat(batch.reward)
            reward_to_go_batch = torch.cat(batch.reward_to_go).to(self.device)


            # Calculate Advantage
            self.value.eval()
            V = self.value(state_batch).squeeze()
            A_k = reward_to_go_batch - V.detach()  # ALG STEP 5
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # The learning part

            # policy update
            self.policy.train()
            for j in range(self.n_epochs_policy):
                _, curr_log_probs, entropy = self.evaluate(state_batch, action_batch)
                ratios = torch.exp(curr_log_probs - batch_results["batch_log_probs"])  # P(a_t|s_t) / P_old(a_t|s_t)

                # calculate surrogate losses
                self.policy_optim.zero_grad()
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                policy_loss = (-torch.min(surr1, surr2)).mean()
                entropy_loss = entropy.mean()
                policy_loss = policy_loss - self.ent_coef * entropy_loss

                policy_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optim.step()

            self.policy.eval()
            self.value.train()

            # critic update
            for j in range(self.n_epochs_critic):
                # Calculate V_phi and pi_theta(a_t | s_t)
                self.critic_optim.zero_grad()
                V, curr_log_probs, _ = self.evaluate(batch_results["batch_obs"], batch_results["batch_actions"])

                critic_loss = nn.MSELoss()(V, batch_results["batch_rtgs"])
                # Calculate gradients and perform backward propagation for critic network

                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)  # Clip gradients
                self.critic_optim.step()

            self.value.eval()
            frac = k / max_iter
            new_lr = self.lr * (1.0 - frac)
            new_lr = max(new_lr, 5e-5)
            self.policy_optim.param_groups[0]["lr"] = new_lr
            self.critic_optim.param_groups[0]["lr"] = new_lr

            self.cov_var = self.cov_var * exploration_decay
            self.cov_mat = torch.diag(self.cov_var).cuda()

            k += 1
        torch.save(self.policy, "../RL_EMS/models/policy_final.pt")
        torch.save(self.value, "../RL_EMS/models/value_final.pt")
        print("max iter reached")
        self.env.close()

        self.value.eval()
        self.policy.eval()
        return self.stats, self.policy

    def show_trajectory(self, policy):
        self.env.show_sample(policy)

    def calculate_gae(self, rewards, values, dones) -> torch.Tensor:
        """
        This method calculates the Generalized Advantage Estimation
        :param rewards: rewards collected from the environment
        :param values: values predicted by the critic
        :param dones: a list of booleans indicating if the episode is done or not
        :return: the GAE
        """
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

    def evaluate(self, batch_obs: torch.Tensor, batch_acts: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Evaluates the current policy on a batch of observations and actions by computing the log probability
        :param batch_obs:
        :param batch_acts:
        :return:
        """
        # Calculate log probabilities using the most recent network
        # self.policy.train()
        mean = self.policy(batch_obs)
        # cov_matrix = torch.diag(std.squeeze(1)).cuda()
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts).cuda()
        self.value.eval()
        V = self.value(batch_obs).squeeze()

        return V, log_probs, dist.entropy()

    def rollout(self) -> dict[str, torch.Tensor]:
        """
        Collects data from the environment.
        :return: A dictionary containing batches of data.
        """
        self.policy.eval()
        self.value.eval()
        batch_results = {'batch_obs': [],
                         'batch_actions': [],
                         'batch_log_probs': [],
                         'batch_rtgs': [],
                         'batch_lens': [],
                         'cumulative_rewards': [],
                         'batch_rews': [],
                         'batch_values': [],
                         'batch_dones': [],
                         'ep_dones': []}

        # collect a set of trajectories. This set is called batch

        i = 0
        total_timesteps = 0
        while total_timesteps < self.timesteps_per_batch:  # while i < self.episodes_per_batch:
            traj_rews = []
            ep_values = []
            ep_dones = []
            done = False
            aux_eps = 0
            obs, info = self.env.reset()
            # for eps_t in range(self.max_timesteps_per_episodes):  # for eps_t in range(self.max_timesteps_per_episodes):
            while not done:
                ep_dones.append(done)

                batch_results["batch_obs"].append(obs)  # collect the current observation

                action, log_prob = get_action(self.policy, obs, self.cov_mat)  # compute the action and the log(prob)

                val = self.value(obs)

                obs, rew, terminated, truncated, info = self.env.step(action)  # transition to the next state
                done = terminated or truncated

                # collect the data into the batch (reward, action and log(prob) )
                traj_rews.append(rew)
                ep_values.append(val.flatten())
                batch_results["batch_actions"].append(action)
                batch_results["batch_log_probs"].append(log_prob)
                total_timesteps += 1

            batch_results["batch_lens"].append(aux_eps + 1)
            batch_results["batch_rews"].append(traj_rews)
            batch_results["batch_values"].append(ep_values)
            batch_results["batch_dones"].append(ep_dones)
            i += 1

        # reshape data as a tensor
        batch_results["batch_obs"] = torch.tensor(np.array(batch_results["batch_obs"]), dtype=torch.float32).cuda()
        batch_results["batch_actions"] = torch.tensor(np.array(batch_results["batch_actions"]),
                                                      dtype=torch.float32).cuda()
        batch_results["batch_log_probs"] = torch.tensor(batch_results["batch_log_probs"], dtype=torch.float32).cuda()

        # for step 4, implement Rewards to-go
        batch_rtgs, cumulative_rewards, total_rews = self.compute_rgts(batch_results["batch_rews"])
        batch_results["batch_rtgs"] = batch_rtgs
        batch_results["cumulative_rewards"] = cumulative_rewards
        batch_results["total_rewards"] = total_rews

        # step 4.1 add the values to the replay memory

        for index, _ in enumerate(batch_results["batch_obs"][:-1]):
            self.memory.push(batch_results["batch_obs"][index], batch_results["batch_actions"][index],
                             batch_results["batch_obs"][index + 1], batch_results["batch_rews"][index],
                             batch_results["batch_rtgs"][index])

        return batch_results

    def compute_rgts(self, batch_rews) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Computes the rewards to go. Plus, it computes the cumulative discounted rewards and the sum of the rewards.
        :param batch_rews: rewards collected from the environment
        :return: rewards to go, cumulative discounted rewards and the sum of the rewards
        """
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
        batch_rgts = torch.tensor(np.array(batch_rgts), dtype=torch.float32).cuda()
        cumulative_rewards = torch.tensor(np.array(cumulative_rewards), dtype=torch.float32).cpu()

        return batch_rgts.squeeze(1), cumulative_rewards, total_rewards

    def _init_hyperparameters(self, options=None):
        """
        This method initializes the hyperparameters of the algorithm
        :param options: a dictionary containing the hyperparameters
        :return:
        """
        if options is None:
            self.timesteps_per_batch = 3000
            self.max_timesteps_per_episodes = 288
            self.episodes_per_batch = 15
            self.gamma = 0.91
            self.n_epochs_critic = 10
            self.n_epochs_policy = 6
            self.clip = 0.2
            self.lr = 0.001
            self.ent_coef = 0.01
            self.max_grad_norm = 0.5
            self.lam = 0.95
        else:
            self.timesteps_per_batch = options['timesteps_per_batch']
            self.max_timesteps_per_episodes = options['max_timesteps_per_episodes']
            self.episodes_per_batch = options['episodes_per_batch']
            self.gamma = options['gamma']
            self.n_epochs_critic = options['n_epochs_critic']
            self.n_epochs_policy = options['n_epochs_policy']
            self.clip = options['clip']
            self.lr = options['lr']
            self.ent_coef = options['ent_coef']
            self.max_grad_norm = options['max_grad_norm']
            self.lam = options['lam']

    def get_set_of_reward(self):
        pass
