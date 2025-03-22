from typing import Callable, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
import gym
from cs285.infrastructure import pytorch_util as ptu


class ModelBasedAgent(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        make_dynamics_model: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        ensemble_size: int,
        mpc_horizon: int,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.env = env
        self.mpc_horizon = mpc_horizon
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        assert mpc_strategy in (
            "random",
            "cem",
        ), f"'{mpc_strategy}' is not a valid MPC strategy"

        # ensure the environment is state-based
        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1

        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        self.ensemble_size = ensemble_size
        self.dynamics_models = nn.ModuleList(
            [
                make_dynamics_model(
                    self.ob_dim,
                    self.ac_dim,
                )
                for _ in range(ensemble_size)
            ]
        )
        self.optimizer = make_optimizer(self.dynamics_models.parameters())
        self.loss_fn = nn.MSELoss()

        # keep track of statistics for both the model input (obs & act) and
        # output (obs delta)
        self.register_buffer(
            "obs_acs_mean", torch.zeros(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_acs_std", torch.ones(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_mean", torch.zeros(self.ob_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_std", torch.ones(self.ob_dim, device=ptu.device)
        )

    def update(self, i: int, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            next_obs: (batch_size, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        # TODO(student): update self.dynamics_models[i] using the given batch of data
        # HINT: make sure to normalize the NN input (observations and actions)
        # *and* train it with normalized outputs (observation deltas) 
        # HINT 2: make sure to train it with observation *deltas*, not next_obs
        # directly
        # HINT 3: make sure to avoid any risk of dividing by zero when
        # normalizing vectors by adding a small number to the denominator!
        norm_diff = (next_obs - obs - self.obs_delta_mean) / (self.obs_delta_std + 1e-6)
        ob_acs = torch.cat([obs, acs], dim=1)
        ob_acs = (ob_acs - self.obs_acs_mean) / (self.obs_acs_std + 1e-6)
        loss = torch.nn.functional.mse_loss(self.dynamics_models[i].forward(ob_acs), norm_diff)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return ptu.to_numpy(loss)

    @torch.no_grad()
    def update_statistics(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update the statistics used to normalize the inputs and outputs of the dynamics models.

        Args:
            obs: (n, ob_dim)
            acs: (n, ac_dim)
            next_obs: (n, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        # TODO(student): update the statistics
        self.obs_acs_mean = torch.mean(torch.cat([obs, acs], dim=1), dim=0)
        self.obs_acs_std = torch.std(torch.cat([obs, acs], dim=1), dim=0)
        self.obs_delta_mean = torch.mean(next_obs - obs, dim=0)
        self.obs_delta_std = torch.std(next_obs - obs, dim=0)

    @torch.no_grad()
    def get_dynamics_predictions(
        self, i: int, obs: np.ndarray, acs: np.ndarray
    ) -> np.ndarray:
        """
        Takes a batch of each current observation and action and outputs the
        predicted next observations from self.dynamics_models[i].

        Args:
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
        Returns: (batch_size, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        # TODO(student): get the model's predicted `next_obs`
        # HINT: make sure to *unnormalize* the NN outputs (observation deltas)
        # Same hints as `update` above, avoid nasty divide-by-zero errors when
        # normalizing inputs!
        ob_acs = torch.cat([obs, acs], dim=1)
        ob_acs = (ob_acs - self.obs_acs_mean) / (self.obs_acs_std + 1e-6)
        delta = self.dynamics_models[i].forward(ob_acs)
        delta = delta * (self.obs_delta_std + 1e-6) + self.obs_delta_mean
        assert delta.shape == obs.shape, delta.shape
        return ptu.to_numpy(obs + delta)

    def evaluate_action_sequences(self, obs: np.ndarray, action_sequences: np.ndarray):
        """
        Evaluate a batch of action sequences using the ensemble of dynamics models.

        Args:
            obs: starting observation, shape (ob_dim,)
            action_sequences: shape (mpc_num_action_sequences, horizon, ac_dim)
        Returns:
            sum_of_rewards: shape (mpc_num_action_sequences,)
        """
        # We are going to predict (ensemble_size * mpc_num_action_sequences)
        # distinct rollouts, and then average over the ensemble dimension to get
        # the reward for each action sequence.

        # We start by initializing an array to keep track of the reward for each
        # of these rollouts.
        sum_of_rewards = np.zeros(
            (self.ensemble_size, self.mpc_num_action_sequences), dtype=np.float32
        )
        assert obs.shape == (self.ob_dim,), obs.shape
        # We need to repeat our starting obs for each of the rollouts.
        obs = np.tile(obs, (self.ensemble_size, self.mpc_num_action_sequences, 1))

        # TODO(student): for each batch of actions in in the horizon...
        action_sequences = action_sequences.swapaxes(0, 1)
        for acs in action_sequences:
            assert acs.shape == (self.mpc_num_action_sequences, self.ac_dim), acs.shape
            assert obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            # TODO(student): predict the next_obs for each rollout
            # HINT: use self.get_dynamics_predictions
            next_obs = np.stack([self.get_dynamics_predictions(i, obs[i], acs) for i in range(self.ensemble_size)])
            assert next_obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            # TODO(student): get the reward for the current step in each rollout
            # HINT: use `self.env.get_reward`. `get_reward` takes 2 arguments:
            # `next_obs` and `acs` with shape (n, ob_dim) and (n, ac_dim),
            # respectively, and returns a tuple of `(rewards, dones)`. You can 
            # ignore `dones`. You might want to do some reshaping to make
            # `next_obs` and `acs` 2-dimensional.
            all_obs = np.concatenate(next_obs, axis=0)
            assert all_obs.shape == (self.ensemble_size * self.mpc_num_action_sequences, self.ob_dim), all_obs.shape
            acs_repeat = np.repeat(acs, self.ensemble_size, axis=0)
            assert acs_repeat.shape[:1] == all_obs.shape[:1]
            rewards, _ = self.env.get_reward(all_obs, acs_repeat)
            rewards = np.reshape(rewards, (self.ensemble_size, self.mpc_num_action_sequences))
            assert rewards.shape == (self.ensemble_size, self.mpc_num_action_sequences), rewards.shape

            sum_of_rewards += rewards

            obs = next_obs

        # now we average over the ensemble dimension
        return sum_of_rewards.mean(axis=0)

    def get_action(self, obs: np.ndarray):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # always start with uniformly random actions
        action_sequences = np.random.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
            size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim),
        )

        if self.mpc_strategy == "random":
            # evaluate each action sequence and return the best one
            rewards = self.evaluate_action_sequences(obs, action_sequences)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = np.argmax(rewards)
            return action_sequences[best_index][0]
        elif self.mpc_strategy == "cem":
            elite_mean, elite_std = None, None
            for i in range(self.cem_num_iters):
                rewards = self.evaluate_action_sequences(obs, action_sequences)
                elites = np.argsort(rewards)[-self.cem_num_elites:]
                assert elites.shape == (self.cem_num_elites,)
                new_elite_actions = action_sequences[elites]
                if i == 0:
                    # others = np.argsort(rewards)[:-self.cem_num_elites]
                    elite_mean = np.mean(action_sequences, axis=0)
                    elite_std = np.std(action_sequences, axis=0)
                    assert elite_mean.shape == (self.mpc_horizon, self.ac_dim)

                new_mean = np.mean(new_elite_actions, axis=0)
                new_std = np.std(new_elite_actions, axis=0)
                elite_mean = elite_mean * (1 - self.cem_alpha) + self.cem_alpha * new_mean
                elite_std = elite_std * (1 - self.cem_alpha) + self.cem_alpha * new_std
                new_sample = np.random.normal(elite_mean, elite_std, action_sequences.shape)
                assert new_sample.shape == action_sequences.shape
                action_sequences = new_sample
                # TODO(student): implement the CEM algorithm
                # HINT: you need a special case for i == 0 to initialize
                # the elite mean and std
            return elite_mean[0]
        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")
