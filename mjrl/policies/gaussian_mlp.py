import numpy as np
from mjrl.utils.fc_network import FCNetwork
import torch
from torch.autograd import Variable


class MLP:
    def __init__(self, env_spec,
                 hidden_sizes=(64,64),
                 min_log_std=-3,
                 init_log_std=0,
                 m_f=1,
                 n_f=1,
                 reinitialize=False,
                 seed=None):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation, if None, it will be the output of MLP
        :param seed: random seed
        """
        self.n = env_spec.observation_dim  # number of states
        self.m = env_spec.action_dim  # number of actions
        self.min_log_std = min_log_std
        self.mean_factor = m_f
        self.noise_factor = n_f
        self.reinitialize = reinitialize

        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        output_dim = 2 * self.m if init_log_std is None else self.m

        # Policy network
        # ------------------------
        self.model = FCNetwork(self.reinitialize, self.n, output_dim, hidden_sizes)
        # make weights small
        for param in list(self.model.parameters())[-2:]:  # only last layer
           param.data = 1e-2 * param.data
        self.trainable_params = list(self.model.parameters())
        self.log_std = None
        if not init_log_std is None:
            self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)
            self.trainable_params = self.trainable_params + [self.log_std]

        # Old Policy network
        # ------------------------
        self.old_model = FCNetwork(self.reinitialize, self.n, output_dim, hidden_sizes)
        self.old_params = list(self.old_model.parameters())
        self.old_log_std = None
        if not init_log_std is None:
            self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
            self.old_params = self.old_params + [self.old_log_std]
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()

        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel()) if not self.log_std is None else np.nan
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                 for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params, set_new=True, set_old=True):
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            if not self.log_std is None:
                self.trainable_params[-1].data = \
                    torch.clamp(self.trainable_params[-1], self.min_log_std).data
            # update log_std_val for sampling
            if not self.log_std is None:
                self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            if not self.old_log_std is None:
                self.old_params[-1].data = \
                    torch.clamp(self.old_params[-1], self.min_log_std).data

    # Main functions
    # ============================================
    def get_action(self, observation):
        o = np.float32(observation.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        mean = self.model(self.obs_var).data.numpy().ravel()
        if self.log_std is None:
            self.log_std_val = mean[..., self.m:]
            mean = mean[..., :self.m]
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise
        # print(">>> mean{}".format(mean))
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]

    def get_action_eval(self, observation):
        o = np.float32(observation.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        mean = self.model(self.obs_var).data.numpy().ravel()
        if self.log_std is None:
            self.log_std_val = mean[..., self.m:]
            mean = mean[..., :self.m]
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise
        # print(">>> mean{}".format(mean))
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]

    def mean_LL(self, observations, actions, model=None, log_std=None):
        model = self.model if model is None else model
        if type(observations) is not torch.Tensor:
            obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False)
        else:
            obs_var = observations
        if type(actions) is not torch.Tensor:
            act_var = Variable(torch.from_numpy(actions).float(), requires_grad=False)
        else:
            act_var = actions
        mean = model(obs_var) * self.mean_factor
        if self.log_std is None:
            self.log_std_var = mean[..., self.m:]
            mean = mean[..., :self.m]
        log_std = self.log_std_var if log_std is None else log_std
        zs = (act_var - mean) / torch.exp(log_std)
        LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
             - torch.sum(log_std) + \
             - 0.5 * self.m * np.log(2 * np.pi)
        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):
        mean, LL = self.mean_LL(observations, actions, model, log_std)
        return LL.data.numpy()

    def old_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std)
        if self.old_log_std is None:
            log_std_var = self.log_std_var
            del self.log_std_var # avoid being copied when recognized as a best policy
            return [LL, mean, log_std_var]
        else:
            return [LL, mean, self.old_log_std]

    def new_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std)
        if self.log_std is None:
            log_std_var = self.log_std_var
            del self.log_std_var # avoid being copied when recognized as a best policy
            return [LL, mean, log_std_var]
        else:
            return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):
        old_log_std = old_dist_info[2]
        new_log_std = new_dist_info[2]
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]
        new_mean = new_dist_info[1]
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)
