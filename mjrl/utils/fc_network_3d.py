import numpy as np
import torch
import torch.nn as nn


class FCNetwork3D(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64,64),
                 nonlinearity='tanh',   # either 'tanh' or 'relu'
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None):
        super(FCNetwork3D, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (obs_dim, ) + hidden_sizes + (act_dim, )
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.conv_1 = nn.Sequential(nn.Conv3d(1, 1, 3, padding=1), nn.ReLU()) # out: 32 
        self.conv_1_1 = nn.Sequential(nn.Conv3d(1, 1, 3, padding=1), nn.ReLU()) # out: 32 
        self.conv_2 = nn.Sequential(nn.Conv3d(1, 1, 3, padding=1), nn.ReLU()) # out: 16 
        self.conv_2_1 = nn.Sequential(nn.Conv3d(1, 1, 3, padding=1), nn.ReLU()) # out: 16 
        self.conv_3 = nn.Sequential(nn.Conv3d(1, 1, 3, padding=1), nn.ReLU()) # out: 8 
        self.conv_3_1 = nn.Sequential(nn.Conv3d(1, 1, 3, padding=1), nn.ReLU()) # out: 8
        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                         for i in range(len(self.layer_sizes) -1)])
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(in_shift=in_shift,
                           in_scale=in_scale,
                           out_shift=out_shift,
                           out_scale=out_scale
                          )
        self.in_shift  = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale  = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)

    def forward(self, x):
        # TODO(Aravind): Remove clamping to CPU
        # This is a temp change that should be fixed shortly
        if x.is_cuda:
            out = x.to('cpu')
        else:
            out = x
        voxel_obs = out[:, -216:].reshape(1,1,6,6,6)
        voxel_obs_1 = self.conv_1(voxel_obs)
        voxel_obs_1_1 = self.conv_1_1(voxel_obs_1)
        voxel_obs_2 = self.conv_2(voxel_obs_1_1)
        voxel_obs_2_1 = self.conv_2_1(voxel_obs_2)
        voxel_obs_3 = self.conv_3(voxel_obs_2_1)
        voxel_obs_3_1 = self.conv_3_1(voxel_obs_3)
        out_obs = voxel_obs_3_1.reshape(1, 216)
        # print(">>> out {} out_obs{}".format(out.shape, out_obs.shape))
        out[:, -216:] = out_obs[:, :]
        out = (out - self.in_shift)/(self.in_scale + 1e-8)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        out = out * self.out_scale + self.out_shift
        return out
