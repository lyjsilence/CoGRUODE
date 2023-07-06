import numpy as np
import torch
import torch.nn as nn
from itertools import chain
from torch.nn import Parameter
from torchdiffeq import odeint_adjoint as odeint

import utils

'''Marginal GRU-ODE Cell'''
class mgn_GRUODE(nn.Module):
    def __init__(self, input_size, n_dim, bias=True):
        super(mgn_GRUODE, self).__init__()
        self.input_size = input_size
        self.n_dim = n_dim

        self.U_r_c = Parameter(torch.randn(input_size, n_dim, n_dim))
        self.U_z_c = Parameter(torch.randn(input_size, n_dim, n_dim))
        self.U_h_c = Parameter(torch.randn(input_size, n_dim, n_dim))

        if bias:
            self.b_r_c = Parameter(torch.zeros(input_size, n_dim))
            self.b_z_c = Parameter(torch.zeros(input_size, n_dim))
            self.b_h_c = Parameter(torch.zeros(input_size, n_dim))
        else:
            self.register_parameter('b_r_c', None)
            self.register_parameter('b_z_c', None)
            self.register_parameter('b_h_c', None)

        self.reset_parameters()

    def forward(self, t, mgn_h):
        mgn_h = mgn_h.reshape(mgn_h.shape[0], self.input_size, self.n_dim)

        mgn_r = torch.sigmoid(torch.einsum("bij,ijk->bik", mgn_h, self.U_r_c) + self.b_r_c)
        mgn_z = torch.sigmoid(torch.einsum("bij,ijk->bik", mgn_h, self.U_z_c) + self.b_z_c)
        mgn_h_tilde = torch.tanh(torch.einsum("bij,ijk->bik", mgn_r * mgn_h, self.U_h_c) + self.b_h_c)

        mgn_h = (1 - mgn_z) * (mgn_h_tilde - mgn_h)
        mgn_h = mgn_h.reshape(mgn_h.shape[0], self.input_size * self.n_dim)

        return mgn_h

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.constant_(param.data, 0.0)
            elif 'U' in name:
                nn.init.orthogonal_(param.data)


'''Marginal GRU Cell'''

class mgn_GRU(nn.Module):
    def __init__(self, input_size, n_dim, bias=True):
        super(mgn_GRU, self).__init__()
        self.n_dim = n_dim
        self.input_size = input_size

        self.W_r_d = Parameter(torch.randn(input_size, 2, n_dim))
        self.W_z_d = Parameter(torch.randn(input_size, 2, n_dim))
        self.W_h_d = Parameter(torch.randn(input_size, 2, n_dim))
        self.U_r_d = Parameter(torch.randn(input_size, n_dim, n_dim))
        self.U_z_d = Parameter(torch.randn(input_size, n_dim, n_dim))
        self.U_h_d = Parameter(torch.randn(input_size, n_dim, n_dim))

        if bias:
            self.b_r_d = Parameter(torch.zeros(input_size, n_dim))
            self.b_z_d = Parameter(torch.zeros(input_size, n_dim))
            self.b_h_d = Parameter(torch.zeros(input_size, n_dim))
        else:
            self.register_parameter('b_r_d', None)
            self.register_parameter('b_z_d', None)
            self.register_parameter('b_h_d', None)

        self.reset_parameters()

    def forward(self, X, M, h):
        h = h.reshape(X.shape[0], self.input_size, self.n_dim)
        M = M.unsqueeze(2)
        h_clone = h.clone()
        r = torch.sigmoid(torch.einsum('bij,ijk->bik', X, self.W_r_d) + \
                          torch.einsum("bij,ijk->bik", h, self.U_r_d) + self.b_r_d)
        z = torch.sigmoid(torch.einsum('bij,ijk->bik', X, self.W_z_d) + \
                          torch.einsum("bij,ijk->bik", h, self.U_z_d) + self.b_z_d)
        h_tilde = torch.tanh(torch.einsum('bij,ijk->bik', X, self.W_h_d) + \
                             torch.einsum("bij,ijk->bik", r * h, self.U_h_d) + self.b_h_d)

        h = z * h + (1 - z) * h_tilde
        h = h_clone * (1 - M) + h * M
        h = h.reshape(X.shape[0], self.input_size * self.n_dim)

        return h

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.constant_(param.data, 0.0)
            elif 'U' in name:
                nn.init.orthogonal_(param.data)
            elif 'W' in name:
                nn.init.xavier_uniform_(param.data)


'''Marginal GRU-update Cell'''


class mgn_GRUBayes(nn.Module):
    def __init__(self, input_size, n_dim, bias=True):
        super().__init__()
        self.n_dim = n_dim
        self.mgn_GRU = mgn_GRU(input_size, n_dim, bias=bias)

    def forward(self, mgn_h, X_obs, M_obs, i_obs):
        temp_h = mgn_h.clone()
        temp_h[i_obs] = self.mgn_GRU(X_obs, M_obs, mgn_h[i_obs])
        mgn_h = temp_h
        return mgn_h

'''Marginal GRU-ODE Layer'''


class mgn_Cell(nn.Module):
    def __init__(self, input_size, n_dim, solver, device):
        super(mgn_Cell, self).__init__()
        self.solver = solver
        self.device = device

        # mgn_ode updates h and h_tilde by ODE function without data
        self.mgn_ode = mgn_GRUODE(input_size, n_dim)

        # gru_update updates h and h_tilde by GRU cell with observed data
        self.mgn_update = mgn_GRUBayes(input_size, n_dim)

    def forward(self, current_time, mgn_h, delta_t, X_obs=None, M_obs=None, i_obs=None, update=False):
        if update:
            assert X_obs is not None
            assert i_obs is not None

            mgn_h = self.mgn_update(mgn_h, X_obs, M_obs, i_obs)
        else:
            t_list = torch.cat([torch.as_tensor(current_time).unsqueeze(0),
                                torch.as_tensor(current_time + delta_t).unsqueeze(0)]).to(self.device)
            mgn_h = odeint(self.mgn_ode, mgn_h, t_list, method=self.solver)[-1, :, :]
        return mgn_h


'''Joint GRU-ODE Cell'''


class joint_GRUODE(nn.Module):
    def __init__(self, input_size, joint_hidden_size, n_dim):
        super(joint_GRUODE, self).__init__()
        self.input_size = input_size
        self.n_dim = n_dim

        self.U_r = Parameter(torch.Tensor(joint_hidden_size, joint_hidden_size))
        self.U_z = Parameter(torch.Tensor(joint_hidden_size, joint_hidden_size))
        self.U_h = Parameter(torch.Tensor(joint_hidden_size, joint_hidden_size))

        self.b_r = Parameter(torch.Tensor(joint_hidden_size))
        self.b_z = Parameter(torch.Tensor(joint_hidden_size))
        self.b_h = Parameter(torch.Tensor(joint_hidden_size))

        self.reset_parameters()

    def forward(self, t, joint_h):
        joint_r = torch.sigmoid(torch.mm(joint_h, self.U_r) + self.b_r)
        joint_z = torch.sigmoid(torch.mm(joint_h, self.U_z) + self.b_z)
        joint_h_tilde = torch.tanh(torch.mm(joint_h * joint_r, self.U_h) + self.b_h)
        joint_dh = (1 - joint_z) * (joint_h_tilde - joint_h)

        return joint_dh

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.constant_(param.data, 0.0)
            elif 'W' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'U' in name:
                nn.init.orthogonal_(param.data)


'''Joint GRU Cell'''


class joint_GRU(nn.Module):
    def __init__(self, input_size, joint_hidden_size, n_dim, bias=True):
        super(joint_GRU, self).__init__()

        self.joint_hidden_size = joint_hidden_size
        self.n_dim = n_dim

        self.W_r = Parameter(torch.Tensor(input_size * n_dim, joint_hidden_size))
        self.W_z = Parameter(torch.Tensor(input_size * n_dim, joint_hidden_size))
        self.W_h = Parameter(torch.Tensor(input_size * n_dim, joint_hidden_size))
        self.U_r = Parameter(torch.Tensor(joint_hidden_size, joint_hidden_size))
        self.U_z = Parameter(torch.Tensor(joint_hidden_size, joint_hidden_size))
        self.U_h = Parameter(torch.Tensor(joint_hidden_size, joint_hidden_size))

        if bias:
            self.b_r = Parameter(torch.Tensor(joint_hidden_size))
            self.b_z = Parameter(torch.Tensor(joint_hidden_size))
            self.b_h = Parameter(torch.Tensor(joint_hidden_size))
        else:
            self.register_parameter('b_r', None)
            self.register_parameter('b_z', None)
            self.register_parameter('b_h', None)
        self.reset_parameters()

    def forward(self, joint_h, mgn_h):
        joint_r = torch.sigmoid(torch.mm(mgn_h, self.W_r) + torch.mm(joint_h, self.U_r) + self.b_r)
        joint_z = torch.sigmoid(torch.mm(mgn_h, self.W_z) + torch.mm(joint_h, self.U_z) + self.b_z)
        joint_h_tilde = torch.tanh(torch.mm(mgn_h, self.W_h) + torch.mm(joint_h * joint_r, self.U_h) + self.b_h)
        joint_h = joint_h_tilde + joint_z * (joint_h - joint_h_tilde)
        return joint_h

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.constant_(param.data, 0.0)
            elif 'W' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'U' in name:
                nn.init.orthogonal_(param.data)


'''Joint GRU-update Cell'''


class joint_GRUBayes(nn.Module):

    def __init__(self, input_size, joint_hidden_size, n_dim, bias=True):
        super(joint_GRUBayes, self).__init__()
        self.GRUcell = joint_GRU(input_size, joint_hidden_size, n_dim, bias=bias)

        self.hidden_size = joint_hidden_size
        self.input_size = input_size
        self.n_dim = n_dim

    def forward(self, joint_h, mgn_h, i_obs):
        temp = joint_h.clone()
        temp[i_obs] = self.GRUcell(joint_h[i_obs], mgn_h[i_obs])
        h = temp
        return h


'''Joint GRU-ODE layer'''


class joint_Cell(nn.Module):
    def __init__(self, input_size, joint_hidden_size, n_dim, solver, device):
        super(joint_Cell, self).__init__()
        self.solver = solver
        self.device = device

        self.joint_ode = joint_GRUODE(input_size, joint_hidden_size, n_dim)
        self.joint_update = joint_GRUBayes(input_size, joint_hidden_size, n_dim)

    def forward(self, current_time, joint_h, delta_t, mgn_h=None, i_obs=None, update=False):
        if update:
            assert i_obs is not None

            joint_h = self.joint_update(joint_h, mgn_h, i_obs)
        else:
            t_list = torch.cat([torch.as_tensor(current_time).unsqueeze(0),
                                torch.as_tensor(current_time + delta_t).unsqueeze(0)]).to(self.device)
            joint_h = odeint(self.joint_ode, joint_h, t_list, method=self.solver)[-1, :, :]
        return joint_h


class CoGRUODE(nn.Module):
    def __init__(self, args, device):
        super(CoGRUODE, self).__init__()
        self.input_size = args.input_size
        self.n_dim = args.n_dim
        self.joint_hidden_size = args.n_dim * args.input_size
        self.solver = args.solver
        self.dropout = args.dropout
        self.device = device

        self.p_model = nn.Sequential(
            torch.nn.Linear(self.joint_hidden_size * 2, self.joint_hidden_size * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.joint_hidden_size * 2, self.input_size-1),
        )

        # construct marginal block
        self.mgn_Cell = mgn_Cell(self.input_size, self.n_dim, self.solver, device)

        # construct dependence/joint block
        self.joint_Cell = joint_Cell(self.input_size, self.joint_hidden_size, self.n_dim, self.solver, device)

        # initialize the classifier or p_model
        self.apply(self.init_weights)


    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def init_hidden_state(self, batch_size, mode='mgn'):
        if mode == 'mgn':
            return torch.zeros([batch_size, self.input_size * self.n_dim]).to(self.device)
        elif mode == 'joint':
            return torch.zeros([batch_size, self.joint_hidden_size]).to(self.device)


    def compute_loss(self, loss_mse, loss_mae, loss_mape, total_M_obs, X_obs, p_obs, M_obs):

        X_obs, M_obs = X_obs[:, 1:], M_obs[:, 1:]
        loss_mse = loss_mse + (torch.pow(X_obs - p_obs, 2) * M_obs).sum()
        loss_mae = loss_mae + (torch.abs(X_obs - p_obs) * M_obs).sum()
        loss_mape = loss_mape + (torch.abs(X_obs - p_obs)/(X_obs + 1e-8) * M_obs).sum()
        total_M_obs = total_M_obs + M_obs.sum()

        return loss_mse, loss_mae, loss_mape, total_M_obs

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, dt, return_path=False):

        '''
        :param obs_times: the total observed times in the multivariate time series
        :param event_pt: the number of events at each time step
        :param sample_idx: the reindex of samples in one batch
        :param X: data of multi-time series
        :param M: mask for each time series with 0 for no observations and 1 for observations
        :param batch_idx: the origin index of samples
        :param dt: the time step of integral
        :param return_path: whether return the whole predicted time series

        :return: MSE loss
        '''

        current_time = 0.0
        loss_mse, loss_mae, loss_mape = torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0)
        pred_list, true_list = [], []
        total_M_obs = 0

        if return_path:
            path = {}
            path['path_t'] = []
            path['path_p'] = []

        # initial the hidden state of marginal block and joint block
        mgn_h = self.init_hidden_state(sample_idx.shape[0], mode='mgn')
        joint_h = self.init_hidden_state(sample_idx.shape[0], mode='joint')

        for j, time in enumerate(obs_times):
            # if no observation at current time, using ODE function updates the hidden state
            while current_time < time:
                # integral and update the marginal hidden state
                mgn_h = self.mgn_Cell(current_time, mgn_h, dt)
                # integral and update the joint hidden state
                joint_h = self.joint_Cell(current_time, joint_h, dt)

                current_time = current_time + dt

            # update the hidden state when observations exists
            if current_time >= time:
                # get samples which have new observations at current_time
                start = event_pt[j]
                end = event_pt[j + 1]

                X_obs = X[start:end]
                M_obs = M[start:end]
                i_obs = batch_idx[start:end].type(torch.LongTensor)

                # predict the values variables at observed time points
                p = self.p_model(torch.cat([mgn_h, joint_h], dim=-1))

                if return_path:
                    path['path_t'].append(current_time)
                    path['path_p'].append(p)

                if current_time > 0:
                    loss_mse, loss_mae, loss_mape, total_M_obs \
                        = self.compute_loss(loss_mse, loss_mae, loss_mape, total_M_obs, X_obs[:, :, 0], p[i_obs], M_obs)

                # update h in marginal component
                mgn_h = self.mgn_Cell(current_time, mgn_h, dt, X_obs, M_obs, i_obs, update=True)

                # update h in joint component
                joint_h = self.joint_Cell(current_time, joint_h, dt, mgn_h, i_obs, update=True)


        if return_path:
            return np.array(path['path_t']), torch.stack(path['path_p']).detach().cpu().numpy().squeeze()
        else:
            return loss_mse / total_M_obs, loss_mae / total_M_obs, loss_mape / total_M_obs
