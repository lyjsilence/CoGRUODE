import numpy as np
import math
import utils
import torch
import torch.nn as nn
from torch.nn import Parameter
from odeint import odeint

'''Marginal GRU-ODE Cell'''


class mgn_GRUODECell(nn.Module):
    def __init__(self, input_size, n_dim, bias=True):
        super(mgn_GRUODECell, self).__init__()
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

        mgn_dh = (1 - mgn_z) * (mgn_h_tilde - mgn_h)
        mgn_dh = mgn_dh.reshape(mgn_h.shape[0], self.input_size * self.n_dim)
        mgn_h_tilde = mgn_h_tilde.reshape(mgn_h.shape[0], self.input_size * self.n_dim)

        return mgn_dh, mgn_h_tilde

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.constant_(param.data, 0.0)
            elif 'U' in name:
                nn.init.orthogonal_(param.data)


'''Marginal minimal GRU-ODE Cell'''


class mgn_mGRUODECell(nn.Module):
    def __init__(self, input_size, n_dim, bias=True):
        super(mgn_mGRUODECell, self).__init__()
        self.input_size = input_size
        self.n_dim = n_dim
        self.U_z_c = Parameter(torch.randn(input_size, n_dim, n_dim))
        self.U_h_c = Parameter(torch.randn(input_size, n_dim, n_dim))

        if bias:
            self.b_z_c = Parameter(torch.Tensor(input_size, n_dim))
            self.b_h_c = Parameter(torch.Tensor(input_size, n_dim))
        else:
            self.register_parameter('b_z_c', None)
            self.register_parameter('b_h_c', None)
        self.reset_parameters()

    def forward(self, t, mgn_h):
        mgn_h = mgn_h.reshape(mgn_h.shape[0], self.input_size, self.n_dim)
        mgn_z = torch.sigmoid(torch.einsum("bij,ijk->bik", mgn_h, self.U_z_c) + self.b_z_c)
        mgn_h_tilde = torch.tanh(torch.einsum("bij,ijk->bik", mgn_z * mgn_h, self.U_h_c) + self.b_h_c)
        mgn_dh = (1 - mgn_z) * (mgn_h_tilde - mgn_h)
        mgn_dh = mgn_dh.reshape(mgn_h.shape[0], self.input_size * self.n_dim)
        mgn_h_tilde = mgn_h_tilde.reshape(mgn_h.shape[0], self.input_size * self.n_dim)

        return mgn_dh, mgn_h_tilde

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.constant_(param.data, 0.0)
            elif 'U' in name:
                nn.init.orthogonal_(param.data)


'''Marginal GRU Cell'''


class GRUCell(nn.Module):
    def __init__(self, input_size, n_dim, bias=True):
        super(GRUCell, self).__init__()
        self.n_dim = n_dim
        self.input_size = input_size

        self.W_r_d = Parameter(torch.randn(input_size, 1, n_dim))
        self.W_z_d = Parameter(torch.randn(input_size, 1, n_dim))
        self.W_h_d = Parameter(torch.randn(input_size, 1, n_dim))
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

    def forward(self, X, h):
        h = h.reshape(X.shape[0], self.input_size, self.n_dim)
        r = torch.sigmoid(torch.einsum('bij,ijk->bik', X.unsqueeze(1).permute(0, 2, 1), self.W_r_d) + \
                          torch.einsum("bij,ijk->bik", h, self.U_r_d) + self.b_r_d)
        z = torch.sigmoid(torch.einsum('bij,ijk->bik', X.unsqueeze(1).permute(0, 2, 1), self.W_z_d) + \
                          torch.einsum("bij,ijk->bik", h, self.U_z_d) + self.b_z_d)
        h_tilde = torch.tanh(torch.einsum('bij,ijk->bik', X.unsqueeze(1).permute(0, 2, 1), self.W_h_d) + \
                             torch.einsum("bij,ijk->bik", r * h, self.U_h_d) + self.b_h_d)

        h = z * h + (1 - z) * h_tilde
        h = h.reshape(X.shape[0], self.input_size * self.n_dim)
        h_tilde = h_tilde.reshape(X.shape[0], self.input_size * self.n_dim)
        return h, h_tilde

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.constant_(param.data, 0.0)
            elif 'U' in name:
                nn.init.orthogonal_(param.data)

'''Marginal minimal GRU Cell'''


class mGRUCell(nn.Module):
    def __init__(self, input_size, n_dim, bias=True):
        super(mGRUCell, self).__init__()
        self.n_dim = n_dim
        self.input_size = input_size

        self.W_z_d = Parameter(torch.randn(input_size, 1, n_dim))
        self.W_h_d = Parameter(torch.randn(input_size, 1, n_dim))
        self.U_z_d = Parameter(torch.randn(input_size, n_dim, n_dim))
        self.U_h_d = Parameter(torch.randn(input_size, n_dim, n_dim))

        if bias:
            self.b_z_d = Parameter(torch.zeros(input_size, n_dim))
            self.b_h_d = Parameter(torch.zeros(input_size, n_dim))
        else:
            self.register_parameter('b_z_d', None)
            self.register_parameter('b_h_d', None)

    def forward(self, X, h):
        h = h.reshape(X.shape[0], self.input_size, self.n_dim)
        z = torch.sigmoid(torch.einsum('bij,ijk->bik', X.unsqueeze(1).permute(0, 2, 1), self.W_z_d) + \
                          torch.einsum("bij,ijk->bik", h, self.U_z_d) + self.b_z_d)
        h_tilde = torch.tanh(torch.einsum('bij,ijk->bik', X.unsqueeze(1).permute(0, 2, 1), self.W_h_d) + \
                             torch.einsum("bij,ijk->bik", z * h, self.U_h_d) + self.b_h_d)
        h = z * h + (1 - z) * h_tilde
        h = h.reshape(X.shape[0], self.input_size * self.n_dim)
        h_tilde = h_tilde.reshape(X.shape[0], self.input_size * self.n_dim)

        return h, h_tilde

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.constant_(param.data, 0.0)
            elif 'U' in name:
                nn.init.orthogonal_(param.data)

'''Marginal GRU-update Cell'''


class mgn_GRUODEObsCell(nn.Module):
    def __init__(self, input_size, n_dim, minimal, bias=True):
        super().__init__()
        self.n_dim = n_dim

        if minimal:
            self.GRUCell = mGRUCell(input_size, n_dim, bias=bias)
        else:
            self.GRUCell = GRUCell(input_size, n_dim, bias=bias)

    def forward(self, mgn_h, mgn_h_tilde, X_obs, i_obs):
        temp_h = mgn_h.clone()
        temp_h_tilde = mgn_h_tilde.clone()
        temp_h[i_obs], temp_h_tilde[i_obs] = self.GRUCell(X_obs, mgn_h[i_obs])
        mgn_h = temp_h
        mgn_h_tilde = temp_h_tilde
        return mgn_h, mgn_h_tilde


'''Marginal GRU-ODE Layer'''


class mgn_GRUODE(nn.Module):
    def __init__(self, input_size, n_dim, bias, solver, minimal, device):
        super(mgn_GRUODE, self).__init__()
        self.solver = solver
        self.device = device

        # mgn_ode updates h and h_tilde by ODE function without data
        if minimal:
            self.mgn_ode = mgn_mGRUODECell(input_size, n_dim, bias)
        else:
            self.mgn_ode = mgn_GRUODECell(input_size, n_dim, bias)

        # gru_update updates h and h_tilde by GRU cell with observed data
        self.mgn_update = mgn_GRUODEObsCell(input_size, n_dim, minimal, bias)

    def forward(self, current_time, mgn_h, delta_t, mgn_h_tilde=None, X_obs=None, i_obs=None, update=False):
        if update:
            assert X_obs is not None
            assert i_obs is not None
            assert mgn_h_tilde is not None

            mgn_h, mgn_h_tilde = self.mgn_update(mgn_h, mgn_h_tilde, X_obs, i_obs)
        else:
            t_list = torch.cat([torch.as_tensor(current_time).unsqueeze(0),
                                torch.as_tensor(current_time + delta_t).unsqueeze(0)]).to(self.device)
            mgn_h, mgn_h_tilde = odeint(self.mgn_ode, None, mgn_h, t_list, method=self.solver, mode='mgn')
        return mgn_h, mgn_h_tilde


'''Joint GRU-ODE Cell'''


class joint_GRUODECell(nn.Module):
    def __init__(self, input_size, joint_hidden_size, n_dim, bias=True):
        super(joint_GRUODECell, self).__init__()
        self.input_size = input_size
        self.n_dim = n_dim

        self.W_z = Parameter(torch.Tensor(input_size * n_dim, joint_hidden_size))
        self.W_h = Parameter(torch.Tensor(input_size * n_dim, joint_hidden_size))

        if bias:
            self.b_z = Parameter(torch.Tensor(joint_hidden_size))
            self.b_h = Parameter(torch.Tensor(joint_hidden_size))
        else:
            self.register_parameter('b_z', None)
            self.register_parameter('b_h', None)
        self.reset_parameters()

    def forward(self, t, mgn_h_tilde, joint_h):
        joint_z = torch.sigmoid(torch.mm(mgn_h_tilde, self.W_z) + self.b_z)
        joint_h_tilde = torch.tanh(torch.mm(mgn_h_tilde, self.W_h) + self.b_h)
        joint_dh = (1 - joint_z) * (joint_h_tilde - joint_h)

        return joint_dh

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.constant_(param.data, 0.0)
            elif 'W' in name:
                nn.init.xavier_uniform_(param.data)

'''Joint GRU Cell'''

class joint_GRUCell(nn.Module):
    def __init__(self, input_size, joint_hidden_size, n_dim, bias=True):
        super(joint_GRUCell, self).__init__()

        self.joint_hidden_size = joint_hidden_size
        self.n_dim = n_dim

        self.W_z = Parameter(torch.Tensor(input_size * n_dim, joint_hidden_size))
        self.W_h = Parameter(torch.Tensor(joint_hidden_size, joint_hidden_size))

        if bias:
            self.b_z = Parameter(torch.Tensor(joint_hidden_size))
            self.b_h = Parameter(torch.Tensor(joint_hidden_size))

        self.reset_parameters()

    def forward(self, joint_h, mgn_h_tilde):

        joint_z = torch.sigmoid(torch.mm(mgn_h_tilde, self.W_z) + self.b_z)
        joint_h_tilde = torch.tanh(torch.mm(mgn_h_tilde, self.W_h) + self.b_h)

        joint_h = joint_h_tilde + joint_z * (joint_h - joint_h_tilde)
        return joint_h

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.constant_(param.data, 0.0)
            elif 'W' in name:
                nn.init.xavier_uniform_(param.data)


'''Joint GRU-update Cell'''


class joint_GRUODEObsCell(nn.Module):

    def __init__(self, input_size, joint_hidden_size, n_dim, bias=True):
        super(joint_GRUODEObsCell, self).__init__()
        self.GRUcell = joint_GRUCell(input_size, joint_hidden_size, n_dim, bias=bias)

        self.hidden_size = joint_hidden_size
        self.input_size = input_size
        self.n_dim = n_dim

    def forward(self, joint_h, mgn_h_tilde, i_obs):
        temp = joint_h.clone()
        temp[i_obs] = self.GRUcell(joint_h[i_obs], mgn_h_tilde[i_obs])
        h = temp
        return h


'''Joint GRU-ODE layer'''


class joint_GRUODE(nn.Module):
    def __init__(self, input_size, joint_hidden_size, n_dim, bias, solver, device):
        super(joint_GRUODE, self).__init__()
        self.solver = solver
        self.device = device

        self.joint_ode = joint_GRUODECell(input_size, joint_hidden_size, n_dim, bias)
        self.joint_update = joint_GRUODEObsCell(input_size, joint_hidden_size, n_dim, bias)

    def forward(self, current_time, mgn_h_tilde, joint_h, delta_t, i_obs=None, update=False):
        if update:
            assert i_obs is not None

            joint_h = self.joint_update(joint_h, mgn_h_tilde, i_obs)
        else:
            t_list = torch.cat([torch.as_tensor(current_time).unsqueeze(0),
                                torch.as_tensor(current_time + delta_t).unsqueeze(0)]).to(self.device)
            joint_h, _ = odeint(self.joint_ode, mgn_h_tilde, joint_h, t_list, method=self.solver, mode='joint')
        return joint_h


class CoGRUODE(nn.Module):
    def __init__(self, CoGRUODE_params, device):
        super(CoGRUODE, self).__init__()
        self.input_size = CoGRUODE_params['input_size']
        self.joint_hidden_size = CoGRUODE_params['hidden_size']
        self.init_hidden_state = CoGRUODE_params['init_hidden_state']
        self.n_dim = CoGRUODE_params['n_dim']
        self.bias = CoGRUODE_params['bias']
        self.batch_size = CoGRUODE_params['batch_size']
        self.solver = CoGRUODE_params['solver']
        self.minimal = CoGRUODE_params['minimal']
        self.num_class = CoGRUODE_params['num_class']
        self.dropout = CoGRUODE_params['dropout']
        self.memory = CoGRUODE_params['memory']
        self.device = device

        # If the task is classification, we need build a classifier
        if self.num_class != None:
            # Using both marginal memory and dependence memory
            if self.memory == 'both':
                self.classify = nn.Sequential(
                    torch.nn.Linear(self.joint_hidden_size * 2, self.joint_hidden_size * 2, bias=self.bias),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(self.dropout),
                    torch.nn.Linear(self.joint_hidden_size * 2, self.num_class, bias=self.bias),
                )
            # Using only marginal memory or dependence memory
            else:
                self.classify = nn.Sequential(
                    torch.nn.Linear(self.joint_hidden_size, self.joint_hidden_size, bias=self.bias),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(self.dropout),
                    torch.nn.Linear(self.joint_hidden_size, self.num_class, bias=self.bias),
                )

        if self.memory == 'both':
            self.p_model = nn.Sequential(
                torch.nn.Linear(self.joint_hidden_size * 2, self.joint_hidden_size * 2, bias=self.bias),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.joint_hidden_size * 2, self.input_size, bias=self.bias),
            )
        else:
            self.p_model = nn.Sequential(
                torch.nn.Linear(self.joint_hidden_size, self.joint_hidden_size, bias=self.bias),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.joint_hidden_size, self.input_size, bias=self.bias),
            )

        # construct marginal block
        self.mgn_GRUODE = mgn_GRUODE(self.input_size, self.n_dim, self.bias, self.solver,self.minimal, device)

        # construct dependence/joint block
        self.joint_GRUODE = joint_GRUODE(self.input_size, self.joint_hidden_size, self.n_dim, self.bias, self.solver, device)

        # if initial hidden state
        if self.init_hidden_state:
            self.mgn_h = Parameter(torch.Tensor(self.batch_size, self.joint_hidden_size))
            self.joint_h = Parameter(torch.Tensor(self.batch_size, self.joint_hidden_size))
            self.reset_hidden()

        # initialize the classifier or p_model
        self.apply(self.init_weights)

    def reset_hidden(self) -> None:
        for name, param in self.named_parameters():
            if 'mgn_h' in name or 'joint_h' in name:
                nn.init.xavier_uniform_(param.data)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, device, T=None, return_path=False,
                classify=False, class_per_time=False, target=None, prop_to_end=True, loss_mae=True, dt=0.05):

        '''
        :param obs_times: the total observed times in the multivariate time series
        :param event_pt: the number of events at each time step
        :param sample_idx: the reindex of samples in one batch
        :param X: data of multi-time series
        :param M: mask for each time series with 0 for no observations and 1 for observations
        :param batch_idx: the origin index of samples
        :param device: cpu or gpu
        :param T: the final time for multivariate time series
        :param return_path: whether return the whole predicted time series
        :param dt: the time step of integral
        :param classify: whether there is a classification task
        :param target: the targets of classification task
        :return: loss of negative log likelihood and path if applicable
        '''

        current_time = 0.0
        loss_reg = torch.as_tensor(0.0)
        loss_CE = torch.as_tensor(0.0)
        total_M_obs = 0

        if class_per_time:
            if return_path:
                path = {}
                path['path_t'] = []
                path['path_p'] = []
                path['path_y'] = []
        else:
            if return_path:
                path = {}
                path['path_t'] = []
                path['path_p'] = []

        if self.init_hidden_state:
            mgn_h = self.mgn_h[0: sample_idx.shape[0]].to(device)
            joint_h = self.joint_h[0: sample_idx.shape[0]].to(device)
        else:
            mgn_h = torch.zeros([sample_idx.shape[0], self.joint_hidden_size]).to(device)
            joint_h = torch.zeros([sample_idx.shape[0], self.joint_hidden_size]).to(device)

        for j, time in enumerate(obs_times):
            # if no observation at current time, using ODE function updates the hidden state
            while current_time < time:
                # integral and update the marginal hidden state
                mgn_h, mgn_h_tilde = self.mgn_GRUODE(current_time, mgn_h, dt)
                # integral and update the joint hidden state
                joint_h = self.joint_GRUODE(current_time, mgn_h_tilde, joint_h, dt)

                if self.memory == 'both':
                    p = self.p_model(torch.cat([mgn_h, joint_h], dim=-1))
                elif self.memory == 'joint':
                    p = self.p_model(joint_h)
                elif self.memory == 'mgn':
                    p = self.p_model(mgn_h)

                current_time = current_time + dt

            # update the hidden state when observations exists
            if current_time >= time:
                # get samples which have new observations at current_time
                start = event_pt[j]
                end = event_pt[j + 1]

                X_obs = X[start:end]
                M_obs = M[start:end]
                if target is not None:
                    target_obs = target[start:end]
                i_obs = batch_idx[start:end].type(torch.LongTensor)

                if loss_mae:
                    if current_time > 0:
                        # loss function (MAE)
                        loss_reg = loss_reg + (torch.abs(X_obs - p[i_obs]) * M_obs).sum()
                        total_M_obs = total_M_obs + M_obs.sum()

                # update h and h_tilde in marginal component
                mgn_h, mgn_h_tilde = self.mgn_GRUODE(current_time, mgn_h, dt, mgn_h_tilde, X_obs, i_obs, update=True)

                # update h and h_tilde in joint component
                joint_h = self.joint_GRUODE(current_time, mgn_h_tilde, joint_h, dt, i_obs,update=True)

                # if regression problem, predict the next value
                if self.memory == 'both':
                    p = self.p_model(torch.cat([mgn_h, joint_h], dim=-1))
                elif self.memory == 'joint':
                    p = self.p_model(joint_h)
                elif self.memory == 'mgn':
                    p = self.p_model(mgn_h)

                # if oer-time classification problem, predict the next class
                if class_per_time:
                    if self.memory == 'both':
                        pred_prob = self.classify(torch.cat([mgn_h, joint_h], dim=-1))
                    elif self.memory == 'joint':
                        pred_prob = self.classify(joint_h)
                    elif self.memory == 'mgn':
                        pred_prob = self.classify(mgn_h)

                    loss_CE_per_time = torch.nn.CrossEntropyLoss()(pred_prob[i_obs], target_obs.long()).sum()
                    loss_CE = loss_CE + loss_CE_per_time

                    if return_path:
                        path['path_t'].append(current_time)
                        path['path_p'].append(pred_prob[i_obs])
                        path['path_y'].append(target_obs)
                else:
                    if return_path:
                        path['path_t'].append(current_time)
                        path['path_p'].append(p)

        if prop_to_end:
            # if need propagating the hidden state until T after every observation has been processed
            while current_time < T - 0.001 * dt:

                mgn_h, mgn_h_tilde = self.mgn_GRUODE(current_time, mgn_h, dt)
                joint_h = self.joint_GRUODE(current_time, mgn_h_tilde, joint_h, dt)

                if self.memory == 'both':
                    p = self.p_model(torch.cat([mgn_h, joint_h], dim=-1))
                elif self.memory == 'joint':
                    p = self.p_model(joint_h)
                elif self.memory == 'mgn':
                    p = self.p_model(mgn_h)

                if class_per_time:
                    pass
                else:
                    if return_path:
                        path['path_t'].append(current_time)
                        path['path_p'].append(p)

                current_time = current_time + dt

        if classify:
            if self.memory == 'both':
                pred_prob = self.classify(torch.cat([mgn_h, joint_h], dim=-1))
            elif self.memory == 'joint':
                pred_prob = self.classify(joint_h)
            elif self.memory == 'mgn':
                pred_prob = self.classify(mgn_h)

        if return_path:
            if classify:
                return loss_reg, pred_prob
            elif class_per_time:
                return np.array(path['path_t']), torch.cat(path['path_p']), torch.cat(path['path_y'])
            else:
                return np.array(path['path_t']), torch.stack(path['path_p'])
        else:
            if classify:
                return loss_reg, pred_prob
            elif class_per_time:
                return loss_reg, loss_CE
            else:
                return loss_reg, loss_reg / total_M_obs