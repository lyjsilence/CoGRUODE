import numpy as np
import torch
import utils
import torch.nn as nn
from torch.nn import Parameter
from itertools import chain
import torchcde
from torchdiffeq import odeint_adjoint as odeint

'''
This part of code are mainly implemented according GRU-ODE-Bayes
https://arxiv.org/abs/1905.12374
'''


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class MGRUODECell(torch.nn.Module):
    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bias = bias

        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.apply(init_weights)

    def forward(self, t, h):
        x = torch.zeros_like(h)
        z = torch.sigmoid(x + self.lin_hz(h))
        n = torch.tanh(x + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh


class GRUODECell(torch.nn.Module):

    def __init__(self, hidden_size, bias=True):
        super().__init__()

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.apply(init_weights)

    def forward(self, t, h):
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))
        dh = (1 - z) * (u - h)
        return dh


class GRUObsCell(torch.nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size * 2
        self.hidden_size = hidden_size
        self.gru_d = torch.nn.GRUCell(self.input_size, self.hidden_size, bias=bias)

    def forward(self, h, X_obs, i_obs):
        temp = h.clone()
        X_obs = X_obs.reshape(X_obs.shape[0], -1)
        temp[i_obs] = self.gru_d(X_obs, h[i_obs])
        h = temp
        return h


class GRU_ODE(nn.Module):
    def __init__(self, args, device):
        super(GRU_ODE, self).__init__()
        # params of GRU_ODE Networks
        self.input_size = args.input_size
        self.n_dim = args.n_dim
        self.hidden_size = args.hidden_size
        self.solver = args.solver
        self.dropout = args.dropout
        self.task = args.task
        self.device = device
        self.minimal = args.minimal

        self.p_model = nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_size, self.input_size-1),
        )

        # Whether using GRU-ODE or Minimal GRU-ODE
        if self.minimal:
            self.gru_c = MGRUODECell(self.hidden_size)
        else:
            self.gru_c = GRUODECell(self.hidden_size)

        self.gru_obs = GRUObsCell(self.input_size, self.hidden_size)

        assert self.solver in ["euler", "midpoint", "rk4", "explicit_adams", "implicit_adams",
                               "dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun"]

        self.apply(init_weights)

    def reset_hidden(self) -> None:
        for name, param in self.named_parameters():
            if 'h_init' in name:
                nn.init.xavier_uniform_(param.data)

    def ode_step(self, h, delta_t):
        if self.solver == 'euler':
            dh = self.gru_c(None, h)
            h = h + delta_t * dh
        else:
            if self.solver == 'euler':
                dh = self.gru_c(None, h)
                h = h + delta_t * dh
            else:
                solution = odeint(self.gru_c, h, torch.tensor([0, delta_t]).to(h.device), method=self.solver)
                h = solution[1, :, :]

        return h

    def compute_loss(self, loss_mse, loss_mae, loss_mape, total_M_obs, X_obs, p_obs, M_obs):

        X_obs, M_obs = X_obs[:, 1:], M_obs[:, 1:]
        loss_mse = loss_mse + (torch.pow(X_obs - p_obs, 2) * M_obs).sum()
        loss_mae = loss_mae + (torch.abs(X_obs - p_obs) * M_obs).sum()
        loss_mape = loss_mape + (torch.abs(X_obs - p_obs)/(X_obs + 1e-8) * M_obs).sum()
        total_M_obs = total_M_obs + M_obs.sum()

        return loss_mse, loss_mae, loss_mape, total_M_obs

    def append_classify_res(self, pred_list, true_list, Y_obs, p_obs):
        Y_obs = Y_obs[:, 1:]
        index = [Y_obs != 0]
        Y_loss, p_loss = Y_obs[index], p_obs[index]
        pred_list.append(p_loss)
        true_list.append(Y_loss)

        return pred_list, true_list

    def compute_classfy_loss(self, pred_list, true_list):
        pred_list = torch.stack(list(chain(*pred_list)))
        true_list = torch.stack(list(chain(*true_list)))

        criteria = torch.nn.BCEWithLogitsLoss()
        true_list = true_list * 0.5 + 0.5
        loss_ce = criteria(pred_list, true_list)

        sigmoid = nn.Sigmoid()
        pred_list = sigmoid(pred_list)

        loss_acc = utils.acc(true_list.cpu().detach().numpy(), pred_list.cpu().detach().numpy())
        loss_auc, loss_pr = utils.auc_pr(true_list.cpu().detach().numpy(), pred_list.cpu().detach().numpy())

        return loss_ce, loss_acc, loss_auc, loss_pr

    def forward(self,  obs_times, event_pt, sample_idx, X, M, batch_idx, dt, Y=None, return_path=False):

        current_time = 0.0
        loss_mse, loss_mae, loss_mape = torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0)
        pred_list, true_list = [], []
        total_M_obs = 0

        if return_path:
            path = {}
            path['path_t'] = []
            path['path_p'] = []
            path['path_y'] = []

        # create the hidden state for each sampled time series
        h = torch.zeros([sample_idx.shape[0], self.hidden_size]).to(self.device)

        for i, obs_time in enumerate(obs_times):
            # do not reach the observation, using ODE to update hidden state
            while current_time < obs_time:
                h = self.ode_step(h, dt)
                current_time = current_time + dt

            # Reached an observation, using GRU cell to update hidden state
            start = event_pt[i]
            end = event_pt[i + 1]

            X_obs = X[start:end]
            M_obs = M[start:end]
            i_obs = batch_idx[start:end].type(torch.LongTensor)

            p = self.p_model(h)

            if current_time > 0:
                if self.task == 'regression':
                    loss_mse, loss_mae, loss_mape, total_M_obs \
                        = self.compute_reg_loss(loss_mse, loss_mae, loss_mape, total_M_obs, X_obs[:, :, 0], p[i_obs], M_obs)
                elif self.task == 'classification':
                    Y_obs = Y[start:end]
                    pred_list, true_list = self.append_classify_res(pred_list, true_list, Y_obs, p[i_obs])

            # Using GRUObservationCell to update h. Also updating p and loss
            h = self.gru_obs(h, X_obs, i_obs)

        if self.task == 'classification':
            loss_ce, loss_acc, loss_auc, loss_pr = self.compute_classfy_loss(pred_list, true_list)

        if return_path:
            return np.array(path['path_t']), torch.stack(path['path_p'])
        else:
            if self.task == 'regression':
                return loss_mse / total_M_obs, loss_mae / total_M_obs, loss_mape / total_M_obs
            elif self.task == 'classification':
                return loss_ce, loss_acc, loss_auc, loss_pr

'''
GRU delta-t
'''


class GRU_delta_t(nn.Module):
    def __init__(self, args, device):
        super(GRU_delta_t, self).__init__()

        self.input_size = args.input_size
        self.sub_series = args.sub_series
        self.n_dim = args.n_dim
        self.hidden_size = args.hidden_size
        self.solver = args.solver
        self.dropout = args.dropout
        self.task = args.task
        self.device = device

        self.p_model = nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_size, self.input_size-1),
        )

        self.GRUCell = nn.GRUCell(self.input_size * self.sub_series + 1, self.hidden_size)

        self.apply(init_weights)

    def reset_hidden(self) -> None:
        for name, param in self.named_parameters():
            if 'h_init' in name:
                nn.init.xavier_uniform_(param.data)

    def compute_loss(self, loss_mse, loss_mae, loss_mape, total_M_obs, X_obs, p_obs, M_obs):

        X_obs, M_obs = X_obs[:, 1:], M_obs[:, 1:]
        loss_mse = loss_mse + (torch.pow(X_obs - p_obs, 2) * M_obs).sum()
        loss_mae = loss_mae + (torch.abs(X_obs - p_obs) * M_obs).sum()
        loss_mape = loss_mape + (torch.abs(X_obs - p_obs)/(X_obs + 1e-8) * M_obs).sum()
        total_M_obs = total_M_obs + M_obs.sum()

        return loss_mse, loss_mae, loss_mape, total_M_obs

    def append_classify_res(self, pred_list, true_list, Y_obs, p_obs):
        Y_obs = Y_obs[:, 1:]
        index = [Y_obs != 0]
        Y_loss, p_loss = Y_obs[index], p_obs[index]
        pred_list.append(p_loss)
        true_list.append(Y_loss)

        return pred_list, true_list

    def compute_classfy_loss(self, pred_list, true_list):
        pred_list = torch.stack(list(chain(*pred_list)))
        true_list = torch.stack(list(chain(*true_list)))

        criteria = torch.nn.BCEWithLogitsLoss()
        true_list = true_list * 0.5 + 0.5
        loss_ce = criteria(pred_list, true_list)

        sigmoid = nn.Sigmoid()
        pred_list = sigmoid(pred_list)

        loss_acc = utils.acc(true_list.cpu().detach().numpy(), pred_list.cpu().detach().numpy())
        loss_auc, loss_pr = utils.auc_pr(true_list.cpu().detach().numpy(), pred_list.cpu().detach().numpy())

        return loss_ce, loss_acc, loss_auc, loss_pr

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, dt, Y=None, return_path=False):

        current_time = 0.0
        loss_mse, loss_mae, loss_mape = torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0)
        pred_list, true_list = [], []
        loss_CE = torch.as_tensor(0.0)
        total_M_obs = 0

        if return_path:
            path = {}
            path['path_t'] = []
            path['path_p'] = []
            path['path_y'] = []

        h = torch.zeros([sample_idx.shape[0], self.hidden_size]).to(self.device)

        # remember the last time of updating
        last_t = torch.zeros(sample_idx.shape[0])

        for i, obs_time in enumerate(obs_times):
            current_time = obs_time
            start = event_pt[i]
            end = event_pt[i + 1]

            X_obs = X[start:end, :]
            M_obs = M[start:end, :]

            i_obs = batch_idx[start:end].type(torch.LongTensor)

            p = self.p_model(h)
            if current_time > 0:
                if self.task == 'regression':
                    loss_mse, loss_mae, loss_mape, total_M_obs \
                        = self.compute_reg_loss(loss_mse, loss_mae, loss_mape, total_M_obs, X_obs[:, :, 0], p[i_obs], M_obs)
                elif self.task == 'classification':
                    Y_obs = Y[start:end]
                    pred_list, true_list = self.append_classify_res(pred_list, true_list, Y_obs, p[i_obs])


            X_obs = X_obs.reshape(X_obs.shape[0], -1)
            input = torch.cat([X_obs, (current_time - last_t[i_obs]).unsqueeze(1).to(self.device)], dim=-1)

            # update the last observation time and hidden state
            temp_last_t, temp_h = last_t.clone(), h.clone()
            temp_last_t[i_obs] = current_time
            temp_h[i_obs] = self.GRUCell(input, h[i_obs])
            last_t, h = temp_last_t, temp_h

        if self.task == 'classification':
            loss_ce, loss_acc, loss_auc, loss_pr = self.compute_classfy_loss(pred_list, true_list)

        if return_path:
            return np.array(path['path_t']), torch.stack(path['path_p'])
        else:
            if self.task == 'regression':
                return loss_mse / total_M_obs, loss_mae / total_M_obs, loss_mape / total_M_obs
            elif self.task == 'classification':
                return loss_ce, loss_acc, loss_auc, loss_pr


'''
This part of code are mainly implemented according GRU-D
https://arxiv.org/abs/1606.01865
'''

class GRU_D_cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU_D_cell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.W_r = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.V_r = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.U_r = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))

        self.W_z = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.V_z = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.U_z = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))

        self.W_h = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.V_h = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.U_h = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))

        self.b_r = nn.Parameter(torch.randn(self.hidden_size))
        self.b_z = nn.Parameter(torch.randn(self.hidden_size))
        self.b_h = nn.Parameter(torch.randn(self.hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            torch.nn.init.uniform_(w, -stdv, stdv)

    def forward(self, h, X_hat, M_obs, gamma_h):
        h = gamma_h * h
        r = torch.sigmoid(torch.mm(X_hat, self.W_r) + torch.mm(h, self.U_r) + torch.mm(M_obs, self.V_r) + self.b_r)
        z = torch.sigmoid(torch.mm(X_hat, self.W_z) + torch.mm(h, self.U_z) + torch.mm(M_obs, self.V_z) + self.b_z)
        h_tilde = torch.tanh(
            torch.mm(X_hat, self.W_h) + torch.mm(r * h, self.U_h) + torch.mm(M_obs, self.V_h) + self.b_h)
        h = (1 - z) * h + z * h_tilde

        return h

class GRU_D(nn.Module):
    def __init__(self, args, device):
        super(GRU_D, self).__init__()

        self.input_size = args.input_size
        self.sub_series = args.sub_series
        self.n_dim = args.n_dim
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.task = args.task
        self.device = device

        # decay parameters
        self.lin_gamma_x = nn.Linear(self.input_size * self.sub_series, self.input_size * self.sub_series, bias=False)
        self.lin_gamma_h = nn.Linear(self.input_size * self.sub_series, self.hidden_size, bias=False)

        # mapping function from hidden state to real data
        self.p_model = nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_size, self.input_size-1)
        )

        self.gru_d = GRU_D_cell(self.input_size * self.sub_series, self.hidden_size)
        self.apply(init_weights)

    def append_classify_res(self, pred_list, true_list, Y_obs, p_obs):
        Y_obs = Y_obs[:, 1:]
        index = [Y_obs != 0]
        Y_loss, p_loss = Y_obs[index], p_obs[index]
        pred_list.append(p_loss)
        true_list.append(Y_loss)

        return pred_list, true_list

    def compute_classfy_loss(self, pred_list, true_list):
        pred_list = torch.stack(list(chain(*pred_list)))
        true_list = torch.stack(list(chain(*true_list)))

        criteria = torch.nn.BCEWithLogitsLoss()
        true_list = true_list * 0.5 + 0.5
        loss_ce = criteria(pred_list, true_list)

        sigmoid = nn.Sigmoid()
        pred_list = sigmoid(pred_list)

        loss_acc = utils.acc(true_list.cpu().detach().numpy(), pred_list.cpu().detach().numpy())
        loss_auc, loss_pr = utils.auc_pr(true_list.cpu().detach().numpy(), pred_list.cpu().detach().numpy())

        return loss_ce, loss_acc, loss_auc, loss_pr

    def compute_loss(self, loss_mse, loss_mae, loss_mape, total_M_obs, X_obs, p_obs, M_obs):

        X_obs, M_obs = X_obs[:, 1:], M_obs[:, 1:]
        loss_mse = loss_mse + (torch.pow(X_obs - p_obs, 2) * M_obs).sum()
        loss_mae = loss_mae + (torch.abs(X_obs - p_obs) * M_obs).sum()
        loss_mape = loss_mape + (torch.abs(X_obs - p_obs)/(X_obs + 1e-8) * M_obs).sum()
        total_M_obs = total_M_obs + M_obs.sum()

        return loss_mse, loss_mae, loss_mape, total_M_obs

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, dt, Y=None, return_path=False):
        current_time = 0.0
        loss_mse, loss_mae, loss_mape = torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0)
        pred_list, true_list = [], []
        total_M_obs = 0

        if return_path:
            path = {}
            path['path_t'] = []
            path['path_p'] = []
            path['path_y'] = []

        # create the hidden state for each sampled time series
        h = torch.zeros([sample_idx.shape[0], self.hidden_size]).to(self.device)

        # create and store the last observation time and value for each sampled time series
        last_t = torch.zeros([sample_idx.shape[0], self.input_size * self.sub_series]).to(self.device)
        last_x = torch.zeros([sample_idx.shape[0], self.input_size * self.sub_series]).to(self.device)

        for i, obs_time in enumerate(obs_times):
            current_time = obs_time
            start = event_pt[i]
            end = event_pt[i + 1]

            X_obs = X[start:end, :]
            M_obs = M[start:end, :]
            i_obs = batch_idx[start:end].type(torch.LongTensor)

            # compute loss
            p = self.p_model(h)
            if current_time > 0:
                if self.task == 'regression':
                    loss_mse, loss_mae, loss_mape, total_M_obs \
                        = self.compute_reg_loss(loss_mse, loss_mae, loss_mape, total_M_obs, X_obs[:, :, 0], p[i_obs], M_obs)
                elif self.task == 'classification':
                    Y_obs = Y[start:end]
                    pred_list, true_list = self.append_classify_res(pred_list, true_list, Y_obs, p[i_obs])

            X_obs = X_obs.reshape(X_obs.shape[0], -1)
            M_obs = torch.repeat_interleave(M_obs, 2, dim=1)
            # update saved X at the last time point
            last_x[i_obs, :] = last_x[i_obs, :] * (1 - M_obs) + X_obs * M_obs
            # compute the mean of each variables
            mean_x = torch.sum(X_obs, dim=0, keepdim=True) / torch.sum(M_obs+1e-6, dim=0, keepdim=True)
            # get the time interval
            interval = obs_time - last_t
            # update the time point of last observation
            last_t[i_obs, :] = last_t[i_obs, :] * (1 - M_obs) + obs_time * M_obs

            gamma_x = torch.exp(-torch.maximum(torch.tensor(0.0).to(self.device), self.lin_gamma_x(interval)))
            gamma_h = torch.exp(-torch.maximum(torch.tensor(0.0).to(self.device), self.lin_gamma_h(interval)))

            X_hat = M_obs * X_obs + (1 - M_obs) * gamma_x[i_obs] * last_x[i_obs] + \
                    (1 - M_obs) * (1 - gamma_x[i_obs]) * mean_x

            temp = h.clone()
            temp[i_obs] = self.gru_d(h[i_obs], X_hat, M_obs, gamma_h[i_obs])
            h = temp

        if self.task == 'classification':
            loss_ce, loss_acc, loss_auc, loss_pr = self.compute_classfy_loss(pred_list, true_list)


        if return_path:
            return np.array(path['path_t']), torch.stack(path['path_p'])
        else:
            if self.task == 'regression':
                return loss_mse / total_M_obs, loss_mae / total_M_obs, loss_mape / total_M_obs
            elif self.task == 'classification':
                return loss_ce, loss_acc, loss_auc, loss_pr




'''
This part of code are mainly implemented according ODE-LSTM
https://arxiv.org/pdf/2006.04418.pdf
'''


class ODEFunc(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.ODEf = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        )

    def forward(self, t, h):
        return self.ODEf(h)


class ODELSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, sub_series):
        super(ODELSTMCell, self).__init__()
        self.input_size = input_size
        self.sub_series = sub_series
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(self.input_size * self.sub_series, self.hidden_size)
        self.ODEFunc = ODEFunc(hidden_size)
        self.apply(init_weights)

    def forward(self, X, h, c, delta_t, solver, update):
        if update:
            h, c = self.lstm(X, (h, c))
            return h, c
        else:
            h = odeint(self.ODEFunc, h, torch.tensor([0, delta_t]).to(h.device), method=solver)[1]
            return h


class ODELSTM(nn.Module):
    def __init__(self, args, device):
        super(ODELSTM, self).__init__()

        self.input_size = args.input_size
        self.sub_series = args.sub_series
        self.n_dim = args.n_dim
        self.hidden_size = args.hidden_size
        self.cell_size = args.hidden_size
        self.solver = args.solver
        self.dropout = args.dropout
        self.task = args.task
        self.device = device

        self.p_model = nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_size, self.input_size-1),
        )

        # ODE-LSTM Cell
        self.odelstm = ODELSTMCell(self.input_size, self.hidden_size, self.sub_series)

        self.apply(init_weights)

    def compute_loss(self, loss_mse, loss_mae, loss_mape, total_M_obs, X_obs, p_obs, M_obs):

        X_obs, M_obs = X_obs[:, 1:], M_obs[:, 1:]
        loss_mse = loss_mse + (torch.pow(X_obs - p_obs, 2) * M_obs).sum()
        loss_mae = loss_mae + (torch.abs(X_obs - p_obs) * M_obs).sum()
        loss_mape = loss_mape + (torch.abs(X_obs - p_obs)/(X_obs + 1e-8) * M_obs).sum()
        total_M_obs = total_M_obs + M_obs.sum()

        return loss_mse, loss_mae, loss_mape, total_M_obs

    def append_classify_res(self, pred_list, true_list, Y_obs, p_obs):
        Y_obs = Y_obs[:, 1:]
        index = [Y_obs != 0]
        Y_loss, p_loss = Y_obs[index], p_obs[index]
        pred_list.append(p_loss)
        true_list.append(Y_loss)

        return pred_list, true_list

    def compute_classfy_loss(self, pred_list, true_list):
        pred_list = torch.stack(list(chain(*pred_list)))
        true_list = torch.stack(list(chain(*true_list)))

        criteria = torch.nn.BCEWithLogitsLoss()
        true_list = true_list * 0.5 + 0.5
        loss_ce = criteria(pred_list, true_list)

        sigmoid = nn.Sigmoid()
        pred_list = sigmoid(pred_list)

        loss_acc = utils.acc(true_list.cpu().detach().numpy(), pred_list.cpu().detach().numpy())
        loss_auc, loss_pr = utils.auc_pr(true_list.cpu().detach().numpy(), pred_list.cpu().detach().numpy())

        return loss_ce, loss_acc, loss_auc, loss_pr

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, dt, Y=None, return_path=False):
        current_time = 0.0
        loss_mse, loss_mae, loss_mape = torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0)
        pred_list, true_list = [], []
        total_M_obs = 0

        if return_path:
            path = {}
            path['path_t'] = []
            path['path_p'] = []
            path['path_y'] = []

        h = torch.zeros([sample_idx.shape[0], self.hidden_size]).to(self.device)
        c = torch.zeros([sample_idx.shape[0], self.cell_size]).to(self.device)

        for i, obs_time in enumerate(obs_times):
            # do not reach the observation, using ODE to update hidden state
            while current_time < obs_time:
                h = self.odelstm(None, h, None, dt, self.solver, update=False)
                current_time = current_time + dt

            start = event_pt[i]
            end = event_pt[i + 1]

            X_obs = X[start:end, :]
            M_obs = M[start:end, :]
            i_obs = batch_idx[start:end].type(torch.LongTensor)

            p = self.p_model(h)
            if current_time > 0:
                if self.task == 'regression':
                    loss_mse, loss_mae, loss_mape, total_M_obs \
                        = self.compute_reg_loss(loss_mse, loss_mae, loss_mape, total_M_obs, X_obs[:, :, 0], p[i_obs], M_obs)
                elif self.task == 'classification':
                    Y_obs = Y[start:end]
                    pred_list, true_list = self.append_classify_res(pred_list, true_list, Y_obs, p[i_obs])

            X_obs = X_obs.reshape(X_obs.shape[0], -1)
            temp_c, temp_h = c.clone(), h.clone()
            # if there exists observations, using LSTM updated
            temp_h[i_obs], temp_c[i_obs] = self.odelstm(X_obs, h[i_obs], c[i_obs], dt, self.solver, update=True)
            c, h = temp_c, temp_h

        if self.task == 'classification':
            loss_ce, loss_acc, loss_auc, loss_pr = self.compute_classfy_loss(pred_list, true_list)

        if return_path:
            return np.array(path['path_t']), torch.stack(path['path_p'])
        else:
            if self.task == 'regression':
                return loss_mse / total_M_obs, loss_mae / total_M_obs, loss_mape / total_M_obs
            elif self.task == 'classification':
                return loss_ce, loss_acc, loss_auc, loss_pr


'''
This part of code are mainly implemented according ODE-RNN
https://arxiv.org/abs/1907.03907
'''


class ODERNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, sub_series):
        super(ODERNNCell, self).__init__()
        self.input_size = input_size
        self.sub_series = sub_series
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(self.input_size * self.sub_series, self.hidden_size)
        self.ODEFunc = ODEFunc(hidden_size)
        self.apply(init_weights)

    def forward(self, X, h, delta_t, solver, update):
        if update:
            h = self.gru(X, h)
            return h
        else:
            h = odeint(self.ODEFunc, h, torch.tensor([0, delta_t]).to(h.device), method=solver)[1]
            return h


class ODERNN(nn.Module):
    def __init__(self, args, device):
        super(ODERNN, self).__init__()

        self.input_size = args.input_size
        self.sub_series = args.sub_series
        self.n_dim = args.n_dim
        self.hidden_size = args.hidden_size
        self.solver = args.solver
        self.dropout = args.dropout
        self.task = args.task
        self.device = device

        self.p_model = nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_size, self.input_size-1),
        )

        # ODERNN Cell
        self.odernn = ODERNNCell(self.input_size, self.hidden_size, self.sub_series)

        self.apply(init_weights)

    def compute_loss(self, loss_mse, loss_mae, loss_mape, total_M_obs, X_obs, p_obs, M_obs):

        X_obs, M_obs = X_obs[:, 1:], M_obs[:, 1:]
        loss_mse = loss_mse + (torch.pow(X_obs - p_obs, 2) * M_obs).sum()
        loss_mae = loss_mae + (torch.abs(X_obs - p_obs) * M_obs).sum()
        loss_mape = loss_mape + (torch.abs(X_obs - p_obs)/(X_obs + 1e-8) * M_obs).sum()
        total_M_obs = total_M_obs + M_obs.sum()

        return loss_mse, loss_mae, loss_mape, total_M_obs

    def append_classify_res(self, pred_list, true_list, Y_obs, p_obs):
        Y_obs = Y_obs[:, 1:]
        index = [Y_obs != 0]
        Y_loss, p_loss = Y_obs[index], p_obs[index]
        pred_list.append(p_loss)
        true_list.append(Y_loss)

        return pred_list, true_list

    def compute_classfy_loss(self, pred_list, true_list):
        pred_list = torch.stack(list(chain(*pred_list)))
        true_list = torch.stack(list(chain(*true_list)))

        criteria = torch.nn.BCEWithLogitsLoss()
        true_list = true_list * 0.5 + 0.5
        loss_ce = criteria(pred_list, true_list)

        sigmoid = nn.Sigmoid()
        pred_list = sigmoid(pred_list)

        loss_acc = utils.acc(true_list.cpu().detach().numpy(), pred_list.cpu().detach().numpy())
        loss_auc, loss_pr = utils.auc_pr(true_list.cpu().detach().numpy(), pred_list.cpu().detach().numpy())

        return loss_ce, loss_acc, loss_auc, loss_pr

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, dt, Y=None, return_path=False):
        current_time = 0.0
        loss_mse, loss_mae, loss_mape = torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0)
        pred_list, true_list = [], []
        total_M_obs = 0

        if return_path:
            path = {}
            path['path_t'] = []
            path['path_p'] = []
            path['path_y'] = []

        h = torch.zeros([sample_idx.shape[0], self.hidden_size]).to(self.device)

        for i, obs_time in enumerate(obs_times):
            # do not reach the observation, using ODE to update hidden state
            while current_time < obs_time:
                h = self.odernn(None, h, dt, self.solver, update=False)
                current_time = current_time + dt

            start = event_pt[i]
            end = event_pt[i + 1]

            X_obs = X[start:end, :]
            M_obs = M[start:end, :]
            i_obs = batch_idx[start:end].type(torch.LongTensor)

            p = self.p_model(h)
            if current_time > 0:
                if self.task == 'regression':
                    loss_mse, loss_mae, loss_mape, total_M_obs \
                        = self.compute_reg_loss(loss_mse, loss_mae, loss_mape, total_M_obs, X_obs[:, :, 0], p[i_obs], M_obs)
                elif self.task == 'classification':
                    Y_obs = Y[start:end]
                    pred_list, true_list = self.append_classify_res(pred_list, true_list, Y_obs, p[i_obs])

            X_obs = X_obs.reshape(X_obs.shape[0], -1)
            temp_h = h.clone()
            # if there exists observations, using LSTM updated
            temp_h[i_obs] = self.odernn(X_obs, h[i_obs], dt, self.solver, update=True)
            h = temp_h

        if self.task == 'classification':
            loss_ce, loss_acc, loss_auc, loss_pr = self.compute_classfy_loss(pred_list, true_list)

        if return_path:
            return np.array(path['path_t']), torch.stack(path['path_p'])
        else:
            if self.task == 'regression':
                return loss_mse / total_M_obs, loss_mae / total_M_obs, loss_mape / total_M_obs
            elif self.task == 'classification':
                return loss_ce, loss_acc, loss_auc, loss_pr


'''
This part of code are mainly implemented according Neural CDE
https://arxiv.org/abs/2005.08926
'''

class FinalTanh(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_hidden_size, num_hidden_layers):
        super(FinalTanh, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_hidden_size = hidden_hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_size, hidden_hidden_size)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_size, hidden_hidden_size)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_size, input_size * hidden_size)
        self.apply(init_weights)

    def extra_repr(self):
        return "input_size: {}, hidden_size: {}, hidden_hidden_size: {}, num_hidden_layers: {}" \
               "".format(self.input_size, self.hidden_size, self.hidden_hidden_size, self.num_hidden_layers)

    def forward(self, t, z):
        z = self.linear_in(z)
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_size, self.input_size)
        z = z.tanh()
        return z


class Neural_CDE(nn.Module):
    def __init__(self, NCDE_params):
        super(Neural_CDE, self).__init__()

        self.hidden_size = NCDE_params['hidden_size']
        self.input_size = NCDE_params['input_size'] + 1
        self.bias = NCDE_params['bias']
        self.solver = NCDE_params['solver']
        self.num_class = NCDE_params['num_class']
        self.dropout = NCDE_params['dropout']

        self.initial = torch.nn.Linear(self.input_size, self.hidden_size)

        self.ncde = FinalTanh(self.input_size, self.hidden_size, hidden_hidden_size=150, num_hidden_layers=4)

        # mapping function from hidden state to real data
        self.p_model = nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_size, self.input_size, bias=self.bias),
        )

        if self.num_class != None:
            # classification net
            self.classify = nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hidden_size, self.num_class, bias=self.bias),
            )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_coeffs, batch_idx, device, T=None, return_path=False,
                classify=False, class_per_time=False, target=None, prop_to_end=True, loss_mae=True, dt=0.05):

        current_time = 0
        loss = torch.as_tensor(0.0)
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

        '''interpolate the observations using rectilinear method for online prediction'''
        X_path = torchcde.CubicSpline(batch_coeffs)
        X0 = X_path.evaluate(X_path.interval[0])
        h0 = self.initial(X0)

        ht = torchcde.cdeint(X=X_path, func=self.ncde, z0=h0, t=torch.FloatTensor(obs_times).to(device),
                            backend='torchdiffeq', method=self.solver, adjoint=True, options={'step_size': dt})

        for i, obs_time in enumerate(obs_times):
            start = event_pt[i]
            end = event_pt[i + 1]

            if target is not None:
                target_obs = target[start:end]
            i_obs = batch_idx[start:end].type(torch.LongTensor)

            if class_per_time:
                pred_prob = self.classify(ht[:, i, :])
                loss_CE_per_time = torch.nn.CrossEntropyLoss()(pred_prob[i_obs], target_obs.long()).sum()
                loss_CE = loss_CE + loss_CE_per_time

                if return_path:
                    path['path_t'].append(current_time)
                    path['path_p'].append(pred_prob[i_obs])
                    path['path_y'].append(target_obs)

        if classify:
            pred_prob = self.classify(ht[:, -1, :])
            loss_CE = utils.cross_entropy(pred_prob, target)
            acc = utils.sum_accuracy(torch.argmax(pred_prob, dim=1).to(device), target.to(device))

        if return_path:
            if classify:
                return pred_prob
            elif class_per_time:
                return np.array(path['path_t']), torch.cat(path['path_p']), torch.cat(path['path_y'])
        else:
            if classify:
                return pred_prob
            elif class_per_time:
                return loss_CE