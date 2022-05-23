import numpy as np
import torch
import utils
import torch.nn as nn
from torch.nn import Parameter
#from torchdiffeq import odeint

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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_d = torch.nn.GRUCell(input_size, hidden_size, bias=bias)

    def forward(self, h, p, X_obs, M_obs, i_obs):
        # only compute losses on observations
        p_obs = p[i_obs]
        losses = torch.abs(X_obs - p_obs) * M_obs

        temp = h.clone()
        temp[i_obs] = self.gru_d(X_obs, h[i_obs])
        h = temp
        return h, losses

class GRU_ODE(nn.Module):
    def __init__(self, GRU_ODE_params):
        super(GRU_ODE, self).__init__()

        # params of GRU_ODE Networks
        self.hidden_size = GRU_ODE_params['hidden_size']
        self.bias = GRU_ODE_params['bias']
        self.input_size = GRU_ODE_params['input_size']
        self.minimal = GRU_ODE_params['minimal']
        self.solver = GRU_ODE_params['solver']
        self.num_class = GRU_ODE_params['num_class']
        self.dropout = GRU_ODE_params['dropout']

        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_size, self.input_size, bias=self.bias)
        )
        
        if self.num_class != None:
            self.classify = nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hidden_size, self.num_class, bias=self.bias),
            )

        # Whether using GRU-ODE or Minimal GRU-ODE
        if self.minimal:
            self.gru_c = MGRUODECell(self.hidden_size, bias=self.bias)
        else:
            self.gru_c = GRUODECell(self.hidden_size, bias=self.bias)

        self.gru_obs = GRUObsCell(self.input_size, self.hidden_size, bias=self.bias)

        assert self.solver in ["euler", "midpoint", "rk4", "explicit_adams", "implicit_adams",
                          "dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun"]

        self.apply(init_weights)

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
        p = self.p_model(h)
        return h, p

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, device, T=None, return_path=False,
                classify=False, class_per_time=False, target=None, prop_to_end=True, loss_mae=True, dt=0.05):

        # create the hidden state for each sampled time series
        h = torch.zeros([sample_idx.shape[0], self.hidden_size]).to(device)
        # create the mapping function from hidden state to real data
        p = self.p_model(h)

        current_time = 0.0
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

        for i, obs_time in enumerate(obs_times):
            # do not reach the observation, using ODE to update hidden state
            while current_time < obs_time:
                h, p = self.ode_step(h, dt)
                current_time = current_time + dt

                if class_per_time:
                    pass
                else:
                    if return_path:
                        if current_time < obs_time:
                            path['path_t'].append(current_time)
                            path['path_p'].append(p)

            # Reached an observation, using GRU cell to update hidden state
            start = event_pt[i]
            end = event_pt[i + 1]

            X_obs = X[start:end]
            M_obs = M[start:end]
            if target is not None:
                target_obs = target[start:end]
            i_obs = batch_idx[start:end].type(torch.LongTensor)

            # Using GRUObservationCell to update h. Also updating p and loss
            h, losses = self.gru_obs(h, p, X_obs, M_obs, i_obs)

            if loss_mae:
                loss = loss + losses.sum()
                total_M_obs = total_M_obs + M_obs.sum()

            p = self.p_model(h)

            if class_per_time:
                pred_prob = self.classify(h)
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
            # propagating until T after every observation has been processed
            while current_time < T - 0.0001*dt:
                h, p = self.ode_step(h, delta_t=dt)
                current_time += dt

                # Storing the predictions
                if class_per_time:
                    pass
                else:
                    if return_path:
                        path['path_t'].append(current_time)
                        path['path_p'].append(p)

        if classify:
            pred_prob = self.classify(h)

        if return_path:
            if classify:
                return loss / total_M_obs, pred_prob
            elif class_per_time:
                return np.array(path['path_t']), torch.cat(path['path_p']), torch.cat(path['path_y'])
            else:
                return np.array(path['path_t']), torch.stack(path['path_p'])
        else:
            if classify:
                return loss / total_M_obs, pred_prob
            elif class_per_time:
                return loss, loss_CE
            else:
                return loss, loss / total_M_obs


'''
This part of code are mainly implemented according CT-GRU
https://arxiv.org/abs/1710.04110
'''

class CTGRU(nn.Module):
    def __init__(self, CTGRU_params):
        super(CTGRU, self).__init__()
        self.hidden_size = CTGRU_params['hidden_size']
        self.input_size = CTGRU_params['input_size']
        self.bias = CTGRU_params['bias']
        self.scale = CTGRU_params['scale']
        self.num_class = CTGRU_params['num_class']
        self.dropout = CTGRU_params['dropout']
        
        # mapping function from hidden state to real data
        self.p_model = nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_size, self.input_size, bias=self.bias),
        )
        if self.num_class != None:
            self.classify = nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hidden_size, self.num_class, bias=self.bias),
            )

        # compute the baseline tau_tilde
        self.log_tau_list = np.empty(self.scale)
        self.tau_list = np.empty(self.scale)
        tau = 1.0
        for i in range(self.scale):
            self.log_tau_list[i] = np.log(tau)
            self.tau_list[i] = tau
            tau = tau * np.sqrt(10)
        self.tau_list = torch.as_tensor(self.tau_list, dtype=torch.float32)
        self.log_tau_list = torch.as_tensor(self.log_tau_list, dtype=torch.float32)

        self.retrieve = nn.Linear(self.input_size + self.hidden_size, self.hidden_size * self.scale, bias=self.bias)
        self.signal = nn.Linear(self.input_size + self.hidden_size, self.hidden_size, bias=self.bias)
        self.store = nn.Linear(self.input_size + self.hidden_size, self.hidden_size * self.scale, bias=self.bias)
        
        self.apply(init_weights)

    def mae(self, p_obs, X_obs, M_obs):
        losses = torch.abs(X_obs - p_obs) * M_obs
        return losses.sum()

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, device, T=None, return_path=False,
                classify=False, target=None, class_per_time=False, prop_to_end=True, loss_mae=True, dt=0.1):
        # create the hidden state for each sampled time series
        h_hat = torch.zeros([sample_idx.shape[0], self.hidden_size, self.scale], dtype=torch.float32).to(device)
        h = torch.sum(h_hat, dim=2)

        # create and store the last observation time
        last_t = np.zeros(sample_idx.shape[0])
        # mapping hidden state to real data
        p = self.p_model(h)

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

        loss = torch.as_tensor(0.0)
        loss_CE = torch.as_tensor(0.0)
        total_M_obs = 0
        current_time = 0
        
        for i, obs_time in enumerate(obs_times):
            current_time = obs_time
            start = event_pt[i]
            end = event_pt[i + 1]
            i_obs = batch_idx[start:end].type(torch.LongTensor)
            X_obs = X[start:end, :]
            M_obs = M[start:end, :]
            if target is not None:
                target_obs = target[start:end]

            if loss_mae:
                # Compute the loss
                batch_loss = self.mae(p[i_obs], X_obs, M_obs)
                loss = loss + batch_loss
                total_M_obs = total_M_obs + M_obs.sum()
            
            # retrieval from memory
            retrieval_input = torch.cat([X_obs, h[i_obs]], dim=-1)
            log_tau_R = self.retrieve(retrieval_input).view(X_obs.shape[0], self.hidden_size, self.scale)
            r_i = torch.softmax(-torch.square(log_tau_R - self.log_tau_list.to(device)), dim=2)

            # detect event signal
            signal_input = torch.cat([X_obs, torch.sum(r_i * h_hat[i_obs], dim=2)], dim=-1)
            h_tilde = torch.tanh(self.signal(signal_input))
            h_tilde = torch.unsqueeze(h_tilde, dim=2)

            # store in memory
            store_input = torch.cat([X_obs, h[i_obs]], dim=-1)
            log_tau_S = self.store(store_input).view(X_obs.shape[0], self.hidden_size, self.scale)
            z_i = torch.softmax(-torch.square(log_tau_S - self.log_tau_list.to(device)), dim=2)

            # get the time interval and update the time point of last observation
            if len(i_obs) > 1:
                interval = torch.unsqueeze(torch.as_tensor(obs_time - last_t[i_obs], dtype=torch.float32), dim=1).to(device)
            else:
                interval = torch.unsqueeze(torch.unsqueeze(torch.as_tensor(obs_time - last_t[i_obs], dtype=torch.float32), dim=0), dim=1).to(device)
            last_t[i_obs] = obs_time

            # update
            exp = torch.exp(- interval / self.tau_list.to(device))
            exp = torch.unsqueeze(exp, dim=1)
            temp_h_hat = h_hat.clone()
            temp_h = h.clone()
            temp_h_hat[i_obs] = ((1-z_i) * h_hat[i_obs] + z_i * h_tilde) * exp
            temp_h[i_obs] = torch.sum(h_hat[i_obs], dim=2)
            h_hat = temp_h_hat
            h = temp_h

            # Compute predicted value of time series by hidden states
            p = self.p_model(h)

            if class_per_time:
                pred_prob = self.classify(h)
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
            # if no observation exists before T, propagating it to the final time point
            while current_time < T - 0.0001*dt:
                current_time = current_time + dt

                # retrieval from memory
                X_obs = torch.zeros(sample_idx.shape[0], X.shape[1]).to(device)

                retrieval_input = torch.cat([X_obs, h], dim=-1)
                log_tau_R = self.retrieve(retrieval_input).view(X_obs.shape[0], self.hidden_size, self.scale)
                r_i = torch.softmax(-torch.square(log_tau_R - self.log_tau_list.to(device)), dim=2)

                # detect event signal
                signal_input = torch.cat([X_obs, torch.sum(r_i * h_hat, dim=2)], dim=-1)
                h_tilde = torch.tanh(self.signal(signal_input))
                h_tilde = torch.unsqueeze(h_tilde, dim=2)

                # store in memory
                store_input = torch.cat([X_obs, h], dim=-1)
                log_tau_S = self.store(store_input).view(X_obs.shape[0], self.hidden_size, self.scale)
                z_i = torch.softmax(-torch.square(log_tau_S - self.log_tau_list.to(device)), dim=2)

                # get the time interval and update the time point of last observation
                interval = torch.unsqueeze(torch.FloatTensor(torch.zeros(sample_idx.shape[0], 1).fill_(dt)), dim=1).to(device)

                # update
                exp = torch.exp(- interval / self.tau_list.to(device))
                h_hat = ((1 - z_i) * h_hat + z_i * h_tilde) * exp
                h = torch.sum(h_hat, dim=2)

                # Compute predicted value of time series by hidden states
                p = self.p_model(h)
                if class_per_time:
                    pass
                else:
                    if return_path:
                        path['path_t'].append(current_time)
                        path['path_p'].append(p)
        if classify:
            pred_prob = self.classify(h)
            loss_CE = utils.cross_entropy(pred_prob, target)
            acc = utils.sum_accuracy(torch.argmax(pred_prob, dim=1).to(device), target.to(device))

        if return_path:
            if classify:
                return loss / total_M_obs , pred_prob
            elif class_per_time:
                return np.array(path['path_t']), torch.cat(path['path_p']), torch.cat(path['path_y'])
            else:
                return np.array(path['path_t']), torch.stack(path['path_p'])
        else:
            if classify:
                return loss / total_M_obs, pred_prob
            elif class_per_time:
                return loss, loss_CE
            else:
                return loss, loss / total_M_obs
'''
GRU delta-t
'''
class GRU_delta_t(nn.Module):
    def __init__(self, GRU_delta_t_params):
        super(GRU_delta_t, self).__init__()
        self.hidden_size = GRU_delta_t_params['hidden_size']
        self.input_size = GRU_delta_t_params['input_size']
        self.bias = GRU_delta_t_params['bias']
        self.num_class = GRU_delta_t_params['num_class']
        self.dropout = GRU_delta_t_params['dropout']
        
        # mapping function from hidden state to real data
        self.p_model = nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_size, self.input_size, bias=self.bias),
        )
        self.apply(init_weights)
        
        if self.num_class != None:
            # classification net
            self.classify = nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hidden_size, self.num_class, bias=self.bias),
            )

        self.GRUCell = nn.GRUCell(self.input_size + 1, self.hidden_size, bias=self.bias)

    def mae(self, p_obs, X_obs, M_obs):
        losses = torch.abs(X_obs - p_obs) * M_obs
        return losses.sum()

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, device, T=None, return_path=False,
                classify=False, class_per_time=False, target=None, prop_to_end=True, loss_mae=True, dt=0.1):

        h = torch.zeros([sample_idx.shape[0], self.hidden_size]).to(device)

        # remember the last time of updating
        last_t = torch.zeros(sample_idx.shape[0])
        p = self.p_model(h)

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

        loss = torch.as_tensor(0.0)
        loss_CE = torch.as_tensor(0.0)
        total_M_obs = 0
        current_time = 0.0
        for i, obs_time in enumerate(obs_times):
            current_time = obs_time
            start = event_pt[i]
            end = event_pt[i + 1]

            X_obs = X[start:end, :]
            M_obs = M[start:end, :]
            if target is not None:
                target_obs = target[start:end]
            i_obs = batch_idx[start:end].type(torch.LongTensor)

            if loss_mae:
                # Compute the loss
                batch_loss = self.mae(p[i_obs], X_obs, M_obs)
                loss = loss + batch_loss
                total_M_obs = total_M_obs + M_obs.sum()
            
            input = torch.cat([X_obs, (current_time-last_t[i_obs]).unsqueeze(1).to(device)], dim=-1)

            # update the last observation time and hidden state
            temp_last_t, temp_h = last_t.clone(), h.clone()
            temp_last_t[i_obs] = current_time
            temp_h[i_obs] = self.GRUCell(input, h[i_obs])
            last_t, h = temp_last_t, temp_h

            # Compute predicted value of time series by hidden states
            p = self.p_model(h)
            if class_per_time:
                pred_prob = self.classify(h)
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
            # if no observation exists before T, propagating it to the final time point
            while current_time < T - 0.0001*dt:
                current_time = current_time + dt

                # retrieval from memory
                X_obs = torch.zeros(sample_idx.shape[0], X.shape[1]).to(device)
                input = torch.cat([X_obs, (current_time-last_t).unsqueeze(1).to(device)], dim=-1)
                h = self.GRUCell(input, h)

                # Compute predicted value of time series by hidden states
                p = self.p_model(h)
                if class_per_time:
                    pass
                else:
                    if return_path:
                        path['path_t'].append(current_time)
                        path['path_p'].append(p)

        if classify:
            pred_prob = self.classify(h)
            loss_CE = utils.cross_entropy(pred_prob, target)
            acc = utils.sum_accuracy(torch.argmax(pred_prob, dim=1).to(device), target.to(device))

        if return_path:
            if classify:
                return loss / total_M_obs, pred_prob
            elif class_per_time:
                return np.array(path['path_t']), torch.cat(path['path_p']), torch.cat(path['path_y'])
            else:
                return np.array(path['path_t']), torch.stack(path['path_p'])
        else:
            if classify:
                return loss / total_M_obs, pred_prob
            elif class_per_time:
                return loss, loss_CE
            else:
                return loss, loss / total_M_obs

'''
This part of code are mainly implemented according GRU-D
https://arxiv.org/abs/1606.01865
'''
class GRU_D(nn.Module):
    def __init__(self, GRU_D_params):
        super(GRU_D, self).__init__()

        self.lin_gamma_x = nn.Linear(1, 1, bias=True)
        self.lin_gamma_h = nn.Linear(1, 1, bias=True)

        self.hidden_size = GRU_D_params['hidden_size']
        self.input_size = GRU_D_params['input_size']
        self.bias = GRU_D_params['bias']
        self.num_class = GRU_D_params['num_class']
        self.dropout = GRU_D_params['dropout']
        
        self.lin_r = nn.Linear(self.hidden_size + 2 * self.input_size, self.hidden_size, bias=True)
        self.lin_z = nn.Linear(self.hidden_size + 2 * self.input_size, self.hidden_size, bias=True)
        self.lin_h = nn.Linear(self.hidden_size + 2 * self.input_size, self.hidden_size, bias=True)
    
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
        self.apply(init_weights)

    def mae(self, p_obs, X_obs, M_obs):
        losses = torch.abs(X_obs - p_obs) * M_obs
        return losses.sum()

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, device, T=None, return_path=False,
                classify=False, class_per_time=False, target=None, prop_to_end=True, loss_mae=True, dt=0.1):

        # create the hidden state for each sampled time series
        h = torch.zeros([sample_idx.shape[0], self.hidden_size]).to(device)
        p = self.p_model(h)

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

        # create and store the last observation time and value for each sampled time series
        last_t = np.zeros(sample_idx.shape[0])
        last_x = np.zeros([sample_idx.shape[0], self.input_size])

        loss = torch.as_tensor(0.0)
        loss_CE = torch.as_tensor(0.0)
        total_M_obs = 0
        current_time = 0
        for i, obs_time in enumerate(obs_times):
            current_time = obs_time
            # reset the update mask at each time point
            M_obs = np.zeros([sample_idx.shape[0], self.input_size])

            start = event_pt[i]
            end = event_pt[i + 1]
            if target is not None:
                target_obs = target[start:end]
            i_obs = batch_idx[start:end].numpy().astype(np.int)

            if loss_mae:
                # Compute the loss
                batch_loss = self.mae(p[torch.LongTensor(i_obs)], X[start:end, :], M[start:end, :])
                loss = loss + batch_loss
                total_M_obs = total_M_obs + M_obs.sum()
            
            M_obs[i_obs, :] = M[start:end, :].cpu().numpy()
            # update the last_X if there is observations this time point, keep last_x where no observation exists
            last_x[i_obs, :] = (last_x[i_obs, :] * (1 - M_obs[i_obs, :]) + X[start:end, :].cpu().numpy())
            # get the time interval
            interval = torch.unsqueeze(torch.FloatTensor(obs_time - last_t), dim=1).to(device)
            # update the time point of last observation
            last_t[i_obs] = obs_time

            gamma_x = torch.exp(-torch.maximum(torch.tensor(0.0).to(device), self.lin_gamma_x(interval)))
            gamma_h = torch.exp(-torch.maximum(torch.tensor(0.0).to(device), self.lin_gamma_h(interval)))

            M_obs_, last_x_ = torch.FloatTensor(M_obs).to(device), torch.FloatTensor(last_x).to(device)
            X_hat = M_obs_ * last_x_ + (1 - M_obs_) * (gamma_x * last_x_)
            h_hat = gamma_h * h

            input = torch.cat([X_hat, h_hat, M_obs_], dim=-1)
            r = torch.sigmoid(self.lin_r(input))
            z = torch.sigmoid(self.lin_z(input))
            reset_input = torch.cat([X_hat, r * h_hat, M_obs_], dim=-1)
            h_tilde = torch.tanh(self.lin_h(reset_input))

            # Compute new hidden state
            h = (1 - z) * h_hat + z * h_tilde
            # Compute predicted value of time series by hidden states
            p = self.p_model(h)

            if class_per_time:
                pred_prob = self.classify(h)
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
            # if no observation exists before T, propagating it to the final time point
            while current_time < T - 0.0001*dt:
                current_time = current_time + dt

                # get the time interval
                interval = torch.unsqueeze(torch.FloatTensor(current_time - last_t), dim=1).to(device)

                gamma_x = torch.exp(-torch.maximum(torch.tensor(0.0).to(device), self.lin_gamma_x(interval)))
                gamma_h = torch.exp(-torch.maximum(torch.tensor(0.0).to(device), self.lin_gamma_h(interval)))

                # the time series has been normalization before training, thus the empirical mean is 0 for each dimension
                M_obs_, last_x_ = torch.zeros([sample_idx.shape[0], self.input_size]).to(device), torch.FloatTensor(last_x).to(device)
                X_hat = gamma_x * last_x_
                h_hat = gamma_h * h

                input = torch.cat([X_hat, h_hat, M_obs_], dim=-1)
                r = torch.sigmoid(self.lin_r(input))
                z = torch.sigmoid(self.lin_z(input))
                reset_input = torch.cat([X_hat, r * h_hat, M_obs_], dim=-1)
                h_tilde = torch.tanh(self.lin_h(reset_input))

                # Compute new hidden state
                h = (1 - z) * h_hat + z * h_tilde
                # Compute predicted value of time series by hidden states
                p = self.p_model(h)
                if class_per_time:
                    pass
                else:
                    if return_path:
                        path['path_t'].append(current_time)
                        path['path_p'].append(p)

        if classify:
            pred_prob = self.classify(h)
            loss_CE = utils.cross_entropy(pred_prob, target)
            acc = utils.sum_accuracy(torch.argmax(pred_prob, dim=1).to(device), target.to(device))

        if return_path:
            if classify:
                return loss / total_M_obs, pred_prob
            elif class_per_time:
                return np.array(path['path_t']), torch.cat(path['path_p']), torch.cat(path['path_y'])
            else:
                return np.array(path['path_t']), torch.stack(path['path_p'])
        else:
            if classify:
                return loss  / total_M_obs, pred_prob
            elif class_per_time:
                return loss, loss_CE
            else:
                return loss, loss / total_M_obs

'''
This part of code are mainly implemented according ODE-LSTM
https://arxiv.org/pdf/2006.04418.pdf
'''
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias):
        super(LSTMCell, self).__init__()
        self.lin_i = nn.Linear(input_size + hidden_size, hidden_size, bias)
        self.lin_f = nn.Linear(input_size + hidden_size, hidden_size, bias)
        self.lin_o = nn.Linear(input_size + hidden_size, hidden_size, bias)
        self.lin_c_tilde = nn.Linear(input_size + hidden_size, hidden_size, bias)
        self.apply(init_weights)
        
    def forward(self, X, h, c):
        input = torch.cat([X, h], dim=-1)
        i = torch.sigmoid(self.lin_i(input))
        f = torch.sigmoid(self.lin_f(input))
        o = torch.sigmoid(self.lin_o(input))
        c_tilde = torch.tanh(self.lin_c_tilde(input))
        c = f * c + i * c_tilde
        h = o * torch.tanh(c)
        return h, c

class ODELSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias):
        super(ODELSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = LSTMCell(self.input_size, self.hidden_size, bias=bias)
        # 1 hidden layer NODE
        self.ODEFunc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.apply(init_weights)
        
    def forward(self, X, h, c, delta_t, solver, update):
        if update:
            h, c = self.lstm(X, h, c)
            return h, c
        else:
            if solver == 'euler':
                dh = self.ODEFunc(h)
                h = h + delta_t * dh
            return h

class ODELSTM(nn.Module):
    def __init__(self, ODELSTM_params):
        super(ODELSTM, self).__init__()

        self.hidden_size = ODELSTM_params['hidden_size']
        self.cell_size = ODELSTM_params['cell_size']
        self.input_size = ODELSTM_params['input_size']
        self.bias = ODELSTM_params['bias']
        self.solver = ODELSTM_params['solver']
        self.num_class = ODELSTM_params['num_class']
        self.dropout = ODELSTM_params['dropout']
        
        # ODE-LSTM Cell
        self.odelstm = ODELSTMCell(self.input_size, self.hidden_size, self.bias)

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
        self.apply(init_weights)
        
    def mae(self, p_obs, X_obs, M_obs):
        losses = torch.abs(X_obs - p_obs) * M_obs
        return losses.sum()

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, device, T=None, return_path=False,
                classify=False, class_per_time=False, target=None, prop_to_end=True, loss_mae=True, dt=0.05):

        h = torch.zeros([sample_idx.shape[0], self.hidden_size]).to(device)
        c = torch.zeros([sample_idx.shape[0], self.cell_size]).to(device)
        # create the mapping function from hidden state to real data
        p = self.p_model(h)

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

        for i, obs_time in enumerate(obs_times):
            # do not reach the observation, using ODE to update hidden state
            while current_time < obs_time:
                h = self.odelstm(None, h, None, dt, self.solver, update=False)
                p = self.p_model(h)
                current_time = current_time + dt

                if class_per_time:
                    pass
                else:
                    if return_path:
                        path = {}
                        path['path_t'] = []
                        path['path_p'] = []

            start = event_pt[i]
            end = event_pt[i + 1]

            X_obs = X[start:end, :]
            M_obs = M[start:end, :]
            if target is not None:
                target_obs = target[start:end]
            i_obs = batch_idx[start:end].type(torch.LongTensor)

            if loss_mae:
                # Compute the loss
                batch_loss = self.mae(p[i_obs], X_obs, M_obs)
                loss = loss + batch_loss
                total_M_obs = total_M_obs + M_obs.sum()
            
            temp_c, temp_h = c.clone(), h.clone()
            # if there exists observations, using LSTM updated
            temp_h[i_obs], temp_c[i_obs] = self.odelstm(X_obs, h[i_obs], c[i_obs], dt, self.solver, update=True)
            c, h = temp_c, temp_h

            # Compute predicted value of time series by hidden states
            p = self.p_model(h)

            if class_per_time:
                pred_prob = self.classify(h)
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
            while current_time < T - 0.0001 * dt:
                h = self.odelstm(None, h, None, dt, self.solver, update=False)
                p = self.p_model(h)
                current_time = current_time + dt

                if class_per_time:
                    pass
                else:
                    if return_path:
                        path['path_t'].append(current_time)
                        path['path_p'].append(p)

        if classify:
            pred_prob = self.classify(h)
            loss_CE = utils.cross_entropy(pred_prob, target)
            acc = utils.sum_accuracy(torch.argmax(pred_prob, dim=1).to(device), target.to(device))

        if return_path:
            if classify:
                return loss  / total_M_obs, pred_prob
            elif class_per_time:
                return np.array(path['path_t']), torch.cat(path['path_p']), torch.cat(path['path_y'])
            else:
                return np.array(path['path_t']), torch.stack(path['path_p'])
        else:
            if classify:
                return loss / total_M_obs, pred_prob
            elif class_per_time:
                return loss, loss_CE
            else:
                return loss, loss / total_M_obs

'''
This part of code are mainly implemented according ODE-RNN
https://arxiv.org/abs/1907.03907
'''
class ODERNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias):
        super(ODERNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.RNNCell(self.input_size, self.hidden_size, bias=bias)

        self.ODEFunc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.apply(init_weights)
        
    def forward(self, X, h, delta_t, solver, update):
        if update:
            h = self.rnn(X, h)
            return h
        else:
            if solver == 'euler':
                dh = self.ODEFunc(h)
                h = h + delta_t * dh
            return h

class ODERNN(nn.Module):
    def __init__(self, ODERNN_params):
        super(ODERNN, self).__init__()
        self.hidden_size = ODERNN_params['hidden_size']
        self.input_size = ODERNN_params['input_size']
        self.bias = ODERNN_params['bias']
        self.solver = ODERNN_params['solver']
        self.num_class = ODERNN_params['num_class']
        self.dropout = ODERNN_params['dropout']
        
        self.odernn = ODERNNCell(self.input_size, self.hidden_size, self.bias)

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
                
    def mae(self, p_obs, X_obs, M_obs):
        losses = torch.abs(X_obs - p_obs) * M_obs
        return losses.sum()

    def forward(self, obs_times, event_pt, sample_idx, X, M, batch_idx, device, T=None, return_path=False,
                classify=False, class_per_time=False, target=None, prop_to_end=True, loss_mae=True, dt=0.05):

        h = torch.zeros([sample_idx.shape[0], self.hidden_size]).to(device)
        # create the mapping function from hidden state to real data
        p = self.p_model(h)

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

        for i, obs_time in enumerate(obs_times):
            # do not reach the observation, using ODE to update hidden state
            while current_time < obs_time:
                h = self.odernn(None, h, dt, self.solver, update=False)
                p = self.p_model(h)
                current_time = current_time + dt

                if class_per_time:
                    pass
                else:
                    if return_path:
                        path = {}
                        path['path_t'] = []
                        path['path_p'] = []

            start = event_pt[i]
            end = event_pt[i + 1]

            X_obs = X[start:end, :]
            M_obs = M[start:end, :]
            if target is not None:
                target_obs = target[start:end]
            i_obs = batch_idx[start:end].type(torch.LongTensor)

            if loss_mae:
                # Compute the loss
                batch_loss = self.mae(p[i_obs], X_obs, M_obs)
                loss = loss + batch_loss
                total_M_obs = total_M_obs + M_obs.sum()
            
            temp_h = h.clone()
            # if there exists observations, using LSTM updated
            temp_h[i_obs] = self.odernn(X_obs, h[i_obs], dt, self.solver, update=True)
            h = temp_h

            # Compute predicted value of time series by hidden states
            p = self.p_model(h)

            if class_per_time:
                pred_prob = self.classify(h)
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
            while current_time < T - 0.0001 * dt:
                h = self.odernn(None, h, dt, self.solver, update=False)
                p = self.p_model(h)
                current_time = current_time + dt

                if class_per_time:
                    pass
                else:
                    if return_path:
                        path['path_t'].append(current_time)
                        path['path_p'].append(p)

        if classify:
            pred_prob = self.classify(h)
            loss_CE = utils.cross_entropy(pred_prob, target)
            acc = utils.sum_accuracy(torch.argmax(pred_prob, dim=1).to(device), target.to(device))

        if return_path:
            if classify:
                return loss  / total_M_obs, pred_prob
            elif class_per_time:
                return np.array(path['path_t']), torch.cat(path['path_p']), torch.cat(path['path_y'])
            else:
                return np.array(path['path_t']), torch.stack(path['path_p'])
        else:
            if classify:
                return loss / total_M_obs, pred_prob
            elif class_per_time:
                return loss, loss_CE
            else:
                return loss, loss / total_M_obs