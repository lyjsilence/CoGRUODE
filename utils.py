import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from itertools import chain
import torchcde

def preprocessing(data, norm=True):
    data.columns = ['idx', 'time', 'ts_p1', 'ts_v1', 'ts_p2', 'ts_v2', 'ts_p3', 'ts_v3', 'ts_p4', 'ts_v4',
                    'ts_p5', 'ts_v5', 'ts_p6', 'ts_v6', 'ts_p7', 'ts_v7', 'ts_p8', 'ts_v8', 'ts_p9', 'ts_v9', 'ts_p10', 'ts_v10',
                    'ts_p11', 'ts_v11', 'ts_p12', 'ts_v12', 'ts_p13', 'ts_v13', 'ts_p14', 'ts_v14', 'ts_p15', 'ts_v15', 'ts_p16', 'ts_v16',
                    'mask_1', 'mask_2', 'mask_3', 'mask_4', 'mask_5', 'mask_6', 'mask_7', 'mask_8', 'mask_9', 'mask_10',
                    'mask_11', 'mask_12', 'mask_13', 'mask_14', 'mask_15', 'mask_16']
    if norm:
        data['ts_p1'] = (data['ts_p1'] - np.min(data['ts_p1']))/(np.max(data['ts_p1']) - np.min(data['ts_p1']))

        for i in range(1, 17):
            data[f'ts_v{str(i)}'] = (data[f'ts_v{str(i)}'] - np.min(data[f'ts_v{str(i)}'])) / (np.max(data[f'ts_v{str(i)}']) - np.min(data[f'ts_v{str(i)}']))
    return data


class option_dataset(Dataset):
    def __init__(self, ts):
        assert ts is not None

        # data format: [index, time, ts_1, ts_2..., mask_1, mask_2...]
        # Extract the time series with certain index and reindex the time series
        self.ts = ts.copy()
        # the number of unique index of time series
        self.length = len(np.unique(np.array(self.ts['idx'])))

        map_dict = dict(zip(self.ts["idx"].unique(), np.arange(self.ts["idx"].nunique())))
        self.ts["idx"] = self.ts["idx"].map(map_dict)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_ts = self.ts[self.ts['idx'] == idx]
        return batch_ts


class option_static_dataset(Dataset):
    def __init__(self, ts):
        assert ts is not None
        df_res = pd.DataFrame(columns=['st', 'm', 'tau', 'ct'])
        for d in range(2, 17):
            idx_d = ts[f'mask_{str(d)}'] == 1
            ct = ts[idx_d][f'ts_p{str(d)}'].values
            st = ts[idx_d][f'ts_p1'].values
            tau = [20] * len(ct)
            m = [0.94 + 0.01 * d] * len(ct)
            kt = st * m
            df_d = pd.DataFrame(np.stack([st, m, kt, ct], axis=1), columns=['st', 'm', 'tau', 'ct'])
            df_res = pd.concat([df_res, df_d], axis=0)
        self.df_res = np.array(df_res)
        self.length = self.df_res.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_ts = self.df_res[idx, :]
        return batch_ts

def option_collate_fn(batch):
    batch_ts = pd.concat(b for b in batch)

    # all sample index in this batch
    sample_idx = pd.unique(batch_ts['idx'])

    # create a obs_idx which indicates the index of samples in this batch
    map_dict = dict(zip(sample_idx, np.arange(len(sample_idx))))
    for idx in sample_idx:
        batch_ts.loc[batch_ts['idx'] == idx, 'batch_idx'] = map_dict[idx]

    # sort the data according by time sequentially
    batch_ts.sort_values(by=['time'], inplace=True)

    # calculating number of events at every time
    obs_times, counts = np.unique(batch_ts.time.values, return_counts=True)
    event_pt = np.concatenate([[0], np.cumsum(counts)])

    # convert data to tensor
    S = torch.FloatTensor(batch_ts.loc[:, [c.startswith("ts_p") for c in batch_ts.columns]].values)
    V = torch.FloatTensor(batch_ts.loc[:, [c.startswith("ts_v") for c in batch_ts.columns]].values)
    X = torch.cat([S.unsqueeze(2), V.unsqueeze(2)], dim=2)
    Y = torch.FloatTensor(batch_ts.loc[:, [c.startswith("ts_t") for c in batch_ts.columns]].values)
    M = torch.FloatTensor(batch_ts.loc[:, [c.startswith("mask") for c in batch_ts.columns]].values)
    batch_idx = torch.FloatTensor(batch_ts.loc[:, 'batch_idx'].values)

    res = {}
    res["sample_idx"] = sample_idx
    res["obs_times"] = obs_times
    res["event_pt"] = event_pt
    res["X"] = X
    res["Y"] = Y
    res["M"] = M
    res["batch_idx"] = batch_idx

    return res





def option_static_collate_fn(batch):
    X = torch.FloatTensor(np.array(batch))
    res = {}
    res["X"] = X
    return res




def map_to_closest(input, reference):
    output = np.zeros_like(input)
    for idx, element in enumerate(input):
        closest_idx = (np.abs(reference-element)).argmin()
        output[idx] = reference[closest_idx]
    return(output)

def extract_obs(t_vec, p_vec, obs_times_val, batch_idx_val):
    t_vec = np.around(t_vec, 2).astype(np.float32)
    times_val = obs_times_val.astype(np.float32)

    t_vec, unique_index = np.unique(t_vec, return_index=True)
    p_vec = p_vec[unique_index, :, :]

    # Whether the time points which need to be predicted have the prediction value
    present_mask = np.isin(times_val, t_vec)
    # If the time points have not been evaluated, map the closest prediction to this time points
    times_val[~present_mask] = map_to_closest(times_val[~present_mask], t_vec)
    mapping = dict(zip(t_vec, np.arange(t_vec.shape[0])))
    time_idx = np.vectorize(mapping.get)(times_val)

    return (p_vec[time_idx, batch_idx_val, :])


def normalization(train_X, val_X, test_X):
    train_col, val_col, test_col = train_X.columns, val_X.columns, test_X.columns
    train_idx_time = np.array(train_X[['idx', 'time']])
    val_idx_time = np.array(val_X[['idx', 'time']])
    test_idx_time = np.array(test_X[['idx', 'time']])
    train_X_ts = np.array(train_X.loc[:, [c.startswith("ts") for c in train_X.columns]])
    train_X_mask = np.array(train_X.loc[:, [c.startswith("mask") for c in train_X.columns]])
    val_X_ts = np.array(val_X.loc[:, [c.startswith("ts") for c in val_X.columns]])
    val_X_mask = np.array(val_X.loc[:, [c.startswith("mask") for c in val_X.columns]])
    test_X_ts = np.array(test_X.loc[:, [c.startswith("ts") for c in test_X.columns]])
    test_X_mask = np.array(test_X.loc[:, [c.startswith("mask") for c in test_X.columns]])
    for col in range(train_X_ts.shape[1]):
        train_keep_idx = train_X_mask[:, col] == 1
        val_keep_idx = val_X_mask[:, col] == 1
        test_keep_idx = test_X_mask[:, col] == 1
        ts_mean = np.mean(train_X_ts[train_keep_idx, col])
        ts_std = np.std(train_X_ts[train_keep_idx, col])
        train_X_ts[train_keep_idx, col] = (train_X_ts[train_keep_idx, col] - ts_mean) / ts_std
        val_X_ts[val_keep_idx, col] = (val_X_ts[val_keep_idx, col] - ts_mean) / ts_std
        test_X_ts[test_keep_idx, col] = (test_X_ts[test_keep_idx, col] - ts_mean) / ts_std
    train_X = pd.DataFrame(np.concatenate([train_idx_time, train_X_ts, train_X_mask], axis=-1), columns=train_col)
    val_X = pd.DataFrame(np.concatenate([val_idx_time, val_X_ts, val_X_mask], axis=-1), columns=val_col)
    test_X = pd.DataFrame(np.concatenate([test_idx_time, test_X_ts, test_X_mask], axis=-1), columns=test_col)
    return train_X, val_X, test_X

def normalize_masked_data(data, mask, att_min, att_max):
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if np.isnan(data_norm).any():
        raise Exception("nans!")
    # set masked out elements back to zero
    data_norm[mask == 0] = 0
    return data_norm, att_min, att_max


def mae(P_mean, X, M):
    losses = torch.abs(P_mean-X) * M
    return losses.sum().cpu().detach().numpy()

def nll(P_mean, P_logvar, X, M):
    nll = nn.GaussianNLLLoss(reduction='none')
    losses = nll(P_mean, X, P_logvar.exp()) * M
    return losses.sum().cpu().detach().numpy()

def cross_entropy(pred, target):
    batch_loss = 0.
    for i in range(pred.shape[0]):
        num = torch.exp(pred[i, target[i]])
        den = torch.sum(torch.exp(pred[i, :]))
        loss = -torch.log(num / den)
        batch_loss = batch_loss + loss
    return batch_loss

def auc_pr(Y, pred_prob):
    auc = metrics.roc_auc_score(Y, pred_prob)
    (precisions, recalls, thresholds) = metrics.precision_recall_curve(Y, pred_prob)
    auprc = metrics.auc(recalls, precisions)
    return auc, auprc

def acc(Y, pred_prob):
    pred_trend = (pred_prob >= 0.5) * 1.0
    acc = metrics.accuracy_score(Y.astype(int), pred_trend.astype(int))
    return acc

def multi_auc(Y, pred_prob):
    auc = metrics.roc_auc_score(Y, pred_prob, multi_class='ovr')
    return auc

def mse(P_mean, X, M):
    losses = torch.pow(P_mean-X, 2) * M
    return losses.sum().cpu().detach().numpy()

def sum_mae(P, X, M):
    losses = np.abs(P-X) * M
    return losses.sum()

def sum_mse(P, X, M):
    losses = np.power(P-X, 2) * M
    return losses.sum()

def sum_accuracy(P, X):
    return (P == X).sum()

def sum_re(P, X, M):
    return ((np.abs(P - X) / X) * M).sum()

def conf_mat(P, X):
    accuracy = metrics.accuracy_score(X, P)
    recall = metrics.recall_score(X, P, average='macro')
    precision = metrics.precision_score(X, P, average='macro')
    f1 = metrics.f1_score(X, P, average='macro')
    return accuracy, recall, precision, f1