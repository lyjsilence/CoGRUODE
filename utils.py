import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class sim_dataset(Dataset):
    def __init__(self, ts, idx, args, val=False, val_threshold=15, test=False, test_threshold=15):
        assert ts is not None
        assert idx is not None
        assert val is not bool
        assert test is not bool

        # data format: [index, time, ts_1, ts_2..., mask_1, mask_2...]
        # Extract the time series with certain index and reindex the time series
        self.ts = ts[ts['idx'].isin(idx)].copy()
        # the number of unique index of time series
        self.length = len(np.unique(self.ts['idx']))

        map_dict = dict(zip(self.ts["idx"].unique(), np.arange(self.ts["idx"].nunique())))
        self.ts["idx"] = self.ts["idx"].map(map_dict)

        self.val = val
        self.test = test

        if self.val:
            assert val_threshold is not None
            # data with time less than val_threshold for training
            # data with time more than val_threshold for validation, only the closest will be chosen
            self.ts_train = self.ts[self.ts['time'] < val_threshold].copy()
            self.ts_val = self.ts[self.ts['time'] >= val_threshold].copy()
            self.ts_val.sort_values(by=["idx", "time"], inplace=True)
            self.ts_val = self.ts_val.groupby('idx').head(args.num_test)
            self.ts = self.ts_train
        else:
            self.ts_val = None

        if self.test:
            assert test_threshold is not None
            # data with time less than test_threshold for training
            # data with time more than test_threshold for test, only the closest will be chosen
            self.ts_train = self.ts[self.ts['time'] < test_threshold].copy()
            self.ts_test = self.ts[self.ts['time'] >= test_threshold].copy()
            self.ts_test.sort_values(by=["idx", "time"], inplace=True)
            self.ts_test = self.ts_test.groupby('idx').head(args.num_test)
            self.ts = self.ts_train
        else:
            self.ts_test = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_ts = self.ts[self.ts['idx'] == idx]
        if self.val:
            batch_ts_val = self.ts_val[self.ts_val['idx'] == idx]
        else:
            batch_ts_val = None

        if self.test:
            batch_ts_test = self.ts_test[self.ts_test['idx'] == idx]
        else:
            batch_ts_test = None

        return {'ts': batch_ts, 'ts_val': batch_ts_val, 'ts_test': batch_ts_test}

class UEA_dataset(Dataset):
    def __init__(self, ts, targets):
        assert ts is not None
        assert targets is not None

        # data format: [index, time, ts_1, ts_2..., mask_1, mask_2...]
        self.ts = ts.copy()
        self.target = targets.copy()
        # the number of unique index of time series
        self.length = len(np.unique(self.ts['idx']))

        map_dict = dict(zip(self.ts["idx"].unique(), np.arange(self.ts["idx"].nunique())))
        self.ts["idx"] = self.ts["idx"].map(map_dict)
        self.target["idx"] = self.target["idx"].map(map_dict)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_ts = self.ts[self.ts['idx'] == idx]
        batch_target = self.target[self.target['idx'] == idx]
        return {'ts': batch_ts, 'ts_targets': batch_target}

class Activity_dataset(Dataset):
    def __init__(self, ts):
        assert ts is not None

        # data format: [index, time, ts_1, ts_2..., mask_1, mask_2..., targets]
        self.ts = ts.copy()
        # the number of unique index of time series
        self.length = len(np.unique(self.ts['idx']))
        map_dict = dict(zip(self.ts["idx"].unique(), np.arange(self.ts["idx"].nunique())))
        self.ts["idx"] = self.ts["idx"].map(map_dict)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_ts = self.ts[self.ts['idx'] == idx]
        return {'ts': batch_ts}

class PhysioNet_dataset(Dataset):
    def __init__(self, ts, targets):
        assert ts is not None
        assert targets is not None

        # data format: [index, time, ts_1, ts_2..., mask_1, mask_2...]
        self.ts = ts.copy()
        self.target = targets.copy()
        # the number of unique index of time series
        self.length = len(np.unique(self.ts['idx']))

        map_dict = dict(zip(self.ts["idx"].unique(), np.arange(self.ts["idx"].nunique())))
        self.ts["idx"] = self.ts["idx"].map(map_dict)
        self.target["idx"] = self.target["idx"].map(map_dict)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_ts = self.ts[self.ts['idx'] == idx]
        batch_target = self.target[self.target['idx'] == idx]
        return {'ts': batch_ts, 'ts_targets': batch_target}

class Stock_dataset(Dataset):
    def __init__(self, ts, mode):
        assert ts is not None
        assert mode is not None
        # data format: [index, time, ts_1, ts_2..., mask_1, mask_2..., targets]
        self.ts = ts.copy()

        # the number of unique index of time series
        self.length = 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_ts = self.ts
        return {'ts': batch_ts}

def sim_collate_fn(batch):
    batch_ts = pd.concat(b['ts'] for b in batch)
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
    X = torch.FloatTensor(batch_ts.loc[:, [c.startswith("ts") for c in batch_ts.columns]].values)
    M = torch.FloatTensor(batch_ts.loc[:, [c.startswith("mask") for c in batch_ts.columns]].values)
    batch_idx = torch.FloatTensor(batch_ts.loc[:, 'batch_idx'].values)

    if batch[0]['ts_val'] is not None:
        batch_val_ts = pd.concat(b['ts_val'] for b in batch)
        sample_idx_val = pd.unique(batch_val_ts['idx'])
        for idx in sample_idx_val:
            batch_val_ts.loc[batch_val_ts['idx'] == idx, 'batch_idx'] = map_dict[idx]

        batch_val_ts.sort_values(by=["idx", "time"], inplace=True)
        X_val = torch.tensor(batch_val_ts.loc[:, [c.startswith("ts") for c in batch_val_ts.columns]].values)
        M_val = torch.tensor(batch_val_ts.loc[:, [c.startswith("mask") for c in batch_val_ts.columns]].values)
        obs_times_val = batch_val_ts["time"].values
        batch_idx_val = batch_val_ts.loc[:, 'batch_idx'].values
    else:
        X_val = None
        M_val = None
        obs_times_val = None
        batch_idx_val = None

    if batch[0]['ts_test'] is not None:
        batch_test_ts = pd.concat(b['ts_test'] for b in batch)
        batch_idx_test = pd.unique(batch_test_ts['idx'])
        for idx in batch_idx_test:
            batch_test_ts.loc[batch_test_ts['idx'] == idx, 'batch_idx'] = map_dict[idx]

        batch_test_ts.sort_values(by=["idx", "time"], inplace=True)
        X_test = torch.tensor(batch_test_ts.loc[:, [c.startswith("ts") for c in batch_test_ts.columns]].values)
        M_test = torch.tensor(batch_test_ts.loc[:, [c.startswith("mask") for c in batch_test_ts.columns]].values)
        obs_times_test = batch_test_ts["time"].values
        batch_idx_test = batch_test_ts.loc[:, 'batch_idx'].values
    else:
        X_test = None
        M_test = None
        obs_times_test = None
        batch_idx_test = None

    res = {}
    res["sample_idx"] = sample_idx
    res["obs_times"] = obs_times
    res["event_pt"] = event_pt
    res["X"] = X
    res["M"] = M
    res["batch_idx"] = batch_idx

    res["X_val"] = X_val
    res["M_val"] = M_val
    res["obs_times_val"] = obs_times_val
    res["batch_idx_val"] = batch_idx_val

    res["X_test"] = X_test
    res["M_test"] = M_test
    res["obs_times_test"] = obs_times_test
    res["batch_idx_test"] = batch_idx_test

    return res

def UEA_collate_fn(batch):
    batch_ts = pd.concat(b['ts'] for b in batch)
    batch_target = pd.concat(b['ts_targets'] for b in batch)
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
    X = torch.FloatTensor(batch_ts.loc[:, [c.startswith("ts") for c in batch_ts.columns]].values)
    M = torch.FloatTensor(batch_ts.loc[:, [c.startswith("mask") for c in batch_ts.columns]].values)
    batch_idx = torch.FloatTensor(batch_ts.loc[:, 'batch_idx'].values)

    batch_target = torch.as_tensor(np.array(batch_target['targets']), dtype=torch.int64)

    res = {}
    res["sample_idx"] = sample_idx
    res["obs_times"] = obs_times
    res["event_pt"] = event_pt
    res["X"] = X
    res["M"] = M
    res["batch_idx"] = batch_idx
    res["targets"] = batch_target

    return res


def Activity_collate_fn(batch):
    batch_ts = pd.concat(b['ts'] for b in batch)
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
    X = torch.FloatTensor(batch_ts.loc[:, [c.startswith("ts") for c in batch_ts.columns]].values)
    M = torch.FloatTensor(batch_ts.loc[:, [c.startswith("mask") for c in batch_ts.columns]].values)
    batch_target = torch.FloatTensor(batch_ts.loc[:, 'targets'].values)
    batch_idx = torch.FloatTensor(batch_ts.loc[:, 'batch_idx'].values)

    res = {}
    res["sample_idx"] = sample_idx
    res["obs_times"] = obs_times
    res["event_pt"] = event_pt
    res["X"] = X
    res["M"] = M
    res["batch_idx"] = batch_idx
    res["targets"] = batch_target

    return res

def PhysioNet_collate_fn(batch):
    batch_ts = pd.concat(b['ts'] for b in batch)
    batch_target = pd.concat(b['ts_targets'] for b in batch)
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
    X = torch.FloatTensor(batch_ts.loc[:, [c.startswith("ts") for c in batch_ts.columns]].values)
    M = torch.FloatTensor(batch_ts.loc[:, [c.startswith("mask") for c in batch_ts.columns]].values)
    batch_idx = torch.FloatTensor(batch_ts.loc[:, 'batch_idx'].values)

    batch_target = torch.as_tensor(np.array(batch_target['targets']), dtype=torch.int64)

    res = {}
    res["sample_idx"] = sample_idx
    res["obs_times"] = obs_times
    res["event_pt"] = event_pt
    res["X"] = X
    res["M"] = M
    res["batch_idx"] = batch_idx
    res["targets"] = batch_target

    return res

def Stock_collate_fn(batch):
    sample_idx = np.array(np.expand_dims([1], axis=0))


    # calculating number of events at every time
    obs_times, counts = np.unique(batch[0]['ts'].time.values, return_counts=True)
    event_pt = np.concatenate([[0], np.cumsum(counts)])

    # convert data to tensor
    X = torch.FloatTensor(batch[0]['ts'].loc[:, [c.startswith("ts") for c in batch[0]['ts'].columns]].values)
    M = torch.FloatTensor(batch[0]['ts'].loc[:, [c.startswith("mask") for c in batch[0]['ts'].columns]].values)
    batch[0]['ts']['batch_idx'] = 0

    batch_idx = torch.FloatTensor( batch[0]['ts'].loc[:, 'batch_idx'].values)
    batch_target = torch.as_tensor(np.array(batch[0]['ts']['targets']), dtype=torch.int64)

    res = {}
    res["sample_idx"] = sample_idx
    res["obs_times"] = obs_times
    res["event_pt"] = event_pt
    res["X"] = X
    res["M"] = M
    res["batch_idx"] = batch_idx
    res["targets"] = batch_target

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