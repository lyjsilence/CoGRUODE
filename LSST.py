import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import utils
import config
from models_HM import CoGRUODE as CoGRUODE_HM
from models_HV import CoGRUODE as CoGRUODE_HV

from baseline_models import GRU_ODE, GRU_D, CTGRU, ODELSTM, GRU_delta_t, ODERNN
from model_training import LSST_training

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# model_list = ['CoGRUODE_HV', 'ComGRUODE_HV', 'CoGRUODE_HM', 'ComGRUODE_HM', 'GRUODE', 'mGRUODE',
#               'ODELSTM', 'ODERNN', 'GRU-D', 'CTGRU', 'GRU_delta_t']
parser = argparse.ArgumentParser(description="CoGRUODE for LSST datasets")
parser.add_argument('--seed', type=int, default=0, help='The random seed')
parser.add_argument('--save_dirs', type=str, default='results', help='The dirs for saving results')
parser.add_argument('--log', type=bool, default=True, help='Whether log the information of training process')

parser.add_argument('--dataset', type=str, default='LSST', help='The dataset the need to train')
parser.add_argument('--model_name', type=str, default='CoGRUODE_HV', help='The model want to implement')
parser.add_argument('--num_exp', type=int, default=5, help='The number of experiment')
parser.add_argument('--batch_size', type=int, default=256, help='The batch size when training NN')
parser.add_argument('--memory', type=str, default='both', help='The memory want to implement')
parser.add_argument('--n_dim', type=int, default=20, help='Dimension of marginal memory for one variable')
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--dt', type=float, default=0.02)
parser.add_argument('--missing_rate', type=str, default='0.5')

args = parser.parse_args()

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_path = os.path.join(args.save_dirs, 'LSST')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_path = os.path.join('data', args.dataset)

    X = pd.read_csv(os.path.join(data_path, 'X_'+str(args.missing_rate)+'.csv'))
    Y = pd.read_csv(os.path.join(data_path, 'Y.csv'))

    # train, val, test split
    data_idx = np.arange(len(Y))
    train_idx = np.random.choice(data_idx, int(0.7 * len(Y)), replace=False)
    data_idx = data_idx[~np.in1d(data_idx, train_idx)]
    val_idx = np.random.choice(data_idx, int(0.15 * len(Y)), replace=False)
    data_idx = data_idx[~np.in1d(data_idx, val_idx)]
    test_idx = data_idx

    train_row, val_row, test_row = [], [], []
    for row in range(len(X)):
        if X.loc[row, 'idx'] in train_idx:
            train_row.append(row)
        elif X.loc[row, 'idx'] in val_idx:
            val_row.append(row)
        elif X.loc[row, 'idx'] in test_idx:
            test_row.append(row)

    train_X, val_X, test_X = X.iloc[train_row], X.iloc[val_row], X.iloc[test_row]
    train_Y, val_Y, test_Y = Y.iloc[train_idx], Y.iloc[val_idx], Y.iloc[test_idx]

    train_X, val_X, test_X = utils.normalization(train_X, val_X, test_X)
    train_data = utils.UEA_dataset(train_X, train_Y)
    val_data = utils.UEA_dataset(val_X, val_Y)
    test_data = utils.UEA_dataset(test_X, test_Y)

    dl_train = DataLoader(dataset=train_data, collate_fn=utils.UEA_collate_fn,
                          shuffle=True, batch_size=args.batch_size, num_workers=1, pin_memory=False)
    dl_val = DataLoader(dataset=val_data, collate_fn=utils.UEA_collate_fn,
                          shuffle=False, batch_size=args.batch_size, num_workers=1, pin_memory=False)
    dl_test = DataLoader(dataset=test_data, collate_fn=utils.UEA_collate_fn,
                         shuffle=False, batch_size=args.batch_size, num_workers=1, pin_memory=False)


    for exp_id in range(args.num_exp):
        model_name = args.model_name
        if model_name == 'CoGRUODE_HM':
            model = CoGRUODE_HM(config.config(model_name, args), device).to(device)
        if model_name == 'CoGRUODE_HV':
            model = CoGRUODE_HV(config.config(model_name, args), device).to(device)
        elif model_name == 'ComGRUODE_HM':
            model = CoGRUODE_HM(config.config(model_name, args), device).to(device)
        elif model_name == 'ComGRUODE_HV':
            model = CoGRUODE_HV(config.config(model_name, args), device).to(device)
        elif model_name == 'GRUODE':
            model = GRU_ODE(config.config(model_name, args)).to(device)
        elif model_name == 'mGRUODE':
            model = GRU_ODE(config.config(model_name, args)).to(device)
        elif model_name == 'ODELSTM':
            model = ODELSTM(config.config(model_name, args)).to(device)
        elif model_name == 'ODERNN':
            model = ODERNN(config.config(model_name, args)).to(device)
        elif model_name == 'GRU-D':
            model = GRU_D(config.config(model_name, args)).to(device)
        elif model_name == 'GRU_delta_t':
            model = GRU_delta_t(config.config(model_name, args)).to(device)
        elif model_name == 'CTGRU':
            model = CTGRU(config.config(model_name, args)).to(device)
        else:
            ModuleNotFoundError(f'Module {model_name} not found')

        print(f'Training model {model_name}')
        LSST_training(model, model_name, dl_train, dl_val, dl_test, args, device, exp_id)