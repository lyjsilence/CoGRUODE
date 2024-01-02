import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import utils
from models import CoGRUODE
import random
from baselines import GRU_ODE, GRU_D, ODELSTM, GRU_delta_t, ODERNN, Neural_CDE
from model_training import Trainer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description="CoGRUODE for TCH datasets")
parser.add_argument('--save_dirs', type=str, default='results', help='The dirs for saving results')
parser.add_argument('--log', type=bool, default=True, help='Whether log the information of training process')

parser.add_argument('--dataset', type=str, default='Call', help='The dataset the need to train')
parser.add_argument('--model_name', type=str, default='CoGRUODE', help='The model want to implement')
parser.add_argument('--num_exp', type=int, default=5, help='The number of experiment')
parser.add_argument('--epoch_max', type=int, default=150, help='The number of epoch for one experiment')
parser.add_argument('--lr', type=float, default=0.003, help='The learning rate when training NN')
parser.add_argument('--batch_size', type=int, default=128, help='The batch size when training NN')

parser.add_argument('--input_size', type=int, default=16, help='Input size')
parser.add_argument('--sub_series', type=int, default=2)
parser.add_argument('--n_dim', type=int, default=5, help='Dimension of marginal memory for one variable')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--solver', type=str, default='dopri5')
parser.add_argument('--dt', type=float, default=0.1)
args = parser.parse_args()


if __name__ == '__main__':
    set_seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_path = os.path.join(args.save_dirs, args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.dataset == 'Call':
        data = pd.read_csv('data/TCH_option_3min_call.csv')
    elif args.dataset == 'Put':
        data = pd.read_csv('data/TCH_option_3min_put.csv')
    else:
        raise 'No such dataset!'

    data = utils.preprocessing(data, norm=True)
    # train, val, test split
    data_idx = np.unique(np.array((data['idx'])))
    train_idx, val_idx = train_test_split(data_idx, train_size=0.7, random_state=42)
    val_idx, test_idx = train_test_split(val_idx, train_size=0.5, random_state=42)

    train_data = utils.option_dataset(args, data[data['idx'].isin(train_idx)])
    val_data = utils.option_dataset(args, data[data['idx'].isin(val_idx)])
    test_data = utils.option_dataset(args, data[data['idx'].isin(test_idx)])
    collate_fn = utils.option_collate_fn

    dl_train = DataLoader(dataset=train_data, collate_fn=utils.option_collate_fn,
                          shuffle=True, batch_size=args.batch_size, num_workers=1, pin_memory=False)
    dl_val = DataLoader(dataset=val_data, collate_fn=utils.option_collate_fn,
                          shuffle=False, batch_size=args.batch_size, num_workers=1, pin_memory=False)
    dl_test = DataLoader(dataset=test_data, collate_fn=utils.option_collate_fn,
                         shuffle=False, batch_size=args.batch_size, num_workers=1, pin_memory=False)

    for exp_id in range(args.num_exp):
        model_name = args.model_name
        if model_name == 'CoGRUODE':
            model = CoGRUODE(args, device).to(device)
        elif model_name == 'Neural-CDE':
            args.hidden_size = 50
            model = Neural_CDE(args, device).to(device)
        elif model_name == 'GRUODE':
            args.minimal = False
            args.hidden_size = 105
            model = GRU_ODE(args, device).to(device)
        elif model_name == 'mGRUODE':
            args.minimal = True
            args.hidden_size = 113
            model = GRU_ODE(args, device).to(device)
        elif model_name == 'ODELSTM':
            args.hidden_size = 97
            model = ODELSTM(args, device).to(device)
        elif model_name == 'ODERNN':
            args.hidden_size = 105
            model = ODERNN(args, device).to(device)
        elif model_name == 'GRU-D':
            args.hidden_size = 122
            model = GRU_D(args, device).to(device)
        elif model_name == 'GRU_delta_t':
            args.hidden_size = 135
            model = GRU_delta_t(args, device).to(device)
        else:
            ModuleNotFoundError(f'Module {model_name} not found')

        print(f'Training model: {model_name}, Experiment: {exp_id}, '
              f'Num of training parameters: {sum(p.numel() for p in model.parameters())}')
        Trainer(model, dl_train, dl_val, dl_test, args, device, exp_id)