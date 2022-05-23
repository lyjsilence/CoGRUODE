import numpy as np
import pandas as pd
import os
import argparse
import utils
import config
from models_HM import CoGRUODE as CoGRUODE_HM
from models_HV import CoGRUODE as CoGRUODE_HV
from baseline_models import GRU_ODE, GRU_D, CTGRU, ODELSTM, GRU_delta_t, ODERNN
from model_training import PhysioNet_training
import matplotlib.pyplot as plt
import tarfile
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url

# get minimum and maximum for each feature across the whole dataset
def get_data_min_max(records):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0]

    for b, (record_id, tt, vals, mask, labels) in enumerate(records):
        vals, mask = vals[:, 4:], mask[:, 4:]
        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:, i][mask[:, i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(inf.to(device))
                batch_max.append(-inf.to(device))
            else:
                batch_min.append(torch.min(non_missing_vals).to(device))
                batch_max.append(torch.max(non_missing_vals).to(device))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    return data_min, data_max

# Adapted from: https://github.com/rtqichen/time-series-datasets
class PhysioNet(object):

    urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz',
        'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz',
        'https://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz'
    ]

    outcome_urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt',
        'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt',
        'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt'
    ]

    params = ["Age", "Gender", "Height", "ICUType", 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
        'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
        'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
    ]

    params_dict = {k: i for i, k in enumerate(params)}

    labels = ["SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death"]
    labels_dict = {k: i for i, k in enumerate(labels)}

    def __init__(self, root, train=True, download=False,
                 quantization=0.016, n_samples=None, device=torch.device("cpu")):

        self.root = root
        self.train = train
        self.reduce = "average"
        self.quantization = quantization

        if os.path.exists(os.path.join(self.processed_folder, 'Outcomes-a.pt')) and \
           os.path.exists(os.path.join(self.processed_folder, 'Outcomes-b.pt')) and \
           os.path.exists(os.path.join(self.processed_folder, 'Outcomes-c.pt')) and \
           os.path.exists(os.path.join(self.processed_folder, 'set-a_0.016.pt')) and \
           os.path.exists(os.path.join(self.processed_folder, 'set-b_0.016.pt')) and \
           os.path.exists(os.path.join(self.processed_folder, 'set-c_0.016.pt')):
            pass
        else:
            if download:
                self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        data_file_a = 'set-a_{}.pt'.format(self.quantization)
        data_file_b = 'set-b_{}.pt'.format(self.quantization)
        data_file_c = 'set-b_{}.pt'.format(self.quantization)

        if device == torch.device("cpu"):
            self.data_a = torch.load(os.path.join(self.processed_folder, data_file_a), map_location='cpu')
            self.data_b = torch.load(os.path.join(self.processed_folder, data_file_b), map_location='cpu')
            self.train_data = self.data_a + self.data_b

            self.data_c = torch.load(os.path.join(self.processed_folder, data_file_c), map_location='cpu')
            self.test_data = self.data_c
        else:
            self.data_a = torch.load(os.path.join(self.processed_folder, data_file_a), map_location='cpu')
            self.data_b = torch.load(os.path.join(self.processed_folder, data_file_b), map_location='cpu')
            self.train_data = self.data_a + self.data_b

            self.data_c = torch.load(os.path.join(self.processed_folder, data_file_c), map_location='cpu')
            self.test_data = self.data_c

        self.make_csv(self.processed_folder)

    def download(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        for count, url in enumerate(self.urls):
            # Download outcome data
            outcome_url = self.outcome_urls[count]
            filename = outcome_url.rpartition('/')[2]
            download_url(outcome_url, self.raw_folder, filename, None)

            txtfile = os.path.join(self.raw_folder, filename)
            with open(txtfile) as f:
                lines = f.readlines()
                outcomes = {}
                for l in lines[1:]:
                    l = l.rstrip().split(',')
                    record_id, labels = l[0], np.array(l[1:]).astype(float)
                    outcomes[record_id] = torch.Tensor(labels).to(self.device)

                torch.save(
                    labels,
                    os.path.join(self.processed_folder, filename.split('.')[0] + '.pt')
                )

            # download events data
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename, None)
            tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
            tar.extractall(self.raw_folder)
            tar.close()

            print('Processing {}...'.format(filename))

            dirname = os.path.join(self.raw_folder, filename.split('.')[0])
            patients = []
            total = 0
            for txtfile in os.listdir(dirname):
                record_id = txtfile.split('.')[0]
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = 0
                    tt = [0.]
                    vals = [torch.zeros(len(self.params)).to(self.device)]
                    mask = [torch.zeros(len(self.params)).to(self.device)]
                    nobs = [torch.zeros(len(self.params))]
                    for l in lines[1:]:
                        total += 1
                        time, param, val = l.split(',')
                        # Time in hours
                        time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.
                        # round up the time stamps (up to 6 min by default)
                        # used for speed -- we actually don't need to quantize it in Latent ODE
                        time = round(time / self.quantization) * self.quantization

                        if time != prev_time:
                            tt.append(time)
                            vals.append(torch.zeros(len(self.params)).to(self.device))
                            mask.append(torch.zeros(len(self.params)).to(self.device))
                            nobs.append(torch.zeros(len(self.params)).to(self.device))
                            prev_time = time

                        if param in self.params_dict:
                            # vals[-1][self.params_dict[param]] = float(val)
                            n_observations = nobs[-1][self.params_dict[param]]
                            if self.reduce == 'average' and n_observations > 0:
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                vals[-1][self.params_dict[param]] = float(val)
                            mask[-1][self.params_dict[param]] = 1
                            nobs[-1][self.params_dict[param]] += 1

                tt = torch.tensor(tt).to(self.device)
                vals = torch.stack(vals)
                mask = torch.stack(mask)

                labels = None
                if record_id in outcomes:
                    # Only training set has labels
                    labels = outcomes[record_id]
                    # Out of 5 label types provided for Physionet, take only the last one -- mortality
                    labels = labels[4]

                patients.append((record_id, tt, vals, mask, labels))

            torch.save(
                patients,
                os.path.join(self.processed_folder,
                             filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            )

        print('Done!')

    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition('/')[2]
            a = os.path.join(self.processed_folder,
                                 filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            if not os.path.exists(
                    os.path.join(self.processed_folder,
                                 filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            ):
                return False
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def visualize(self, timesteps, data, mask, plot_name):
        width = 15
        height = 15

        non_zero_attributes = (torch.sum(mask, 0) > 2).numpy()
        non_zero_idx = [i for i in range(len(non_zero_attributes)) if non_zero_attributes[i] == 1.]
        n_non_zero = sum(non_zero_attributes)

        mask = mask[:, non_zero_idx]
        data = data[:, non_zero_idx]

        params_non_zero = [self.params[i] for i in non_zero_idx]
        params_dict = {k: i for i, k in enumerate(params_non_zero)}

        n_col = 3
        n_row = n_non_zero // n_col + (n_non_zero % n_col > 0)
        fig, ax_list = plt.subplots(n_row, n_col, figsize=(width, height), facecolor='white')

        # for i in range(len(self.params)):
        for i in range(n_non_zero):
            param = params_non_zero[i]
            param_id = params_dict[param]

            tp_mask = mask[:, param_id].long()

            tp_cur_param = timesteps[tp_mask == 1.]
            data_cur_param = data[tp_mask == 1., param_id]

            ax_list[i // n_col, i % n_col].plot(tp_cur_param.numpy(), data_cur_param.numpy(), marker='o')
            ax_list[i // n_col, i % n_col].set_title(param)

        fig.tight_layout()
        fig.savefig(plot_name)
        plt.close(fig)

    def make_csv(self, save_path):
        df_train_X, df_test_X = pd.DataFrame(), pd.DataFrame()
        df_train_Y, df_test_Y = pd.DataFrame(columns=['idx', 'targets']), pd.DataFrame(columns=['idx', 'targets'])
        for b, (record_id, time, vals, mask, label) in enumerate(self.train_data):
            time, vals, mask, label = time.cpu().numpy(), vals.cpu().numpy(), mask.cpu().numpy(), label.cpu().numpy()
            time = np.round(np.expand_dims(time, axis=1), 2) + 0.01
            # delete the first four features that does not change with time, i.e., "Age", "Gender", "Height", "ICUType"
            X = np.concatenate([np.expand_dims(np.array([b] * len(time)), axis=1), time, vals[:, 4:], mask[:, 4:]], axis=1)
            df_train_X = pd.concat([df_train_X, pd.DataFrame(X)], axis=0)
            df_train_Y.loc[b, 'idx'], df_train_Y.loc[b, 'targets'] = b, label

        df_train_X.columns = ['idx', 'time']+['ts_'+str(i) for i in range(37)]+['mask_'+str(i) for i in range(37)]
        df_train_Y.columns = ['idx', 'targets']

        self.data_min, self.data_max = get_data_min_max(self.train_data)

        combined_vals = np.array(df_train_X.loc[:, [c.startswith("ts") for c in df_train_X.columns]].values)
        combined_mask = np.array(df_train_X.loc[:, [c.startswith("mask") for c in df_train_X.columns]].values)
        combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask,
                                                          att_min=self.data_min.cpu().numpy(), att_max=self.data_max.cpu().numpy())
        df_train_X.loc[:, [c.startswith("ts") for c in df_train_X.columns]] = combined_vals

        for b, (record_id, time, vals, mask, label) in enumerate(self.test_data):
            time, vals, mask, label = time.cpu().numpy(), vals.cpu().numpy(), mask.cpu().numpy(), label.cpu().numpy()
            time = np.round(np.expand_dims(time, axis=1), 2) + 0.01
            # delete the first four features that does not change with time, i.e., "Age", "Gender", "Height", "ICUType"
            X = np.concatenate([np.expand_dims(np.array([b] * len(time)), axis=1), time, vals[:, 4:], mask[:, 4:]], axis=1)
            df_test_X = pd.concat([df_test_X, pd.DataFrame(X)], axis=0)
            df_test_Y.loc[b, 'idx'], df_test_Y.loc[b, 'targets'] = b, label

        df_test_X.columns = ['idx', 'time']+['ts_'+str(i) for i in range(37)]+['mask_'+str(i) for i in range(37)]
        df_test_Y.columns = ['idx', 'targets']

        combined_vals = np.array(df_test_X.loc[:, [c.startswith("ts") for c in df_test_X.columns]].values)
        combined_mask = np.array(df_test_X.loc[:, [c.startswith("mask") for c in df_test_X.columns]].values)
        combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask,
                                                          att_min=self.data_min.cpu().numpy(), att_max=self.data_max.cpu().numpy())
        df_test_X.loc[:, [c.startswith("ts") for c in df_test_X.columns]] = combined_vals

        df_train_X.to_csv(os.path.join(save_path, 'train_X.csv'), index=False)
        df_train_Y.to_csv(os.path.join(save_path, 'train_Y.csv'), index=False)
        df_test_X.to_csv(os.path.join(save_path, 'test_X.csv'), index=False)
        df_test_Y.to_csv(os.path.join(save_path, 'test_Y.csv'), index=False)


parser = argparse.ArgumentParser(description="PhysioNet dataset training")
parser.add_argument('--seed', type=int, default=0, help='The random seed')
parser.add_argument('--save_dirs', type=str, default='results', help='The dirs for saving results')
parser.add_argument('--log', type=bool, default=True, help='Whether log the information of training process')

parser.add_argument('--dataset', type=str, default='PhysioNet')
parser.add_argument('--model_name', type=str, default='CoGRUODE_HV', help='The model want to implement')
parser.add_argument('--num_exp', type=int, default=5, help='The number of experiment')
parser.add_argument('--batch_size', type=int, default=500, help='The batch size when training NN')
parser.add_argument('--memory', type=str, default='both', help='The memory want to implement')
parser.add_argument('--n_dim', type=int, default=20)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--dt', type=float, default=0.1)

args = parser.parse_args()



if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # if you are first running this program, you need preprocessing this dataset in PersonActivity
    data_path = './data/PhysioNet/processed'
    if not os.path.exists(os.path.join(data_path, 'train_X.csv')):
        PhysioNet('data', train=True, quantization=0.016, download=True, device=device)
        PhysioNet('data', train=False, quantization=0.016, download=True, device=device)
    else:
        pass
    data = pd.read_csv(os.path.join(data_path, 'train_X.csv'))
    data_idx = np.arange(len(set(data['idx'])))

    train_X, train_Y = pd.read_csv(os.path.join(data_path, 'train_X.csv')), pd.read_csv(os.path.join(data_path, 'train_Y.csv'))
    test_X, test_Y = pd.read_csv(os.path.join(data_path, 'test_X.csv')), pd.read_csv(os.path.join(data_path, 'test_Y.csv'))

    train_data = utils.PhysioNet_dataset(train_X, train_Y)
    test_data = utils.PhysioNet_dataset(test_X, test_Y)

    dl_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=utils.PhysioNet_collate_fn)
    dl_test = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=utils.PhysioNet_collate_fn)

    for exp_id in range(0, args.num_exp):
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
        PhysioNet_training(model, model_name, dl_train, dl_test, args, device, exp_id)