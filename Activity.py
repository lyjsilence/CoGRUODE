import numpy as np
import pandas as pd
import torch
import os
import argparse
from sklearn import model_selection
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader
import utils
import config
from models_HM import CoGRUODE as CoGRUODE_HM
from models_HV import CoGRUODE as CoGRUODE_HV
from baseline_models import GRU_ODE, GRU_D, CTGRU, ODELSTM, GRU_delta_t, ODERNN
from model_training import Activity_training

class PersonActivity(object):
    urls = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt',
    ]

    tag_ids = [
        "010-000-024-033",  # "ANKLE_LEFT",
        "010-000-030-096",  # "ANKLE_RIGHT",
        "020-000-033-111",  # "CHEST",
        "020-000-032-221"  # "BELT"
    ]

    tag_dict = {k: i for i, k in enumerate(tag_ids)}

    label_names = [
        "walking",
        "falling",
        "lying down",
        "lying",
        "sitting down",
        "sitting",
        "standing up from lying",
        "on all fours",
        "sitting on the ground",
        "standing up from sitting",
        "standing up from sit on grnd"
    ]

    # label_dict = {k: i for i, k in enumerate(label_names)}

    # Merge similar labels into one class
    label_dict = {
        "walking": 0,
        "falling": 1,
        "lying": 2,
        "lying down": 2,
        "sitting": 3,
        "sitting down": 3,
        "standing up from lying": 4,
        "standing up from sitting": 4,
        "standing up from sit on grnd": 4,
        "on all fours": 5,
        "sitting on the ground": 6
    }

    def __init__(self, root, download=False, reduce='average', max_seq_length=50, n_samples=None, device=torch.device("cpu")):

        self.root = root
        self.reduce = reduce
        self.max_seq_length = max_seq_length

        if download:
            if os.path.exists(os.path.join(self.processed_folder, 'data.pt')):
                pass
            else:
                self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        if device == torch.device("cpu"):
            self.data = torch.load(os.path.join(self.processed_folder, self.data_file), map_location='cpu')
        else:
            self.data = torch.load(os.path.join(self.processed_folder, self.data_file))

        if n_samples is not None:
            self.data = self.data[:n_samples]

        self.make_csv(self.processed_folder)

    def download(self):
        if self._check_exists():
            return

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        def save_record(records, record_id, tt, vals, mask, labels):
            tt = torch.tensor(tt).to(self.device)

            vals = torch.stack(vals)
            mask = torch.stack(mask)
            labels = torch.stack(labels)

            # flatten the measurements for different tags
            vals = vals.reshape(vals.size(0), -1)
            mask = mask.reshape(mask.size(0), -1)
            assert (len(tt) == vals.size(0))
            assert (mask.size(0) == vals.size(0))
            assert (labels.size(0) == vals.size(0))

            # records.append((record_id, tt, vals, mask, labels))

            seq_length = len(tt)
            # split the long time series into smaller ones
            offset = 0
            slide = self.max_seq_length // 2

            while (offset + self.max_seq_length < seq_length):
                idx = range(offset, offset + self.max_seq_length)

                first_tp = tt[idx][0]
                records.append((record_id, tt[idx] - first_tp, vals[idx], mask[idx], labels[idx]))
                offset += slide

        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename, None)

            print('Processing {}...'.format(filename))

            dirname = os.path.join(self.raw_folder)
            records = []
            first_tp = None

            for txtfile in os.listdir(dirname):
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = -1
                    tt = []

                    record_id = None
                    for l in lines:
                        cur_record_id, tag_id, time, date, val1, val2, val3, label = l.strip().split(',')
                        value_vec = torch.Tensor((float(val1), float(val2), float(val3))).to(self.device)
                        time = float(time)

                        if cur_record_id != record_id:
                            if record_id is not None:
                                save_record(records, record_id, tt, vals, mask, labels)
                            tt, vals, mask, nobs, labels = [], [], [], [], []
                            record_id = cur_record_id

                            tt = [torch.zeros(1).to(self.device)]
                            vals = [torch.zeros(len(self.tag_ids), 3).to(self.device)]
                            mask = [torch.zeros(len(self.tag_ids), 3).to(self.device)]
                            nobs = [torch.zeros(len(self.tag_ids)).to(self.device)]
                            labels = [torch.zeros(len(self.label_names)).to(self.device)]

                            first_tp = time
                            time = round((time - first_tp) / 10 ** 5)
                            prev_time = time
                        else:
                            # for speed -- we actually don't need to quantize it in Latent ODE
                            time = round((time - first_tp) / 10 ** 5)  # quatizing by 100 ms. 10,000 is one millisecond, 10,000,000 is one second

                        if time != prev_time:
                            tt.append(time)
                            vals.append(torch.zeros(len(self.tag_ids), 3).to(self.device))
                            mask.append(torch.zeros(len(self.tag_ids), 3).to(self.device))
                            nobs.append(torch.zeros(len(self.tag_ids)).to(self.device))
                            labels.append(torch.zeros(len(self.label_names)).to(self.device))
                            prev_time = time

                        if tag_id in self.tag_ids:
                            n_observations = nobs[-1][self.tag_dict[tag_id]]
                            if (self.reduce == 'average') and (n_observations > 0):
                                prev_val = vals[-1][self.tag_dict[tag_id]]
                                new_val = (prev_val * n_observations + value_vec) / (n_observations + 1)
                                vals[-1][self.tag_dict[tag_id]] = new_val
                            else:
                                vals[-1][self.tag_dict[tag_id]] = value_vec

                            mask[-1][self.tag_dict[tag_id]] = 1
                            nobs[-1][self.tag_dict[tag_id]] += 1

                            if label in self.label_names:
                                if torch.sum(labels[-1][self.label_dict[label]]) == 0:
                                    labels[-1][self.label_dict[label]] = 1
                        else:
                            assert tag_id == 'RecordID', 'Read unexpected tag id {}'.format(tag_id)
                    save_record(records, record_id, tt, vals, mask, labels)

            torch.save(
                records, os.path.join(self.processed_folder, 'data.pt')
            )

        print('Done!')

    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition('/')[2]
            if not os.path.exists(
                    os.path.join(self.processed_folder, 'data.pt')
            ):
                return False
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def data_file(self):
        return 'data.pt'

    def make_csv(self, save_path):
        if os.path.exists(os.path.join(save_path, 'X.csv')) and os.path.exists(os.path.join(save_path, 'Y.csv')):
            pass
        else:
            df = pd.DataFrame()

            for b, (record_id, time, vals, mask, labels) in enumerate(self.data):
                time, vals, mask, labels = time.cpu().numpy(), vals.cpu().numpy(), mask.cpu().numpy(), labels.cpu().numpy()
                time = np.round(np.expand_dims(time / np.max(time), axis=1), 2) + 0.01
                data = np.concatenate([np.expand_dims(np.array([b] * len(time)), axis=1), time, vals, mask,
                                       np.expand_dims(np.argmax(labels, axis=1), axis=1)], axis=1)
                df = pd.concat([df, pd.DataFrame(data)], axis=0)

            df.columns = ['idx', 'time']+['ts_'+str(i) for i in range(12)]+['mask_'+str(i) for i in range(12)]+['targets']
            # normalization
            for col in range(12):
                df['ts_'+str(col)] = df['ts_'+str(col)] / np.max(df['ts_'+str(col)])
            df.to_csv(os.path.join(save_path, 'data.csv'), index=False)

parser = argparse.ArgumentParser(description="Human Activatity dataset training")
parser.add_argument('--seed', type=int, default=0, help='The random seed')
parser.add_argument('--save_dirs', type=str, default='results', help='The dirs for saving results')
parser.add_argument('--log', type=bool, default=True, help='Whether log the information of training process')

parser.add_argument('--dataset', type=str, default='Activity')
parser.add_argument('--model_name', type=str, default='CoGRUODE_HV', help='The model want to implement')
parser.add_argument('--num_exp', type=int, default=5, help='The number of experiment')
parser.add_argument('--batch_size', type=int, default=64, help='The batch size when training NN')
parser.add_argument('--memory', type=str, default='both', help='The memory want to implement')
parser.add_argument('--n_dim', type=int, default=20)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--dt', type=float, default=0.01)

args = parser.parse_args()

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # if you are first running this program, you need preprocessing this dataset in PersonActivity
    data_path = './data/PersonActivity/processed/data.csv'
    if not os.path.exists(data_path):
        PersonActivity('data', download=True, device=device)
    else:
        pass
    data = pd.read_csv(data_path)
    data_idx = np.arange(len(set(data['idx'])))

    # train-test split
    train_idx, test_idx = model_selection.train_test_split(data_idx, train_size=0.8, random_state=42)
    train_data = utils.Activity_dataset(data.loc[data['idx'].isin(train_idx), :])
    test_data = utils.Activity_dataset(data.loc[data['idx'].isin(test_idx), :])

    dl_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=utils.Activity_collate_fn)
    dl_test = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=utils.Activity_collate_fn)

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
        Activity_training(model, model_name, dl_train, dl_test, args, device, exp_id)



