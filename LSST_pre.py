from scipy.io import arff
import numpy as np
import pandas as pd
import os
import pathlib
import requests
import zipfile


def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))

def unzip(dataset, data_root_dir):
    zip_file = os.path.join(data_root_dir, dataset + '.zip')
    this_data_dir = os.path.join(data_root_dir, dataset)
    if not os.path.isdir(this_data_dir):
        os.makedirs(this_data_dir)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(this_data_dir)

def Preprocessing(dataset, data_root_dir, missing_rate):
    root_dir = os.path.join(data_root_dir, dataset)

    train_data_dir = os.path.join(root_dir, dataset + '_TRAIN.arff')
    test_data_dir = os.path.join(root_dir, dataset + '_TEST.arff')
    training_data = arff.loadarff(train_data_dir)
    test_data = arff.loadarff(test_data_dir)
    df_train, df_test = pd.DataFrame(training_data[0]), pd.DataFrame(test_data[0])
    data_column, label_columns = df_train.columns[0], df_test.columns[1]
    train_data_list, test_data_list = [], []

    for _, this_row in df_train[data_column].iteritems():
        train_data_list.append(np.array(this_row.tolist(), dtype='float32'))
    train_X = np.transpose(np.array(train_data_list), (0, 2, 1))
    train_Y = df_train[label_columns].astype('category').cat.codes
    for _, this_row in df_test[data_column].iteritems():
        test_data_list.append(np.array(this_row.tolist(), dtype='float32'))
    test_X = np.transpose(np.array(test_data_list), (0, 2, 1))
    test_Y = df_test[label_columns].astype('category').cat.codes

    data_X = np.concatenate([train_X, test_X], axis=0)
    data_Y = np.concatenate([train_Y, test_Y], axis=0)
    data_Y = pd.concat([pd.DataFrame(np.arange(len(data_Y))), pd.DataFrame(data_Y)], axis=1)
    data_Y.columns = ['idx', 'targets']
    data_Y.to_csv(os.path.join('data', 'LSST', 'Y.csv'), index=False)

    data_mat = pd.DataFrame()

    for idx, P_mat in enumerate(data_X):
        idx_mat = np.zeros(P_mat.shape[0]) + idx
        time_mat = np.arange(0.1, P_mat.shape[0] / 10 + 0.1, 0.1)
        mask_mat = np.ones([P_mat.shape[0], P_mat.shape[1]])

        # random delete samples for each variable at a given missing rate
        for col in range(P_mat.shape[1]):
            missing_idx = np.random.choice(np.arange(P_mat.shape[0]), int(missing_rate * P_mat.shape[0]), replace=False)
            P_mat[missing_idx, col] = 0
            mask_mat[missing_idx, col] = 0

        # find the values which is NAN and mask it as 0
        nan_idx = np.isnan(P_mat)
        mask_mat[nan_idx] = 0
        P_mat[nan_idx] = 0

        delete_index = []
        for row in range(P_mat.shape[0]):
            if (mask_mat[row, :] == 0).all(0):
                delete_index.append(row)

        P_mat = np.delete(P_mat, delete_index, axis=0)
        idx_mat = np.expand_dims(np.delete(idx_mat, delete_index), axis=1)
        time_mat = np.expand_dims(np.delete(time_mat, delete_index), axis=1)
        mask_mat = np.delete(mask_mat, delete_index, axis=0)

        sample_mat = np.concatenate([idx_mat, time_mat, P_mat, mask_mat], axis=-1)
        data_mat = pd.concat([data_mat, pd.DataFrame(sample_mat)], axis=0)
    data_mat.columns = (['idx', 'time'] +
                        ['ts_' + str(i) for i in range(data_X.shape[2])] +
                        ['mask_' + str(i + 1) for i in range(data_X.shape[2])])

    data_mat.to_csv(os.path.join('data', 'LSST', 'X_'+str(missing_rate)+'.csv'), index=False)


if __name__ == '__main__':
    np.random.seed(0)
    data_root_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'data', 'LSST')
    if not os.path.isdir(data_root_dir):
        os.makedirs(data_root_dir)

    print("Downloading...")
    this_url = 'http://www.timeseriesclassification.com/Downloads/LSST.zip'
    download(this_url, data_root_dir)

    print("Unzipping...")
    unzip('LSST', data_root_dir)

    missing_rate_list = [0.25, 0.5, 0.75]
    for missing_rate in missing_rate_list:
        print('Preprocessing...missing_rate: ', missing_rate)
        Preprocessing('LSST', data_root_dir, missing_rate)


