import os
import time
import numpy as np
import pandas as pd
import torch
import utils

def LSST_training(model, model_name, dl_train, dl_val, dl_test, args, device, exp_id):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True, min_lr=0.0001)
    epoch_max = 100
    max_acc = 0

    criterion = torch.nn.CrossEntropyLoss()
    save_path = os.path.join(args.save_dirs, 'LSST', 'exp_' + str(exp_id))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()

    for epoch in range(1, epoch_max + 1):
        '''Model training'''
        model.train()
        epoch_start_time = time.time()
        num_obs = 0

        training_epoch_loss_CE = []
        training_epoch_loss_GRU = []

        prob_train_list, pred_train_list, targets_train_list = [], [], []
        for i, batch_ts in enumerate(dl_train):
            optimizer.zero_grad()
            sample_idx = batch_ts["sample_idx"]
            obs_times = batch_ts["obs_times"]
            event_pt = batch_ts["event_pt"]
            X = batch_ts["X"].to(device)
            M = batch_ts["M"].to(device)
            batch_idx = batch_ts["batch_idx"]
            batch_targets = batch_ts["targets"].to(device)

            loss_GRU, prob = model(obs_times, event_pt, sample_idx, X, M, batch_idx, device, classify=True, T=3.6,
                                   class_per_time=False, loss_mae=False, target=batch_targets, prop_to_end=True, dt=0.05)

            loss_CE = criterion(prob, batch_targets)

            prediction = torch.argmax(prob, dim=1)
            softmax = torch.nn.Softmax(dim=1)
            prob = softmax(prob)

            prob_train_list.append(prob)
            pred_train_list.append(prediction)
            targets_train_list.append(batch_targets)

            loss = loss_CE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_epoch_loss_GRU.append(loss_GRU.detach().cpu().numpy())
            training_epoch_loss_CE.append(loss_CE.detach().cpu().numpy())

            num_obs = num_obs + len(batch_targets)

            print('\rEpoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, time elapsed: {:.2f}, '
                  .format(epoch, epoch_max, i + 1, len(dl_train), np.sum(training_epoch_loss_CE) / len(training_epoch_loss_CE),
                  time.time() - epoch_start_time), end='')

        prob_train = torch.cat(prob_train_list)
        pred_train = torch.cat(pred_train_list)
        target_train = torch.cat(targets_train_list)

        acc_train = utils.sum_accuracy(pred_train.cpu().numpy(), target_train.cpu().numpy()) / num_obs

        cum_training_loss_CE = np.sum(training_epoch_loss_CE) / len(training_epoch_loss_CE)
        time_elapsed = time.time() - epoch_start_time
        print(f"acc_train={acc_train:.4f}, ", end='')

        if args.log:
            df_log_val.loc[epoch, 'Epoch'] = epoch
            df_log_val.loc[epoch, 'Time elapsed'] = time_elapsed
            df_log_val.loc[epoch, 'Train Loss CE'] = cum_training_loss_CE
            df_log_val.loc[epoch, 'ACC_train'] = acc_train

        '''Modeling val'''
        with torch.no_grad():
            model.eval()
            num_obs = 0
            prob_val_list, pred_val_list, targets_val_list = [], [], []
            for i, batch_ts_val in enumerate(dl_val):
                sample_idx = batch_ts_val["sample_idx"]
                obs_times = batch_ts_val["obs_times"]
                event_pt = batch_ts_val["event_pt"]
                X = batch_ts_val["X"].to(device)
                M = batch_ts_val["M"].to(device)
                batch_idx = batch_ts_val["batch_idx"]
                batch_targets = batch_ts_val["targets"].to(device)

                # run the model for test
                loss_GRU, prob = model(obs_times, event_pt, sample_idx, X, M, batch_idx, device, classify=True, T=3.6,
                                       class_per_time=False, loss_mae=False, target=batch_targets, prop_to_end=True,
                                       return_path=True, dt=0.05)

                prediction = torch.argmax(prob, dim=1)
                softmax = torch.nn.Softmax(dim=1)
                prob = softmax(prob)

                prob_val_list.append(prob)
                pred_val_list.append(prediction)
                targets_val_list.append(batch_targets)

                num_obs = num_obs + len(batch_targets)

            prob_val = torch.cat(prob_val_list)
            pred_val = torch.cat(pred_val_list)
            target_val = torch.cat(targets_val_list)

            acc_val = utils.sum_accuracy(pred_val.cpu().numpy(), target_val.cpu().numpy()) / num_obs

            # save the best model parameter
            if acc_val > max_acc:
                max_acc = acc_val
                if args.model_name in ['CoGRUODE_HV', 'ComGRUODE_HV', 'CoGRUODE_HM', 'ComGRUODE_HM']:
                    model_save_path = os.path.join(save_path, model_name+'_'+str(args.memory) +
                                                   '_miss_'+str(args.missing_rate)+'_LSST.pkl')
                else:
                    model_save_path = os.path.join(save_path, model_name+'_miss_'+str(args.missing_rate)+'_LSST.pkl')
                torch.save(model.state_dict(), model_save_path)

            if args.log:
                df_log_val.loc[epoch, 'ACC_val'] = acc_val

            print(f"Validation: acc_val={acc_val:.4f}")

        scheduler.step(acc_val)

    '''Modeling test'''
    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load(model_save_path))

        num_obs = 0
        prob_test_list, pred_test_list, targets_test_list = [], [], []
        for i, batch_ts_test in enumerate(dl_test):
            sample_idx = batch_ts_test["sample_idx"]
            obs_times = batch_ts_test["obs_times"]
            event_pt = batch_ts_test["event_pt"]
            X = batch_ts_test["X"].to(device)
            M = batch_ts_test["M"].to(device)
            batch_idx = batch_ts_test["batch_idx"]
            batch_targets = batch_ts_test["targets"].to(device)

            # run the model for test
            loss_GRU, prob = model(obs_times, event_pt, sample_idx, X, M, batch_idx, device, classify=True, T=3.6,
                                   class_per_time=False, loss_mae=False, target=batch_targets, prop_to_end=True,
                                   return_path=True, dt=0.05)

            prediction = torch.argmax(prob, dim=1)
            softmax = torch.nn.Softmax(dim=1)
            prob = softmax(prob)

            prob_test_list.append(prob)
            pred_test_list.append(prediction)
            targets_test_list.append(batch_targets)

            num_obs = num_obs + len(batch_targets)

        prob_test = torch.cat(prob_test_list)
        pred_test = torch.cat(pred_test_list)
        target_test = torch.cat(targets_test_list)

        acc_test = utils.sum_accuracy(pred_test.cpu().numpy(), target_test.cpu().numpy()) / num_obs
        print('Test accuracy:', acc_test)

        if args.log:
            df_log_test.loc[0, 'ACC_test'] = acc_test

    if args.log:
        if args.model_name in ['CoGRUODE_HV', 'ComGRUODE_HV', 'CoGRUODE_HM', 'ComGRUODE_HM']:
            val_log_save_path_test = os.path.join(save_path, model_name + '_' + str(args.memory) +'_miss_'+str(args.missing_rate) + '_val_log.csv')
            test_log_save_path_test = os.path.join(save_path, model_name+'_'+str(args.memory) + '_miss_'+str(args.missing_rate) + '_test_log.csv')
        else:
            val_log_save_path_test = os.path.join(save_path, model_name + '_' + str(args.memory) + '_miss_'+str(args.missing_rate) + '_val_log.csv')
            test_log_save_path_test = os.path.join(save_path, model_name+'_'+str(args.memory) + '_miss_'+str(args.missing_rate)+ '_test_log.csv')
        df_log_val.to_csv(val_log_save_path_test)
        df_log_test.to_csv(test_log_save_path_test)

def Activity_training(model, model_name, dl_train, dl_test, args, device, exp_id):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0035, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True, min_lr=0.0001)
    epoch_max = 50
    max_acc = 0
    dt = 0.01
    
    save_path = os.path.join(args.save_dirs, 'Activity', 'exp_' + str(exp_id))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_test = pd.DataFrame()

    print(f'Experiment: {exp_id}')
    for epoch in range(1, epoch_max+1):
        '''Model training'''
        epoch_start_time = time.time()
        model.train()
        training_epoch_loss_CE = []
        num_obs = 0

        for i, batch_ts in enumerate(dl_train):
            sample_idx = batch_ts["sample_idx"]
            obs_times = batch_ts["obs_times"]
            event_pt = batch_ts["event_pt"]
            X = batch_ts["X"].to(device)
            M = batch_ts["M"].to(device)
            batch_idx = batch_ts["batch_idx"]
            batch_targets = batch_ts["targets"].to(device)

            loss_GRU, loss_CE = model(obs_times, event_pt, sample_idx, X, M, batch_idx, device, classify=False, class_per_time=True,
                                      loss_mae=False, target=batch_targets, prop_to_end=False, return_path=False, T=1.0, dt=dt)

            training_epoch_loss_CE.append(loss_CE.detach().cpu().numpy())

            num_obs = num_obs + len(batch_targets)

            loss = loss_CE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('\rEpoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, time elapsed: {:.2f}, '
                  .format(epoch, epoch_max, i + 1, len(dl_train), np.sum(training_epoch_loss_CE) / len(training_epoch_loss_CE),
                  time.time() - epoch_start_time), end='')

        time_elapsed = time.time() - epoch_start_time

        if args.log:
            df_log_test.loc[epoch, 'Epoch'] = epoch
            df_log_test.loc[epoch, 'Time elapsed'] = time_elapsed
            df_log_test.loc[epoch, 'Train Loss CE'] = np.sum(training_epoch_loss_CE) / len(training_epoch_loss_CE)

        '''Modeling test'''
        with torch.no_grad():
            model.eval()
            num_obs = 0

            pred_test_list, targets_test_list = [], []
            for i, batch_ts_test in enumerate(dl_test):
                sample_idx = batch_ts_test["sample_idx"]
                obs_times = batch_ts_test["obs_times"]
                event_pt = batch_ts_test["event_pt"]
                X = batch_ts_test["X"].to(device)
                M = batch_ts_test["M"].to(device)
                batch_idx = batch_ts_test["batch_idx"]
                batch_targets = batch_ts_test["targets"].to(device)

                # run the model for test
                t_vec, p_vec, y_vec = model(obs_times, event_pt, sample_idx, X, M, batch_idx, device,  classify=False, class_per_time=True,
                                   loss_mae=False, target=batch_targets, prop_to_end=False, T=1.0, return_path=True, dt=dt)

                prediction = torch.argmax(p_vec, dim=1)
                pred_test_list.append(prediction)
                targets_test_list.append(batch_targets)
                num_obs = num_obs + len(batch_targets)

            pred_test = torch.cat(pred_test_list)
            target_test = torch.cat(targets_test_list)

            acc_test = utils.sum_accuracy(pred_test.cpu().numpy(), target_test.cpu().numpy()) / num_obs

            # save the best model parameter
            if acc_test > max_acc:
                max_acc = acc_test
                if args.model_name in ['CoGRUODE_HV', 'ComGRUODE_HV', 'CoGRUODE_HM', 'ComGRUODE_HM']:
                    model_save_path = os.path.join(save_path, model_name+'_'+str(args.memory)+'_Act.pkl')
                else:
                    model_save_path = os.path.join(save_path, model_name+'_Act.pkl')
                torch.save(model.state_dict(), model_save_path)

            if args.log:
                df_log_test.loc[epoch, 'ACC_test'] = acc_test

            print(f"Test: acc_test={acc_test:.4f}, num_obs={num_obs}")

        scheduler.step(acc_test)

    if args.log:
        if args.model_name in ['CoGRUODE_HV', 'ComGRUODE_HV', 'CoGRUODE_HM', 'ComGRUODE_HM']:
            log_save_path_test = os.path.join(save_path, model_name+'_'+str(args.memory)+'_test_log.csv')
        else:
            log_save_path_test = os.path.join(save_path, model_name+'_test_log.csv')
        df_log_test.to_csv(log_save_path_test)

def PhysioNet_training(model, model_name, dl_train, dl_test, args, device, exp_id):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True, min_lr=0.0001)
    epoch_max = 200
    max_auc = 0
    dt = 0.1

    criterion = torch.nn.CrossEntropyLoss()
    save_path = os.path.join(args.save_dirs, 'PhysioNet', 'exp_' + str(exp_id))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_test = pd.DataFrame()

    print(f'Experiment: {exp_id}')
    for epoch in range(1, epoch_max + 1):
        '''Model training'''
        epoch_start_time = time.time()
        model.train()
        num_obs = 0
        training_epoch_loss_GRU = []
        training_epoch_loss_CE = []

        prob_train_list, pred_train_list, targets_train_list = [], [], []
        for i, batch_ts in enumerate(dl_train):
            optimizer.zero_grad()
            sample_idx = batch_ts["sample_idx"]
            obs_times = batch_ts["obs_times"]
            event_pt = batch_ts["event_pt"]
            X = batch_ts["X"].to(device)
            M = batch_ts["M"].to(device)
            batch_idx = batch_ts["batch_idx"]
            batch_targets = batch_ts["targets"].to(device)

            loss_GRU, prob = model(obs_times, event_pt, sample_idx, X, M, batch_idx, device, classify=True, T=48.10,
                                   class_per_time=False, loss_mae=False, target=batch_targets, prop_to_end=True, dt=dt)

            loss_CE = criterion(prob, batch_targets)

            prediction = torch.argmax(prob, dim=1)
            softmax = torch.nn.Softmax(dim=1)
            prob = softmax(prob)

            prob_train_list.append(prob)
            pred_train_list.append(prediction)
            targets_train_list.append(batch_targets)

            loss = loss_CE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_epoch_loss_GRU.append(loss_GRU.detach().cpu().numpy())
            training_epoch_loss_CE.append(loss_CE.detach().cpu().numpy())

            num_obs = num_obs + len(batch_targets)

            print('\rEpoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, time elapsed: {:.2f}, '
                  .format(epoch, epoch_max, i + 1, len(dl_train), np.sum(training_epoch_loss_CE) / len(training_epoch_loss_CE),
                  time.time() - epoch_start_time), end='')

        prob_train = torch.cat(prob_train_list)
        pred_train = torch.cat(pred_train_list)
        target_train = torch.cat(targets_train_list)

        acc_train = utils.sum_accuracy(pred_train.cpu().numpy(), target_train.cpu().numpy()) / num_obs
        auc_train, pr_train = utils.auc_pr(target_train.detach().cpu().numpy(), prob_train[:, -1].detach().cpu().numpy())

        cum_training_loss_GRU = np.sum(training_epoch_loss_GRU) / len(training_epoch_loss_GRU)
        cum_training_loss_CE = np.sum(training_epoch_loss_CE) / len(training_epoch_loss_CE)
        time_elapsed = time.time() - epoch_start_time
        print(f"acc_train={acc_train:.4f}, auc_train={auc_train:.4f}, pr_train={pr_train:.4f}")

        if args.log:
            df_log_test.loc[epoch, 'Epoch'] = epoch
            df_log_test.loc[epoch, 'Time elapsed'] = time_elapsed
            df_log_test.loc[epoch, 'Train Loss CE'] = cum_training_loss_CE
            df_log_test.loc[epoch, 'ACC_train'] = acc_train
            df_log_test.loc[epoch, 'AUC_train'] = auc_train
            df_log_test.loc[epoch, 'PR_train'] = pr_train

        '''Modeling test'''
        with torch.no_grad():
            model.eval()
            num_obs = 0

            prob_test_list, pred_test_list, targets_test_list = [], [], []
            for i, batch_ts_test in enumerate(dl_test):
                sample_idx = batch_ts_test["sample_idx"]
                obs_times = batch_ts_test["obs_times"]
                event_pt = batch_ts_test["event_pt"]
                X = batch_ts_test["X"].to(device)
                M = batch_ts_test["M"].to(device)
                batch_idx = batch_ts_test["batch_idx"]
                batch_targets = batch_ts_test["targets"]

                # run the model for test
                loss, prob = model(obs_times, event_pt, sample_idx, X, M, batch_idx, device, T=48.01,
                                   classify=True, target=batch_targets, prop_to_end=True, return_path=True)
                prediction = torch.argmax(prob, dim=1)
                softmax = torch.nn.Softmax(dim=1)
                prob = softmax(prob)

                prob_test_list.append(prob)
                pred_test_list.append(prediction)
                targets_test_list.append(batch_targets)

                num_obs = num_obs + len(batch_targets)

            prob_test = torch.cat(prob_test_list)
            pred_test = torch.cat(pred_test_list)
            target_test = torch.cat(targets_test_list)

            acc_test = utils.sum_accuracy(pred_test.cpu().numpy(), target_test.cpu().numpy()) / num_obs
            auc_test, pr_test = utils.auc_pr(target_test.cpu().numpy(), prob_test[:, -1].cpu().numpy())

            # save the best model parameter
            if auc_test > max_auc:
                max_acc = auc_test
                if args.model_name in ['CoGRUODE_HV', 'ComGRUODE_HV', 'CoGRUODE_HM', 'ComGRUODE_HM']:
                    model_save_path = os.path.join(save_path, model_name+'_'+str(args.memory)+'_Phy.pkl')
                else:
                    model_save_path = os.path.join(save_path, model_name+'_Phy.pkl')
                torch.save(model.state_dict(), model_save_path)

            if args.log:
                df_log_test.loc[epoch, 'ACC_test'] = acc_test
                df_log_test.loc[epoch, 'AUC_test'] = auc_test
                df_log_test.loc[epoch, 'PR_test'] = pr_test

            print(f"Test loss at epoch {epoch}: acc_test={acc_test:.4f}, auc_test={auc_test:.4f}, pr_test={pr_test:.4f},"
                  f"num_obs={num_obs}")

        scheduler.step(auc_test)

    if args.log:
        if args.model_name in ['CoGRUODE_HV', 'ComGRUODE_HV', 'CoGRUODE_HM', 'ComGRUODE_HM']:
            log_save_path_test = os.path.join(save_path, model_name+'_'+str(args.memory)+'_test_log.csv')
        else:
            log_save_path_test = os.path.join(save_path, model_name+'_test_log.csv')
        df_log_test.to_csv(log_save_path_test)
