import os
import time
import numpy as np
import pandas as pd
import torch
import utils


def Trainer(model, dl_train, dl_val, dl_test, args, device, exp_id):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True, min_lr=0.0001)
    min_loss = np.inf

    save_path = os.path.join(args.save_dirs, args.dataset, 'exp_' + str(exp_id))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()

    for epoch in range(1, args.epoch_max + 1):
        '''Model training'''
        model.train()
        epoch_start_time = time.time()


        training_mse_loss, training_mae_loss, training_mape_loss = [], [], []

        for i, batch_ts in enumerate(dl_train):
            sample_idx = batch_ts["sample_idx"]
            batch_idx = batch_ts["batch_idx"]
            obs_times = batch_ts["obs_times"]
            event_pt = batch_ts["event_pt"]
            X = batch_ts["X"].to(device)
            M = batch_ts["M"].to(device)

            loss_mse, loss_mae, loss_mape = model(obs_times, event_pt, sample_idx, X, M, batch_idx, dt=args.dt)
            loss = loss_mse

            training_mse_loss.append(loss_mse.detach().cpu().numpy())
            training_mae_loss.append(loss_mae.detach().cpu().numpy())
            training_mape_loss.append(loss_mape.detach().cpu().numpy())

            print('\rEpoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, time elapsed: {:.2f}, '
                  .format(epoch, args.epoch_max, i + 1, len(dl_train), np.mean(training_mse_loss),
                          time.time() - epoch_start_time), end='')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        time_elapsed = time.time() - epoch_start_time

        if args.log:
            df_log_val.loc[epoch, 'Epoch'] = epoch
            df_log_val.loc[epoch, 'Time elapsed'] = time_elapsed
            df_log_val.loc[epoch, 'Train_Loss'] = np.mean(training_mse_loss)


        '''Modeling val'''

        with torch.no_grad():
            model.eval()
            val_mse_loss, val_mae_loss, val_mape_loss = [], [], []

            for i, batch_ts_val in enumerate(dl_val):
                sample_idx = batch_ts_val["sample_idx"]
                batch_idx = batch_ts_val["batch_idx"]
                obs_times = batch_ts_val["obs_times"]
                event_pt = batch_ts_val["event_pt"]
                X = batch_ts_val["X"].to(device)
                M = batch_ts_val["M"].to(device)

                val_loss_mse, val_loss_mae, val_loss_mape = model(obs_times, event_pt, sample_idx, X, M, batch_idx, dt=args.dt)
                val_mse_loss.append(val_loss_mse.detach().cpu().numpy())
                val_mae_loss.append(val_loss_mae.detach().cpu().numpy())
                val_mape_loss.append(val_loss_mape.detach().cpu().numpy())

            val_mse_loss = np.mean(val_mse_loss)
            val_mae_loss = np.mean(val_mae_loss)
            val_mape_loss = np.mean(val_mape_loss)

            print('Val, MSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}:'.format(val_mse_loss, val_mae_loss, val_mape_loss))

            # save the best model parameter
            if val_mse_loss < min_loss:
                min_loss = val_mse_loss

                model_save_path = os.path.join(save_path, args.model_name + '.pkl')
                torch.save(model.state_dict(), model_save_path)

            if args.log:
                df_log_val.loc[epoch, 'MSE_loss'] = val_mse_loss
                df_log_val.loc[epoch, 'MAE_loss'] = val_mae_loss
                df_log_val.loc[epoch, 'MAPE_loss'] = val_mape_loss

            scheduler.step(val_mse_loss)

    '''Modeling test'''

    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load(model_save_path))

        test_mse_loss, test_mae_loss, test_mape_loss = [], [], []

        for i, batch_ts_test in enumerate(dl_test):
            sample_idx = batch_ts_test["sample_idx"]
            batch_idx = batch_ts_test["batch_idx"]
            obs_times = batch_ts_test["obs_times"]
            event_pt = batch_ts_test["event_pt"]
            X = batch_ts_test["X"].to(device)
            M = batch_ts_test["M"].to(device)

            # run the model for test
            test_loss_mse, test_loss_mae, test_loss_mape = model(obs_times, event_pt, sample_idx, X, M, batch_idx, dt=args.dt)
            test_mse_loss.append(test_loss_mse.detach().cpu().numpy())
            test_mae_loss.append(test_loss_mae.detach().cpu().numpy())
            test_mape_loss.append(test_loss_mape.detach().cpu().numpy())

        test_mse_loss = np.mean(test_mse_loss)
        test_mae_loss = np.mean(test_mae_loss)
        test_mape_loss = np.mean(test_mape_loss)

        print('Test, MSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}:'.format(test_mse_loss, test_mae_loss, test_mape_loss))

        if args.log:
            df_log_test.loc[epoch, 'MSE_loss'] = test_mse_loss
            df_log_test.loc[epoch, 'MAE_loss'] = test_mae_loss
            df_log_test.loc[epoch, 'MAPE_loss'] = test_mape_loss

    if args.log:
        val_log_save_path_test = os.path.join(save_path, args.model_name + '_val_log.csv')
        test_log_save_path_test = os.path.join(save_path, args.model_name+'_test_log.csv')
        df_log_val.to_csv(val_log_save_path_test)
        df_log_test.to_csv(test_log_save_path_test)

