import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from preprocess.Dataset_user import user_get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data_train(name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            data_train = data[0:int(len(data) * 0.7)]
            data_valid = data[int(len(data) * 0.7):]
            num_types = 0
            max = 0
            for item in data:
                num_types = len(item) + num_types
                if item[0]['cascade_id'] > max:
                    max = item[0]['cascade_id']
            return data_train, data_valid, num_types, max


    print('[Info] Loading train data...')
    train_data, data_valid, num_types, max_cascade_id = load_data_train(opt.data)


    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(data_valid, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types, max_cascade_id


def train_epoch(model, training_data, optimizer, pred_loss_func, pred_loss_func2, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    total_mape_number = []
    total_msle_number = []
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type, cascade_id, time_factor, num_participants = map(lambda x: x.to(opt.device), batch)
        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time, time_factor, cascade_id)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        mse, mape, msle = Utils.nums_loss(prediction[1], num_participants, pred_loss_func2)

        # time prediction
        se = Utils.time_loss(prediction[0], event_time)
        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss = event_loss + se / scale_time_loss + mse
        # loss = event_loss + pred_loss + se / scale_time_loss
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
        total_mape_number.append(mape.item())
        total_msle_number.append(msle.item())
        print('{},{},{}'.format(mse, mape, msle))

    rmse = np.sqrt(total_time_se / total_num_pred)
    mape = np.mean(np.array(total_mape_number))
    msle = np.mean(np.array(total_msle_number))
    return total_event_ll / total_num_event, rmse, mape, msle

def eval_epoch(model, validation_data, pred_loss_func, mse_loss_func, opt):
    """ Epoch operation in evaluation phase. """
    model.eval()
    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions

    total_mape_number = []
    total_msle_number = []

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type, cascade_id, time_factor, num_participants = map(lambda x: x.to(opt.device), batch)
            """ forward """
            enc_out, prediction = model(event_type, event_time, time_factor, cascade_id)
            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            se = Utils.time_loss(prediction[0], event_time)
            mse, mape, msle = Utils.nums_loss(prediction[1], num_participants, mse_loss_func)

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
            total_mape_number.append(mape.item())
            total_msle_number.append(msle.item())
            print('{},{},{}'.format(mse, mape, msle))
    rmse = np.sqrt(total_time_se / total_num_pred)
    mape = np.mean(np.array(total_mape_number))
    msle = np.mean(np.array(total_msle_number))
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, mape, msle


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, mse_loss_func, opt):
    """ Start training. """

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')
        start = time.time()
        train_event, rmse, mape, msle = train_epoch(model, training_data, optimizer, pred_loss_func, mse_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'msle: {msle: 8.5f}, MAPE:{mape:8.5f}'
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, msle=msle, mape=mape, elapse=(time.time() - start) / 60))
        start = time.time()
        valid_event, valid_time, rmse, mape, msle = eval_epoch(model, validation_data, pred_loss_func, mse_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'msle: {msle: 8.5f}, MAPE:{mape:8.5f}'
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, msle=msle, mape=mape, elapse=(time.time() - start) / 60))
        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {mape: 8.5f}, {msle: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, mape=mape, msle=msle))
        scheduler.step()


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='preprocess/AERpkl_file.pkl')
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.2)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-smooth', type=float, default=0)
    parser.add_argument('-log', type=str, default='log.txt')

    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cuda')

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, testloader, num_types, max_cascade_id = prepare_dataloader(opt)
    # user_trainloader, user_testloader, user_num_types, user_max_cascade_id = user_get_dataloader(opt)

    """ prepare model """
    model = Transformer(
        num_types=num_types,
        num_cascade_id=max_cascade_id,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    mse_loss_func = nn.MSELoss()

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, mse_loss_func, opt)


if __name__ == '__main__':
    main()
