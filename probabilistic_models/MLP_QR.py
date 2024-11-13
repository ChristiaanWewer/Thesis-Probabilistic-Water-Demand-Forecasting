#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import torch
import torch.nn as nn
import os
import wandb

import sys
sys.path.append('../')

from ForecastingModel.models.MLP import MLP
from ForecastingModel.Forecaster import Forecaster

# name for wandb
NAME = '00_MLP_QR_rpb_FINAL99'

# number of runs
COUNT = 100

# number of epochs
num_epochs = 250

# load data
data_folder = './data/sequences/'
ds = '24h_out_all_no_weather'
data_folder = os.path.join(data_folder, ds)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# save path
save_path = './probabilistic_models/results'

# DMAs
DMAs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

os.chdir('../')

# load data
x_train = torch.load(os.path.join(data_folder, 'train_x_{}_full_sequence.pt'.format(ds)), map_location=device)
y_train = torch.load(os.path.join(data_folder, 'train_y_{}_full_sequence.pt'.format(ds)), map_location=device)
x_val = [torch.load(os.path.join(data_folder, 'val_x_{}_{}.pt'.format(ds, dma)), map_location=device) for dma in DMAs]
y_val = [torch.load(os.path.join(data_folder, 'val_y_{}_{}.pt'.format(ds, dma)), map_location=device) for dma in DMAs]
naive_val = [torch.load(os.path.join(data_folder, 'val_naive_{}_{}.pt'.format(ds, dma)), map_location=device) for dma in DMAs]
x_test = torch.load(os.path.join(data_folder, 'test_x_{}_full_sequence.pt'.format(ds)), map_location=device)
y_test = torch.load(os.path.join(data_folder, 'test_y_{}_full_sequence.pt'.format(ds)), map_location=device)
scaler = torch.load('./data/processed/scaler_all.pt')


# In[4]:
configuration = {
    'historic_sequence_length':            {'values': [168]}, # number of lagged input variables
    'forecast_sequence_length':            {'values': [24]}, # also forecast horizon
    'dropout_rate':                        {'values': [0.1, 0.15, 0.2]},
    'hidden_size':                         {'values': [64, 128, 256]}, # also number of neurons per layer
    'hidden_layers':                       {'values': [0,1,2]},
    'data_keys':                           {'values': [['historic']]}, # data keys we use for this model
    'learning_rate':                       {'min': 1e-5, 'max': 1e-2},
    'num_epochs':                          {'values': [num_epochs]},
    'batch_size':                          {'values': [256]}, 
    'echo_iter':                           {'values': [10]},
    'l1_penalty':                          {'values': [0]},
    'weight_decay':                        {'values': [0]},
    'early_stopping':                      {'values': [False]},
    'patience_1':                          {'values': [0]},
    'delta_1':                             {'values': [0]},
    'patience_2':                          {'values': [0]},
    'clip_grad':                           {'values': [None]},
    'wandb_logging':                       {'values': [True]},
    'criterion':                           {'values': ['QuantileLoss']},
    'probabilistic_method':                {'values': ['quantile_regression']},
    'quantiles':                           {'values': [[0.025, 0.5, 0.975]]},
    'crit_nr_to_save':                     {'values': [1]}, # criterion we save 1=relative_pb', 2=cov gap, 3=robust pinaw
    'crit_nr_to_optimize':                 {'values': [1]}  # criterion we optimize the hyperparameter optimization with 1=relative_pb (relative quantile score in report), 2=cov gap, 3=robust pinaw
}
sweep_config = {
'name': NAME,
'method': 'bayes'
}

metric = {
    'name': 'crit_hpam',
    'goal': 'minimize'
}

sweep_config['metric'] = metric
sweep_config['parameters'] = configuration


# In[5]:
# losses of naive/benchmark model
naive_q_losses = [0.23801901936531067,
                  0.07998813688755035,
                  0.08969905227422714,
                  0.46468520164489746,
                  0.37573495507240295,
                  0.18749231100082397,
                  0.2399633675813675,
                  0.23348844051361084,
                  0.271905779838562,
                  0.2729724049568176]


def hyperparameter_optimization(config=None):
    with wandb.init(config=config, settings=wandb.Settings(_service_wait=300), job_type='fs'):

        config=wandb.config

        config = dict(config)
        
        FC = Forecaster(
            model=MLP,
            config=config,
            name=NAME,
            x_val=x_val,
            y_val=y_val,
            naive_val=[naive_q_losses],
            save_path=save_path,
            compile=False,
            seed=42,
            scale1s=scaler.scale_,
            scale2s=scaler.mean_,
        )
        
        FC.fit(x_train, y_train)

        # evaluate model on ALL DMAs 
        # y_pred = FC.predict(x_test)

        # test_mse_loss = nn.MSELoss()(y_pred, y_test).item()
        wandb.log({'train_time': FC.train_time})
        
        # evaluate model on ALL DMAs seperately
        for i, dma in enumerate(DMAs):

            # load scaler
            x_test_dma = torch.load(os.path.join(data_folder, 'test_x_{}_{}.pt'.format(ds, dma)), map_location=device)
            y_test_dma = torch.load(os.path.join(data_folder, 'test_y_{}_{}.pt'.format(ds, dma)), map_location=device)
            y_pred_dma = FC.predict(x_test_dma, batch_size=256)
            print(y_pred_dma.shape)
            scale1 = scaler.scale_[i]
            scale2 = scaler.mean_[i]
            FC.log_results_interval_forecast(y_pred_dma, y_test_dma, f'{dma}', scale1, scale2)
                    
        FC.save_results()

sweep_id = wandb.sweep(sweep_config, project=NAME)
print(f'Sweep ID: {sweep_id}')
wandb.agent(sweep_id, hyperparameter_optimization, count=COUNT)


# In[ ]:




