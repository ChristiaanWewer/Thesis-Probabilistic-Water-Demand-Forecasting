##!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import os
import wandb
import sys
sys.path.append('../')
from ForecastingModel.models.MLP import MLP
from ForecastingModel.Forecaster import Forecaster

# name for wandb
NAME = '00_MLP_VANILLA_FINAL2'

# number of runs
COUNT = 100

# load data
data_folder = '../data/sequences/'
ds = '24h_out_all_no_weather'
data_folder = os.path.join(data_folder, ds)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = '../results'
DMAs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# load data
x_train = torch.load(os.path.join(data_folder, 'train_x_{}_full_sequence.pt'.format(ds)), map_location=device)
y_train = torch.load(os.path.join(data_folder, 'train_y_{}_full_sequence.pt'.format(ds)), map_location=device)
x_val = [torch.load(os.path.join(data_folder, 'val_x_{}_{}.pt'.format(ds, dma)), map_location=device) for dma in DMAs]
y_val = [torch.load(os.path.join(data_folder, 'val_y_{}_{}.pt'.format(ds, dma)), map_location=device) for dma in DMAs]
naive_val = [torch.load(os.path.join(data_folder, 'val_naive_final_{}_{}.pt'.format(ds, dma)), map_location=device) for dma in DMAs]
x_test = torch.load(os.path.join(data_folder, 'test_x_{}_full_sequence.pt'.format(ds)), map_location=device)
y_test = torch.load(os.path.join(data_folder, 'test_y_{}_full_sequence.pt'.format(ds)), map_location=device)
scaler = torch.load('../data/processed/scaler_all.pt')

# %%
configuration = {
    'historic_sequence_length':            {'values': [168]},             # number of lagged input variables                        
    'forecast_sequence_length':            {'values': [24]},              # forecasting horizon                           
    'dropout_rate':                        {'values': [0.1, 0.15, 0.2]},                                                          
    'hidden_size':                         {'values': [64, 128, 256]},    # number of neurons per layer                           
    'hidden_layers':                       {'values': [0,1,2]},                                                                 
    'data_keys':                           {'values': [['historic']]},    # input data key                                                        
    'learning_rate':                       {'min': 1e-5, 'max': 1e-2},
    'num_epochs':                          {'values': [250]}, 
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
    'criterion':                           {'values': ['L1Loss']},
    'probabilistic_method':                {'values': [False]},
    'crit_nr_to_save':                     {'values': [1]}, # criterion to save best model with 1=RMAE, 2=MAPE, 3=MAE
    'crit_nr_to_optimize':                 {'values': [1]}  # criterion for hyperparameter optimization with 1=RMAE, 2=MAPE, 3=MAE
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

# %%
def hyperparameter_optimization(config=None):
    with wandb.init(config=config, settings=wandb.Settings(_service_wait=300), job_type='fs'):
        
        # get config
        config=wandb.config
        config = dict(config)
        
        # create forecaster
        FC = Forecaster(
            model=MLP,
            config=config,
            name=NAME,
            x_val=x_val,
            y_val=y_val,
            naive_val=naive_val,
            save_path=save_path,
            compile=False,
            seed=42,
            scale1s = scaler.scale_,
            scale2s = scaler.mean_
        )

        # fit model
        FC.fit(x_train, y_train)

        # evaluate model on ALL DMAs 
        y_pred = FC.predict(x_test)

        test_loss = nn.L1Loss()(y_pred, y_test).item()
        wandb.log({'test_loss': test_loss, 'train_time': FC.train_time})
        
        # evaluate model on ALL DMAs seperately
        for i, dma in enumerate(DMAs):

            # load scaler
            x_test_dma = torch.load(os.path.join(data_folder, 'test_x_{}_{}.pt'.format(ds, dma)), map_location=device)
            y_test_dma = torch.load(os.path.join(data_folder, 'test_y_{}_{}.pt'.format(ds, dma)), map_location=device)
            y_naive_dma = torch.load(os.path.join(data_folder, 'test_naive_final_{}_{}.pt'.format(ds, dma)), map_location=device)
            y_pred_dma = FC.predict(x_test_dma)
            scale1 = scaler.scale_[i]
            scale2 = scaler.mean_[i]
            FC.log_results_pt_forecast(y_pred_dma, y_test_dma, y_naive_dma, f'{dma}', scale1, scale2)
        
        FC.save_results()

sweep_id = wandb.sweep(sweep_config, project=NAME)
print(f'Sweep ID: {sweep_id}')
wandb.agent(sweep_id, hyperparameter_optimization, count=COUNT)


# %%



