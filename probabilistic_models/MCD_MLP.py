import numpy as np
import torch
import torch.nn as nn
import os
import wandb
import sys
sys.path.append('../')
from ForecastingModel.models.MLP import MLP
from ForecastingModel.Forecaster import Forecaster

# set variables
# name of the project, for wandb
NAME = 'TESTING_MLP_MCD_VAL2'
COUNT = 1 # we only run the experiment once

# these are fitted dropout rates
predefined_dropout_rates = [0.528125,
                            0.2890625,
                            0.359375,
                            0.471875,
                            0.10625000000000001,
                            0.5843750000000001,
                            0.2328125,
                            0.1765625,
                            0.5,
                            0.415625]

# use the above dropout rates or 
use_predefined_dropout_rates = True


# name of the project we apply the monte carlo dropout on
MCD_ON_PROJECT = '00_MLP_VANILLA_FINAL2'

# save path
save_path = './show_results/results'

# load data
data_folder = './data/sequences/'
ds = '24h_out_all_no_weather'
data_folder = os.path.join(data_folder, ds)
device = 'cpu'

# DMAs
DMAs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
os.chdir('../')

# load scaler and config
scaler = torch.load('./data/processed/scaler_all.pt')
cfg = torch.load(r'./show_results/results/' +MCD_ON_PROJECT+ '/config.pt', map_location=device)
cfg['wandb_logging'] = False

# set the number of samples of our monte carlo dropout
configuration = {
    'sample_size': {'values': [1000]},
}

# we do a grid search because its just one variable
sweep_config = {
'name': NAME,
'method': 'grid'
}

# we fill in the metric we want to optimize, but we do not really use it as its defined in the MCD method in Forecasting class
metric = {
    'name': 'cov_gap',
    'goal': 'minimize'
}

sweep_config['metric'] = metric
sweep_config['parameters'] = configuration

FC = Forecaster(
    model=MLP,
    config=cfg,
    x_val=np.zeros((256,24)),
    y_val=np.zeros((256,24)),
    naive_val = np.zeros((256,24)),
    device=device,
    save_path=save_path,
    name=NAME,
    compile=False,
    scale1s=scaler.scale_,
    scale2s=scaler.mean_,
)

# load model
FC.load_model(r'./show_results/results/'+ MCD_ON_PROJECT +'/state_dict_model.pt')

# we loop over the DMAs
for i, dma in enumerate(DMAs):
    NAME_DMA = NAME + '_' + dma
    sweep_config['name'] = NAME_DMA

    # we define the hyperparameter optimization function
    def hyperparameter_optimization(config=None):
        with wandb.init(config=config, settings=wandb.Settings(_service_wait=300), job_type='fs'):
            
            # get the config
            config = wandb.config
            config = dict(config)
            sample_size = config['sample_size']
            scale1 = scaler.scale_[i]
            scale2 = scaler.mean_[i]

            # load data
            x_val_dma = torch.load(os.path.join(data_folder, 'val_x_{}_{}.pt'.format(ds, dma)), map_location=device)
            y_val_dma = torch.load(os.path.join(data_folder, 'val_y_{}_{}.pt'.format(ds, dma)), map_location=device)
            naive_val_dma = torch.load(os.path.join(data_folder, 'val_naive_final_{}_{}.pt'.format(ds, dma)), map_location=device)
            
            # fit the dropout rate using the dichotomic search
            if not use_predefined_dropout_rates:
                FC.fit_dropout_rate(
                    r_low=0.05,
                    r_high=0.95,
                    calib_set=[x_val_dma, y_val_dma],
                    n_samples=sample_size,
                    max_cov_gap=0.005,
                    batch_size=256,
                    log_wandb=True,
                )

            # load test data
            x_test_dma = torch.load(os.path.join(data_folder, 'test_x_{}_{}.pt'.format(ds, dma)), map_location=device)
            y_test_dma = torch.load(os.path.join(data_folder, 'test_y_{}_{}.pt'.format(ds, dma)), map_location=device)
            naive_test_dma = torch.load(os.path.join(data_folder, 'test_naive_final_{}_{}.pt'.format(ds, dma)), map_location=device)

            # apply the monte carlo dropout on test set
            y_preds = FC.predict_mcd(
                x_test_dma, 
                n_samples=sample_size, 
                r='fitted' if not use_predefined_dropout_rates else predefined_dropout_rates[i], 
                batch_size=256, 
                return_samples=True, # we want to return the samples
            )

            # save the samples in case we want to plot them
            # torch.save(y_preds, '/run/media/Christiaan/My Passport/masters_thesis/github/Water Demand Forecasting/probabilistic_hyperparameter_optimization/ensemblepred.pt')


            # get the quantiles
            quantiles = torch.Tensor([0.025,0.5, 0.975]).to(device)
            mcd_preds = torch.quantile(y_preds, quantiles, dim=-1).transpose(0,-1).transpose(0,1)
            y_point = mcd_preds[:,:,1]

            # put name
            NAME_DMA_TEST = NAME_DMA + '_test'

            # log the results
            FC.log_results_pt_forecast(y_point, y_test_dma, naive_test_dma, NAME_DMA_TEST, scale1=scale1, scale2=scale2)
            FC.log_results_interval_forecast(mcd_preds, y_test_dma, name=NAME_DMA_TEST, scale1=scale1, scale2=scale2)
            FC.results_probabilistic[NAME_DMA_TEST]['fitted_dropout_rate'] = cfg['dropout_rate']
            FC.results_probabilistic[NAME_DMA_TEST]['dropout_rate'] = FC.fitted_dropout_rate if not use_predefined_dropout_rates else predefined_dropout_rates[i]

            # predict on the validation set
            mcd_preds = FC.predict_mcd(
                x_val_dma, 
                n_samples=sample_size, 
                r='fitted' if not use_predefined_dropout_rates else predefined_dropout_rates[i], 
                batch_size=256,
                quantiles=[0.025,0.5, 0.975], 
                return_samples=False
            )

            y_point = mcd_preds[:,:,1]

            NAME_DMA_VAL = NAME_DMA + '_val'
            FC.log_results_pt_forecast(y_point, y_val_dma, naive_val_dma, NAME_DMA_VAL, scale1=scale1, scale2=scale2)
            FC.log_results_interval_forecast(mcd_preds, y_val_dma, name=NAME_DMA_VAL, scale1=scale1, scale2=scale2)
            FC.results_probabilistic[NAME_DMA_VAL]['fitted_dropout_rate'] = cfg['dropout_rate']
            FC.results_probabilistic[NAME_DMA_VAL]['dropout_rate'] = FC.fitted_dropout_rate if not use_predefined_dropout_rates else predefined_dropout_rates[i]

        FC.save_results(probabilistic=True, point=True)  


    sweep_id = wandb.sweep(sweep_config, project=NAME)
    print(f'Sweep ID: {sweep_id}')
    wandb.agent(sweep_id, hyperparameter_optimization, count=COUNT)
