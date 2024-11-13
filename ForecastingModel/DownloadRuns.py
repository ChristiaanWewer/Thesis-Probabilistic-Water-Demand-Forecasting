import wandb
import os
import torch
import pandas as pd
from ForecastingModel.Forecaster import Forecaster
import json

def get_best_run_info_from_last_sweep(project, results_folder, download=False):
    """
    Function to get the best run from the last sweep of a project.
    The function will download the files of the best run and save them in the results folder.
    The function will return the configuration and the summary of the best run.

    Input Arguments:
    project: Name of the project on wandb

    results_folder: Folder where the results are saved

    download: Boolean to download the files of the best run

    """

    results_project_folder = os.path.join(results_folder, project)
    if not os.path.exists(results_project_folder):
        os.makedirs(results_project_folder)

    api = wandb.Api()
    s = api.project(project).sweeps()
    print('Sweeps of Project:', s)
    best_run = s[0].best_run()
    print('Best Run ID', best_run.id)
    if download:
        for f in best_run.files():
            f.download(replace=True, root=results_project_folder)
    info = {**best_run.config, **best_run.summary}
    return info

def get_best_run_info_from_ith_sweep(project, results_folder, i, download=False):
    """
    Function to get the best run from the ith sweep of a project.
    The function will download the files of the best run and save them in the results folder.
    The function will return the configuration and the summary of the best run.

    Input Arguments:
    project: Name of the project on wandb

    results_folder: Folder where the results are saved

    i: Index of the sweep that is used to get the best run

    download: Boolean to download the files of the best run

    """

    results_project_folder = os.path.join(results_folder, project)
    if not os.path.exists(results_project_folder):
        os.makedirs(results_project_folder)
    api = wandb.Api()
    s = api.project(project).sweeps()
    print('Sweeps of Project:', s)
    best_run = s[i].best_run()
    print('Best Run ID', best_run.id)
    if download:
        for f in best_run.files():
            f.download(replace=True, root=results_project_folder)
    info = {**best_run.config, **best_run.summary}
    return info


def get_best_run_info_from_each_sweep(project, results_folder, download, DMAs=None):
    """
    Function to get the best run from each sweep of a project.
    The function will download the files of the best run and save them in the results folder.
    The function will return the configuration and the summary of the best run.

    Input Arguments:
    project: Name of the project on wandb

    results_folder: Folder where the results are saved

    download: Boolean to download the files of the best run

    DMAs: List of DMAs that are used to train the model

    """

    results_project_folder = os.path.join(results_folder, project)
    if not os.path.exists(results_project_folder):
        os.makedirs(results_project_folder)
    api = wandb.Api()
    s = api.project(project).sweeps()
    best_runs = []
    for i, sweep in enumerate(s):
        if DMAs is None:
            dma = sweep.name[-1:]
        else:
            dma = DMAs[i]
        best_run = sweep.best_run()
        if download:
            dl_path = os.path.join(results_project_folder, dma)
            if not os.path.exists(dl_path):
                os.makedirs(dl_path)
            for f in best_run.files():
                f.download(replace=True, root=dl_path)
        info = {**best_run.config, **best_run.summary}
        best_runs.append(info)
    return best_runs


def get_results_pt_model_per_dma(project, 
                                 model, 
                                 DMAs=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], 
                                 device='cpu',
                                 data_folder='../data/sequences/',
                                 dataset='24h_out_all_no_weather',
                                 location_scaler='../data/processed/scaler_all.pt'
                                 ):
    """
    Function to get the results of the models that are trained per DMA.
    The function will load the model, the configuration, the validation and test data and the scalers.
    The function will predict the validation and test data and log the results.
    The results are saved in a dictionary and returned.

    Input Arguments:
    model: Class of the model that is used to make the predictions

    DMAs: List of DMAs that are used to train the model

    device: String of device that is used to make the predictions 'cpu' or 'cuda'

    data_folder: Folder where the data is stored

    dataset: Name of the dataset that is used to train the model

    location_scaler: Location of the scaler that is used to scale the data

    """
    
    scaler = torch.load(location_scaler, map_location=device)

    df_results_test = pd.DataFrame()
    df_results_val = pd.DataFrame()
    df_hyperparameters = pd.DataFrame()
    
    df_result_val_2d_RMAE= pd.DataFrame() 
    df_result_val_2d_mae = pd.DataFrame()
    df_result_val_2d_MAPE = pd.DataFrame()
    df_result_val_2d_mae_naive = pd.DataFrame()
    df_result_val_2d_MAPE_naive = pd.DataFrame()

    df_result_test_2d_RMAE= pd.DataFrame()
    df_result_test_2d_mae = pd.DataFrame()
    df_result_test_2d_MAPE = pd.DataFrame()
    df_result_test_2d_mae_naive = pd.DataFrame()
    df_result_test_2d_MAPE_naive = pd.DataFrame()

    preds_test = {}
    truth_test = {}
    preds_val = {}
    truth_val = {}

    index_results = ['MAE',         'RMAE',         'MAPE', 
                     'Lowest AE',  'Lowest APE',  'Lowest RAE', 
                     'Highest AE', 'Highest APE', 'Highest RAE', 
                     'Median AE',  'Median APE',  'Median RAE']
    
    # evaluate model on ALL DMAs seperately
    for i, dma in enumerate(DMAs):

        # path to parameters
        state_dict = 'results/'+project+'/'+dma+'/state_dict_model.pt'

        # load config
        config = torch.load('results/'+project+'/'+dma+'/config.pt', map_location='cpu')

        # turn wandb logging false, we have already done any training
        config['wandb_logging'] = False

        # load validation data
        x_val = torch.load(os.path.join(data_folder,dataset, 'val_x_{}_{}.pt'.format(dataset, dma)))
        y_val = torch.load(os.path.join(data_folder, dataset,'val_y_{}_{}.pt'.format(dataset, dma)))
        naive_val = torch.load(os.path.join(data_folder, dataset,'val_naive_final_{}_{}.pt'.format(dataset, dma)))

        # load scalers
        scale1 = scaler.scale_[i]
        scale2 = scaler.mean_[i]

        # make model class
        FC = Forecaster(
            model=model,
            config=config,
            name='test',
            x_val=[x_val],
            y_val=[y_val],
            naive_val=[naive_val],
            save_path=False,
            compile=False,
            seed=42,
            scale1s=[scale1],
            scale2s=[scale2],
        )

        # open json file wandb-summary.json 'results/'+project+'wandb-summary.json', these are the results with logs from training
        with open('results/'+project+'/'+dma+'/wandb-summary.json') as f:
            wandbsummary = json.load(f)

        # add training time to config
        config['train_time'] = wandbsummary['train_time']/60 # minutes
        f.close()

        # load the parameters of the model in the Forecaster class
        FC.load_model(state_dict)

        # compute the exact number of parameters of the model
        nr_params = FC.compute_parameters_model()

        # load test data
        x_test_dma = torch.load(os.path.join(data_folder, dataset, 'test_x_{}_{}.pt'.format(dataset, dma)), map_location=device)
        y_test_dma = torch.load(os.path.join(data_folder, dataset, 'test_y_{}_{}.pt'.format(dataset, dma)), map_location=device)
        naive_test_dma = torch.load(os.path.join(data_folder, dataset, 'test_naive_final_{}_{}.pt'.format(dataset, dma)), map_location=device)

        # Predict model on the validation set
        y_pred_val_dma = FC.predict(x_val, batch_size=256)

        # log the results of the model on the validation set
        FC.log_results_pt_forecast(y_pred_val_dma, y_val, naive_val, dma, scale1, scale2)
        series_dma_val =  pd.Series(data=FC.results[dma]['metrics']).iloc[1:]
        series_dma_val.index = index_results
        df_results_val[dma] = series_dma_val
        df_result_val_2d_RMAE[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_RMAE_2d'])
        df_result_val_2d_mae[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAE_2d'])
        df_result_val_2d_MAPE[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAPE_2d'])
        df_result_val_2d_mae_naive[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAE_naive_2d'])
        df_result_val_2d_MAPE_naive[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAPE_naive_2d'])

        # log the results of the model on the testing set
        y_pred_test_dma = FC.predict(x_test_dma, batch_size=256)
        FC.log_results_pt_forecast(y_pred_test_dma, y_test_dma, naive_test_dma, dma, scale1, scale2)
        series_dma_test =  pd.Series(data=FC.results[dma]['metrics']).iloc[1:]
        series_dma_test.index = index_results
        df_results_test[dma] = series_dma_test
        df_result_test_2d_RMAE[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_RMAE_2d'])
        df_result_test_2d_mae[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAE_2d'])
        df_result_test_2d_MAPE[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAPE_2d'])
        df_result_test_2d_mae_naive[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAE_naive_2d'])
        df_result_test_2d_MAPE_naive[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAPE_naive_2d'])

        # save the configuration of the model
        series_hyperparameters = pd.Series(data=config)

        # make sure to have the output data in the correct scale
        preds_test[dma] = y_pred_test_dma * scale1 + scale2
        truth_test[dma] = y_test_dma * scale1 + scale2
        preds_val[dma] = y_pred_val_dma * scale1 + scale2
        truth_val[dma] = y_val * scale1 + scale2

        # add the number of parameters to the configuration
        series_hyperparameters.loc['Number Of Parameters'] = nr_params

        # add the configuration to the dataframe                                                              
        df_hyperparameters[dma] = series_hyperparameters

    df_results_test['Average'] = df_results_test.mean(axis=1)
    df_results_val['Average'] = df_results_val.mean(axis=1)
    
    df_result_test_2d_RMAE.index         = df_result_test_2d_RMAE.index+1
    df_result_test_2d_mae.index          = df_result_test_2d_mae.index+1
    df_result_test_2d_MAPE.index         = df_result_test_2d_MAPE.index+1
    df_result_test_2d_mae_naive.index    = df_result_test_2d_mae_naive.index+1
    df_result_test_2d_MAPE_naive.index   = df_result_test_2d_MAPE_naive.index+1

    df_result_val_2d_RMAE.index         = df_result_val_2d_RMAE.index+1
    df_result_val_2d_mae.index          = df_result_val_2d_mae.index+1
    df_result_val_2d_MAPE.index         = df_result_val_2d_MAPE.index+1
    df_result_val_2d_mae_naive.index    = df_result_val_2d_mae_naive.index+1
    df_result_val_2d_MAPE_naive.index   = df_result_val_2d_MAPE_naive.index+1
    
    # put all the results in a dictionary together
    dict_results = {
        'val':{
            'y_pred': preds_val,
            'y_truth': truth_val,
            'results': df_results_val,
            'results_2d_RMAE': df_result_val_2d_RMAE,
            'results_2d_mae': df_result_val_2d_mae,
            'results_2d_MAPE': df_result_val_2d_MAPE,
            'results_2d_mae_naive': df_result_val_2d_mae_naive,
            'results_2d_MAPE_naive': df_result_val_2d_MAPE_naive,
        },
        'test':{
            'y_pred': preds_test,
            'y_truth': truth_test,
            'results': df_results_test,
            'results_2d_RMAE': df_result_test_2d_RMAE,
            'results_2d_mae': df_result_test_2d_mae,
            'results_2d_MAPE': df_result_test_2d_MAPE,
            'results_2d_mae_naive': df_result_test_2d_mae_naive,
            'results_2d_MAPE_naive': df_result_test_2d_MAPE_naive,
        },
        'config': df_hyperparameters
    }

    return dict_results
    

def get_results_pt_dma_together(project, 
                                model, 
                                DMAs=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
                                device='cpu',
                                data_folder='../data/sequences/',
                                dataset='24h_out_all_no_weather',
                                location_scaler='../data/processed/scaler_all.pt'
                                ):
    """
    Function to get the results of the models that are trained on all DMAs together.
    The function will load the model, the configuration, the validation and test data and the scalers.
    The function will predict the validation and test data and log the results.
    The results are saved in a dictionary and returned.

    Input Arguments:
    model: Class of the model that is used to make the predictions

    DMAs: List of DMAs that are used to train the model

    device: String of device that is used to make the predictions 'cpu' or 'cuda'

    data_folder: Folder where the data is stored

    dataset: Name of the dataset that is used to train the model

    location_scaler: Location of the scaler that is used to scale the data

    """

    # make the path to the state dict
    state_dict = 'results/'+project+'/state_dict_model.pt'

    # load the config
    config = torch.load('results/'+project+'/config.pt', map_location='cpu')
    config['wandb_logging'] = False

    # load the validatoin data
    name = str(model).split('.')[-1][:-2]
    x_val = [torch.load(os.path.join(data_folder,dataset, 'val_x_{}_{}.pt'.format(dataset, dma)), map_location=device) for dma in DMAs]
    y_val = [torch.load(os.path.join(data_folder, dataset,'val_y_{}_{}.pt'.format(dataset, dma)), map_location=device) for dma in DMAs]
    naive_val = [torch.load(os.path.join(data_folder, dataset,'val_naive_final_{}_{}.pt'.format(dataset, dma)), map_location=device) for dma in DMAs]

    # load the scaler
    scaler = torch.load(location_scaler, map_location=device)

    # index for the results
    index_results = ['MAE',         'RMAE',         'MAPE', 
                     'Lowest AE',  'Lowest APE',  'Lowest RAE', 
                     'Highest AE', 'Highest APE', 'Highest RAE', 
                     'Median AE',  'Median APE',  'Median RAE']

    # make the forecaster class
    FC = Forecaster(
        model=model,
        config=config,
        name=name,
        x_val=x_val,
        y_val=y_val,
        naive_val=naive_val,
        save_path=False,
        compile=False,
        seed=42,
        scale1s=scaler.scale_,
        scale2s=scaler.mean_,
    )

    # open the wandb-summary.json file to get the training time
    with open('results/'+project+'/wandb-summary.json') as f:
        wandbsummary = json.load(f)
        config['train_time'] = wandbsummary['train_time']/60 # minutes
    f.close()

    # load the state dict
    FC.load_model(state_dict)

    # compute the number of parameters
    nr_params = FC.compute_parameters_model()

    # make the results dataframes
    df_results_test = pd.DataFrame()
    df_results_val = pd.DataFrame()
    df_result_val_2d_RMAE= pd.DataFrame() 
    df_result_val_2d_mae = pd.DataFrame()
    df_result_val_2d_MAPE = pd.DataFrame()
    df_result_val_2d_mae_naive = pd.DataFrame()
    df_result_val_2d_MAPE_naive = pd.DataFrame()
    df_result_test_2d_RMAE= pd.DataFrame()
    df_result_test_2d_mae = pd.DataFrame()
    df_result_test_2d_MAPE = pd.DataFrame()
    df_result_test_2d_mae_naive = pd.DataFrame()
    df_result_test_2d_MAPE_naive = pd.DataFrame()
    preds_test = {}
    truth_test = {}
    preds_val = {}
    truth_val = {}

    # evaluate model on ALL DMAs seperately
    for i, dma in enumerate(DMAs):

        # load test data
        x_test_dma = torch.load(os.path.join(data_folder, dataset, 'test_x_{}_{}.pt'.format(dataset, dma)), map_location=device)
        y_test_dma = torch.load(os.path.join(data_folder, dataset, 'test_y_{}_{}.pt'.format(dataset, dma)), map_location=device)
        naive_test_dma = torch.load(os.path.join(data_folder, dataset, 'test_naive_final_{}_{}.pt'.format(dataset, dma)), map_location=device)
        
        # load validationd ata
        x_val = torch.load(os.path.join(data_folder,dataset, 'val_x_{}_{}.pt'.format(dataset, dma)))
        y_val = torch.load(os.path.join(data_folder, dataset,'val_y_{}_{}.pt'.format(dataset, dma)))
        naive_val = torch.load(os.path.join(data_folder, dataset,'val_naive_final_{}_{}.pt'.format(dataset, dma)))

        # predict the validation data and predict the test data
        y_pred_val_dma = FC.predict(x_val, batch_size=256)
        y_pred_test_dma = FC.predict(x_test_dma, batch_size=256)

        # load the scalers
        scale1 = scaler.scale_[i]
        scale2 = scaler.mean_[i]

        # log validation results
        FC.log_results_pt_forecast(y_pred_val_dma, y_val, naive_val, dma, scale1, scale2)
        series_dma_val =  pd.Series(data=FC.results[dma]['metrics']).iloc[1:]
        series_dma_val.index = index_results
        df_results_val[dma] = series_dma_val
        df_result_val_2d_RMAE[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_RMAE_2d'])
        df_result_val_2d_mae[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAE_2d'])
        df_result_val_2d_MAPE[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAPE_2d'])
        df_result_val_2d_mae_naive[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAE_naive_2d'])
        df_result_val_2d_MAPE_naive[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAPE_naive_2d'])

        # log test results
        FC.log_results_pt_forecast(y_pred_test_dma, y_test_dma, naive_test_dma, dma, scale1, scale2)
        series_dma_test =  pd.Series(data=FC.results[dma]['metrics']).iloc[1:]
        series_dma_test.index = index_results
        df_results_test[dma] = series_dma_test
        df_result_test_2d_RMAE[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_RMAE_2d'])
        df_result_test_2d_mae[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAE_2d'])
        df_result_test_2d_MAPE[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAPE_2d'])
        df_result_test_2d_mae_naive[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAE_naive_2d'])
        df_result_test_2d_MAPE_naive[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAPE_naive_2d'])
        
        preds_test[dma] = y_pred_test_dma * scale1 + scale2
        truth_test[dma] = y_test_dma * scale1 + scale2
        preds_val[dma] = y_pred_val_dma * scale1 + scale2
        truth_val[dma] = y_val * scale1 + scale2

    df_results_test['Average'] = df_results_test.mean(axis=1)
    df_results_val['Average'] = df_results_val.mean(axis=1)

    df_result_test_2d_RMAE.index         = df_result_test_2d_RMAE.index+1
    df_result_test_2d_mae.index          = df_result_test_2d_mae.index+1
    df_result_test_2d_MAPE.index         = df_result_test_2d_MAPE.index+1
    df_result_test_2d_mae_naive.index    = df_result_test_2d_mae_naive.index+1
    df_result_test_2d_MAPE_naive.index   = df_result_test_2d_MAPE_naive.index+1
    df_result_val_2d_RMAE.index         = df_result_val_2d_RMAE.index+1
    df_result_val_2d_mae.index          = df_result_val_2d_mae.index+1
    df_result_val_2d_MAPE.index         = df_result_val_2d_MAPE.index+1
    df_result_val_2d_mae_naive.index    = df_result_val_2d_mae_naive.index+1
    df_result_val_2d_MAPE_naive.index   = df_result_val_2d_MAPE_naive.index+1

    # add the number of parameters to the configuration
    config['nr_params'] = nr_params

    dict_results = {
        'val':{
            'y_pred': preds_val,
            'y_true': truth_val,
            'results': df_results_val,
            'results_2d_RMAE': df_result_val_2d_RMAE,
            'results_2d_mae': df_result_val_2d_mae,
            'results_2d_MAPE': df_result_val_2d_MAPE,
            'results_2d_mae_naive': df_result_val_2d_mae_naive,
            'results_2d_MAPE_naive': df_result_val_2d_MAPE_naive,
        },
        'test':{
            'y_pred': preds_test,
            'y_true': truth_test,
            'results': df_results_test,
            'results_2d_RMAE': df_result_test_2d_RMAE,
            'results_2d_mae': df_result_test_2d_mae,
            'results_2d_MAPE': df_result_test_2d_MAPE,
            'results_2d_mae_naive': df_result_test_2d_mae_naive,
            'results_2d_MAPE_naive': df_result_test_2d_MAPE_naive,
        },
        'config': config
    }

    return dict_results


def get_results_prob(
                    project, 
                    model, 
                    DMAs=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
                    device='cpu',
                    data_folder='../data/sequences/',
                    dataset='24h_out_all_no_weather',
                    location_scaler='../data/processed/scaler_all.pt'
                    ):
    """
    Function to get the results of the models that are trained on all DMAs together.
    The function will load the model, the configuration, the validation and test data and the scalers.
    The function will predict the validation and test data and log the results.
    The results are saved in a dictionary and returned.

    Input Arguments:
    model: Class of the model that is used to make the predictions

    DMAs: List of DMAs that are used to train the model

    device: String of device that is used to make the predictions 'cpu' or 'cuda'

    data_folder: Folder where the data is stored

    dataset: Name of the dataset that is used to train the model

    location_scaler: Location of the scaler that is used to scale the data

    """
     
    state_dict = 'results/'+project+'/state_dict_model.pt'
    config = torch.load('results/'+project+'/config.pt', map_location='cpu')
    config['forecast_sequence_length'] = 24
    config['wandb_logging'] = False

    # val data
    x_val = [torch.load(os.path.join(data_folder,dataset, 'val_x_{}_{}.pt'.format(dataset, dma)), map_location=device) for dma in DMAs]
    y_val = [torch.load(os.path.join(data_folder, dataset,'val_y_{}_{}.pt'.format(dataset, dma)), map_location=device) for dma in DMAs]
    naive_val = [torch.load(os.path.join(data_folder, dataset,'val_naive_{}_{}.pt'.format(dataset, dma)), map_location=device) for dma in DMAs]
    scaler = torch.load(location_scaler, map_location='cpu')

    # make the forecaster class
    FC = Forecaster(
        model=model,
        config=config,
        name=project,
        x_val=x_val,
        y_val=y_val,
        naive_val=naive_val,
        save_path=False,
        compile=False,
        seed=42,
        scale1s=scaler.scale_,
        scale2s=scaler.mean_
    )

    # open the wandb-summary.json file to get the training time
    with open('results/'+project+'/wandb-summary.json') as f:
        wandbsummary = json.load(f)
        config['train_time'] = wandbsummary['train_time']/60 
    f.close()

    # load the state dict
    FC.load_model(state_dict)

    # make the results dataframes
    nr_params = FC.compute_parameters_model()

    # make the results dataframes
    df_results_test = pd.DataFrame()
    df_results_val = pd.DataFrame()
    df_result_val_2d_RMAE= pd.DataFrame() 
    df_result_val_2d_mae = pd.DataFrame()
    df_result_val_2d_MAPE = pd.DataFrame()
    df_result_val_2d_mae_naive = pd.DataFrame()
    df_result_val_2d_MAPE_naive = pd.DataFrame()
    df_result_test_2d_RMAE= pd.DataFrame()
    df_result_test_2d_mae = pd.DataFrame()
    df_result_test_2d_MAPE = pd.DataFrame()
    df_result_test_2d_mae_naive = pd.DataFrame()
    df_result_test_2d_MAPE_naive = pd.DataFrame()
    df_results_test = pd.DataFrame()
    df_results_val = pd.DataFrame()

    # index results point prediction
    index_results = ['MAE',         'RMAE',         'MAPE', 
                     'Lowest AE',  'Lowest APE',  'Lowest RAE', 
                     'Highest AE', 'Highest APE', 'Highest RAE', 
                     'Median AE',  'Median APE',  'Median RAE']

    # index results probabilistic prediction
    index_results_probabilistic = ['Cov. Gap', 'PICP', 'PINAW', 'Robust Pinaw', 'Width']

    df_results_interval_test = pd.DataFrame()
    df_results_interval_val = pd.DataFrame()
    df_results_test_2d_picp = pd.DataFrame()
    df_results_test_2d_pinaw = pd.DataFrame()
    df_results_test_2d_cov_gap = pd.DataFrame()
    df_results_test_2d_robust_pinaw = pd.DataFrame()
    df_results_val_2d_picp = pd.DataFrame()
    df_results_val_2d_pinaw = pd.DataFrame()
    df_results_val_2d_cov_gap = pd.DataFrame()
    df_results_val_2d_robust_pinaw = pd.DataFrame()
    y_preds_test = []
    y_preds_val = []
    scale1s = []
    scale2s = []
    y_tests = []
    y_vals = []


    for i, dma in enumerate(DMAs):

        # load test data
        test_data_dma_x = torch.load(f'../data/sequences/24h_out_all_no_weather/test_x_24h_out_all_no_weather_{dma}.pt', map_location='cpu')
        test_data_dma_y = torch.load(f'../data/sequences/24h_out_all_no_weather/test_y_24h_out_all_no_weather_{dma}.pt', map_location='cpu')
        naive_test = torch.load(os.path.join(data_folder, dataset,'test_naive_final_{}_{}.pt'.format(dataset, dma)))

        # load val data        
        val_data_dma_x = torch.load(f'../data/sequences/24h_out_all_no_weather/val_x_24h_out_all_no_weather_{dma}.pt', map_location='cpu')
        val_data_dma_y = torch.load(f'../data/sequences/24h_out_all_no_weather/val_y_24h_out_all_no_weather_{dma}.pt', map_location='cpu')
        naive_val = torch.load(os.path.join(data_folder, dataset,'val_naive_final_{}_{}.pt'.format(dataset, dma)))

        # get predictions of probabilistic model
        if config['probabilistic_method'] == 'quantile_regression':
            pred_val = FC.predict(val_data_dma_x)
            pred_test = FC.predict(test_data_dma_x)
        elif config['probabilistic_method'] == 'mixture_density_network':
            pred_val = FC.predict_mdn(val_data_dma_x, quantile=[0.025, 0.5, 0.975], num_samples=1000)
            pred_test = FC.predict_mdn(test_data_dma_x,  quantile=[0.025, 0.5, 0.975], num_samples=1000)
        
        # get scalers
        scale1 = scaler.scale_[i]
        scale2 = scaler.mean_[i]

        # save results
        y_preds_test.append(pred_test)
        y_preds_val.append(pred_val)
        scale1s.append(scale1)
        scale2s.append(scale2)
        y_tests.append(test_data_dma_y)
        y_vals.append(val_data_dma_y)
        pred_val_pt = pred_val[:,:,1]
        pred_test_pt = pred_test[:,:,1]

        # log point forecasts forecasts testing
        FC.log_results_pt_forecast(pred_test_pt, test_data_dma_y, naive_test, dma, scale1, scale2)
        series_test_dma =  pd.Series(data=FC.results[dma]['metrics']).iloc[1:]
        series_test_dma.index = index_results
        df_results_test[dma] = series_test_dma
        df_result_test_2d_RMAE[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_RMAE_2d'])
        df_result_test_2d_mae[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAE_2d'])
        df_result_test_2d_MAPE[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAPE_2d'])
        df_result_test_2d_mae_naive[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAE_naive_2d'])
        df_result_test_2d_MAPE_naive[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAPE_naive_2d'])

        # log point forecasts forecasts validation
        FC.log_results_pt_forecast(pred_val_pt, val_data_dma_y, naive_val, dma, scale1, scale2)
        series_val_dma =  pd.Series(data=FC.results[dma]['metrics']).iloc[1:]
        series_val_dma.index = index_results
        df_results_val[dma] = series_val_dma
        df_result_val_2d_RMAE[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_RMAE_2d'])
        df_result_val_2d_mae[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAE_2d'])
        df_result_val_2d_MAPE[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAPE_2d'])
        df_result_val_2d_mae_naive[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAE_naive_2d'])
        df_result_val_2d_MAPE_naive[dma] = pd.Series(data=FC.results[dma]['metrics_2d'][f'{dma}_MAPE_naive_2d'])
        
        # log interval forecast testing
        FC.log_results_interval_forecast(q_preds=pred_test[:,:,[0,-1]], y_test=test_data_dma_y, name=dma, scale1=scale1, scale2=scale2)
        series_dma_test = pd.Series(FC.results_probabilistic[dma]['metrics'])
        series_dma_test.index = index_results_probabilistic
        df_results_interval_test[dma] = series_dma_test
        df_results_interval_val[dma] = series_dma_test
        df_results_test_2d_picp[dma] = pd.Series(data=FC.results_probabilistic[dma]['metrics_2d'][f'{dma}_picp_2d'])
        df_results_test_2d_pinaw[dma] = pd.Series(data=FC.results_probabilistic[dma]['metrics_2d'][f'{dma}_pinaw_2d'])
        df_results_test_2d_cov_gap[dma] = pd.Series(data=FC.results_probabilistic[dma]['metrics_2d'][f'{dma}_cov_gap_2d'])
        df_results_test_2d_robust_pinaw[dma] = pd.Series(data=FC.results_probabilistic[dma]['metrics_2d'][f'{dma}_robust_pinaw_2d'])

        # log interval forecast validation
        FC.log_results_interval_forecast(q_preds=pred_val[:,:,[0,-1]], y_test=val_data_dma_y, name=dma, scale1=scale1, scale2=scale2)
        series_dma_val = pd.Series(FC.results_probabilistic[dma]['metrics'])
        series_dma_val.index = index_results_probabilistic
        df_results_interval_val[dma] = series_dma_val
        df_results_interval_val[dma] = series_dma_val
        df_results_val_2d_picp[dma] = pd.Series(data=FC.results_probabilistic[dma]['metrics_2d'][f'{dma}_picp_2d'])
        df_results_val_2d_pinaw[dma] = pd.Series(data=FC.results_probabilistic[dma]['metrics_2d'][f'{dma}_pinaw_2d'])
        df_results_val_2d_cov_gap[dma] = pd.Series(data=FC.results_probabilistic[dma]['metrics_2d'][f'{dma}_cov_gap_2d'])
        df_results_val_2d_robust_pinaw[dma] = pd.Series(data=FC.results_probabilistic[dma]['metrics_2d'][f'{dma}_robust_pinaw_2d'])
    
    # get the average results
    df_results_test['Average'] = df_results_test.mean(axis=1)
    df_results_val['Average'] = df_results_val.mean(axis=1)

    # get resutls
    df_result_test_2d_RMAE.index = df_result_test_2d_RMAE.index + 1
    df_result_test_2d_mae.index = df_result_test_2d_mae.index + 1
    df_result_test_2d_MAPE.index = df_result_test_2d_MAPE.index + 1
    df_result_test_2d_mae_naive.index = df_result_test_2d_mae_naive.index + 1
    df_result_test_2d_MAPE_naive.index = df_result_test_2d_MAPE_naive.index + 1
    df_results_test_2d_picp.index = df_results_test_2d_picp.index + 1
    df_results_test_2d_pinaw.index = df_results_test_2d_pinaw.index + 1
    df_results_test_2d_cov_gap.index = df_results_test_2d_cov_gap.index + 1
    df_results_test_2d_robust_pinaw.index = df_results_test_2d_robust_pinaw.index + 1
    df_result_val_2d_RMAE.index = df_result_val_2d_RMAE.index + 1
    df_result_val_2d_mae.index = df_result_val_2d_mae.index + 1
    df_result_val_2d_MAPE.index = df_result_val_2d_MAPE.index + 1
    df_result_val_2d_mae_naive.index = df_result_val_2d_mae_naive.index + 1
    df_result_val_2d_MAPE_naive.index = df_result_val_2d_MAPE_naive.index + 1
    df_results_val_2d_picp.index = df_results_val_2d_picp.index + 1
    df_results_val_2d_pinaw.index = df_results_val_2d_pinaw.index + 1
    df_results_val_2d_cov_gap.index = df_results_val_2d_cov_gap.index + 1
    df_results_val_2d_robust_pinaw.index = df_results_val_2d_robust_pinaw.index + 1

    # reuslts dict
    results_dict = {
        'test':{
            'q_preds':              y_preds_test,
            'scale1':               scale1s,
            'scale2':               scale2s,
            'y_test':               y_tests,
            '1d_metrics':           df_results_test,
            '1d_metrics_interval':  df_results_interval_test,
            '2d_RMAE':              df_result_test_2d_RMAE,
            '2d_mae':               df_result_test_2d_mae,
            '2d_MAPE':              df_result_test_2d_MAPE,
            '2d_mae_naive':         df_result_test_2d_mae_naive,
            '2d_MAPE_naive':        df_result_test_2d_MAPE_naive,
            '2d_picp':              df_results_test_2d_picp,
            '2d_pinaw':             df_results_test_2d_pinaw,
            '2d_cov_gap':           df_results_test_2d_cov_gap,
            '2d_robust_pinaw':      df_results_test_2d_robust_pinaw
        },
        'val':{
            'q_preds':              y_preds_val,
            'scale1':               scale1s,
            'scale2':               scale2s,
            'y_val':                y_vals,
            '1d_metrics':           df_results_val,
            '1d_metrics_interval':  df_results_interval_val,
            '2d_RMAE':              df_result_val_2d_RMAE,
            '2d_mae':               df_result_val_2d_mae,
            '2d_MAPE':              df_result_val_2d_MAPE,
            '2d_mae_naive':         df_result_val_2d_mae_naive,
            '2d_MAPE_naive':        df_result_val_2d_MAPE_naive,
            '2d_picp':              df_results_val_2d_picp,
            '2d_pinaw':             df_results_val_2d_pinaw,
            '2d_cov_gap':           df_results_val_2d_cov_gap,
            '2d_robust_pinaw':      df_results_val_2d_robust_pinaw
        }
    }
        
    config['nr_params'] = nr_params
    results_dict['config'] = config

    return results_dict

