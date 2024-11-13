# In[1]
import torch
import os
import sys
sys.path.append('../')
from ForecastingModel.models.MLP import MLP
from ForecastingModel.Forecaster import Forecaster
from ForecastingModel.ConformalPrediction import ConformalPrediction
from ForecastingModel.ScoreFunctions import *

# In[2]

# load data and set variables
# data folder
data_folder = './data/sequences/'

# dataset name
ds = '24h_out_all_no_weather'
data_folder = os.path.join(data_folder, ds)

# set the device
device = 'cpu'

# project name
project_name = 'MLP_CP_FINAL_no_update'

# save path
save_path = './show_results/results'

# DMAs
DMAs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# go back one fodler
os.chdir('../')

# name of the data
name_data = '24h_out_all_no_weather'

# name of the model we apply the conformal prediction on
CP_ON_PROJECT = '00_MLP_QR_rpb_FINAL99'
cfg = torch.load(r'./show_results/results/' +CP_ON_PROJECT+ '/config.pt', map_location=device)
cfg['wandb_logging'] = False
cfg['forecast_sequence_length'] = 24

# In[3]
# In[]
# load validation data
print('load val data')
x_val = [torch.load(os.path.join(data_folder, 'val_x_{}_{}.pt'.format(ds, dma)), map_location=device) for dma in DMAs]
y_val = [torch.load(os.path.join(data_folder, 'val_y_{}_{}.pt'.format(ds, dma)), map_location=device) for dma in DMAs]
naive_val = [torch.load(os.path.join(data_folder, 'val_naive_final_{}_{}.pt'.format(ds, dma)), map_location=device) for dma in DMAs]
naive_test = [torch.load(os.path.join(data_folder, 'test_naive_final_{}_{}.pt'.format(ds, dma)), map_location=device) for dma in DMAs]

# load scaler
scaler = torch.load('./data/processed/scaler_all.pt')

print('data loaded')

# make conformalprediction class
cpfc = ConformalPrediction(
    model=MLP,
    modelclass=Forecaster,
    config=cfg,
    device=device,
    model_name='test_cp',
    scale1s=scaler.scale_,
    scale2s=scaler.mean_,
)

print('Initialized CP class')

dict_results = {}

p = os.path.join(save_path , project_name)

# load the model from the project we want to apply conformal prediction on
cpfc.load_models(
    x_val = x_val,
    y_val = y_val,
    naive_val=naive_val,
    save_path=False,
    names=['test'],
    load_paths=[r'./show_results/results/' +CP_ON_PROJECT+ '/state_dict_model.pt']
    )

print('loaded model')

# iterate over the DMAs and apply conformal prediction
for ii, dma in enumerate(DMAs):

    # get scaler for the DMA
    scale1 = scaler.scale_[ii]
    scale2 = scaler.mean_[ii]

    # load the data
    data_test_dma_x = torch.load(os.path.join('./data/sequences/', name_data, f'test_x_{name_data}_{dma}.pt'))
    data_test_dma_y = torch.load(os.path.join('./data/sequences/', name_data, f'test_y_{name_data}_{dma}.pt'))
    data_test_dma_ind = torch.load(os.path.join('./data/sequences/', name_data, f'test_ind_{name_data}_{dma}.pt'))
    data_val_dma_x = x_val[ii]
    data_val_dma_y = y_val[ii]
    data_val_dma_ind = torch.load(os.path.join(data_folder, 'val_ind_{}_{}.pt'.format(ds, dma)), map_location=device)
    
    # compute nonconformity scores
    cpfc.compute_nonconformity_scores(
        model='all',
        x_calibrate=x_val[ii],
        y_calibrate=y_val[ii],
        )

    print('Computed nonconformity measures')
    CQR_preds_val = cpfc.predict_CQR(
        x_test=data_val_dma_x,
        ind_test=data_val_dma_ind,
        batch_size=256
    )
    CQR_preds = cpfc.predict_CQR(
        x_test=data_test_dma_x,
        ind_test=data_test_dma_ind,
        update_nonconformity_scores=False,
        batch_size=256
    )


    # transform results back to original scale
    CQR_preds_unscaled = CQR_preds * scale1 + scale2
    data_test_dma_y_unscaled = data_test_dma_y * scale1 + scale2
    data_val_dma_y_unscaled = data_val_dma_y * scale1 + scale2
    CQR_preds_unscaled_quantiles_test = CQR_preds_unscaled[:, :, [0, -1]]
    CQR_preds_unscaled_val = CQR_preds_val * scale1 + scale2
    CQR_preds_unscaled_quantiles_val = CQR_preds_unscaled_val[:, :, [0, -1]]
    naive_val[ii] = naive_val[ii] * scale1 + scale2
    naive_test[ii] = naive_test[ii] * scale1 + scale2

    # In[7]

    # get scores
    test_picp_1d = compute_picp_1d(data_test_dma_y_unscaled, CQR_preds_unscaled_quantiles_test)
    test_pinaw_1d = compute_pinaw_1d(data_test_dma_y_unscaled, CQR_preds_unscaled_quantiles_test)
    test_robust_pinaw_1d = compute_robust_pinaw_1d(data_test_dma_y_unscaled, CQR_preds_unscaled_quantiles_test)
    test_width_1d = compute_width_1d(CQR_preds_unscaled_quantiles_test)
    test_cov_gap_1d = compute_cov_gap_1d(data_test_dma_y_unscaled, CQR_preds_unscaled_quantiles_test)
    test_cov_gap_2d = compute_cov_gap_2d(data_test_dma_y_unscaled, CQR_preds_unscaled_quantiles_test)
    test_picp_2d = compute_picp_2d(data_test_dma_y_unscaled, CQR_preds_unscaled_quantiles_test)
    test_pinaw_2d = compute_pinaw_2d(data_test_dma_y_unscaled, CQR_preds_unscaled_quantiles_test)
    test_robust_pinaw_2d = compute_robust_pinaw_2d(data_test_dma_y_unscaled, CQR_preds_unscaled_quantiles_test)
    CQR_preds_unscaled_pt = CQR_preds_unscaled[:,:,1]
    test_mae_1d = compute_MAE_1d(data_test_dma_y_unscaled, CQR_preds_unscaled_pt)
    test_RMAE_1d = compute_RMAE_1d(data_test_dma_y_unscaled, CQR_preds_unscaled_pt, naive_test[ii])
    test_MAPE_1d = compute_MAPE_1d(data_test_dma_y_unscaled, CQR_preds_unscaled_pt)
    test_mae_2d = compute_MAE_2d(data_test_dma_y_unscaled, CQR_preds_unscaled_pt)
    test_RMAE_2d = compute_RMAE_2d(data_test_dma_y_unscaled, CQR_preds_unscaled_pt, naive_test[ii])
    test_MAPE_2d = compute_MAPE_2d(data_test_dma_y_unscaled, CQR_preds_unscaled_pt)
    val_picp_1d = compute_picp_1d(data_val_dma_y_unscaled, CQR_preds_unscaled_quantiles_val)
    val_pinaw_1d = compute_pinaw_1d(data_val_dma_y_unscaled, CQR_preds_unscaled_quantiles_val)
    val_robust_pinaw_1d = compute_robust_pinaw_1d(data_val_dma_y_unscaled, CQR_preds_unscaled_quantiles_val)
    val_width_1d = compute_width_1d(CQR_preds_unscaled_quantiles_val)
    val_cov_gap_1d = compute_cov_gap_1d(data_val_dma_y_unscaled, CQR_preds_unscaled_quantiles_val)
    val_cov_gap_2d = compute_cov_gap_2d(data_val_dma_y_unscaled, CQR_preds_unscaled_quantiles_val)
    val_picp_2d = compute_picp_2d(data_val_dma_y_unscaled, CQR_preds_unscaled_quantiles_val)
    val_pinaw_2d = compute_pinaw_2d(data_val_dma_y_unscaled, CQR_preds_unscaled_quantiles_val)
    val_robust_pinaw_2d = compute_robust_pinaw_2d(data_val_dma_y_unscaled, CQR_preds_unscaled_quantiles_val)
    CQR_preds_unscaled_pt = CQR_preds_unscaled_val[:,:,1]
    val_mae_1d = compute_MAE_1d(data_val_dma_y_unscaled, CQR_preds_unscaled_pt)
    val_RMAE_1d = compute_RMAE_1d(data_val_dma_y_unscaled, CQR_preds_unscaled_pt, naive_val[ii])
    val_MAPE_1d = compute_MAPE_1d(data_val_dma_y_unscaled, CQR_preds_unscaled_pt)
    val_mae_2d = compute_MAE_2d(data_val_dma_y_unscaled, CQR_preds_unscaled_pt)
    val_RMAE_2d = compute_RMAE_2d(data_val_dma_y_unscaled, CQR_preds_unscaled_pt, naive_val[ii])
    val_MAPE_2d = compute_MAPE_2d(data_val_dma_y_unscaled, CQR_preds_unscaled_pt)

    # make dictionary with all the metrics
    dict_result_dma_val = {
        'cov_gap_1d':       val_cov_gap_1d,
        'picp_1d':          val_picp_1d,
        'width_1d':         val_width_1d,
        'pinaw_1d':         val_pinaw_1d,
        'robust_pinaw_1d':  val_robust_pinaw_1d,
        'preds':            CQR_preds_val,
        'true':             data_val_dma_y,
        'index':            data_val_dma_ind,
        'mae_1d':           val_mae_1d,
        'RMAE_1d':          val_RMAE_1d,
        'MAPE_1d':          val_MAPE_1d,
        'mae_2d':           val_mae_2d,
        'RMAE_2d':          val_RMAE_2d,
        'MAPE_2d':          val_MAPE_2d,
        'cov_gap_2d':       val_cov_gap_2d,
        'picp_2d':          val_picp_2d,
        'pinaw_2d':         val_pinaw_2d,
        'robust_pinaw_2d':  val_robust_pinaw_2d,
    }

    dict_result_dma_test = {
        'cov_gap_1d':       test_cov_gap_1d,
        'picp_1d':          test_picp_1d,
        'pinaw_1d':         test_pinaw_1d,
        'width_1d':         test_width_1d,
        'robust_pinaw_1d':  test_robust_pinaw_1d,
        'preds':            CQR_preds,
        'true':             data_test_dma_y,
        'index':            data_test_dma_ind,
        'mae_1d':           test_mae_1d,
        'RMAE_1d':          test_RMAE_1d,
        'MAPE_1d':          test_MAPE_1d,
        'mae_2d':           test_mae_2d,
        'RMAE_2d':          test_RMAE_2d,
        'MAPE_2d':          test_MAPE_2d,
        'cov_gap_2d':       test_cov_gap_2d,
        'picp_2d':          test_picp_2d,
        'pinaw_2d':         test_pinaw_2d,
        'robust_pinaw_2d':  test_robust_pinaw_2d,
    }

    print(f'DMA {dma} Cov. GAP: {test_cov_gap_1d}, picp: {test_picp_1d}, PINAW: {test_pinaw_1d}, Robust PINAW: {test_robust_pinaw_1d}')

    dict_results[dma] = {'test':dict_result_dma_test, 'val':dict_result_dma_val}


save_folder = f'./show_results/results/{project_name}'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

torch.save(dict_results, f'./show_results/results/{project_name}/cp_results_final.pt')


