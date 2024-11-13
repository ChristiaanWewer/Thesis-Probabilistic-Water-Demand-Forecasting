import torch
from ForecastingModel.ProbabilisticLosses import QuantileLoss

def compute_MAE_2d(y_true, y_pred):
    """
    Compute the MAE over the (Mean Absolute Error) over the forecasting sequence length

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The MAE tensor of shape (forecasting sequence length, ).
    """

    mae = torch.mean(torch.abs(y_true - y_pred), axis=0)

    return mae

def compute_RMAE_2d(y_true, y_pred, y_naive):
    """
    Compute the Relative Mean Absolute Error (RMAE) over the forecasting sequence length

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The MAPE tensor of shape (1).
    """  
    
    abs_mae = torch.mean(torch.abs(y_true - y_pred),axis=0)
    abs_mae_naive = torch.mean(torch.abs(y_true - y_naive),axis=0)
    RMAE  = abs_mae/abs_mae_naive
    return RMAE

def compute_lowest_AE(y_true, y_pred):
    """
    Compute the lowest Absolute Error (AE)

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The lowest AE tensor of shape (1).
    """

    abs_mae = torch.abs(y_true - y_pred)
    return abs_mae.min()

def compute_highest_AE(y_true, y_pred):
    """
    Compute the highest Absolute Error (AE)

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The highest AE tensor of shape (1).
    """

    abs_mae = torch.abs(y_true - y_pred)
    return abs_mae.max()

def compute_median_AE(y_true, y_pred):
    """
    Compute the median Absolute Error (AE)

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The median AE tensor of shape (1).
    """
        
    abs_mae = torch.abs(y_true - y_pred)
    return abs_mae.median()

def compute_lowest_RAE(y_true, y_pred, y_naive):
    """
    Compute the lowest Relative Absolute Error (RAE)
    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The lowest RAE tensor of shape (1).
    """
        
    abs_mae = torch.abs(y_true - y_pred)
    abs_mae_naive = torch.mean(torch.abs(y_true - y_naive))
    RMAE  = abs_mae/abs_mae_naive
    return RMAE.min()

def compute_highest_RAE(y_true, y_pred, y_naive):
    """
    Compute the highest Relative Absolute Error (RAE)

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).
    y_naive: torch.Tensor: The naive forecast tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The highest RAE tensor of shape (1).
    """
        
    abs_mae = torch.abs(y_true - y_pred)
    abs_mae_naive = torch.mean(torch.abs(y_true - y_naive))
    RMAE  = abs_mae/abs_mae_naive
    return RMAE.max()

def compute_median_RAE(y_true, y_pred, y_naive):
    """
    Compute the median Relative Absolute Error (RAE)

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).
    y_naive: torch.Tensor: The naive forecast tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The median RAE tensor of shape (1).
    """
        
    abs_mae = torch.abs(y_true - y_pred)
    abs_mae_naive = torch.mean(torch.abs(y_true - y_naive))
    RMAE  = abs_mae/abs_mae_naive
    return RMAE.median()

def compute_lowest_APE(y_true, y_pred):
    """
    Compute the lowest Absolute Percentage Error (APE)

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The lowest APE tensor of shape (1).
    """
    
    MAPE = 100 * torch.abs(y_true - y_pred) / torch.abs(y_true)
    return MAPE.min()

def compute_highest_APE(y_true, y_pred):
    """
    Compute the highest Absolute Percentage Error (APE)

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).
    y_naive: torch.Tensor: The naive forecast tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The highest RAE tensor of shape (1).
    """
        
    MAPE = 100 * torch.abs(y_true - y_pred) / torch.abs(y_true)
    return MAPE.max()

def compute_median_APE(y_true, y_pred):
    """
    Compute the median Absolute Percentage Error (APE)

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).
    y_naive: torch.Tensor: The naive forecast tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The median RAE tensor of shape (1).
    """
            
    MAPE = 100 * torch.abs(y_true - y_pred) / torch.abs(y_true)
    return MAPE.median()


def compute_MAPE_2d(y_true, y_pred):
    """
    Compute the Mean Absolute Percentage Error (MAPE) over the forecasting sequence length

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The MAPE tensor of shape (1).
    """

    MAPE = 100 * torch.mean(torch.abs(y_true - y_pred) / torch.abs(y_true), axis=0)

    return MAPE


def compute_relative_pb(y_true, y_pred, naive_q_loss):
    """
    Compute the Relative Pinball Loss (RPL) 

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The RPL tensor of shape (1).
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    quantiles = torch.tensor([0.025, 0.5, 0.975])
    qloss = QuantileLoss(quantiles, device)
    loss_y_naive = qloss(y_pred, y_true)
    relative_pinball = loss_y_naive / naive_q_loss
    return relative_pinball


def compute_RMAE_1d(y_true, y_pred, y_naive):
    """
    Compute the Relative Mean Absolute Error (RMAE)

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The RMAE tensor of shape (1).
    """

    RMAE_2d = compute_RMAE_2d(y_true, y_pred, y_naive)
    RMAE = torch.mean(RMAE_2d)
    return RMAE


def compute_MAPE_1d(y_true, y_pred):
    """
    Compute the Mean Absolute Percentage Error (MAPE)

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The MAPE tensor of shape (1).
    """

    # MAPE = 100 * torch.mean(torch.abs(y_true - y_pred) / ((torch.abs(y_true))))
    MAPE_2d = compute_MAPE_2d(y_true, y_pred)
    MAPE = torch.mean(MAPE_2d)

    return MAPE

def compute_MAE_1d(y_true, y_pred):
    """
    Compute the Mean Absolute Error (MAE) over the forecasting sequence length

    Input Arguments:
    y_true: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    y_pred: torch.Tensor: The predicted tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The MAE tensor of shape (1).
    """

    mae_2d = compute_MAE_2d(y_true, y_pred)
    mae = torch.mean(mae_2d)
    return mae


def compute_picp_1d(y, q_y_hat):
    """
    Compute the Prediction Interval Coverage Probability (PICP) 

    Input Arguments:
    y: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    q_y_hat: torch.Tensor:The predicted quantiles tensor of shape (batch_size, sequence_length, 2).

    Returns:
    torch.Tensor: The PICP tensor of shape (forecasting sequence length,).
    """

    picp = torch.mean((q_y_hat[:, :, -1] >= y) & (q_y_hat[:, :, 0] <= y),dtype=float)

    return picp

def compute_cov_gap_1d(y, q_y_hat, PI=0.95):
    """
    Compute the 1D Coverage Gap 

    Input Arguments:
        y: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
        q_y_hat: torch.Tensor: The predicted quantiles tensor of shape (batch_size, sequence_length, 2).
        PI float: The desired prediction interval.

    Returns:
        torch.Tensor: The Coverage Gap tensor of shape (1).
    """

    picp = compute_picp_2d(y, q_y_hat)
    cov_gap = torch.mean(torch.abs(picp - PI))

    return cov_gap

def compute_width_1d(q_y_hat):
    """
    Compute the 1D width of the prediction interval 

    Input Arguments:
        q_y_hat: torch.Tensor: The predicted quantiles tensor of shape (batch_size, sequence_length, 2).

    Returns:
        torch.Tensor: The width tensor of shape (1).
    """

    width = torch.mean(q_y_hat[:, :, -1] - q_y_hat[:, :, 0])

    return width

def compute_pinaw_1d(y, q_y_hat):
    """
    Compute the 1D Prediction Interval Normalized Average Width (PINAW) 

    Input Arguments:
        y: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
        q_y_hat: torch.Tensor: The predicted quantiles tensor of shape (batch_size, sequence_length, 2).

    Returns:
        torch.Tensor: The PINAW tensor of shape (1).
    """

    interval_width = q_y_hat[:, :, -1] - q_y_hat[:, :, 0]
    R = y.max() - y.min()
    pinaw = interval_width.mean() / R
    
    return pinaw

def compute_robust_pinaw_1d(y, q_y_hat):
    """
    Compute the 1D Robust Prediction Interval Normalized Average Width (Robust PINAW) 

    Input Arguments:
        y: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
        q_y_hat: torch.Tensor: The predicted quantiles tensor of shape (batch_size, sequence_length, 2).

    Returns:
        torch.Tensor: The Robust PINAW tensor of shape (1).
    """

    interval_width = q_y_hat[:, :, -1] - q_y_hat[:, :, 0]
    
    # compute interquantile range of y
    IQR = y.quantile(0.75) - y.quantile(0.25)
    
    # robust pinaw
    robust_pinaw = interval_width.mean() / IQR

    return robust_pinaw

def compute_picp_2d(y: torch.Tensor, q_y_hat: torch.Tensor):
    """
    Compute the Prediction Interval Coverage Probability (PICP)  over the forecasting sequence length

    Input Arguments:
        y: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
        q_y_hat: torch.Tensor: The predicted quantiles tensor of shape (batch_size, sequence_length, 2).

    Returns:
        torch.Tensor: The PICP tensor of shape (forecasting sequence length,).
    """

    picp = torch.mean((q_y_hat[:, :, -1] >= y) & (q_y_hat[:, :, 0] <= y), dtype=float, axis=0)
    return picp


def compute_cov_gap_2d(y: torch.Tensor, q_y_hat: torch.Tensor, PI: float=0.95):
    """
    Compute the Coverage Gap over the forecasting sequence length

    Input Arguments:
    y: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
    q_y_hat: torch.Tensor: The predicted quantiles tensor of shape (batch_size, sequence_length, 2).
    PI: float: The desired prediction interval.

    Returns:
        torch.Tensor: The Coverage Gap tensor of shape (forecasting sequence length,).
    """

    picp = compute_picp_2d(y, q_y_hat)
    cov_gap = torch.abs(picp - PI)
    return cov_gap


def compute_width_2d(q_y_hat: torch.Tensor):
    """
    Compute the width of the prediction interval over the forecasting sequence length

    Input Arguments:
    q_y_hat: torch.Tensor: The predicted quantiles tensor of shape (batch_size, sequence_length, 2).

    Returns:
    torch.Tensor: The width tensor of shape (forecasting sequence length,).
    """

    width = torch.mean(q_y_hat[:, :, -1] - q_y_hat[:, :, 0], axis=0)
    return width


def compute_pinaw_2d(y: torch.Tensor, q_y_hat: torch.Tensor):
    """
    Compute the Prediction Interval Normalized Average Width (PINAW) over the forecasting sequence length

    Input Arguments:
        y: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
        q_y_hat: torch.Tensor: The predicted quantiles tensor of shape (batch_size, sequence_length, 2).

    Returns:
        torch.Tensor: The PINAW tensor of shape (forecasting sequence length,).
    """

    interval_width = q_y_hat[:, :, -1] - q_y_hat[:, :, 0]
    R = y.max(axis=0).values - y.min(axis=0).values
    pinaw = interval_width.mean(axis=0) / R
    return pinaw


def compute_winkler_score_2d(y: torch.Tensor, q_y_hat: torch.Tensor):
    """
    Compute the Winkler Score over the forecasting sequence length

    Input Arguments:
        y: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
        q_y_hat: torch.Tensor: The predicted quantiles tensor of shape (batch_size, sequence_length, 2).

    Returns:
        torch.Tensor:The Winkler Score tensor of shape (forecasting sequence length,).
    """

    l = q_y_hat[:, :, 0]
    u = q_y_hat[:, :, -1]
    y = y
    alpha = 0.05
    winkler_i = u-l + (2/alpha)*(l-y)*(y<l) + (2/alpha)*(y-u)*(y>u)

    return winkler_i.mean(axis=0)


def compute_winkler_score_1d(y: torch.Tensor, q_y_hat: torch.Tensor):
    """
    Compute the Winkler Score

    Input Arguments:
        y: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
        q_y_hat: torch.Tensor: The predicted quantiles tensor of shape (batch_size, sequence_length, 2).

    Returns:
        torch.Tensor:The Winkler Score tensor of shape (1).
    """

    return compute_winkler_score_2d(y, q_y_hat).mean()


def compute_robust_pinaw_2d(y: torch.Tensor, q_y_hat: torch.Tensor):
    """
    Compute the Robust Prediction Interval Normalized Average Width (Robust PINAW) 

    Input Arguments:
        y: torch.Tensor: The ground truth tensor of shape (batch_size, sequence_length).
        q_y_hat: torch.Tensor: The predicted quantiles tensor of shape (batch_size, sequence_length, 2).

    Returns:
        torch.Tensor: The Robust PINAW tensor of shape (forecasting sequence length,).
    """

    interval_width = q_y_hat[:, :, -1] - q_y_hat[:, :, 0]
    IQR = y.quantile(0.75, axis=0) - y.quantile(0.25, axis=0)
    robust_pinaw = interval_width.mean(axis=0) / IQR
    return robust_pinaw

def compute_loss_per_hour(x_true, y_true, y_pred, loss_func):
    '''
    Compute the loss per hour of the day for a given loss function.
    '''

    errors = loss_func(reduce=False)(y_pred, y_true)
    errors_per_hour = torch.zeros(y_true.shape[-1], device=y_pred.device)
    for i, x_t in enumerate(x_true):
        hour = torch.argmax(x_t['Hour_future_one_hot'].to(dtype=torch.int8), dim=1)
        errors_per_hour = errors_per_hour + errors[i][hour]
    errors_per_hour = errors_per_hour / (i+1)
    return errors_per_hour

def compute_2d_score_per_hour(x_true, y_true, y_pred, score_func):
    '''
    Compute the score per hour of the day for a given score function.

    '''

    # errors = score_func(y_pred, y_true)
    errors_per_hour = torch.zeros(y_true.shape[-1], device=y_pred.device)
    for i, x_t in enumerate(x_true, device=y_pred.device):

        pred_t = y_pred[i].unsqueeze(0)
        true_t = y_true[i].unsqueeze(0)

        error_t = score_func(true_t, pred_t)
        
        hour = torch.argmax(x_t['Hour_future_one_hot'].to(dtype=torch.int8), dim=1)
        errors_per_hour = errors_per_hour + error_t[hour]
    errors_per_hour = errors_per_hour / (i+1)
    return errors_per_hour