import torch
import torch.nn as nn
import numpy as np
import time
import copy
import os
import wandb
from ForecastingModel.ScoreFunctions import *
from ForecastingModel.ProbabilisticLosses import QuantileLoss, GMMLoss

class Forecaster():
    def __init__(
            self, 
            model, 
            name,
            config, 
            save_path, 
            x_val, 
            y_val,
            naive_val,
            scale1s,
            scale2s,
            compile=False, 
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            seed=42,
            ):
        """
        Make the class that defines the forecaster.

        Input Arguments:
        Model: Class: of the Neural Network

        Name: Str: Name of our experiment

        Config: Dict: Configuration of the model

        save_path: Str: path to save the model at

        x_val: torch.Tensor: validation input data

        y_val: torch.Tensor: validation target data

        naive_val: torch.Tensor: naive forecast of the validation data

        scale1s: torch.Tensor: scale1 of the validation data (mean)

        scale2s: torch.Tensor: scale2 of the validation data (std)
        
        compile: Bool: if we want to compile the model or not

        device: torch.device: device to run the model on

        seed: int: seed to set the random seed
        """
        
        # set variables to run
        self.device = device
        self.seed = seed
        self.results = {}
        self.results_probabilistic = {}
        self.scale1s = torch.tensor(scale1s, device=self.device)
        self.scale2s = torch.tensor(scale2s, device=self.device)
        self.name = name
        self.config = copy.deepcopy(config)
        self.torch_seed = torch.manual_seed(self.seed)
        self.model_uninitialized = model
        self.criterion_str = self.config['criterion']

        # set criterion for deterministic/probabilsitic models
        if self.criterion_str == 'QuantileLoss':
            # self.cov_gap_regularization =  self.config['cov_gap_regularization'] 
            self.criterion = QuantileLoss(self.config['quantiles'], self.device)
        elif self.criterion_str == 'GMMLoss':
            self.criterion = GMMLoss(variance_regularization=self.config['mdn_variance_regularization'], weights_regularization=self.config['mdn_weights_regularization'])
        else:
            self.criterion = getattr(nn, self.criterion_str)()

        # set output transform of output variables. for example QR requires a different output transform than a GMM.
        # The general output is just a [batch_size, outputs]. For QR for example we need [batch_size, forecast horizon, quantiles]
        self.probabilistic_transform = self.__no_probabilistic_transform
        if self.config['probabilistic_method'] == 'quantile_regression':
            # adapt forecasting sequence to include each of the quantiles
            self.config['forecast_sequence_length'] = self.config['forecast_sequence_length'] * len(self.config['quantiles'])
            self.quantiles = torch.tensor(self.config['quantiles']).to(device=self.device)
            self.probabilistic_transform = self.__probabilistic_transform_qr
            self.val_criteria = ['relative_pb', 'cov gap', 'robust pinaw', 'best_crit_save_model', 'crit_hpam']
        elif self.config['probabilistic_method'] == 'mixture_density_network':
            self.config['forecast_sequence_length'] = self.config['forecast_sequence_length'] * (self.config['number_of_gaussians'] * 3)
            self.probabilistic_transform = self.__probabilistic_transform_mdn
            self.quantiles = torch.tensor(self.config['quantiles']).to(device=self.device)

            # set validation criteria that are saved for the probabilistic model
            self.val_criteria = ['relative_pb', 'cov gap', 'robust pinaw', 'best_crit_save_model', 'crit_hpam']
        else:

            # set validation criteria that are saved for the deterministic model
            self.val_criteria = ['RMAE', 'MAPE', 'MAE', 'best_crit_save_model', 'crit_hpam']

        # save the NN class to be used for the model
        self.model = self.model_uninitialized(self.config)
        
        # save if we compile the model or not
        self.compile = compile
        if self.compile:
            self.model = torch.compile(self.model)

        # send the model already to our GPU if available
        self.model.to(self.device)

        # set the criterion of the hyperparameter optimization
        self.criteria_hpam = np.inf


        # set save path
        self.save_path = save_path

        # l2 regularization if we need
        self.weight_decay = self.config['weight_decay']

        # set optimizer of our model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=self.weight_decay)

        # empty lists to save the losses
        self.train_losses=[]
        self.val_losses=[]

        # make placeholder for the train time
        self.train_time = 0

        # all sorts of settings that speak for themselves
        self.trained_ = False
        self.num_epochs = self.config['num_epochs']
        self.echo_iter = self.config['echo_iter']
        self.l1_penalty = self.config['l1_penalty']
        self.early_stopping = self.config['early_stopping']
        self.patience_1 = self.config['patience_1']
        self.delta_1 = self.config['delta_1']
        self.patience_2 = self.config['patience_2']
        self.clip_grad = self.config['clip_grad']
        self.wandb_logging = self.config['wandb_logging']
        self.crit_to_save_model = self.config['crit_nr_to_save'] 
        self.crit_to_optimize = self.config['crit_nr_to_optimize']
        self.val_criteria_best_safe_model = np.inf
        self.criteria_hpam = np.inf

        # placeholder for the validation results
        self.val_results = np.zeros(
            (self.num_epochs, 1+len(self.val_criteria))
            )

        # make directories if we want to save the model
        if self.save_path != False:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            self.save_path = os.path.join(self.save_path, name)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        # if we want to log to wandb, we save the model with the wandb id
        if self.wandb_logging:
            self.savepath_model = os.path.join(self.save_path, 'model_' + str(wandb.run.id) + '.pt')
            torch.save(dict(self.config), os.path.join(self.save_path, 'config_' + str(wandb.run.id) + '.pt'))
            torch.save(dict(self.config), os.path.join(wandb.run.dir, 'config.pt'))
        elif self.save_path != False:
            self.savepath_model = os.path.join(self.save_path, 'model.pt')

        # set the validation loaders
        self.x_val = x_val
        self.y_val = y_val
        self.naive_val = naive_val
        if self.x_val is not None and self.y_val is not None:
            if type(self.x_val) != list:
                self.x_val = [self.x_val]
            if type(self.y_val) != list:
                self.y_val = [self.y_val]
            
            self.val_loaders = [torch.utils.data.DataLoader(list(zip(xv, yv, nv)), batch_size=config['batch_size'], shuffle=False)
                               for (xv, yv, nv) in zip(self.x_val, self.y_val, self.naive_val)]
            

    def __probabilistic_transform_qr(self, y_pred):
        """
        Hidden function that defines the transform of the output of the NN if we use quantile regression

        Input Arguments:
        y_pred: torch.Tensor: direct output of the NN


        """

        return y_pred.view(y_pred.shape[0], len(self.config['quantiles']), -1).permute(0, 2, 1)

    def __no_probabilistic_transform(self, y_pred):
        """
        Hidden function that defines the transform if we do not want a transform of the output of the NN

        Input Arguments:    
        y_pred: torch.Tensor: direct output of the NN
    

        """

        return y_pred
    
    def __probabilistic_transform_mdn(self, y_pred):
        """
        Hidden function that defines the transform of the output of the NN if we use the Gaussian Mixture Density Network

        Input Arguments:
        y_pred: torch.Tensor: direct output of the NN

        """

        # get the right dimensions
        nr_gaussians = self.config['number_of_gaussians']
        batch_size = y_pred.shape[0]
        y_pred = y_pred.view(batch_size, -1, nr_gaussians, 3)
        weights = y_pred[:,:,:,0]
        weights = torch.nn.functional.softmax(weights, dim=-1) + 1e-15
        means = y_pred[:,:,:,1]
        sigma = y_pred[:,:,:,2]

        # make sure the std is positive using the ELU
        sigma = torch.nn.ELU()(sigma) + 1.1
        
        # clip the std at between 0.1 and 10 to make sure we do not have a collapsing distribution or practically no distribution
        sigma = torch.clamp(sigma, 0.1, 10)

        return weights, means, sigma 
    
    def compute_parameters_model(self):
        """
        simple function that computes the nr of parame

        """

        n_params = sum(p.numel() for p in self.model.parameters())
        return n_params
           

    def predict_mcd(self, x, n_samples=1000, quantiles=[0.025, 0.975],  r=0.5, return_samples=False, batch_size=1, empirical_quantiles=True):
        """
        Function that predicts the output of the model using Monte Carlo Dropout

        Input Arguments:
        x: torch.Tensor: input data, x

        n_samples: int: number of samples to draw from the model

        quantiles: list: list of quantiles to compute the prediction intervals

        r: float: dropout rate

        return_samples: bool: if we want to return the samples or the prediction intervals

        batch_size: int: batch size of the data

        empirical_quantiles: bool: if we want to compute the quantiles empirically or use a Gaussian approximation

        """

        x_batch = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=False)

        if r == 'fitted':
            r = self.fitted_dropout_rate

        # prepare dropout model
        self.dropout_model_state_dict = copy.deepcopy(self.best_model.state_dict())
        self.dropout_config = copy.deepcopy(self.config)
        self.dropout_config['dropout_rate'] = r
        self.dropout_model = self.model_uninitialized(self.dropout_config)
        if self.compile:
            self.dropout_model = torch.compile(self.dropout_model)
        self.dropout_model.load_state_dict(self.dropout_model_state_dict)

        # get ensemble forecast
        self.dropout_model.train()
        y_preds = []
        for _ in range(n_samples):
            with torch.no_grad():
                y_pred = [self.dropout_model(x_batch_i) for x_batch_i in x_batch]
                y_pred = torch.cat(y_pred, dim=0)
                y_preds.append(y_pred)
        y_preds = torch.stack(y_preds, dim=-1)

        # get prediction intervals
        if return_samples:
            return y_preds
        
        else:
            if empirical_quantiles:
                quantiles = torch.Tensor(quantiles).to(self.device)
                y_preds_PI = torch.quantile(y_preds, quantiles, dim=-1).transpose(0,-1).transpose(0,1)
            else:
                # use gaussian approximation
                mu = torch.mean(y_preds, dim=-1)
                sigma = torch.std(y_preds, dim=-1)

                # create gaussian with mu and sigma
                gaus_preds = torch.distributions.Normal(mu, sigma, device=self.device)

                # get PI from gaussian
                lower = gaus_preds.icdf(quantiles[0])
                higher = gaus_preds.icdf(quantiles[1])
                # combine lower and higher to get y_preds, join on last axis
                y_preds_PI = torch.stack((lower, higher), dim=-1)

            return y_preds_PI

    def __evaluate_model(self, epoch):
        """
        Hidden function that evaluates the model on each of the validation sets and computes the averages of these

        Input Arguments:
        epoch: int: current epoch

        """

        self.model.eval()

        nr_val_loaders = len(self.val_loaders)

        # obtain the results of each of the validation sets
        for i, val_loader in enumerate(self.val_loaders):

            outputs_concat = []
            targets_concat = []
            naive_concat = []

            # get the outputs batch wise (in case the model has batch normalization)
            for inputs, targets, naive in val_loader:
                outputs = self.model(inputs)
                outputs_concat.append(outputs)
                targets_concat.append(targets)
                naive_concat.append(naive)

            # concatenate the outputs and targets
            outputs_concat = torch.cat(outputs_concat, dim=0)
            targets_concat = torch.cat(targets_concat, dim=0)
            naive_concat = torch.cat(naive_concat, dim=0)
            
            # perform the transform in case we have a probabilistic models
            outputs_concat = self.probabilistic_transform(outputs_concat)

            # compute loss
            loss = self.criterion(outputs_concat, targets_concat).item() / nr_val_loaders
            self.val_results[epoch, 0] += loss

            # in case our model is a mixture density network, we have the parameterization
            # but not yet the drawn samples. We draw the samples here
            if self.config['probabilistic_method'] == 'mixture_density_network':
                outputs_concat = self.__mdn_data_into_pred(
                    mdn_pred=outputs_concat,
                    num_samples=self.config['num_mdn_samples'],
                    batch_size_samples=self.config['batch_size']
                )

            # scale the outputs and targets back to the original scale
            s1 = self.scale1s[i]
            s2 = self.scale2s[i]
            # print(outputs_concat)
            outputs_concat = outputs_concat * s1 + s2
            targets_concat = targets_concat * s1 + s2

            # compute the validation criteria if model is deterministic
            if not self.config['probabilistic_method']:
                RMAE = compute_RMAE_1d(targets_concat, outputs_concat, y_naive=naive_concat).item() / nr_val_loaders
                MAPE = compute_MAPE_1d(targets_concat, outputs_concat).item() / nr_val_loaders
                MAE = compute_MAE_1d(targets_concat, outputs_concat).item() / nr_val_loaders
                self.val_results[epoch, 1] += RMAE
                self.val_results[epoch, 2] += MAPE
                self.val_results[epoch, 3] += MAE

            # compute the validation criteria if model is probabilistic
            elif self.config['probabilistic_method'] == 'quantile_regression' or self.config['probabilistic_method'] == 'mixture_density_network':

                relative_pb = compute_relative_pb(targets_concat, outputs_concat, naive_concat[i]).item() / nr_val_loaders
                cov_gap = compute_cov_gap_1d(targets_concat, outputs_concat).item() / nr_val_loaders
                # pinaw = compute_pinaw_1d(targets_concat, outputs_concat).item() / nr_val_loaders

                robust_pinaw = compute_robust_pinaw_1d(targets_concat, outputs_concat).item() / nr_val_loaders

                self.val_results[epoch, 1] += relative_pb
                self.val_results[epoch, 2] += cov_gap
                self.val_results[epoch, 3] += robust_pinaw
            else:
                raise ValueError('Probabilistic method not recognized')


    def fit(self, x_train, y_train):
        """
        Function that fits the model to the training data
        This is the function we use for fitting the model to the training data

        Input Arguments: 
        x_train: torch.Tensor: input data, x
        y_train: torch.Tensor: target data, y

        """

        # make train loader
        self.train_loader = torch.utils.data.DataLoader(list(zip(x_train, y_train)), batch_size=self.config['batch_size'], shuffle=True)

        # train and validate
        self.__train_and_validate()
        self.trained_ = True


    def predict(self, x_test, batch_size=1, return_tensor=True):
        """
        Function that predicts the output of the model

        Input Arguments:
        x_test: torch.Tensor: input data, x

        batch_size: int: batch size of the data

        return_tensor: bool: if we want to return the output as a tensor or as a numpy array

        """

        x_test_batch = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=False)
       
        self.best_model.eval()
        self.model.eval()

        with torch.no_grad():
            y_pred = [self.best_model(x_test_b) for x_test_b in x_test_batch]
            y_pred = torch.cat(y_pred, dim=0)
            y_pred = self.probabilistic_transform(y_pred)
            if return_tensor:               
                return y_pred
            else:
                return y_pred.cpu().numpy()
            

    def __mdn_data_into_pred(self, mdn_pred, num_samples, batch_size_samples=256):
        """
        Hidden function that transforms the output of the MDN into the prediction intervals
        
        Input Arguments:
        mdn_pred: torch.Tensor: output of the MDN

        num_samples: int: number of samples to draw from the MDN

        batch_size_samples: int: batch size of drawing the samples

        """

        # make loader to draw the samples and draw them
        mdn_loader = torch.utils.data.DataLoader(torch.stack(mdn_pred,dim=1), batch_size=batch_size_samples, shuffle=False)
        y_preds = torch.Tensor([]).to(device=self.device)
        quantile = self.quantiles
        for b in mdn_loader:
            weights = b[:,0,:,:]
            means = b[:,1,:,:]
            sigmas = b[:,2,:,:]
            mixture = torch.distributions.Categorical(weights)
            components = torch.distributions.Normal(means, sigmas)
            mixed = torch.distributions.MixtureSameFamily(mixture, components)
            samples = mixed.sample([num_samples])
            y_preds_batch = torch.quantile(samples, quantile, dim=0).transpose(0,-1).transpose(0,1)
            y_preds = torch.cat((y_preds, y_preds_batch), dim=0).to(device=self.device)

        return y_preds
        
    def predict_mdn(self, x_test, batch_size_pred=256, batch_size_samples=256, num_samples=1000, quantile=[0.025, 0.975], return_tensor=True):
        """
        Function that predicts the output of the model using the Gaussian Mixture Density Network

        Input Arguments:
        x_test: torch.Tensor: input data, x

        batch_size_pred: int: batch size of the prediction

        batch_size_samples: int: batch size of drawing the samples

        num_samples: int: number of samples to draw from the MDN

        quantile: list: list of quantiles to compute the prediction intervals

        return_tensor: bool: if we want to return the output as a tensor or as a numpy array

        """

        mdn_pred = self.predict(
            x_test, 
            batch_size=batch_size_pred, 
            return_tensor=True
            )

        quantile = torch.Tensor(quantile).to(device=self.device)

        y_preds = self.__mdn_data_into_pred(mdn_pred, num_samples, batch_size_samples=batch_size_samples)

        if return_tensor:
            return y_preds
        else:
            return y_preds.cpu().numpy()

    def fit_dropout_rate(self, 
            r_low, 
            r_high, 
            calib_set,
            n_samples=1000, 
            max_cov_gap=0.005, 
            max_iterations=5, 
            PI=0.95, 
            batch_size=256, 
            log_wandb=False,
            ):
        """
        Function that fits the dropout rate using the dichotomic search

        Input Arguments:
        r_low: float: lower bound of the dropout rate

        r_high: float: upper bound of the dropout rate

        calib_set: [torch.Tensor, torch.Tensor] calibration data x and y to find dropout rate with

        n_samples: int: number of samples to draw from the model

        max_cov_gap: float: maximum coverage gap

        max_iterations: int: maximum number of iterations

        PI: float: prediction interval

        batch_size: int: batch size of the data

        log_wandb: bool: if we want to log the results to wandb

        """
        
        print('Start Dropout Rate Optimization')
        coverage_gap=1
        quantiles=[0.5-PI/2, 0.5+PI/2]

        ds_x = calib_set[0]
        ds_y = calib_set[1]

        c=0
        while (coverage_gap > max_cov_gap):

            if c > max_iterations:
                print('Max Iterations Reached')
                print('PICP:', picp, 'Coverage Gap:', coverage_gap, 'r:', r,)
            
                break 

            r = (r_low + r_high) / 2
            results = self.predict_mcd(ds_x, n_samples=n_samples, quantiles=quantiles, r=r, batch_size=batch_size)
            picp = compute_picp_1d(ds_y, results)

            coverage_gap = torch.abs(picp - PI)
            
            if log_wandb:
                wandb.log({'dropout_rate':r, 'PICP': picp})

            if picp < PI:
                r_low = r
            else:
                r_high = r
            print('PICP:', picp, 'Coverage Gap:', coverage_gap, 'r:', r)
            c+=1


        self.fitted_dropout_rate = r
        


    def __train_and_validate(self):
        """
        Hidden function that trains and validates the model

        """
        
        self.train_losses = []

        n_weights = 0
        for name, weights in self.model.named_parameters():
            if 'bias' not in name:
                n_weights += weights.numel()

        start_time = time.time()  # Start training time
        c=0
        for epoch in range(self.num_epochs):
            # Training Phase
            self.model.train()
            total_train_loss = 0
            for inputs, targets in self.train_loader:

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs = self.probabilistic_transform(outputs)
                loss = self.criterion(outputs, targets)

                # L1 Regularization
                if self.l1_penalty > 0:
                    L1_term = torch.tensor(0., requires_grad=True).to(self.device)
                    for name, weights in self.model.named_parameters():
                        weights_sum = torch.sum(torch.abs(weights))
                        L1_term = L1_term + weights_sum
                    L1_term = L1_term / n_weights
                    loss = loss + self.l1_penalty * L1_term

                loss.backward()

                # gradient clipping
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

                # update weights
                self.optimizer.step()

                # add loss
                total_train_loss += loss.item()

            # make sure we have the average loss
            avg_train_loss = total_train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            # evaluate model
            self.__evaluate_model(epoch)

            # compute train time
            self.train_time = time.time() - start_time

            # save the results of the validation set
            val_criteria_this_epoch = self.val_results[epoch, self.crit_to_save_model]
            criteria_hpam_this_epoch = self.val_results[epoch, self.crit_to_optimize]

            # make sure we save the best model
            if val_criteria_this_epoch < self.val_criteria_best_safe_model:
                print('keeping model at epoch:', epoch, 'with val criteria:', val_criteria_this_epoch, 'best previous val criteria:', self.val_criteria_best_safe_model)

                self.val_criteria_best_safe_model = val_criteria_this_epoch
                self.criteria_hpam = criteria_hpam_this_epoch

                self.best_model = copy.deepcopy(self.model)
                if self.save_path != False:
                    with open(self.savepath_model, 'wb') as f:
                        torch.save(self.model.state_dict(), self.savepath_model)
                    f.close()
                c=0

            else:
                c+=1

            self.val_results[epoch, -2] = self.val_criteria_best_safe_model
            self.val_results[epoch, -1] = self.criteria_hpam

            # log the results
            log_dict = dict(zip(self.val_criteria, self.val_results[epoch, 1:]))
            log_dict = {
                    **log_dict,
                    'val_loss'                          : self.val_results[epoch, 0],
                    "train_loss_epoch"                  : avg_train_loss, 
                    'epoch'                             : epoch,
                    'training_time'                        : self.train_time,
            }
            
            if self.wandb_logging:
                wandb.log(
                    log_dict
                )
            if self.echo_iter > 0:
                if (epoch + 1) % self.echo_iter == 0:
                    print(log_dict)

        self.epochs_trained = epoch
        if self.wandb_logging:
            torch.save(self.best_model.state_dict(),  os.path.join(wandb.run.dir, f'state_dict_model.pt'))
        if self.echo_iter > 0:
            print("Training complete.")

    def load_model(self, path):
        """
        Function that loads a model from a given path

        Input Arguments:
    
        path: str: path to the model

        """

        state_dict = torch.load(path, map_location=self.device)
        self.model = self.model_uninitialized(self.config)
        if self.compile:
            self.model = torch.compile(self.model)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.best_model = copy.deepcopy(self.model)
        self.trained_ = True
    
    def log_results_interval_forecast(self, q_preds, y_test, name, scale1, scale2):
        """
        Function that logs the results of the interval forecast

        Input Arguments:
        q_preds: torch.Tensor: predicted quantiles [number of forecasts, shape forecast sequence length, lower and higher quantile]

        y_test: torch.Tensor: target data, shape [number of forecasts, forecast sequence length]

        name: str: name of the model

        scale1: torch.Tensor: scale1 of the data

        scale2: torch.Tensor: scale2 of the data
        
        """

        q_preds_untransformed = q_preds * scale1 + scale2
        y_test_untransformed = y_test * scale1 + scale2

        # 1d metrics
        cov_gap_1d = compute_cov_gap_1d(y_test_untransformed, q_preds_untransformed, PI=0.95).item()
        picp_1d = compute_picp_1d(y_test_untransformed, q_preds_untransformed).item()
        pinaw_1d = compute_pinaw_1d(y_test_untransformed, q_preds_untransformed).item()
        robust_pinaw_1d = compute_robust_pinaw_1d(y_test_untransformed, q_preds_untransformed).item()
        width_1d = compute_width_1d(q_preds_untransformed).item()
        # relative_pb = compute_relative_pb(y_test_untransformed, q_preds_untransformed,y_naive)
        # loss = self.criterion(q_preds, y_test).item()

        picp_2d = compute_picp_2d(y_test_untransformed, q_preds_untransformed)
        cov_gap_2d = compute_cov_gap_2d(y_test_untransformed, q_preds_untransformed, PI=0.95)
        pinaw_2d = compute_pinaw_2d(y_test_untransformed, q_preds_untransformed)
        robust_pinaw_2d = compute_robust_pinaw_2d(y_test_untransformed, q_preds_untransformed)
        width_2d = compute_width_2d(q_preds_untransformed)

        results_probabilistic = {
            # name+'_loss'                      : loss,
            name+'_cov_gap_1d'                : cov_gap_1d,
            name+'_picp_1d'                    : picp_1d,
            name+'_pinaw_1d'                  : pinaw_1d,
            name+'_robust_pinaw_1d'           : robust_pinaw_1d,
            name+'_width_1d'                  : width_1d,

        }
    
        results_probabilistic_2d = {
            name+'_cov_gap_2d'                : cov_gap_2d,
            name+'_picp_2d'                    : picp_2d,
            name+'_pinaw_2d'                  : pinaw_2d,
            name+'_width_2d'                  : width_2d,
            name+'_robust_pinaw_2d'           : robust_pinaw_2d,
        }

        series = {
            'q_preds': q_preds,
            'y_test': y_test,
            'scale1': scale1,
            'scale2': scale2,
        }

        if self.wandb_logging:
            wandb.log(results_probabilistic)

        results = {
            'name': name,
            'series': series,
            'metrics': results_probabilistic,
            'metrics_2d': results_probabilistic_2d
        }

        self.results_probabilistic[name] = results


        # shape y_pred: (batch size, forecast sequence length, lower quantile and upper quantile)
        
    def log_results_pt_forecast(self, y_pred, y_test, y_naive, name, scale1, scale2):
        """
        Function that logs the results of the point forecast

        Input Arguments:
        y_pred: torch.Tensor: point forecast, shape [number of forecasts, forecast sequence length]

        y_test: torch.Tensor: target data, shape [number of forecasts, forecast sequence length]

        name: str: name of the model

        scale1: torch.Tensor: scale1 of the data

        scale2: torch.Tensor: scale2 of the data
        
        """



        loss = compute_MAE_1d(y_pred, y_test)

        y_test_transformed =  y_test *  scale1 + scale2
        y_pred_transformed =  y_pred *  scale1 + scale2
        y_naive_transformed = y_naive * scale1 + scale2

        MAE = compute_MAE_1d(y_test_transformed, y_pred_transformed)
        RMAE = compute_RMAE_1d(y_test_transformed, y_pred_transformed, y_naive=y_naive_transformed)
        MAPE = compute_MAPE_1d(y_test_transformed, y_pred_transformed)
        lowest_AE = compute_lowest_AE(y_test_transformed, y_pred_transformed)
        lowest_APE = compute_lowest_APE(y_test_transformed, y_pred_transformed)
        lowest_RAE = compute_lowest_RAE(y_test_transformed, y_pred_transformed, y_naive=y_naive_transformed)
        highest_AE = compute_highest_AE(y_test_transformed, y_pred_transformed)
        highest_APE = compute_highest_APE(y_test_transformed, y_pred_transformed)
        highest_RAE = compute_highest_RAE(y_test_transformed, y_pred_transformed, y_naive=y_naive_transformed)
        median_AE = compute_median_AE(y_test_transformed, y_pred_transformed)
        median_APE = compute_median_APE(y_test_transformed, y_pred_transformed)
        median_RAE = compute_median_RAE(y_test_transformed, y_pred_transformed, y_naive=y_naive_transformed)

        mae2d = compute_MAE_2d(y_test_transformed, y_pred_transformed)
        RMAE2d = compute_RMAE_2d(y_test_transformed, y_pred_transformed, y_naive=y_naive_transformed)
        MAPE2d = compute_MAPE_2d(y_test_transformed, y_pred_transformed)

        mae_naive_2d = compute_MAE_2d(y_test_transformed, y_naive_transformed)
        MAPE_naive_2d = compute_MAPE_2d(y_test_transformed, y_naive_transformed)  
    
        save_metrics = {
            name+'_loss'                    : loss.item(),
            name+'_MAE'                     : MAE.item(), 
            name+'_RMAE'                    : RMAE.item(),
            name+'_MAPE'                    : MAPE.item(),
        }

        
        save_metrics_2d = {
            name+'_MAE_2d'                  : mae2d,
            name+'_RMAE_2d'                 : RMAE2d,
            name+'_MAPE_2d'                 : MAPE2d,
            name+'_MAE_naive_2d'             : mae_naive_2d,
            name+'_MAPE_naive_2d'           : MAPE_naive_2d,
        }

        if self.wandb_logging:
            wandb.log(save_metrics)

        save_metrics[name+'_lowest_AE'] = lowest_AE.item()
        save_metrics[name+'_lowest_APE'] = lowest_APE.item()
        save_metrics[name+'_lowest_RAE'] = lowest_RAE.item()
        save_metrics[name+'_highest_AE'] = highest_AE.item()
        save_metrics[name+'_highest_APE'] = highest_APE.item()
        save_metrics[name+'_highest_RAE'] = highest_RAE.item()
        save_metrics[name+'_median_AE'] = median_AE.item()
        save_metrics[name+'_median_APE'] = median_APE.item()
        save_metrics[name+'_median_RAE'] = median_RAE.item()

        series = {
            'y_pred': y_pred,
            'y_test': y_test,
            'scale1': scale1,
            'scale2': scale2,
        }

        result = {
            'name': name,
            'series': series,
            'metrics': save_metrics,
            'metrics_2d': save_metrics_2d
        }

        self.results[name] = result


    def save_results(self, probabilistic=True, point=True):
        """
        Function that saves the results and logs them, both if asked

        Input Arguments:
        probabilistic: bool: True if we want to save the interval forecasts

        point: bool: True if we want to save the point forecasts
        """
        
        if probabilistic:
            if self.wandb_logging:
                torch.save(self.results_probabilistic, os.path.join(wandb.run.dir, 'results_probabilistic.pt'))
            if self.save_path:
                torch.save(self.results_probabilistic, os.path.join(self.save_path, f'results_probabilistic.pt'))
        
        if point:
            if self.wandb_logging:
                torch.save(self.results, os.path.join(wandb.run.dir, 'results.pt'))
            if self.save_path:
                torch.save(self.results, os.path.join(self.save_path, f'results.pt'))
        
        