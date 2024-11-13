import torch

class ConformalPrediction:
    def __init__(self, 
                 model, 
                 modelclass, 
                 config, 
                 scale1s,
                 scale2s,
                 model_name, 
                 device='cpu',
        ):
        """
        Class for the Conformal Prediction

        Input Arguments:
        model: Class: model e.g. MLP class

        modelclass: Class: Forecaster class

        config: dict: dictionary with the configuration

        scale1s: torch.Tensor: scale with the means

        scale2s: torch.Tensor: scale with the stds

        model_name: str: name of the model

        device: str: 'cpu or 'cuda'
        """

        self.model = model
        self.modelclass = modelclass
        self.config = config
        self.quantiles = self.config['quantiles']
        self.model_name = model_name
        self.fitted_models = []
        self.device = device
        self.scale1s = scale1s
        self.scale2s = scale2s
    
    def compute_kfold_nonconformity_measures(
            self, 
            x_calib_fold_data, 
            y_calib_fold_data, 
            reset_nonconformity_scores=True
            ):
        """
        Compute the nonconformity scores for each fold in case of k-fold cross-validation to produce nonconformity scores

        Input Arguments:
        x_calib_fold_data: list: list of tensors with the x calibration data for each fold

        y_calib_fold_data: list: list of tensors with the y calibration data for each fold

        reset_nonconformity_scores: bool: if True, reset the nonconformity scores      
        
        """
        
        if reset_nonconformity_scores:
            self.nonconformity_scores = None

        if type(x_calib_fold_data) != list:
            x_calib_fold_data = [x_calib_fold_data]
        if type(y_calib_fold_data) != list:
            y_calib_fold_data = [y_calib_fold_data]
        

        for i in range(len(x_calib_fold_data)):

            x_calib_i = x_calib_fold_data[i]
            y_calib_i = y_calib_fold_data[i]

            self.compute_nonconformity_scores(
                model=i,
                x_calibrate=x_calib_i, 
                y_calibrate=y_calib_i,
                reset_nonconformity_scores=True if i==0 else False
                )   
            
            print('Computed Nonconformity Measures for fold: ', i)

    def compute_nonconformity_scores(
            self, 
            model, 
            x_calibrate, 
            y_calibrate, 
            reset_nonconformity_scores=True
            ):
        """
        Compute the nonconformity scores for a produce nonconformity scores

        Input Arguments:
        model: int or str: index of the model r 'all' 

        x_calibrate: torch.Tensor: x calibration data

        y_calibrate: torch.Tensor: y calibration data

        reset_nonconformity_scores: bool: if True, reset the nonconformity scores
        """
        
        # predict the model on the calibration data
        y_preds = self.predict(x_calibrate, model=model)

        # expand y_calibrate to the same shape as y_preds
        y_calibrate = y_calibrate.unsqueeze(-1).expand_as(y_preds)

        # compute the nonconformity scores
        # if we reet the nonconformity scores, we set them to the residuals
        # otherwise, we concatenate the residuals
        if reset_nonconformity_scores:
            self.nonconformity_scores = y_calibrate - y_preds
        else:
            self.nonconformity_scores = torch.cat([self.nonconformity_scores, y_calibrate - y_preds], dim=0)

        # update the length of the nonconformity scores
        self.len_nonconformity_score = len(self.nonconformity_scores)
    
    def load_models(self, x_val, y_val, naive_val, names, save_path=False, load_paths=False):
        """
        load pretrained models from the save path

        Input Arguments:
        x_val: torch.Tensor: x validation data

        y_val: torch.Tensor: y validation data

        naive_val: torch.Tensor: naive model validation data 

        names: list: list of names for the models

        save_path: str: path to the save folder

        load_paths: list: list of paths to the models
        """

        # load the models
        self.fitted_models=[]
        for name, load_path in zip(names, load_paths):
            model = self.modelclass(
                model=self.model,
                config=self.config,
                x_val=x_val,
                y_val=y_val,
                naive_val=naive_val,
                name=name,
                device=self.device,
                save_path=save_path,
                scale1s=self.scale1s,
                scale2s=self.scale2s,
            )

            model.load_model(load_path)
            self.fitted_models.append(model)


    def fit_model(self, x_train, y_train, x_val, y_val, naive_val, save_path=False):
        """
        Fit the quantile forecasting model

        Input Arguments:
        x_train: torch.Tensor or list: x training data

        y_train: torch.Tensor or list: y training data

        x_val: torch.Tensor: x validation data

        y_val: torch.Tensor: y validation data

        naive_val: torch.Tensor: naive model validation data

        save_path: str: path to the save folder

        """
        
        if type(x_train) != list:
            x_train = [x_train]

        if type(y_train) != list:
            y_train = [y_train]
        
        for i, (x_train_i, y_train_i) in enumerate(zip(x_train, y_train)):
            
            model = self.modelclass(
                model=self.model,
                config=self.config,
                x_val=x_val,
                y_val=y_val, 
                naive_val=naive_val,
                name=f'{self.model_name}_fold_{i}',
                device=self.device,
                save_path=save_path,
                scale1s=self.scale1s,
                scale2s=self.scale2s,
                seed=52
            )

            model.fit(x_train_i, y_train_i)
            self.fitted_models.append(model)

    def predict(self, x_test, batch_size=256, model='all'):
        """
        Predict the probabilistic model

        Input Arguments:
        x_test: torch.Tensor: x test data

        batch_size: int: batch size for the prediction

        model: int or str: index of the model or 'all'
        """

        if model == 'all':
            y_pred = torch.stack([model.predict(x_test, batch_size=batch_size) for model in self.fitted_models])
            y_pred = y_pred.mean(dim=0, dtype=float)
        elif type(model) == int:
            y_pred = self.fitted_models[model].predict(x_test, batch_size=batch_size)
        else:
            raise ValueError('model must be "all" or an integer')
    
        return y_pred


    def _update_nonconformity_measures(self, nonconformity_score):
        """
        Update the nonconformity measures

        Input Arguments:
        nonconformity_score: torch.Tensor: nonconformity scores
        """
        
        if len(nonconformity_score.shape) == 2:
            nonconformity_score = nonconformity_score.unsqueeze(0)

        self.nonconformity_scores = torch.cat([self.nonconformity_scores, nonconformity_score], dim=0)


    def predict_CQR(self, ind_test, x_test, batch_size=256, model='all', update_nonconformity_scores=False):
        """
        Predict the CQR model

        Input Arguments:
        ind_test: torch.Tensor: indices of the test data

        x_test: torch.Tensor: x test data

        batch_size: int: batch size for the prediction

        model: int or str: index of the model or 'all'

        update_nonconformity_scores: bool: if True, update the nonconformity scores

        """

        if update_nonconformity_scores is False:
            y_preds = self.predict(x_test, batch_size=batch_size, model=model)
            fsr = (self.len_nonconformity_score+1)/(self.len_nonconformity_score)
            y_pred_calibrated = torch.ones_like(y_preds)

            for i, q in enumerate(self.quantiles):
                residuals_q = self.nonconformity_scores[:,:,i]
                d_i = torch.quantile(residuals_q, q*fsr, dim=0).unsqueeze(0).expand_as(y_preds[:,:,i])
                y_pred_calibrated[:,:,i] = y_preds[:,:,i] + d_i
            return y_pred_calibrated
        
        else:
            y_test = update_nonconformity_scores
            nonconformity_scores_queue = []
            last_inds = []
            y_preds = []

            for x, ind, yt in zip(x_test, ind_test, y_test):

                # update nonconformity scores
                first_ind = ind[0]
                for i, moment in enumerate(last_inds):
                    if moment < first_ind:

                        # update nonconformity scores
                        self._update_nonconformity_measures(nonconformity_scores_queue[i])

                        # remove from queue
                        last_inds.pop(i)
                        nonconformity_scores_queue.pop(i)

                x = [x]
                
                # compute prediction
                y_pred = self.predict(x, batch_size=batch_size, model=model)
                yt = yt.reshape(1,-1,1).expand_as(y_pred)

                # compute nonconformity scores
                nonconformity_scores = yt - y_pred
                last_inds.append(ind[-1])
                nonconformity_scores_queue.append(nonconformity_scores)

                # compute calibrated prediction
                y_pred_calibrated = torch.ones_like(y_pred)

                # updat the quantile prediction
                for i, q in enumerate(self.quantiles):
                    residuals_q = self.nonconformity_scores[:,:,i]
                    d_i = torch.quantile(residuals_q, q*(self.len_nonconformity_score+1)/(self.len_nonconformity_score), dim=0)\
                            .unsqueeze(0).expand_as(y_pred[:,:,i])
                    y_pred_calibrated[:,:,i] = y_pred[:,:,i] + d_i
                    
                y_preds.append(y_pred_calibrated)

            y_preds = torch.cat(y_preds, dim=0)
            return y_preds
