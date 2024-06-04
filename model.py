
import numpy as np
from sklearn.linear_model import Ridge

class OLS():
    def __init__(self, 
                 dataset_train, 
                 context_length, 
                 horizon, 
                 instance_norm=True, 
                 individual=False,
                 alpha=0.000001,
                 seed=42,
                 verbose=False,
                 max_train_N=None):
        """
        OLS wrapper to simplify some of the OLS fitting. 
        args:
            dataset_train: training dataset, will be turned into training instances with the numpy sliding_window_view trick
            context_length: features that the linear model will see
            horizon: up until where the linear model will forecast
            instance_norm: switch on or off instance normalisation, which equates to subtracting the mean for a linear model
            individual: determines if separate model should be learned per channel or one shared across channels. Enabled only for 'weather' dataset as in DLinear.
            seed: for repeatability when using SVD solver
            max_train_N: set this if your dataset (N_samples * N_variables) is very large and you want to subsample
        """
        self.context_length = context_length
        self.horizon = horizon
        self.dataset = dataset_train
        self.individual = individual
        self.instance_norm = instance_norm
        #  Disable 'fit_intercept' in Ridge regresion when instance normalization is used. This adjustment is necessary
        #  because the bias (intercept) term is implicitly handled through normalization, specifically by appending the
        #  context standard deviation as a feature to each instance. Refer to Table 1 in paper for details on this setup.
        fit_intercept = False if instance_norm else True
        self.max_train_N = max_train_N
        self.verbose = verbose

        if self.individual:
            self.models = []
            for _ in range(dataset_train.shape[1]):
                self.models.append(Ridge(alpha=alpha,  # This has no appreciable impact for regularisation but is instead set for stability 
                                         fit_intercept=fit_intercept, 
                                         tol=0.00001, 
                                         copy_X=True, 
                                         max_iter=None, 
                                         solver='svd', 
                                         random_state=seed))
        else:
            self.model = Ridge(alpha=alpha,  # This has no appreciable impact for regularisation but is instead set for stability
                               fit_intercept=fit_intercept, 
                               tol=0.00001, 
                               copy_X=True, 
                               max_iter=None, 
                               solver='svd', 
                               random_state=seed)
        
        self.fit_ols_solutions()
    
    def fit_ols_solutions(self):
        """
        Fit the OLS solutions for each series or in a global mode.
        self.dataset.shape = (D, V), where D is the length (in time) and V is the number of variables.
        """
        # Instances are of shape (D_trimmed, V, context+horizon), where D_trimmed is determined by the settings, 
        # see: np.lib.stride_tricks.sliding_window_view
        instances = np.lib.stride_tricks.sliding_window_view(self.dataset, (self.context_length+self.horizon), axis=0)
        

        if self.instance_norm:
            if self.verbose:
                print('Subtracting means')
            context_means = np.mean(instances[:,:,:self.context_length], axis=2, keepdims=True)  # (D_trimmed,V,1)
            instances = instances - context_means
        
        X = instances[:,:,:self.context_length]  # (D_trimmed, V, context)
        y = instances[:,:,self.context_length:]  # (D_trimmed, V, horizon)
        
        if self.instance_norm:
            # Concatenate the standard deviation when doing instance norm
            # This is to account for fit_intercept. See table 1. 
            context_stds = np.sqrt(np.var(instances[:,:,:self.context_length], axis=2, keepdims=True) + 1e-5)
            X = np.concatenate((X, context_stds), axis=2)  # (D_trimmed, V, context+1)
        
        if self.verbose:
            print('Fitting')

        if self.individual:
            for series_idx in range(X.shape[1]):
                if self.verbose:
                    print('\t Fitting in individual mode, series idx {series_idx}')

                X_data = X[:,series_idx,:]
                y_data = y[:,series_idx,:]
                if self.max_train_N is not None and X_data.shape[0]>self.max_train_N:
                    idxs = np.arange(X_data.shape[0])
                    idxs = np.random.choice(idxs, size=self.max_train_N, replace=False)
                    self.models[series_idx].fit(X_data[idxs], y_data[idxs])
                else:
                    self.models[series_idx].fit(X_data, y_data)
        else:
            # Flatten 3D data into 2D data: training instances across all variables for 'global' mode
            X_data = np.reshape(X, (X.shape[0]*X.shape[1], -1))  # (D_trimmed*V, context+(1 if instance_norm else 0))
            y_data = np.reshape(y, (y.shape[0]*y.shape[1], -1))  # (D_trimmed*V, horizon)
            
            # Randomly select some data if this is set (mostly for speed and debugging purposes)
            if self.max_train_N is not None and X_data.shape[0]>self.max_train_N:
                idxs = np.arange(X_data.shape[0])
                idxs = np.random.choice(idxs, size=self.max_train_N, replace=False)
                self.model.fit(X_data[idxs], y_data[idxs])
            else:
                self.model.fit(X_data, y_data)
            self.weight_matrix = self.model.coef_   
            self.bias = self.model.intercept_  
    
    
    def predict(self, X):
        """
        Using the pre-fitted models and context, x, predict to horizon
        """
        D, V = X.shape[0], X.shape[1]
        if self.instance_norm:
            x_mean = np.mean(X, axis=2, keepdims=True)
            X = X - x_mean
            x_std = np.sqrt(np.var(X, axis=2, keepdims=True) + 1e-5)
            X = np.concatenate((X, x_std), axis=2)
            
        
        if self.individual:
            preds = []
            for series_idx in range(X.shape[1]):
                pred_i = self.models[series_idx].predict(X[:,series_idx])
                preds.append(pred_i[:,np.newaxis])
            preds = np.concatenate(preds, axis=1)
        else:
            pred = self.model.predict(X.reshape(D*V, -1))
            preds = pred.reshape(D, V,-1)
        
        
        if self.instance_norm:
            return preds + x_mean # Undo instance norm
        else:
            return preds
        
        
if __name__=='__main__':
    """
    Simple test of OLS solution on ETTh1
    """
    from data.datasets import dataset_selector
    context_length = 720
    datasets_to_test = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
    # datasets_to_test = ['electricity']
    for dataset in datasets_to_test:
        for horizon in [96, 192, 336, 720]:
            dataset_train, dataset_val, dataset_test = dataset_selector(dataset, context_length, horizon)
            model = OLS(dataset_train.data, 
                            context_length, 
                            horizon, 
                            instance_norm=True, 
                            individual=False if dataset!='weather' else True, 
                            verbose=False)
            
            test_instances = np.lib.stride_tricks.sliding_window_view(dataset_test.data, (context_length+horizon), axis=0)
            X = test_instances[:,:,:context_length]
            y = test_instances[:,:,context_length:]
            preds = model.predict(X)
            print(f'{dataset}. Horizon={horizon}. MSE={np.mean((y-preds)**2):0.3f}; MAE={np.mean(np.abs(y-preds)):0.3f}.')
