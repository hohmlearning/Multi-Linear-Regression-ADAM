# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 20:18:10 2022
@author: Manuel
"""

import numpy as np
'''
import sys
path = r'E:\Eigene Dateien\2022\Project_Git_hub'
if path not in sys.path:
    sys.path.append(path)
'''
from Evaluation_Metric import Metric_regression
    
class Gradient():
    '''
    ADAM (Adaptive Moment Estimation)
    Adam implements momentum, while correcting the momentum with the second order
    moments. Adam decreases the work required for searchibg for the correct
    learning rate. 
    
    Class for calclulation of the weights (theta) and the updating the weights.
    The ADAM is an adaptive algorithm for minimizing or maximizing objective 
    function based first order gradients with exponential decaying. 
    The gradient is updated with the first order moments while correcting with 
    the second order moments.
    
    
    '''
    def __init__(self, dimension):
        '''
        A Maxtrix (dimension_features x 4) is initialized with ceros filled.
        In the 1 dimension the weights theta, the first moment, second moment,
        and the gradient is stored.
        Parameters
        ----------
        dimension : integer
        Returns
        -------
        None.
        '''
        self.name_columns = ['theta', 'first moment', 'second moment', 'gradient']
        self.matrix =  np.zeros([dimension, len(self.name_columns)])
        
    def correct_gradient (self, beta_1, beta_2, epsilon, t):
        '''
        For calculation of the gradient, momentum is applied. This avoids "jumps"
        in the loss function and ensures faster convergence. The gradient is then 
        scaled with the second moment. Both are bias corrected by beta_1**t and
        beta_2**t, respectively.
        https://arxiv.org/abs/1412.6980 (2022|01|02)
        Parameters
        ----------
        beta_1 : float - in range (0,1) default value beta_1 = 0.9
                 exponential decay for the first moments.
        beta_2 : float - in range (0,1) default value beta_1 = 0.99
                 exponential decay for the second moments.
        epsilon : float - avoids division through 0
        t : integer - Bias correction. For each batch t = t + 1
        Returns
        -------
        
        '''
        first_moment = self.matrix[:,1]
        second_moment =  self.matrix[:,2]
        gradient =  self.matrix[:,3]
        
        first_moment = beta_1 * first_moment + (1-beta_1) * gradient
        first_moment = first_moment / (1-np.power(beta_1, t))
        second_moment = beta_2 * second_moment + (1-beta_2) * np.power(gradient, 2)
        second_moment = second_moment / (1-np.power(beta_2, t))
        gradient =  first_moment / (np.power(second_moment, 0.5) + epsilon)
        
        self.matrix[:,1] = first_moment
        self.matrix[:,2] = second_moment
        self.matrix[:,3] = gradient
        
    def update_theta (self, eta):
        '''
        Derived from directional derivation, the gradient is updadet.
        
        Parameters
        ----------
        eta : float - learning rate default 0.001
        Returns
        -------
        None.
        '''
        theta = self.matrix[:,0]
        gradient = self.matrix[:,3]
        theta = theta - eta * gradient
        self.matrix[:,0] = theta 

        
class ADAM(Metric_regression):
    def __init__(self, max_epoch, batch_size=1, eta=0.001, beta_1=0.9, beta_2=0.99, epsilon=1E-8, MSE_epsilon=1E-14):
        '''
        For explanation: https://arxiv.org/abs/1412.6980 (2022|01|02)
        Parameters
        ----------
        max_epoch : integer
        batch_size : integer
        objective : string - 'MIN' or 'MAX' The default is 'MAX'.
        eta : learning rate - The default is 0.001.
        beta_1 : TYPE, optional
            DESCRIPTION. The default is 0.9.
        beta_2 : TYPE, optional
            DESCRIPTION. The default is 0.99.
        epsilon : TYPE, optional
            DESCRIPTION. The default is 1E-8.
        MSE_epsilon : float - Additional criterium for MSE on epoch -  The default is 1E-14.
        Raises
        ------
        Exception
            DESCRIPTION.
        Returns
        -------
        None.
        '''
        np.random.seed(42)
        self.eta = eta
        self.max_epoch = max_epoch
        self.beta_1 = beta_1
        self.beta_2 = beta_1
        self.epsilon = epsilon
        self.MSE_epsilon = MSE_epsilon
        self.epoch = 0
        self.batch_size = batch_size
        self.batch = 0
        self.MSE_epoch = 100
        self.fitted = False
        self.name = 'SGD with Adam'
            
    def set_parameters_ADAM(self, x_data, y_true):
        '''
        If not retraining all paramters are initilized to cero.
        For retraining only epoch and batch is set to cero.
        Parameters
        ----------
        x_data : numpy array (datapoints x dimension) - X-matrix
        y_true : numpy array (datapoints) - prediction
        Returns
        -------
        None.
        '''
        if self.fitted == True:
            self.epoch = 0
            self.batch = 0
        else:
            try:
                self.dimension = x_data.shape[1]
                self.x = x_data
            except (IndexError):
                self.dimension = 1
                self.x = np.expand_dims(x_data, axis=-1)
            self.y_true = y_true   
            
            self.Gradient = Gradient(dimension=self.dimension) 
            self.Gradient_0 = Gradient(dimension=1)
            self.theta = np.zeros(self.dimension)
            self.theta_0 = 0
            self.steps_epoch = x_data.shape[0] // self.batch_size
            self.number_elements =  x_data.shape[0]
            self.ordered_elements = self.shuffle()
            self.fitted = True
        
    def MSE (self, x, y):
        '''
        Calclulated Mean Squared Error with intern thetas.
    
        Parameters
        ----------
        x_data : numpy array (datapoints x dimension) - X-matrix
        y_true : numpy array (datapoints) - prediction
    
        Returns
        -------
        MSE : float
    
        '''
        y_pred = np.dot(x, self.theta) + (np.ones(y.shape) * self.theta_0)
        MSE_ = self.fun_MSE(y, y_pred)
        return (MSE_)

        
    def update_batch(self):
        '''
        Updates batch and epoch. Calculates the MSE after each epoch
        and shuffles the elements after each epoch.
        Returns
        -------
        None.
        '''
        if self.batch+1 < self.steps_epoch:
            self.batch = self.batch + 1
        else:
            self.epoch = self.epoch + 1 
            self.batch = 0
            self.ordered_elements = self.shuffle()
            self.MSE_epoch = self.MSE(self.x, self.y_true)
            if self.epoch % int(self.max_epoch / 10) == 0:
                print('Epoch: {} | MSE_train: {:.2f}'.format(self.epoch, self.MSE_epoch))
        
    def calc_gradient(self, x, y):
        '''
        The gradient of theta and theta_0 is calculated according to
        the loss function MSE.
        Parameters
        ----------
        x_data : numpy array (datapoints x dimension) - X-matrix
        y_true : numpy array (datapoints) - prediction
        Returns
        -------
        None.
        '''        
        theta = self.Gradient.matrix[:,0]
        theta_0 = self.Gradient_0.matrix[:,0]
        y_pred = np.dot(x, theta) + (np.ones(y.shape) * theta_0)
        gradient_theta = -2*np.dot((y - y_pred), x)
        gradient_theta = gradient_theta# / self.steps_epoch
        gradient_theta_0 = -2*(y - y_pred).sum() # / self.steps_epoch
       
        self.Gradient.matrix[:,3] = gradient_theta
        self.Gradient_0.matrix[:,3] = gradient_theta_0
    
    def shuffle(self):
        '''
        For faster and non biased convergence, the dataset is shufled using 
        the indices of X_train, y_train.
        Returns
        -------
        None.
        '''
        n_shufled = np.random.permutation(self.number_elements)
        return(n_shufled)
    
    def get_item (self):
        '''
        For each batch, selects the corresponding data from X_train and y_train.
        Returns
        -------
        x_batch : numpy array (datapoints x dimension) - X-matrix
        y_batch : numpy array (datapoints) - prediction
        '''
        start = self.batch * self.batch_size
        end = start + self.batch_size
        n_batch = self.ordered_elements[start:end]
        x_batch = self.x[n_batch]
        y_batch = self.y_true[n_batch]
        return(x_batch, y_batch)
 
    def __call__(self, x_data, y_true):  
        '''
        Calculates Multiple Linear Regression with ADAM while end of epoch is
        not reached and the MSE_epoch has not reached the given difference.
        
        For explanation: https://arxiv.org/abs/1412.6980 (2022|01|02)
        Parameters
        ----------
        x_data : numpy array (datapoints x dimension) - X-matrix
        y_true : numpy array (datapoints) - prediction
        Returns
        -------
        None.
        '''
        self.set_parameters_ADAM(x_data, 
                                y_true)
        self.t = 1
        self.MSE_epoch = 100
        while self.epoch <= self.max_epoch and self.MSE_epoch>self.MSE_epsilon:
            x_batch, y_batch = self.get_item() 
            self.calc_gradient(x_batch, y_batch)
            
            self.Gradient.correct_gradient(beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon, t=self.t)
            self.Gradient_0.correct_gradient(beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon, t=self.t)
            
            self.Gradient.update_theta (eta=self.eta)
            self.Gradient_0.update_theta (eta=self.eta)

            self.theta = self.Gradient.matrix[:,0]
            self.theta_0 = self.Gradient_0.matrix[:,0]
            self.update_batch()
            self.t = self.t + 1 
        print('#'*50)

class ADAM_learning_rate_decay(ADAM):
    def __init__(self, max_epoch, batch_size=1, eta=0.001, beta_1=0.9, beta_2=0.99, epsilon=1E-8, MSE_epsilon=1E-14, patience=1E4):
        super().__init__(max_epoch, batch_size, eta, beta_1, beta_2, epsilon, MSE_epsilon)
        self.patience = patience
        self.name = 'SGD with ADAM and learning rate decay'
        
    def set_parameters_ADAM(self, x_data, y_true):
        super().set_parameters_ADAM(x_data, y_true)
        self.best_theta = np.zeros(self.dimension)
        self.best_theta_0 = np.zeros(1)
        self.best_MSE = self.MSE(self.x, self.y_true)
        self.epoch_best_model = 0
        
    def update_batch(self):
        '''
        Updates batch and epoch. Calculates the MSE after each epoch
        and shuffles the elements after each epoch.
        Returns
        -------
        None.
        '''
        if type(self.X_val) == str:
            if self.batch+1 < self.steps_epoch:
                self.batch = self.batch + 1
            else:
                self.epoch = self.epoch + 1 
                self.batch = 0
                self.ordered_elements = self.shuffle()
                self.MSE_train_epoch = self.MSE(self.x, self.y_true)
                self.epoch_best_model = self.epoch_best_model + 1
                if self.epoch % int(self.max_epoch / 10) == 0:
                    print('Epoch: {} | MSE_train: {:.2f}'.format(self.epoch, self.MSE_train_epoch))
                
                if self.MSE_train_epoch < self.best_MSE:
                    self.best_theta = self.Gradient.matrix[:,0]
                    self.best_theta_0 =self.Gradient_0.matrix[:,0]
                    self.best_MSE = self.MSE_train_epoch
                    self.epoch_best_model = 0
                    
        else:
            if self.batch+1 < self.steps_epoch:
                self.batch = self.batch + 1
            else:
                self.epoch = self.epoch + 1 
                self.batch = 0
                self.ordered_elements = self.shuffle()
                self.MSE_train_epoch = self.MSE(self.x, self.y_true)
                self.MSE_val = self.MSE(self.X_val, self.y_val)
                self.epoch_best_model = self.epoch_best_model + 1
                if self.epoch % int(self.max_epoch / 10) == 0:
                    print('Epoch: {} | MSE_train: {:.2f} | MSE_val: {:.2f}'.format(self.epoch, self.MSE_train_epoch, self.MSE_val))
                    
                
                if self.MSE_val < self.best_MSE:
                    self.best_theta = self.Gradient.matrix[:,0]
                    self.best_theta_0 =self.Gradient_0.matrix[:,0]
                    self.best_MSE = self.MSE_val
                    self.epoch_best_model = 0
                
    def __call__(self, x_data, y_true, X_val='Not given', y_val='Not given'):  
        '''
        Calculates Multiple Linear Regression with ADAM while end of epoch is
        not reached and the MSE_epoch has not reached the given difference.
        
        For explanation: https://arxiv.org/abs/1412.6980 (2022|01|02)
        Parameters
        ----------
        x_data : numpy array (datapoints x dimension) - X-matrix
        y_true : numpy array (datapoints) - prediction
        Returns
        -------
        None.
        '''
        if type(X_val)==str:
            self.X_val = 'Not given'
        else:
            self.X_val = X_val
            self.y_val = y_val
            
        self.set_parameters_ADAM(x_data, 
                                y_true)
        self.t = 1
        while self.epoch <= self.max_epoch:
            x_batch, y_batch = self.get_item() 
            self.calc_gradient(x_batch, y_batch)
            
            self.Gradient.correct_gradient(beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon, t=self.t)
            self.Gradient_0.correct_gradient(beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon, t=self.t)
            
            self.Gradient.update_theta (eta=self.eta)
            self.Gradient_0.update_theta (eta=self.eta)

            self.theta = self.Gradient.matrix[:,0]
            self.theta_0 = self.Gradient_0.matrix[:,0]
            self.update_batch()
            self.t = self.t + 1 
            
            if self.epoch_best_model < self.patience :
                if self.eta < 1E-5:
                    break 
            else:
                self.eta = self.eta / np.power(10, 0.5)
                self.epoch_best_model = 0
                self.epoch = 0
                self.batch = 0
                print('New learning rate: {:.2g}'.format(self.eta))
                
        print('#'*50)
            
class ADAM_learning_rate_decay_full_train(ADAM_learning_rate_decay):     
    def __init__(self, max_epoch, batch_size=1, eta=0.001, beta_1=0.9, beta_2=0.99, epsilon=1E-8, MSE_epsilon=1E-14, patience=1E4):
        super().__init__(max_epoch, batch_size, eta, beta_1, beta_2, epsilon, MSE_epsilon, patience)
        self.start_eta = eta
        self.name = 'SGD with ADAM and learning rate decay and full training'
        
    def set_parameters_ADAM(self, x_data, y_true):
        if self.fitted == True:
            self.epoch = 0
            self.batch = 0
        else:
            super().set_parameters_ADAM(x_data, y_true)

    def update_batch(self):
        '''
        Updates batch and epoch. Calculates the MSE after each epoch
        and shuffles the elements after each epoch.
        Returns
        -------
        None.
        '''
       
        if self.batch+1 < self.steps_epoch:
            self.batch = self.batch + 1
        else:
            self.epoch = self.epoch + 1 
            self.batch = 0
            self.ordered_elements = self.shuffle()
            self.MSE_train_epoch = self.MSE(self.x, self.y_true)
            self.MSE_val = self.MSE(self.X_val, self.y_val)
            self.epoch_best_model = self.epoch_best_model + 1
            if self.epoch % int(self.max_epoch / 10) == 0:
                print('Epoch: {} | MSE_train: {:.2f} | MSE_val: {:.2f}'.format(self.epoch, self.MSE_train_epoch, self.MSE_val))
            
            if self.MSE_val < self.best_MSE:
                self.best_theta = self.Gradient.matrix[:,0]
                self.best_theta_0 =self.Gradient_0.matrix[:,0]
                self.best_MSE = self.MSE_val
                self.epoch_best_model = 0
                self.best_eta = self.eta
                    
    def __call__(self, x_data, y_true):  
        '''
        Calculates Multiple Linear Regression with ADAM while end of epoch is
        not reached and the MSE_epoch has not reached the given difference.
        
        Splits the data set internal in (X_sub_train, y_sub_train) and 
        (X_val, y_val). Then, performs ADAM with learning rate decay.
        Finally, trains with (X_sub_train + X_val, y_sub_train + y_val)
        till MSE(X_val, y_val) is reached or maximum epochs is reached
        (self.max_epoch * 100):
        
        For explanation: https://arxiv.org/abs/1412.6980 (2022|01|02)
        Parameters
        ----------
        x_data : numpy array (datapoints x dimension) - X-matrix
        y_true : numpy array (datapoints) - prediction
        Returns
        -------
        None.
        '''
        n_split = int(y_true.shape[0] * 0.8)
         
        self.X_val = x_data[n_split:, :]
        self.y_val = y_true[n_split:]
        
        self.X_sub_train = x_data[:n_split, :]
        self.y_sub_train = y_true[:n_split]
            
        self.set_parameters_ADAM(self.X_sub_train, 
                                self.y_sub_train)
        self.t = 1
        self.best_eta = self.eta
        train_flag = True
        while self.epoch <= self.max_epoch and train_flag == True:
            x_batch, y_batch = self.get_item() 
            self.calc_gradient(x_batch, y_batch)
            
            self.Gradient.correct_gradient(beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon, t=self.t)
            self.Gradient_0.correct_gradient(beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon, t=self.t)
            
            self.Gradient.update_theta (eta=self.eta)
            self.Gradient_0.update_theta (eta=self.eta)

            self.theta = self.Gradient.matrix[:,0]
            self.theta_0 = self.Gradient_0.matrix[:,0]
            self.update_batch()
            self.t = self.t + 1 
            
            if self.epoch_best_model < self.patience :
                if self.eta < 1E-5:
                    train_flag = False 
            else:
                self.eta = self.eta / np.power(10, 0.5)
                self.epoch_best_model = 0
                self.epoch = 0
                self.batch = 0
                print('New learning rate: {:.2g}'.format(self.eta))
        
        MSE_sub_train = self.MSE(self.X_sub_train, self.y_sub_train)
        self.x = x_data
        self.y_true = y_true
        self.eta = self.start_eta
        self.epoch_best_model = 0
        print('Start full training!')
        train_flag = True
        while self.best_MSE > MSE_sub_train and self.epoch  < self.max_epoch and train_flag == True:
            x_batch, y_batch = self.get_item() 
            self.calc_gradient(x_batch, y_batch)
            
            self.Gradient.correct_gradient(beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon, t=self.t)
            self.Gradient_0.correct_gradient(beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon, t=self.t)
            
            self.Gradient.update_theta (eta=self.eta)
            self.Gradient_0.update_theta (eta=self.eta)

            self.theta = self.Gradient.matrix[:,0]
            self.theta_0 = self.Gradient_0.matrix[:,0]
            self.update_batch()
            
            if self.epoch_best_model < self.patience :
                if self.eta < 1E-5:
                    train_flag = False 
            else:
                self.eta = self.eta / np.power(10, 0.5)
                self.epoch_best_model = 0
                self.epoch = 0
                self.batch = 0
                print('New learning rate: {:.2g}'.format(self.eta))  
        print('#'*50)
