#!/usr/bin/env python
# coding: utf-8

# # Testing the r1 model
# 
# we test the r1 model against a number of chalenging molecules. We follow a leave one out scheme in testing. That is, all the molecules are used to train the model leaving one molecule out for testing. This process is repeated for each molecule.

# ## 1.Import libraries and objects

# In[ ]:


import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import sklearn as sk
from matplotlib.backends.backend_pdf import PdfPages
import re
import seaborn as sns
from matplotlib import pyplot
import time
import math
from math import sqrt
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error


# # 2. Inheritance

# ## Increasing the maximum number of iterations for the optmizer of the Gaussian processes object

# In[ ]:


class MyGPR(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=1000000, gtol=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter
        self._gtol = gtol

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,tol=self._gtol, options={'maxiter':self._max_iter, 'disp':True})
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
            
        return theta_opt, func_min


# # 3. Functions

# ## 3.1 Load data

# In[ ]:


def load(handel,per_tab=r"peridic.csv"): #Load is a function that takes the handles of the two CSV files containing the full data set (including old and new data) and the data set containing the data from Liu et al. 2021 and returns multiple pandas data frames of the data as defined below
    dfe=pd.read_csv(handel,index_col=None)
    df1=pd.read_csv(per_tab,index_col=None) #Includes information from the periodic table for each element
    dfe= dfe.loc[:, ~dfe.columns.str.contains('^Unnamed')]
    g=dfe
    g.loc[g.atom2=='H','p2']=[1]*len(g.loc[g.atom2=='H']['p2'])
    g.loc[g.atom1=='H','p1']=[1]*len(g.loc[g.atom1=='H']['p1'])
    g.loc[g.atom2=='H','g2_lan_act']=[1]*len(g.loc[g.atom2=='H']['g2_lan_act'])
    g.loc[g.atom1=='H','g1_lan_act']=[1]*len(g.loc[g.atom1=='H']['g1_lan_act'])
    #the 'lan_act' extension to 'g1' and 'g2' indicates that Lanthanides and Actinides are included and both are indicated by group number 3.
    
    #Defining the groups and periods of the Hydrogen isotopologues
    g.loc[g.atom2=='D','p2']=[1]*len(g.loc[g.atom2=='D']['p2'])
    g.loc[g.atom1=='D','p1']=[1]*len(g.loc[g.atom1=='D']['p1'])
    g.loc[g.atom2=='T','p2']=[1]*len(g.loc[g.atom2=='T']['p2'])
    g.loc[g.atom1=='T','p1']=[1]*len(g.loc[g.atom1=='T']['p1'])
    g.loc[g.atom2=='D','g2']=[1]*len(g.loc[g.atom2=='D']['g2'])
    g.loc[g.atom1=='D','g1']=[1]*len(g.loc[g.atom1=='D']['g1'])
    g.loc[g.atom2=='T','g2']=[1]*len(g.loc[g.atom2=='T']['g2'])
    g.loc[g.atom1=='T','g1']=[1]*len(g.loc[g.atom1=='T']['g1'])
    g.loc[g.atom2=='D','g2_lan_act']=[1]*len(g.loc[g.atom2=='D']['g2_lan_act'])  
    g.loc[g.atom1=='D','g1_lan_act']=[1]*len(g.loc[g.atom1=='D']['g1_lan_act'])
    g.loc[g.atom2=='T','g2_lan_act']=[1]*len(g.loc[g.atom2=='T']['g2_lan_act'])
    g.loc[g.atom1=='T','g1_lan_act']=[1]*len(g.loc[g.atom1=='T']['g1_lan_act']) 
    g= g.loc[:, ~g.columns.str.contains('^Unnamed')]
    g_dict=g.to_dict(orient='list')
    #creating different pandas dataframes for different purposes
    gr=g[g["Re (\AA)"].isna()==False] #gr only contains molecules that have R_e available 
    # permuting the properties of atoms 1 and 2 in the diatomic molecules as described in Liu et al., 2021 and in the manuscript, to create expanded data frames containing both A-B and B-A molecules
    reverse=['A1','A2','g1','g2','p1','p2','g1_lan_act','g2_lan_act','atom1','atom2','type1','type2']
    for key,value in g_dict.items():
        if key in reverse:
            continue 
        else:
            g_dict[key]=value+value
    s=0        
    for i in range(len(reverse)):
            if s==len(reverse):
                break
            A=g_dict[reverse[s]]+g_dict[reverse[s+1]]
            B=g_dict[reverse[s+1]]+g_dict[reverse[s]]
            g_dict[reverse[s]]=A
            g_dict[reverse[s+1]]=B
            s=s+2
    #the extension '-expand' indicates the inclusion of both A-B and B-A variations of the diatomic molecule in the dataframe 
    g_expand=pd.DataFrame.from_dict(g_dict, orient='columns')
    g_expand.drop_duplicates(subset=['A1','A2','g1','g2','p1','p2','g1_lan_act','g2_lan_act','atom1','atom2','type1','type2'], keep='first', inplace=True, ignore_index=False)
    gr_expand=g_expand[g_expand["Re (\AA)"].isna()==False]

    return gr,gr_expand


# ## 3.3 Function that perfroms the loo splits, train the GPR and make predictions

# In[ ]:


def ml_model(data,features,prior_features,logtarget,target,nu,normalize_y): #function used for implementing the leave one out GPR model for testing
    r_y_train_preds={} # Initiate a dictionary to store training predictions
    r_y_test_preds={} # Initiate a dictionary to store testing predictions
    r_train_stds={} # Initiate a dictionary to store training standard deviations
    r_test_stds={} # Initiate a dictionary to store testing standard deviations
    trval={} #intiate a dictionary to store optimized kernels and scores
    start_time = time.time() #Timing the algorithm
    RMSE=[] # Intiate a list to store the test RMSE of all loo steps
    RMSLE=[] # Intiate a list to store the test RMSLE of all loo steps
    MAE=[] # Intiate a list to store the test MAE of all loo steps
    R=[] # Intiate a list to store the test normalized RMSE % of all loo steps
    Train_RMSE=[] # Intiate a list to store the train RMSE of all loo steps
    Train_RMSLE=[] # Intiate a list to store the train RMSLE of all loo steps
    Train_MAE=[] # Intiate a list to store the train MAE of all loo steps
    Train_R=[] # Intiate a list to store the train normalized RMSE % of all loo steps
    mean_std=[] # Intiate a list to store the mean test std of all loo steps
    train=[]
    test=[] 

    loo = LeaveOneOut() # Using the leave one out object from sklearn for final testing
    
    s=0
    
    for train_index, test_index in loo.split(data): # leave one out is used for testing. That is the whole data set is used in the kernel to make one prediction
        
        re_train_set1 = data.iloc[train_index] # The dataframe's training rows returend from loo.split(data)
        re_train_set1['ind']=train_index #The dataframe's training rows' indicies returend from loo.split(data)
        re_test_set1 = data.iloc[test_index] # The dataframe's testing rows returend from loo.split(data)
        re_test_set1['ind']=test_index #The dataframe's testing rows' indicies returend from loo.split(data)
        
        
        re_train_set=re_train_set1[~re_train_set1['Molecule'].isin(re_test_set1['Molecule'].tolist())] #Removing A-B molecules from the training set if their mirror molecules (B-A molecules) are in the testing set
        re_test_set=pd.concat([re_test_set1,re_train_set1[re_train_set1['Molecule'].isin(re_test_set1['Molecule'].tolist())]]) #Placing miror molecules from the training set in the testing set so that A-B and B-A moleculesa re both in the testing set
        
        for i in re_train_set['Molecule'].isin([re_test_set['Molecule']]):
            if i ==True:
                print('Warning: A molecule in the test set is aslo in the training set')

        train.append(re_train_set['Molecule'])
        if (re_test_set['Molecule'].tolist()) in test:
            break

        test.append(re_test_set['Molecule'].tolist())
        



        trval[str(s)]={} # intiate a dictionary that stores the three parameters values after optimization for each split s
        
        if re_test_set['Molecule'].tolist()[0] not in ['MoC','NbC','NiC','NiO','NiS','PbI','PdC','RuC','SnI','UO','WC','YC','ZnBr','ZnCl','WO','ZnI','ZnF','HCl','DCl','CrC','CoO','IrSi','UF','ZrC']:
            continue # only make prediction is the molecule is in the following list ['MoC','NbC','NiC','NiO','NiS','PbI','PdC','RuC','SnI','UO','WC','YC','ZnBr','ZnCl','WO','ZnI','ZnF','HCl','DCl','CrC','CoO','IrSi','UF','ZrC']
            
        reg = LinearRegression().fit(re_train_set[prior_features], re_train_set[logtarget]) #Liner regression model to fix the constatns coefficients of the prirori mean function in each loo step
        
        re_train_set['prior_mean']=reg.predict(re_train_set[prior_features])
        re_test_set['prior_mean']=reg.predict(re_test_set[prior_features])
        
        prior_mean='prior_mean'
        signal_variance=(re_train_set[logtarget].var()) #Intiate constant cooefcient of the Matern kernel function
        length_scale=(re_train_set[features].std()).mean() #Intiate length scale of the Matern kernel function
        gpr = MyGPR(kernel=ConstantKernel(constant_value=signal_variance)*Matern(length_scale=length_scale, nu=nu)+WhiteKernel(noise_level=re_train_set[target].std()/np.sqrt(2),noise_level_bounds=(10**-15,1)),n_restarts_optimizer=20,normalize_y=normalize_y,random_state=42) #Using MYGPR class with the matern Kernel with multiplicative constant and additive white noise kernel as defined in the manuscript
        gpr.fit(re_train_set[features], re_train_set[logtarget]-re_train_set[prior_mean]) # Optmizing the kernel parameters using the fitting data (the target is offset by the prior mean)


        r_y_train_pred_log,r_std_train=gpr.predict(re_train_set[features], return_std=True)  #train predictions, and train standard deviations
        r_y_test_pred_log,r_std_test=gpr.predict(re_test_set[features], return_std=True) #test predictions, and train standard deviations
        
        r_y_train_pred_log=r_y_train_pred_log+np.array(re_train_set[prior_mean]) #adding the prior mean back
        r_y_test_pred_log=r_y_test_pred_log+np.array(re_test_set[prior_mean]) #adding the prior mean back
        
        r_y_train_pred=r_y_train_pred_log #log transformation was not used in predicting R_e
        r_y_test_pred=r_y_test_pred_log #log transformation was not used in predicting R_e

        
        for m in range(len(r_y_test_pred)):
            if r_y_test_pred[m]<0:
                print('negative result') #indicates negative results if any 
        r_y_test_pred=(np.array(r_y_test_pred))
        for m in range(len(r_y_test_pred)):
            if r_y_test_pred[m]<0:
                print('negative result') #indicates negative results if any 
        
        
        for  mol in  re_test_set['Molecule'].tolist():
            test.append(mol)
        
        
        mean_std.append(np.array(r_std_test).mean())  #calculating mean of the standard deviations returned from gpr predictions
        
        trval[str(s)]['mean_std']=mean_std[-1] # mean of the standard deviations returned from gpr predictions of split s

        RMSE.append(np.sqrt(mean_squared_error(re_test_set[target],r_y_test_pred))) #calculating test RMSE of the split and appending it to the Train_RMSE list

        trval[str(s)]['RMSE']=RMSE[-1] #RMSE of split s
        
        Train_RMSE.append(np.sqrt(mean_squared_error(re_train_set[target],r_y_train_pred)))  #calculating train RMSE of the split and appending it to the Train_RMSE list

        trval[str(s)]['Train_RMSE']=Train_RMSE[-1] #Train RMSE of split s
        
        RMSLE.append(np.sqrt(mean_squared_error(re_test_set[logtarget],r_y_test_pred_log))) #calculating test RMSLE of the split and appending it to the test RMSLE list

        trval[str(s)]['RMSLE']=RMSLE[-1] #Test RMSLE of split s
        
                
        Train_RMSLE.append(np.sqrt(mean_squared_error(re_train_set[logtarget],r_y_train_pred_log))) #calculating train RMSLE of the split and appending it to the Train_RMSLE list 

        trval[str(s)]['Train_RMSLE']=Train_RMSLE[-1] #Train RMSLE of split s

        MAE.append(sum(abs(re_test_set[target]-(r_y_test_pred)))/len(re_test_set[target])) #calculating test MAE of the split and appending it to the test MAE list

        trval[str(s)]['MAE']=MAE[-1] #Train MAE of split s
        
        Train_MAE.append(sum(abs(re_train_set[target]-(r_y_train_pred)))/len(re_train_set[target])) #calculating train MAE of the split and appending it to the Train_MAE list

        trval[str(s)]['Train_MAE']=Train_MAE[-1] #Train MAE of split s

        R.append(100*(np.sqrt(mean_squared_error(re_test_set[target],r_y_test_pred)))/((data[target]).max()-(data[target]).min())) #calculating test R of the split and appending it to the test R list 

        trval[str(s)]['R']=R[-1] #Test R of split s


    
        s=s+1 # incrementing the loo split counter
        

        for i in range(len(re_train_set.ind)):
            if re_train_set.ind.tolist()[i] not in r_y_train_preds:   
                r_y_train_preds[re_train_set.ind.tolist()[i]]=[r_y_train_pred[i]] #adding loo train prediction list of a molecule of index 'i' which is not yet in re_train_set dictionary
                r_train_stds[re_train_set.ind.tolist()[i]]=[r_std_train[i]] #adding loo train GPR standard deviation list of a molecule of index 'i' not yet in r_train_stds dictionary
            else:
                r_y_train_preds[re_train_set.ind.tolist()[i]].append(r_y_train_pred[i])  #apeending new loo train prediction to an existing list of predictions of a molecule indexed i in the r_y_train_preds dictionary
                r_train_stds[re_train_set.ind.tolist()[i]].append(r_std_train[i]) #apeending new loo train standard deviation to an existing list of predictions of a molecule indexed i in the r_train_stds dictionary
                
        for i in range(len(re_test_set.ind)):
            if re_test_set.ind.tolist()[i] not in r_y_test_preds:
                r_y_test_preds[re_test_set.ind.tolist()[i]]=[r_y_test_pred[i]] #adding loo test prediction list of a molecule of index 'i' which is not yet in re_test_set dictionary
                r_test_stds[re_test_set.ind.tolist()[i]]=[r_std_test[i]] #adding loo test GPR standard deviation list of a molecule of index 'i' not yet in r_test_stds dictionary
            else:
                r_y_test_preds[re_test_set.ind.tolist()[i]].append(r_y_test_pred[i]) #apeending new loo test prediction to an existing list of predictions of a molecule indexed 'i' in the r_y_test_preds dictionary
                r_test_stds[re_test_set.ind.tolist()[i]].append(r_std_test[i]) #apeending new loo test standard deviation to an existing list of predictions of a molecule indexed 'i' in the r_test_stds dictionary
    end_time = time.time()
    retime=end_time-start_time
    retime # timing the validation stage
    return trval,train,test,mean_std,Train_MAE,Train_RMSE,Train_R,Train_RMSLE,MAE,RMSE,R,RMSLE,r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds        


# # 4. Main Body

# ## 4.1 Loading data

# In[ ]:


gr,gr_expand=load(handel=r"r1_gr_expand_test.csv")


# In[ ]:


gr_expand=gr_expand[~gr_expand['Molecule'].isin(['AgBi','Hg2','XeCl','HgCl','HgBr',"HgI"])] # Removing moleules with uncertain experimental values of their spectroscopic constnants
gr_expand['mu^(1/2)']=(np.sqrt(gr_expand['Reduced mass'])) # The square root of the reduced mass is used as a feature


# ## 4.2 Making predictions

# In[ ]:


trval,train,test,mean_std,Train_MAE,Train_RMSE,Train_R,Train_RMSLE,MAE,RMSE,R,RMSLE,r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds=ml_model(data=gr_expand,features=['p1','p2','g1_lan_act','g2_lan_act'],prior_features=['p1','p2','g1_lan_act','g2_lan_act'],target='Re (\AA)',logtarget='Re (\AA)',nu=1/2,normalize_y=False)


# ## 4.3 Saving the results

# In[ ]:


test_molecules=gr_expand[gr_expand['Molecule'].isin(['MoC','NbC','NiC','NiO','NiS','PbI','PdC','RuC','SnI','UO','WC','YC','ZnBr','ZnCl','WO','ZnI','ZnF','HCl','DCl','CrC','CoO','IrSi','UF','ZrC'])].drop_duplicates(subset='Molecule')['Molecule'].tolist()
true_values=gr_expand[gr_expand['Molecule'].isin(['MoC','NbC','NiC','NiO','NiS','PbI','PdC','RuC','SnI','UO','WC','YC','ZnBr','ZnCl','WO','ZnI','ZnF','HCl','DCl','CrC','CoO','IrSi','UF','ZrC'])].drop_duplicates(subset='Molecule')['Re (\AA)'].tolist()
re_test_preds=[]
re_test_std=[]
out=[]
for index in r_y_test_preds:
        if index >= 344:
               continue
        re_test_preds.append(((r_y_test_preds[index][0])))
        re_test_std.append(((r_test_stds[index][0])))
testing_results = pd.DataFrame(list(zip(test_molecules, true_values,re_test_preds,re_test_std)), columns =['Molecule', 'true $R_e (\AA)$','Predicted $R_e (\AA)$','error bars'])
testing_results.to_csv(r'r1_testing_results.csv')

