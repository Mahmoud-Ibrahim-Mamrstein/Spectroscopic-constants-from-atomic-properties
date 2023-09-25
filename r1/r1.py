#!/usr/bin/env python
# coding: utf-8

# # Model r1
# 
# Model r1 is the same as in Liu et al.[[1]](#1). A GPR model with a Matern 1/2 kernel and with groups and periods of the constituent atoms as features. We explicitly express the model's prior mean functions as linear functions in the groups and periods of the diatomic molecules' constituent atoms.
# 
# $m_{r1-r2} = \beta_0^{r1-r2}+\beta_{1}^{r1-r2}(p_1+p_2) + \beta_{2}^{r1-r2}(g_1+g_2)$
# 
# where  $\beta_k^{r1-r2}$, $k \in \{0,1,2\}$ are the linear coefficients of  $m_{r1-r2}$.
# 
# ## References
# <a id="1">[1]</a> 
# X. Liu, S. Truppe, G. Meijer and J. Pérez-Ríos, Journal of
# Cheminformatics, 2020, 12, 31.

# ## 1.Import libraries and objects

# In[ ]:


import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import sklearn as sk
from matplotlib import pyplot
import time
import math
from math import sqrt
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error


# # 2. Inheritance

# ## Increasing the maximum number of iterations for the optmizer of the Gaussianprocesses object

# In[ ]:


class MyGPR(GaussianProcessRegressor): #MyGPR(GaussianProcessRegressor) class specify the maximum number of iterations, tolerance, and optimizer explicitly in the sklearn.gaussian_process GaussianProcessRegressor object 
    def __init__(self, *args, max_iter=1000000, gtol=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter #maximum number of iterations
        self._gtol = gtol #tolerance

    def _constrained_optimization(self, obj_func, initial_theta, bounds): 
        if self.optimizer == "fmin_l_bfgs_b": #optmizer 
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


# ## 3.3 Function that perfroms the MC-CV splits, train the GPR and make predictions

# In[ ]:


def ml_model(data,strata,test_size,features,prior_features,logtarget,target,nu,normalize_y,n_splits=1000): #function used for implementing the MC-CV GPR model
    r_y_train_preds={} # Initiate a dictionary to store training predictions
    r_y_test_preds={} # Initiate a dictionary to store testing predictions
    r_train_stds={} # Initiate a dictionary to store training standard deviations
    r_test_stds={} # Initiate a dictionary to store testing standard deviations
    trval={} #intiate a dictionary to store optimized kernels and scores
    start_time = time.time() #Timing the algorithm
    RMSE=[] # Intiate a list to store the test RMSE of all MC-CV steps
    RMSLE=[] # Intiate a list to store the test RMSLE of all MC-CV steps
    MAE=[] # Intiate a list to store the test MAE of all MC-CV steps
    R=[] # Intiate a list to store the test normalized RMSE % of all MC-CV steps
    Train_RMSE=[] # Intiate a list to store the train RMSE of all MC-CV steps
    Train_RMSLE=[] # Intiate a list to store the train RMSLE of all MC-CV steps
    Train_MAE=[] # Intiate a list to store the train MAE of all MC-CV steps
    Train_R=[] # Intiate a list to store the train normalized RMSE % of all MC-CV steps
    mean_std=[] # Intiate a list to store the mean test std of all MC-CV steps
    train=[] # Intiate a list of lists to store molecules used for training in each split 
    test=[] # Intiate a list of lists to store molecules used for testing in each split
    mcs = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,random_state=42) # Using stratified shuffle split object from sklearn for the MC-CV scheme 
    s=0

    for train_index, test_index in mcs.split(data,strata):
        # The naming 'testing' is used insted of 'validatining' since in each MC-CV step the validation set is used to siulate the testing stage
        re_train_set1 = data.iloc[train_index] # The dataframe's training rows returend from mcs.split(data,strata)
        re_train_set1['ind']=train_index #The dataframe's training rows' indicies returend from mcs.split(data,strata)
        re_test_set1 = data.iloc[test_index] # The dataframe's testing rows returend from mcs.split(data,strata)
        re_test_set1['ind']=test_index #The dataframe's testing rows' indicies returend from mcs.split(data,strata)
        
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
        
        
        reg = LinearRegression().fit(re_train_set[prior_features], re_train_set[logtarget]) #Liner regression model to fix the constatns coefficients of the prirori mean function in each MC-CV step
        
        re_train_set['prior_mean']=reg.predict(re_train_set[prior_features])
        re_test_set['prior_mean']=reg.predict(re_test_set[prior_features])
        
        
        prior_mean='prior_mean'
        signal_variance=(re_train_set[logtarget].var()) #Intiate constant cooefcient of the Matern kernel function 
        length_scale=(re_train_set[features].std()).mean() #Intiate length scale of the Matern kernel function 
        gpr = MyGPR(kernel=ConstantKernel(constant_value=signal_variance)*Matern(length_scale=length_scale, nu=nu)+WhiteKernel(noise_level=re_train_set[target].std()/np.sqrt(2),noise_level_bounds=(10**-15,1)),n_restarts_optimizer=20,normalize_y=normalize_y,random_state=42) #Using MYGPR class with the matern Kernel with multiplicative constant and additive white noise kernel as defined in the manuscript
        gpr.fit(re_train_set[features], re_train_set[logtarget]-re_train_set[prior_mean]) # Optmizing the kernel parameters using the fitting data (the target is offset by the prior mean)


        r_y_train_pred_log,r_std_train=gpr.predict(re_train_set[features], return_std=True) #train predictions, and train standard deviations 
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
        mean_std.append(np.array(r_std_test).mean()) #calculating mean of the standard deviations returned from gpr predictions

        trval[str(s)]['mean_std']=mean_std[-1] #mean of the standard deviations returned from gpr predictions of split s

        RMSE.append(np.sqrt(mean_squared_error(re_test_set[target],r_y_test_pred))) #calculating test RMSE of the split and appending it to the test RMSE list 

        trval[str(s)]['RMSE']=RMSE[-1] #RMSE of split s
        
        Train_RMSE.append(np.sqrt(mean_squared_error(re_train_set[target],r_y_train_pred))) #calculating train RMSE of the split and appending it to the Train_RMSE list

        trval[str(s)]['Train_RMSE']=Train_RMSE[-1] #Train RMSE of split s
        
        RMSLE.append(np.sqrt(mean_squared_error(re_test_set[logtarget],r_y_test_pred_log))) #calculating test RMSLE of the split and appending it to the test RMSLE list 

        trval[str(s)]['RMSLE']=RMSLE[-1] #Test RMSLE of split s
        
                
        Train_RMSLE.append(np.sqrt(mean_squared_error(re_train_set[logtarget],r_y_train_pred_log))) #calculating train RMSLE of the split and appending it to the Train_RMSLE list 

        trval[str(s)]['Train_RMSLE']=Train_RMSLE[-1] #Train RMSE of split s

        MAE.append(sum(abs(re_test_set[target]-(r_y_test_pred)))/len(re_test_set[target])) #calculating test MAE of the split and appending it to the test MAE list 

        trval[str(s)]['MAE']=MAE[-1] #Test MAE of split s
        
        Train_MAE.append(sum(abs(re_train_set[target]-(r_y_train_pred)))/len(re_train_set[target])) #calculating train MAE of the split and appending it to the Train_MAE list 

        trval[str(s)]['Train_MAE']=Train_MAE[-1] #Train MAE of split s

        R.append(100*(np.sqrt(mean_squared_error(re_test_set[target],r_y_test_pred)))/((data[target]).max()-(data[target]).min())) #calculating test R of the split and appending it to the test R list 

        trval[str(s)]['R']=R[-1] #Test R of split s

    
        s=s+1 # incrementing the MC-CV split counter
        

        for i in range(len(re_train_set.ind)):
            if re_train_set.ind.tolist()[i] not in r_y_train_preds:   
                r_y_train_preds[re_train_set.ind.tolist()[i]]=[r_y_train_pred[i]] #adding MC-CV train prediction list of a molecule of index 'i' which is not yet in re_train_set dictionary
                r_train_stds[re_train_set.ind.tolist()[i]]=[r_std_train[i]] #adding MC-CV train GPR standard deviation list of a molecule of index 'i' not yet in r_train_stds dictionary
            else:
                r_y_train_preds[re_train_set.ind.tolist()[i]].append(r_y_train_pred[i]) #apeending new MC-CV train prediction to an existing list of predictions of a molecule indexed i in the r_y_train_preds dictionary
                r_train_stds[re_train_set.ind.tolist()[i]].append(r_std_train[i]) #apeending new MC-CV train standard deviation to an existing list of predictions of a molecule indexed i in the r_train_stds dictionary
                
        for i in range(len(re_test_set.ind)):
            if re_test_set.ind.tolist()[i] not in r_y_test_preds:
                r_y_test_preds[re_test_set.ind.tolist()[i]]=[r_y_test_pred[i]] #adding MC-CV test prediction list of a molecule of index 'i' which is not yet in re_test_set dictionary
                r_test_stds[re_test_set.ind.tolist()[i]]=[r_std_test[i]] #adding MC-CV test GPR standard deviation list of a molecule of index 'i' not yet in r_test_stds dictionary
            else:
                r_y_test_preds[re_test_set.ind.tolist()[i]].append(r_y_test_pred[i]) #apeending new MC-CV test prediction to an existing list of predictions of a molecule indexed 'i' in the r_y_test_preds dictionary
                r_test_stds[re_test_set.ind.tolist()[i]].append(r_std_test[i]) #apeending new MC-CV test standard deviation to an existing list of predictions of a molecule indexed 'i' in the r_test_stds dictionary
    end_time = time.time()
    retime=end_time-start_time
    retime # timing the validation stage
    return trval,train,test,mean_std,Train_MAE,Train_RMSE,Train_R,Train_RMSLE,MAE,RMSE,R,RMSLE,r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds


# ## 3.4 Ploting Function

# In[ ]:


def plot_results(df,x,y,target,r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds): #funtion to create scatter plots of training and testing predictions
    re_train_preds=[] # initiating a list to store the average training predictions for each molecule over the MC-CV splits 
    re_train_std=[] # initiating a list to store the average training standard deviations for each molecule over the MC-CV splits
    re_test_preds=[]  # initiating a list to store the average testing predictions for each molecule over the MC-CV splits 
    re_test_std=[] # initiating a list to store the average testing standard deviations for each molecule over the MC-CV splits
    out=[] # intiating a list that stores the indices of the molecules that has never been used in testing (validation) set 
    for index in range(len(df.index)):
            re_train_preds.append(np.array(r_y_train_preds[index]).mean()) 
            re_train_std.append(np.std(np.array(r_y_train_preds[index]))+np.array(r_train_stds[index]).mean())
            re_test_preds.append((np.array(r_y_test_preds[index])).mean())
            re_test_std.append(np.std(np.array(r_y_test_preds[index]))+np.array(r_test_stds[index]).mean())
    fig, ax =pyplot.subplots(figsize=(7,7))
    pyplot.xticks(fontsize=16)
    pyplot.yticks(fontsize=16)
    ax.errorbar(df[target], re_train_preds, yerr=re_train_std, fmt ='o',label='Training set')
    ax.errorbar(df[target], re_test_preds, yerr=re_test_std, fmt ='o',label='Validation set')

    line=df[target].tolist()
    line.append(0)
    line.append(np.ceil(np.array(re_test_preds).max()))
    ax.plot(line,line,'--k')
    pyplot.xticks(ticks=np.linspace(1, 4, num=4))
    pyplot.yticks(ticks=np.linspace(1, 4, num=4))
    pyplot.xlim(np.array(line).min(),np.ceil(np.array(line).max()))
    pyplot.ylim(np.array(line).min(),np.ceil(np.array(line).max()))
    ax.legend(prop={'size': 12})
    pyplot.xlabel(x,fontdict={'size': 16})
    pyplot.ylabel(y,fontdict={'size': 16})
    return re_train_preds,re_train_std,re_test_preds,re_test_std,out,fig,ax


# ## 3.5 A function to repors a statistical summary

# In[ ]:


def results(data_describtion,df,target,re_test_preds,no_molecules,MAE,RMSE,R,handle): # A function that returns a data frame of the final results and scores of the model 
    results={}
    results[data_describtion]={}
    results[data_describtion]['Number of molecules in the whole data set']=no_molecules
    results[data_describtion]['MAE']=str(np.array(MAE).mean())+' (+/-) '+str(round(abs(np.mean(abs(np.mean(np.array(df[target]))-(np.mean(re_test_preds)+np.std(re_test_preds))))-(np.mean(abs(np.mean(np.array(df[target]))-(np.mean(re_test_preds)-np.std(re_test_preds))))))/2,4))
    results[data_describtion]['RMSE']=str(np.array(RMSE).mean())+' (+/-) '+str((abs(np.sqrt(np.mean(np.mean(np.array(df[target]))-(np.mean(re_test_preds)+np.std(re_test_preds)))**2)-(np.sqrt(np.mean((df[target].mean()-(np.mean(re_test_preds)-np.std(re_test_preds))))**2))).round(decimals=4))/2)
    results[data_describtion]['$r%$']=str((np.array(R).mean()))+' (+/-) '+str(round((((abs(np.sqrt(np.mean(np.mean(np.array(df[target]))-(np.mean(re_test_preds)+np.std(re_test_preds)))**2)-(np.sqrt(np.mean((df[target].mean()-(np.mean(re_test_preds)-np.std(re_test_preds))))**2))))/2)*100/(df[target].max()-df[target].min())),4))
    results=pd.DataFrame.from_dict(results) 
    results.to_csv(handle, index=True)  
    return results


# # 4. Main Body

# ## 4.1 Loading data

# In[ ]:


gr,gr_expand=load(handel=r"r1_gr_expand_pred.csv")


# ## 4.2 Stratify data according to the levels of the target values

# In[ ]:


gr_expand=gr_expand[~gr_expand['Molecule'].isin(['XeCl','AgBi','Hg2','HgCl'])] # the molecules XeCl, AgBi , Hg2 and HgCl has been removed due to uncertainties in their experimental spectroscopic constants values 
gr_expand['rcat']=gr_expand['Re (\AA)'] # gr_expand['rcat'] will be used to define strata for the process of stratified sampling
gr_expand_unique=np.unique(gr_expand['rcat']) # gr_expand_unique define the unique values of gr_expand['rcat']
ind=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,309] # indicies used to defined strata for the stratified random sampling 
for i in range(len(ind)-1): 
    gr_expand['rcat'].where((gr_expand['rcat']>gr_expand_unique[ind[i+1]])|(gr_expand['rcat']<=gr_expand_unique[ind[i]]),gr_expand_unique[ind[i]],inplace=True) # stratification according to the levels of the target variables


# ## 4.3 Run MC-CV 

# In[ ]:


trval,train,test,mean_std,Train_MAE,Train_RMSE,Train_R,Train_RMSLE,MAE,RMSE,R,RMSLE,r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds=ml_model(data=gr_expand,strata=gr_expand['rcat'],test_size=31,features=['p1','p2','g1_lan_act','g2_lan_act'],prior_features=['p1','p2','g1_lan_act','g2_lan_act'],target='Re (\AA)',logtarget='Re (\AA)',nu=1/2,normalize_y=False,n_splits=1000) #MC-CV gpr 


# ## 4.4 Plot and save results

# In[ ]:


re_train_preds,re_train_std,re_test_preds,re_test_std,out,fig,ax=plot_results(gr_expand,'True $R_e(\AA)$','Predicted $R_e(\AA)$','Re (\AA)',r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds); # plotting scatter plots 
pyplot.savefig('r1_scatter.svg')
for i in range(len(re_test_preds)):
    if abs(gr_expand['Re (\AA)'].tolist()[i]-re_test_preds[i])<0.1:
        continue
    ax.annotate(gr_expand['Molecule'].tolist()[i], (gr_expand['Re (\AA)'].tolist()[i], re_test_preds[i])) # annotating molecules with 0.1 angstrom absolute error 
pyplot.savefig('r1_scatter_ann.svg') # saving the annotated scatter plot


# In[ ]:


results('r1 model',gr_expand,'Re (\AA)',re_test_preds,314,MAE,RMSE,R,r"stat_summ.csv") # saving results and scores
gr_expand['re_test_preds']=re_test_preds # saving the average testing predictions for each molecule over the MC-CV splits
gr_expand['re_test_std']=re_test_std # saving the average testing standard deviations for each molecule over the MC-CV splits
gr_expand['re_train_preds']=re_train_preds # saving the average training predictions for each molecule over the MC-CV splits
gr_expand['re_train_std']=re_train_std # saving the average training standard deviations for each molecule over the MC-CV splits
gr_expand.to_csv('r1_gr_expand_pred.csv')
split_stat = pd.DataFrame(list(zip(Train_MAE,Train_RMSE,Train_R,MAE,RMSE,R)),columns =['Train_MAE','Train_RMSE','Train_R','MAE','RMSE','R']) # saving final scores for each split
split_stat.to_csv('r1_split_stat.csv') # saving final scores for each split in a CSV file

