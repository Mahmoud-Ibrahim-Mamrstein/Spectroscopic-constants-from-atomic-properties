#!/usr/bin/env python
# coding: utf-8

# # Model w4
#  Model w4 uses the periods and the groups of the constituent atoms and the reduced mass of the molecule along with $R_e$ as features in a Matern 3/2 kernels. $R_e$ is included as a feature in the prior mean function as well.
# \begin{equation}
# \label{eq15}
#     \begin{gathered}  
#         m_{w3-w4} = \beta_0^{w3-w4}+\beta_1^{w3-w4}(p_1+p_2) + \beta_2^{w3-w4}(g_1+g_2) \\ + \beta_3^{w3-w4} R_e +\beta_4^{w3-w4} \ln{(\mu^{1/2})},
#     \end{gathered}  
# \end{equation}
# where $\beta_k^{w4}$, $k \in \{0,1,2,3,4\}$ are the linear coefficients

# # 1.Import libraries and objects

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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Product, Matern, WhiteKernel, RBF, DotProduct, ExpSineSquared
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
            #_check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
            
        return theta_opt, func_min


# # 3. Functions

# ## 3.1 Load data

# In[ ]:


def load(handel,old_handel): #Load is a function that takes the handles of the two CSV files containing the full data set (including old and new data) and the data set containing the data from Liu et al. 2021 and returns multiple pandas data frames of the data as defined below
    dfe=pd.read_csv(handel,index_col=None)
    df1=pd.read_csv(r"peridic.csv",index_col=None) #Includes information from the periodic table for each element
    dfe= dfe.loc[:, ~dfe.columns.str.contains('^Unnamed')]
    nul=[np.NaN]*len(dfe.Molecule)
    for char in ['e1','e2']: #creating two columns that take in the number of electrons of the elements compromising the diatomic molecules
        dfe[char]=nul
    for char in df1.Symbol:
        ind1=dfe.loc[dfe['Molecule'].str.contains(r'^'+char+r'\D')].index.values
        ind2=dfe.loc[dfe['Molecule'].str.contains(char+r'$')].index.values
        ind3=dfe.loc[dfe['Molecule'].str.contains(r'^'+char+r'2')].index.values
        dfe.loc[ind1,'e1']=df1[df1.Symbol==char].NumberofElectrons.values[0]
        dfe.loc[ind2,'e2']=df1[df1.Symbol==char].NumberofElectrons.values[0]
        dfe.loc[ind3,'e1']=df1[df1.Symbol==char].NumberofElectrons.values[0]
        dfe.loc[ind3,'e2']=df1[df1.Symbol==char].NumberofElectrons.values[0]
    g=dfe
    g['n1']=g.A1-g.e1 #number of neutrons of element 1 in a diatomic molecule
    g['n2']=g.A2-g.e2 #number of neutrons of element 2 in a diatomic molecule
    g.loc[g.atom2=='H','p2']=[1]*len(g.loc[g.atom2=='H']['p2'])
    g.loc[g.atom1=='H','p1']=[1]*len(g.loc[g.atom1=='H']['p1'])
    g.loc[g.atom2=='H','g2_lan_act']=[1]*len(g.loc[g.atom2=='H']['g2_lan_act'])
    g.loc[g.atom1=='H','g1_lan_act']=[1]*len(g.loc[g.atom1=='H']['g1_lan_act'])
    #the 'lan_act' extension to 'g1' and 'g2' indicates that Lanthanides and Actinides are included and both are indicated by group number 3.

    
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
    g.loc[g.atom2=='D','g2_lan_act_iso']=[0]*len(g.loc[g.atom2=='D']['g2_lan_act_iso']) #including the isotopic character in some atoms through the groups of the perspective elements  
    g.loc[g.atom1=='D','g1_lan_act_iso']=[0]*len(g.loc[g.atom1=='D']['g1_lan_act_iso']) #including the isotopic character in some atoms through the groups of the perspective elements 
    g.loc[g.atom2=='T','g2_lan_act_iso']=[-1]*len(g.loc[g.atom2=='T']['g2_lan_act_iso']) #including the isotopic character in some atoms through the groups of the perspective elements 
    g.loc[g.atom1=='T','g1_lan_act_iso']=[-1]*len(g.loc[g.atom1=='T']['g1_lan_act_iso']) #including the isotopic character in some atoms through the groups of the perspective elements 
    g.loc[g.atom2=='H','g2_lan_act_iso']=[1]*len(g.loc[g.atom2=='H']['g2_lan_act_iso']) #including the isotopic character in some atoms through the groups of the perspective elements 
    g.loc[g.atom1=='H','g1_lan_act_iso']=[1]*len(g.loc[g.atom1=='H']['g1_lan_act_iso']) #including the isotopic character in some atoms through the groups of the perspective elements
    #creating different variations of features and targets 
    g['sum_p']=g['p1']+g['p2']
    g['sum_g']=g.g1_lan_act+g.g2_lan_act
    g['diff_p']=abs(g['p1']-g['p2'])
    g['diff_g']=abs(g['g1_lan_act']-g['g2_lan_act'])
    g['product_p']=g['p1']*g['p2']
    g['product_g']=g['g1_lan_act']*g['g1_lan_act']
    g['Reduced_g']=(g.g1_lan_act*g.g2_lan_act)/(g.g1_lan_act+g.g2_lan_act)
    g['Reduced_p']=(g.p1*g.p2)/(g.p1+g.p2)
    g['g_average']=(g.g1+g.g2)/2
    g['g_average_lan_act']=(g.g1_lan_act+g.g2_lan_act)/2
    g['g_average_lan_act_iso']=(g.g1_lan_act_iso+g.g2_lan_act_iso)/2
    g['Re (\AA)^-1']=1/((g['Re (\AA)']))
    g['Re (au)']=((g['Re (\AA)'])*1.8897259885789)
    g['Re (au)^-1']=1/((g['Re (\AA)'])*1.8897259885789)
    g['Re (au)^2']=((g['Re (\AA)'])*1.8897259885789)**2
    g['Re (au)^-2']=1/g['Re (au)^2']
    g['4*(np.pi**2)*Re (au)']=(4*(np.pi**2)*(g['Re (\AA)'])*1.8897259885789)
    g['4*(np.pi**2)*Re (au)^-1']=1/(4*(np.pi**2)*(g['Re (\AA)'])*1.8897259885789)
    g['omega_e (au)^-2']=1/((g['omega_e (cm^{-1})']*0.0000046)**2)
    g['4*(np.pi**2)*omega_e (au)^-2']=1/((2*(np.pi)*g['omega_e (cm^{-1})']*0.0000046)**2)
    g['omega_e (au)']=(g['omega_e (cm^{-1})']*(0.0000046))
    g['omega_e (au)^2']=((g['omega_e (cm^{-1})']*0.0000046)**2)
    g['4*(np.pi**2)*omega_e (au)^2']=4*(np.pi**2)*((g['omega_e (cm^{-1})']*(0.0000046))**2)
    g['K']=(4*(np.pi**2)*((g['omega_e (cm^{-1})'])**2)*g['Reduced mass'])
    g['sqrt(K)']=np.sqrt(4*(np.pi**2)*((g['omega_e (cm^{-1})'])**2)*g['Reduced mass'])
    g['4*(np.pi**2)*omega_e (au)']=4*(np.pi**2)*((g['omega_e (cm^{-1})']*(0.0000046)))
    g['4*(np.pi**2)*Re (au)*omega_e (au)^2']=g['4*(np.pi**2)*Re (au)']*((g['omega_e (cm^{-1})']*0.0000046)**2)
    g['ve1']=g['g1_lan_act']
    g['ve2']=g['g2_lan_act']
    g['log(D_e)/(R_e^3*Z_1*Z_2)']=np.log((g["D0 (eV)"]*0.037)/((g["Re (au)"]**3)*g.e1*g.e2))
    g.loc[g.g2_lan_act==18,'ve2']=[0]*len(g.loc[g.g2_lan_act==18]['g2_lan_act'])
    g.loc[g.g1_lan_act==18,'ve1']=[0]*len(g.loc[g.g1_lan_act==18]['g1_lan_act'])
    g.loc[g.g2_lan_act==17,'ve2']=[7]*len(g.loc[g.g2_lan_act==17]['g2_lan_act'])
    g.loc[g.g1_lan_act==17,'ve1']=[7]*len(g.loc[g.g1_lan_act==17]['g1_lan_act'])
    g.loc[g.g2_lan_act==16,'ve2']=[6]*len(g.loc[g.g2_lan_act==16]['g2_lan_act'])
    g.loc[g.g1_lan_act==16,'ve1']=[6]*len(g.loc[g.g1_lan_act==16]['g1_lan_act'])
    g.loc[g.g2_lan_act==15,'ve2']=[5]*len(g.loc[g.g2_lan_act==15]['g2_lan_act'])
    g.loc[g.g1_lan_act==15,'ve1']=[5]*len(g.loc[g.g1_lan_act==15]['g1_lan_act'])
    g.loc[g.g2_lan_act==14,'ve2']=[4]*len(g.loc[g.g2_lan_act==14]['g2_lan_act'])
    g.loc[g.g1_lan_act==14,'ve1']=[4]*len(g.loc[g.g1_lan_act==14]['g1_lan_act'])
    g.loc[g.g2_lan_act==13,'ve2']=[3]*len(g.loc[g.g2_lan_act==13]['g2_lan_act'])
    g.loc[g.g1_lan_act==13,'ve1']=[3]*len(g.loc[g.g1_lan_act==13]['g1_lan_act'])
    g.loc[g.type2=='Transition Metal','ve2']=[2]*len(g.loc[g.type2=='Transition Metal']['g2_lan_act'])
    g.loc[g.type1=='Transition Metal','ve1']=[2]*len(g.loc[g.type1=='Transition Metal']['g1_lan_act'])
    #redfinining valence electrons
    g= g.loc[:, ~g.columns.str.contains('^Unnamed')]
    g_dict=g.to_dict(orient='list')
    old=pd.read_csv(old_handel) #loading data from the Liu et al. 2021 paper
    old=old[old["Te (cm^{-1})"]==0]
    old.drop_duplicates(inplace=True)
    #creating different pandas dataframes for different purposes
    gr=g[g["Re (\AA)"].isna()==False] #gr only contains molecules that have R_e available 
    gw=gr[gr["omega_e (cm^{-1})"].isna()==False] #gw only contains molecules that have R_e and omega_e available
    g_new=g.loc[g['Molecule'].isin(old.Molecule)==False] #g_new contains only new data
    g_old=g.loc[g['Molecule'].isin(old.Molecule)] #g_new contains only old data from liu et al., 2021
    gr_old=g_old[g_old["Re (\AA)"].isna()==False] #gr_old only contains molecules that have R_e available from liu et al., 2021
    gw_old=gr_old[gr_old["omega_e (cm^{-1})"].isna()==False] #gw_old only contains molecules that have R_e and omega_e available from liu et al., 2021
    gr_new=g_new[g_new["Re (\AA)"].isna()==False] #gr_new only contains new molecules that have R_e available
    gw_new=gr_new[gr_new["omega_e (cm^{-1})"].isna()==False] #gw_new only contains new molecules that have R_e and omega_e available
    
    # permuting the properties of atoms 1 and 2 in the diatomic molecules as described in Liu et al., 2021 and in the manuscript, to create expanded data frames containing both A-B and B-A molecules
    reverse=['A1','A2','g1','g2','p1','p2','g1iso','g2iso','g1_lan_act','g2_lan_act','g1_lan_act_iso','g2_lan_act_iso','atom1','atom2','type1','type2','e1','e2','ve1','ve2']
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
    #the extension '-expand' indicates the inclusion of both A-B and B-A variations of the diatomic molecule in the datafram
    g_expand=pd.DataFrame.from_dict(g_dict, orient='columns')
    g_expand.drop_duplicates(subset=['A1','A2','g1','g2','p1','p2','g1iso','g2iso','g1_lan_act','g2_lan_act','g1_lan_act_iso','g2_lan_act_iso','atom1','atom2','type1','type2','ve1','ve2'], keep='first', inplace=True, ignore_index=False)
    gr_expand=g_expand[g_expand["Re (\AA)"].isna()==False]
    gw_expand=gr_expand[gr_expand["omega_e (cm^{-1})"].isna()==False]

    g_new_expand=g_expand.loc[g_expand['Molecule'].isin(old.Molecule)==False]
    g_old_expand=g_expand.loc[g_expand['Molecule'].isin(old.Molecule)]
    gr_old_expand=g_old_expand[g_old_expand["Re (\AA)"].isna()==False]
    gw_old_expand=gr_old_expand[gr_old_expand["omega_e (cm^{-1})"].isna()==False]
    gr_new_expand=g_new_expand[g_new_expand["Re (\AA)"].isna()==False]
    gw_new_expand=gr_new_expand[gr_new_expand["omega_e (cm^{-1})"].isna()==False]
    return g,gr,gw, g_old, g_new, gr_old, gw_old, gr_new, gw_new, g_expand, gr_expand, gw_expand, g_old_expand, g_new_expand, gr_old_expand, gw_old_expand, gr_new_expand, gw_new_expand


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
    train=[]
    test=[]
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
                print(i)
                print('warning: A molecule in the test (validation) set is aslo in the training set')
        train.append(re_train_set['Molecule'])
        if (re_test_set['Molecule'].tolist()) in test:
            break

        test.append(re_test_set['Molecule'].tolist())
 
        



        trval[str(s)]={} # intiate a dictionary that stores the three parameters values after optimization for each split s
        trval[str(s)]['$\sigma^2$']=1 # intiate the value of the multiplicative constant of the kernel
        trval[str(s)]['length scale']=1 # intiate the value of the length scale
        trval[str(s)]['noise level']=1 # intiate the value of the noise level in the additive white kernel
      
        
        
        reg = LinearRegression().fit(re_train_set[prior_features], re_train_set[logtarget]) #Liner regression model to fix the constatns coefficients of the prirori mean function in each MC-CV step
        
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
        
        r_y_train_pred=np.exp(np.array(r_y_train_pred_log))*(1+0.5*r_std_train**2) #log transformation was used to convert the resulting $ln(\omega_e)$ predictive distribution to $\omega_e$ predictive distribution
        r_y_test_pred=np.exp(np.array(r_y_test_pred_log))*(1+0.5*r_std_test**2) #log transformation was used to convert the resulting $ln(\omega_e)$ predictive distribution to $\omega_e$ predictive distribution
        
        r_std_train=r_y_train_pred*r_std_train #log transformation was used to convert the resulting $ln(\omega_e)$ predictive distribution to $\omega_e$ predictive distribution
        r_std_test=r_y_test_pred*r_std_test #log transformation was used to convert the resulting $ln(\omega_e)$ predictive distribution to $\omega_e$ predictive distribution
        
        for m in range(len(r_y_test_pred)):
            if r_y_test_pred[m]<0:
                print('negative result') #indicates negative results if any 
        r_y_test_pred=(np.array(r_y_test_pred))
        for m in range(len(r_y_test_pred)):
            if r_y_test_pred[m]<0:
                print('negative result') #indicates negative results if any 
        
        
        for  mol in  re_test_set['Molecule'].tolist():
            test.append(mol)
        mean_std.append(np.array(r_std_test).mean())  #calculating mean standard deviations

        trval[str(s)]['mean_std']=mean_std[-1]

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
        print("Split:",s)
        #print('Molecule',re_test_set['Molecule'].tolist()[-1])
       # print('Train MAE', np.array(Train_MAE).mean(),trval[str(s)]['Train_MAE'])
        print('Test MAE', np.array(MAE).mean(),trval[str(s)]['MAE'])
        #print('Test R%: ',np.array(R).mean(),trval[str(s)]['R'])
        #print('Train RMSE: ',np.array(Train_RMSE).mean(),trval[str(s)]['Train_RMSE'])
        print('Test RMSE: ',np.array(RMSE).mean(),trval[str(s)]['RMSE'])
      #  print('Train RMSLE: ',np.array(Train_RMSLE).mean(),trval[str(s)]['Train_RMSLE'])
     #   print('Test RMSLE: ',np.array(RMSLE).mean(),trval[str(s)]['RMSLE'])
              
              
        
    
        s=s+1 # incrementing the MC-CV split counter
        

        for i in range(len(re_train_set.ind)):
            if re_train_set.ind.tolist()[i] not in r_y_train_preds:   
                r_y_train_preds[re_train_set.ind.tolist()[i]]=[r_y_train_pred[i]] #adding MC-CV train prediction list of a molecule of index 'i' which is not yet in re_train_set dictionary
                r_train_stds[re_train_set.ind.tolist()[i]]=[r_std_train[i]] #adding MC-CV train GPR standard deviation list of a molecule of index 'i' not yet in r_train_stds dictionary

            else:
                r_y_train_preds[re_train_set.ind.tolist()[i]].append(r_y_train_pred[i])  #apeending new MC-CV train prediction to an existing list of predictions of a molecule indexed i in the r_y_train_preds dictionary
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


# ## 3.4 Plotting Function

# In[ ]:


def plot_results(df,x,y,target,r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds):
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
    pyplot.xticks(fontsize=14)
    pyplot.yticks(fontsize=14)
    ax.errorbar(df[target], re_train_preds, yerr=re_train_std, fmt ='o',label='Training set')
    ax.errorbar(df[target], re_test_preds, yerr=re_test_std, fmt ='o',label='Validation set')

    line=df[target].tolist()
    line.append(0)
    line.append(4500)
    ax.plot(line,line,'-k')
    ax.plot([0, 1], [0, 1],'-k', transform=ax.transAxes)
    ax.legend(prop={'size': 18})
    pyplot.xlabel(x,fontdict={'size': 16})
    pyplot.ylabel(y,fontdict={'size': 16})
    return re_train_preds,re_train_std,re_test_preds,re_test_std,out,fig,ax


# ## 3.5 A function to report a statistical summary

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


g,gr,gw, g_old, g_new, gr_old, gw_old, gr_new, gw_new, g_expand, gr_expand, gw_expand, g_old_expand, g_new_expand, gr_old_expand, gw_old_expand, gr_new_expand, gw_new_expand=load(handel=r"w4_gw_expand_pred.csv",old_handel=r"list of molecules used in Xiangue and Jesus paper.csv")


# ## 4.2 Stratify data according to the levels of the target values

# In[ ]:


gw_expand=gw_expand[~gw_expand['Molecule'].isin(['AgBi','Hg2','XeCl','HgCl'])] #Remmove molecules with uncertain data
gw_expand["ln(omega_e (cm^{-1}))"]=np.log(gw_expand["omega_e (cm^{-1})"]) # used as target
gw_expand['wcat']=gw_expand["ln(omega_e (cm^{-1}))"]  # gw_expand['wcat'] is used to define strata for the process of stratified sampling
gw_expand_unique=np.unique(gw_expand['wcat'])
ind=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,307]
for i in range(len(ind)-1):
    gw_expand['wcat'].where((gw_expand['wcat']>gw_expand_unique[ind[i+1]])|(gw_expand['wcat']<=gw_expand_unique[ind[i]]),gw_expand_unique[ind[i]],inplace=True)


# ## 4.3 Train the ml model and make predictions

# In[ ]:


gw_expand['mu^(1/2)']=(np.sqrt(gw_expand['Reduced mass'])) #square root of the reduced mass will be used as a feature
gw_expand['ln(mu^(1/2))']=np.log(np.sqrt(gw_expand['Reduced mass'])) #the natural logarithm of the square root of the reduced mass will be used as a feature in the prior mean function


# In[ ]:


trval,train,test,mean_std,Train_MAE,Train_RMSE,Train_R,Train_RMSLE,MAE,RMSE,R,RMSLE,r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds=ml_model(data=gw_expand,strata=gw_expand['wcat'],test_size=30,features=['Re (au)','p1','p2','g1_lan_act','g2_lan_act','mu^(1/2)'],prior_features=['Re (au)','p1','p2','g1_lan_act','g2_lan_act','ln(mu^(1/2))'],target='omega_e (cm^{-1})',logtarget="ln(omega_e (cm^{-1}))",nu=3/2,normalize_y=True,n_splits=1000)


# ## 4.4 Plot and save results

# In[ ]:


from matplotlib.transforms import Bbox
re_train_preds,re_train_std,re_test_preds,re_test_std,out,fig,ax=plot_results(gw_expand,'True $\omega_e (cm^{-1})$','Predicted $\omega_e (cm^{-1})$','omega_e (cm^{-1})',r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds);
pyplot.savefig('w4_scatter.svg',bbox_inches=Bbox([[-1,-1],fig.get_size_inches()]))
for i in range(len(re_test_preds)):
    if abs(gw_expand['omega_e (cm^{-1})'].tolist()[i]-re_test_preds[i])<100:
        continue
    ax.annotate(gw_expand['Molecule'].tolist()[i], (gw_expand['omega_e (cm^{-1})'].tolist()[i], re_test_preds[i]))
pyplot.savefig('w4_scatter_ann.svg',bbox_inches=Bbox([[-1,-1],fig.get_size_inches()]))


# In[ ]:


results('Model w4',gw_expand,'omega_e (cm^{-1})',re_test_preds,308,MAE,RMSE,R,r"stat_summ.csv")
gw_expand['re_test_preds']=re_test_preds
gw_expand['re_test_std']=re_test_std
gw_expand['re_train_preds']=re_train_preds
gw_expand['re_train_std']=re_train_std
gw_expand.to_csv('w4_gw_expand_pred.csv')
split_stat = pd.DataFrame(list(zip(Train_MAE,Train_RMSE,MAE,RMSE)),columns =['Train_MAE','Train_RMSE','MAE','RMSE'])
split_stat.to_csv('split_stat.csv') 


# ## 4.5 Predicting $\omega_e$  of homonuclear molecules from hetronuclear data

# In[ ]:


features=['Re (au)','p1','p2','g1_lan_act','g2_lan_act','mu^(1/2)']
prior_features=['Re (au)','p1','p2','g1_lan_act','g2_lan_act','ln(mu^(1/2))']
logtarget="ln(omega_e (cm^{-1}))"
target="omega_e (cm^{-1})"
nu=3/2
normalize_y=True
gw_expand.loc[gw_expand.atom2=='D','atom2']=['H']*len(gw_expand.loc[gw_expand.atom2=='D']['atom2']) #Isotoplogues are treated as homonuclear molecules (ensuring none of hydrogen isotoplogus is used to train the model)
gw_expand.loc[gw_expand.atom1=='D','atom1']=['H']*len(gw_expand.loc[gw_expand.atom1=='D']['atom1']) #Isotoplogues are treated as homonuclear molecules (ensuring none of hydrogen isotoplogus is used to train the model)
gw_expand.loc[gw_expand.atom2=='T','atom2']=['H']*len(gw_expand.loc[gw_expand.atom2=='T']['atom2']) #Isotoplogues are treated as homonuclear molecules (ensuring none of hydrogen isotoplogus is used to train the model)
gw_expand.loc[gw_expand.atom1=='T','atom1']=['H']*len(gw_expand.loc[gw_expand.atom1=='T']['atom1']) #Isotoplogues are treated as homonuclear molecules (ensuring none of hydrogen isotoplogus is used to train the model)
re_train_set=gw_expand[gw_expand['atom1']!=gw_expand['atom2']] # training set consisting of only hetronuclear molecules
re_test_set=gw_expand[gw_expand['atom1']==gw_expand['atom2']]  # testing set consisting of only homonuclear molecules

signal_variance=(re_train_set[logtarget].var())
length_scale=(re_train_set[features].std()).mean()

reg = LinearRegression().fit(re_train_set[prior_features], re_train_set[logtarget])
        
re_train_set['prior_mean']=reg.coef_[0]*re_train_set[prior_features[0]]+reg.coef_[1]*re_train_set[prior_features[1]]+reg.coef_[2]*re_train_set[prior_features[2]]+reg.coef_[3]*re_train_set[prior_features[3]]+reg.coef_[4]*re_train_set[prior_features[4]]+reg.coef_[5]*re_train_set[prior_features[5]]+reg.intercept_
re_test_set['prior_mean']=reg.coef_[0]*re_test_set[prior_features[0]]+reg.coef_[1]*re_test_set[prior_features[1]]+reg.coef_[2]*re_test_set[prior_features[2]]+reg.coef_[3]*re_test_set[prior_features[3]]+reg.coef_[4]*re_test_set[prior_features[4]]+reg.coef_[5]*re_test_set[prior_features[5]]+reg.intercept_

prior_mean='prior_mean'



gpr = MyGPR(kernel=ConstantKernel(constant_value=signal_variance)*Matern(length_scale=length_scale, nu=nu)+WhiteKernel(noise_level=re_train_set[target].std()/np.sqrt(2),noise_level_bounds=(10**-15,1)),n_restarts_optimizer=20,normalize_y=normalize_y,random_state=42)
gpr.fit(re_train_set[features], re_train_set[logtarget]-re_train_set[prior_mean])

r_y_train_pred_log,r_std_train=gpr.predict(re_train_set[features], return_std=True)
r_y_test_pred_log,r_std_test=gpr.predict(re_test_set[features], return_std=True)

r_y_train_pred_log=r_y_train_pred_log+np.array(re_train_set[prior_mean])
r_y_test_pred_log=r_y_test_pred_log+np.array(re_test_set[prior_mean])

r_y_train_pred=np.exp(np.array(r_y_train_pred_log))*(1+0.5*r_std_train**2)
r_y_test_pred=np.exp(np.array(r_y_test_pred_log))*(1+0.5*r_std_test**2)

r_std_train=r_y_train_pred*r_std_train
r_std_test=r_y_test_pred*r_std_test


print(re_test_set[target],r_y_test_pred)


# In[ ]:


df=gw_expand[gw_expand['atom1']==gw_expand['atom2']]
fig, ax =pyplot.subplots(figsize=(7,7))
pyplot.xticks(fontsize=14)
pyplot.yticks(fontsize=14)
ax.errorbar(-200, -200,fmt='-o')
ax.errorbar(df[target], r_y_test_pred, yerr=r_std_test, fmt ='o',label='Homonuclear molecules in the test set')
line=df[target].tolist()
line.append(-100)
line.append(5000)
ax.plot(line,line,'-k')
ax.plot([0, 1], [0, 1],'-k', transform=ax.transAxes)
pyplot.xlim(np.array(line).min(),np.ceil(np.array(line).max()))
pyplot.ylim(np.array(line).min(),np.ceil(np.array(line).max()))
ax.legend(prop={'size': 12})
pyplot.xlabel('True $\omega_e (cm^{-1})$',fontdict={'size': 16})
pyplot.ylabel('Predicted $\omega_e (cm^{-1})$',fontdict={'size': 16})
pyplot.savefig('w4_homo_from_hetro.svg')
for i in range(len(df)):
    if abs(df['omega_e (cm^{-1})'].tolist()[i]-r_y_test_pred[i])<100:
        continue
    ax.annotate(df['Molecule'].tolist()[i], (df['omega_e (cm^{-1})'].tolist()[i],  r_y_test_pred[i]))
pyplot.savefig('w4_homo_from_hetro_ann.svg',bbox_inches=Bbox([[-1,-1],fig.get_size_inches()])) 

