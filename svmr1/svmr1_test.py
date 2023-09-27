#!/usr/bin/env python
# coding: utf-8

# # Testing model svmr1 

# # 1. Import libraries and objects

# In[ ]:


import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import sklearn as sk
from matplotlib.backends.backend_pdf import PdfPages
import re
import seaborn as sns
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
import time
import math
from math import sqrt
import scipy
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.gaussian_process.kernels import ConstantKernel, Product, Matern, WhiteKernel, RBF, DotProduct, ExpSineSquared


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


# ## 3.3 Function that perfroms the loo splits, train the SVR and make predictions

# In[ ]:


def ml_model(data,constant_value,length_scale,C,epsilon,nu,features,logtarget,target,n_splits=1000): #function used for implementing the leave one out GPR model for testing
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
                print('warning: A molecule in the test set is aslo in the training set')
        train.append(re_train_set['Molecule'])
        if (re_test_set['Molecule'].tolist()) in test:
            break

        test.append(re_test_set['Molecule'].tolist())
        



        trval[str(s)]={} # intiate a dictionary that stores the three parameters values after optimization for each split s
        trval[str(s)]['$\sigma^2$']=1 # intiate the value of the multiplicative constant of the kernel
        trval[str(s)]['length scale']=1 # intiate the value of the length scale
        trval[str(s)]['noise level']=1 # intiate the value of the noise level in the additive white kernel
        
        if re_test_set['Molecule'].tolist()[0] not in ['MoC','NbC','NiC','NiO','NiS','PbI','PdC','RuC','SnI','UO','WC','YC','ZnBr','ZnCl','WO','ZnI','ZnF','HCl','DCl','CrC','CoO','IrSi','UF','ZrC']:
            continue # only make prediction is the molecule is in the following list ['MoC','NbC','NiC','NiO','NiS','PbI','PdC','RuC','SnI','UO','WC','YC','ZnBr','ZnCl','WO','ZnI','ZnF','HCl','DCl','CrC','CoO','IrSi','UF','ZrC']
      
        kernel = kernel=ConstantKernel(constant_value=constant_value)*Matern(length_scale=length_scale, nu=3/2)+WhiteKernel(noise_level=10**-15,noise_level_bounds=(10**-15,1))
        
        reg = make_pipeline(SVR(kernel=kernel,C=C, epsilon=epsilon)).fit(re_train_set[features], re_train_set[logtarget])
              
        r_y_train_pred_log=reg.predict(re_train_set[features])
        
        r_y_test_pred_log=reg.predict(re_test_set[features])
        
        
        r_y_train_pred=r_y_train_pred_log
        r_y_test_pred=r_y_test_pred_log
        
        for m in range(len(r_y_test_pred)):
            if r_y_test_pred[m]<0:
                print('negative result') #indicates negative results if any 
        r_y_test_pred=(np.array(r_y_test_pred))
        for m in range(len(r_y_test_pred)):
            if r_y_test_pred[m]<0:
                print('negative result') #indicates negative results if any 

        
        for  mol in  re_test_set['Molecule'].tolist():
            test.append(mol)

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
               

            else:
                r_y_train_preds[re_train_set.ind.tolist()[i]].append(r_y_train_pred[i])  #apeending new loo train prediction to an existing list of predictions of a molecule indexed i in the r_y_train_preds dictionary
                
        for i in range(len(re_test_set.ind)):
            if re_test_set.ind.tolist()[i] not in r_y_test_preds:
                r_y_test_preds[re_test_set.ind.tolist()[i]]=[r_y_test_pred[i]] #adding loo test prediction list of a molecule of index 'i' which is not yet in re_test_set dictionary
            else:
                r_y_test_preds[re_test_set.ind.tolist()[i]].append(r_y_test_pred[i]) #apeending new loo test prediction to an existing list of predictions of a molecule indexed 'i' in the r_y_test_preds dictionary
    end_time = time.time()
    retime=end_time-start_time
    retime # timing the validation stage
    return trval,train,test,Train_MAE,Train_RMSE,Train_R,Train_RMSLE,MAE,RMSE,R,RMSLE,r_y_train_preds,r_y_test_preds       


# # 4. Main Body

# ## 4.1 Loading data

# In[ ]:


g,gr,gw, g_old, g_new, gr_old, gw_old, gr_new, gw_new, g_expand, gr_expand, gw_expand, g_old_expand, g_new_expand, gr_old_expand, gw_old_expand, gr_new_expand, gw_new_expand=load(handel=r"svmr1_gr_expand_test.csv",old_handel=r"list of molecules used in Xiangue and Jesus paper.csv")


# In[ ]:


gr_expand=gr_expand[~gr_expand['Molecule'].isin(['AgBi','Hg2','XeCl','HgCl','HgBr',"HgI"])] # Removing moleules with uncertain experimental values of their spectroscopic constnants
gr_expand['mu^(1/2)']=(np.sqrt(gr_expand['Reduced mass'])) # The square root of the reduced mass is used as a feature
gr_expand['ln(mu^(1/2))']=np.log(np.sqrt(gr_expand['Reduced mass'])) # The logaritm of the square root of the reduced mass is used as a feature
gr_expand['ln(w)']=np.log(gr_expand['omega_e (cm^{-1})']) #The natural logaritm of the square root of $\omega_e$ will be used as a feature in the kernel and in the prior mean function


# ## 4.2 Making predictions

# In[ ]:


best_params_=([('C', 100), ('epsilon', 0.001), ('kernel__k1__k1__constant_value', 1000.0), ('kernel__k1__k2__length_scale', 24.06295579569555)]) # the best parameters rturned from the Bayes search cv,these results may be used to replicate the results in the manuscript
trval,train,test,Train_MAE,Train_RMSE,Train_R,Train_RMSLE,MAE,RMSE,R,RMSLE,r_y_train_preds,r_y_test_preds=ml_model(data=gr_expand,C=best_params_[0][1],epsilon=best_params_[1][1],constant_value=best_params_[2][1],length_scale=best_params_[3][1],nu=3/2,features=['p1','p2','g1_lan_act','g2_lan_act'],target='Re (\AA)',logtarget='Re (\AA)',n_splits=1000)


# In[ ]:


test_molecules=gw_expand[gw_expand['Molecule'].isin(['MoC','NbC','NiC','NiO','NiS','PbI','PdC','RuC','SnI','UO','WC','YC','ZnBr','ZnCl','WO','ZnI','ZnF','HCl','DCl'])]['Molecule'].tolist()
true_values=gw_expand[gw_expand['Molecule'].isin(['MoC','NbC','NiC','NiO','NiS','PbI','PdC','RuC','SnI','UO','WC','YC','ZnBr','ZnCl','WO','ZnI','ZnF','HCl','DCl'])]['Re (\AA)'].tolist()
re_test_preds=[]
re_test_std=[]
out=[]
s=0
i=(-1)
for index in r_y_test_preds:
    z=(i)**s
    s=s+1
    if z==(-1):
        continue
    re_test_preds.append(((r_y_test_preds[index][0])))
testing_results = pd.DataFrame(list(zip(test_molecules, true_values,re_test_preds)), columns =['Molecule', 'true $R_e (\AA)$','Predicted $R_e (\AA)$'])
testing_results.to_csv(r'svmr1_testing_results.csv')

