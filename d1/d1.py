import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import sklearn as sk
from matplotlib.backends.backend_pdf import PdfPages
import re
import seaborn as sns
from matplotlib import pyplot
%matplotlib inline
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
----------------------------------------------------------------------------
class MyGPR(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=1000000, gtol=1e-3, **kwargs):
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
--------------------------------------------
def load(handel,old_handel):
    dfe=pd.read_csv(handel,index_col=None)
    df1=pd.read_csv(r"data/peridic.csv",index_col=None)
    dfe= dfe.loc[:, ~dfe.columns.str.contains('^Unnamed')]
    nul=[np.NaN]*len(dfe.Molecule)
    for char in ['e1','e2']:
        dfe[char]=nul
    for char in df1.Symbol:
        ind1=dfe.loc[dfe['Molecule'].str.contains(r'^'+char+r'\D')].index.values
        ind2=dfe.loc[dfe['Molecule'].str.contains(char+r'$')].index.values
        ind3=dfe.loc[dfe['Molecule'].str.contains(r'^'+char+r'2')].index.values
        #print(char)
        #print(df1[df1.Symbol==char].Period.values)
        dfe.loc[ind1,'e1']=df1[df1.Symbol==char].NumberofElectrons.values[0]
        dfe.loc[ind2,'e2']=df1[df1.Symbol==char].NumberofElectrons.values[0]
        dfe.loc[ind3,'e1']=df1[df1.Symbol==char].NumberofElectrons.values[0]
        dfe.loc[ind3,'e2']=df1[df1.Symbol==char].NumberofElectrons.values[0]
    g=dfe
    g['n1']=g.A1-g.e1
    g['n2']=g.A2-g.e2
    g.loc[g.atom2=='H','p2']=[1]*len(g.loc[g.atom2=='H']['p2'])
    g.loc[g.atom1=='H','p1']=[1]*len(g.loc[g.atom1=='H']['p1'])
    g.loc[g.atom2=='H','g2_lan_act']=[1]*len(g.loc[g.atom2=='H']['g2_lan_act'])
    g.loc[g.atom1=='H','g1_lan_act']=[1]*len(g.loc[g.atom1=='H']['g1_lan_act'])
    
    
    
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
    g.loc[g.atom2=='D','g2_lan_act_iso']=[0]*len(g.loc[g.atom2=='D']['g2_lan_act_iso'])
    g.loc[g.atom1=='D','g1_lan_act_iso']=[0]*len(g.loc[g.atom1=='D']['g1_lan_act_iso'])
    g.loc[g.atom2=='T','g2_lan_act_iso']=[-1]*len(g.loc[g.atom2=='T']['g2_lan_act_iso'])
    g.loc[g.atom1=='T','g1_lan_act_iso']=[-1]*len(g.loc[g.atom1=='T']['g1_lan_act_iso'])
    g.loc[g.atom2=='H','g2_lan_act_iso']=[1]*len(g.loc[g.atom2=='H']['g2_lan_act_iso'])
    g.loc[g.atom1=='H','g1_lan_act_iso']=[1]*len(g.loc[g.atom1=='H']['g1_lan_act_iso'])
    g['sum_p']=g['p1']+g['p2']
    g['sum_g']=g.g1_lan_act+g.g2_lan_act
    g['diff_p']=abs(g['p1']-g['p2'])
    g['diff_g']=abs(g['g1_lan_act']-g['g1_lan_act'])
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
    g['K']=(4*(np.pi**2)*((g['omega_e (cm^{-1})']*(0.00012))**2)*g['Reduced mass'])
    g['sqrt(K)']=np.sqrt(4*(np.pi**2)*((g['omega_e (cm^{-1})']*(0.0000046))**2)*g['Reduced mass'])
    g['4*(np.pi**2)*omega_e (au)']=4*(np.pi**2)*((g['omega_e (cm^{-1})']*(0.0000046)))
    g['4*(np.pi**2)*Re (au)*omega_e (au)^2']=g['4*(np.pi**2)*Re (au)']*((g['omega_e (cm^{-1})']*0.0000046)**2)
    g['ve1']=g['g1_lan_act']
    g['ve2']=g['g2_lan_act']
    g['log(D_0/(R_e^3*Z_1*Z_2))']=np.log((g["D0 (eV)"]*0.037)/((g["Re (au)"]**3)*g.e1*g.e2))
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
    g= g.loc[:, ~g.columns.str.contains('^Unnamed')]
    g_dict=g.to_dict(orient='list')
    old=pd.read_excel(old_handel)
    old=old[old["Te (cm^{-1})"]==0]
    old.drop_duplicates(inplace=True)
    gr=g[g["Re (\AA)"].isna()==False]
    gw=gr[gr["omega_e (cm^{-1})"].isna()==False]
    g_new=g.loc[g['Molecule'].isin(old.Molecule)==False]
    g_old=g.loc[g['Molecule'].isin(old.Molecule)]
    gr_old=g_old[g_old["Re (\AA)"].isna()==False]
    gw_old=gr_old[gr_old["omega_e (cm^{-1})"].isna()==False]
    gr_new=g_new[g_new["Re (\AA)"].isna()==False]
    gw_new=gr_new[gr_new["omega_e (cm^{-1})"].isna()==False]
    reverse=['A1','A2','g1','g2','p1','p2','g1iso','g2iso','g1_lan_act','g2_lan_act','g1_lan_act_iso','g2_lan_act_iso','atom1','atom2','type1','type2','e1','e2','ve1','ve2']
    for key,value in g_dict.items():
        if key in reverse:
            continue 
        else:
            g_dict[key]=value+value
            #=g_dict[key].append(g_dict[key])
    s=0        
    for i in range(len(reverse)):
            if s==len(reverse):
                break
            A=g_dict[reverse[s]]+g_dict[reverse[s+1]]
            B=g_dict[reverse[s+1]]+g_dict[reverse[s]]
            g_dict[reverse[s]]=A
            g_dict[reverse[s+1]]=B
            s=s+2
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
------------------------------------------------------------------------------------------------
def ml_model(data,strata,test_size,features,prior_features,logtarget,target,nu,normalize_y,n_splits=1000):
    r_y_train_preds={}
    r_y_test_preds={}
    r_train_stds={}
    r_test_stds={}
    trval={} #intiate a dictionary to store optmized kernels and scores
    start_time = time.time()
    RMSE=[]
    RMSLE=[]
    MAE=[]
    R=[]
    Train_RMSE=[]
    Train_RMSLE=[]
    Train_MAE=[]
    Train_R=[]
    mean_std=[]
    train=[]
    test=[]
    mcs = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,random_state=42)
    #skf = StratifiedKFold(n_splits=0, shuffle=True, random_state=42)
    loo = LeaveOneOut()
    #print(loo.get_n_splits(data))
    s=0

    #for train_index, test_index in loo.split(data):
    for train_index, test_index in mcs.split(data,strata):
        #print(train_index)
        re_train_set1 = data.iloc[train_index]
        re_train_set1['ind']=train_index
        re_test_set1 = data.iloc[test_index]
        re_test_set1['ind']=test_index
        print('size of training set before removing mirror molecules',len(re_train_set1))
        
        re_train_set=re_train_set1[~re_train_set1['Molecule'].isin(re_test_set1['Molecule'].tolist())]
        re_test_set=pd.concat([re_test_set1,re_train_set1[re_train_set1['Molecule'].isin(re_test_set1['Molecule'].tolist())]])
        
        for i in re_train_set['Molecule'].isin([re_test_set['Molecule']]):
            if i ==True:
                print(i)
        print('size of training set after removing mirror molecules',len(re_train_set))
        #print('size of test set after adding mirror molecules',len(re_test_set))
        train.append(re_train_set['Molecule'])
        if (re_test_set['Molecule'].tolist()) in test:
            #continue
            break

        test.append(re_test_set['Molecule'].tolist())
        #print(test)
        



        trval[str(s)]={}
        trval[str(s)]['$\sigma^2$']=1
        trval[str(s)]['length scale']=1
        trval[str(s)]['noise level']=1
        
        #if re_test_set['Molecule'].tolist()[0] not in ['InBr','MoC','NbC','NiC','NiO','NiS','PbI','PdC','RuC','SnI','UO','WC','YC','ZnBr','ZnCl','WO','ZnI','ZnF','HgBr','HgI','HCl','DCl']:
         #   continue
      
        
        
        reg = LinearRegression().fit(re_train_set[prior_features], re_train_set[logtarget])
        
        re_train_set['prior_mean']=reg.predict(re_train_set[prior_features])
        #reg.coef_[0]*re_train_set[prior_features[0]]+reg.coef_[1]*re_train_set[prior_features[1]]+reg.coef_[2]*re_train_set[prior_features[2]]+reg.coef_[3]*re_train_set[prior_features[3]]+reg.coef_[4]*re_train_set[prior_features[4]]+reg.coef_[5]*re_train_set[prior_features[5]]+reg.coef_[6]*re_train_set[prior_features[6]]+reg.intercept_
        re_test_set['prior_mean']=reg.predict(re_test_set[prior_features])
        #reg.coef_[0]*re_test_set[prior_features[0]]+reg.coef_[1]*re_test_set[prior_features[1]]+reg.coef_[2]*re_test_set[prior_features[2]]+reg.coef_[3]*re_test_set[prior_features[3]]+reg.coef_[4]*re_test_set[prior_features[4]]+reg.coef_[5]*re_test_set[prior_features[5]]+reg.coef_[6]*re_test_set[prior_features[6]]+reg.intercept_
        
        prior_mean='prior_mean'
        signal_variance=(re_train_set[logtarget].var())
        length_scale=(re_train_set[features].std()).mean()
        #gpr = MyGPR(kernel=ConstantKernel(constant_value=trval[str(s)]['best $\sigma^2$'],constant_value_bounds='fixed')*Matern(length_scale=trval[str(s)]['best length scale'],length_scale_bounds='fixed' ,nu=nu)+WhiteKernel(noise_level=trval[str(s)]['best noise level'],noise_level_bounds='fixed'),n_restarts_optimizer=20,normalize_y=normalize_y,random_state=42)
        gpr = MyGPR(kernel=ConstantKernel(constant_value=signal_variance)*Matern(length_scale=length_scale, nu=nu)+WhiteKernel(noise_level=re_train_set[target].std()/np.sqrt(2),noise_level_bounds=(10**-15,1)),n_restarts_optimizer=20,normalize_y=normalize_y,random_state=42)
        #ConstantKernel(constant_value=signal_variance)*Matern(length_scale=length_scale, nu=nu)+WhiteKernel(noise_level=re_train_set[target].std()/np.sqrt(2),noise_level_bounds=(10**-15,1))
        gpr.fit(re_train_set[features], re_train_set[logtarget]-re_train_set[prior_mean])
        #gpr.log_marginal_likelihood(theta=None, eval_gradient=False, clone_kernel=False)
        
        #trval[str(s)]['$\sigma^2$']=gpr.kernel_.get_params(deep=True)['k1__k1__constant_value']
        #trval[str(s)]['length scale']=gpr.kernel_.get_params(deep=True)['k1__k2__length_scale']
        #trval[str(s)]['noise level']=gpr.kernel_.get_params(deep=True)['k2__noise_level']

        #re_test_set=re_test_set[~re_test_set['Molecule'].isin(['XeCl'])]

        r_y_train_pred_log,r_std_train=gpr.predict(re_train_set[features], return_std=True)
        r_y_test_pred_log,r_std_test=gpr.predict(re_test_set[features], return_std=True)
        
        r_y_train_pred_log=r_y_train_pred_log+np.array(re_train_set[prior_mean])
        r_y_test_pred_log=r_y_test_pred_log+np.array(re_test_set[prior_mean])
        
        r_y_train_pred=r_y_train_pred_log
        r_y_test_pred=r_y_test_pred_log
        
        #r_std_train=r_y_train_pred*r_std_train
        #r_std_test=r_y_test_pred*r_std_test
        
        '''for m in range(len(r_y_test_pred)):
            if r_y_test_pred[m]<0:
                print('yalaaahwy')
        r_y_test_pred=(np.array(r_y_test_pred))
        for m in range(len(r_y_test_pred)):
            if r_y_test_pred[m]<0:
                print('y5rabaaaay')'''
                
                
        #if (100*(np.sqrt(mean_squared_error(re_test_set['Re (\AA)'],r_y_test_pred)))/((data['Re (\AA)']).max()-(data['Re (\AA)']).min())) > 3.0:
         #   print(re_test_set['Molecule'].isin(test))
          #  continue
        
        
        for  mol in  re_test_set['Molecule'].tolist():
            test.append(mol)
        mean_std.append(np.array(r_std_test).mean())

        trval[str(s)]['mean_std']=mean_std[-1]

        RMSE.append(np.sqrt(mean_squared_error(re_test_set[target],r_y_test_pred)))

        trval[str(s)]['RMSE']=RMSE[-1]
        
        Train_RMSE.append(np.sqrt(mean_squared_error(re_train_set[target],r_y_train_pred)))

        trval[str(s)]['Train_RMSE']=Train_RMSE[-1]
        
        RMSLE.append(np.sqrt(mean_squared_error(re_test_set[logtarget],r_y_test_pred_log)))

        trval[str(s)]['RMSLE']=RMSLE[-1]
        
                
        Train_RMSLE.append(np.sqrt(mean_squared_error(re_train_set[logtarget],r_y_train_pred_log)))

        trval[str(s)]['Train_RMSLE']=Train_RMSLE[-1]

        MAE.append(sum(abs(re_test_set[target]-(r_y_test_pred)))/len(re_test_set[target]))

        trval[str(s)]['MAE']=MAE[-1]
        
        Train_MAE.append(sum(abs(re_train_set[target]-(r_y_train_pred)))/len(re_train_set[target]))

        trval[str(s)]['Train_MAE']=Train_MAE[-1]

        R.append(100*(np.sqrt(mean_squared_error(re_test_set[target],r_y_test_pred)))/((data[target]).max()-(data[target]).min()))

        trval[str(s)]['R']=R[-1]
        
        #if trval[str(s)]['R'] > 3.0:
         #   continue
            #print(re_test_set['Molecule'])

        #print("Molecule",re_test_set["Molecule"],'-------')
        #print('sigma: ',trval[str(s)]['$\sigma^2$'],"length scale: ",trval[str(s)]['length scale'],'noise level: ',trval[str(s)]['noise level'])
        
        print("Split:",s)
        print('Molecule',re_test_set['Molecule'].tolist()[-1])
       # print('Train MAE', np.array(Train_MAE).mean(),trval[str(s)]['Train_MAE'])
        print('Test MAE', np.array(MAE).mean(),trval[str(s)]['MAE'])
        print('Test R%: ',np.array(R).mean(),trval[str(s)]['R'])
        print('Train RMSE: ',np.array(Train_RMSE).mean(),trval[str(s)]['Train_RMSE'])
        print('Test RMSE: ',np.array(RMSE).mean(),trval[str(s)]['RMSE'])
      #  print('Train RMSLE: ',np.array(Train_RMSLE).mean(),trval[str(s)]['Train_RMSLE'])
     #   print('Test RMSLE: ',np.array(RMSLE).mean(),trval[str(s)]['RMSLE'])
              
        
    
        s=s+1
        

        for i in range(len(re_train_set.ind)):
            if re_train_set.ind.tolist()[i] not in r_y_train_preds:   
                r_y_train_preds[re_train_set.ind.tolist()[i]]=[r_y_train_pred[i]]
                r_train_stds[re_train_set.ind.tolist()[i]]=[r_std_train[i]]

                #print("Molecule: ",re_train_set.loc[train_index[i],'Molecule'],"true: ",gr.loc[train_index[i],'Re (\AA)'],"pred: ",r_y_train_pred[i],"standard deviation: ",r_std_train[i])

            else:
                r_y_train_preds[re_train_set.ind.tolist()[i]].append(r_y_train_pred[i])
                r_train_stds[re_train_set.ind.tolist()[i]].append(r_std_train[i])
                
        for i in range(len(re_test_set.ind)):
            if re_test_set.ind.tolist()[i] not in r_y_test_preds:
                r_y_test_preds[re_test_set.ind.tolist()[i]]=[r_y_test_pred[i]]
                r_test_stds[re_test_set.ind.tolist()[i]]=[r_std_test[i]]
            else:
                r_y_test_preds[re_test_set.ind.tolist()[i]].append(r_y_test_pred[i])
                r_test_stds[re_test_set.ind.tolist()[i]].append(r_std_test[i])
    end_time = time.time()
    retime=end_time-start_time
    retime
    return trval,train,test,mean_std,Train_MAE,Train_RMSE,Train_R,Train_RMSLE,MAE,RMSE,R,RMSLE,r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds
-------------------------------------------------------------------
def plot_results(df,x,y,target,r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds):
    re_train_preds=[]
    re_train_std=[]
    re_test_preds=[]
    re_test_std=[]
    out=[]
    for index in range(len(df.index)):
            re_train_preds.append(np.array(r_y_train_preds[index]).mean())
            re_train_std.append(np.std(np.array(r_y_train_preds[index]))+np.array(r_train_stds[index]).mean())
            re_test_preds.append((np.array(r_y_test_preds[index])).mean())
            re_test_std.append(np.std(np.array(r_y_test_preds[index]))+np.array(r_test_stds[index]).mean())
    fig, ax =pyplot.subplots(figsize=(7,7))
    pyplot.xticks(fontsize=14)
    pyplot.yticks(fontsize=14)
    #ax.set_xlim(0, df[target].max())
    ax.errorbar(df[target], re_train_preds,yerr=re_train_std, fmt ='o',label='Training set')
    ax.errorbar(df[target], re_test_preds, yerr=re_test_std ,fmt ='o',label='Validation set')

    line=df[target].tolist()
    line.append(np.array(line).min()-1)
    line.append((np.array(line).max())+1)
    ax.plot(line,line,'-k')
    #pyplot.xticks(ticks=np.linspace(1, 4, num=4))
    #pyplot.yticks(ticks=np.linspace(1, 4, num=4))
    ax.plot([0, 1], [0, 1],'-k',transform=ax.transAxes)
    pyplot.xlim(np.array(line).min(),(np.array(line).max()))
    pyplot.ylim(np.array(line).min(),(np.array(line).max()))
    ax.legend(prop={'size': 12})
    pyplot.xlabel(x,fontdict={'size': 16})
    pyplot.ylabel(y,fontdict={'size': 16})
    return re_train_preds,re_train_std,re_test_preds,re_test_std,out,fig,ax
---------------------------------------------------------
def hyper_parameters(target,handel):
    length_scale=[]
    MAE=[]
    RMSE=[]
    R=[]
    for s in trval:
        #print(s)
        for key in trval[str(s)]:
            #print(trval[str(s)][key])
            if key=='length scale':
                #print(trval[str(s)][key])
                length_scale.append(trval[str(s)][key])
            elif key=='MAE':
                MAE.append(trval[str(s)][key])
            elif key=='RMSE':
                RMSE.append(trval[str(s)][key])
            elif key=='R':
                R.append(trval[str(s)][key])
    hyper = pd.DataFrame(list(zip(length_scale, MAE,RMSE,R)),
                   columns =['length_scale', 'MAE','RMSE',"R"])
    hyper['length_scale_rounded_1']=hyper['length_scale'].round(decimals=1)
    hyper_group_1=hyper.groupby('length_scale_rounded_1').mean()
    hyper['length_scale_rounded_0']=hyper['length_scale'].round(decimals=0)
    hyper_group_0=hyper.groupby('length_scale_rounded_0').mean()
    hyper_group_0
    #hyper.to_csv(handel+'\\'+target+' '+'results per split.csv', index=True)
    #hyper_group_1.to_csv(handel+'\\'+' '+target+'results per split grouped and averged.csv', index=True)
    #hyper_group_0.to_csv(handel+'\\'+' '+target+'results per split grouped and averged.csv', index=True)
    return hyper, hyper_group_1, hyper_group_0
---------------------------------------------------------------------------
g,gr,gw, g_old, g_new, gr_old, gw_old, gr_new, gw_new, g_expand, gr_expand, gw_expand, g_old_expand, g_new_expand, gr_old_expand, gw_old_expand, gr_new_expand, gw_new_expand=load(handel=r"data/g.csv",old_handel=r"data/list of molecules used in Xiangue and Jesus paper.csv")
gw_expand=gw_expand[gw_expand["D0 (eV)"].isna()==False]
len(np.unique(gw_expand[gw_expand["D0 (eV)"].isna()==False]["D0 (eV)"]))
---------------------------------------------------------------------------------------------------
gw_expand['ln(Re (\AA))']=np.log(gw_expand['Re (\AA)'])  #ln of the target
gw_expand['mu^(1/2)']=np.sqrt(gw_expand['Reduced mass'])
gw_expand['mu^(-1/2)']=1/np.sqrt(gw_expand['Reduced mass'])
gw_expand["ln(D0 (eV))"]=np.log(gw_expand["D0 (eV)"])
gw_expand=gw_expand[~gw_expand['Molecule'].isin(['AgBi','Hg2','XeCl','VF','HgCl','Ne2'])]
#gw_expand=gw_expand[~gw_expand['type1'].isin(['H'])]
#gw_expand=gw_expand[~gw_expand['type2'].isin(['H'])]
gw_expand=gw_expand[gw_expand["D0 (eV)"].isna()==False]
gr_expand_nodub=gw_expand
#.drop_duplicates(subset=['p1','p2','g1_lan_act','g2_lan_act'],ignore_index=True)
gr_expand_nodub["com"]=((gr_expand_nodub["omega_e (au)"]*gr_expand_nodub["Re (au)"]**2))
gr_expand_nodub["ln(com)"]=np.log((gr_expand_nodub["omega_e (au)"]*gr_expand_nodub["Re (au)"]**2))
#gr_expand_nodub=gw_expand
#gr_expand_nodub=gr_expand_nodub[gr_expand_nodub['log(D_e)/(R_e^3*Z_1*Z_2)'].isna()==False]
gr_expand_nodub['rcat']=gr_expand_nodub["ln(D0 (eV))"]

gr_expand_nodub_unique=np.unique(gr_expand_nodub['rcat'])
gr_expand_nodub['Re (au)^-3']=gr_expand_nodub['Re (au)']**-3  
ind=[0,10,20,30,40,50,60,70,80,89,95,100,110,120,130,140,149,159,167,177,187,196,205]
#ind=[0,35,75,125,170,215,267]
for i in range(len(ind)-1):
    
    gr_expand_nodub['rcat'].where((gr_expand_nodub['rcat']>gr_expand_nodub_unique[ind[i+1]])|(gr_expand_nodub['rcat']<=gr_expand_nodub_unique[ind[i]]),gr_expand_nodub_unique[ind[i]],inplace=True)
np.unique((gr_expand_nodub['rcat']),return_counts=True) 
-----------------------------------------------------------------------
gw_expand['ln(mu^(1/2))']=np.log(gw_expand['mu^(1/2)'])
gw_expand['ln(Re (au))']=np.log(gw_expand['Re (au)'])
gr_expand_nodub["ln(omega_e (au))"]=np.log(gr_expand_nodub["omega_e (au)"])
gr_expand_nodub["ln(omega_e (cm^{-1}))"]=np.log(gr_expand_nodub['omega_e (cm^{-1})'])
gr_expand_nodub["exp(Re (au))"]=np.log(gr_expand_nodub["Re (au)"])
gr_expand_nodub['average_p']=gr_expand_nodub["sum_p"]/2
#gr_nodub['average_p']=gr_nodub["sum_p"]/2
gr_expand_nodub["ln(D0 (eV))"]=-np.log(gr_expand_nodub["D0 (eV)"])
gr_expand_nodub['log(D_0/(Z_1*Z_2))']=np.log((gr_expand_nodub["D0 (eV)"])/(gr_expand_nodub.e1+gr_expand_nodub.e2))
gr_expand_nodub['D_0/(Z_1*Z_2)']=(gr_expand_nodub["D0 (eV)"])/(gr_expand_nodub.e1*gr_expand_nodub.e2)
#gr_nodub['D_0/(Z_1*Z_2)']=(gr_nodub["D0 (eV)"])/(gr_nodub.e1*gr_nodub.e2)
gr_expand_nodub["ln(Re (au))"]=np.log(gr_expand_nodub["Re (au)"])
gr_expand_nodub["Re (au)^3"]=(gr_expand_nodub["Re (au)"])**3
gr_expand_nodub["ln(Re (au)^3)"]=np.log(gr_expand_nodub["Re (au)^3"])
gr_expand_nodub["log(e1*e2)"]=(gr_expand_nodub["e1"]*gr_expand_nodub["e2"])
gr_expand_nodub["log(e1)"]=np.log(gr_expand_nodub["e1"])
gr_expand_nodub["log(e2)"]=np.log(gr_expand_nodub["e2"])/np.log(gr_expand_nodub["e2"])
gr_expand_nodub['D_0(au)']=(gr_expand_nodub["D0 (eV)"]*0.037)
gr_nodub["ln(omega_e (au))"]=np.log(gr_expand_nodub["omega_e (au)"])
gr_nodub["ln(omega_e (cm^{-1}))"]=np.log(gr_expand_nodub['omega_e (cm^{-1})'])
gr_nodub['average_p']=gr_expand_nodub["sum_p"]/2
gr_nodub['average_p']=gr_nodub["sum_p"]/2
gr_nodub["ln(D0 (eV))"]=np.log(gr_expand_nodub["D0 (eV)"])
gr_nodub['log(D_0/(Z_1*Z_2))']=np.log((gr_expand_nodub["D0 (eV)"]*0.037)/(gr_expand_nodub.e1*gr_expand_nodub.e2))
gr_nodub['D_0/(Z_1*Z_2)']=(gr_expand_nodub["D0 (eV)"]*0.037)/(gr_expand_nodub.e1*gr_expand_nodub.e2)
gr_nodub['D_0(au)']=(gr_expand_nodub["D0 (eV)"]*0.037)
gr_nodub['D_0/(Z_1*Z_2)']=(gr_nodub["D0 (eV)"]*0.037)/(gr_nodub.e1*gr_nodub.e2)
gr_nodub["ln(Re (au))"]=np.log(gr_expand_nodub["Re (au)"])
gr_nodub["Re (au)^3"]=(gr_expand_nodub["Re (au)"])**3
gr_nodub["ln(Re (au)^3)"]=np.log(gr_expand_nodub["Re (au)^3"])
gr_nodub["log(e1*e2)"]=np.log(gr_expand_nodub["e1"]*gr_expand_nodub["e2"])
gr_nodub["log(e1)"]=np.log(gr_expand_nodub["e1"])
gr_nodub["log(e2)"]=np.log(gr_expand_nodub["e2"])/np.log(gr_expand_nodub["e2"])
gr_expand_nodub["(e1*e2)"]=(gr_expand_nodub["e1"]+gr_expand_nodub["e2"])
gr_expand_nodub['1/ln(omega_e (au))']=(1/gr_expand_nodub["ln(omega_e (au))"])
gr_expand_nodub['mu^(1/2)']*(gr_expand_nodub["ln(omega_e (au))"])
#gr_expand_nodub['1/omega_e (au)']=(1/gr_expand_nodub["omega_e (au)"])
gr_expand_nodub['1/D0 (eV)']=(1/gr_expand_nodub["D0 (eV)"])
(gr_expand_nodub["ln(K)"])=np.log(gr_expand_nodub["K"])
gr_expand_nodub["ln('omega_ex_e (cm^{-1})')"]=np.log(gr_expand_nodub['omega_ex_e (cm^{-1})'])
gr_expand_nodub['ln(mu)']=np.log(gw_expand['Reduced mass'])
-----------------------------------------------------------------------------------------------
trval,train,test,mean_std,Train_MAE,Train_RMSE,Train_R,Train_RMSLE,MAE,RMSE,R,RMSLE,r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds=ml_model(data=gr_expand_nodub,strata=gr_expand_nodub['rcat'],test_size=22,features=['g1_lan_act','g2_lan_act','p1','p2','mu^(1/2)'],prior_features=["ln(K)",'ln(Re (au))','g1_lan_act','g2_lan_act','p1','p2'],logtarget="ln(D0 (eV))",target="ln(D0 (eV))",nu=3/2,normalize_y=True,n_splits=1000)
------------------------------------------------------------------------------------------------------------
from matplotlib.transforms import Bbox
re_train_preds,re_train_std,re_test_preds,re_test_std,out,fig,ax=plot_results(gr_expand_nodub[gr_expand_nodub["ln(D0 (eV))"]<6],'True  $-ln(D_0)$','Pedicted  $-ln(D_0)$',"ln(D0 (eV))",r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds)
#pyplot.savefig('experiment1.svg',bbox_inches=Bbox([[-2,-2],fig.get_size_inches()])) 
for i in range(len(gr_expand_nodub)):
    if abs(gr_expand_nodub["ln(D0 (eV))"].tolist()[i]-re_test_preds[i])<0.2:
        continue
    ax.annotate(gr_expand_nodub['Molecule'].tolist()[i], (gr_expand_nodub["ln(D0 (eV))"].tolist()[i], re_test_preds[i]))
#pyplot.savefig('experiment1_ann.svg',bbox_inches=Bbox([[-2,-2],fig.get_size_inches()]))    
----------------------------------------------------------------------------------------------------
results('$\ln{(D0 (eV))}"$ statistical summary',gr_expand_nodub,"ln(D0 (eV))",re_test_preds,208,MAE,RMSE,R,r"stat_summ.csv")
------------------------------------------------------------------------------------------------------------
#learning curves
Train_rmse_means=[]
Train_rmse_stds=[]
Valid_rmse_means=[]
Valid_rmse_stds=[]
Train_rmse_means=[]
Train_rmse_stds=[]
Valid_rmse_means=[]
Valid_rmse_stds=[]
for i in [0.5,0.4,0.3,0.2,0.1,22]:
    trval,train,test,mean_std,Train_MAE,Train_RMSE,Train_R,Train_RMSLE,MAE,RMSE,R,RMSLE,r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds=ml_model(data=gr_expand_nodub,strata=gr_expand_nodub['rcat'],test_size=i,features=['g1_lan_act','g2_lan_act','p1','p2','mu^(1/2)'],prior_features=["ln(K)",'ln(Re (au))','g1_lan_act','g2_lan_act','p1','p2'],logtarget="ln(D0 (eV))",target="ln(D0 (eV))",nu=3/2,normalize_y=True,n_splits=500)
    Train_rmse_means.append(np.array(Train_RMSE).mean())
    Train_rmse_stds.append(np.array(Train_RMSE).std())
    Valid_rmse_means.append(np.array(RMSE).mean())
    Valid_rmse_stds.append(np.array(RMSE).std())
    print(Train_rmse_means)
    print(Valid_rmse_means)
Train_rmse_means.append(np.array(Train_RMSE).mean())
Train_rmse_stds.append(np.array(Train_RMSE).std())
Valid_rmse_means.append(np.array(RMSE).mean())
Valid_rmse_stds.append(np.array(RMSE).std())    
---------------------------------------------------------
from matplotlib.transforms import Bbox
fig, ax =pyplot.subplots(figsize=(7,7))
ax.errorbar(np.array([0.5,0.6,0.7,0.8,0.9,0.95])*100, Train_rmse_means,fmt='-o',label='Training set')
ax.fill_between(np.array([0.5,0.6,0.7,0.8,0.9,0.95])*100, np.array(Train_rmse_means) - 0.5*np.array(Train_rmse_stds), np.array(Train_rmse_means) + 0.5*np.array(Train_rmse_stds),alpha=0.2)
ax.errorbar(np.array([0.5,0.6,0.7,0.8,0.9,0.95])*100, Valid_rmse_means,fmt='-o',label='Validation set')
ax.fill_between(np.array([0.5,0.6,0.7,0.8,0.9,0.95])*100, np.array(Valid_rmse_means) - 0.5*np.array(Valid_rmse_stds), np.array(Valid_rmse_means) + 0.5*np.array(Valid_rmse_stds),alpha=0.2)
#pyplot.xticks(ticks=np.linspace(1, 4, num=4))
#pyplot.yticks(ticks=np.linspace(0, 5, num=4))
ax.legend(prop={'size': 18})
pyplot.xlabel('portion of the training set %',fontdict={'size': 18})
pyplot.ylabel("RMSE of $ln(\\frac{D_0}{ Z_{1}Z_{2}})$",fontdict={'size': 18})
pyplot.xticks(fontsize=16)
pyplot.yticks(fontsize=16);
---------------------------------------------------------------
learning_curve = pd.DataFrame(list(zip([0.5,0.6,1-0.3,1-0.2,1-0.1,0.95],Train_rmse_means,Train_rmse_stds,Valid_rmse_means,Valid_rmse_stds)),columns =['portion of the data set','Train_rmse_means','Train_rmse_stds','valid_rmse_means','valid_rmse_stds'])
learning_curve.to_csv('learning_curves.csv') 
split_stat = pd.DataFrame(list(zip(Train_MAE,Train_RMSE,Train_R,MAE,RMSE)),columns =['Train_MAE','Train_RMSE','Train_R','MAE','RMSE'])
split_stat.to_csv('split_stat.csv')
gw_expand.drop('we_pred',axis=1,inplace=True)
gw_expand['-ln(De)_pred']=re_test_preds
gw_expand.to_csv('gw_expand_pred.csv')
------------------------------------------------------------------------
#Homo_Hetro
features=['g1_lan_act','g2_lan_act','p1','p2','mu^(1/2)']
prior_features=['ln(Re (au))','g1_lan_act','g2_lan_act','p1','p2',"ln(omega_e (cm^{-1}))",'ln(mu^(1/2))']
logtarget="ln(D0 (eV))"
target="ln(D0 (eV))"
nu=3/2
normalize_y=True
gw_expand.loc[gw_expand.atom2=='D','atom2']=['H']*len(gw_expand.loc[gw_expand.atom2=='D']['atom2'])
gw_expand.loc[gw_expand.atom1=='D','atom1']=['H']*len(gw_expand.loc[gw_expand.atom1=='D']['atom1'])
gw_expand.loc[gw_expand.atom2=='T','atom2']=['H']*len(gw_expand.loc[gw_expand.atom2=='T']['atom2'])
gw_expand.loc[gw_expand.atom1=='T','atom1']=['H']*len(gw_expand.loc[gw_expand.atom1=='T']['atom1'])
re_train_set=gr_expand_nodub[gr_expand_nodub['atom1']!=gr_expand_nodub['atom2']]
re_test_set=gr_expand_nodub[gr_expand_nodub['atom1']==gr_expand_nodub['atom2']]
signal_variance=(re_train_set[logtarget].var())
length_scale=(re_train_set[features].std()).mean()

        
reg = LinearRegression().fit(re_train_set[prior_features], re_train_set[logtarget])

re_train_set['prior_mean']=reg.coef_[0]*re_train_set[prior_features[0]]+reg.coef_[1]*re_train_set[prior_features[1]]+reg.coef_[2]*re_train_set[prior_features[2]]+reg.coef_[3]*re_train_set[prior_features[3]]+reg.coef_[4]*re_train_set[prior_features[4]]+reg.coef_[5]*re_train_set[prior_features[5]]+reg.coef_[6]*re_train_set[prior_features[6]]+reg.intercept_
re_test_set['prior_mean']=reg.coef_[0]*re_test_set[prior_features[0]]+reg.coef_[1]*re_test_set[prior_features[1]]+reg.coef_[2]*re_test_set[prior_features[2]]+reg.coef_[3]*re_test_set[prior_features[3]]+reg.coef_[4]*re_test_set[prior_features[4]]+reg.coef_[5]*re_test_set[prior_features[5]]+reg.coef_[6]*re_test_set[prior_features[6]]+reg.intercept_

prior_mean='prior_mean'
signal_variance=(re_train_set[logtarget].var())
length_scale=(re_train_set[features].std()).mean()
#gpr = MyGPR(kernel=ConstantKernel(constant_value=trval[str(s)]['best $\sigma^2$'],constant_value_bounds='fixed')*Matern(length_scale=trval[str(s)]['best length scale'],length_scale_bounds='fixed' ,nu=nu)+WhiteKernel(noise_level=trval[str(s)]['best noise level'],noise_level_bounds='fixed'),n_restarts_optimizer=20,normalize_y=normalize_y,random_state=42)
gpr = MyGPR(kernel=ConstantKernel(constant_value=signal_variance)*Matern(length_scale=length_scale, nu=nu)+WhiteKernel(noise_level=re_train_set[target].std()/np.sqrt(2),noise_level_bounds=(10**-15,1)),n_restarts_optimizer=20,normalize_y=normalize_y,random_state=42)
#ConstantKernel(constant_value=signal_variance)*Matern(length_scale=length_scale, nu=nu)+WhiteKernel(noise_level=re_train_set[target].std()/np.sqrt(2),noise_level_bounds=(10**-15,1))
gpr.fit(re_train_set[features], re_train_set[logtarget]-re_train_set[prior_mean])
#gpr.log_marginal_likelihood(theta=None, eval_gradient=False, clone_kernel=False)

#trval[str(s)]['$\sigma^2$']=gpr.kernel_.get_params(deep=True)['k1__k1__constant_value']
#trval[str(s)]['length scale']=gpr.kernel_.get_params(deep=True)['k1__k2__length_scale']
#trval[str(s)]['noise level']=gpr.kernel_.get_params(deep=True)['k2__noise_level']

#re_test_set=re_test_set[~re_test_set['Molecule'].isin(['XeCl'])]

r_y_train_pred_log,r_std_train=gpr.predict(re_train_set[features], return_std=True)
r_y_test_pred_log,r_std_test=gpr.predict(re_test_set[features], return_std=True)

r_y_train_pred_log=r_y_train_pred_log+np.array(re_train_set[prior_mean])
r_y_test_pred_log=r_y_test_pred_log+np.array(re_test_set[prior_mean])

r_y_train_pred=r_y_train_pred_log
r_y_test_pred=r_y_test_pred_log
print(re_test_set[target],r_y_test_pred)
---------------------------------------------------------------------------------------------
#plot Homo_Hetro
from matplotlib.transforms import Bbox
fig, ax =pyplot.subplots(figsize=(7,7))

ax.errorbar(-4, -3, fmt ='o',label='True values')
ax.errorbar(re_test_set[target], r_y_test_pred, yerr=r_std_test, fmt ='o',label='Homonuclear molecules in the test set')

line=re_test_set[target].tolist()
#line.append(0)
line.append(np.ceil(np.array( r_y_test_pred).min())-1)
line.append(np.ceil(np.array( r_y_test_pred).max()))
ax.plot(line,line,'-k')
#pyplot.xticks(fontdict={'size': 18})
#pyplot.yticks(fontdict={'size': 18})
#ax.plot([0, 1], [0, 1], transform=ax.transAxes)
pyplot.xlim(np.array(line).min(),np.ceil(np.array(line).max()))
pyplot.ylim(np.array(line).min(),np.ceil(np.array(line).max()))
#ax.legend(prop={'size': 18})
pyplot.xlabel('True  $-ln(D_0^0(eV))$',fontdict={'size': 18})
pyplot.ylabel('Pedicted  $-ln(D_0^0(eV))$',fontdict={'size': 18})
pyplot.xticks(fontsize=14)
pyplot.yticks(fontsize=14)
#for i in range(len(re_test_set)):
    #if abs(gr_expand_nodub['D_0/(Z_1*Z_2)'].tolist()[i]-test_preds[i])>0:
        #continue
#    ax.annotate(re_test_set['Molecule'].tolist()[i], (re_test_set['log(D_0/(Z_1*Z_2))'].tolist()[i], r_y_test_pred[i]))
pyplot.savefig('experiment1_homofromhetro.svg',bbox_inches=Bbox([[-0.2,0],fig.get_size_inches()]))
for i in range(len(re_test_set)):
    if abs(re_test_set["ln(D0 (eV))"].tolist()[i]-r_y_test_pred[i])<0.2:
        continue
    ax.annotate(re_test_set['Molecule'].tolist()[i], (re_test_set["ln(D0 (eV))"].tolist()[i], r_y_test_pred[i]))
pyplot.savefig('experiment1_homo_hetro_ann.svg',bbox_inches=Bbox([[-2,-2],fig.get_size_inches()]))    
