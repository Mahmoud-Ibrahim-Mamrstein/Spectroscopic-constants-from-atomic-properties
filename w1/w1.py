import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import sklearn as sk
from matplotlib.backends.backend_pdf import PdfPages
import re
import seaborn as sns
from matplotlib import pyplot
from matplotlib.transforms import Bbox
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
def load(handel,old_handel):
    dfe=pd.read_csv(handel,index_col=None)
    df1=pd.read_csv(r"/gpfs/home/maaibrahim/gpr/peridic.csv",index_col=None)
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
    g= g.loc[:, ~g.columns.str.contains('^Unnamed')]
    g_dict=g.to_dict(orient='list')
    old=pd.read_csv(old_handel)
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
        
        #testawy=g_test.loc[g_test['Molecule'].isin(g['Molecule'])==False]
        #testawy=testawy[testawy['Molecule'].isin(['ZnCl','InBr','MoC','NbC','NiC','NiO','NiS','PbI','PdC','RuC','SnI','UO','WC','YC','ZnBr'])]
        #re_test_set=testawy[testawy["omega_e (cm^{-1})"].isna()==False]
        
        #re_test_set['Re (au)']=((re_test_set['Re (\AA)'])*1.8897259885789)
        #re_test_set['mu^(1/2)']=(np.sqrt(re_test_set['Reduced mass']))
        #re_test_set['ln(mu^(1/2))']=np.log(np.sqrt(re_test_set['Reduced mass']))
        #re_test_set["ln(omega_e (cm^{-1}))"]=np.log(re_test_set["omega_e (cm^{-1})"])



        trval[str(s)]={}
        trval[str(s)]['$\sigma^2$']=1
        trval[str(s)]['length scale']=1
        trval[str(s)]['noise level']=1
      
        #if re_test_set['Molecule'].tolist()[0] not in ['InBr','MoC','NbC','NiC','NiO','NiS','PbI','PdC','RuC','SnI','UO','WC','YC','ZnBr','ZnCl','WO','ZnI','ZnF','HgBr','HgI','HCl','DCl']:
         #   continue
        
        reg = LinearRegression().fit(re_train_set[prior_features], re_train_set[logtarget])
        re_train_set['prior_mean']=reg.predict(re_train_set[prior_features])
        #re_train_set['prior_mean']=reg.coef_[0]*re_train_set[prior_features[0]]+reg.coef_[1]*re_train_set[prior_features[1]]+reg.coef_[2]*re_train_set[prior_features[2]]+reg.coef_[3]*re_train_set[prior_features[3]]+reg.coef_[4]*re_train_set[prior_features[4]]+reg.intercept_
        re_test_set['prior_mean']=reg.predict(re_test_set[prior_features])
        #re_test_set['prior_mean']=reg.coef_[0]*re_test_set[prior_features[0]]+reg.coef_[1]*re_test_set[prior_features[1]]+reg.coef_[2]*re_test_set[prior_features[2]]+reg.coef_[3]*re_test_set[prior_features[3]]+reg.coef_[4]*re_test_set[prior_features[4]]+reg.intercept_
        
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
        
        r_y_train_pred=np.exp(np.array(r_y_train_pred_log))*(1+0.5*r_std_train**2)
        r_y_test_pred=np.exp(np.array(r_y_test_pred_log))*(1+0.5*r_std_test**2)
        
        r_std_train=r_y_train_pred*r_std_train
        r_std_test=r_y_test_pred*r_std_test
        
        
        #r_std_train=r_y_train_pred*r_std_train
        #r_std_test=r_y_test_pred*r_std_test
        
        for m in range(len(r_y_test_pred)):
            if r_y_test_pred[m]<0:
                print('yalaaahwy')
        r_y_test_pred=(np.array(r_y_test_pred))
        for m in range(len(r_y_test_pred)):
            if r_y_test_pred[m]<0:
                print('y5rabaaaay')
                
                
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
        
        Train_R.append(100*(np.sqrt(mean_squared_error(re_test_set[target],r_y_test_pred)))/((data[target]).max()-(data[target]).min()))

        trval[str(s)]['R']=R[-1]
        
        #if trval[str(s)]['R'] > 3.0:
         #   continue
            #print(re_test_set['Molecule'])

        print("Molecule",re_test_set["Molecule"],'-------')
        print("true",re_test_set[target],'-------')
        print("predict",r_y_test_pred,'-------')
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
    ax.errorbar(df[target], re_train_preds, yerr=re_train_std, fmt ='o',label='Training set')
    ax.errorbar(df[target], re_test_preds, yerr=re_test_std, fmt ='o',label='Validation set')

    line=df[target].tolist()
    line.append(0)
    line.append(max(np.array(re_test_preds))+max(np.array(re_test_std)))
    ax.plot(line,line,'-k')
    #pyplot.xticks(ticks=np.linspace(0, 4000, num=4))
    #pyplot.yticks(ticks=np.linspace(0, 4000, num=4))
    ax.plot([0, 1], [0, 1],'-k', transform=ax.transAxes)
    #pyplot.xlim(np.array(line).min(),np.ceil(np.array(line).max()))
    #pyplot.ylim(np.array(line).min(),np.ceil(np.array(line).max()))
    ax.legend(prop={'size': 18})
    pyplot.xlabel(x,fontdict={'size': 16})
    pyplot.ylabel(y,fontdict={'size': 16})
    return re_train_preds,re_train_std,re_test_preds,re_test_std,out,fig,ax
def results(data_describtion,df,target,re_test_preds,no_molecules,MAE,RMSE,R,handle):
    results={}
    results[data_describtion]={}
    results[data_describtion]['Number of molecules in the whole data set']=no_molecules
    results[data_describtion]['MAE']=str(np.array(MAE).mean())+' (+/-) '+str(round(abs(np.mean(abs(np.mean(np.array(df[target]))-(np.mean(re_test_preds)+np.std(re_test_preds))))-(np.mean(abs(np.mean(np.array(df[target]))-(np.mean(re_test_preds)-np.std(re_test_preds))))))/2,4))
    results[data_describtion]['RMSE']=str(np.array(RMSE).mean())+' (+/-) '+str((abs(np.sqrt(np.mean(np.mean(np.array(df[target]))-(np.mean(re_test_preds)+np.std(re_test_preds)))**2)-(np.sqrt(np.mean((df[target].mean()-(np.mean(re_test_preds)-np.std(re_test_preds))))**2))).round(decimals=4))/2)
    results[data_describtion]['$r%$']=str((np.array(R).mean()))+' (+/-) '+str(round((((abs(np.sqrt(np.mean(np.mean(np.array(df[target]))-(np.mean(re_test_preds)+np.std(re_test_preds)))**2)-(np.sqrt(np.mean((df[target].mean()-(np.mean(re_test_preds)-np.std(re_test_preds))))**2))))/2)*100/(df[target].max()-df[target].min())),4))
    results=pd.DataFrame.from_dict(results) 
    results.to_csv(handle, index=True)  
    return results
g,gr,gw, g_old, g_new, gr_old, gw_old, gr_new, gw_new, g_expand, gr_expand, gw_expand, g_old_expand, g_new_expand, gr_old_expand, gw_old_expand, gr_new_expand, gw_new_expand=load(handel=r"/gpfs/home/maaibrahim/gpr/g.csv",old_handel=r"/gpfs/home/maaibrahim/gpr/list of molecules used in Xiangue and Jesus paper.csv")
gw_expand=gw_expand[~gw_expand['Molecule'].isin(['AgBi','Hg2','XeCl','HgCl'])] #Remmove molecules with uncertain data

gw_expand["ln(omega_e (cm^{-1}))"]=np.log(gw_expand["omega_e (cm^{-1})"])
gw_expand["ln(omega_e (au))"]=np.log(gw_expand["omega_e (au)"])
gw_expand['wcat']=gw_expand["ln(omega_e (au))"]
#gw_expand['wcat']=gw_expand['Re (\AA)']
#np.sqrt(gw_expand['Reduced mass']**2+gw_expand['p1']**2+gw_expand['p2']**2+gw_expand['g1_lan_act']**2+gw_expand['g2_lan_act']**2)
gw_expand_unique=np.unique(gw_expand['wcat'])
ind=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,307]
#ind=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,303]
print(len(gw_expand_unique))
for i in range(len(ind)-1):
    
    gw_expand['wcat'].where((gw_expand['wcat']>gw_expand_unique[ind[i+1]])|(gw_expand['wcat']<=gw_expand_unique[ind[i]]),gw_expand_unique[ind[i]],inplace=True)
np.unique(gw_expand['wcat'],return_counts=True)
#len(np.unique(gw_expand['wcat']))
gw_expand['mu^(1/2)']=(np.sqrt(gw_expand['Reduced mass']))
gw_expand['ln(mu^(1/2))']=np.log(np.sqrt(gw_expand['Reduced mass']))
gw_expand['zero']=np.log(np.sqrt(gw_expand['Reduced mass']))*0
trval,train,test,mean_std,Train_MAE,Train_RMSE,Train_R,Train_RMSLE,MAE,RMSE,R,RMSLE,r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds=ml_model(data=gw_expand,strata=gw_expand['wcat'],test_size=30,features=['Re (\AA)^-1','p1','p2','g1_lan_act_iso','g2_lan_act_iso','g_average_lan_act_iso'],prior_features=['zero','zero','zero','zero','zero'],target='omega_e (cm^{-1})',logtarget="ln(omega_e (cm^{-1}))",nu=5/2,normalize_y=True,n_splits=1000)
from matplotlib.transforms import Bbox
re_train_preds,re_train_std,re_test_preds,re_test_std,out,fig,ax=plot_results(gw_expand,'True $\omega_e (cm^{-1})$','Predicted $\omega_e (cm^{-1})$','omega_e (cm^{-1})',r_y_train_preds,r_train_stds,r_y_test_preds,r_test_stds);
pyplot.savefig('w1_scatter.svg',bbox_inches=Bbox([[-1,-1],fig.get_size_inches()]))
for i in range(len(re_test_preds)):
    if abs(gw_expand['omega_e (cm^{-1})'].tolist()[i]-re_test_preds[i])<100:
        continue
    ax.annotate(gw_expand['Molecule'].tolist()[i], (gw_expand['omega_e (cm^{-1})'].tolist()[i], re_test_preds[i]))
pyplot.savefig('w1_scatter_ann.svg',bbox_inches=Bbox([[-1,-1],fig.get_size_inches()]))
results('hob reg3na leeko',gw_expand,'omega_e (cm^{-1})',re_test_preds,308,MAE,RMSE,R,r"w1_stat_summ.csv")
gw_expand['re_test_preds']=re_test_preds
gw_expand['re_test_std']=re_test_std
gw_expand['re_train_preds']=re_train_preds
gw_expand['re_train_std']=re_train_std
gw_expand.to_csv('w1_gr_expand_pred.csv')
split_stat = pd.DataFrame(list(zip(Train_MAE,Train_RMSE,MAE,RMSE)),columns =['Train_MAE','Train_RMSE','MAE','RMSE'])
split_stat.to_csv('w1_split_stat.csv') 