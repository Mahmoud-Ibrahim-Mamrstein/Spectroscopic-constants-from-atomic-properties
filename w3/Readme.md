# Model w3
 Model w3 uses the periods and the groups of the constiuent atoms and the reduced mass of the molecule as features in a Matern 3/2 kernels. $R_e$ is included as a feature in the prior mean function.
\begin{equation}
\label{eq15}
    \begin{gathered}  
        m_{w3-w4} = \beta_0^{w3-w4}+\beta_1^{w3-w4}(p_1+p_2) + \beta_2^{w3-w4}(g_1+g_2) \\ + \beta_3^{w3-w4} R_e +\beta_4^{w3-w4} \ln{(\mu^{1/2})},
    \end{gathered}  
\end{equation}
where $\beta_k^{w4}$, $k \in \{0,1,2,3,4\}$ are the linear coefficients
## Files description 
[w3.ipynb](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/w3/w3.ipynb) : A Jupiter notebook of the w3 model\
[w3_gr_expand_pred.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/w3/w3_gr_expand_pred.csv): A csv file containing prediction of the w3 model. Columns we_test_preds, we_test_std, we_train_preds, and we_train_std include the testing predictions and standard deviations and training predictions and standard deviations for each molecule in the training validation set.\
[w3_learninig_curves.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/19d4c9834a2bb9521bcfde277eb46e59ded7ae3a/w3/w3_learning_curves.csv): A csv file containing a statistical summary of each step used for producing the learning curves. To get these results the ml_model function in [w3.ipynb](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/w3/w3.ipynb) was run several times with different testing set sizes, each was run for 500 MC-CV splits. Column 'portion_of_data_set' indicate the portion used for training the model. \
[w3_split_stat.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/9ba61b3b2dad50f9deddf955f9303b0adc310fae/w3/w3_split_stat.csv): A csv file containing a statistical summary of each MC-CV step.\
[stat_summ](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b4a0b821ac4d3341ebb8f74178527c816e036641/w3/stat_summ.csv): statistical summary of the model. \

