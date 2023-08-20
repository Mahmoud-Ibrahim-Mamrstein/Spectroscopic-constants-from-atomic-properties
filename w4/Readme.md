# Model w4
 Model w3 uses the periods and the groups of the constiuent atoms and the reduced mass of the molecule along with $R_e$ as features in a Matern 3/2 kernels. $R_e$ is included as a feature in the prior mean function as well.
$$m_{w3-w4} = \beta_0^{w3-w4}+\beta_1^{w3-w4}(p_1+p_2) + \beta_2^{w3-w4}(g_1+g_2) + \beta_3^{w3-w4} R_e +\beta_4^{w3-w4} \ln{(\mu^{1/2})}$$
where $\beta_k^{w4}$, $k \in \{0,1,2,3,4\}$ are the linear coefficients

## Files description 
[w4.ipynb](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/w4/w4.ipynb) : A Jupiter notebook of the w4 model\
[w4_gr_expand_pred.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/w4/w4_gr_expand_pred.csv) : A csv file containing prediction of the w4 model. Columns we_test_preds, we_test_std, we_train_preds, and we_train_std include the testing predictions and standard deviations and training predictions and standard deviations for each molecule in the training validation set.\
[w4_learninig_curves.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/19d4c9834a2bb9521bcfde277eb46e59ded7ae3a/w4/w4_learning_curves.csv): A csv file containing a statistical summary of each step used for producing the learning curves. To get these results the ml_model function in [w4.ipynb](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/w4/w4.ipynb) was run several times with different testing set sizes, each was run for 500 MC-CV splits. Column 'portion_of_data_set' indicate the portion used for training the model. \
[w4_split_stat.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/9ba61b3b2dad50f9deddf955f9303b0adc310fae/w4/w4_split_stat.csv): A csv file containing a statistical summary of each MC-CV step.\
[stat_summ](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b4a0b821ac4d3341ebb8f74178527c816e036641/w4/stat_summ.csv): statistical summary of the model. \
[w4_test.ipynb](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/60c2ff448d4a4fe9cec11760452be0c2cb8d1cfd/w4/w4_test.ipynb): code to test model w4 \
[w4_testing_results.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/60c2ff448d4a4fe9cec11760452be0c2cb8d1cfd/w4/w4_testing_results.csv): Contains testing results


