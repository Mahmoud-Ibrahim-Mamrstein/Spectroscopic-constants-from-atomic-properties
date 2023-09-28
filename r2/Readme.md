
# Model r2

A GPR model with a Matern 3/2 kernel and with groups and periods of the constituent atoms and the reduced mass of the molecule as features. We explicitly express the model's prior mean functions as linear functions in the groups and periods of the diatomic molecules' constituent atoms.
$$m_{r1-r2} = \beta_0^{r1-r2}+\beta_1^{r1-r2}(p_1+p_2) + \beta_2^{r1-r2}(g_1+g_2)$$
where  $\beta_k^{r1-r2}$, $k \in \{0,1,2\}$ are the linear coefficients of  $m_{r1-r2}$.
## Files description 
[r2.ipynb](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/r2/r2.ipynb) : A Jupiter notebook of the model\
[r2.py](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/r2/r2.py) : A python file of the model\
[r2_gr_expand_pred.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/r2/r2_gr_expand_pred.csv) : A csv file used for training the model. Columns re_test_preds, re_test_std, re_train_preds, and re_train_std include the testing predictions and standard deviations and training predictions and standard deviations for each molecule in the training validation set. (This file is an older version of  [data.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/560feedb6e0468d7400730de19a6d2ab31d2adac/data/data.csv), it was used to train the model at the time of submitting the manuscript and should be used to reproduce the training results in the manuscript. The most updated version of the dataset is included [The data folder](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/tree/6ec4a08f434a5dc0ae38345fc155a10db0b5ff49/data) )\
[r2_learninig_curves.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/19d4c9834a2bb9521bcfde277eb46e59ded7ae3a/r2/r2_learning_curves.csv): A csv file containing a statistical summary of each step used for producing the learning curves. To get these results the ml_model function in [r2.ipynb](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/r2/r2.ipynb) was run several times with different testing set sizes, each was run for 500 MC-CV splits. Column 'portion_of_data_set' indicates the portion used for training the model. \
[r2_split_stat.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/9ba61b3b2dad50f9deddf955f9303b0adc310fae/r2/r2_split_stat.csv): A csv file containing a statistical summary of each MC-CV step.\
[stat_summ](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b4a0b821ac4d3341ebb8f74178527c816e036641/r2/stat_summ.csv): statistical summary of the model.\
[periodic.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/cb121f56b411820aee8c37a67363ad192c939044/r2/peridic.csv): A csv file of the periodic table of elements\
[r2_gr_expand_test.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/03fb8e821479dfc6c13eb2018370bd2f914d6de6/r2/r2_gr_expand_test.csv): A csv files consisting of data of molecules used for testing in addition to molecules used for training and validation.\
[r2_testing_results.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/1eb8c5f0ff5f28c88635ce23f0bec026340aadbe/r2/r2_testing_results.csv): A csv of the models predictions on the testing set\
[list molecules used in Xiangue and Jesus paper.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/9041f8a5e3190998c0a8db29c5ffac11ad53a9fa/r2/list%20of%20molecules%20used%20in%20Xiangue%20and%20Jesus%20paper.csv): A list of molecules used to train the models Liu et al.[[1]](#1).

## References
<a id="1">[1]</a> 
X. Liu, S. Truppe, G. Meijer and J. Pérez-Ríos, Journal of
Cheminformatics, 2020, 12, 31.
