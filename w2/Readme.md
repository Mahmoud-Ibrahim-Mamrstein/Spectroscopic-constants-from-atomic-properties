# Model w2
model w2 only includes groups and periods of the constituent atoms and the reduced mass of the molecule as features within a Matern 3/2 kernel. The prior mean of model w2 is given by 
$$m_{w2} =\beta_0^{w2}+\beta_1^{w2}(p_1+p_2) + \beta_2^{w2}(g_1+g_2)+\beta_3^{w2} \ln{(\mu^{1/2})}$$
where $\beta_k^{w2}$, $k \in \{0,1,2,3\}$ are linear coefficients.
## Files description 
[w2.ipynb](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/w2/w2.ipynb) : A Jupiter notebook of the model\
[w2.py](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/w2/w2.py) : A python file of the model\
[w2_gr_expand_pred.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/w2/w2_gr_expand_pred.csv) : A csv file used for training the model. Columns re_test_preds, re_test_std, re_train_preds, and re_train_std include the testing predictions and standard deviations and training predictions and standard deviations for each molecule in the training validation set. (This file is an older version of  [data.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/560feedb6e0468d7400730de19a6d2ab31d2adac/data/data.csv), it was used to train the model at the time of submitting the manuscript and should be used to reproduce the training results in the manuscript. The most updated version of the dataset is included [The data folder](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/tree/6ec4a08f434a5dc0ae38345fc155a10db0b5ff49/data) )\
[w2_learninig_curves.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/19d4c9834a2bb9521bcfde277eb46e59ded7ae3a/w2/w2_learning_curves.csv): A csv file containing a statistical summary of each step used for producing the learning curves. To get these results the ml_model function in [w2.ipynb](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/w2/w2.ipynb) was run several times with different testing set sizes, each was run for 500 MC-CV splits. Column 'portion_of_data_set' indicates the portion used for training the model. \
[w2_split_stat.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/9ba61b3b2dad50f9deddf955f9303b0adc310fae/w2/w2_split_stat.csv): A csv file containing a statistical summary of each MC-CV step.\
[stat_summ](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b4a0b821ac4d3341ebb8f74178527c816e036641/w2/stat_summ.csv): statistical summary of the model.\
[periodic.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/cb121f56b411820aee8c37a67363ad192c939044/w2/peridic.csv): A csv file of the periodic table of elements\
[w2_gr_expand_test.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/03fb8e821479dfc6c13eb2018370bd2f914d6de6/w2/w2_gr_expand_test.csv): A csv files consisting of data of molecules used for testing in addition to molecules used for training and validation.\
[w2_testing_results.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/1eb8c5f0ff5f28c88635ce23f0bec026340aadbe/w2/w2_testing_results.csv): A csv of the models predictions on the testing set\
[list molecules used in Xiangue and Jesus paper.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/9041f8a5e3190998c0a8db29c5ffac11ad53a9fa/w2/list%20of%20molecules%20used%20in%20Xiangue%20and%20Jesus%20paper.csv): A list of molecules used to train the models in [[1]](#1).

## References
<a id="1">[1]</a> 
X. Liu, S. Truppe, G. Meijer and J. Pérez-Ríos, Journal of
Cheminformatics, 2020, 12, 31.


 
