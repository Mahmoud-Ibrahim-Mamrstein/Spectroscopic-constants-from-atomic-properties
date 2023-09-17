# Model r1

Model r1 is the same as in Liu et al.[[1]](#1). A GPR model with a Matern 1/2 kernel and with groups and periods of the constituent atoms as features. We explicitly express the model's prior mean functions as linear functions in the groups and periods of the diatomic molecules' constituent atoms. \
$$m_{r1-r2} = \beta_0^{r1-r2}+\beta_1^{r1-r2}(p_1+p_2) + \beta_2^{r1-r2}(g_1+g_2)$$
\
where  $\beta_k^{r1-r2}$, $k \in \{0,1,2\}$ are the linear coefficients of  $m_{r1-r2}$.
## Files description 
[r1.ipynb](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/r1/r1.ipynb) : A Jupiter notebook of the r1 model\
[r1.py](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/r1/r1.py) : A python file of the r1 model\
[r1_gr_expand_pred.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/r1/r1_gr_expand_pred.csv) : A csv file used for training the r1 model. Columns re_test_preds, re_test_std, re_train_preds, and re_train_std include the testing predictions and standard deviations and training predictions and standard deviations for each molecule in the training validation set. (This file is an older version of  [g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/560feedb6e0468d7400730de19a6d2ab31d2adac/data/g.csv), it was used to train the model at the time of submitting the manuscript, some minor updates have been made since we submitted the paper, and these updates are included in [The data folder](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/tree/6ec4a08f434a5dc0ae38345fc155a10db0b5ff49/data) )\
[r1_learninig_curves.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/19d4c9834a2bb9521bcfde277eb46e59ded7ae3a/r1/r1_learning_curves.csv): A csv file containing a statistical summary of each step used for producing the learning curves. To get these results the ml_model function in [r1.ipynb](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/r1/r1.ipynb) was run several times with different testing set sizes, each was run for 500 MC-CV splits. Column 'portion_of_data_set' indicates the portion used for training the model. \
[r1_split_stat.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/9ba61b3b2dad50f9deddf955f9303b0adc310fae/r1/r1_split_stat.csv): A csv file containing a statistical summary of each MC-CV step.\
[stat_summ](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b4a0b821ac4d3341ebb8f74178527c816e036641/r1/stat_summ.csv): statistical summary of the model.

## References
<a id="1">[1]</a> 
X. Liu, S. Truppe, G. Meijer and J. Pérez-Ríos, Journal of
Cheminformatics, 2020, 12, 31.
