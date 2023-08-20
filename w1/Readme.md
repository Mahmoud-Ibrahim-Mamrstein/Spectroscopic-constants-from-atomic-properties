# Model w1
Model w1 is the best-performing model of Liu et al.[[1]](#1). It is characterized by six features, including atomic and molecular properties. Namely, the groups and periods of the constituent atoms, the average group, $\bar{g}=(g^{iso}_1+g^{iso}_2)/2$, and $R_e^{-1}$. $g^{iso}$ encodes isotopic information, such that $g_i^{iso}=0$ for deuterium, $g_i^{iso}=-1$ for tritium, and $g_i^{iso}=g_i$ for every other element. The prior mean function is set to zero.
## Files description 
[w1.ipynb](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/w1/w1.ipynb) : A Jupiter notebook of the w1 model\
[w1_gr_expand_pred.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/w1/w1_gr_expand_pred.csv): A csv file containing prediction of the w1 model. Columns we_test_preds, we_test_std, we_train_preds, and we_train_std include the testing predictions and standard deviations and training predictions and standard deviations for each molecule in the training validation set.\
[w1_learninig_curves.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/19d4c9834a2bb9521bcfde277eb46e59ded7ae3a/w1/w1_learning_curves.csv): A csv file containing a statistical summary of each step used for producing the learning curves. To get these results the ml_model function in [w1.ipynb](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/675a7ef80706594b614d08ff2983706efb1f9aab/w1/w1.ipynb) was run several times with different testing set sizes, each was run for 500 MC-CV splits. Column 'portion_of_data_set' indicate the portion used for training the model. \
[w1_split_stat.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/9ba61b3b2dad50f9deddf955f9303b0adc310fae/w1/w1_split_stat.csv): A csv file containing a statistical summary of each MC-CV step.\
[stat_summ](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b4a0b821ac4d3341ebb8f74178527c816e036641/w1/stat_summ.csv): statistical summary of the model. \

## References
<a id="1">[1]</a> 
X. Liu, S. Truppe, G. Meijer and J. Pérez-Ríos, Journal of
Cheminformatics, 2020, 12, 31.
