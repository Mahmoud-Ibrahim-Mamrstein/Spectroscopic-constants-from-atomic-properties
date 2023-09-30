# Spectroscopic constants from atomic properties: a machine learning approach
![Alt text](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/f9e2af4f2267a6294878909900b4c210f42e0df3/Art%20work.jpg)

We present a machine-learning approach toward predicting spectroscopic constants based on atomic properties. After collecting spectroscopic information on diatomics and generating an extensive database, we employ Gaussian process regression to identify the most efficient characterization of molecules to predict the equilibrium distance, vibrational harmonic frequency, and dissociation energy. As a result, we show that it is possible to predict the equilibrium distance with an absolute error of 0.04 Å and vibrational harmonic frequency with an absolute error of 36 $\text{cm}^{-1}$, including only atomic properties. These results can be improved by including prior information on molecular properties leading to an absolute error of 0.02 Å and 28 $\text{cm}^{-1}$ for the equilibrium distance and vibrational harmonic frequency, respectively. In contrast, the dissociation energy is predicted with an absolute error $\lesssim 0.4$ eV. Alongside these results, we prove that it is possible to predict spectroscopic constants of homonuclear molecules from heteronuclear's atomic and molecular properties. Finally, based on our results, we present a new way to classify diatomic molecules beyond chemical bond properties.

## Code 
In each repository, there is a Readme file that summarizes the model and describes its file contents. All the codes in the repository are presented in Jupyter notebooks and documented via markdown cells and comments. Python 3.9.7 and conda 23.1.0 were used with the following packages

Name|                  Version|                  Build|                  Channel|
--------|               --------|                  --------|            --------|
anaconda|                  custom|                   py39_1|                    | 
pandas|                    1.4.2|            py39hd77b12b_0|                    |
numpy|                     1.21.5|           py39h7a0a035_1|
numpy-base|                1.21.5|           py39hca35cd5_1|                    |
numpydoc|                  1.2|                pyhd3eb1b0_0|
scikit-learn|              1.0.2|            py39hf11a4ad_1|                    |
scikit-learn-intelex|      2021.5.0|         py39haa95532_0|
matplotlib|                3.5.1|            py39haa95532_1|                    |
matplotlib-base|           3.5.1|            py39hd77b12b_1|                    |
matplotlib-inline|         0.1.2|              pyhd3eb1b0_2|                    |
matplotlib-venn|           0.11.9|                   pypi_0|                pypi|
scipy|                     1.7.3|            py39h0a974cb_0|                    |
seaborn|                   0.11.2|             pyhd3eb1b0_0|                    |
scikit-optimize|          0.9.0|
## Data gathering

In this project spectroscopic constants of diatomic molecules have been gathered from various published books, papers, and online accessible databases (e.g., [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) and [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/))
### Homonuclear molecules:
Most homonuclear molecules' spectroscopic data have been gathered from Huber and Herzberg's book [[1]](#1). The references used to collect homonuclear molecules' spectroscopic information from sources other than the Huber and Herzberg book are cited in the references column in the [g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv) and [g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv). files.

### Heteronuclear molecules:

Heteronuclear molecules were gathered from multiple sources, mainly from [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) and [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) which are based on the Huber and Herzberg book [[1]](#1). [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) was downloaded as a CSV file from the website. A list of these molecules is given in [here](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/4b6ece83b0bcaf2c5a5627ecb45ba02f5a4d9612/data/list%20of%20molecules%20used%20in%20Xiangue%20and%20Jesus%20paper.csv).

Other heteronuclear molecules were gathered from various published papers. Most are experimental studies, some are review papers, and some are theoretical. Theoretical papers usually compare their results with previously published experimental results. These papers helped us to find several experimental studies regarding molecules of theoretical and experimental interest. Both experimental and theoretical studies are cited in the manuscript and the references column in [g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv) and [g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv).       

## Data cleaning
Experimental data from cited books, papers, or online accessible databases usually does not suffer from data entry errors. However, careful reading of the text describing the raw data was required. Footnotes in the Huber and Herzberg book helped eliminate some of the molecules or initiate a search to find more recent experimental results. For instance, AgBi was eliminated from the dataset due to the footnote regarding the calculated value of $R_e$ from the rotational constants. The $R_e$ values of XeCl and HgCl are uncertain, as listed in [[1]](#1). We have found some discrepancies in the experimental values of some molecules, for instance, $\text{Hg}_2$ [[2]](#2), [[3]](#3), and other cases discussed in detail in the manuscript (e.g., AuF and ZnBr). To overcome the potential susceptibility of the data to errors stemming from variations in experimental techniques or conditions during the measurement of spectroscopic constants, we try to find various experimental studies that agree on the same value of a spectroscopic constant. If there is a discrepancy in the experimental value of some spectroscopic constants of some molecule (e.g., AuF and ZnBr) we turn to theoretical studies to gain an insight about the most probable value. To the best of our knowledge, we were careful in reviewing such cases where there are discrepancies among various reported experimental results. However, this exercise requires continuous revisions of the data and monitoring of the most recent experimental studies. The following histogram shows a comparison in the number of references per publishing date between [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) v.01 and v.02. The figure shows that in v.02 of the database, a more careful review of various published experimental and theoretical studies and gathering of up-to-date data was a significant priority for the authors.   
![Alt text](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/71bebc2184c9a4746bf4c106f14a8287386f23cf/refrences%20_compare.svg)

During the development of this work, we have realized that, historically, uncertainties about the dissociation energy experimental values had restrained the development of empirical relations connecting them to other atomic and molecular properties and have led several authors to focus their efforts on the $\omega_e$ - $R_e$ relation [[8]](#8), [[9]](#9), [[10]](#10). The data used to train model d1 is primarily collected from Huber and Herzberg's constants of diatomic molecules, first published in 1979 [[1]](#1). More recently, Fu et al. used an ML model to predict dissociation energies for diatomic molecules, exploiting microscopic and macroscopic properties[[6]](#6). They tested their model against CO and highlighted that the reported experimental dissociation energy in the literature had increased by 100 kcal/mol over the course of 78 years from 1936 to 2014 [[4]](#4), [[9]](#9), [[10]](#10) (in Table 1 of Ref.[[4]](#4). Unlike experimental values of $R_e$ and $\omega_e$, since 1980, a significant number of $D_0^0$ values have been updated [[5]](#5). To name a few, MgD, MgBr, MgO, CaCl. CaO, SrI, SrO, TiS, NbO, AgF, AgBr, and BrF all have their experimental values updated with at least  $\pm 2.3 \ \text{kcal/mol}$ difference from their values in Huber and Herzberg [[1]](#1), [[5]](#5). Moreover, the uncertainties in $D_0^0$ experimental values are not within chemical accuracy for some molecules. For instance, MgH, CaCl, CaO, CaS, SrH, BaO, BaS, ScF, Tif, NbO, and BrF have uncertainties ranging from $\pm 1 \ \text{kcal/mol} \ \text{up to} \pm 8 \  \text{kcal/mol}$ [[5]](#5). Unlike $R_e$ and $\omega_e$, it is most likely that uncertainties around $D_0^0$ experimental values drive from various systematic effects.


To avoid data entry errors, the authors reviewed the data set several times during the project using standard techniques like plotting the data, looking at the distributions of the data points, and directly looking at single data points in the CSV files. The data is clean to the best of the author's knowledge.


## Data description
[g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv): contains data used for training and validation. References are provided for each molecule experimental data \
[g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv): contains data used for training and validation in addition to data used for testing. References are provided for each molecule experimental data \
[list of molecules used in Xiangue and Jesus paper.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/ece9ce778381e0e7a83e75dc29c02950d5a4bd62/data/list%20of%20molecules%20used%20in%20Xiangue%20and%20Jesus%20paper.csv): A list of molecules used in Liu et al., 2021 \
[peridic.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/ece9ce778381e0e7a83e75dc29c02950d5a4bd62/data/peridic.csv): contains information from the periodic table for each element

[g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv) and [g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv): A1 and A2 columns contain the mass number of the two atoms making up a molecule. 


[all_new_data.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/8fb6627a3cf221a32150bc9b43dcf659b3174bb7/data/all_new_data.csv): Conssists of all the newly gatherd data with refrences and authors' notes
Electronic state: electronic state and/or symmetry symbol

$T_e$  (cm $^{-1}$):  Minimum electronic energy 

$\omega_e$ (cm $^{-1}$): Vibrational constant – first term

$\omega_ex_e$ (cm $^{-1}$): Vibrational constant – second term 

$B_e$ (cm $^{-1}$): Rotational constant in equilibrium position

$\alpha_e$ (cm $^{-1}$): Rotational constant – first term

$D_e$ ($10^{-7}$ cm $^{-1}$): 	Centrifugal distortion constant

$R_e$ (Å): Internuclear distance

$D_O^O$ (eV): Dissociation energy

$IP$ (eV): Ionization potential

lan_act: Indicates the use of group 3 as the group for both lanthanides and actinides

iso: Indicates of the use of 0 as the group number of deuterium and -1 as the group number of tritium.
 
## Gaussian process regression
We define our data set as $D=\{(\textbf{x}_i,y_i)|i=1,\cdot\cdot\cdot,n\}$, where $\textbf{x}_i$ is a feature vector of some dimension $D$ associated with the $i$-th element of the dataset, $y_i$ is a scalar target label, and $n$ is the number of observations--the number of elements in the dataset. The set of all feature vectors and corresponding labels can be grouped in random variables $X$ and $\textbf{y}$, respectively, where $X=(\textbf{x}_1,\cdot\cdot\cdot,\textbf{x}_n)$ and $\textbf{y}=(y_1,\cdot\cdot\cdot,y_n)$. Here, $\textbf{y}$ consists of values of molecular properties to be learned. $y_i$ is $R_e$, $\omega_e$, or $D_0^0$ of the $i$-th molecule, whereas $\textbf{x}_i$ is a vector containing atomic or molecular properties of the same molecule.

We are interested in mapping features to target labels via a regression model $y_i=f(\textbf{x}_i)+\varepsilon_i$, where $f(\textbf{x}_i)$ is the regression function, and $\varepsilon_i$ is an additive noise. We further assume that $\varepsilon_i$ follows an independent, identically distributed (i.i.d) Gaussian distribution with variance $\sigma_n^2$.

Gaussian process regression (GPR) assumes a Gaussian process prior $\mathcal{GP}$ over the space of functions. It is specified by a prior mean function $m(\textbf{x})$ and a covariance function(kernel) $k(\textbf{x},\textbf{x}')$. A posterior distribution of the value of $f(\textbf{x}^{\*})$ at some point of interest, $\textbf{x}^{\*}$, is determined through the Bayes theorem. The mean of the resulting predictive posterior distribution, $\mu^{\*}$, is used to obtain a point estimate of the value of $f(\textbf{x}^{\*})$, and its covariance $\Sigma^{\*}$ provides a confidence interval.  

In GPR, the regression model is completely specified by the kernel $k(\textbf{x},\textbf{x}')$. The kernel is a similarity measure that specifies the correlation between a pair of values $f(\textbf{x})$ and $f(\textbf{x}')$ using the distance between a pair of feature vectors $\textbf{x}$ and $\textbf{x}'$ as its input variable. Specifying a kernel, we encode high-level structural assumptions (e.g., smoothness, periodicity) about the regression function. Here, we focus on the Mat\'ern class characterized by a length scale parameter $l$ and the order $\nu$ of the modified Bessel function of the second kind as a hyperparameter. Values of $\nu$ suitable for regression applications are 1/2, 3/2, 5/2, and $\infty$.

We encode our physical intuition by specifying a prior mean function $m(\textbf{x})$. A common choice of the prior mean function is $m(\textbf{x})=0$. This choice is satisfactory in most cases, especially in interpolation tasks. However, selecting an appropriate prior mean function can simplify the learning process (delivering better results using fewer data). The mean function can also guide the model for better predictions as $k(\textbf{x}_p,\textbf{x}_q) \rightarrow 0$; this is necessary for extrapolation and interpolation among sparse data points. Further, a model with a specified mean function is more interpretable.

## Monte Carlo cross-validation

To determine the parameters and hyperparameters, we divide the dataset $D$ into two subsets: $D_{\text{tv}}$ and $D_{\text{test}}$. First, $D_{\text{tv}}$ is used for the training and validation stage, in which we determine the model's hyperparameters. Then, $D_{\text{test}}$, known as the test set, is left out for model final testing and evaluation and does not take any part in determining the parameters nor the hyperparameters of the model. 

To design a model, we choose an $X$ suitable to learn $\textbf{y}$ through a GPR. We then choose a convenient prior mean function $m(X)$ based on physical intuition, alongside the last hyperparameter $\nu \in \{ 1/2,3/2,5/2,\infty \}$ is determined by running four models, each with a possible value of $\nu$, and we chose the one that performs the best on the training data to be the final model. Precisely, a cross-validation (CV) scheme is used to evaluate the performance of each model iteratively: we split $D_{\text{tv}}$ into a training set $D_{\text{train}}$ ($\sim 90 \%$ of $D_{\text{tv}}$) and a validation set $D_{\text{valid}}$. We use $D_{\text{train}}$ to fit the model and determine its parameters by maximizing the log-marginal likelihood. The fitted model is then used to predict the target labels of $D_{\text{valid}}$. We repeat the process with a different split in each iteration such that each element in $D_{\text{tv}}$ has been sampled at least once in both $D_{\text{train}}$ and $D_{\text{valid}}$. After many iterations, we can determine the average performance of the model. We compare the average performance of the four models after the CV process. Finally, We determine the value of $\nu$ to be its value for the best-performing model. 

We adopt a Monte Carlo (MC) splitting scheme to generate the CV splits. We expose the models to various data compositions using the MC splitting scheme. To generate a single split, we use stratified sampling. First, we stratify the training set into smaller strata based on the target label. Stratification will be such that molecules in each stratum have values within some lower and upper bounds of the target label (spectroscopic constant) of interest. Then, we sample the validation set so that each stratum is represented. Stratified sampling minimizes the change in the proportions of the data set composition upon MC splitting, ensuring that the trained model can make predictions over the full range of the target variable. Using the Monte Carlo splitting scheme with cross-validation (MC-CV) allows our models to train on $D_{\text{tv}}$ in full, as well as make predictions for each molecule in $D_{\text{tv}}$. In each iteration, $D_{\text{valid}}$ simulates the testing set; thus, by the end of the MC-CV process, it provides an evaluation of the model performance against ~ 90% of the molecules in the data set before the final testing stage. 

![Alt text](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/62b29764facc0fae2687daa4800a927e7ef57a21/MCCV.svg)
## Choosing features
The number of features and the features themselves are hyperparameters. Our philosophy is to keep the features as simple as possible and the number of features as low as possible. The groups and periods of the diatomic molecules' constituent atoms and the reduced mass of the molecules are used as features in all the newly developed models. The features are kept in their simplest forms. At the same time, the square root of the reduced mass is used to be on the same scale as the other features, which takes values that lay within the values of 1 and 18 for groups and 1 and 7 for periods. Those five numbers ($p1$, $p2$, $g1$, $g2$ and $\sqrt(mu)$) discriminates between all diatomics including istoplogues. The chief reason periodic properties are used as features in our models is due to the strong relation between spectroscopic constants and periodicity found in the literature, as highlighted in  [[13]](#13).    
 ## References
<a id="1">[1]</a> 
Klaus-Peter Huber. Molecular spectra and molecular structure: IV. Constants of diatomic
molecules. Springer Science & Business Media, 2013.

<a id="2">[2]</a> 
Stefanov, B., 1985. Comment: On the equilibrium of Hg2 molecule. The Journal of chemical physics, 83(5), pp.2621-2621.

<a id="3">[3]</a> 
Hilpert, K. and Jones, R.O., 1985. Reply to ‘‘Comment: On the equilibrium of Hg2 molecule’’. The Journal of Chemical Physics, 83(5), pp.2622-2622.

<a id="4">[4]</a> Fu, J., Wan, Z., Yang, Z., Liu, L., Fan, Q., Xie, F., Zhang, Y. and Ma, J., 2022. Combining ab initio and machine learning method to improve the prediction of diatomic vibrational energies. International Journal of Quantum Chemistry, 122(18), p.e26953.

<a id="5">[5]</a> Luo, Y.R., 2007. Comprehensive handbook of chemical bond energies. CRC press.

<a id="6">[6]</a>Badger, R.M., 1934. A relation between internuclear distances and bond force constants. The Journal of Chemical Physics, 2(3), pp.128-131.

<a id="7">[7] Jhung, K.S., Kim, I.H., Oh, K.H., Hahn, K.B. and Jhung, K.H.C., 1990. Universal nature of diatomic potentials. Physical Review A, 42(11), p.6497.

<a id="8">[8] The determination of internuclear distances and of dissociation energies from force constants

<a id="9">[9] Kȩpa, R., Ostrowska-Kopeć, M., Piotrowska, I., Zachwieja, M., Hakalla, R., Szajna, W. and Kolek, P., 2014. Ångström (B1Σ+→ A1Π) 0–1 and 1–1 bands in isotopic CO molecules: further investigations. Journal of Physics B: Atomic, Molecular and Optical Physics, 47(4), p.045101.

<a id="10">[10] Vol'Kenshtein, M.V., 1955. Structure and physical properties of molecules. Izd. Inostr. Lit., Moscow.

<a id="11">[11]	arXiv:2308.08933 [physics.chem-ph]
