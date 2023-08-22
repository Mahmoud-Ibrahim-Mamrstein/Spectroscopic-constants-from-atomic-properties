# Spectroscopic constants from atomic properties: a machine learning approach
![Alt text](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/f9e2af4f2267a6294878909900b4c210f42e0df3/Art%20work.jpg)

We present a machine-learning approach toward predicting spectroscopic constants based on atomic properties. After collecting spectroscopic information on diatomics and generating an extensive database, we employ Gaussian process regression to identify the most efficient characterization of molecules to predict the equilibrium distance, vibrational harmonic frequency, and dissociation energy. As a result, we show that it is possible to predict the equilibrium distance with an absolute error of 0.04 Å and vibrational harmonic frequency with an absolute error of 36 $\text{cm}^{-1}$, including only atomic properties. These results can be improved by including prior information on molecular properties leading to an absolute error of 0.02 Å and 28 $\text{cm}^{-1}$ for the equilibrium distance and vibrational harmonic frequency, respectively. In contrast, the dissociation energy is predicted with an absolute error $\lesssim 0.4$ eV. Alongside these results, we prove that it is possible to predict spectroscopic constants of homonuclear molecules from the atomic and molecular properties of heteronuclear. Finally, based on our results, we present a new way to classify diatomic molecules beyond chemical bond properties.

## Code 
In each repository, there is a Readme file that summarizes the model and describes its file contents. All the codes in the repository are presented in Jupyter notebooks and documented via markdown cells and comments. Python 3.9.7 and conda 23.1.0 were used with the following packages

Name|                  Version|                  Build|                  Channel|
--------|               --------|                --------|              --------|
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
matplotlib-venn|           0.11.9|                   pypi_0|    pypi|
scipy|                     1.7.3|            py39h0a974cb_0|                    |
seaborn|                   0.11.2|             pyhd3eb1b0_0|                    |

## Data gathering

In this project spectroscopic constants of diatomic molecules have been gathered from various published books, papers, and online accessible databases (e.g, [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) and [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/))
### Homonuclear molecules:
Most homonuclear molecules' spectroscopic data have been gathered from Huber and Herzberg's book [[1]](#1). The references used to gather Homonuclear molecules' spectroscopic information from sources other than the Huber and Herzbeg book are cited in the references column in the [g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv) and [g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv). files.

### Heteronuclear molecules:

Heteronuclear molecules are gathered from multiple sources, Mainly from [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) and [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) which are based on the Huber and Herzberg book [[1]](#1). [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) was downloaded as a CSV file from the website. A list of these molecules is given in [here](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/4b6ece83b0bcaf2c5a5627ecb45ba02f5a4d9612/data/list%20of%20molecules%20used%20in%20Xiangue%20and%20Jesus%20paper.csv).

Other Heteronuclear molecules were gathered from various published papers. Most are experimental studies, some are review papers, and some are theoretical studies. Theoretical papers usually compare their results with previously published experimental results. These papers helped us to find several experimental studies regarding molecules of theoretical and experimental interest. Both experimental and theoretical studies are cited in In the manuscript and the references column in [g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv) and [g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv).       

## Data cleaning
Experimental data gathered from cited books, papers, or online accessible databases usually does not suffer from data entry errors. However, careful reading of the text describing the raw data was required. Footnotes in the Huber and Herzberg book were useful in eliminating some of the molecules or initiating a search to find more recent experimental results. For instance, AgBi was eliminated from the dataset due to the footnote regarding the calculated value of $r_e$ from the rotational constants. The $\omega_e$ value of HgH was found to be updated several times to higher values than the one found in the Huber and Herzberg book [[1]](#1), [[4]](#4), [[5]](#5). We have found some discrepancies in the experimental values of some molecules for instance $\text{Hg}_2$ [[2]](#2), [[3]](#3) and other cases discussed in detail in the manuscript (e.g, AuF and ZnBr). To overcome the potential susceptibility of the data to errors stemming from variations in experimental techniques or conditions during the measurement of spectroscopic constants, we try to find various experimental studies that agree on the same value of a spectroscopic constant. In case there is a discrepancy in the experimental value of some spectroscopic constants of some molecule (e.g., AuF and ZnBr) we turn to theoretical studies to gain an insight about the most probable value. we were careful in reviewing such cases where there are discrepancies among various reported experimental results to the best of our knowledge, but this exercise requires continuous revisions of the data and monitoring of the most recent experimental studies. The following histogram shows a comparison in the number of references per publishing date between The diatomic molecular spectroscopy database v.01 and v.02. The figure shows that in v.02 of the database more carful review of various experimental studies and gathering of up-to-date data was a major priority for the authors.   
![Alt text](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/2faee4e0d8eed6021f5e83d34015a5aab90fa4d8/data/refrences%20_compare.svg)


To avoid data entry errors, the data set has been reviewed several times during the project by the authors using standard techniques like plotting the data, looking at the distributions of the data points and directly looking at single data points in the CSV files. The data is clean to the best of the author's knowledge.

Other issues regarding the experimental values of the dissociation energies of diatomic molecules are discussed in detail in the [manuscript](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/a68fdfea6013637c6956adae616f566831263065/Manuscript.pdf). 

## Data description
[g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv): contains data used for training and validation. References are provided for each molecule experimental data \
[g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv): contains data used for training and validation in addition to data used for testing. References are provided for each molecule experimental data \
[list of molecules used in Xiangue and Jesus paper.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/ece9ce778381e0e7a83e75dc29c02950d5a4bd62/data/list%20of%20molecules%20used%20in%20Xiangue%20and%20Jesus%20paper.csv): A list of molecules used in Liu et al., 2021 \
[peridic.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/ece9ce778381e0e7a83e75dc29c02950d5a4bd62/data/peridic.csv): contains information from the periodic table for each element

[g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv) and [g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv): A1 and A2 columns contain the mass number of the two atoms making up a molecule. 

Electronic state: electronic state and/or symmetry symbol

$T_e$  (cm $^{-1}$):  minimum electronic energy 

$\omega_e$ (cm $^{-1}$): vibrational constant – first term

$\omega_eX_e$ (cm $^{-1}$): vibrational constant – second term 

$B_e$ (cm $^{-1}$): rotational constant in equilibrium position

$\alpha_e$ (cm $^{-1}$): rotational constant – first term

$D_e$ ($10^{-7}$ cm $^{-1}$): 	centrifugal distortion constant

$R_e$ (\AA): internuclear distance

$D_O^O$ (eV): Dissociation energy

$IP$ (eV): Ionozation potential

lan_act: Indicate the use of group 3 as the group for both lanthanides and actinides

iso: indicates of the use of 0 as the group number of Deuterium and -1 as the group number of Tritium.
 
## Gaussian process regression
We define our data set $D=\{(\textbf{x}_i,y_i)|i=1,\cdot\cdot\cdot,n\}$, where $\textbf{x}_i$ is a feature vector of some dimension $D$ associated with the $i$-th element of the dataset, $y_i$ is a scalar target label, and $n$ is the number of observations, i.e., the number of elements in the dataset. The set of all feature vectors and corresponding labels can be grouped in the random variables $X$ and $\textbf{y}$, respectively, where $X=(\textbf{x}_1,\cdot\cdot\cdot,\textbf{x}_n)$ and $\textbf{y}=(y_1,\cdot\cdot\cdot,y_n)$. Here, $\textbf{y}$ consists of values of molecular properties to be learned. $y_i$ is $R_e$, $\omega_e$, or $D_0^0$ of the $i$-th molecule, whereas $\textbf{x}_i$ is a vector containing atomic or molecular properties of the same molecule.

We are interested in mapping features to target labels via a regression model $y_i=f(\textbf{x}_i)+\varepsilon_i$, where $f(\textbf{x}_i)$ is the regression function, and $\varepsilon_i$ is an additive noise. We further assume that $\varepsilon_i$ follows an independent, identically distributed (i.i.d) Gaussian distribution with variance $\sigma_n^2$.

Gaussian process regression (GPR), assumes a Gaussian process prior $\mathcal{GP}$ over the space of functions. It is specified by a prior mean function $m(\textbf{x})$ and a covariance function(kernel) $k(\textbf{x},\textbf{x}')$. A posterior distribution of the value of $f(\textbf{x}^{\*})$ at some point of interest, $\textbf{x}^{\*}$, is determined through the Bayes theorem. The mean of the resulting predictive posterior distribution, $\mu^{\*}$, is used to obtain a point estimate of the value of $f(\textbf{x}^{\*})$, and its covariance $\Sigma^{\*}$ provides a confidence interval.  

In GPR, the regression model is completely specified by the kernel $k(\textbf{x},\textbf{x}')$.  The kernel is a similarity measure that specifies the correlation between a pair of values $f(\textbf{x})$ and $f(\textbf{x}')$ by only using the distance between a pair of feature vectors $\textbf{x}$ and $\textbf{x}'$ as its input variable. Specifying a kernel, we encode high-level structural assumptions (e.g., smoothness, periodicity, etc.) about the regression function. Here, we focus on the Mat\'ern class which is characterized by a length scale parameter $l$ and the order $\nu$ of the modified Bessel function of the second kind as a hyperparameter. Values of $\nu$  that are suitable for regression applications are 1/2, 3/2, 5/2, and $\infty$.

We encode our physical intuition by specifying a prior mean function $m(\textbf{x})$. A common choice of the prior mean function is $m(\textbf{x})=0$. This choice is satisfactory in most cases, especially in interpolation tasks. However, selecting an appropriate prior mean function can simplify the learning process (delivering better results using fewer data). The mean function can also guide the model for better predictions as $k(\textbf{x}_p,\textbf{x}_q) \rightarrow 0$; this is necessary for extrapolation and interpolation among sparse data points. Further, a model with a specified mean function is more interpretable.

## Monte Carlo cross-validation

To determine the parameters and the hyperparameters, we divide the dataset $D$ into two subsets: $D_{\text{tv}}$ and $D_{\text{test}}$. First, $D_{\text{tv}}$ is used for the training and validation stage, in which we determine the model's hyperparameters. Then, $D_{\text{test}}$, known as the test set, is left out for model final testing and evaluation and does not take any part in determining the parameters nor the hyperparameters of the model. 

To design a model, we choose an $X$ suitable to learn $\textbf{y}$ through a GPR. We then choose a convenient prior mean function $m(X)$ based on physical intuition, alongside the last hyperparameter $\nu \in \{ 1/2,3/2,5/2,\infty \}$ is determined by running four models, each with a possible value of $\nu$, and we chose the one that performs the best on the training data to be the final model. Precisely, a cross-validation (CV) scheme is used to evaluate the performance of each model iteratively: we split $D_{\text{tv}}$ into a training set $D_{\text{train}}$ ($\sim 90 \%$ of $D_{\text{tv}}$) and a validation set $D_{\text{valid}}$. We use $D_{\text{train}}$ to fit the model and determine its parameters by maximizing the log-marginal likelihood. The fitted model is then used to predict the target labels of $D_{\text{valid}}$. We repeat the process with a different split in each iteration such that each element in $D_{\text{tv}}$ has been sampled at least once in both $D_{\text{train}}$ and $D_{\text{valid}}$. After many iterations, we can determine the average performance of the model. We compare the average performance of the four models after the CV process. Finally, We determine the value of $\nu$ to be its value for the best-performing model. 

We adopt a Monte Carlo (MC) splitting scheme to generate the CV splits. Using the MC splitting scheme, we expose the models to various data compositions. To generate a single split, we use stratified sampling. First, we stratify the training set into smaller strata based on the target label. Stratification will be such that molecules in each stratum have values within some lower and upper bounds of the target label (spectroscopic constant) of interest. Then, we sample the validation set so that each stratum is represented. Stratified sampling minimizes the change in the proportions of the data set composition upon MC splitting, ensuring that the trained model can make predictions over the full range of the target variable. Using the Monte Carlo splitting scheme with cross-validation (MC-CV) allows our models to train on $D_{\text{tv}}$ in full, as well as make predictions for each molecule in $D_{\text{tv}}$. In each iteration, $D_{\text{valid}}$ simulates the testing set; thus, by the end of the MC-CV process, it provides an evaluation of the model performance against ~ 90% of the molecules in the data set before the final testing stage. 

![Alt text](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/62b29764facc0fae2687daa4800a927e7ef57a21/MCCV.svg)

 ## References
<a id="1">[1]</a> 
Klaus-Peter Huber. Molecular spectra and molecular structure: IV. Constants of diatomic
molecules. Springer Science & Business Media, 2013.

<a id="2">[2]</a> 
Stefanov, B., 1985. Comment: On the equilibrium of Hg2 molecule. The Journal of chemical physics, 83(5), pp.2621-2621.

<a id="3">[3]</a> 
Hilpert, K. and Jones, R.O., 1985. Reply to ‘‘Comment: On the equilibrium of Hg2 molecule’’. The Journal of Chemical Physics, 83(5), pp.2622-2622.


<a id="3">[4]</a>  J. Dufayard, B. Majournat and O. Nedelec, Chem. Phys. 128, 537 (1988).

<a id="3">[5]</a> W. C. Stwalley, J. Chem. Phys. 63, 3062 (1975)
