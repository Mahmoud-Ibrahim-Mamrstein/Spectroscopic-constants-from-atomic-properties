# Spectroscopic constants from atomic properties: a machine learning approach
![Alt text](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/f9e2af4f2267a6294878909900b4c210f42e0df3/Art%20work.jpg)

We present a machine-learning approach toward predicting spectroscopic constants based on atomic properties. After collecting spectroscopic information on diatomics and generating an extensive database, we employ Gaussian process regression to identify the most efficient characterization of molecules to predict the equilibrium distance, vibrational harmonic frequency, and dissociation energy. As a result, we show that it is possible to predict the equilibrium distance with an absolute error of 0.04 Å and vibrational harmonic frequency with an absolute error of 36 $\text{cm}^{-1}$, including only atomic properties. These results can be improved by including prior information on molecular properties leading to an absolute error of 0.02 Å and 28 $\text{cm}^{-1}$ for the equilibrium distance and vibrational harmonic frequency, respectively. In contrast, the dissociation energy is predicted with an absolute error $\lesssim 0.4$ eV. Alongside these results, we prove that it is possible to predict spectroscopic constants of homonuclear molecules from the atomic and molecular properties of heteronuclear. Finally, based on our results, we present a new way to classify diatomic molecules beyond chemical bond properties.

## Code and data description as in the manuscript 
In each repository, there is a Readme file that summarizes the model and describes its file contents. All the codes in the repository are presented in Jupyter notebooks and documented via markdown cells and comments. All the data are contained in csv files, the data is described in [data/Readme.md](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/03d76f1438373fbb8f8a21b718b0947ffcfbd3b5/data/Readme.md)   
## Gaussian process regression
We define our data set $D=\{(\textbf{x}_i,y_i)|i=1,\cdot\cdot\cdot,n\}$, where $\textbf{x}_i$ is a feature vector of some dimension $D$ associated with the $i$-th element of the dataset, $y_i$ is a scalar target label, and $n$ is the number of observations, i.e., the number of elements in the dataset. The set of all feature vectors and corresponding labels can be grouped in the random variables $X$ and $\textbf{y}$, respectively, where $X=(\textbf{x}_1,\cdot\cdot\cdot,\textbf{x}_n)$ and $\textbf{y}=(y_1,\cdot\cdot\cdot,y_n)$. Here, $\textbf{y}$ consists of values of molecular properties to be learned. $y_i$ is $R_e$, $\omega_e$, or $D_0^0$ of the $i$-th molecule, whereas $\textbf{x}_i$ is a vector containing atomic or molecular properties of the same molecule.

We are interested in mapping features to target labels via a regression model $y_i=f(\textbf{x}_i)+\varepsilon_i$, where $f(\textbf{x}_i)$ is the regression function, and $\varepsilon_i$ is an additive noise. We further assume that $\varepsilon_i$ follows an independent, identically distributed (i.i.d) Gaussian distribution with variance $\sigma_n^2$
Gaussian process regression (GPR), assumes a Gaussian process prior $\mathcal{GP}$ over the space of functions. It is specified by a mean function $m(\textbf{x})$ and a covariance function(kernel) $k(\textbf{x},\textbf{x}')$. A posterior distribution of the value of $f(\textbf{x}^{\*})$ at some point of interest, $\textbf{x}^{\*}$, is determined through the Bayes theorem. The mean of the resulting predictive posterior distribution, $\mu^{\*}$, is used to obtain a point estimate of the value of $f(\textbf{x}^{\*})$, and its covariance $\Sigma^{\*}$ provides a confidence interval.  

In GPR, the regression model is completely specified by the kernel $k(\textbf{x},\textbf{x}')$.  The kernel is a similarity measure that specifies the correlation between a pair of values $f(\textbf{x})$ and $f(\textbf{x}')$ by only using the distance between a pair of feature vectors $\textbf{x}$ and $\textbf{x}'$ as its input variable. Specifying a kernel, we encode high-level structural assumptions (e.g., smoothness, periodicity, etc.) about the regression function. Here, we focus on the Mat\'ern class which is characterized by a length scale parameter $l$ and the order $\nu$ of the modified Bessel function of the second kind as a hyperparameter. Values of $\nu$  that are suitable for regression applications are 1/2, 3/2, 5/2, and $\infty$ 

## Monte Carlo cross-validation

To determine the parameters and the hyperparameters, we divide the dataset $D$ into two subsets: $D_{\text{tv}}$ and $D_{\text{test}}$. First, $D_{\text{tv}}$ is used for the training and validation stage, in which we determine the model's hyperparameters. Then, $D_{\text{test}}$, known as the test set, is left out for model final testing and evaluation and does not take any part in determining the parameters nor the hyperparameters of the model. 

To design a model, we choose an $X$ suitable to learn $\textbf{y}$ through a GPR. We then choose a convenient prior mean function $m(X)$ based on physical intuition, alongside the last hyperparameter $\nu \in \{ 1/2,3/2,5/2,\infty \}$ is determined by running four models, each with a possible value of $\nu$, and we chose the one that performs the best on the training data to be the final model. Precisely, a cross-validation (CV) scheme is used to evaluate the performance of each model iteratively: we split $D_{\text{tv}}$ into a training set $D_{\text{train}}$ ($\sim 90 \%$ of $D_{\text{tv}}$) and a validation set $D_{\text{valid}}$. We use $D_{\text{train}}$ to fit the model and determine its parameters by maximizing the log-marginal likelihood. The fitted model is then used to predict the target labels of $D_{\text{valid}}$. We repeat the process with a different split in each iteration such that each element in $D_{\text{tv}}$ has been sampled at least once in both $D_{\text{train}}$ and $D_{\text{valid}}$. After many iterations, we can determine the average performance of the model. We compare the average performance of the four models after the CV process. Finally, We determine the value of $\nu$ to be its value for the best-performing model. 

We adopt a Monte Carlo (MC) splitting scheme to generate the CV splits. Using the MC splitting scheme, we expose the models to various data compositions. To generate a single split, we use stratified sampling. First, we stratify the training set into smaller strata based on the target label. Stratification will be such that molecules in each stratum have values within some lower and upper bounds of the target label (spectroscopic constant) of interest. Then, we sample the validation set so that each stratum is represented. Stratified sampling minimizes the change in the proportions of the data set composition upon MC splitting, ensuring that the trained model can make predictions over the full range of the target variable. Using the Monte Carlo splitting scheme with cross-validation (MC-CV) allows our models to train on $D_{\text{tv}}$ in full, as well as make predictions for each molecule in $D_{\text{tv}}$. In each iteration, $D_{\text{valid}}$ simulates the testing set; thus, by the end of the MC-CV process, it provides an evaluation of the model performance against ~ 90% of the molecules in the data set before the final testing stage. 

![Alt text](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/62b29764facc0fae2687daa4800a927e7ef57a21/MCCV.svg)
