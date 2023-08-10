# Data

## Data gathering

In this project spectroscopic constants of diatomic molecules have been gathered from various published books, papers, and online accessible databases (e.g, [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) and [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/))
### Homonuclear molecules:
Most homonuclear molecules' spectroscopic data have been gathered from Huber and Herzberg's book [[1]](#1). The references used to gather Homonuclear molecules' spectroscopic information from sources other than the Huber and Herzbeg book are cited in the references column in the g.csv and g-test-2.csv files.

### Heteronuclear molecules:

Heteronuclear molecules are gathered from multiple sources, Mainly from [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) and [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) which are based on the Huber and Herzberg book [[1]](#1). 
\\
[The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) was downloaded as a CSV file from the website. A list of these molecules is given in [here](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/4b6ece83b0bcaf2c5a5627ecb45ba02f5a4d9612/data/list%20of%20molecules%20used%20in%20Xiangue%20and%20Jesus%20paper.csv).

Other Heteronuclear molecules were gathered from various published papers. Some are review papers, some are experimental studies and some are theoretical studies. Theoretical papers usually compare their results with previously published experimental results. These papers helped us to find several experimental studies regarding molecules of theoretical and experimental interest. Both experimental and theoretical studies are cited in In the manuscript and the references column in [g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv) and [g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv).       

## Data cleaning
Experimental data gathered from cited books papers or online accessible databases in general does not suffer errors. However, some of the data required careful reading of the text accompanying the raw data was required. Footnotes in the Huber and Herzberg book were useful in eliminating some of the molecules or initiating a search to find more recent experimental results. For instance, AgBi was eliminated from the dataset due to the footnote regarding the calculated value of $r_e$ from the rotational constants. The $\omega_e$ value of HgCl was found to be updated several times to higher values than the one found in the Huber and Herzberg book [[1]](#1). We have found some discrepancies in the experimental values of some molecules for instance $\text{Hg}_2$ [[2]](#2), [[3]](#3) and other cases discussed in detail in the manuscript (e.g, AuF and ZnBr). The authors were careful in reviewing such cases where there are discrepancies among various reported experimental results to the best of their knowledge. 

To avoid data entry errors, the data set has been reviewed several times during the project by the authors using standard techniques like plotting the data, looking at the distributions of the data points and directly looking at single data points in the CSV files. The data is clean to the best of the author's knowledge.

Other issues regarding the experimental values of the dissociation energies of diatomic molecules are discussed in detail in the manuscript. 
   
 ## References
<a id="1">[1]</a> 
Klaus-Peter Huber. Molecular spectra and molecular structure: IV. Constants of diatomic
molecules. Springer Science & Business Media, 2013.

<a id="2">[2]</a> 
Stefanov, B., 1985. Comment: On the equilibrium of Hg2 molecule. The Journal of chemical physics, 83(5), pp.2621-2621.

<a id="3">[3]</a> 
Hilpert, K. and Jones, R.O., 1985. Reply to ‘‘Comment: On the equilibrium of Hg2 molecule’’. The Journal of Chemical Physics, 83(5), pp.2622-2622.






g.csv : contains data used for traianing and validation. Refrences are provided for each molecule experimental data \
g-test-2.csv: contains data used for traianing and validation in addition to data used for testing. Refrences are provided for each molecule experimental data \
list of molecules used in Xiangue and Jesus paper.csv : A list of molecules used in Liu et al., 2021 \
peridic.csv : contains information from the periodic table for each element 
