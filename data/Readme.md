# Data gathering

In this project spectroscopic constants of diatomic molecules have been gathered from various published books, papers, and online accessible databases (e.g, [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) and [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/))
 ### Homonuclear molecules:
Most homonuclear molecules' spectroscopic data have been gathered from Huber and Herzberg's book [[1]](#1). The references used to gather Homonuclear molecules' spectroscopic information from sources other than the Huber and Herzbeg book are cited in the references column in the g.csv and g-test-2.csv files.

## Heteronuclear molecules:

Heteronuclear molecules are gathered from multiple sources, Mainly from [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) and [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) which are based on the Huber and Herzberg book [[1]](#1). 
\\
[The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) was downloaded as a CSV file from the website. A list of these molecules is given in [here](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/4b6ece83b0bcaf2c5a5627ecb45ba02f5a4d9612/data/list%20of%20molecules%20used%20in%20Xiangue%20and%20Jesus%20paper.csv).

Other Heteronuclear molecules were gathered from various published papers. Some are review papers, some are experimental studies and some are theoretical studies. Theoretical papers usually compare their results with previously published experimental results. These papers helped us to find several experimental studies regarding molecules of theoretical and experimental interest. Both experimental and theoretical studies are cited in In the manuscript and the references column in      

 ## References
<a id="1">[1]</a> 
Klaus-Peter Huber. Molecular spectra and molecular structure: IV. Constants of diatomic
molecules. Springer Science & Business Media, 2013.











g.csv : contains data used for traianing and validation. Refrences are provided for each molecule experimental data \
g-test-2.csv: contains data used for traianing and validation in addition to data used for testing. Refrences are provided for each molecule experimental data \
list of molecules used in Xiangue and Jesus paper.csv : A list of molecules used in Liu et al., 2021 \
peridic.csv : contains information from the periodic table for each element 
