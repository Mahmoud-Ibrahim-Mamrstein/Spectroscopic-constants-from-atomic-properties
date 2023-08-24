## Data description
[g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv): contains data used for training and validation. References are provided for each molecule experimental data \
[g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv): contains data used for training and validation in addition to data used for testing. References are provided for each molecule experimental data \
[list of molecules used in Xiangue and Jesus paper.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/ece9ce778381e0e7a83e75dc29c02950d5a4bd62/data/list%20of%20molecules%20used%20in%20Xiangue%20and%20Jesus%20paper.csv): A list of molecules used in Liu et al., 2021 \
[peridic.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/ece9ce778381e0e7a83e75dc29c02950d5a4bd62/data/peridic.csv): contains information from the periodic table for each element

[g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv) and [g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv): A1 and A2 columns contain the mass number of the two atoms making up a molecule. 


[all_new_data.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/8fb6627a3cf221a32150bc9b43dcf659b3174bb7/data/all_new_data.csv): Conssists of all the newly gatherd data with refrences and authors' notes
Electronic state: electronic state and/or symmetry symbol

$T_e$  (cm $^{-1}$):  minimum electronic energy 

$\omega_e$ (cm $^{-1}$): vibrational constant – first term

$\omega_ex_e$ (cm $^{-1}$): vibrational constant – second term 

$B_e$ (cm $^{-1}$): rotational constant in equilibrium position

$\alpha_e$ (cm $^{-1}$): rotational constant – first term

$D_e$ ($10^{-7}$ cm $^{-1}$): 	centrifugal distortion constant

$R_e$ (\AA): internuclear distance

$D_O^O$ (eV): Dissociation energy

$IP$ (eV): Ionozation potential

lan_act: Indicates the use of group 3 as the group for both lanthanides and actinides

iso: indicates of the use of 0 as the group number of Deuterium and -1 as the group number of Tritium.
 
## Data gathering

In this project spectroscopic constants of diatomic molecules have been gathered from various published books, papers, and online accessible databases (e.g, [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) and [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/))
### Homonuclear molecules:
Most homonuclear molecules' spectroscopic data have been gathered from Huber and Herzberg's book [[1]](#1). The references used to gather Homonuclear molecules' spectroscopic information from sources other than the Huber and Herzbeg book are cited in the references column in the [g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv) and [g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv). files.

### Heteronuclear molecules:

Heteronuclear molecules were gathered from multiple sources, Mainly from [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) and [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) which are based on the Huber and Herzberg book [[1]](#1). [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) was downloaded as a CSV file from the website. A list of these molecules is given in [here](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/4b6ece83b0bcaf2c5a5627ecb45ba02f5a4d9612/data/list%20of%20molecules%20used%20in%20Xiangue%20and%20Jesus%20paper.csv).

Other Heteronuclear molecules were gathered from various published papers. Most are experimental studies, some are review papers, and some are theoretical studies. Theoretical papers usually compare their results with previously published experimental results. These papers helped us to find several experimental studies regarding molecules of theoretical and experimental interest. Both experimental and theoretical studies are cited in In the manuscript and the references column in [g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv) and [g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv).       

## Data cleaning
Experimental data gathered from cited books, papers, or online accessible databases usually does not suffer from data entry errors. However, careful reading of the text describing the raw data was required. Footnotes in the Huber and Herzberg book were useful in eliminating some of the molecules or initiating a search to find more recent experimental results. For instance, AgBi was eliminated from the dataset due to the footnote regarding the calculated value of $r_e$ from the rotational constants. The $\omega_e$ value of HgH was found to be updated several times to higher values than the one found in the Huber and Herzberg book [[1]](#1), [[4]](#4), [[5]](#5). We have found some discrepancies in the experimental values of some molecules for instance $\text{Hg}_2$ [[2]](#2), [[3]](#3) and other cases discussed in detail in the manuscript (e.g, AuF and ZnBr). To overcome the potential susceptibility of the data to errors stemming from variations in experimental techniques or conditions during the measurement of spectroscopic constants, we try to find various experimental studies that agree on the same value of a spectroscopic constant. In case there is a discrepancy in the experimental value of some spectroscopic constants of some molecule (e.g., AuF and ZnBr) we turn to theoretical studies to gain an insight about the most probable value. we were careful in reviewing such cases where there are discrepancies among various reported experimental results to the best of our knowledge, but this exercise requires continuous revisions of the data and monitoring of the most recent experimental studies. The following histogram shows a comparison in the number of references per publishing date between [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de/index.php) v.01 and v.02. The figure shows that in v.02 of the database, a more careful review of various published experimental and theoretical studies and gathering of up-to-date data was a major priority for the authors.   
![Alt text](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/71bebc2184c9a4746bf4c106f14a8287386f23cf/refrences%20_compare.svg)

During the development of this work, we have realized that, historically, uncertainties about the dissociation energy experimental values had restrained the development of empirical relations connecting them to other atomic and molecular properties and have led several authors to focus their efforts on the $\omega_e$ - $R_e$ relation [[8]](#8), [[9]](#9), [[10]](#10). The data used to train model d1 is primarily collected from Huber and Herzberg's constants of diatomic molecules, first published in 1979 [[1]](#1).More recently, Fu et al. used an ML model to predict dissociation energies for diatomic molecules, exploiting microscopic and macroscopic properties[[6]](#6). They tested their model against CO and highlighted that the reported experimental dissociation energy in the literature had increased by 100 kcal/mol over the course of 78 years from 1936 to 2014 [[6]](#6), [[12]](#12), [[12]](#12) (in Table 1 of Ref.[[6]](#6). Unlike experimental values of $R_e$ and $\omega_e$, since 1980, a significant number of $D_0^0$ values have been updated [[7]](#7). To name a few, MgD, MgBr, MgO, CaCl. CaO, SrI, SrO, TiS, NbO, AgF, AgBr, and BrF all have their experimental values updated with at least  $\pm 2.3 \ \text{kcal/mol}$ difference from their values in Huber and Herzberg [[1]](#1), [[7]](#7). Moreover, for some molecules, the uncertainties in $D_0^0$ experimental values are not within chemical accuracy. For instance, MgH, CaCl, CaO, CaS, SrH, BaO, BaS, ScF, Tif, NbO, and BrF have uncertainties ranging from $\pm 1 \ \text{kcal/mol} \ \text{up to} \pm 8 \  \text{kcal/mol}$ [[7]](#7). Unlike $R_e$ and $\omega_e$, it is most likely that uncertainties around $D_0^0$ experimental values drive from various systematic effects.


To avoid data entry errors, the data set has been reviewed several times during the project by the authors using standard techniques like plotting the data, looking at the distributions of the data points and directly looking at single data points in the CSV files. The data is clean to the best of the author's knowledge.

Other issues regarding the experimental values of the dissociation energies of diatomic molecules are discussed in detail in the [manuscript](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/a68fdfea6013637c6956adae616f566831263065/Manuscript.pdf). 

## Data description
[g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv): contains data used for training and validation. References are provided for each molecule experimental data \
[g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv): contains data used for training and validation in addition to data used for testing. References are provided for each molecule experimental data \
[list of molecules used in Xiangue and Jesus paper.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/ece9ce778381e0e7a83e75dc29c02950d5a4bd62/data/list%20of%20molecules%20used%20in%20Xiangue%20and%20Jesus%20paper.csv): A list of molecules used in Liu et al., 2021 \
[peridic.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/ece9ce778381e0e7a83e75dc29c02950d5a4bd62/data/peridic.csv): contains information from the periodic table for each element

[g.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g.csv) and [g-test-2.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/b7fe30b53feceacd9d0e9ae47eeb9ef755adcce5/data/g-test-2.csv): A1 and A2 columns contain the mass number of the two atoms making up a molecule. 


[all_new_data.csv](https://github.com/Mahmoud-Ibrahim-Mamrstein/Spectroscopic-constants-from-atomic-properties/blob/8fb6627a3cf221a32150bc9b43dcf659b3174bb7/data/all_new_data.csv): Conssists of all the newly gatherd data with refrences and authors' notes
Electronic state: electronic state and/or symmetry symbol

$T_e$  (cm $^{-1}$):  minimum electronic energy 

$\omega_e$ (cm $^{-1}$): vibrational constant – first term

$\omega_ex_e$ (cm $^{-1}$): vibrational constant – second term 

$B_e$ (cm $^{-1}$): rotational constant in equilibrium position

$\alpha_e$ (cm $^{-1}$): rotational constant – first term

$D_e$ ($10^{-7}$ cm $^{-1}$): 	centrifugal distortion constant

$R_e$ (\AA): internuclear distance

$D_O^O$ (eV): Dissociation energy

$IP$ (eV): Ionozation potential

lan_act: Indicates the use of group 3 as the group for both lanthanides and actinides

iso: indicates of the use of 0 as the group number of Deuterium and -1 as the group number of Tritium.
 
 ## References
<a id="1">[1]</a> 
Klaus-Peter Huber. Molecular spectra and molecular structure: IV. Constants of diatomic
molecules. Springer Science & Business Media, 2013.

<a id="2">[2]</a> 
Stefanov, B., 1985. Comment: On the equilibrium of Hg2 molecule. The Journal of chemical physics, 83(5), pp.2621-2621.

<a id="3">[3]</a> 
Hilpert, K. and Jones, R.O., 1985. Reply to ‘‘Comment: On the equilibrium of Hg2 molecule’’. The Journal of Chemical Physics, 83(5), pp.2622-2622.


<a id="4">[4]</a>  J. Dufayard, B. Majournat and O. Nedelec, Chem. Phys. 128, 537 (1988).

<a id="5">[5]</a> W. C. Stwalley, J. Chem. Phys. 63, 3062 (1975)

<a id="6">[6]</a> Fu, J., Wan, Z., Yang, Z., Liu, L., Fan, Q., Xie, F., Zhang, Y. and Ma, J., 2022. Combining ab initio and machine learning method to improve the prediction of diatomic vibrational energies. International Journal of Quantum Chemistry, 122(18), p.e26953.

<a id="7">[7]</a> Luo, Y.R., 2007. Comprehensive handbook of chemical bond energies. CRC press.

<a id="8">[8]</a>Badger, R.M., 1934. A relation between internuclear distances and bond force constants. The Journal of Chemical Physics, 2(3), pp.128-131.

<a id="9">[9] Jhung, K.S., Kim, I.H., Oh, K.H., Hahn, K.B. and Jhung, K.H.C., 1990. Universal nature of diatomic potentials. Physical Review A, 42(11), p.6497.

<a id="10">[10] The determination of internuclear distances and of dissociation energies from force constants

<a id="11">[11] Kȩpa, R., Ostrowska-Kopeć, M., Piotrowska, I., Zachwieja, M., Hakalla, R., Szajna, W. and Kolek, P., 2014. Ångström (B1Σ+→ A1Π) 0–1 and 1–1 bands in isotopic CO molecules: further investigations. Journal of Physics B: Atomic, Molecular and Optical Physics, 47(4), p.045101.

<a id="12">[12] Vol'Kenshtein, M.V., 1955. Structure and physical properties of molecules. Izd. Inostr. Lit., Moscow.
