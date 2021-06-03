# chemometrics
chemometrics is a free and open source chemometric library for Python. Its main focus lies on the chemometric analysis of spectroscopic data (e.g. UV/Vis, NIR, Raman, NMR and MS). chemometrics builds on scikit-learn and extends its functionalities to support chemometric data analysis. The package provides methods for plotting, preprocessing and fitting data. In contrast to scikit-learn, chemometrics is mainly intended for interactive work.

chemometrics is work in progress. The current features are mainly revolving around the (supervised) analysis of spectroscopic data.

## Example applications
Plotting:
```python
import matplotlib.pyplot as plt
import chemometrics as cm
cm.plot_colored_series(X.T, reference=Y)
plt.xlabel('Wavenumber / nm')
plt.ylabel('Absorbance / mAU')
```

![spectra](https://github.com/maruedt/chemometrics/blob/master/examples/peaches.png "NIR spectra")

Preprocessing:
```python
X_deriv = cm.Whittaker(constraint_order=3, deriv=2).fit_transform(X)
cm.plot_colored_series(X_deriv.T, reference=Y)
plt.xlabel('Wavenumber / nm')
plt.ylabel('$d^2A/dl^2$ / $mAU/nm^2$')
```
![derived spectra](https://github.com/maruedt/chemometrics/blob/master/examples/peaches_deriv.png "Second derivative NIR spectra")


Regression:
```python
cm.fit_pls(X_deriv, Y)
```
![CV scores](https://github.com/maruedt/chemometrics/blob/master/examples/pls_cv.png "Cross-validation scores")
![PLS analytics](https://github.com/maruedt/chemometrics/blob/master/examples/pls_analysis.png "PLS analytical plots")

More examples with explanations and additional code are shown in the examples folder. For an initial overview of a typical regression workflow, a look at [this example](https://github.com/maruedt/chemometrics/blob/master/examples/basic_pls_example.ipynb) might be most interesting (the example Jupyter Notebooks are not always rendered correctly by Github. They may be displayed by copying the url to https://nbviewer.jupyter.org/).

## Requirements and installation
- Python >= 3.8
- NumPy >= 1.19.2
- SciPy >= 1.5.2
- scikit-learn >= 0.23.2
- matplotlib >= 3.3.2

Earlier versions of the required libraries may work but have not been tested.

chemometrics is distributed over PyPI. The simplest way to install chemometrics is by running

```
python -m pip install chemometrics
```
The source code is available from: https://github.com/maruedt/chemometrics.


## Code quality, testing and future development
chemometrics is written with a strong focus on code quality in mind. The first implementation of many similar functionalities was done in Matlab but chemometrics has since been rewritten from scratch to omit licensing issues and to provide an improved and coherent code structure. The code includes quite an extensive set of unit tests to ensure that it actually does what it is supposed to do. Next to ensuring that the code runs, the tests aim to mathematically ensure the correctness of the different procedures.

As mentioned above, chemometrics is work in progress. I will be working on integrating additional analytical methods and provide more of the common tools/plots used in chemometrics. This may include:
- Summary statistics/plots for PCA
- Further develop the already available tools for PLS models
- Potentially MCR

## Nomenclature
chemometrics generally complies with the glossary of scikit-learn: https://scikit-learn.org/stable/glossary.html

## Copyright
Copyright 2021 Matthias Rüdt
