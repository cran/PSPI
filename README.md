# PSPI
  <!-- badges: start -->
  [![R-CMD-check](https://github.com/zjg540066169/SBMtrees/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/zjg540066169/PSPI/actions/workflows/R-CMD-check.yaml)
  [![License: GPL-2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html)
  [![version](https://img.shields.io/badge/version-1.1-green.svg)](https://github.com/zjg540066169/PSPI)
  ![R](https://img.shields.io/badge/language-R-blue)
  ![C++](https://img.shields.io/badge/language-C%2B%2B-green)
  <!-- badges: end -->

The R package **PSPI** (Propensity Score Predictive Inference) provides a suite of Propensity Score Predictive Inference (PSPI) methods to generalize treatment effects in trials to target populations. The package includes an existing model Bayesian Causal Forest (BCF) and four PSPI models (BCF-PS, FullBART, SplineBART, DSplineBART). These methods leverage Bayesian Additive Regression Trees (BART) to adjust for high-dimensional covariates and nonlinear associations, while SplineBART and DSplineBART further use propensity score based splines to address covariate shift between trial data and target population. 

## Installation
This package is based on `Rcpp`, `RcppArmadillo`, `RcppDist`, and `pg`, please make sure these three packages can be installed.

This package can be installed from R CRAN:
```
install.packages("PSPI")
```
or Github:
```
require("devtools")
install_github("https://github.com/zjg540066169/PSPI")
library(PSPI)
```

## Attribution

This package includes code derived from the [BART3](https://github.com/rsparapa/bnptools/tree/master) package, originally developed by Rodney Sparapani. 

The original source code, licensed under the [GNU General Public License version 2 (GPL-2)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html), has been modified as follows:
- We include part of the C++ code in BART3, primarily about functions about `wbart` and `cpwart`. We also modify some files to make sure our package can be successfully compiled.
- Modifications were made by Jungang Zou, 2024.

### Licensing

- The original BART3 package is licensed under the GNU General Public License version 2 (GPL-2).
- This package, as a derived work, is also licensed under the GNU General Public License version 2 (GPL-2) to comply with the licensing terms.

### Here are some acronyms:
* Zou: Author` last name.
* PSPI: Propensity Score Predictive Inference
* BART: Bayesian Additive Regression Trees.
* BCF: Bayesian Causal Forest.
* BCF-PS: Name of a PSPI model.
* FullBART: Name of a PSPI model.
* SplineBART: Name of a PSPI model.
* DSplineBART: Name of a PSPI model.
* MCMC: Monte Carlo Markov chain.
* Cloglog: complementary log–log transform.
* InvGumbel: Inverse Gumbel function or Gumbel quantile function.

