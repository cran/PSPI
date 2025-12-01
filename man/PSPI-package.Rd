\name{PSPI-package}
\alias{PSPI-package}
\alias{PSPI}
\docType{package}
\title{
  Propensity Score Predictive Inference for Generalizability
}
\description{
  Provides a suite of Propensity Score Predictive Inference (PSPI) methods to generalize treatment effects in trials to target populations. The package includes an existing model Bayesian Causal Forest (BCF) and four PSPI models (BCF-PS, FullBART, SplineBART, DSplineBART). These methods leverage Bayesian Additive Regression Trees (BART) to adjust for high-dimensional covariates and nonlinear associations, while SplineBART and DSplineBART further use propensity score based splines to address covariate shift between trial data and target population. 
}
\details{
PSPI provides Bayesian methods for generalizing treatment effects from clinical trials to target populations. It implements five models-\code{BCF}, \code{BCF_P}, \code{FullBART}, \code{SplineBART}, and \code{DSplineBART}-built on Bayesian
Additive Regression Trees (BART). Spline-based variants (\code{SplineBART} and \code{DSplineBART}) use propensity score transformations and spline terms to handle covariate shift between datasets.
Core computations rely on efficient MCMC routines implemented in C++.

  This package modifies and extends C++ code originally derived from the
  BART3 package, developed by Rodney Sparapani,
  which is licensed under the GNU General Public License version 2 (GPL-2).

  The modified code is redistributed in accordance with the GPL-2 license.
  For more details on the modifications, see the package's documentation.
}
\references{
  BART3 package: https://github.com/rsparapa/bnptools/tree/master, originally developed by Rodney Sparapani. 
}
\examples{
   sim <- sim_data(scenario = "linear", n_trial = 60)
 
 fit <- PSPI_generalizability(
   X = as.matrix(sim$trials[, paste0("X", 1:10)]),
   Y = sim$trials$Y,
   A = sim$trials$A,
   pi = sim$population$ps[sim$population$selected],
   X_pop = as.matrix(sim$population[, paste0("X", 1:10)]),
   pi_pop = sim$population$ps,
   model = "SplineBART",
   transformation = "InvGumbel",
   verbose = FALSE,
   nburn = 1, npost = 1
)
   str(fit)

}
