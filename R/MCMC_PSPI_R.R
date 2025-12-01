#' Propensity Scores Predictive Inference for Generalizability
#'
#' @description
#' This is the main function of the **PSPI** package. It runs Bayesian models that
#' generalize findings from a clinical trial to a target population, estimating
#' the average treatment effects and potential outcomes. Propensity scores of 
#' trial participation play the central role for generalizability analysis. 
#' When covariate shift is an issue, we recommend PSPI-SplineBART and PSPI-DSplineBART,
#' which leveraging Bayesian Additive Regression Trees (BART) to model high-dimensional covariates,
#' and propensity scores based splines to extrapolate smoothly.
#' 
#' 
#' Users provide trial data (covariates, outcomes, treatment, and propensity scores)
#' along with population-level covariates and propensity scores. Propensity scores
#' can be the true values or estimated from some models. The function then
#' performs Monte Carlo  Markov chain (MCMC) for the posterior inference.
#'
#'
#' @details
#' **Model choices**
#' 
#' The `model` argument selects the type of PSPI model to be fitted:
#' 
#' \itemize{
#'   \item \code{"BCF"} – Bayesian Causal Forests (Hahn et al., 2020).
#'   \item \code{"BCF_P"} – BCF with the propensity score as an additional predictor.
#'   \item \code{"FullBART"} – Uses three BARTs to estimate treatment effects.
#'   \item \code{"SplineBART"} – Incorporates a natural cubic spline for heterogeneous treatment effects.
#'   \item \code{"DSplineBART"} – Adds another natural cubic spline for the prognostic score.
#' }

#'
#' **Propensity score transformations**
#'
#' Since splines are sensitive to scales of predictor, robust transformation is needed.
#' The propensity scores (\code{pi} for trial, \code{pi_pop} for population) can be
#' optionally transformed before modeling using one of the following:
#' 
#' \itemize{
#'   \item \code{"Identity"} – uses the raw propensity scores directly (no transformation).
#'   \item \code{"Logit"} – applies the logit transform: \eqn{g(p) = \log(p / (1 - p))}.
#'   \item \code{"Cloglog"} – complementary log–log transform: \eqn{g(p) = \log(-\log(1 - p))}.
#'   \item \code{"InvGumbel"} – inverse Gumbel transform: \eqn{g(p) = -\log(-\log(p))}. Default choice.
#' }
#'
#' Users can experiment with different transformations to assess model sensitivity.
#'
#' **Spline settings**
#'
#' Spline-based models (\code{"SplineBART"} and \code{"DSplineBART"}) allow flexible
#' extrapolation to address covariate shift. The number and order of spline basis functions can be
#' customized through the following parameters:
#' \itemize{
#'   \item \code{n_knots_inter}, \code{order_inter}: number and order of spline knots for
#'         treatment-interaction effects. Available for both \code{SplineBART} and
#'         \code{DSplineBART}.
#'   \item \code{n_knots_main}, \code{order_main}: number and order of spline knots for
#'         main effects. Available only for \code{DSplineBART}.
#' }
#' 
#' If any of these are left as \code{NULL}, default values are chosen automatically based
#' on the cube root of the sample size (ensuring a reasonable smoothness level).
#'
#'
#'
#' @param X Matrix of covariates for the trial data.
#' @param Y Numeric vector of observed outcomes in the trial.
#' @param A Binary vector of treatment assignments (0 = control, 1 = intervention).
#' @param pi Numeric vector of trial propensity scores (probability of trial participation).
#' @param X_pop Matrix of covariates for the target population data.
#' @param pi_pop Numeric vector of the target population propensity scores.
#' @param model Character string specifying which PSPI model to use (see Details).
#' @param transformation Character string indicating the transformation applied to the
#'   propensity scores. Options are \code{"Identity"}, \code{"Logit"}, \code{"Cloglog"},
#'   or \code{"InvGumbel"} (default).
#' @param nburn Number of burn-in iterations (default = 4000).
#' @param npost Number of posterior iterations saved after burn-in (default = 4000).
#' @param n_knots_main,n_knots_inter Number of spline knots for main and interaction effects.
#'   If \code{NULL}, defaults are chosen automatically. 
#'   \code{n_knots_inter} is available for \code{SplineBART} and \code{DSplineBART};
#'   \code{n_knots_main} is available only for \code{DSplineBART}.
#' @param order_main,order_inter Order of spline basis functions (default = 3).
#'   \code{order_inter} applies to both \code{SplineBART} and \code{DSplineBART};
#'   \code{order_main} applies only to \code{DSplineBART}.
#' @param ntrees_s Number of trees used for the BART component (default = 200).
#' @param verbose Logical; if TRUE, prints progress messages.
#' @param seed Optional random seed for reproducibility.
#'
#' @return
#' A list containing posterior samples and model summaries produced by the C++
#' sampler. Typical elements include:
#' \describe{
#' \item{post_outcome1}{Each row is a posterior draw for individual potential outcome under treatment}
#' \item{post_outcome0}{Each row is a posterior draw for individual potential outcome under control}
#' \item{post_te}{Each row is a posterior draw for individual treatment effects}
#' }
#'
#' @examples
#' # Example with simulated data
#' sim <- sim_data(scenario = "linear", n_trial = 60)
#' 
#' fit <- PSPI_generalizability(
#'   X = as.matrix(sim$trials[, paste0("X", 1:10)]),
#'   Y = sim$trials$Y,
#'   A = sim$trials$A,
#'   pi = sim$population$ps[sim$population$selected],
#'   X_pop = as.matrix(sim$population[, paste0("X", 1:10)]),
#'   pi_pop = sim$population$ps,
#'   model = "SplineBART",
#'   transformation = "InvGumbel",
#'   verbose = FALSE,
#'   nburn = 1, npost = 1
#' )
#' 
#' str(fit)
#'
#'
#' @note
#' This function utilizes modified C++ code originally derived from the
#' BART3 package (Bayesian Additive Regression Trees). The original package
#' was developed by Rodney Sparapani and is licensed
#' under GPL-2. Modifications were made by Jungang Zou, 2024.
#' For more information about the original BART3 package, see:
#' https://github.com/rsparapa/bnptools/tree/master/BART3
#' @useDynLib PSPI, .registration = TRUE
#' @importFrom dplyr case_when
#' @importFrom stringr str_to_upper
#' @importFrom Rcpp sourceCpp
#' @importFrom stats rnorm runif
#' @importFrom methods is
#' @export
PSPI_generalizability = function(X, Y, A, pi, X_pop, pi_pop, model, transformation = "InvGumbel", nburn = 4000, npost = 4000, n_knots_main = NULL, n_knots_inter = NULL, order_main = 3, order_inter = 3, ntrees_s = 200, verbose = FALSE, seed = NULL){
  
  if(!methods::is(model, "character")){
    stop("Invalid model_name. Please specify a character name for model. Available options: BCF, BCF_P, FullBART, SplineBART, DSplineBART.")
  }
  
  
  model = stringr::str_to_upper(model)
  model = dplyr::case_when(
    model == "BCF" ~ 2,
    model == "BCF-P" | model == "BCF_P" | model == "BCF-PS" | model == "BCF_PS" ~ 3,
    model == "PSPI_BCF-P" | model == "PSPI_BCF_P" | model == "PSPI_BCF-PS" | model == "PSPI_BCF_PS" ~ 3,
    model == "FULLBART" | model == "PSPI-FULLBART" | model == "PSPI_FULLBART" ~ 4,
    model == "SPLINEBART" | model == "PSPI-SPLINEBART" | model == "PSPI_SPLINEBART" ~ 5,
    model == "DSPLINEBART" | model == "PSPI-DSPLINEBART" | model == "PSPI_DSPLINEBART" ~ 6,
    TRUE ~ NA
  )
  if(is.na(model)){
    stop("Invalid model_name. Available options: BCF, BCF_P, FullBART, SplineBART, DSplineBART.")
  }
  
  if(!methods::is(transformation, "character")){
    stop("Invalid transformation. Please specify a character name for transformation. Available options: Identity, Invlogit, Cloglog, InvGumbel.")
  }
  
  message(paste0("Start to run the model: ", c("BCF", "BCF_P", "FullBART", "SplineBART", "DSplineBART")[model - 1]))
  

  
  if(model == 5 | model == 6){
    if(is.null(n_knots_main)){
      n_knots_main = round(sum(A==0)^(1/3))
    }
    if(is.null(n_knots_inter)){
      n_knots_inter = round(sum(A==1)^(1/3))
    }
    n_knots_main = max(n_knots_main, 2)
    n_knots_inter = max(n_knots_inter, 2)
  }
  
  if(is.null(n_knots_main)){
    n_knots_main = 0
  }
  if(is.null(n_knots_inter)){
    n_knots_inter = 0
  }
  
  transformation = stringr::str_to_upper(transformation)
  if(transformation == "LOGIT"){
    pi = arm::logit(pi)
    pi_pop = arm::logit(pi_pop)
    message("Transformation on the propensity scores: Logit")
  }
  if(transformation == "CLOGLOG"){
    pi = log(-log(1- pi))
    pi_pop = log(-log(1- pi_pop))
    message("Transformation on the propensity scores: Cloglog")
  }
  if(transformation == "INVGUMBEL"){
    pi = -log(-log(pi))
    pi_pop =-log(-log(pi_pop))
    message("Transformation on the propensity scores: InvGumbel")
    
  }
  if(transformation == "IDENTITY"){
    message("Transformation on the propensity scores: Identity")
  }
  
  if(!is.null(seed))
    set.seed(seed)
  mcmc_results = MCMC_PSPI_generalizability(X, Y, A, pi, X_pop, pi_pop, model, nburn, npost, n_knots_main, n_knots_inter, order_main, order_inter, ntrees_s, verbose)
  return(mcmc_results)
}

