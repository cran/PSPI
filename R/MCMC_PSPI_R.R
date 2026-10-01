#' Propensity Scores Predictive Inference for Generalizability and Transportability
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
#'   \item \code{"DSplineBART"} (alias \code{"MSplineBART"}) – Adds another natural cubic spline for the prognostic score.
#' }

#'
#' **Multi-arm treatment support**
#'
#' All models support \eqn{J \ge 2} treatment arms. The model structure is:
#' \deqn{Y_i = h(X_i, \hat\pi_{i,0}) + \sum_{j=1}^{J-1} I(A_i = j) \cdot s_j(X_i, \hat\pi_{i,j}) + \epsilon_i}
#' where arm 0 is the reference (control) group, \eqn{h(\cdot)} is the prognostic
#' function, and each \eqn{s_j(\cdot)} captures the heterogeneous treatment effect
#' for arm \eqn{j} relative to the reference.
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
#' @param Y Numeric vector or logical vector of observed outcomes in the trial.
#'   Numeric vector is for continuous variable, while logical vector is for binary outcome.
#' @param A Integer vector of treatment assignments. Values should be
#'   \code{0, 1, ..., J-1} where \code{0} is the reference (control) arm and
#'   \code{J} is the total number of treatment arms.
#' @param pi Numeric matrix (\eqn{n \times J}) of propensity scores for trial
#'   participants. Column \eqn{j} contains the propensity score associated with
#'   arm \eqn{j}. In a standard analysis with a single participation propensity
#'   score \eqn{\hat\pi = \Pr(S = 1 \mid X)}, the same score can be supplied in
#'   all columns. The number of columns must equal \code{length(unique(A))}.
#' @param X_pop Matrix of covariates for the target population data.
#' @param A_pop Integer vector of treatment assignments for population data (default = NULL).
#' @param pi_pop Numeric matrix (\eqn{N \times J}) of propensity scores for the
#'   target population. Must have the same number of columns as \code{pi}.
#' @param model Character string specifying which PSPI model to use (see Details).
#' @param transformation Character string indicating the transformation applied to the
#'   propensity scores. Options are \code{"Identity"}, \code{"Logit"}, \code{"Cloglog"},
#'   or \code{"InvGumbel"} (default).
#' @param restrict_covariates Restrict population data within support of trial samples.
#'   This parameter specifies the names of covariates to restrict (default = NULL).
#' @param nburn Number of burn-in iterations (default = 4000).
#' @param npost Number of posterior iterations saved after burn-in (default = 4000).
#' @param n_knots Integer vector specifying the number of spline knots per arm.
#'   Length \eqn{J - 1} for \code{"SplineBART"} or \eqn{J} for \code{"DSplineBART"}.
#'   If \code{NULL}, defaults are chosen automatically.
#' @param order Order of spline basis functions (default = 3).
#' @param ntrees_s Number of trees used for the BART component (default = 200).
#' @param sparse Whether to perform variable selection based on a sparse Dirichlet prior rather than simply uniform; see Linero 2016.
#' @param augment  Whether data augmentation is to be performed in sparse variable selection; see Linero 2016.
#' @param verbose Logical; if TRUE, prints progress messages.
#' @param seed Optional random seed for reproducibility.
#' @param pate_bootstrap Logical (default \code{TRUE}). If \code{TRUE},
#'   after MCMC the function attaches a \code{[npost, J]} matrix
#'   \code{post_outcome_PATE} containing Bayesian-bootstrap (Dirichlet
#'   re-weighting) draws of the target-population mean \eqn{E[Y(a)]} for
#'   each treatment arm. This propagates the uncertainty of the empirical
#'   covariate distribution \eqn{\widehat{F}_X} into the posterior,
#'   giving population (PATE) intervals in addition to the standard
#'   mixed-ATE (MATE) intervals obtained by averaging
#'   \code{post_outcome[, , j]} uniformly across the target sample
#'   (Li, Ding & Mealli, 2022). Set \code{FALSE} to skip and save memory.
#'
#' @return
#' A list containing posterior samples and model summaries:
#' \describe{
#'   \item{\code{post_outcome}}{A three-dimensional array of dimension
#'     \code{[npost, N, J]}. Entry \code{[m, i, j]} is the \eqn{m}-th posterior
#'     draw of the potential outcome for population unit \eqn{i} under treatment
#'     arm \eqn{j}. Population-level potential outcome means are obtained by
#'     averaging over units: \code{rowMeans(post_outcome[,,j])}.}
#'   \item{\code{post_interaction}}{A three-dimensional array of dimension
#'     \code{[npost, N, J-1]}. Entry \code{[m, i, j]} is the \eqn{m}-th posterior
#'     draw of the treatment effect for unit \eqn{i} comparing arm \eqn{j} to the
#'     reference arm (arm 0). The population average treatment effect for arm
#'     \eqn{j} is \code{rowMeans(post_interaction[,,j])}.}
#'   \item{\code{post_outcome_train}}{Matrix \code{[npost, n]} of fitted values
#'     on the trial sample.}
#'   \item{\code{post_sigma}}{Vector of length \code{npost} containing posterior
#'     draws of the residual standard deviation.}
#'   \item{\code{post_beta}}{(SplineBART and DSplineBART only) List of matrices
#'     containing posterior draws of the natural cubic spline coefficients, one
#'     matrix per spline component.}
#'   \item{\code{post_gamma2}}{(SplineBART and DSplineBART only) Matrix of
#'     posterior draws of the spline shrinkage parameters.}
#' }
#'
#' @examples
#' # Example with simulated data
#' sim <- sim_trans(scenario = "linear", n_trial = 60)
#'
#' fit <- PSPI_generalizability(
#'   X = as.matrix(sim$trials[, paste0("X", 1:10)]),
#'   Y = sim$trials$Y,
#'   A = sim$trials$A,
#'   pi = cbind(sim$trials$ps_trial, sim$trials$ps_trial),
#'   X_pop = as.matrix(sim$population[, paste0("X", 1:10)]),
#'   pi_pop = cbind(sim$population$ps_target, sim$population$ps_target),
#'   model = "SplineBART",
#'   transformation = "InvGumbel",
#'   #restrict_covariates = c("X1", "X2"),
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
#' @importFrom arm logit
#' @export
PSPI_generalizability = function(X, Y, A, pi, X_pop, A_pop = NULL, pi_pop, model, transformation = "InvGumbel", restrict_covariates = NULL, nburn = 4000, npost = 4000, n_knots = NULL, order = 3, ntrees_s = 200, sparse = FALSE, augment = FALSE, verbose = FALSE, seed = NULL, pate_bootstrap = TRUE){
  
  
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
    model == "MSPLINEBART" | model == "PSPI-MSPLINEBART" | model == "PSPI_MSPLINEBART" ~ 6,
    TRUE ~ NA
  )
  if(is.na(model)){
    stop("Invalid model_name. Available options: BCF, BCF_P, FullBART, SplineBART, DSplineBART.")
  }
  
  if(!methods::is(transformation, "character")){
    stop("Invalid transformation. Please specify a character name for transformation. Available options: Identity, Logit, Cloglog, InvGumbel.")
  }
  
  message(paste0("Start to run the model: ", c("BCF", "BCF_P", "FullBART", "SplineBART", "DSplineBART")[model - 1]))
  if(ncol(pi) != ncol(pi_pop)){
    stop("Mismatch of number of columns of pi and pi_pop")
  }
  
  J = ncol(pi)
  if(J != length(unique(A))){
    stop("Mismatch of number of treatment options and columns of pi")
  }
  
  
  
  if(model == 5 | model == 6){
    if(is.null(n_knots)){
      n_knots = sapply(0:(J-1), function(j){
        if(j == 0) return(round(length(A)^(1/3)))
        else return(round(sum(A==j)^(1/3)))
      })
      
      n_knots = pmax(n_knots, 2)
      if(model == 5)
        n_knots = n_knots[-1]
    }
    message("Number of knots for splines: ", paste0(n_knots, collapse = ", "))
    if(length(order) == 1){
      if(model == 5)
        order = rep(order, J - 1)
      else
        order = rep(order, J)
    }
  }
  
  if(is.null(n_knots)){
    n_knots = 0
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
    pi = invgumbel(pi)
    pi_pop = invgumbel(pi_pop)
    message("Transformation on the propensity scores: InvGumbel")
    
  }
  if(transformation == "IDENTITY"){
    message("Transformation on the propensity scores: Identity")
  }
  
  if(is.logical(Y)){
    message("Outcome type: Binary")
  }else{
    message("Outcome type: Continuous")
  }
  
  
  if (!is.null(restrict_covariates)) {
    message("Restrict covariates: ", paste(restrict_covariates, collapse = ", "))
    
    stopifnot(
      !is.null(colnames(X)),
      !is.null(colnames(X_pop)),
      all(restrict_covariates %in% colnames(X)),
      all(restrict_covariates %in% colnames(X_pop))
    )
    
    
    keep <- rep(TRUE, nrow(X_pop))
    
    for (v in restrict_covariates) {
      xp <- X_pop[,v]
      x  <- X[,v]
      
      # treat non-continuous as categorical: factor/character/logical
      if (is.factor(x) || is.character(x) || is.logical(x)) {
        vals <- unique(x[!is.na(x)])
        keep <- keep & (is.na(xp) | xp %in% vals)
        
      } else {
        # numeric/integer -> continuous: restrict to [min, max] in X
        lo <- suppressWarnings(min(x, na.rm = TRUE))
        hi <- suppressWarnings(max(x, na.rm = TRUE))
        
        if (is.finite(lo) && is.finite(hi)) {
          keep <- keep & (is.na(xp) | (xp >= lo & xp <= hi))
        }
      }
    }
    
    X_pop <- X_pop[keep, , drop = FALSE]
    pi_pop <- pi_pop[keep, , drop = FALSE]
    
  }
  
  
  
  if(!is.null(seed))
    set.seed(seed)
  if(length(unique(Y)) == 2)
    Y = as.logical(Y)
  
  mcmc_results = MCMC_PSPI_generalizability(as.matrix(X), Y, is.logical(Y), A, pi, as.matrix(X_pop), pi_pop, model, nburn, npost, n_knots, order, ntrees_s, verbose, sparse, augment)
  
  if(model == 5 | model == 6){
    mcmc_results$post_beta = lapply(1:length(mcmc_results$post_beta[[1]]), function(j){
      t(sapply(1:npost, function(np){
        mcmc_results$post_beta[[np]][[j]]
      }))
    })
    
    mcmc_results$post_gamma2 = t(sapply(1:npost, function(np){
      mcmc_results$post_gamma2[[np]]
    }))
    if(dim(mcmc_results$post_gamma2)[1] == 1)
      mcmc_results$post_gamma2 = t(mcmc_results$post_gamma2)
    
    # Convert spline_train and spline_test lists of matrices to 3D arrays
    # Dimensions: [npost, n_obs, n_splines]
    if(!is.null(mcmc_results$post_spline_train) && length(mcmc_results$post_spline_train) > 0){
      mcmc_results$post_spline_train = aperm(simplify2array(mcmc_results$post_spline_train), c(3, 1, 2))
    }
    if(!is.null(mcmc_results$post_spline_test) && length(mcmc_results$post_spline_test) > 0){
      mcmc_results$post_spline_test = aperm(simplify2array(mcmc_results$post_spline_test), c(3, 1, 2))
    }
  }
  
  # mcmc_results$in_trial = in_trial(X_pop, X)
  # if("spline_inter" %in% names(mcmc_results)){
  #   mcmc_results$spline_inter$pi_transformed = pi[,2]
  #   mcmc_results$spline_inter$pi_pop_transformed = pi_pop[,2]
  # }
  # if("spline_main" %in% names(mcmc_results)){
  #   mcmc_results$spline_main$pi_transformed = pi[,1]
  #   mcmc_results$spline_main$pi_pop_transformed = pi_pop[,1]
  # }
  # 
  # if(!is.null(A_pop)){
  #   mcmc_results$post_outcome1 = mcmc_results$post_outcome1[, A_pop == 1]
  #   mcmc_results$post_outcome0 = mcmc_results$post_outcome0[, A_pop == 0]
  #   mcmc_results$post_te = rowMeans(mcmc_results$post_outcome1) - rowMeans(mcmc_results$post_outcome0)
  # }
  
  if(npost > 0){
    mcmc_results$post_outcome = aperm(simplify2array(mcmc_results$post_outcome), c(3, 1, 2))
    mcmc_results$post_interaction = aperm(simplify2array(mcmc_results$post_interaction), c(3, 1, 2))
  }

  # ---- Bayesian-bootstrap PATE post-processing ---------------------------
  # MATE: per posterior draw, average of CATE over the OBSERVED target sample
  #       (i.e. uniform weights 1/N).  Already obtainable via
  #       rowMeans(post_outcome[,,j]).
  # PATE: re-weight units by Dirichlet(1,...,1) per draw, yielding a
  #       Bayesian-bootstrap estimate of the target-population mean
  #       (Li, Ding & Mealli, 2022, Section 3.1).
  if(isTRUE(pate_bootstrap) && npost > 0){
    n_tgt <- dim(mcmc_results$post_outcome)[2]
    J_arms <- dim(mcmc_results$post_outcome)[3]
    W <- matrix(stats::rgamma(npost * n_tgt, shape = 1, rate = 1),
                nrow = npost, ncol = n_tgt)
    W <- W / rowSums(W)
    mcmc_results$post_outcome_PATE <- matrix(0, nrow = npost, ncol = J_arms)
    for(j in seq_len(J_arms)){
      mcmc_results$post_outcome_PATE[, j] <- rowSums(W * mcmc_results$post_outcome[, , j])
    }
  }

  class(mcmc_results) <- c("PSPI_generalizability", "PSPI_fitresult", "list")
  return(mcmc_results)
}


#' #' @export
#' PSPI_generalizability_ps = function(X, Y, A, X_pop, A_pop = NULL, S, model, restrict_covariates = NULL, nburn = 4000, npost = 4000, n_knots_main = NULL, n_knots_inter = NULL, order_main = 3, order_inter = 3, ntrees_s = 200, sparse = FALSE, augment = FALSE, verbose = FALSE, seed = NULL){
#'   
#'   if(!methods::is(model, "character")){
#'     stop("Invalid model_name. Please specify a character name for model. Available options: BCF, BCF_P, FullBART, SplineBART, MSplineBART.")
#'   }
#'   
#'   
#'   model = stringr::str_to_upper(model)
#'   model = dplyr::case_when(
#'     model == "BCF" ~ 2,
#'     model == "BCF-P" | model == "BCF_P" | model == "BCF-PS" | model == "BCF_PS" ~ 3,
#'     model == "PSPI_BCF-P" | model == "PSPI_BCF_P" | model == "PSPI_BCF-PS" | model == "PSPI_BCF_PS" ~ 3,
#'     model == "FULLBART" | model == "PSPI-FULLBART" | model == "PSPI_FULLBART" ~ 4,
#'     model == "SPLINEBART" | model == "PSPI-SPLINEBART" | model == "PSPI_SPLINEBART" ~ 5,
#'     model == "MSplineBART" | model == "PSPI-MSplineBART" | model == "PSPI_MSplineBART" ~ 6,
#'     TRUE ~ NA
#'   )
#'   if(is.na(model)){
#'     stop("Invalid model_name. Available options: BCF, BCF_P, FullBART, SplineBART, MSplineBART.")
#'   }
#'   
#'   message(paste0("Start to run the model: ", c("BCF", "BCF_P", "FullBART", "SplineBART", "MSplineBART")[model - 1]))
#'   
#'   
#'   
#'   if(model == 5 | model == 6){
#'     if(is.null(n_knots_main)){
#'       n_knots_main = round(length(A)^(1/3))
#'     }
#'     if(is.null(n_knots_inter)){
#'       #n_knots_inter = round(length(A)^(1/3))
#'       n_knots_inter = round(sum(A==1)^(1/3))
#'     }
#'     n_knots_main = max(n_knots_main, 2)
#'     n_knots_inter = max(n_knots_inter, 2)
#'     if(model == 6)
#'       message("Number of knots for splines in the main-effect term: ", n_knots_main)
#'     message("Number of knots for splines in the interaction term: ", n_knots_inter)
#'   }
#'   
#'   if(is.null(n_knots_main)){
#'     n_knots_main = 0
#'   }
#'   if(is.null(n_knots_inter)){
#'     n_knots_inter = 0
#'   }
#'   
#'   
#'   if(is.logical(Y)){
#'     message("Outcome type: Binary")
#'   }else{
#'     message("Outcome type: Continuous")
#'   }
#'   
#'   
#'   if (!is.null(restrict_covariates)) {
#'     message("Restrict covariates: ", paste(restrict_covariates, collapse = ", "))
#'     
#'     stopifnot(
#'       !is.null(colnames(X)),
#'       !is.null(colnames(X_pop)),
#'       all(restrict_covariates %in% colnames(X)),
#'       all(restrict_covariates %in% colnames(X_pop))
#'     )
#'     
#'     
#'     keep <- rep(TRUE, nrow(X_pop))
#'     
#'     for (v in restrict_covariates) {
#'       xp <- X_pop[,v]
#'       x  <- X[,v]
#'       
#'       # treat non-continuous as categorical: factor/character/logical
#'       if (is.factor(x) || is.character(x) || is.logical(x)) {
#'         vals <- unique(x[!is.na(x)])
#'         keep <- keep & (is.na(xp) | xp %in% vals)
#'         
#'       } else {
#'         # numeric/integer -> continuous: restrict to [min, max] in X
#'         lo <- suppressWarnings(min(x, na.rm = TRUE))
#'         hi <- suppressWarnings(max(x, na.rm = TRUE))
#'         
#'         if (is.finite(lo) && is.finite(hi)) {
#'           keep <- keep & (is.na(xp) | (xp >= lo & xp <= hi))
#'         }
#'       }
#'     }
#'     
#'     X_pop <- X_pop[keep, , drop = FALSE]
#'     
#'   }
#'   
#'   
#'   
#'   if(!is.null(seed))
#'     set.seed(seed)
#'   if(length(unique(Y)) == 2)
#'     Y = as.logical(Y)
#'   
#'   mcmc_results = MCMC_PSPI_generalizability2(as.matrix(X), Y, is.logical(Y), A, as.matrix(X_pop), S, model, nburn, npost, n_knots_main, n_knots_inter, order_main, order_inter, ntrees_s, verbose, sparse, augment)
#'   
#'   mcmc_results$in_trial = in_trial(X_pop, X)
#'   
#'   
#'   if(!is.null(A_pop)){
#'     mcmc_results$post_outcome1 = mcmc_results$post_outcome1[, A_pop == 1]
#'     mcmc_results$post_outcome0 = mcmc_results$post_outcome0[, A_pop == 0]
#'     mcmc_results$post_te = rowMeans(mcmc_results$post_outcome1) - rowMeans(mcmc_results$post_outcome0)
#'   }
#'   
#'   return(mcmc_results)
#' }