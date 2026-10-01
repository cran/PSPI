#' Predict from a Fitted PSPI Model on Population Data
#'
#' @description
#' Generates posterior predictions of potential outcomes and treatment effects
#' for a target population, using a fitted PSPI model object produced by
#' \code{\link{PSPI_fit}}. This function only requires population-level
#' covariates and propensity scores — no trial data is needed.
#'
#' @param fit An object of class \code{"PSPI_fit"} returned by
#'   \code{\link{PSPI_fit}}, or loaded via \code{readRDS()}.
#' @param X_pop Numeric matrix of covariates for the target population (N x p).
#'   Must have the same columns (same variables, same order) as the trial X
#'   used in \code{PSPI_fit}.
#' @param pi_pop Numeric matrix (N x J) of propensity scores for the target
#'   population. Must have the same number of columns as the \code{pi} used
#'   in \code{PSPI_fit}.
#' @param restrict_covariates Character vector of covariate names to restrict
#'   population data to the support of trial samples (default = NULL).
#'   Requires column names on \code{X_pop}.
#' @param X_trial Optional trial covariate matrix, needed only when
#'   \code{restrict_covariates} is not NULL, to determine the support.
#' @param verbose Logical; print progress (default = FALSE).
#' @param pate_bootstrap Logical (default \code{TRUE}). If \code{TRUE},
#'   attaches a \code{[npost, J]} matrix \code{post_outcome_PATE} containing
#'   Bayesian-bootstrap (Dirichlet re-weighting) draws of the
#'   target-population mean \eqn{E[Y(a)]} for each arm, propagating the
#'   uncertainty of the empirical covariate distribution
#'   \eqn{\widehat{F}_X} into the posterior (population-ATE estimand of
#'   Li, Ding & Mealli, 2022). Set \code{FALSE} to skip.
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{post_outcome}}{Array \code{[npost, N, J]} of posterior draws
#'     of potential outcomes.}
#'   \item{\code{post_interaction}}{Array \code{[npost, N, J-1]} of posterior
#'     draws of treatment effects relative to arm 0.}
#' }
#'
#' @examples
#' \dontrun{
#' fit <- readRDS("pspi_fit.rds")
#' result <- PSPI_predict(fit, X_pop, pi_pop = cbind(ps_pop, ps_pop))
#'
#' # Population ATE for arm 1 vs arm 0
#' ate <- rowMeans(result$post_interaction[,,1])
#' quantile(ate, c(0.025, 0.5, 0.975))
#' }
#'
#' @seealso \code{\link{PSPI_fit}}, \code{\link{PSPI_generalizability}}
#' @export
PSPI_predict = function(fit, X_pop, pi_pop, restrict_covariates = NULL, X_trial = NULL, verbose = FALSE, pate_bootstrap = TRUE){

  if(!inherits(fit, "PSPI_fit")){
    stop("fit must be an object of class 'PSPI_fit' from PSPI_fit()")
  }

  if(ncol(pi_pop) != fit$J){
    stop(paste0("pi_pop must have ", fit$J, " columns (J = ", fit$J, ")"))
  }

  # Apply the same transformation used during fitting
  transformation = stringr::str_to_upper(fit$transformation)
  if(transformation == "LOGIT"){
    pi_pop = arm::logit(pi_pop)
  }
  if(transformation == "CLOGLOG"){
    pi_pop = log(-log(1 - pi_pop))
  }
  if(transformation == "INVGUMBEL"){
    pi_pop = invgumbel(pi_pop)
  }

  # Optional covariate restriction
  if(!is.null(restrict_covariates)){
    if(is.null(X_trial)){
      stop("X_trial must be provided when restrict_covariates is not NULL")
    }
    stopifnot(
      !is.null(colnames(X_pop)),
      !is.null(colnames(X_trial)),
      all(restrict_covariates %in% colnames(X_trial)),
      all(restrict_covariates %in% colnames(X_pop))
    )

    keep <- rep(TRUE, nrow(X_pop))
    for(v in restrict_covariates){
      xp <- X_pop[, v]
      x  <- X_trial[, v]
      if(is.factor(x) || is.character(x) || is.logical(x)){
        vals <- unique(x[!is.na(x)])
        keep <- keep & (is.na(xp) | xp %in% vals)
      } else {
        lo <- suppressWarnings(min(x, na.rm = TRUE))
        hi <- suppressWarnings(max(x, na.rm = TRUE))
        if(is.finite(lo) && is.finite(hi)){
          keep <- keep & (is.na(xp) | (xp >= lo & xp <= hi))
        }
      }
    }
    X_pop <- X_pop[keep, , drop = FALSE]
    pi_pop <- pi_pop[keep, , drop = FALSE]
  }

  mcmc_results = MCMC_PSPI_predict(fit, as.matrix(X_pop), pi_pop, verbose)

  npost = fit$npost
  if(npost > 0){
    mcmc_results$post_outcome = aperm(simplify2array(mcmc_results$post_outcome), c(3, 1, 2))
    mcmc_results$post_interaction = aperm(simplify2array(mcmc_results$post_interaction), c(3, 1, 2))
  }

  # ---- Bayesian-bootstrap PATE post-processing ---------------------------
  # See PSPI_generalizability for details; same logic.
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

  class(mcmc_results) <- c("PSPI_predict", "PSPI_fitresult", "list")
  return(mcmc_results)
}
