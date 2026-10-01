#' Fit a PSPI Model (Without Population Data)
#'
#' @description
#' Fits a PSPI model using trial data only, without requiring population-level
#' covariates. The returned fit object can be saved with \code{saveRDS()} and
#' transferred to a remote server where population data resides, then used with
#' \code{\link{PSPI_predict}} to obtain population-level treatment effect estimates.
#'
#' This two-stage workflow supports federated analysis where trial data and
#' population data cannot be co-located.
#'
#' @param X Matrix of covariates for the trial data.
#' @param Y Numeric vector or logical vector of observed outcomes in the trial.
#' @param A Integer vector of treatment assignments (0, 1, ..., J-1).
#' @param pi Numeric matrix (n x J) of propensity scores for trial participants.
#' @param model Character string specifying which PSPI model to use.
#'   Options: \code{"BCF"}, \code{"BCF_P"}, \code{"FullBART"},
#'   \code{"SplineBART"}, \code{"DSplineBART"} (alias \code{"MSplineBART"}).
#' @param transformation Character string for propensity score transformation.
#'   Options: \code{"Identity"}, \code{"Logit"}, \code{"Cloglog"},
#'   \code{"InvGumbel"} (default).
#' @param nburn Number of burn-in iterations (default = 4000).
#' @param npost Number of posterior iterations saved (default = 4000).
#' @param n_knots Integer vector of spline knots per arm. If \code{NULL},
#'   chosen automatically.
#' @param order Order of spline basis functions (default = 3).
#' @param ntrees_s Number of trees for each BART component (default = 200).
#' @param sparse Logical; sparse Dirichlet prior (default = FALSE).
#' @param augment Logical; data augmentation for variable selection (default = FALSE).
#' @param verbose Logical; print progress (default = FALSE).
#' @param seed Optional random seed.
#'
#' @return An object of class \code{"PSPI_fit"} containing serialized BART tree
#'   structures, spline parameters, and metadata. Can be saved with
#'   \code{saveRDS()} and used with \code{\link{PSPI_predict}}.
#'
#' @examples
#' sim <- sim_generalizability(scenario = "linear", n_trial = 60)
#' ps_trial <- sim$population$ps[sim$population$selected]
#'
#' fit <- PSPI_fit(
#'   X = as.matrix(sim$trials[, paste0("X", 1:10)]),
#'   Y = sim$trials$Y,
#'   A = sim$trials$A,
#'   pi = cbind(ps_trial, ps_trial),
#'   model = "SplineBART",
#'   nburn = 1, npost = 1
#' )
#'
#'
#' @seealso \code{\link{PSPI_predict}}, \code{\link{PSPI_generalizability}}
#' @export
PSPI_fit = function(X, Y, A, pi, model, transformation = "InvGumbel", nburn = 4000, npost = 4000, n_knots = NULL, order = 3, ntrees_s = 200, sparse = FALSE, augment = FALSE, verbose = FALSE, seed = NULL){

  if(!methods::is(model, "character")){
    stop("Invalid model_name. Available options: BCF, BCF_P, FullBART, SplineBART, DSplineBART.")
  }

  model_name = model
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

  message(paste0("PSPI_fit: ", c("BCF", "BCF_P", "FullBART", "SplineBART", "DSplineBART")[model - 1]))

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

  if(!methods::is(transformation, "character")){
    stop("Invalid transformation.")
  }

  transformation_name = transformation
  transformation = stringr::str_to_upper(transformation)
  if(transformation == "LOGIT"){
    pi = arm::logit(pi)
    message("Transformation: Logit")
  }
  if(transformation == "CLOGLOG"){
    pi = log(-log(1 - pi))
    message("Transformation: Cloglog")
  }
  if(transformation == "INVGUMBEL"){
    pi = invgumbel(pi)
    message("Transformation: InvGumbel")
  }
  if(transformation == "IDENTITY"){
    message("Transformation: Identity")
  }

  if(is.logical(Y)){
    message("Outcome type: Binary")
  } else {
    message("Outcome type: Continuous")
  }

  if(!is.null(seed))
    set.seed(seed)
  if(length(unique(Y)) == 2)
    Y = as.logical(Y)

  result = MCMC_PSPI_fit(as.matrix(X), Y, is.logical(Y), A, pi, model, nburn, npost, n_knots, order, ntrees_s, verbose, sparse, augment)

  result$transformation = transformation_name
  result$model_name = model_name
  class(result) = "PSPI_fit"
  return(result)
}
