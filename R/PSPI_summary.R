#' Summarize a fitted PSPI object
#'
#' @description
#' S3 \code{summary} method for objects returned by
#' \code{\link{PSPI_generalizability}} or \code{\link{PSPI_predict}}.
#' Produces posterior point estimates and credible intervals for both the
#' \emph{mixed average treatment effect (MATE)} and the
#' \emph{population average treatment effect (PATE)}, on each potential
#' outcome \eqn{E[Y(a)]} and on each treatment contrast
#' \eqn{E[Y(a) - Y(0)]}.
#'
#' \strong{MATE} averages the conditional treatment effect uniformly over
#' the observed target sample (the de-facto default of most Bayesian
#' causal inference; Li, Ding & Mealli 2022, eq.~(3.2)). \strong{PATE}
#' uses Bayesian-bootstrap (Dirichlet) weights to integrate the
#' conditional effect against the posterior of the covariate distribution,
#' giving the strictly larger and more honest population-level uncertainty
#' (Li, Ding & Mealli 2022, eq.~(2.2)).
#'
#' @param object A fitted object of class \code{"PSPI_generalizability"} or
#'   \code{"PSPI_predict"}, typically holding the array
#'   \code{post_outcome} (npost x N x J) and, if \code{pate_bootstrap = TRUE}
#'   was used during fitting, the matrix \code{post_outcome_PATE}
#'   (npost x J).
#' @param level Coverage level for the credible intervals (default
#'   \code{0.95}).
#' @param ... Ignored (S3 conformity).
#'
#' @return A data frame of class \code{"summary.PSPI"} with one row per
#'   (estimand, target) combination and columns \code{estimand}
#'   (\code{"E[Y(a)]"} or \code{"E[Y(a)-Y(0)]"}), \code{target}
#'   (\code{"MATE"} or \code{"PATE"}), \code{mean}, \code{lower},
#'   \code{upper}, \code{width}, \code{sd}.
#'
#' @examples
#' \dontrun{
#' sim <- sim_generalizability(n_trial = 200, n_sample = 1000,
#'                              scenario = "linear", seed = 1)
#' fit <- PSPI_generalizability(
#'   X = as.matrix(sim$trials[, paste0("X", 1:10)]),
#'   Y = sim$trials$Y, A = sim$trials$A,
#'   pi = cbind(sim$trials$ps, sim$trials$ps),
#'   X_pop = as.matrix(sim$population[, paste0("X", 1:10)]),
#'   pi_pop = cbind(sim$population$ps, sim$population$ps),
#'   model = "DSplineBART", nburn = 500, npost = 500)
#' summary(fit)
#' }
#'
#' @seealso \code{\link{PSPI_generalizability}}, \code{\link{PSPI_predict}}
#' @importFrom stats quantile sd
#' @export
summary.PSPI_generalizability <- function(object, level = 0.95, ...){
  .pspi_summary_table(object, level = level)
}

#' @rdname summary.PSPI_generalizability
#' @export
summary.PSPI_predict <- function(object, level = 0.95, ...){
  .pspi_summary_table(object, level = level)
}


# ---------- internal builder ---------------------------------------------
.pspi_summary_table <- function(object, level = 0.95){
  if(is.null(object$post_outcome)){
    stop("No `post_outcome` array found on the fit object; nothing to summarise.")
  }
  po <- object$post_outcome              # [npost, N, J]
  npost <- dim(po)[1]
  J     <- dim(po)[3]
  has_pate <- !is.null(object$post_outcome_PATE)

  a <- (1 - level) / 2
  q <- function(x) stats::quantile(x, c(a, 1 - a), names = FALSE, na.rm = TRUE)

  rows <- list()
  # E[Y(a)] for each arm
  Y_MATE <- vapply(seq_len(J), function(j) rowMeans(po[, , j]),
                   numeric(npost))                            # [npost, J]
  Y_PATE <- if(has_pate) object$post_outcome_PATE else NULL    # [npost, J] or NULL

  add_row <- function(label, target, draws){
    qq <- q(draws)
    rows[[length(rows) + 1L]] <<- data.frame(
      estimand = label, target = target,
      mean     = mean(draws),
      lower    = qq[1], upper = qq[2],
      width    = qq[2] - qq[1],
      sd       = stats::sd(draws),
      stringsAsFactors = FALSE)
  }

  # marginal means E[Y(a)]
  for(j in seq_len(J)){
    add_row(sprintf("E[Y(%d)]", j - 1L), "MATE", Y_MATE[, j])
    if(has_pate) add_row(sprintf("E[Y(%d)]", j - 1L), "PATE", Y_PATE[, j])
  }
  # contrasts E[Y(a) - Y(0)] vs reference
  if(J >= 2L){
    for(j in 2:J){
      add_row(sprintf("E[Y(%d)-Y(0)]", j - 1L), "MATE",
              Y_MATE[, j] - Y_MATE[, 1])
      if(has_pate)
        add_row(sprintf("E[Y(%d)-Y(0)]", j - 1L), "PATE",
                Y_PATE[, j] - Y_PATE[, 1])
    }
  }
  out <- do.call(rbind, rows)
  class(out) <- c("summary.PSPI", "data.frame")
  attr(out, "level")    <- level
  attr(out, "n_target") <- dim(po)[2]
  attr(out, "npost")    <- npost
  out
}

#' @export
print.summary.PSPI <- function(x, digits = 3, ...){
  lvl   <- attr(x, "level")    %||% 0.95
  nt    <- attr(x, "n_target") %||% NA
  npost <- attr(x, "npost")    %||% NA
  cat(sprintf("PSPI posterior summary (level = %.0f%%, n_target = %d, npost = %d)\n",
              100 * lvl, nt, npost))
  cat("MATE = mixed ATE (uniform 1/N average over observed target sample)\n")
  cat("PATE = population ATE (Bayesian-bootstrap of F_X; eq. (2.2) in Li, Ding & Mealli 2022)\n\n")
  body <- as.data.frame(x)
  body[, c("mean","lower","upper","width","sd")] <-
    lapply(body[, c("mean","lower","upper","width","sd")], round, digits = digits)
  print(body, row.names = FALSE)
  invisible(x)
}

# null-coalescing helper (avoids dependency)
`%||%` <- function(a, b) if(is.null(a)) b else a
