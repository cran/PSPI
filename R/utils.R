#' Gumbel quantile transform
#'
#' @description
#' Computes the inverse cumulative distribution function (quantile function)
#' of the standard Gumbel distribution via the
#' probability integral transform:
#' \deqn{Q(p) = -\log\{-\log(p)\}, \quad 0 < p < 1.}
#'
#' @param x Numeric vector or matrix of probabilities in \eqn{(0,1)}.
#'
#' @return A numeric vector or matrix of the same dimensions as \code{x}, containing
#' \eqn{-\log\{-\log(x)\}}.
#'
#'
#' @export
invgumbel = function(x){
  -log(-log(x))
}

#' Identify which population rows are included in a trial subset
#'
#' @description
#' Returns a logical vector indicating, for each row in \code{population},
#' whether that row is included in \code{trial}, where \code{trial} is assumed
#' to be a (row) subset of \code{population}. 
#'
#' @param population A data frame or a matrix
#'   representing the covariates of population
#' @param trial A data frame or a matrix
#'   representing the covariates of trial dataset, assumed to be a subset of \code{population}
#'   by rows.
#'
#' @return A logical vector of length \code{nrow(population)}. The \code{i}-th
#' element is \code{TRUE} if the \code{i}-th row of \code{population} appears
#' in \code{trial}, and \code{FALSE} otherwise.
#'
#'
#' @examples
#' pop <- data.frame(id = 1:5, x = c(0.1, 0.2, 0.3, 0.4, 0.5))
#' tri <- pop[c(2, 4), ]
#' in_trial(pop, tri)
#'
#' @export
in_trial <- function(population, trial){
  do.call(paste, c(as.data.frame(population), sep = "\r")) %in%
    do.call(paste, c(as.data.frame(trial), sep = "\r"))
}
  
  



