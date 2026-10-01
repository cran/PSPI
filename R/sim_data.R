#' Simulate a superpopulation and a randomized trial under PSPI generalizability scenarios
#'
#' @description
#' Generates a superpopulation of 100,000 individuals (treated as the target
#' population) with seven continuous and three binary covariates, constructs
#' potential outcomes \code{Y1} and \code{Y0} according to the chosen scenario,
#' and returns:
#' (i) a target sample of size \code{n_sample} drawn as a simple random
#'     sample from the target population;
#' (ii) a randomized trial of size \code{n_trial} drawn as a selective sample
#'      from the target sample via a logistic participation model; and
#' (iii) the true estimands at both the \emph{mixed-ATE (MATE)} level
#'       (averaged over the target sample) and the \emph{population-ATE
#'       (PATE)} level (averaged over the 100K superpopulation), following
#'       the estimand taxonomy in Li, Ding & Mealli (2022).
#'
#' Treatment assignment \eqn{A} is drawn independently at the superpopulation
#' level with probability \code{prop}, representing marginal randomization.
#' Optional diagnostic plots visualize covariate and outcome overlap between
#' the target sample and the trial.
#'
#' @param n_trial Integer. Trial sample size drawn by weighted sampling from
#'   the target sample. Must satisfy \code{n_trial <= n_sample}.
#' @param n_sample Integer. Target sample size drawn as a simple random
#'   sample from the target population. Defaults to \code{1000}.
#' @param scenario Character. One of \code{"linear"},
#'   \code{"linear+covariate shift"}, \code{"nonlinear"},
#'   \code{"nonlinear+covariate shift"}. Controls both the outcome-generating
#'   process and the participation model.
#' @param seed Optional integer seed for reproducibility. If \code{NULL}, the
#'   current RNG state is used.
#' @param prop Numeric in \code{[0,1]}. Randomization probability
#'   \eqn{\Pr(A = 1)} within the superpopulation.
#' @param plot Logical. If \code{TRUE}, constructs and prints two diagnostic
#'   figures comparing covariate and outcome distributions between the
#'   target sample and the trial. The plot objects are also returned inside
#'   the output list under \code{$plots}. Defaults to \code{FALSE}.
#'
#' @return A named \code{list} with three (or four, when \code{plot = TRUE})
#'   elements:
#' \describe{
#'   \item{\code{population}}{Data frame of size \code{n_sample} (the target
#'         sample). Columns: \code{X1:X10} (covariates), \code{A}
#'         (treatment indicator), \code{Y1}, \code{Y0} (potential outcomes),
#'         \code{ps} (oracle trial participation propensity score), and
#'         \code{selected} (logical; \code{TRUE} if the row is also in
#'         the trial).}
#'   \item{\code{trials}}{Data frame of size \code{n_trial} (the selective
#'         trial sample drawn from \code{population}). Columns: \code{X1:X10},
#'         \code{A}, \code{ps}, and the observed outcome
#'         \code{Y = A*Y1 + (1-A)*Y0}.}
#'   \item{\code{true_values}}{Named list of six true estimands:
#'         \code{true_ATE_MATE}, \code{true_Y1_MATE}, \code{true_Y0_MATE}
#'         (averaged over the \emph{target sample}, size \code{n_sample});
#'         \code{true_ATE_PATE}, \code{true_Y1_PATE}, \code{true_Y0_PATE}
#'         (averaged over the \emph{target population}, here the
#'         superpopulation).}
#'   \item{\code{plots}}{(Only when \code{plot = TRUE}.) Named list with
#'         \code{covariates} and \code{outcomes} ggplot objects.}
#' }
#'
#' @examples
#' set.seed(2025)
#' sim <- sim_generalizability(n_trial = 200, n_sample = 1000,
#'                             scenario = "nonlinear+covariate shift", prop = 0.5,
#'                             plot = TRUE)
#' str(sim$population)
#' table(sim$trials$A)               # trial treatment allocation
#' mean(sim$population$selected)      # fraction of target sample in trial
#' sim$true_values                    # true target-population ATE, E[Y1], E[Y0]
#'
#' # Smaller trial, linear scenario with covariate shift
#' sim2 <- sim_generalizability(n_trial = 60, n_sample = 1000,
#'                              scenario = "linear+covariate shift",
#'                              seed = 1, prop = 0.6)
#' nrow(sim2$trials)                  # 60
#'
#' @importFrom mvtnorm rmvnorm
#' @importFrom stats rbinom rnorm runif
#' @importFrom arm invlogit
#' @export
sim_generalizability = function(n_trial = 200, n_sample = 1000, scenario = "linear",
                                seed = NULL, prop = 0.5, plot = FALSE){
  if (!scenario %in% c("linear", "linear+covariate shift", "nonlinear", "nonlinear+covariate shift")) {
    stop("Invalid scenario name. Available options: linear, linear+covariate shift, nonlinear, nonlinear+covariate shift.")
  }
  
  if (prop > 1 | prop < 0) {
    stop("Invalid randomization proportion. Should be between 0 and 1.")
  }
  
  if (n_trial > n_sample) {
    stop("n_trial must not exceed n_sample.")
  }
  
  
  # Set the seed for reproducibility
  if(!is.null(seed))
    set.seed(seed)
  
  # superpopulation
  n_pop = 100000
  
  
  # Number of continuous and binary variables
  n_continuous <- 7
  n_binary <- 3
  
  # Generate continuous variables (normally distributed)
  continuous_mean = rep(0, n_continuous)
  correlation = matrix(
    c(1, 0.2, 0, 0, 0, 0, 0,
      0.2, 1, 0, 0, 0, 0, 0,
      0, 0, 1, 0.5, 0, 0, 0,
      0, 0, 0.5, 1, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 1),
    nrow = n_continuous, ncol = n_continuous
  )
  continuous_vars <- mvtnorm::rmvnorm(n_pop, mean = continuous_mean, sigma = correlation)
  
  
  
  # Generate binary variables (Bernoulli distributed)
  binary_mean = rep(0.5, n_binary)
  binary_vars <- sapply(1:n_binary, function(i) rbinom(n_pop, size = 1, prob = binary_mean[i]))
  
  # Combine continuous and binary variables into one data frame
  superpopulation_data <- data.frame(continuous_vars, binary_vars)
  
  # Rename the columns
  colnames(superpopulation_data) <- c(paste0("cont_var", 1:n_continuous), paste0("bin_var", 1:n_binary))
  
  superpopulation_data$A = rbinom(dim(superpopulation_data)[1], 1, prop)
  
  
  
  if(scenario %in% c("linear", "linear+covariate shift")){
    superpopulation_data$outcome1 = with(
      superpopulation_data,
      2 * cont_var1 + -1.5 * cont_var2 + 0.5 * cont_var3 + 1 * cont_var4 + 1 * bin_var1 +
        1 * (2 + 3 * cont_var1 + 2 * cont_var2 + 1 * cont_var3 + 1 * cont_var4 + 3 * bin_var1)
    )
    
    
    superpopulation_data$outcome0 = with(
      superpopulation_data,
      2 * cont_var1 + -1.5 * cont_var2 + 0.5 * cont_var3 + 1 * cont_var4 + 1 * bin_var1
    )
  }
  
  
  if(scenario %in% c("nonlinear", "nonlinear+covariate shift")){
    superpopulation_data$outcome1 = with(superpopulation_data,
                                         2 * cont_var1  + 1 * cont_var2 + 3 * cont_var3 + 2 * cont_var4 + 3 * bin_var1 +
                                           2 * cont_var1 * cont_var3 * bin_var1 + 0.8 * cont_var2 * cont_var4 + 0.5 * cont_var1^2 +
                                           (2 + 3 * cont_var1 + 1 * cont_var2 - 1 * cont_var3 - 2 * cont_var4 + 3 * bin_var1 +
                                              1.5 *  cont_var1 * cont_var4 + 0.8 * cont_var3^2 + 0.5 * cont_var2 * bin_var1 + 0.3 * cont_var4^3 * bin_var1 + arm::invlogit(cont_var1 * cont_var2)))
    
    
    superpopulation_data$outcome0 = with(superpopulation_data,
                                         2 * cont_var1  + 1 * cont_var2 + 3 * cont_var3 + 2 * cont_var4 + 3 * bin_var1 +
                                           2 * cont_var1 * cont_var3 * bin_var1 + 0.8 * cont_var2 * cont_var4 + 0.5 * cont_var1^2)
  }
  
  
  superpopulation_data$outcome1 =  superpopulation_data$outcome1 + rnorm(n_pop)
  superpopulation_data$outcome0 =  superpopulation_data$outcome0 + rnorm(n_pop)
  
  
  target_sample_ID = sample(1:n_pop, size = n_sample, replace = FALSE)
  target_sample = superpopulation_data[target_sample_ID, ]

  # ---- True estimands ------------------------------------------------------
  # MATE (mixed ATE):   N^{-1} sum_i  tau(X_i)        averaged over target SAMPLE
  # PATE (population):  E[tau(X)]                     averaged over target POP
  # In this nested design, the target population = superpopulation.
  # See Li, Ding & Mealli (2022), Sections 2 & 3.1, eqs (2.2) and (3.2).
  true_values = list(
    true_ATE_MATE = mean(target_sample$outcome1 - target_sample$outcome0),
    true_Y1_MATE  = mean(target_sample$outcome1),
    true_Y0_MATE  = mean(target_sample$outcome0),
    true_ATE_PATE = mean(superpopulation_data$outcome1 - superpopulation_data$outcome0),
    true_Y1_PATE  = mean(superpopulation_data$outcome1),
    true_Y0_PATE  = mean(superpopulation_data$outcome0)
  )
  
  
  
  
  # selection model
  pi <- with(target_sample, {
    if(scenario == "linear"){
      arm::invlogit(-1.41 + 1.5 * cont_var1 - 0.7 * cont_var2 + 0.5 * cont_var5 + 1 * cont_var4 - 2 * bin_var1)
    } else if(scenario == "linear+covariate shift"){
      arm::invlogit(-2.38 + 1.8*cont_var1 - 1.5*cont_var2^2 - 0.8*cont_var5*bin_var1 - 6*(cont_var4 - 0.5)^2)
    } else if(scenario == "nonlinear"){
      arm::invlogit(-2.5 + 4 * cont_var1 - 1.5 * cont_var2^2 + 1.2 * cont_var4 -
                      2.4 * bin_var1 + 0.6 * cont_var5 + 0.6 * cont_var1 * cont_var4)
    } else if(scenario == "nonlinear+covariate shift"){
      arm::invlogit(-2 + 8 * cont_var1 - 5 * cont_var2^2 - 5 * (cont_var4 - 0.5)^2 - 0.8 * cont_var5 * bin_var1)
    } else {
      stop("Invalid scenario")
    }
  })
  
  
  
  if (any(is.na(pi))) {
    stop("Some rows did not match any scenario/n_trial branch.")
  }
  colnames(target_sample) = c(paste0("X", 1:10), "A", "Y1", "Y0")
  target_sample$ps = pi
  ID = sample(1:n_sample, size = n_trial, replace = FALSE, prob = pi)
  target_sample$selected = FALSE
  target_sample$selected[ID] = TRUE
  
  
  trials = target_sample[ID, ]
  trials$selected = NULL
  trials$Y = ifelse(trials$A == 1, trials$Y1, trials$Y0)
  trials$Y0 = NULL
  trials$Y1 = NULL
  
  
  out = list(population = target_sample, trials = trials, true_values = true_values)
  
  
  if(plot){
    if(!requireNamespace("ggplot2", quietly = TRUE)){
      warning("Package 'ggplot2' is not available; skipping plots.")
    } else {
      trial_super_idx <- target_sample_ID[ID]
      plot_df <- rbind(
        data.frame(group = "Target sample", superpopulation_data[target_sample_ID, ]),
        data.frame(group = "Trial (S=1)",   superpopulation_data[trial_super_idx, ])
      )
      plot_df$group <- factor(plot_df$group,
                              levels = c("Target sample", "Trial (S=1)"))
      
      covars <- c(paste0("cont_var", 1:7), paste0("bin_var", 1:3))
      long_cov <- do.call(rbind, lapply(covars, function(v){
        data.frame(group = plot_df$group, variable = v, value = plot_df[[v]])
      }))
      long_cov$variable <- factor(long_cov$variable, levels = covars)
      
      p_cov <- ggplot2::ggplot(long_cov,
                               ggplot2::aes(x = value, fill = group, color = group)) +
        ggplot2::geom_histogram(ggplot2::aes(y = ggplot2::after_stat(density)),
                                position = "identity", alpha = 0.3, bins = 30) +
        ggplot2::facet_wrap(~ variable, scales = "free", ncol = 4) +
        ggplot2::theme_bw(base_size = 11) +
        ggplot2::labs(title = paste0("Covariate distributions: ", scenario),
                      x = NULL, y = "Density", fill = NULL, color = NULL) +
        ggplot2::theme(legend.position = "bottom",
                       panel.grid.minor = ggplot2::element_blank())
      
      long_y <- rbind(
        data.frame(group = plot_df$group, outcome = "Y0",          value = plot_df$outcome0),
        data.frame(group = plot_df$group, outcome = "Y1",          value = plot_df$outcome1),
        data.frame(group = plot_df$group, outcome = "ITE (Y1-Y0)", value = plot_df$outcome1 - plot_df$outcome0)
      )
      long_y$outcome <- factor(long_y$outcome, levels = c("Y0", "Y1", "ITE (Y1-Y0)"))
      
      p_y <- ggplot2::ggplot(long_y,
                             ggplot2::aes(x = value, fill = group, color = group)) +
        ggplot2::geom_density(alpha = 0.3) +
        ggplot2::facet_wrap(~ outcome, scales = "free", ncol = 3) +
        ggplot2::theme_bw(base_size = 11) +
        ggplot2::labs(title = paste0("Outcome distributions: ", scenario),
                      x = NULL, y = "Density", fill = NULL, color = NULL) +
        ggplot2::theme(legend.position = "bottom",
                       panel.grid.minor = ggplot2::element_blank())
      
      print(p_cov)
      print(p_y)
      
      out$plots <- list(covariates = p_cov, outcomes = p_y)
    }
  }
  
  
  return(out)
}






#' Simulate a superpopulation, a randomized trial, and a target sample under
#' PSPI transportability scenarios
#'
#' @description
#' Generates a superpopulation of 100,000 individuals with seven continuous and
#' three binary covariates, constructs potential outcomes \code{Y1} and
#' \code{Y0} according to the chosen scenario, and returns a randomized trial,
#' a target population, a target sample, and the true estimands at both
#' the \emph{mixed-ATE (MATE)} level (averaged over the target sample) and
#' the \emph{population-ATE (PATE)} level (averaged over the target
#' population), following the estimand taxonomy in Li, Ding & Mealli (2022).
#' Optional diagnostic
#' plots visualize covariate and outcome overlap between the superpopulation,
#' the trial, and the target sample.
#'
#' @param n_trial Integer. Trial sample size drawn from the superpopulation.
#' @param n_target_pop Integer. Target population size drawn from the
#'   non-randomized portion of the superpopulation. Defaults to \code{5000}.
#' @param n_sample Integer. Target sample size drawn as a simple random sample
#'   from the target population. Must satisfy \code{n_sample <= n_target_pop}.
#'   Defaults to \code{1000}.
#' @param scenario Character. One of \code{"linear"},
#'   \code{"linear+covariate shift"}, \code{"nonlinear"},
#'   \code{"nonlinear+covariate shift"}.
#' @param seed Optional integer seed for reproducibility.
#' @param prop Numeric in \code{[0,1]}. Randomization probability
#'   \eqn{\Pr(A = 1)} within the superpopulation.
#' @param target_shift_coef Numeric. Coefficient on \code{X1} in the target
#'   population selection model
#'   \eqn{\Pr(\text{target} \mid X) = \mathrm{invlogit}(c \cdot X_1)}.
#'   Defaults to \code{-0.3}, which produces a mild shift (target slightly
#'   over-samples low \eqn{X_1}, opposite but gentle relative to the trial
#'   participation model). Set to \code{0} for (approximately) simple random
#'   sampling from the non-trial superpopulation; increase in magnitude for
#'   stronger covariate shift.
#' @param plot Logical. If \code{TRUE}, constructs and prints two diagnostic
#'   figures comparing covariate and outcome distributions across the
#'   superpopulation, the trial, and the target sample. The plot objects are
#'   also returned inside the output list under \code{$plots}. Defaults to
#'   \code{FALSE}.
#'
#' @return A named \code{list} with three (or four, when \code{plot = TRUE})
#'   elements:
#' \describe{
#'   \item{\code{population}}{Data frame of size \code{n_sample} (the target
#'         sample drawn as a SRS from the target population). Columns:
#'         \code{X1:X10} (covariates), \code{A} (treatment indicator, unused
#'         for target inference), \code{Y1}, \code{Y0} (potential outcomes,
#'         provided for verification), \code{ps_trial} (oracle trial
#'         participation PS), \code{ps_target} (oracle target enrollment PS).}
#'   \item{\code{trials}}{Data frame of size \code{n_trial} (the randomized
#'         trial). Columns: \code{X1:X10}, \code{A}, observed outcome
#'         \code{Y = A*Y1 + (1-A)*Y0}, \code{ps_trial}, \code{ps_target}.}
#'   \item{\code{true_values}}{Named list of six true estimands:
#'         \code{true_ATE_MATE}, \code{true_Y1_MATE}, \code{true_Y0_MATE}
#'         (averaged over the \emph{target sample}, size \code{n_sample});
#'         \code{true_ATE_PATE}, \code{true_Y1_PATE}, \code{true_Y0_PATE}
#'         (averaged over the \emph{target population}, size
#'         \code{n_target_pop}, corresponding to \eqn{E[Y^a \mid S=0]} and
#'         \eqn{E[Y^1 - Y^0 \mid S=0]}).}
#'   \item{\code{plots}}{(Only when \code{plot = TRUE}.) Named list with
#'         \code{covariates} and \code{outcomes} ggplot objects.}
#' }
#'
#' @examples
#' set.seed(2025)
#' sim <- sim_trans(n_trial = 200, n_target_pop = 5000, n_sample = 1000,
#'                  scenario = "nonlinear", prop = 0.5, plot = TRUE)
#' str(sim$population)
#' table(sim$trials$A)
#' sim$true_values
#'
#' sim2 <- sim_trans(n_trial = 60, scenario = "linear+covariate shift",
#'                   seed = 1, prop = 0.6)
#' nrow(sim2$trials)
#'
#' @importFrom mvtnorm rmvnorm
#' @importFrom stats rbinom rnorm runif
#' @importFrom arm invlogit
#' @export
sim_trans = function(n_trial = 200, n_target_pop = 5000, n_sample = 1000,
                     scenario = "linear", seed = NULL, prop = 0.5,
                     target_shift_coef = -0.3,
                     plot = FALSE){
  if (!scenario %in% c("linear", "linear+covariate shift", "nonlinear", "nonlinear+covariate shift")) {
    stop("Invalid scenario name. Available options: linear, linear+covariate shift, nonlinear, nonlinear+covariate shift.")
  }
  
  if (prop > 1 | prop < 0) {
    stop("Invalid randomization proportion. Should be between 0 and 1.")
  }
  
  if (n_sample > n_target_pop) {
    stop("n_sample must not exceed n_target_pop.")
  }
  
  
  if(!is.null(seed))
    set.seed(seed)
  
  n_pop = 100000
  
  
  n_continuous <- 7
  n_binary <- 3
  
  continuous_mean = rep(0, n_continuous)
  correlation = matrix(
    c(1, 0.2, 0, 0, 0, 0, 0,
      0.2, 1, 0, 0, 0, 0, 0,
      0, 0, 1, 0.5, 0, 0, 0,
      0, 0, 0.5, 1, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 1),
    nrow = n_continuous, ncol = n_continuous
  )
  continuous_vars <- mvtnorm::rmvnorm(n_pop, mean = continuous_mean, sigma = correlation)
  
  
  binary_mean = rep(0.5, n_binary)
  binary_vars <- sapply(1:n_binary, function(i) rbinom(n_pop, size = 1, prob = binary_mean[i]))
  
  superpopulation_data <- data.frame(continuous_vars, binary_vars)
  
  colnames(superpopulation_data) <- c(paste0("cont_var", 1:n_continuous), paste0("bin_var", 1:n_binary))
  
  superpopulation_data$A = rbinom(dim(superpopulation_data)[1], 1, prop)
  
  
  if(scenario %in% c("linear", "linear+covariate shift")){
    superpopulation_data$outcome1 = with(
      superpopulation_data,
      2 * cont_var1 + -1.5 * cont_var2 + 0.5 * cont_var3 + 1 * cont_var4 + 1 * bin_var1 +
        1 * (2 + 3 * cont_var1 + 2 * cont_var2 + 1 * cont_var3 + 1 * cont_var4 + 3 * bin_var1)
    )
    
    superpopulation_data$outcome0 = with(
      superpopulation_data,
      2 * cont_var1 + -1.5 * cont_var2 + 0.5 * cont_var3 + 1 * cont_var4 + 1 * bin_var1
    )
  }
  
  
  if(scenario %in% c("nonlinear", "nonlinear+covariate shift")){
    superpopulation_data$outcome1 = with(
      superpopulation_data,
      2 * cont_var1  + 1 * cont_var2 + 3 * cont_var3 + 2 * cont_var4 + 3 * bin_var1 +
        2 * cont_var1 * cont_var3 * bin_var1 + 0.8 * cont_var2 * cont_var4 + 0.5 * cont_var1^2 +
        (2 + 3 * cont_var1 + 1 * cont_var2 - 1 * cont_var3 - 2 * cont_var4 + 3 * bin_var1 +
           1.5 * cont_var1 * cont_var4 + 0.8 * cont_var3^2 + 0.5 * cont_var2 * bin_var1 + 0.3 * cont_var4^3 * bin_var1 + arm::invlogit(cont_var1 * cont_var2))
    )
    
    superpopulation_data$outcome0 = with(
      superpopulation_data,
      2 * cont_var1  + 1 * cont_var2 + 3 * cont_var3 + 2 * cont_var4 + 3 * bin_var1 +
        2 * cont_var1 * cont_var3 * bin_var1 + 0.8 * cont_var2 * cont_var4 + 0.5 * cont_var1^2
    )
  }
  
  superpopulation_data$outcome1 =  superpopulation_data$outcome1 + rnorm(n_pop)
  superpopulation_data$outcome0 =  superpopulation_data$outcome0 + rnorm(n_pop)
  
  
  pi_trial <- with(superpopulation_data, {
    if(scenario == "linear"){
      arm::invlogit(-1.41 + 1.5 * cont_var1 - 0.7 * cont_var2 + 0.5 * cont_var5 + 1 * cont_var4 - 2 * bin_var1)
    } else if(scenario == "linear+covariate shift"){
      arm::invlogit(-2.38 + 1.8*cont_var1 - 1.5*cont_var2^2 - 0.8*cont_var5*bin_var1 - 6*(cont_var4 - 0.5)^2)
    } else if(scenario == "nonlinear"){
      arm::invlogit(-2.5 + 4 * cont_var1 - 1.5 * cont_var2^2 + 1.2 * cont_var4 -
                      2.4 * bin_var1 + 0.6 * cont_var5 + 0.6 * cont_var1 * cont_var4)
    } else if(scenario == "nonlinear+covariate shift"){
      arm::invlogit(-2 + 8 * cont_var1 - 5 * cont_var2^2 - 5 * (cont_var4 - 0.5)^2 - 0.8 * cont_var5 * bin_var1)
    } else {
      stop("Invalid scenario")
    }
  })
  
  if (any(is.na(pi_trial))) {
    stop("Some rows did not match any scenario branch in the trial participation model.")
  }
  
  # Target-population selection model.  Simple, single-covariate logistic with
  # a tunable coefficient on X1 (see `target_shift_coef`).  At coef=-0.3 this
  # produces a mild opposing shift relative to pi_trial (which generally
  # favours high X1); at coef=0 it reduces to near-SRS of the non-trial
  # superpopulation.  See vignette / notes/superpopulation_logic.md.
  pi_target = with(superpopulation_data,
                   arm::invlogit(target_shift_coef * cont_var1)
  )
  
  # Store BOTH oracle PS on superpopulation so they propagate to both samples
  superpopulation_data$ps_trial  = pi_trial
  superpopulation_data$ps_target = pi_target
  
  
  # Trial: selective sample via pi_trial
  trial_ID = sample(1:n_pop, size = n_trial, replace = FALSE, prob = pi_trial)
  
  # Target pool: exclude trial participants (non-nested design)
  pi_target_adj = pi_target
  pi_target_adj[trial_ID] = 0
  target_pop_ID = sample(1:n_pop, size = n_target_pop, replace = FALSE, prob = pi_target_adj)
  
  
  target_sample_ID = sample(target_pop_ID, size = n_sample, replace = FALSE)
  target_sample = superpopulation_data[target_sample_ID, ]

  # ---- True estimands ------------------------------------------------------
  # MATE (mixed ATE):  N^{-1} sum_i  tau(X_i)  on the target SAMPLE (n_sample)
  # PATE (population): E[tau(X)|S=0]           on the target POPULATION (n_target_pop)
  # See Li, Ding & Mealli (2022), Sections 2 & 3.1, eqs (2.2) and (3.2).
  true_values = list(
    true_ATE_MATE = mean(target_sample$outcome1 - target_sample$outcome0),
    true_Y1_MATE  = mean(target_sample$outcome1),
    true_Y0_MATE  = mean(target_sample$outcome0),
    true_ATE_PATE = mean(superpopulation_data$outcome1[target_pop_ID] -
                        superpopulation_data$outcome0[target_pop_ID]),
    true_Y1_PATE  = mean(superpopulation_data$outcome1[target_pop_ID]),
    true_Y0_PATE  = mean(superpopulation_data$outcome0[target_pop_ID])
  )

  colnames(target_sample) = c(paste0("X", 1:10), "A", "Y1", "Y0", "ps_trial", "ps_target")
  
  
  trials = superpopulation_data[trial_ID, ]
  colnames(trials) = c(paste0("X", 1:10), "A", "Y1", "Y0", "ps_trial", "ps_target")
  trials$Y  = ifelse(trials$A == 1, trials$Y1, trials$Y0)
  trials$Y0 = NULL
  trials$Y1 = NULL
  
  
  out = list(population = target_sample, trials = trials, true_values = true_values)
  
  
  if(plot){
    if(!requireNamespace("ggplot2", quietly = TRUE)){
      warning("Package 'ggplot2' is not available; skipping plots.")
    } else {
      super_idx <- sample(1:n_pop, size = min(5000, n_pop), replace = FALSE)
      plot_df <- rbind(
        data.frame(group = "Superpopulation",       superpopulation_data[super_idx, ]),
        data.frame(group = "Trial (S=1)",           superpopulation_data[trial_ID, ]),
        data.frame(group = "Target sample (S=0)",   superpopulation_data[target_sample_ID, ])
      )
      plot_df$group <- factor(plot_df$group,
                              levels = c("Superpopulation", "Trial (S=1)", "Target sample (S=0)"))
      
      covars <- c(paste0("cont_var", 1:7), paste0("bin_var", 1:3))
      long_cov <- do.call(rbind, lapply(covars, function(v){
        data.frame(group = plot_df$group, variable = v, value = plot_df[[v]])
      }))
      long_cov$variable <- factor(long_cov$variable, levels = covars)
      
      p_cov <- ggplot2::ggplot(long_cov,
                               ggplot2::aes(x = value, fill = group, color = group)) +
        ggplot2::geom_histogram(ggplot2::aes(y = ggplot2::after_stat(density)),
                                position = "identity", alpha = 0.3, bins = 30) +
        ggplot2::facet_wrap(~ variable, scales = "free", ncol = 4) +
        ggplot2::theme_bw(base_size = 11) +
        ggplot2::labs(title = paste0("Covariate distributions: ", scenario),
                      x = NULL, y = "Density", fill = NULL, color = NULL) +
        ggplot2::theme(legend.position = "bottom",
                       panel.grid.minor = ggplot2::element_blank())
      
      long_y <- rbind(
        data.frame(group = plot_df$group, outcome = "Y0",          value = plot_df$outcome0),
        data.frame(group = plot_df$group, outcome = "Y1",          value = plot_df$outcome1),
        data.frame(group = plot_df$group, outcome = "ITE (Y1-Y0)", value = plot_df$outcome1 - plot_df$outcome0)
      )
      long_y$outcome <- factor(long_y$outcome, levels = c("Y0", "Y1", "ITE (Y1-Y0)"))
      
      p_y <- ggplot2::ggplot(long_y,
                             ggplot2::aes(x = value, fill = group, color = group)) +
        ggplot2::geom_density(alpha = 0.3) +
        ggplot2::facet_wrap(~ outcome, scales = "free", ncol = 3) +
        ggplot2::theme_bw(base_size = 11) +
        ggplot2::labs(title = paste0("Outcome distributions: ", scenario),
                      x = NULL, y = "Density", fill = NULL, color = NULL) +
        ggplot2::theme(legend.position = "bottom",
                       panel.grid.minor = ggplot2::element_blank())
      
      print(p_cov)
      print(p_y)
      
      out$plots <- list(covariates = p_cov, outcomes = p_y)
    }
  }
  
  return(out)
}
