
#' Simulate a population and a randomized trial under PSPI scenarios
#'
#' @description
#' Generates a finite population of size 1000 with seven continuous and three
#' binary covariates, constructs potential outcomes \code{Y0} and \code{Y1}
#' according to the chosen scenario, simulates trial participation through a
#' logistic selection model calibrated to target \code{n_trial} = 200 or 60,
#' and returns both the target population and the randomized trial
#' (with treatment assigned at probability \code{prop}).
#'
#'
#' @param n_trial Integer. Target trial size; must be \code{200} or \code{60}.
#' @param scenario Character. One of \code{"linear"}, \code{"linear+covariate shift"},
#'   \code{"nonlinear"}, \code{"nonlinear+covariate shift"}.
#' @param seed Optional integer seed for reproducibility. If \code{NULL}, the current
#'   RNG state is used.
#' @param prop Numeric in \code{[0,1]}. Randomization probability \eqn{P(A=1)} within
#'   the trial.
#'
#' @return A \code{list} with two data frames:
#' \describe{
#'   \item{population}{columns \code{X1:X10}, potential outcomes \code{Y1} and \code{Y0},
#'         \code{selected} (logical), and \code{ps} (true propensity scores of trial participation).}
#'   \item{trials}{columns \code{X1:X10}, \code{A}, and observed \code{Y}.}
#' }
#'
#' @examples
#' set.seed(2025)
#' sim <- sim_data(n_trial = 200, scenario = "nonlinear", prop = 0.5)
#' str(sim$population)
#' table(sim$trials$A)            # treatment allocation
#' mean(sim$population$selected)   # selection rate
#'
#' # A smaller trial size and linear scenario with covariate shift
#' sim2 <- sim_data(n_trial = 60, scenario = "linear+covariate shift", seed = 1, prop = 0.6)
#' nrow(sim2$trials)
#'
#'
#' @importFrom mvtnorm rmvnorm
#' @importFrom stats rbinom rnorm runif
#' @importFrom dplyr case_when
#' @importFrom arm invlogit
#' @export


sim_data = function(n_trial = 200, scenario = "linear", seed = NULL, prop = 0.5){
  if (!scenario %in% c("linear", "linear+covariate shift", "nonlinear", "nonlinear+covariate shift")) {
    stop("Invalid scenario name. Available options: linear, linear+covariate shift, nonlinear, nonlinear+covariate shift.")
  }
  
  if (!n_trial %in% c(200, 60)) {
    stop("Invalid trial sample size. Available options: 200, 60.")
  }
  
  if (prop > 1 | prop < 0) {
    stop("Invalid randomization proportion. Should be between 0 and 1.")
  }
  
  
  # Set the seed for reproducibility
  if(!is.null(seed))
    set.seed(seed)
  
  # population size
  n_samples = 1000
  
  
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
  continuous_vars <- mvtnorm::rmvnorm(n_samples, mean = continuous_mean, sigma = correlation)
  
  
  
  # Generate binary variables (Bernoulli distributed)
  binary_mean = rep(0.5, n_binary)
  binary_vars <- sapply(1:n_binary, function(i) rbinom(n_samples, size = 1, prob = binary_mean[i]))
  
  # Combine continuous and binary variables into one data frame
  data <- data.frame(continuous_vars, binary_vars)
  
  # Rename the columns
  colnames(data) <- c(paste0("cont_var", 1:n_continuous), paste0("bin_var", 1:n_binary))
  
  if(scenario %in% c("linear", "linear+covariate shift")){
    data$outcome1 = 
      2 * data$cont_var1 + -1.5 * data$cont_var2 + 0.5 * data$cont_var3 + 1 * data$cont_var4 + 1 * data$bin_var1 + 
      1 * (2 + 3 * data$cont_var1 + 2 * data$cont_var2 + 1 * data$cont_var3 + 1 * data$cont_var4 + 3 * data$bin_var1) 
    
    data$outcome0 = 
      2 * data$cont_var1 + -1.5 * data$cont_var2 + 0.5 * data$cont_var3 + 1 * data$cont_var4 + 1 * data$bin_var1
  }
  if(scenario %in% c("nonlinear", "nonlinear+covariate shift")){
    data$outcome1 = with(data,
                         2 * cont_var1  + 1 * cont_var2 + 3 * cont_var3 + 2 * cont_var4 + 3 * bin_var1 + 
                           2 * cont_var1 * cont_var3 * bin_var1 + 0.8 * cont_var2 * cont_var4 + 0.5 * cont_var1^2 +
                           (2 + 3 * cont_var1 + 1 * cont_var2 - 1 * cont_var3 - 2 * cont_var4 + 3 * bin_var1 +
                              1.5 *  cont_var1 * cont_var4 + 0.8 * cont_var3^2 + 0.5 * cont_var2 * bin_var1 + 0.3 * cont_var4^3 * bin_var1 + arm::invlogit(cont_var1 * cont_var2)))
    
    
    data$outcome0 = with(data,
                         2 * cont_var1  + 1 * cont_var2 + 3 * cont_var3 + 2 * cont_var4 + 3 * bin_var1 + 
                           2 * cont_var1 * cont_var3 * bin_var1 + 0.8 * cont_var2 * cont_var4 + 0.5 * cont_var1^2) 
  }
  data$outcome1 =  data$outcome1 + rnorm(n_samples)
  data$outcome0 =  data$outcome0 + rnorm(n_samples)
  
  # selection model
  pi = dplyr::case_when(
    scenario == "linear" & n_trial == 60  ~ arm::invlogit(-3.46 + 1.5 * data$cont_var1 - 0.7 * data$cont_var2  + 0.5 * data$cont_var3 + 1 * data$cont_var4 - 2 * data$bin_var1),
    scenario == "linear" & n_trial == 200 ~ arm::invlogit(-1.41 + 1.5 * data$cont_var1 - 0.7 * data$cont_var2  + 0.5 * data$cont_var3 + 1 * data$cont_var4 - 2 * data$bin_var1),
    
    scenario == "linear+covariate shift" & n_trial == 60  ~ arm::invlogit(-2.19 + 3 * data$cont_var1 - 1.5 * data$cont_var2^2  - 0.8 * data$cont_var3 * data$bin_var1 - 6 * (data$cont_var4 - 0.5)^2),
    scenario == "linear+covariate shift" & n_trial == 200 ~ arm::invlogit(1.15 + 3 * data$cont_var1 - 1.5 * data$cont_var2^2  - 0.8 * data$cont_var3 * data$bin_var1 - 6 * (data$cont_var4 - 0.5)^2),
    
    scenario == "nonlinear" & n_trial == 60  ~ arm::invlogit(-5.02 + 3 * data$cont_var1 - 1 * data$cont_var2^2  - 0.8 * data$cont_var3 * data$cont_var4 + 2 * data$bin_var1 * arm::invlogit(data$cont_var4)),
    scenario == "nonlinear" & n_trial == 200 ~ arm::invlogit(-2.38 + 3 * data$cont_var1 - 1 * data$cont_var2^2  - 0.8 * data$cont_var3 * data$cont_var4 + 2 * data$bin_var1 * arm::invlogit(data$cont_var4)),
    
    scenario == "nonlinear+covariate shift" & n_trial == 60  ~ arm::invlogit(-5.57 + 5 * data$cont_var1 - 8 * data$cont_var2^2 - 0.8 * data$cont_var3 * data$cont_var4 + 2 * data$bin_var1 * arm::invlogit(data$cont_var4)),
    scenario == "nonlinear+covariate shift" & n_trial == 200 ~ arm::invlogit(-1.06 + 5 * data$cont_var1 - 8 * data$cont_var2^2 - 0.8 * data$cont_var3 * data$cont_var4 + 2 * data$bin_var1 * arm::invlogit(data$cont_var4)),
    TRUE ~ NA
  )
  
  if (any(is.na(pi))) {
    stop("Some rows did not match any scenario/n_trial branch.")
  }
  
  
  ID = which(pi >= runif(length(pi)))
  
  colnames(data) = c(paste0("X", 1:10), "Y1", "Y0")
  data$selected = F
  data$selected[ID] = T
  data$ps = pi
  
  trials = data[ID,]
  trials$A = rbinom(dim(trials)[1], 1, prop)
  trials$selected = NULL
  trials$Y = ifelse(trials$A == 1, trials$Y1, trials$Y0)
  trials$Y0 = NULL
  trials$Y1 = NULL
  trials$ps = NULL
  
  return(list(population = data, trials = trials))
}