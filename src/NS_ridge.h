#ifndef ARMADILLO_H_
#define ARMADILLO_H_
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#endif

#ifndef RCPP_H_
#define RCPP_H_
#include <Rcpp.h>
#endif

#ifndef NS_H_
#define NS_H_
#include "NS.h"
#endif

#ifndef RCPPDIST_H_
#define RCPPDIST_H_
#include <RcppDist.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
#endif

#include <algorithm>

using namespace Rcpp;
using namespace arma;

class NS_R: public NS {
public:
  NS_R() {}
  
  NS_R(NumericVector X,
       NumericVector y,
       long K,
       double sigma,
       int order = 3,
       double alpha_0 = 1.0,
       double beta_0  = 1.0,
       double alpha_k = 1.0,
       double beta_k  = 1.0,
       bool local = false,
       bool intercept = false)
    : NS(X, y, K, sigma, order, intercept)
  {
    this->alpha_0 = alpha_0;
    this->beta_0  = beta_0;
    this->alpha_k = alpha_k;
    this->beta_k  = beta_k;
    this->local   = local;
    
    const arma::uword d = ns_basis.n_cols;
    
    if(d > 0){
      gamma = 1.0;
      
      if(local){
        lambda_k.set_size(d);
        for(arma::uword k = 0; k < d; ++k){
          lambda_k[k] = std::sqrt(rinvgamma(this->alpha_k, this->beta_k));
        }
      }
    }
    
    refresh_outcome_cache();
  }
  
  // Joint Gibbs update of (lr_coefficient, ns_coefficient).
  // Samples the full coefficient vector from its joint conditional posterior
  // in one step, eliminating slow mixing along ridge-like posteriors that
  // arise when the linear and NS basis columns are posterior-correlated.
  //
  // Model:  y = [lr_basis | ns_basis] * beta + N(0, sigma^2 I)
  // Prior:  flat on lr_coefficient (dims 0..d_lr-1)
  //         N(0, gamma^2) on ns_coefficient (or N(0, (gamma*lambda_k)^2) if local)
  //
  // Equivalent stationary distribution to the old blocked Gibbs, but
  // dramatically faster mixing when (lr, ns) coefficients are correlated.
  void update() override {
    const arma::uword d_lr = lr_basis.n_cols;
    const arma::uword d_ns = ns_basis.n_cols;

    if(d_ns == 0){
      // No NS basis: fall back to flat-prior linear regression on lr_basis.
      lr_coefficient = update_beta(lr_basis, this->y, this->sigma);
      ns_outcome = lr_basis * lr_coefficient;
      return;
    }

    // Stacked design matrix [lr_basis | ns_basis]
    arma::mat X = arma::join_rows(lr_basis, ns_basis);

    // Posterior precision: (1/sigma^2) X'X + prior precision.
    // Prior precision is 0 for lr dims (flat prior), 1/gamma^2 (or local) for ns dims.
    const double inv_sig2 = 1.0 / (this->sigma * this->sigma);
    arma::mat Prec = inv_sig2 * (X.t() * X);

    if(local && lambda_k.n_elem == d_ns){
      for(arma::uword k = 0; k < d_ns; ++k){
        const double prec_k = 1.0 / std::pow(gamma * lambda_k[k], 2.0);
        Prec(d_lr + k, d_lr + k) += prec_k;
      }
    } else {
      const double prec_ns = 1.0 / (gamma * gamma);
      for(arma::uword k = 0; k < d_ns; ++k){
        Prec(d_lr + k, d_lr + k) += prec_ns;
      }
    }

    // Joint draw from multivariate normal
    arma::mat Var = arma::inv_sympd(Prec);
    arma::vec Mean = Var * (inv_sig2 * (X.t() * this->y));
    arma::vec beta_full = arma::vectorise(rmvnorm(1, Mean, Var));

    lr_coefficient = beta_full.head(d_lr);
    ns_coefficient = beta_full.tail(d_ns);

    // Update shrinkage hyperparameters (gamma, lambda_k)
    update_shrinkage();

    // Cache ns_outcome = X * beta (stored for downstream access)
    ns_outcome = X * beta_full;
  }

  void update(double sigma) override {
    this->sigma = sigma;
    update();
  }

  // Heteroskedastic variant: y_i ~ N(X_i beta, sigma_i^2)
  void update(NumericVector sigma) override {
    const arma::uword d_lr = lr_basis.n_cols;
    const arma::uword d_ns = ns_basis.n_cols;

    arma::vec sigma_vec = as<arma::vec>(sigma);
    arma::vec w = 1.0 / (sigma_vec % sigma_vec);

    if(d_ns == 0){
      // Heteroskedastic flat-prior regression on lr_basis only.
      arma::mat Prec = lr_basis.t() * arma::diagmat(w) * lr_basis;
      arma::mat Var  = arma::inv_sympd(Prec);
      arma::vec Mean = Var * (lr_basis.t() * (w % this->y));
      lr_coefficient = arma::vectorise(rmvnorm(1, Mean, Var));
      ns_outcome = lr_basis * lr_coefficient;
      return;
    }

    arma::mat X = arma::join_rows(lr_basis, ns_basis);
    arma::mat Prec = X.t() * arma::diagmat(w) * X;

    if(local && lambda_k.n_elem == d_ns){
      for(arma::uword k = 0; k < d_ns; ++k){
        const double prec_k = 1.0 / std::pow(gamma * lambda_k[k], 2.0);
        Prec(d_lr + k, d_lr + k) += prec_k;
      }
    } else {
      const double prec_ns = 1.0 / (gamma * gamma);
      for(arma::uword k = 0; k < d_ns; ++k){
        Prec(d_lr + k, d_lr + k) += prec_ns;
      }
    }

    arma::mat Var = arma::inv_sympd(Prec);
    arma::vec Mean = Var * (X.t() * (w % this->y));
    arma::vec beta_full = arma::vectorise(rmvnorm(1, Mean, Var));

    lr_coefficient = beta_full.head(d_lr);
    ns_coefficient = beta_full.tail(d_ns);

    update_shrinkage();

    ns_outcome = X * beta_full;
  }
  
protected:
  void refresh_outcome_cache(){
    const arma::uword d = ns_basis.n_cols;
    arma::vec lr_fit = lr_basis * lr_coefficient;
    
    if(d > 0 && ns_coefficient.n_elem == d){
      ns_outcome = lr_fit + ns_basis * ns_coefficient;
    }else{
      ns_outcome = lr_fit;
    }
  }
  
  void sample_spline_homoskedastic(double sigma, const arma::vec& lr_fit){
    const arma::uword d = ns_basis.n_cols;
    if(d == 0) return;
    
    arma::vec r = this->y - lr_fit;
    
    arma::mat XtX = ns_basis.t() * ns_basis;
    arma::vec Xtr = ns_basis.t() * r;
    
    arma::mat Prec = (1.0 / (sigma * sigma)) * XtX;
    
    if(local){
      arma::vec prior_prec = 1.0 / arma::square(gamma * lambda_k);
      Prec.diag() += prior_prec;
    }else{
      Prec.diag() += (1.0 / (gamma * gamma));
    }
    
    arma::mat Var = arma::inv_sympd(Prec);
    arma::vec Mean = Var * ((1.0 / (sigma * sigma)) * Xtr);
    
    ns_coefficient = arma::vectorise(rmvnorm(1, Mean, Var));
  }
  
  void sample_spline_heteroskedastic(const arma::vec& lr_fit){
    const arma::uword d = ns_basis.n_cols;
    if(d == 0) return;
    
    arma::vec r = this->y - lr_fit;
    
    arma::mat XtWX = ns_basis.t() * inv_Sigma * ns_basis;
    arma::vec XtWr = ns_basis.t() * inv_Sigma * r;
    
    arma::mat Prec = XtWX;
    
    if(local){
      arma::vec prior_prec = 1.0 / arma::square(gamma * lambda_k);
      Prec.diag() += prior_prec;
    }else{
      Prec.diag() += (1.0 / (gamma * gamma));
    }
    
    arma::mat Var = arma::inv_sympd(Prec);
    arma::vec Mean = Var * XtWr;
    
    ns_coefficient = arma::vectorise(rmvnorm(1, Mean, Var));
  }
  
  void update_shrinkage(){
    const arma::uword d = ns_basis.n_cols;
    if(d == 0) return;
    
    if(local){
      for(arma::uword k = 0; k < d; ++k){
        double shape = alpha_k + 0.5;
        double scale = beta_k + 0.5 * std::pow(ns_coefficient[k] / gamma, 2.0);
        lambda_k[k] = std::sqrt(rinvgamma(shape, scale));
      }
      
      arma::vec z = ns_coefficient / lambda_k;
      double shape0 = alpha_0 + 0.5 * (double)d;
      double scale0 = beta_0  + 0.5 * arma::dot(z, z);
      gamma = std::sqrt(rinvgamma(shape0, scale0));
    }else{
      double shape0 = alpha_0 + 0.5 * (double)d;
      double scale0 = beta_0  + 0.5 * arma::dot(ns_coefficient, ns_coefficient);
      gamma = std::sqrt(rinvgamma(shape0, scale0));
    }
  }
  
protected:
  double alpha_0 = 1.0;
  double beta_0  = 1.0;
  double alpha_k = 1.0;
  double beta_k  = 1.0;
  
  arma::vec lambda_k;
  bool local = false;
};

