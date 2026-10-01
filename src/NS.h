#ifndef ARMADILLO_H_
#define ARMADILLO_H_
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#endif

#ifndef RCPP_H_
#define RCPP_H_
#include <Rcpp.h>
#endif

#ifndef NS_BASIS_H_
#define NS_BASIS_H_
#include "NS_basis.h"
#endif

#ifndef RCPPDIST_H_
#define RCPPDIST_H_
#include <RcppDist.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
#endif

#include <algorithm>
#include <memory>

using namespace Rcpp;
using namespace arma;

class NS{
public:
  NS() {}
  
  NS(NumericVector X_,
     NumericVector y_,
     long K_,
     double sigma_,
     int order = 3,
     bool intercept = false){
    if(K_ < 2) stop("K must be >= 2.");
    
    X = X_;
    y = as<arma::vec>(y_);
    K = K_;
    n = X.size();
    sigma = sigma_;
    
    basis = std::unique_ptr<NS_basis>(new NS_basis(X, K, order, intercept));
    has_intercept = intercept;
    
    complete_basis = as<arma::mat>(basis->get_basis());
    partition_basis(complete_basis, lr_basis, ns_basis);
    
    arma::vec theta_init = update_beta_mean(complete_basis, y);
    lr_coefficient = theta_init.head(lr_basis.n_cols);
    if(ns_basis.n_cols > 0){
      ns_coefficient = theta_init.tail(ns_basis.n_cols);
    }else{
      ns_coefficient.reset();
    }
    
    update_outcome();
  }
  
  virtual ~NS() = default;
  
  NS(const NS&) = delete;
  NS& operator=(const NS&) = delete;
  
  virtual void update() = 0;
  virtual void update(double sigma) = 0;
  virtual void update(NumericVector sigma) = 0;
  
  void set_Y(NumericVector y_){
    y = as<arma::vec>(y_);
  }
  
  void set_X(NumericVector X_){
    X = X_;
    complete_basis = as<arma::mat>(basis->predict(X));
    partition_basis(complete_basis, lr_basis, ns_basis);
    update_outcome();
  }
  
  void set_basis(NumericMatrix complete_basis_, NumericMatrix lr_basis_, NumericMatrix ns_basis_){
    complete_basis = as<arma::mat>(complete_basis_);
    lr_basis = as<arma::mat>(lr_basis_);
    ns_basis = as<arma::mat>(ns_basis_);
    update_outcome();
  }
  
  void set_basis_replace(NumericVector x){
    complete_basis = as<arma::mat>(basis->predict(x));
    partition_basis(complete_basis, lr_basis, ns_basis);
    update_outcome();
  }
  
  NumericVector predict(NumericVector X_test){
    complete_basis_test = as<arma::mat>(basis->predict(X_test));
    partition_basis(complete_basis_test, lr_basis_test, ns_basis_test);
    
    arma::vec out = lr_basis_test * lr_coefficient;
    if(ns_basis_test.n_cols > 0 && ns_coefficient.n_elem > 0){
      out += ns_basis_test * ns_coefficient;
    }
    return wrap(out);
  }
  
  NumericMatrix get_basis() const { return wrap(complete_basis); }
  
  NumericMatrix get_basis_test() const { return wrap(complete_basis_test); }
  
  NumericMatrix get_lr_part() const { return wrap(lr_basis); }
  
  NumericMatrix get_ns_part() const { return wrap(ns_basis); }
  
  NumericVector get_boundary_knots() const { return basis->get_boundary_knots(); }
  
  NumericVector get_internal_knots() const { return basis->get_internal_knots(); }
  
  NumericVector get_knots() const { return basis->get_knots(); }
  
  NumericVector get_theta() const{
    arma::vec theta = lr_coefficient;
    if(ns_coefficient.n_elem > 0) theta = arma::join_cols(theta, ns_coefficient);
    return wrap(theta);
  }
  
  NumericVector get_ns_outcome() const { return wrap(ns_outcome); }
  
  double get_gamma() const { return gamma; }
  
  arma::vec update_beta_mean(const arma::mat& Xmat, const arma::vec& Yvec){
    arma::mat XtX = Xmat.t() * Xmat;
    arma::vec XtY = Xmat.t() * Yvec;
    
    arma::vec beta;
    bool ok = arma::solve(beta, XtX, XtY,
                          arma::solve_opts::likely_sympd + arma::solve_opts::refine);
    if(!ok) beta = arma::pinv(XtX) * XtY;
    return beta;
  }
  
  arma::vec update_beta(const arma::mat& Xmat, const arma::vec& Yvec, double sigma_){
    arma::vec beta_mean = update_beta_mean(Xmat, Yvec);
    
    arma::mat XtX = Xmat.t() * Xmat;
    arma::mat XtX_inv;
    bool ok = arma::inv_sympd(XtX_inv, XtX);
    if(!ok) XtX_inv = arma::pinv(XtX);
    
    arma::mat beta_var = XtX_inv * (sigma_ * sigma_);
    return arma::vectorise(rmvnorm(1, beta_mean, beta_var));
  }
  
  arma::vec update_beta(const arma::mat& Xmat, const arma::vec& Yvec, const arma::vec& sigma_vec){
    if(sigma_vec.n_elem != Yvec.n_elem) stop("sigma length must match y length.");
    
    arma::vec w = 1.0 / (sigma_vec % sigma_vec);
    arma::mat W = arma::diagmat(w);
    
    arma::mat XtWX = Xmat.t() * W * Xmat;
    
    arma::mat beta_var;
    bool ok = arma::inv_sympd(beta_var, XtWX);
    if(!ok) beta_var = arma::pinv(XtWX);
    
    arma::vec beta_mean = beta_var * (Xmat.t() * W * Yvec);
    return arma::vectorise(rmvnorm(1, beta_mean, beta_var));
  }
  
  List project_residual_basis(NumericVector y_in){
    arma::vec y_tr = as<arma::vec>(y_in);
    if(complete_basis.n_rows != y_tr.n_rows) stop("complete_basis rows must match length(y).");
    
    arma::mat H = design_no_intercept(complete_basis);
    if(H.n_cols == 0) stop("No columns available after dropping intercept.");
    
    arma::mat XtX = H.t() * H;
    arma::vec XtY = H.t() * y_tr;
    
    arma::vec beta;
    bool ok = arma::solve(beta, XtX, XtY,
                          arma::solve_opts::likely_sympd + arma::solve_opts::refine);
    if(!ok) beta = arma::pinv(XtX) * XtY;
    
    arma::vec fitted = H * beta;
    arma::vec resid = y_tr - fitted;
    
    return List::create(
      _["beta"] = beta,
      _["d_used"] = (int)H.n_cols,
      _["resid_tr"] = wrap(resid)
    );
  }
  
  List project_residual_basis_test(NumericVector y_in){
    arma::vec y_tr = as<arma::vec>(y_in);
    if(complete_basis_test.n_rows != y_tr.n_rows) stop("complete_basis_test rows must match length(y).");
    
    arma::mat H = design_no_intercept(complete_basis_test);
    if(H.n_cols == 0) stop("No columns available after dropping intercept.");
    
    arma::mat XtX = H.t() * H;
    arma::vec XtY = H.t() * y_tr;
    
    arma::vec beta;
    bool ok = arma::solve(beta, XtX, XtY,
                          arma::solve_opts::likely_sympd + arma::solve_opts::refine);
    if(!ok) beta = arma::pinv(XtX) * XtY;
    
    arma::vec fitted = H * beta;
    arma::vec resid = y_tr - fitted;
    
    return List::create(
      _["beta"] = beta,
      _["d_used"] = (int)H.n_cols,
      _["resid_tr"] = wrap(resid)
    );
  }
  
  NumericVector predict_project_residual_basis(List project, NumericVector y_pop){
    arma::vec beta = as<arma::vec>(project["beta"]);
    int d_used = as<int>(project["d_used"]);
    
    arma::vec y_te = as<arma::vec>(y_pop);
    if(complete_basis_test.n_rows != y_te.n_rows) stop("complete_basis_test rows must match length(y_pop).");
    
    arma::mat H = design_no_intercept(complete_basis_test);
    if((int)H.n_cols != d_used) stop("Basis column count mismatch for projection.");
    
    arma::vec fitted = H * beta;
    arma::vec resid = y_te - fitted;
    return wrap(resid);
  }
  
  double update_slope(const arma::vec& u, const arma::vec& yvec, double sigma_){
    double w = 1.0 / (sigma_ * sigma_);
    double XtWX = w * arma::dot(u, u);
    double beta_var = 1.0 / XtWX;
    double beta_mean = beta_var * w * arma::dot(u, yvec);
    return beta_mean + std::sqrt(beta_var) * R::rnorm(0.0, 1.0);
  }
  
  void update_outcome(){
    ns_outcome = lr_basis * lr_coefficient;
    if(ns_basis.n_cols > 0 && ns_coefficient.n_elem > 0){
      ns_outcome += ns_basis * ns_coefficient;
    }
  }
  
  double rinvgamma(double a, double b){
    double s = R::rgamma(a, 1.0 / b);
    return 1.0 / s;
  }
  
  double qinvgamma(double p, double a, double b){
    double q = R::qgamma(1.0 - p, a, 1.0 / b, true, false);
    return 1.0 / q;
  }
  
protected:
  void partition_basis(const arma::mat& B, arma::mat& B_lr, arma::mat& B_ns){
    if(B.n_cols == 0) stop("Basis matrix has zero columns.");
    
    arma::uword lr_cols = has_intercept ? 2u : 1u;
    if(B.n_cols < lr_cols) stop("Basis has fewer columns than required for linear part.");
    
    B_lr = B.cols(0, lr_cols - 1);
    
    if(B.n_cols > lr_cols){
      B_ns = B.cols(lr_cols, B.n_cols - 1);
    }else{
      B_ns.set_size(B.n_rows, 0);
    }
  }
  
  arma::mat design_no_intercept(const arma::mat& B) const{
    if(!has_intercept) return B;
    if(B.n_cols <= 1) stop("Cannot drop intercept: basis has <= 1 column.");
    return B.cols(1, B.n_cols - 1);
  }
  
protected:
  std::unique_ptr<NS_basis> basis;
  
  bool has_intercept = false;
  
  long K = 0;
  long n = 0;
  double sigma = 1.0;
  double gamma = NA_REAL;
  
  arma::mat Sigma;
  arma::mat inv_Sigma;
  
  arma::vec lr_coefficient;
  arma::vec ns_coefficient;
  
  NumericVector X;
  arma::vec y;
  
  arma::mat complete_basis;
  arma::mat ns_basis;
  arma::mat lr_basis;
  
  arma::mat complete_basis_test;
  arma::mat ns_basis_test;
  arma::mat lr_basis_test;
  
  arma::vec ns_outcome;
};


// 
// // [[Rcpp::export]]
// List test_NS(NumericVector X, NumericVector X_test, NumericVector y, long K){
//   NS * a = new NS(X, y, K, 1);
//   Rcout << a->get_theta() << std::endl;
//   for(int i = 1; i < 10000; ++i)
//     a->update();
//   Rcout << a->get_theta() << std::endl;
//   return List::create(Named("ns_predict") = a->predict(X_test), Named("eta") = a->get_eta(), Named("gamma") = a->get_gamma(), Named("theta") = a->get_theta());
//   //return List::create(Named("ns_outcome") = wrap(a->ns_outcome), Named("lr_outcome") = wrap(a->lr_outcome), Named("ns_part_outcome") = wrap(a->ns_part_outcome), Named("eta") = a->get_eta(), Named("gamma") = a->get_gamma(), Named("theta") = a->get_theta(), Named("boundary_knots") = a->get_boundary_knots(), Named("knots") = a->get_knots(), Named("internal_knots") = a->get_internal_knots(), Named("ns_part") = a->get_ns_part(), Named("lr_part") = a->get_lr_part(), Named("basis") = a->get_basis());
// };
