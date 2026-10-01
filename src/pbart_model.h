/*
 * BART: Bayesian Additive Regression Trees
 * Modifications by Jungang Zou, 2024.
 */

#ifndef ARMADILLO_H_
#define ARMADILLO_H_
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#endif

#ifndef BART_H_
#define BART_H_
#include "bart_model.h"
#endif

#ifndef DIST_H_
#define DIST_H_
#include <RcppDist.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
#endif

#ifndef RCPP_H_
#define RCPP_H_
#include <Rcpp.h>
#endif

using namespace Rcpp;


class pbart_model {
public:
  pbart_model(){};
  
  pbart_model(NumericMatrix x_train, IntegerVector y_train, bool dart = false, bool aug = false, long numcut=100L, bool usequants = false, bool cont = false, bool rm_const = false, int ntrees = 200, Nullable<double> sigmaf = R_NilValue, double k = 2.0, double power = 2, double base = 0.95, double nu = 3){
    
    this->X = x_train;
    this->Y = y_train;
    this->Y_star = as<NumericVector>(y_train) * 2.0 - 1.0;

    bart = new bart_model(X, Y_star, dart, aug, numcut, usequants, cont, rm_const, ntrees, sigmaf, k, power, base, nu);
    bart->update(1.0, 50, 50, 1, false, 10L);
    n = y_train.size();
    bart_pre = colMeans(bart->predict(X));
    
    for(int i = 0; i < n; ++i){
      if(Y[i] == 0){
        NumericVector mean_y = rtruncnorm(1, bart_pre[i], 1.0, R_NegInf, 0);
        Y_star[i] = mean_y[0];
      }else{
        NumericVector mean_y = rtruncnorm(1, bart_pre[i], 1.0, 0, R_PosInf);
        Y_star[i] = mean_y[0];
      }
    }
    
   
  };
  
  // Update Signatures
  void update(bool verbose = false, long print_every = 100L){
    bart->set_data(X, Y_star);
    bart->update(1.0, 0, 1, 1, verbose, print_every);
    bart_pre = colMeans(bart->predict(X));
    
    for(int i = 0; i < n; ++i){
      if(Y[i] == 0){
        NumericVector mean_y = rtruncnorm(1, bart_pre[i], 1.0, R_NegInf, 0);
        Y_star[i] = mean_y[0];
      }else{
        NumericVector mean_y = rtruncnorm(1, bart_pre[i], 1.0, 0, R_PosInf);
        Y_star[i] = mean_y[0];
      }
    }

  }
  
  

  
  NumericMatrix predict(NumericMatrix x_predict, bool verbose = false){
    return bart->predict(x_predict, verbose);
  };
  
  NumericMatrix predict_prob(NumericMatrix x_predict, bool verbose = false){
    NumericMatrix predicted_Y_star = bart->predict(x_predict, verbose);
    for(int i = 0; i < predicted_Y_star.rows(); ++i){
      for(int j = 0; j < predicted_Y_star.cols(); ++j){
        predicted_Y_star(i, j) = R::pnorm(predicted_Y_star(i, j), 0, 1, true, false);
      }
    }
    return predicted_Y_star; 
  };
  
  Rcpp::NumericMatrix get_varprob(){return bart->get_varprob(); }
  
  Rcpp::IntegerMatrix get_varcount(){return bart->get_varcount();}
  
  void startdart(){
    if(dart)
      bart->startdart();
  }
  
private:
 NumericMatrix X;
 NumericVector Y_star;
 IntegerVector Y;
 NumericVector bart_pre;
 bool dart;
 long n;
 
 bart_model * bart;
};




