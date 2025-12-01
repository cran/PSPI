#ifndef ARMADILLO_H_
#define ARMADILLO_H_
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#endif

#ifndef CBART_H_
#define CBART_H_
#include "BARTforPSPI.h"
#endif


#include "BCF.h"
#include "PSPI_BCF_P.h"
#include "PSPI_FullBART.h"
#include "PSPI_SplineBART.h"
#include "PSPI_DSplineBART.h"
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>

#ifndef RCPP_H_
#define RCPP_H_
#include <Rcpp.h>
#endif


using namespace Rcpp;

// [[Rcpp::export]]
List MCMC_PSPI_generalizability(NumericMatrix X, NumericVector Y, NumericVector Z, NumericVector pi, NumericMatrix X_test, NumericVector pi_test, int model, long nburn, long npost, long n_knots_main, long n_knots_inter, long order_main = 3, long order_inter = 3, int ntrees_s = 200, bool verbose = false){
  BARTforPSPI * cmodel;
  
  NumericMatrix X_ = clone(X);
  NumericVector Y_ = clone(Y);
  NumericVector Z_ = clone(Z);
  NumericVector pi_ = clone(pi);
  
  NumericMatrix X_test_ = clone(X_test);
  NumericVector pi_test_ = clone(pi_test);

  //Rcout << 123 << std::endl;
  //pi_ = log(pi_ / (1 - pi_)); //1 / (1 + exp(-1 * pi_));
  //pi_test_ = log(pi_test_ / (1 - pi_test_)); //1 / (1 + exp(-1 * pi_test_));
  switch (model){
  case 1:
    //cmodel = new vanillaBART(X_, Y_, Z_, pi_, X_test_, binary, logistic, ntrees_s);
    break;
  case 2:
    cmodel = new BCF(X_, Y_, Z_, pi_, X_test_, ntrees_s);
    break;
  case 3:
    cmodel = new PSPI_BCF_P(X_, Y_, Z_, pi_, X_test_, ntrees_s);
    break;
  case 4:
    cmodel = new PSPI_FullBART(X_, Y_, Z_, pi_, X_test_, ntrees_s);
    break;
  case 5:
    cmodel = new PSPI_SplineBART(X_, Y_, Z_, pi_, X_test_, n_knots_inter, order_inter, ntrees_s);
    break;
  case 6:
    cmodel = new PSPI_DSplineBART(X_, Y_, Z_, pi_, X_test_, n_knots_main, n_knots_inter, order_main, order_inter, ntrees_s);
    break;
  }

  long n = X_test_.nrow();
  NumericVector post_sigma(npost);
  NumericMatrix post_outcome1(npost, pi_test_.length());
  NumericMatrix post_outcome0(npost, pi_test_.length());
  NumericMatrix post_te(npost, pi_test_.length());
  
  List spline_main = List::create();
  List spline_inter = List::create();
  //NumericMatrix predict_s(npost, pi_test_.length());
  //NumericMatrix predict_h(npost, pi_test_.length());
  //NumericMatrix cbart_pop(npost, pi_test_.length());
  //Rcout << 123 << std::endl;
  Progress p(nburn + npost, !verbose);
  for(int i = 0 ; i < nburn + npost; ++i){
    if(Progress::check_abort())
      return -1.0;
    p.increment();
    if(verbose)
      Rcout << i << " " << nburn + npost << std::endl;
    cmodel->update(verbose);
    
    if(i >= nburn){
      List posterior = cmodel->get_posterior();
      List predict_outcome = cmodel->predict(pi_test_);
      post_outcome0(i - nburn, _) = (as<NumericVector>(predict_outcome["outcome_0"]));
      post_outcome1(i - nburn, _) = (as<NumericVector>(predict_outcome["outcome_1"]));
      post_te(i - nburn, _) = (post_outcome1(i - nburn, _) - post_outcome0(i - nburn, _));
      post_sigma[i - nburn] = posterior["sigma"];

      // if(model == 4 || model == 5 || model == 6){
      //   
      // }
      // if(model == 5 || model == 6 || model == 7){
      //   predict_s(i - nburn, _) = as<NumericVector>(predict_outcome["predict_s"]);
      //   cbart_pop(i - nburn, _) = as<NumericVector>(predict_outcome["cbart_pop"]);
      //   cbart_pre(i - nburn, _) = as<NumericVector>(posterior["cbart_pre"]);
      //   
      //   ns_beta(i - nburn, _) = as<NumericVector>(posterior["ns_beta"]);
      //   
      //   //post_beta(i - nburn, 0) = as<NumericVector>(posterior["ns_beta"])[0];
      //   //post_beta(i - nburn, 1) = as<NumericVector>(posterior["ns_beta"])[1];
      //   //post_gamma(i - nburn) = as<double>(posterior["gamma"]);
      // }
      // if(model == 6){
      //   predict_h(i - nburn, _) =  as<NumericVector>(predict_outcome["predict_h"]);
      //   //post_beta_main(i - nburn, _) = as<NumericVector>(posterior["ns_beta_main"]);
      // }
    }
  }
  return List::create(Named("post_outcome1") = post_outcome1, Named("post_outcome0") = post_outcome0, Named("post_te") = post_te, Named("post_sigma") = post_sigma);
}
