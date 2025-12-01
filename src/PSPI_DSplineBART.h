#ifndef ARMADILLO_H_
#define ARMADILLO_H_
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#endif

#ifndef NS_H_
#define NS_H_
#include "NS_ridge.h"
#endif


#ifndef CBART_H_
#define CBART_H_
#include "BARTforPSPI.h"
#endif

#ifndef PG_H_
#define PG_H_
#include <pg.h>
// [[Rcpp::depends(RcppArmadillo, pg)]]
#endif

#ifndef RCPPDIST_H_
#define RCPPDIST_H_
#include <RcppDist.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
#endif


#ifndef RCPP_H_
#define RCPP_H_
#include <Rcpp.h>
#endif


using namespace Rcpp;


class PSPI_DSplineBART: public BARTforPSPI{
public:
  PSPI_DSplineBART(NumericMatrix X_, NumericVector Y_, NumericVector Z_, NumericVector pi_, NumericMatrix X_test_, long n_knots_main, long n_knots_inter, long order_main = 3, long order_inter = 3, long ntrees_s = 200) : BARTforPSPI(X_, Y_, Z_, pi_, X_test_, ntrees_s){
    Z_1 = (Z == 1.0);
    
    main_bart = new bart_model(X, Y, 100L, false, false, false, 200);
    main_bart->update(50, 50, 1, false, 10L);
    sigma = main_bart->get_sigma();
    
    bart_pre = colMeans(main_bart->predict(this->X));
    bart_pre_mean = mean(bart_pre);
    bart_pre = bart_pre - bart_pre_mean;
    
    main_bs = new NS_R(as<NumericVector>(pi), Y - bart_pre, n_knots_main, sigma, order_main);
    //main_bs = new NS_R(as<NumericVector>(pi), Y - bart_pre, std::floor(pow(Z.length(), 1.0/3.0)), sigma, 3);
    
    main_bs->update(sigma);
    main_pi_pre = main_bs->get_ns_outcome();
    
    
    Z_1 = (Z == 1.0);
    X_Z = sliceRows(X, Z_1);
    Y_Z = as<NumericVector>(Y[Z_1]) - as<NumericVector>(bart_pre[Z_1]) - as<NumericVector>(main_pi_pre[Z_1]);
    
    bs = new NS_R(as<NumericVector>(pi[Z_1]), Y_Z, n_knots_inter, sigma, order_inter);
    //bs = new NS_R(as<NumericVector>(pi[Z_1]), Y_Z, std::floor(pow(sum(Z_1), 1.0/3.0)), sigma, 3);
    bs->update(sigma);
    clm_pi_pre = bs->get_ns_outcome();
    
    
    cbart = new bart_model(X_Z, Y_Z - clm_pi_pre, 100L, false, false, false, ntrees_s);
    cbart->update(sigma, 50, 50, 1, false, 10L);
    cbart_pre = colMeans(cbart->predict(X_Z));
    cbart_pre_mean = mean(colMeans(cbart->predict(X)));
    cbart_pre = cbart_pre - cbart_pre_mean;

    
    Z_cbart = NumericVector(Y.length());
    this->update_Z_cbart();
  };
  
  void update_Z_cbart(){
    NumericVector Z_cbart_Z_1 = cbart_pre + clm_pi_pre;
    Z_cbart[Z_1] = Z_cbart_Z_1;
  }
  
  void update(bool verbose = false) override{
    main_bart->set_data(X, Y - Z_cbart - main_pi_pre);
    List main_train_result = main_bart->update(sigma, w, 1, 1, 1, false, 10L);
    bart_pre = as<NumericVector>(main_train_result["yhat.train.mean"]);
    bart_pre_mean = mean(bart_pre);
    bart_pre = bart_pre - bart_pre_mean;
    
    
    main_bs->set_Y(Y - Z_cbart - bart_pre);
    main_bs->update(sigma * w);
    main_pi_pre = main_bs->get_ns_outcome();
    
    
    Y_Z = as<NumericVector>(Y[Z_1]) - as<NumericVector>(bart_pre[Z_1]) - as<NumericVector>(main_pi_pre[Z_1]);
    NumericVector Y_Cb = Y_Z - clm_pi_pre;
    NumericVector w_Z = w[Z_1];
    
    cbart->set_data(X_Z, Y_Z - clm_pi_pre);
    List cbart_train_result = cbart->update(sigma, w_Z, 1, 1, 1, false, 10L);
    cbart_pre = as<NumericVector>(cbart_train_result["yhat.train.mean"]);
    cbart_pre_mean = mean(colMeans(cbart->predict(X)));
    cbart_pre = cbart_pre - cbart_pre_mean;
    
    
    NumericVector y_te = Y_Z - cbart_pre;
    
    bs->set_Y(y_te);
    bs->update(sigma * w_Z);
    clm_pi_pre = bs->get_ns_outcome();
    
    this->update_Z_cbart();
    double rss = sum(pow(Y - Z_cbart - bart_pre - main_pi_pre, 2));
    sigma = main_bart->get_invchi(n, rss);
  };
  
  List predict(NumericVector pi_test) override{
    long N = X_test.nrow();
    NumericVector predict_s = bs->predict(pi_test);
    NumericVector predict_h = main_bs->predict(pi_test);
    
    bart_pop = colMeans(main_bart->predict(X_test)) - bart_pre_mean;
    cbart_pop = colMeans(cbart->predict(X_test)) - cbart_pre_mean;
    
    NumericVector outcome_0 = bart_pop + predict_h;
    NumericVector outcome_1 = outcome_0 + cbart_pop + predict_s;
    
    for(int i = 0; i < N; ++i){
      outcome_1[i] = outcome_1[i] + R::rnorm(0, sigma);
      outcome_0[i] = outcome_0[i] + R::rnorm(0, sigma);
    }
    return List::create(Named("outcome_1") = outcome_1, Named("outcome_0") = outcome_0);
  };
  
  List get_posterior() override{
    return List::create(
      Named("sigma") = sigma//,
      // Named("bart_pre") = bart_pre,
      // Named("cbart_pre") = cbart_pre,
      // Named("Z_cbart") = Z_cbart,
      // Named("clm_pi_pre") = clm_pi_pre,
      // Named("cbart_pre_mean") = cbart_pre_mean,
      // Named("ns_beta") = bs->get_theta(),
      // Named("ns_beta_main") = main_bs->get_theta(),
      // Named("gamma") = bs->get_gamma(),
    );
  };
  
  
private:
  
  NumericVector main_pi_pre;
  double sigma;
  
  NS * main_bs;
  NS * bs;
  
  bart_model * cbart;
  
  LogicalVector Z_1;
  NumericMatrix X_Z;
  NumericVector Y_Z;
  NumericMatrix pi_Z;
  NumericMatrix pi_main;
  NumericVector Z_cbart;
  NumericVector clm_pi_pre;
  
  double cbart_pre_mean;
  double bart_pre_mean;
  NumericVector cbart_pre;
  NumericVector cbart_pop;
  NumericVector bart_pop;
  NumericVector bart_pre;
  
};
