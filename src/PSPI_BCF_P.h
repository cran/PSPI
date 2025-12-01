#ifndef ARMADILLO_H_
#define ARMADILLO_H_
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
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


#ifndef RCPP_H_
#define RCPP_H_
#include <Rcpp.h>
#endif


using namespace Rcpp;

class PSPI_BCF_P: public BARTforPSPI{
public:
  PSPI_BCF_P(NumericMatrix X_, NumericVector Y_, NumericVector Z_, NumericVector pi_, NumericMatrix X_test_, long ntrees_s) : BARTforPSPI(X_, Y_, Z_, pi_, X_test_, ntrees_s){
    Z_1 = (Z == 1.0);
    main_bart = new bart_model(cbind(X, pi), Y, 100L, false, false, false, 200);
    main_bart->update(100, 100, 1, false, 10L);
    sigma = main_bart->get_sigma();
    bart_pre = colMeans(main_bart->predict(cbind(this->X, this->pi)));
    X_Z = sliceRows(cbind(X, pi), Z_1);
    Y_Z = Y[Z_1] - bart_pre[Z_1];
    cbart = new bart_model(X_Z, Y_Z, 100L, false, false, false, ntrees_s);
    cbart->update(sigma, 50, 50, 1, false, 10L);
    Z_cbart = NumericVector(Y.length());
    this->update_Z_cbart();
  };
  
  void update_Z_cbart(){
    cbart_pre = colMeans(cbart->predict(X_Z));
    Z_cbart[Z_1] = cbart_pre;
  }
  
  void update(bool verbose = false) override{
    main_bart->set_data(cbind(X, pi), Y - Z_cbart);
    main_bart->update(sigma, w, 1, 1, 1, false, 10L);
    
    bart_pre = colMeans(main_bart->predict(cbind(X, pi)));
    Y_Z = Y[Z_1] - bart_pre[Z_1];
    cbart->set_data(X_Z, Y_Z);
    NumericVector w_Z = w[Z_1];
    cbart->update(sigma, w_Z, 1, 1, 1, false, 10L);
    this->update_Z_cbart();
    double rss = sum(pow(Y - Z_cbart - bart_pre, 2));
    sigma = main_bart->get_invchi(n, rss);
  };
  
  List predict(NumericVector pi_test) override{
    long N = X_test.nrow();
    NumericVector outcome_0 = colMeans(main_bart->predict(cbind(X_test, pi_test)));// - main_bart_mean;
    NumericVector outcome_1 = outcome_0 + colMeans(cbart->predict(cbind(X_test, pi_test)));
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
      // Named("Y_hat") = Y
    );
  };
  
private:
  double sigma;
  NumericVector bart_pre;
  bart_model * cbart;
  LogicalVector Z_1;
  NumericMatrix X_Z;
  NumericVector Y_Z;
  NumericVector Z_cbart;
  NumericVector cbart_pre;
};
