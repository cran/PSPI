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


class PSPI_FullBART: public BARTforPSPI{
public:
  PSPI_FullBART(NumericMatrix X_, NumericVector Y_, NumericVector Z_, NumericVector pi_, NumericMatrix X_test_, long ntrees_s = 200) : BARTforPSPI(X_, Y_, Z_, pi_, X_test_, ntrees_s){
    Z_1 = (Z == 1.0);
    //main_bart = new bart_model(sliceRows(cbind(X, pi), !Z_1), Y[!Z_1], 100L, false, false, false, 200);
    main_bart = new bart_model(cbind(X, pi), Y, 100L, false, false, false, 200);
    main_bart->update(50, 50, 1, false, 10L);
    sigma = main_bart->get_sigma();
    bart_pre = colMeans(main_bart->predict(cbind(this->X, this->pi)));
    
    Z_1 = (Z == 1.0);
    X_Z = sliceRows(X, Z_1);
    Y_Z = Y[Z_1] - bart_pre[Z_1];
    pi_Z = NumericMatrix(Y_Z.length(), 1, as<NumericVector>(pi[Z_1]).begin());
    cbart_pi = new bart_model(pi_Z, Y_Z, 100L, false, false, false, ntrees_s);
    cbart_pi->update(sigma, 50, 50, 1, false, 10L);
    cbart_pi_pre = colMeans(cbart_pi->predict(pi_Z));

    cbart = new bart_model(X_Z, Y_Z - cbart_pi_pre, 100L, false, false, false, ntrees_s);
    cbart->update(sigma, 50, 50, 1, false, 10L);
    cbart_pre = colMeans(cbart->predict(X_Z));
    cbart_pre_mean = mean(colMeans(cbart->predict(X)));
    cbart_pre = cbart_pre - cbart_pre_mean;
    
    Z_cbart = NumericVector(Y.length());
    this->update_Z_cbart();
  };
  
  void update_Z_cbart(){
    NumericVector Z_cbart_Z_1 = cbart_pre + cbart_pi_pre;
    Z_cbart[Z_1] = Z_cbart_Z_1;
  }
  
  void update(bool verbose = false) override{
    
    main_bart->set_data(cbind(X, pi), Y - Z_cbart);
    
    main_bart->update(sigma, w, 1, 1, 1, false, 10L);
    bart_pre = colMeans(main_bart->predict(cbind(X, pi)));

    
    Y_Z = Y[Z_1] - bart_pre[Z_1];
    NumericVector Y_Cb = Y_Z - cbart_pi_pre;
    NumericVector w_Z = w[Z_1];

    cbart->set_data(X_Z, Y_Z - cbart_pi_pre);
    cbart->update(sigma, w_Z, 1, 1, 1, false, 10L);
    cbart_pre = colMeans(cbart->predict(X_Z));
    cbart_pre_mean = mean(colMeans(cbart->predict(X)));;
    cbart_pre = cbart_pre - cbart_pre_mean;

    NumericVector y_te = Y_Z - cbart_pre;
    cbart_pi->set_data(pi_Z, Y_Z - cbart_pre);
    cbart_pi->update(sigma, w_Z, 1, 1, 1, false, 10L);
    cbart_pi_pre = colMeans(cbart_pi->predict(pi_Z));
    this->update_Z_cbart();
    double rss = sum(pow(Y - Z_cbart - bart_pre, 2));
    sigma = main_bart->get_invchi(n, rss);
  };
  
  List predict(NumericVector pi_test) override{
    long N = X_test.nrow();
    cbart_pop = colMeans(cbart->predict(X_test)) - cbart_pre_mean;
    NumericMatrix pi_test_Z = NumericMatrix(N, 1, pi_test.begin());
    NumericVector predict_s = colMeans(cbart_pi->predict(pi_test_Z));
    NumericVector outcome_0 = colMeans(main_bart->predict(cbind(X_test, pi_test)));
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
      // Named("cbart_pi_pre") = cbart_pi_pre,
      // Named("cbart_pre_mean") = cbart_pre_mean
    );
  };
  
private:
  double sigma;
  NumericVector bart_pre;
  bart_model * cbart;
  bart_model * cbart_pi;
  
  
  LogicalVector Z_1;
  NumericMatrix X_Z;
  NumericVector Y_Z;
  NumericMatrix pi_Z;
  NumericVector Z_cbart;
  NumericVector cbart_pre;
  double cbart_pre_mean;
  NumericVector cbart_pi_pre;
  NumericVector cbart_pop;
};
