#ifndef ARMADILLO_H_
#define ARMADILLO_H_
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#endif

#ifndef CBART_H_
#define CBART_H_
#include "BARTforPSPI.h"
#endif

#ifndef NS_H_
#define NS_H_
#include "NS_ridge.h"
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


class PSPI_SplineBART: public BARTforPSPI{
public:
    PSPI_SplineBART(NumericMatrix X_, NumericVector Y_, NumericVector Z_, NumericVector pi_, NumericMatrix X_test_, long n_knots, long order = 3, long ntrees_s = 200) : BARTforPSPI(X_, Y_, Z_, pi_, X_test_, ntrees_s){
    //Rcout << 1234 ;
    
    main_bart = new bart_model(cbind(X, pi), Y, 100L, false, false, false, 200);
    main_bart->update(50, 50, 1, false, 10L);
    sigma = main_bart->get_sigma();
    bart_pre = colMeans(main_bart->predict(cbind(this->X, this->pi)));
    //Rcout << 12344 ;
    Z_1 = (Z == 1.0);
    X_Z = sliceRows(X, Z_1);
    Y_Z = Y[Z_1] - bart_pre[Z_1];
    
    //ns = new NS_R(as<NumericVector>(pi[Z_1]), Y_Z, std::floor(pow(sum(Z_1), 1.0/3.0)), sigma, order);
    ns = new NS_R(as<NumericVector>(pi[Z_1]), Y_Z, n_knots, sigma, order);
    //Rcout << 123 ;
    
    ns->update(sigma);
    //Rcout << 123 ;
    clm_pi_pre = ns->get_ns_outcome();
    //Rcout << 123 ;
    
    cbart = new bart_model(X_Z, Y_Z - clm_pi_pre, 100L, false, false, false, ntrees_s);
    cbart->update(sigma, 50, 50, 1, false, 10L);
    cbart_pre = colMeans(cbart->predict(X_Z));

    cbart_pre_mean = mean(cbart_pre);
    cbart_pre = cbart_pre - cbart_pre_mean;
 
    Z_cbart = NumericVector(Y.length());
    this->update_Z_cbart();
    double rss = sum(pow(Y - Z_cbart - bart_pre, 2));
    sigma = main_bart->get_invchi(n, rss);
    
  };
  
  void update_Z_cbart(){
    NumericVector Z_cbart_Z_1 = cbart_pre + clm_pi_pre;
    Z_cbart[Z_1] = Z_cbart_Z_1;
  }
  
  void update(bool verbose = false) override{
    main_bart->set_data(cbind(X, pi), Y - Z_cbart);
    List main_train_result = main_bart->update(sigma, 1, 1, 1, false, 10L);
    bart_pre = as<NumericVector>(main_train_result["yhat.train.mean"]);
    
    Y_Z = Y[Z_1] - bart_pre[Z_1];
    NumericVector Y_Cb = Y_Z - clm_pi_pre;
    NumericVector w_Z = w[Z_1];

    cbart->set_data(X_Z, Y_Z - clm_pi_pre);
    List cbart_train_result = cbart->update(sigma, 1, 1, 1, false, 10L);
    cbart_pre = as<NumericVector>(cbart_train_result["yhat.train.mean"]);
    cbart_pre_mean = mean(colMeans(cbart->predict(X)));
    cbart_pre = cbart_pre - cbart_pre_mean;
    
    
    NumericVector y_te = Y_Z - cbart_pre;
    
    ns->set_Y(y_te);
    ns->update(sigma * w_Z);
    
    clm_pi_pre = ns->get_ns_outcome();
    
    this->update_Z_cbart();
    double rss = sum(pow(Y - Z_cbart - bart_pre, 2));
    sigma = main_bart->get_invchi(n, rss);
  };
  
  List predict(NumericVector pi_test) override{
    long N = X_test.nrow();
    NumericVector predict_s = ns->predict(pi_test);
    cbart_pop = colMeans(cbart->predict(X_test)) - cbart_pre_mean;
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
      // Named("clm_pi_pre") = clm_pi_pre,
      // Named("cbart_pre_mean") = cbart_pre_mean,
      // Named("Y_hat") = Y,
      // Named("ns_beta") = ns->get_theta(),
      // Named("gamma") = ns->get_gamma()
    );
  };
  
  
private:
  
  NumericVector bart_pre;
  bart_model * cbart;
  NS * ns;
  
  double sigma;
  
  
  LogicalVector Z_1;
  NumericMatrix X_Z;
  NumericVector Y_Z;
  NumericMatrix pi_Z;
  NumericVector Z_cbart;
  NumericVector cbart_pre;
  
  double cbart_pre_mean;
  NumericVector clm_pi_pre;
  NumericVector cbart_pop;
};
