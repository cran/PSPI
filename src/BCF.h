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

#ifndef LAMDBA_H_
#define LAMDBA_H_
#include "BART/lambda.h"
#endif



#ifndef RCPP_H_
#define RCPP_H_
#include <Rcpp.h>
#endif


using namespace Rcpp;


class BCF: public BARTforPSPI{
public:
  BCF(NumericMatrix X_, NumericVector Y_, bool binary_, IntegerVector Z_, NumericMatrix pi_, NumericMatrix X_test_, long ntrees_s = 200, bool dart = false, bool aug = false) : BARTforPSPI(X_, Y_, binary_, Z_, pi_, X_test_, ntrees_s){
    //Rcout << pi_ << std::endl;
    //Rcout << 123 << std::endl;
    X_Z.push_back(cbind(X, pi(_, 0)));
    NumericMatrix X_j = X_Z[0];
    //Rcout << 123 << std::endl;
    bart.push_back(new bart_model(X_j, Y, dart, aug, 100L, false, false, false, ntrees_s));
    //Rcout << 123 << std::endl;
    bart[0]->update(50, 50, 1, false, 10L);
    sigma = bart[0]->get_sigma();
    
    bart_pre = colMeans(bart[0]->predict(X_j));
    for(int j = 1; j < J; ++j){
      LogicalVector Z_j = (Z == j);
      Z_Z.push_back(Z_j);
      NumericMatrix X_j = sliceRows(X, Z_j);
      NumericVector Y_j = Y[Z_j] - bart_pre[Z_j];
      X_Z.push_back(X_j);
      bart.push_back(new bart_model(X_j, Y_j, dart, aug, 100L, false, false, false, ntrees_s));
      bart[j]->update(sigma, 50, 50, 1, false, 10L);
    }
    //Rcout << 123 << std::endl;
    Z_cbart = NumericVector(Y.length());
    this->update_Z_cbart();
    if(binary){
      sigma = 1;
      for(int i = 0; i < n; ++i){
        if(Y_[i] == 0){
          NumericVector mean_y = rtruncnorm(1, bart_pre[i] + Z_cbart[i], sigma, R_NegInf, 0);
          Y[i] = mean_y[0];
        }else{
          NumericVector mean_y = rtruncnorm(1, bart_pre[i] + Z_cbart[i], sigma, 0, R_PosInf);
          Y[i] = mean_y[0];
        }
      }
    }
  };
  
  
  
  BCF(NumericMatrix X_, NumericVector Y_, bool binary_, NumericMatrix pi_, NumericMatrix X_test_, long ntrees_s = 200, bool dart = false, bool aug = false) : BARTforPSPI(X_, Y_, binary_, pi_, X_test_, ntrees_s){
    bart.push_back(new bart_model(cbind(X, pi(_, 0)), Y, dart, aug, 100L, false, false, false, ntrees_s));
    bart[0]->update(50, 50, 1, false, 10L);
    sigma = bart[0]->get_sigma();
    X_Z.push_back(cbind(X, pi(_, 0)));
    NumericMatrix X_j = X_Z[0];
    bart_pre = colMeans(bart[0]->predict(X_j));
    Z_cbart = NumericVector(Y.length());
    this->update_Z_cbart();
    
    if(binary){
      sigma = 1;
      for(int i = 0; i < n; ++i){
        if(Y_[i] == 0){
          NumericVector mean_y = rtruncnorm(1, bart_pre[i] + Z_cbart[i], sigma, R_NegInf, 0);
          Y[i] = mean_y[0];
        }else{
          NumericVector mean_y = rtruncnorm(1, bart_pre[i] + Z_cbart[i], sigma, 0, R_PosInf);
          Y[i] = mean_y[0];
        }
      }
    }
  };
  
  void update_Z_cbart(){
    for(int j = 1; j < J; ++j){
      LogicalVector Z_j = Z_Z[j-1];
      NumericMatrix X_j = X_Z[j];
      Z_cbart[Z_j] = colMeans(bart[j]->predict(X_j));
    }
  }
  
  void update(bool verbose = false) override{
    NumericMatrix X_j = X_Z[0];
    //bart[0]->set_data(X_j, Y - Z_cbart);
    bart[0]->set_Y(Y - Z_cbart);
    bart[0]->update(sigma, w, 0, 1, 1, false, 10L);
    bart_pre = colMeans(bart[0]->predict(X_j));
    for(int j = 1; j < J; ++j){
      LogicalVector Z_j = Z_Z[j-1];
      NumericVector Y_j = Y[Z_j] - bart_pre[Z_j];
      //NumericMatrix X_j = X_Z[j];
      NumericVector w_Z = w[Z_j];
      //bart[j]->set_data(X_j, Y_j);
      bart[j]->set_Y(Y_j);
      bart[j]->update(sigma, w_Z, 0, 1, 1, false, 10L);
    }
    this->update_Z_cbart();
    double rss = sum(pow(Y - Z_cbart - bart_pre, 2));
    sigma = bart[0]->get_invchi(n, rss);
    
    if(binary){
      sigma = 1;
      for(int i = 0; i < n; ++i){
        if(Y_[i] == 0){
          NumericVector mean_y = rtruncnorm(1, bart_pre[i] + Z_cbart[i], sigma, R_NegInf, 0);
          Y[i] = mean_y[0];
        }else{
          NumericVector mean_y = rtruncnorm(1, bart_pre[i] + Z_cbart[i], sigma, 0, R_PosInf);
          Y[i] = mean_y[0];
        }
      }
    }
  };
  
  List predict(NumericMatrix pi_test) override{
    long N = X_test.nrow();
    NumericVector pi0_test = pi_test(_, 0);
    
    NumericMatrix outcome(N, J);
    NumericMatrix inter_model(N, J - 1);
    NumericMatrix outcome_hidden(N, J);
    outcome(_, 0) = colMeans(bart[0]->predict(cbind(X_test, pi0_test)));
    for(int j = 1; j < J; ++j){
      inter_model(_, j - 1) = colMeans(bart[j]->predict(X_test));
      outcome(_, j) = outcome(_, 0) + inter_model(_, j - 1);
    }
    if(binary){
      for(int i = 0; i < N; ++i){
        for(int j = 0; j < J; ++j){
          outcome_hidden(i, j) = R::pnorm(outcome(i, j), 0, 1, true, false);
          outcome(i, j) = R::rbinom(1, outcome_hidden(i, j));
        }
      }
    }else{
      for(int i = 0; i < N; ++i){
        for(int j = 0; j < J; ++j){
          outcome(i, j) = outcome(i, j) + R::rnorm(0, sigma);
        }
      }
    }
    return List::create(Named("outcome") = outcome, Named("outcome_prob") = outcome_hidden, Named("inter_model") = inter_model);
  };
  
  List get_posterior() override{
    return List::create(
      Named("sigma") = sigma,
      Named("outcome_train") = bart_pre + Z_cbart
    );
  };
  
  List get_serialized_state() override {
    List bart_states;
    for(int j = 0; j < J; ++j)
      bart_states.push_back(deep_copy_tree_object(bart[j]));
    return List::create(Named("bart_states") = bart_states,
                        Named("sigma") = sigma);
  }

  void startdart() override{
    for(int j = 0; j < J; ++j){
      bart[j]->startdart();
    }
  }

  void set_pi(NumericMatrix pi_) override {
    pi = clone(pi_);

    // Rebuild mu design matrix
    X_Z[0] = cbind(X, pi(_, 0));
  }

private:
  double sigma;
  NumericVector bart_pre;
  List Z_Z;
  List X_Z;
  NumericVector Z_cbart;

};
