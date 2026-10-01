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
  
  PSPI_FullBART(NumericMatrix X_, NumericVector Y_, bool binary_, IntegerVector Z_, NumericMatrix pi_, NumericMatrix X_test_, long ntrees_s = 200, bool dart = false, bool aug = false) : BARTforPSPI(X_, Y_, binary_, Z_, pi_, X_test_, ntrees_s){
    //Rcout << 123 << std::endl;
    bart_pre_mean = NumericVector(J - 1);
    X_Z.push_back(cbind(X, pi(_, 0)));
    NumericMatrix X_j = X_Z[0];
    bart.push_back(new bart_model(X_j, Y, dart, aug, 100L, false, false, false, ntrees_s));
    bart[0]->update(50, 50, 1, false, 10L);
    sigma = bart[0]->get_sigma();
    
    bart_pre = colMeans(bart[0]->predict(X_j));
    //Rcout << 123 << std::endl;
    for(int j = 1; j < J; ++j){
      LogicalVector Z_j = (Z == j);
      Z_Z.push_back(Z_j);
      NumericMatrix X_j = sliceRows(X, Z_j);
      NumericVector Y_j = Y[Z_j] - bart_pre[Z_j];
      NumericVector pi_j_ = pi(_, j);
      pi_j_ = pi_j_[Z_j];
      NumericMatrix pi_j = NumericMatrix(pi_j_.length(), 1, pi_j_.begin());
      X_Z.push_back(X_j);
      pi_Z.push_back(pi_j);
      
      bart.push_back(new bart_model(X_j, Y_j, dart, aug, 100L, false, false, false, ntrees_s));
      bart[j]->update(sigma, 50, 50, 1, false, 10L);
      NumericVector cbart_pre = colMeans(bart[j]->predict(X_j));
      bart_pre_mean[j - 1] = mean(colMeans(bart[j]->predict(X)));
      cbart_pre = cbart_pre - bart_pre_mean[j - 1];

      bart_pi.push_back(new bart_model(pi_j, Y_j - cbart_pre, dart, aug, 100L, false, false, false, ntrees_s));
      bart_pi[j - 1]->update(sigma, 50, 50, 1, false, 10L);
      NumericVector cbart_pi_pre = colMeans(bart_pi[j - 1]->predict(pi_j));
      pi_bart_pre.push_back(cbart_pi_pre);
      
    }
    Z_cbart = NumericVector(Y.length());
    //Rcout << 123 << std::endl;
    this->update_Z_cbart();
    //Rcout << 123 << std::endl;
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
  
  PSPI_FullBART(NumericMatrix X_, NumericVector Y_, bool binary_, NumericMatrix pi_, NumericMatrix X_test_, long ntrees_s = 200, bool dart = false, bool aug = false) : BARTforPSPI(X_, Y_, binary_, pi_, X_test_, ntrees_s){
    bart_pre_mean = NumericVector(1);
    bart.push_back(new bart_model(X, Y, dart, aug, 100L, false, false, false, ntrees_s));
    bart[0]->update(50, 50, 1, false, 10L);
    sigma = bart[0]->get_sigma();
    X_Z.push_back(X);
    NumericMatrix X_j = X_Z[0];
    bart_pre = colMeans(bart[0]->predict(X_j));
    bart_pre_mean[0] = mean(bart_pre);
    bart_pre = bart_pre - bart_pre_mean[0];
    
    NumericVector pi_j_ = pi(_, 0);
    NumericMatrix pi_j = NumericMatrix(pi_j_.length(), 1, pi_j_.begin());
    pi_Z.push_back(pi_j);
    
    bart_pi.push_back(new bart_model(pi_j, Y - bart_pre, dart, aug, 100L, false, false, false, ntrees_s));
    bart_pi[0]->update(sigma, 50, 50, 1, false, 10L);
    Z_cbart = colMeans(bart_pi[0]->predict(pi_j));
    pi_bart_pre.push_back(Z_cbart);
    
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
      NumericVector r = colMeans(bart[j]->predict(X_j)) - bart_pre_mean[j - 1] + as<NumericVector>(pi_bart_pre[j-1]);
      Z_cbart[Z_j] = r;
    }
  }
  
  void update(bool verbose = false) override{
    NumericMatrix X_j = X_Z[0];
    //bart[0]->set_data(X_j, Y - Z_cbart);
    bart[0]->set_Y(Y - Z_cbart);
    bart[0]->update(sigma, w, 0, 1, 1, false, 10L);
    bart_pre = colMeans(bart[0]->predict(X_j));
    
    if(J > 1){
      for(int j = 1; j < J; ++j){
        LogicalVector Z_j = Z_Z[j-1];
        NumericVector cbart_pi_pre = pi_bart_pre[j-1];
        NumericVector Y_j = Y[Z_j] - bart_pre[Z_j] - cbart_pi_pre;
        //NumericMatrix X_j = X_Z[j];
        NumericVector w_Z = w[Z_j];
        //bart[j]->set_data(X_j, Y_j);
        bart[j]->set_Y(Y_j);
        bart[j]->update(sigma, w_Z, 0, 1, 1, false, 10L);
        NumericMatrix X_j = X_Z[j];
        NumericVector cbart_pre = colMeans(bart[j]->predict(X_j));
        bart_pre_mean[j - 1] = mean(colMeans(bart[j]->predict(X)));
        cbart_pre = cbart_pre - bart_pre_mean[j - 1];
        
        Y_j = Y[Z_j] - bart_pre[Z_j] - cbart_pre;
        bart_pi[j - 1]->set_Y(Y_j);
        bart_pi[j - 1]->update(sigma, w_Z, 0, 1, 1, false, 10L);
        NumericMatrix pi_j = pi_Z[j - 1];
        pi_bart_pre[j-1] = colMeans(bart_pi[j - 1]->predict(pi_j));
      }
      this->update_Z_cbart();
    }else{
      bart_pre_mean[0] = mean(bart_pre);
      bart_pre = bart_pre - bart_pre_mean[0];
      
      NumericVector Y_j = Y - bart_pre;
      bart_pi[0]->set_Y(Y_j);
      bart_pi[0]->update(sigma, w, 0, 1, 1, false, 10L);
      NumericMatrix pi_j = pi_Z[0];
      Z_cbart = colMeans(bart_pi[0]->predict(pi_j));
      pi_bart_pre[0] = Z_cbart;
    }
    
    
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
    
    if(J > 1){
      outcome(_, 0) = colMeans(bart[0]->predict(cbind(X_test, pi0_test)));
      for(int j = 1; j < J; ++j){
        
        inter_model(_, j - 1) = colMeans(bart[j]->predict(X_test)) - bart_pre_mean[j - 1];
        
        
        //outcome(_, j) = outcome(_, 0) + colMeans(bart[j]->predict(X_test)) - bart_pre_mean[j - 1];
        NumericVector pi_j_ = pi_test(_, j);
        NumericMatrix pi_j = NumericMatrix(pi_j_.length(), 1, pi_j_.begin());
        inter_model(_, j - 1) = inter_model(_, j - 1) + colMeans(bart_pi[j - 1]->predict(pi_j));
        outcome(_, j) = outcome(_, 0) + inter_model(_, j - 1);
      }
    }else{
      outcome(_, 0) = colMeans(bart[0]->predict(X_test)) - bart_pre_mean[0];
      NumericVector pi_j_ = pi_test(_, 0);
      NumericMatrix pi_j = NumericMatrix(pi_j_.length(), 1, pi_j_.begin());
      outcome(_, 0) = outcome(_, 0) + colMeans(bart_pi[0]->predict(pi_j));
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
      // Named("bart_pre") = bart_pre,
      // Named("cbart_pre") = cbart_pre,
      // Named("Z_cbart") = Z_cbart,
      // Named("cbart_pi_pre") = cbart_pi_pre,
      // Named("cbart_pre_mean") = cbart_pre_mean
    );
  };
  
  List get_serialized_state() override {
    List bart_states;
    int nb = (J > 1) ? J : 1;
    for(int j = 0; j < nb; ++j)
      bart_states.push_back(deep_copy_tree_object(bart[j]));
    List bart_pi_states;
    int np = (J > 1) ? (J - 1) : 1;
    for(int j = 0; j < np; ++j)
      bart_pi_states.push_back(deep_copy_tree_object(bart_pi[j]));
    return List::create(Named("bart_states") = bart_states,
                        Named("bart_pi_states") = bart_pi_states,
                        Named("bart_pre_mean") = clone(bart_pre_mean),
                        Named("sigma") = sigma);
  }

  void startdart() override{
    for(int j = 0; j < J; ++j){
      bart[j]->startdart();
    }

    for(int j = 0; j < J - 1; ++j){
      bart_pi[j]->startdart();
    }
    if(J == 1)
      bart_pi[0]->startdart();
  }

  void set_pi(NumericMatrix pi_) override {
    pi = clone(pi_);
    X_Z[0] = cbind(X, pi(_, 0));
    for(int j = 1; j < J; ++j){
      LogicalVector Z_j = Z_Z[j-1];
      NumericVector pi_j_ = pi(_, j);
      pi_j_ = pi_j_[Z_j];
      NumericMatrix pi_j = NumericMatrix(pi_j_.length(), 1, pi_j_.begin());
      pi_Z[j - 1] = pi_j;
    }
  }
  
private:
  double sigma;
  NumericVector bart_pre;
  List Z_Z;
  List X_Z;
  List Y_Z;
  List pi_Z;
  List pi_bart_pre;
  NumericVector Z_cbart;
  std::vector<bart_model*> bart_pi;
  NumericVector bart_pre_mean;
  
};
