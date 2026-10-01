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


// MSplineBART (DSplineBART). The BART / NS non-identifiability is resolved by
// dropping the NS intercept in both the main-effect spline (ns[0]) and each
// interaction spline (ns[j], j>=1); BART components are not centered.
class PSPI_MSplineBART: public BARTforPSPI{
public:
  PSPI_MSplineBART(NumericMatrix X_, NumericVector Y_,  bool binary_,  IntegerVector Z_, NumericMatrix pi_, NumericMatrix X_test_, IntegerVector n_knots, IntegerVector order, long ntrees_s = 200, bool dart = false, bool aug = false) : BARTforPSPI(X_, Y_, binary_, Z_, pi_, X_test_, ntrees_s){
    X_Z.push_back(X);
    NumericMatrix X_j = X_Z[0];
    bart.push_back(new bart_model(X_j, Y, dart, aug, 100L, false, false, false, ntrees_s));
    bart[0]->update(50, 50, 1, false, 10L);
    sigma = bart[0]->get_sigma();
    NumericVector cbart_pre = colMeans(bart[0]->predict(X_j));
    // NO centering

    NumericVector pi_j = pi(_, 0);
    // NS with intercept = FALSE
    ns.push_back(new NS_R(pi_j, Y - cbart_pre, n_knots[0], sigma, order[0], 1.0, 1.0, 1.0, 1.0, false, false));
    ns[0]->update(sigma);
    NumericVector clm_pi_pre = ns[0]->get_ns_outcome();
    pi_ns_pre.push_back(clm_pi_pre);
    bart_pre = cbart_pre + clm_pi_pre;

    for(int j = 1; j < J; ++j){
      LogicalVector Z_j = (Z == j);
      Z_Z.push_back(Z_j);
      NumericMatrix X_j = sliceRows(X, Z_j);
      NumericVector Y_j = Y[Z_j] - bart_pre[Z_j];
      NumericVector pi_j = pi(_, j);
      pi_j = pi_j[Z_j];
      X_Z.push_back(X_j);
      pi_Z.push_back(pi_j);

      bart.push_back(new bart_model(X_j, Y_j, dart, aug, 100L, false, false, false, ntrees_s));
      bart[j]->update(sigma, 50, 50, 1, false, 10L);
      NumericVector cbart_pre = colMeans(bart[j]->predict(X_j));
      // NO centering

      // NS with intercept = FALSE
      ns.push_back(new NS_R(pi_j, Y_j - cbart_pre, n_knots[j], sigma, order[j], 1.0, 1.0, 1.0, 1.0, false, false));
      ns[j]->update(sigma);
      NumericVector clm_pi_pre = ns[j]->get_ns_outcome();
      pi_ns_pre.push_back(clm_pi_pre);
    }
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
      LogicalVector Z_j = Z_Z[j - 1];
      NumericMatrix X_j = X_Z[j];
      NumericVector r = colMeans(bart[j]->predict(X_j)) + as<NumericVector>(pi_ns_pre[j]);
      Z_cbart[Z_j] = r;
    }
  }

  void update(bool verbose = false) override{
    NumericMatrix X_j = X_Z[0];
    bart[0]->set_Y(Y - Z_cbart - as<NumericVector>(pi_ns_pre[0]));
    bart[0]->update(sigma, w, 0, 1, 1, false, 10L);
    NumericVector cbart_pre = colMeans(bart[0]->predict(X_j));
    // NO centering

    NumericVector Y_j = Y - Z_cbart - cbart_pre;
    ns[0]->set_Y(Y_j);
    ns[0]->update(sigma);
    NumericVector clm_pi_pre = ns[0]->get_ns_outcome();
    pi_ns_pre[0] = clm_pi_pre;
    bart_pre = cbart_pre + clm_pi_pre;

    if(J > 1){
      for(int j = 1; j < J; ++j){
        LogicalVector Z_j = Z_Z[j-1];
        NumericVector cbart_pi_pre = pi_ns_pre[j];
        NumericVector Y_j = Y[Z_j] - bart_pre[Z_j] - cbart_pi_pre;
        NumericVector w_Z = w[Z_j];
        bart[j]->set_Y(Y_j);
        bart[j]->update(sigma, w_Z, 0, 1, 1, false, 10L);
        NumericMatrix X_j = X_Z[j];
        NumericVector cbart_pre = colMeans(bart[j]->predict(X_j));
        // NO centering

        Y_j = Y[Z_j] - bart_pre[Z_j] - cbart_pre;
        ns[j]->set_Y(Y_j);
        ns[j]->update(sigma);
        pi_ns_pre[j] = ns[j]->get_ns_outcome();
      }
      this->update_Z_cbart();
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
    // J splines total: col 0 is main-effect spline h, col j (j>=1) is interaction s_j
    NumericMatrix spline_test(N, J);

    if(J > 1){
      NumericVector sp0 = ns[0]->predict(pi0_test);
      spline_test(_, 0) = sp0;
      outcome(_, 0) = colMeans(bart[0]->predict(X_test)) + sp0;  // no centering
      for(int j = 1; j < J; ++j){
        NumericVector pi_j = pi_test(_, j);
        NumericVector spj = ns[j]->predict(pi_j);
        spline_test(_, j) = spj;
        inter_model(_, j - 1) = colMeans(bart[j]->predict(X_test));  // no centering
        inter_model(_, j - 1) = inter_model(_, j - 1) + spj;
        outcome(_, j) = outcome(_, 0) + inter_model(_, j - 1);
      }
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
    return List::create(Named("outcome") = outcome, Named("outcome_prob") = outcome_hidden, Named("inter_model") = inter_model, Named("spline_test") = spline_test);
  };

  List get_posterior() override{
    // J splines: col 0 = main-effect (all trial), cols 1..J-1 = interaction (arm j only, NA otherwise)
    NumericMatrix spline_train(n, J);
    spline_train(_, 0) = as<NumericVector>(pi_ns_pre[0]);
    if(J > 1){
      for(int j = 1; j < J; ++j){
        LogicalVector Z_j = Z_Z[j-1];
        NumericVector sp = pi_ns_pre[j];
        NumericVector col(n);
        int k = 0;
        for(int i = 0; i < n; ++i){
          if(Z_j[i]){
            col[i] = sp[k++];
          } else {
            col[i] = NA_REAL;
          }
        }
        spline_train(_, j) = col;
      }
    }
    return List::create(
      Named("sigma") = sigma,
      Named("outcome_train") = bart_pre + Z_cbart,
      Named("beta") = get_theta(),
      Named("gamma") = get_gamma(),
      Named("spline_train") = spline_train
    );
  };

  List get_theta(){
    List theta;
    for(int j = 0 ; j < (int)ns.size(); ++j){
      theta.push_back(ns[j]->get_theta());
    }
    return theta;
  }

  NumericVector get_gamma(){
    NumericVector gamma(J);
    for(int j = 0 ; j < (int)ns.size(); ++j){
      gamma[j] = ns[j]->get_gamma();
    }
    return gamma;
  }

  void startdart() override{
    for(int j = 0; j < (int)bart.size(); ++j){
      bart[j]->startdart();
    }
  }

  void set_pi(NumericMatrix pi_) override {
    pi = clone(pi_);
    pi_Z[0] = pi(_, 0);
    for(int j = 1; j < J; ++j){
      LogicalVector Z_j = Z_Z[j-1];
      NumericVector pi_j = pi(_, j);
      pi_j = pi_j[Z_j];
      pi_Z[j] = pi_j;
    }
  }

  // Serialization: no bart_pre_mean field needed
  List get_serialized_state() override {
    List bart_states;
    int nb = (J > 1) ? J : 1;
    for(int j = 0; j < nb; ++j)
      bart_states.push_back(deep_copy_tree_object(bart[j]));
    List spline_theta;
    for(int j = 0; j < (int)ns.size(); ++j)
      spline_theta.push_back(clone(ns[j]->get_theta()));
    return List::create(Named("bart_states") = bart_states,
                        Named("spline_theta") = spline_theta,
                        Named("sigma") = sigma);
  }

  List get_spline_info() override {
    List info;
    for(int j = 0; j < (int)ns.size(); ++j){
      info.push_back(List::create(
        Named("knots") = clone(ns[j]->get_knots()),
        Named("boundary_knots") = clone(ns[j]->get_boundary_knots())
      ));
    }
    return info;
  }

private:

  double sigma;
  NumericVector bart_pre;
  List Z_Z;
  List X_Z;
  List Y_Z;
  List pi_Z;
  List pi_ns_pre;
  NumericVector Z_cbart;
  std::vector<NS*> ns;
};
