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
#include "NS.h"
#endif

#ifndef RCPPDIST_H_
#define RCPPDIST_H_
#include <RcppDist.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
#endif

#include <algorithm>

using namespace Rcpp;
using namespace arma;



class NS_R: public NS{
public:
  NS_R(){};
  
  NS_R(NumericVector X, NumericVector y, long K, double sigma, int order = 3, double alpha_0 = 1, double beta_0 = 1, double alpha_k = 1, double beta_k = 1, bool local = false) : NS(X, y, K, sigma, order){
    this->alpha_0 = alpha_0;
    this->beta_0 = beta_0;
    this->alpha_k = alpha_k;
    this->beta_k = beta_k;
    this->local = local;

    if(K > 2){
      gamma = 1;//sqrt(rinvgamma(alpha_0, beta_0));
      if(local){
        for(int k = 2; k <= K-1; ++k){
          lambda_k.insert_rows(lambda_k.n_rows, 1);
          lambda_k(lambda_k.n_rows - 1) = sqrt(rinvgamma(alpha_k, beta_k));
        }
      }
    }
    update_outcome();
  };

  
  
  
  void update(){
    if(K > 2){
      lr_coefficient = update_beta(lr_basis, this->y - ns_part_outcome, this->sigma);
    }else{
      lr_coefficient = update_beta(lr_basis, this->y, this->sigma);
    }
    lr_outcome = lr_basis * lr_coefficient;
    //Rcout << "after lr" << std::endl;
    if(K > 2){
      if(local){
        arma::mat Lambda = arma::diagmat(pow(lambda_k, 2)) * pow(gamma, 2);
        arma::mat theta_var = arma::inv_sympd(arma::inv_sympd(Lambda) + 1 / pow(sigma, 2) * ns_basis.t() * ns_basis);
        arma::vec theta_mean = arma::inv_sympd(pow(sigma, 2) * arma::inv_sympd(Lambda) + ns_basis.t() * ns_basis) * ns_basis.t() * (this->y - lr_outcome);
        ns_coefficient = arma::vectorise(rmvnorm(1, theta_mean, theta_var));
      }else{
        arma::mat theta_var = arma::inv_sympd(1 / pow(gamma, 2) * arma::eye<arma::mat>(K - 2, K - 2) + 1 / pow(this->sigma, 2) * ns_basis.t() * ns_basis);
        arma::vec theta_mean = arma::inv_sympd(pow(this->sigma, 2) / pow(gamma, 2) * arma::eye<arma::mat>(K - 2, K - 2) + ns_basis.t() * ns_basis) * ns_basis.t() * (this->y - lr_outcome);
        ns_coefficient = arma::vectorise(rmvnorm(1, theta_mean, theta_var));
      }
      
      ns_part_outcome = ns_basis * ns_coefficient;
      ns_outcome = lr_outcome + ns_part_outcome;
      
      if(local){
        for(int k = 0; k < K-2; ++k){
          lambda_k[k] = sqrt(rinvgamma(alpha_k + 0.5, beta_k + 0.5 * pow(ns_coefficient[k] / gamma, 2)));
        }
        gamma = sqrt(rinvgamma(alpha_0 + (K-2.0) / 2.0, beta_0 + 0.5 * arma::dot(ns_coefficient / lambda_k, ns_coefficient / lambda_k)));
      }else{
        gamma = sqrt(rinvgamma(alpha_0 + (K-2.0) / 2.0, beta_0 + 0.5 * arma::dot(ns_coefficient, ns_coefficient)));
      } 
    }
  }
  
  
  void update(double sigma){
    //Rcout << "begin" << std::endl;
    this->sigma = sigma;
    if(K > 2){
      //Rcout << 456 ;
      lr_coefficient = update_beta(lr_basis, this->y - ns_part_outcome, this->sigma);
      //Rcout << 456 ;
    }else{
      lr_coefficient = update_beta(lr_basis, this->y, this->sigma);
    }
    lr_outcome = lr_basis * lr_coefficient;
    //Rcout << "after lr" << std::endl;
    if(K > 2){
      if(local){
        arma::mat Lambda = arma::diagmat(pow(lambda_k, 2)) * pow(gamma, 2);
        arma::mat theta_var = arma::inv_sympd(arma::inv_sympd(Lambda) + 1 / pow(sigma, 2) * ns_basis.t() * ns_basis);
        arma::vec theta_mean = arma::inv_sympd(pow(sigma, 2) * arma::inv_sympd(Lambda) + ns_basis.t() * ns_basis) * ns_basis.t() * (this->y - lr_outcome);
        ns_coefficient = arma::vectorise(rmvnorm(1, theta_mean, theta_var));
      }else{
        
        arma::mat theta_var = arma::inv_sympd(1 / pow(gamma, 2) * arma::eye<arma::mat>(K - 2, K - 2) + 1 / pow(this->sigma, 2) * ns_basis.t() * ns_basis);
        arma::vec theta_mean = arma::inv_sympd(pow(this->sigma, 2) / pow(gamma, 2) * arma::eye<arma::mat>(K - 2, K - 2) + ns_basis.t() * ns_basis) * ns_basis.t() * (this->y - lr_outcome);
        ns_coefficient = arma::vectorise(rmvnorm(1, theta_mean, theta_var));
        
      }
      
      ns_part_outcome = ns_basis * ns_coefficient;
      ns_outcome = lr_outcome + ns_part_outcome;
      
      if(local){
        for(int k = 0; k < K-2; ++k){
          lambda_k[k] = sqrt(rinvgamma(alpha_k + 0.5, beta_k + 0.5 * pow(ns_coefficient[k] / gamma, 2)));
        }
          gamma = sqrt(rinvgamma(alpha_0 + (K-2.0) / 2.0, beta_0 + 0.5 * arma::dot(ns_coefficient / lambda_k, ns_coefficient / lambda_k)));
      }else{
          gamma = sqrt(rinvgamma(alpha_0 + (K-2.0) / 2.0, beta_0 + 0.5 * arma::dot(ns_coefficient, ns_coefficient)));
      } 
    }
  }
  

  void update(NumericVector sigma){
    sigma = as<arma::vec>(sigma);
    if(K > 2){
      lr_coefficient = update_beta(lr_basis, this->y - ns_part_outcome, sigma);
    }else{
      lr_coefficient = update_beta(lr_basis, this->y, sigma);
    }
    lr_outcome = lr_basis * lr_coefficient;
    //Rcout << "after lr" << std::endl;
    if(K > 2){
      if(local){
        arma::mat Lambda = arma::diagmat(pow(lambda_k, 2)) * pow(gamma, 2);
        arma::mat theta_var = arma::inv_sympd(arma::inv_sympd(Lambda) + ns_basis.t() * inv_Sigma * ns_basis);
        arma::vec theta_mean = theta_var * (ns_basis.t() * inv_Sigma * (this->y - lr_outcome));
        ns_coefficient = arma::vectorise(rmvnorm(1, theta_mean, theta_var));
      }else{
        arma::mat theta_var = arma::inv_sympd(1 / pow(gamma, 2) * arma::eye<arma::mat>(K - 2, K - 2) + ns_basis.t() * inv_Sigma * ns_basis);
        arma::vec theta_mean = theta_var * (ns_basis.t() * inv_Sigma * (this->y - lr_outcome));
        ns_coefficient = arma::vectorise(rmvnorm(1, theta_mean, theta_var));
      }
      
      ns_part_outcome = ns_basis * ns_coefficient;
      ns_outcome = lr_outcome + ns_part_outcome;
      
      if(local){
        for(int k = 0; k < K-2; ++k){
          lambda_k[k] = sqrt(rinvgamma(alpha_k + 0.5, beta_k + 0.5 * pow(ns_coefficient[k] / gamma, 2)));
        }
        gamma = sqrt(rinvgamma(alpha_0 + (K-2.0) / 2.0, beta_0 + 0.5 * arma::dot(ns_coefficient / lambda_k, ns_coefficient / lambda_k)));
      }else{
        gamma = sqrt(rinvgamma(alpha_0 + (K-2.0) / 2.0, beta_0 + 0.5 * arma::dot(ns_coefficient, ns_coefficient)));
      } 
    }
  }  
  

  
protected:

  double alpha_0;
  double beta_0;
  double alpha_k;
  double beta_k;
  
  arma::vec lambda_k;
  arma::vec vk;
  
  bool local;
  bool binary;
  
};



// // [[Rcpp::export]]
// List test_NS(NumericVector X, NumericVector X_test, NumericVector y, long K){
//   NS_R * a = new NS_R(X, y, K, 1);
//   NumericMatrix theta(6000, K);
//   NumericMatrix gamma(6000, 1);
//   //Rcout << a->get_theta() << std::endl;
//   for(int i = 0; i < 6000; ++i){
//     a->update(2.0 + NumericVector(y.length()));
//     //a->update(2.0);
//     theta(i, _) = a->get_theta();
//     gamma(i, 0) = a->get_gamma();
//   }
//   return List::create(Named("ns_predict") = a->predict(X_test), Named("gamma") = gamma, Named("theta") = theta);
//   //return List::create(Named("ns_outcome") = wrap(a->ns_outcome), Named("lr_outcome") = wrap(a->lr_outcome), Named("ns_part_outcome") = wrap(a->ns_part_outcome), Named("eta") = a->get_eta(), Named("gamma") = a->get_gamma(), Named("theta") = a->get_theta(), Named("boundary_knots") = a->get_boundary_knots(), Named("knots") = a->get_knots(), Named("internal_knots") = a->get_internal_knots(), Named("ns_part") = a->get_ns_part(), Named("lr_part") = a->get_lr_part(), Named("basis") = a->get_basis());
// };
