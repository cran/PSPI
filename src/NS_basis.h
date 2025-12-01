#ifndef ARMADILLO_H_
#define ARMADILLO_H_
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#endif

#ifndef RCPP_H_
#define RCPP_H_
#include <Rcpp.h>
#endif

#ifndef RCPPDIST_H_
#define RCPPDIST_H_
#include <RcppDist.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
#endif

#include <algorithm>

using namespace Rcpp;
using namespace arma;



class NS_basis{
public:
  NS_basis(){};
  
  NS_basis(NumericVector x, long K, int order = 3){
    this->K = K;
    this->n = x.length();
    this->order = order;
    boundary_knots.push_back(min(x));
    boundary_knots.push_back(max(x));
    knots = cpp_quantile_seq(x, K);
    if(K > 2)
      internal_knots = knots[Range(1, K - 2)];
    
    basis = NumericMatrix(n, K);
    for(int i = 0 ; i < n; ++i){
      basis(i, 0) = 1;
      basis(i, 1) = x[i];
      for(int j = 0 ; j < K - 2; ++j){
        basis(i, j + 2) = ((pow(std::max(x[i] - knots[j], 0.0), order) - pow(std::max(x[i] - knots[K - 1], 0.0), order)) / (knots[K - 1] - knots[j]) - (pow(std::max(x[i] - knots[K - 2], 0.0), order) - pow(std::max(x[i] - knots[K - 1], 0.0), order)) / (knots[K - 1] - knots[K - 2]));
      }
    }
    if(K > 2)
      ns_part = basis(_ , Range(2, K-1));
    
  };
  
  NumericMatrix get_basis(){
    return basis;
  }
  
  NumericMatrix get_ns_part(){
    return ns_part;
  }
  
  NumericVector get_boundary_knots(){
    return boundary_knots;
  }
  
  NumericVector get_internal_knots(){
    return internal_knots;
  }
  
  NumericVector get_knots(){
    return knots;
  }
  

  
  NumericMatrix predict(NumericVector x){
    NumericMatrix basis_predict = NumericMatrix(x.length(), K);
    for(int i = 0 ; i < x.length(); ++i){
      basis_predict(i, 0) = 1;
      basis_predict(i, 1) = x[i];
      for(int j = 0 ; j < K - 2; ++j){
        basis_predict(i, j + 2) = (pow(std::max(x[i] - knots[j], 0.0), order) - pow(std::max(x[i] - knots[K - 1], 0.0), order)) / (knots[K - 1] - knots[j]) - (pow(std::max(x[i] - knots[K - 2], 0.0), order) - pow(std::max(x[i] - knots[K - 1], 0.0), order)) / (knots[K - 1] - knots[K - 2]);
      }
    }
    return basis_predict;
  }
  
  NumericMatrix predict_ns_part(NumericVector x){
    NumericMatrix basis_predict = NumericMatrix(x.length(), K - 2);
    for(int i = 0 ; i < x.length(); ++i){
      for(int j = 0 ; j < K - 2; ++j){
        basis_predict(i, j) = (pow(std::max(x[i] - knots[j], 0.0), order) - pow(std::max(x[i] - knots[K - 1], 0.0), order)) / (knots[K - 1] - knots[j]) - (pow(std::max(x[i] - knots[K - 2], 0.0), order) - pow(std::max(x[i] - knots[K - 1], 0.0), order)) / (knots[K - 1] - knots[K - 2]);
      }
    }
    return basis_predict;
  }
  
  NumericVector cpp_quantile_seq(NumericVector x, int K) {
    // Handle edge cases
    if (K <= 0) {
      stop("K must be a positive integer.");
    }
    
    int n = x.size();
    
    if (n == 0) {
      return NumericVector(K, NA_REAL);
    }
    
    // Create a sorted copy of x
    NumericVector sorted_x = clone(x);
    std::sort(sorted_x.begin(), sorted_x.end());
    
    // Generate the probabilities seq(0, 1, length.out = K)
    NumericVector probs(K);
    
    if (K == 1) {
      probs[0] = 0.0; // Only the minimum
    } else {
      double step = 1.0 / (K - 1);
      for(int i = 0; i < K; ++i){
        probs[i] = i * step;
      }
    }
    
    // Function to compute a single quantile using Type 7
    auto compute_quantile = [&](double p) -> double {
      if (p <= 0.0) {
        return sorted_x[0];
      }
      if (p >= 1.0) {
        return sorted_x[n - 1];
      }
      
      double pos = 1.0 + (n - 1) * p;
      double index = pos - 1.0; // 0-based index
      int lower = static_cast<int>(std::floor(index));
      int upper = static_cast<int>(std::ceil(index));
      double frac = index - lower;
      
      if (upper >= n) {
        return sorted_x[n - 1];
      }
      
      return sorted_x[lower] + frac * (sorted_x[upper] - sorted_x[lower]);
    };
    
    // Compute all quantiles
    NumericVector quantiles(K);
    for(int i = 0; i < K; ++i){
      quantiles[i] = compute_quantile(probs[i]);
    }
    
    return quantiles;
  }
  
  
  
  
private:
  NumericVector boundary_knots;
  NumericVector internal_knots;
  NumericVector knots;
  
  long K;
  long n;
  
  int order;
  
  NumericMatrix ns_part;
  NumericMatrix basis;
};



// // [[Rcpp::export]]
// List test_NS_basis(NumericVector X, long K, NumericVector X_test){
//   NS_basis * a = new NS_basis(X, K);
//   return List::create(Named("predict") = a->predict(X_test), Named("predict_ns") = a->predict_ns_part(X_test), Named("boundary_knots") = a->get_boundary_knots(), Named("knots") = a->get_knots(), Named("internal_knots") = a->get_internal_knots(), Named("ns_part") = a->get_ns_part(), Named("basis") = a->get_basis());
// };
