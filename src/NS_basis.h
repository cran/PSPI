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
  NS_basis() {}

  // Predict-only constructor: rebuild from saved knots (no training data needed)
  NS_basis(NumericVector knots_, int order_, bool intercept_)
    : knots(clone(knots_)), K(knots_.size()), n(0), order(order_), intercept(intercept_) {
    if(K < 2) stop("K must be >= 2.");
    boundary_knots = NumericVector::create(knots_[0], knots_[K-1]);
    if(K > 2) internal_knots = knots[Range(1, K - 2)];
    else internal_knots = NumericVector(0);
  }

  NS_basis(NumericVector x, long K, int order = 3, bool intercept = false){
    if(K < 2) stop("K must be >= 2.");
    this->K = K;
    this->n = x.size();
    this->order = order;
    this->intercept = intercept;
    
    double xmin = Rcpp::min(x);
    double xmax = Rcpp::max(x);
    if(!(xmax > xmin)) stop("x must have positive range.");
    
    boundary_knots = NumericVector::create(xmin, xmax);
    knots = cpp_quantile_seq(x, (int)K);
    
    if(knots.size() != K) stop("Internal error: knots size mismatch.");
    
    for(long j = 0; j < K - 1; ++j){
      if(knots[j + 1] < knots[j]) stop("Knots must be nondecreasing.");
    }
    if(!(knots[K - 1] > knots[K - 2])) stop("Last two knots must be distinct.");
    for(long j = 0; j < K - 2; ++j){
      if(!(knots[K - 1] > knots[j])) stop("Encountered non-distinct knot causing zero denominator.");
    }
    
    if(K > 2) internal_knots = knots[Range(1, K - 2)];
    else internal_knots = NumericVector(0);
    
    NumericMatrix full = eval_full_basis(x);          // n x K
    ns_part = (K > 2) ? full(_, Range(2, K - 1)) : NumericMatrix(n, 0);
    
    if(intercept){
      basis = full;                                   // n x K
    }else{
      basis = full(_, Range(1, K - 1));               // n x (K-1)
    }
  }
  
  NumericMatrix get_basis() const { return basis; }
  
  NumericMatrix get_ns_part() const { return ns_part; }
  
  NumericVector get_boundary_knots() const { return boundary_knots; }
  
  NumericVector get_internal_knots() const { return internal_knots; }
  
  NumericVector get_knots() const { return knots; }
  
  NumericMatrix predict(NumericVector x) const{
    NumericMatrix full = eval_full_basis(x);          // |x| x K
    if(intercept) return full;
    return full(_, Range(1, K - 1));
  }
  
  NumericMatrix predict_ns_part(NumericVector x) const{
    NumericMatrix full = eval_full_basis(x);          // |x| x K
    if(K <= 2) return NumericMatrix(x.size(), 0);
    return full(_, Range(2, K - 1));
  }
  
private:
  NumericMatrix eval_full_basis(const NumericVector& x) const{
    long m = x.size();
    NumericMatrix out(m, K);
    
    double k_last = knots[K - 1];
    double k_pen  = knots[K - 2];
    
    for(long i = 0; i < m; ++i){
      out(i, 0) = 1.0;
      out(i, 1) = x[i];
      
      for(long j = 0; j < K - 2; ++j){
        double kj = knots[j];
        
        double a1 = std::pow(std::max(x[i] - kj,     0.0), (double)order);
        double a2 = std::pow(std::max(x[i] - k_last, 0.0), (double)order);
        double b1 = std::pow(std::max(x[i] - k_pen,  0.0), (double)order);
        double b2 = a2;
        
        double term1 = (a1 - a2) / (k_last - kj);
        double term2 = (b1 - b2) / (k_last - k_pen);
        
        out(i, j + 2) = term1 - term2;
      }
    }
    return out;
  }
  
  NumericVector cpp_quantile_seq(NumericVector x, int K) const{
    if(K <= 0) stop("K must be positive.");
    int n = x.size();
    if(n == 0) return NumericVector(K, NA_REAL);
    
    NumericVector sorted_x = clone(x);
    std::sort(sorted_x.begin(), sorted_x.end());
    
    NumericVector probs(K);
    if(K == 1){
      probs[0] = 0.0;
    }else{
      double step = 1.0 / (K - 1);
      for(int i = 0; i < K; ++i) probs[i] = i * step;
    }
    
    auto q7 = [&](double p)->double{
      if(p <= 0.0) return sorted_x[0];
      if(p >= 1.0) return sorted_x[n - 1];
      double pos = 1.0 + (n - 1) * p;
      double idx = pos - 1.0;
      int lo = (int)std::floor(idx);
      int hi = (int)std::ceil(idx);
      double frac = idx - lo;
      if(hi >= n) return sorted_x[n - 1];
      return sorted_x[lo] + frac * (sorted_x[hi] - sorted_x[lo]);
    };
    
    NumericVector quant(K);
    for(int i = 0; i < K; ++i) quant[i] = q7(probs[i]);
    return quant;
  }
  
private:
  NumericVector boundary_knots;
  NumericVector internal_knots;
  NumericVector knots;
  
  long K = 0;
  long n = 0;
  int order = 3;
  
  NumericMatrix ns_part;
  NumericMatrix basis;
  
  bool intercept = false;
};



// // [[Rcpp::export]]
// List test_NS_basis(NumericVector X, long K, NumericVector X_test){
//   NS_basis * a = new NS_basis(X, K);
//   return List::create(Named("predict") = a->predict(X_test), Named("predict_ns") = a->predict_ns_part(X_test), Named("boundary_knots") = a->get_boundary_knots(), Named("knots") = a->get_knots(), Named("internal_knots") = a->get_internal_knots(), Named("ns_part") = a->get_ns_part(), Named("basis") = a->get_basis());
// };
