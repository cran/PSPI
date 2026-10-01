#ifndef ARMADILLO_H_
#define ARMADILLO_H_
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#endif

#ifndef DIST_H_
#define DIST_H_
#include <RcppDist.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
#endif


#ifndef BART_H_
#define BART_H_
#include "bart_model.h"
#endif


#ifndef RCPP_H_
#define RCPP_H_
#include <Rcpp.h>
#endif


using namespace Rcpp;



class BARTforPSPI{
public:
  BARTforPSPI(NumericMatrix X_, NumericVector Y_, bool binary_, IntegerVector Z_, NumericMatrix pi_, NumericMatrix X_test_, long ntrees_s = 200){
    //Rcout << 123 << std::endl;
    X = clone(X_);
    binary = binary_;
    if(binary){
      Y = as<NumericVector>(clone(Y_)) * 2 - 1;
    }else{
      Y = Y_;
    }
    Z = clone(Z_);
    this->Y_ = clone(Y_);
    pi = clone(pi_);
    
    J = pi_.cols(); 
    
   
    n = Y.length();
    this->ntrees_s = ntrees_s;
    this->X_test = X_test_;
    this->w = NumericVector(Y_.length()) + 1.0;
  }
  
  
  BARTforPSPI(NumericMatrix X_, NumericVector Y_, bool binary_, NumericMatrix pi_, NumericMatrix X_test_, long ntrees_s = 200){
    X = clone(X_);
    binary = binary_;
    if(binary){
      Y = as<NumericVector>(clone(Y_)) * 2 - 1;
    }else{
      Y = Y_;
    }
    this->Y_ = clone(Y_);
    pi = clone(pi_);
    J = 1; 
    n = Y.length();
    this->ntrees_s = ntrees_s;
    this->X_test = X_test_;
    this->w = NumericVector(Y_.length()) + 1.0;
  }
  
  
  
  virtual void update(bool verbose = false) = 0;
  virtual List predict(NumericMatrix pi_test) = 0;
  virtual List get_posterior() = 0;
  virtual void startdart() = 0;
  virtual void set_pi(NumericMatrix pi_)=0;
  virtual List get_serialized_state() = 0;
  virtual List get_spline_info() { return List::create(); }
  
  NumericMatrix get_X(){
    return X;
  };
  NumericVector get_Y(){
    return Y;
  };
  NumericMatrix get_pi(){
    return pi;
  };
  IntegerVector get_Z(){
    return Z;
  };
  
  void update_X(NumericMatrix X_new){
    X = X_new;
  }
  
  void update_Y(NumericVector Y_new){
    Y = Y_new;
  }
  
  void update_pi(NumericMatrix pi_new){
    pi = pi_new;
  }
  
  void update_Z(IntegerVector Z_new){
    Z = Z_new;
  }
  
  NumericMatrix cbind(NumericMatrix a, NumericVector b){
    int row = a.nrow();
    int col = a.ncol();
    
    NumericMatrix result(row, col+1);
    for(int i = 0 ; i < col; ++i){
      result(_, i) = a(_, i);
    }
    result(_, col) = b;
    return result;
  }
  
  NumericMatrix cbind(NumericMatrix a, NumericVector b, NumericVector c){
    int row = a.nrow();
    int col = a.ncol();
    
    NumericMatrix result(row, col+2);
    for(int i = 0 ; i < col; ++i){
      result(_, i) = a(_, i);
    }
    result(_, col) = b;
    result(_, col + 1) = c;
    return result;
  }
  

  NumericMatrix sliceRows(NumericMatrix mat, LogicalVector vec) {
      int n = mat.nrow();
      int m = mat.ncol();
      int count = sum(vec);
      
      // Create a new NumericMatrix to store the result
      NumericMatrix result(count, m);
      
      // Fill the result matrix with the selected rows
      int rowIndex = 0;
      for (int i = 0; i < n; i++) {
        if (vec[i]) {
          for (int j = 0; j < m; j++) {
            result(rowIndex, j) = mat(i, j);
          }
          rowIndex++;
        }
      }
      return result;
  }
  NumericVector logit(NumericVector x){
    return log(x / (1-x));
  }
  
  NumericVector treatment_slice(NumericMatrix pi_, NumericVector Z_) {
    int n = pi_.nrow();
    int p = pi_.ncol();
    
    // Basic error checking
    if (Z_.size() != n) {
      stop("Length of index vector must match the number of rows.");
    }
    
    NumericVector result(n);
    
    for (int i = 0; i < n; ++i) {
      // Cast the numeric index to an integer
      int col = (int)Z_[i];
      
      // Bounds check to prevent crashing R
      if (col < 0 || col >= p) {
        stop("Column index out of bounds at row %d", i + 1);
      }
      
      result[i] = pi_(i, col);
    }
    
    return result;
  }
  
  // Deep-copy a bart_model's tree_object for safe serialization
  List deep_copy_tree_object(bart_model* bm){
    List to = as<List>(bm->get_tree_object());
    List td = as<List>(to["treedraws"]);
    CharacterVector trees_str = clone(as<CharacterVector>(td["trees"]));
    List cp = clone(as<List>(td["cutpoints"]));
    List td_copy = List::create(Named("trees") = trees_str, Named("cutpoints") = cp);
    double mu = as<double>(to["mu"]);
    return List::create(Named("treedraws") = td_copy, Named("mu") = mu);
  }

  List get_var_split(){
    List var_split = List::create();
    for(int j = 0 ; j < bart.size(); ++j){
      //Rcout << bart[j]->get_varcount() << std::endl;
      IntegerVector varcount = bart[j]->get_varcount()(0,_);;
      NumericVector varprob = bart[j]->get_varprob()(0,_);
      var_split.push_back(List::create(Named("varcount") = varcount, Named("varprob") = varprob));
    }
    return var_split;
  }
  
  
  
protected:
  NumericMatrix X;
  NumericMatrix X_test;
  NumericVector Y;
  LogicalVector Y_;
  IntegerVector Z;
  NumericMatrix pi;
  NumericVector w;
  std::vector<bart_model*> bart;
  long n;
  long ntrees_s;
  arn gen;
  bool binary;
  int J; // number of treatments
};
