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
  BARTforPSPI(NumericMatrix X_, NumericVector Y_, NumericVector Z_, NumericVector pi_, NumericMatrix X_test_, long ntrees_s = 200){
    X = clone(X_);
    Y = Y_;
    Z = clone(Z_);
    this->Y_ = clone(Y_);
    pi = clone(pi_);
    main_bart = NULL;
    n = Y.length();
    this->ntrees_s = ntrees_s;
    this->X_test = X_test_;
    this->w = NumericVector(Y_.length()) + 1.0;
  }
  
  
  BARTforPSPI(NumericMatrix X_, NumericVector Y_, NumericVector pi_, NumericMatrix X_test_, long ntrees_s = 200){
    X = clone(X_);
    Y = Y_;
    this->Y_ = clone(Y_);
    pi = clone(pi_);
    main_bart = NULL;
    n = Y.length();
    this->ntrees_s = ntrees_s;
    this->X_test = X_test_;
    this->w = NumericVector(Y_.length()) + 1.0;
  }
  
  
  virtual void update(bool verbose = false) = 0;
  virtual List predict(NumericVector pi_test) = 0;
  virtual List get_posterior() = 0;
  
  NumericMatrix get_X(){
    return X;
  };
  NumericVector get_Y(){
    return Y;
  };
  NumericVector get_pi(){
    return pi;
  };
  NumericVector get_Z(){
    return Z;
  };
  
  void update_X(NumericMatrix X_new){
    X = X_new;
  }
  
  void update_Y(NumericVector Y_new){
    Y = Y_new;
  }
  
  void update_pi(NumericVector pi_new){
    pi = pi_new;
  }
  
  void update_Z(LogicalVector Z_new){
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
  
protected:
  NumericMatrix X;
  NumericMatrix X_test;
  NumericVector Y;
  LogicalVector Y_;
  NumericVector Z;
  NumericVector pi;
  NumericVector w;
  bart_model * main_bart;
  long n;
  long ntrees_s;
  arn gen;
};
