/*
 * BART: Bayesian Additive Regression Trees
 * Modifications by Jungang Zou, 2024.
 */

#ifndef ARMADILLO_H_
#define ARMADILLO_H_
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#endif

#ifndef BART_MODEL_H_
#define BART_MODEL_H_
#include "BART/tree.h"
#include "BART/treefuns.h"
#include "BART/info.h"
#include "BART/bartfuns.h"
#include "BART/bd.h"
#include "BART/bart.h"
#include "BART/heterbart.h"
#include <stdio.h>
#include "BART/cpwbart.h"
#include "BART/lambda.h"
#endif

#ifndef RCPP_H_
#define RCPP_H_
#include <Rcpp.h>
#endif

#ifndef PG_H_
#define PG_H_
#include <pg.h>
// [[Rcpp::depends(RcppArmadillo, pg)]]
#endif

using namespace Rcpp;

#define TRDRAW(a, b) trdraw(a, b)

// --- Helper: Check for NaNs ---
bool has_nan(NumericMatrix x) {
  for(int i = 0; i < x.size(); ++i) {
    if (Rcpp::traits::is_nan<REALSXP>(x[i])) return true;
  }
  return false;
}

class bart_model {
public:
  bart_model(){};
  
  bart_model(NumericMatrix x_train, NumericVector y_train, bool dart = false, bool aug = false, long numcut=100L, bool usequants = false, bool cont = false, bool rm_const = false, int ntrees = 300, Nullable<double> sigmaf = R_NilValue, double k = 2.0, double power = 2, double base = 0.95, double nu = 3){
    
    // Attempt to grab function from namespace
    Function bartModelMatrix = Environment::namespace_env("PSPI")["bartModelMatrix"];
    //Function bartModelMatrix = Environment::global_env()["bartModelMatrix"];
    
    this->usequants = usequants;
    this->cont = cont;
    this->rm_const = rm_const;
    this->alpha = base;
    this->mybeta = power;
    this->tree_object = List();
    sigma = 1;
    this->nu = nu;
    this->dart = dart;
    this->aug = aug;
    
    n = y_train.length();
    Rcpp::List temp = bartModelMatrix(clone(x_train), numcut, usequants, 7, rm_const, cont);
    
    // Prepare X (Transpose logic)
    NumericMatrix X_trans = transpose(as<NumericMatrix>(temp["X"]));
    
    // --- CRITICAL FIX 1: Store Data Permanently ---
    // Storing as Matrix allows .nrow() and contiguous memory access
    this->stored_xv = X_trans; 
    this->ix = &this->stored_xv[0];
    
    this->numcut = as<IntegerVector>(temp["numcut"]);
    NumericMatrix Xinfo = as<NumericMatrix>(temp["xinfo"]);
    
    if(n != X_trans.ncol())
      throw("The length of y_train and the number of rows in x_train must be identical");
    
    p = this->stored_xv.nrow(); // Safe to call .nrow() now
    
    // Sigma estimation
    double sigest = sd(y_train);
    
    if(p < n){
      arma::mat x_r(x_train.begin(), n, p, false);
      bool has_constant_column = false;
      if(!rm_const){
        for (int j = 0; j < p; ++j) {
          if (arma::all(x_r.col(j) == 1)) {
            has_constant_column = true;
            break;
          }
        }
        //Rcout<<has_constant_column << std::endl;
      }
      if(has_constant_column == false){
        arma::mat allOne(n, 1, arma::fill::ones);      
        x_r.insert_cols(0, allOne); 
      }
      arma::colvec y_r(y_train.begin(), y_train.size(), false);
      arma::colvec coef = arma::solve(x_r, y_r);       
      arma::colvec resid = y_r - x_r*coef;             
      
      double sig2 = arma::as_scalar( arma::trans(resid)*resid/(n-p));
      sigest = pow(sig2, 0.5);
      sigma = sigest;
    }
    NumericVector qch;
    qch.push_back(1 - 0.9);
    double qchi = Rcpp::qchisq(qch, nu, true, false)[0];
    this->lambda = (sigest * sigest * qchi) / nu;
    
    
    this->ntrees = ntrees;
    if(this->rm_const.length() == 0){
      this->rm_const = seq(1, p);
    }
    
    this->fmean = mean(y_train);
    
    // --- CRITICAL FIX 2: Store Y Permanently ---
    this->stored_yv = clone(y_train) - this->fmean;
    this->iy = &this->stored_yv[0];
    
    if(sigmaf.isNull()){
      tau = (max(this->stored_yv) - min(this->stored_yv))/(2*k*sqrt(ntrees));
    }else{
      this->sigmaf = as<double>(sigmaf) / sqrt(ntrees);
    }
    
    bm = heterbart(ntrees);
    
    if(Xinfo.size()>0) {
      xinfo xi_;
      xi_.resize(p);
      for(size_t i=0;i<p;i++) {
        xi_[i].resize(this->numcut[i]);
        for(size_t j=0;j<(size_t)this->numcut[i];j++) xi_[i][j]=Xinfo(i, j);
      }
      bm.setxinfo(xi_);
    }
    
    int *nc = &this->numcut[0];
    
    bm.setprior(alpha, mybeta, tau);
    bm.setdata(p, n, ix, iy, nc);
    double a = 0.5;
    double b = 1.0;
    double rho = p;
    double theta = 0;
    double omega = 1.0;
    bm.setdart(a,b,rho,aug,dart,theta,omega);
    //if(dart) bm.startdart();
  };
  
  // Update Signatures
  List update(long nburn, long npost, int skip, bool verbose = false, long print_every = 100L);
  List update(double sigma, long nburn, long npost, int skip, bool verbose = false, long print_every = 100L);
  List update(double sigma, NumericVector w, long nburn, long npost, int skip, bool verbose = false, long print_every = 100L);
  
  void set_data(NumericMatrix x_train, NumericVector y_train){
    n = y_train.length();
    this->fmean = mean(y_train);
    
    // --- DATA UPDATE FIX ---
    // 1. Update Y storage and pointer
    this->stored_yv = clone(y_train) - this->fmean;
    this->iy = &this->stored_yv[0];
    
    // 2. Update X storage and pointer
    this->stored_xv = transpose(clone(x_train));
    this->ix = &this->stored_xv[0];
    
    p = this->stored_xv.nrow();
    int *nc = &this->numcut[0];
    
    // 3. Pass VALID member pointers to heterbart
    bm.setdata(p, n, this->ix, this->iy, nc);
  };
  
  void set_Y(NumericVector y_train){
    n = y_train.length();
    
    // 1. Update Y
    this->fmean = mean(y_train);
    // Deep copy to member variable
    this->stored_yv = clone(y_train) - this->fmean;
    // Update internal pointer
    this->iy = &this->stored_yv[0];
    
    int *nc = &numcut[0];
    
    // 2. Update BART using existing X pointer (this->ix)
    // Since we updated 'this->ix' in set_data, this correctly uses the current X.
    bm.setdata(p, n, this->ix, this->iy, nc);
  };
  
  NumericMatrix predict(NumericMatrix x_predict, bool verbose = false){
    if(this->tree_object.length() == 0){
      return NumericMatrix();
    }
    NumericMatrix X = transpose(as<NumericMatrix>(clone(x_predict)));
    NumericMatrix predict_y = cpwbart(this->tree_object["treedraws"], X, verbose);
    return predict_y + this->fmean;
  };
  
  SEXP get_tree_object(){ return tree_object; }
  double get_sigma(){ return this->sigma; }
  bool get_usequants(){ return this->usequants; }
  double get_invchi(long n, double rss){
    return sqrt((nu*lambda + rss)/gen.chi_square(n+nu));
  }
  double get_lambda(){ return lambda; }
  double get_nu(){ return nu; }
  
  Rcpp::NumericMatrix get_varprob(){return varprob; }
  
  Rcpp::IntegerMatrix get_varcount(){return varcount;}
  
  void startdart(){
    if(dart)
      bm.startdart();
  }
  
private:
  // --- PERSISTENT STORAGE ---
  // Using Matrix for X allows valid .nrow() calls and contiguous storage
  Rcpp::NumericMatrix stored_xv; 
  Rcpp::NumericVector stored_yv;
  
  IntegerVector numcut;
  bool usequants;
  bool cont; 
  bool dart;
  bool aug;
  IntegerVector rm_const;
  
  long n;
  long p;
  long ntrees;
  double sigmaf;
  double tau;
  
  double *ix;
  double *iy;
  
  double alpha;
  double mybeta;
  double fmean;
  double sigma;
  double nu;
  double lambda;
  
  List tree_object;
  
  arn gen;
  heterbart bm; 
  
  Rcpp::NumericMatrix varprob;
  Rcpp::IntegerMatrix varcount;
};

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF UPDATE METHODS
// -----------------------------------------------------------------------------

// 1. Standard Update (Uses internal Sigma)
List bart_model::update(long nburn, long npost, int skip, bool verbose, long print_every){
  return update(this->sigma, nburn, npost, skip, verbose, print_every);
}

// 2. Sigma Update
List bart_model::update(double sigma, long nburn, long npost, int skip, bool verbose, long print_every){
  this->sigma = sigma;
  Rcpp::NumericVector trmean(n); 
  Rcpp::NumericMatrix trdraw(npost / skip, n);
  Rcpp::NumericMatrix varprb(npost / skip, p);
  Rcpp::IntegerMatrix varcnt(npost / skip, p);
  
  std::stringstream treess;  
  treess.precision(10);
  treess << npost / skip << " " << ntrees << " " << p << endl;
  
  std::vector<double> ivarprb (p, 0.);
  std::vector<size_t> ivarcnt (p, 0);
  
  if(verbose) printf("\nMCMC\n");
  
  size_t trcnt=0;
  size_t treedrawscnt=0; 
  bool keeptreedraw;
  xinfo& xi = bm.getxinfo();
  
  for(int i=0; i < nburn + npost; i++) {
    if(verbose && i % print_every == 0){
      printf("iteration %d", i);
      Rcout << "/" << nburn + npost << std::endl;
    }
    
    bm.draw(this->sigma, gen);
    
    // Update Sigma using Gibbs Step
    double restemp = 0, rss=0.0;
    for(size_t k=0; k<n; k++) {
      restemp = (iy[k] - bm.f(k)); 
      rss += restemp * restemp;
    }
    this->sigma = get_invchi(n, rss);
    
    if(i >= nburn) {
      for(size_t k=0; k<n; k++) trmean[k] += bm.f(k);
      keeptreedraw = npost && (((i-nburn+1) % skip) == 0);
      
      if(keeptreedraw) {
        for(long k=0; k<n; k++) TRDRAW(trcnt, k) = bm.f(k);
        trcnt += 1;
        for(size_t j=0; j<ntrees; j++) {
          treess << bm.gettree(j);
        }
        
#ifndef NoRcpp
        ivarcnt = bm.getnv();
        ivarprb = bm.getpv();
        size_t k_idx = (i-nburn)/skip;
        for(size_t j=0; j<p; j++){
          varcnt(k_idx, j) = ivarcnt[j];
          varprb(k_idx, j) = ivarprb[j];
        }
#else
        varcnt.push_back(bm.getnv());
        varprb.push_back(bm.getpv());
#endif
        
        treedrawscnt += 1;
      }
    }
  }
  
  for(size_t k=0; k<n; k++) trmean[k] /= npost;
  
  varprob = varprb;
  varcount = varcnt;
  
#ifndef NoRcpp
  Rcpp::List ret;
  ret["yhat.train.mean"] = trmean;
  ret["yhat.train"] = trdraw;
  ret["varcount"] = varcnt;
  ret["varprob"] = varprb;
  
  Rcpp::List xiret(xi.size());
  for(size_t i=0; i<xi.size(); i++) {
    Rcpp::NumericVector vtemp(xi[i].size());
    std::copy(xi[i].begin(), xi[i].end(), vtemp.begin());
    xiret[i] = Rcpp::NumericVector(vtemp);
  }
  
  Rcpp::List treesL;
  treesL["cutpoints"] = xiret;
  treesL["trees"] = Rcpp::CharacterVector(treess.str());
  
  ret["treedraws"] = treesL;
  ret["mu"] = fmean;
  ret["yhat.train.mean"] = trmean + fmean;
  ret["yhat.train"] = trdraw + fmean;
  ret["sigma"] = this->sigma;
  
  this->tree_object = ret;
  return ret;
#else
  return List::create();
#endif
}

// 3. Weighted Update
List bart_model::update(double sigma, NumericVector w, long nburn, long npost, int skip, bool verbose, long print_every){
  this->sigma = sigma;
  
  // FIX: Use std::vector to handle memory automatically (prevents leak)
  std::vector<double> svec(n);
  for(size_t i=0; i<n; i++) svec[i] = w[i] * sigma;
  
  Rcpp::NumericVector trmean(n); 
  Rcpp::NumericMatrix trdraw(npost / skip, n);
  Rcpp::NumericMatrix varprb(npost / skip, p);
  Rcpp::IntegerMatrix varcnt(npost / skip, p);
  
  std::stringstream treess;  
  treess.precision(10);
  treess << npost / skip << " " << ntrees << " " << p << endl;
  std::vector<double> ivarprb (p, 0.);
  std::vector<size_t> ivarcnt (p, 0);
  
  if(verbose) printf("\nMCMC\n");
  
  size_t trcnt=0;
  size_t treedrawscnt=0; 
  bool keeptreedraw;
  xinfo& xi = bm.getxinfo();
  
  for(int i=0; i < nburn + npost; i++) {
    if(verbose && i % print_every == 0){
      printf("iteration %d", i);
      Rcout << "/" << nburn + npost << std::endl;
    }
    
    // Pass the vector data pointer
    bm.draw(svec.data(), gen);
    
    if(i >= nburn) {
      for(size_t k=0; k<n; k++) trmean[k] += bm.f(k);
      keeptreedraw = npost && (((i-nburn+1) % skip) == 0);
      
      if(keeptreedraw) {
        for(long k=0; k<n; k++) TRDRAW(trcnt, k) = bm.f(k);
        trcnt += 1;
        for(size_t j=0; j<ntrees; j++) {
          treess << bm.gettree(j);
        }
        
#ifndef NoRcpp
        ivarcnt = bm.getnv();
        ivarprb = bm.getpv();
        size_t k_idx = (i-nburn)/skip;
        for(size_t j=0; j<p; j++){
          varcnt(k_idx, j) = ivarcnt[j];
          varprb(k_idx, j) = ivarprb[j];
        }
#else
        varcnt.push_back(bm.getnv());
        varprb.push_back(bm.getpv());
#endif
        
        treedrawscnt += 1;
      }
    }
  }
  
  // Standard Return logic
  for(size_t k=0; k<n; k++) trmean[k] /= npost;
  
  varprob = varprb;
  varcount = varcnt;
  
#ifndef NoRcpp
  Rcpp::List ret;
  ret["yhat.train.mean"] = trmean;
  ret["yhat.train"] = trdraw;
  ret["varcount"] = varcnt;
  ret["varprob"] = varprb;
  
  Rcpp::List xiret(xi.size());
  for(size_t i=0; i<xi.size(); i++) {
    Rcpp::NumericVector vtemp(xi[i].size());
    std::copy(xi[i].begin(), xi[i].end(), vtemp.begin());
    xiret[i] = Rcpp::NumericVector(vtemp);
  }
  
  Rcpp::List treesL;
  treesL["cutpoints"] = xiret;
  treesL["trees"] = Rcpp::CharacterVector(treess.str());
  
  ret["treedraws"] = treesL;
  ret["mu"] = fmean;
  ret["yhat.train.mean"] = trmean + fmean;
  ret["yhat.train"] = trdraw + fmean;
  ret["sigma"] = this->sigma;
  
  this->tree_object = ret;
  return ret;
#else
  return List::create();
#endif
};

// [[Rcpp::export]]
SEXP bart_train(NumericMatrix X, NumericVector Y, long nburn = 100, long npost = 1000, bool verbose = true){
  // 1. Allocate model
  bart_model * m = new bart_model(X, Y);
  
  // 2. Initial Burn-in
  List a = m->update(1, 1, 1, verbose);
  
  NumericMatrix y_pre = m->predict(X);
  NumericMatrix post_y(npost, Y.length());
  
  for(int i = 0; i < nburn + npost; ++i){
    // Pass m->get_sigma() so we don't reset sigma to 1.0 every step
    a = m->update(m->get_sigma(), 1, 1, 1, verbose, 200);
    
    y_pre = m->predict(X);
    m->set_Y(y_pre);
    if(i >= nburn)
      post_y(i-nburn, _) = y_pre;
  }
  
  // 3. CLEAN UP MEMORY
  delete m;
  
  return List::create(Named("y_pre") = y_pre, Named("post_y") = post_y);
}
