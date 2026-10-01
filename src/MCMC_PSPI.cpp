#ifndef ARMADILLO_H_
#define ARMADILLO_H_
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#endif

#ifndef CBART_H_
#define CBART_H_
#include "BARTforPSPI.h"
#endif

#ifndef PBART_H_
#define PBART_H_
#include "pbart_model.h"
#endif


#include "BCF.h"
#include "PSPI_BCF_P.h"
#include "PSPI_FullBART.h"
#include "PSPI_SplineBART.h"
#include "PSPI_MSplineBART.h"
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>

#ifndef RCPP_H_
#define RCPP_H_
#include <Rcpp.h>
#endif


using namespace Rcpp;

// [[Rcpp::export]]
List MCMC_PSPI_generalizability(NumericMatrix X, NumericVector Y, bool binary, IntegerVector Z, NumericMatrix pi, NumericMatrix X_test, NumericMatrix pi_test, int model, long nburn, long npost, IntegerVector n_knots, IntegerVector order, int ntrees_s = 200, bool verbose = false, bool dart = false, bool aug = false){
  //Rcout << dart << " " << aug << std::endl;
  
  BARTforPSPI * cmodel;
  NumericMatrix X_ = clone(X);
  NumericVector Y_ = clone(Y);
  IntegerVector Z_ = clone(Z);
  NumericMatrix pi_ = clone(pi);
  
  NumericMatrix X_test_ = clone(X_test);
  NumericMatrix pi_test_ = clone(pi_test);
  //pi_ = log(pi_ / (1 - pi_)); //1 / (1 + exp(-1 * pi_));
  //pi_test_ = log(pi_test_ / (1 - pi_test_)); //1 / (1 + exp(-1 * pi_test_));
  switch (model){
  case 1:
    //cmodel = new vanillaBART(X_, Y_, Z_, pi_, X_test_, binary, logistic, ntrees_s);
    break;
  case 2:
    cmodel = new BCF(X_, Y_, binary, Z_, pi_, X_test_, ntrees_s, dart, aug);
    break;
  case 3:
    cmodel = new PSPI_BCF_P(X_, Y_, binary, Z_, pi_, X_test_, ntrees_s, dart, aug);
    break;
  case 4:
    cmodel = new PSPI_FullBART(X_, Y_, binary, Z_, pi_, X_test_, ntrees_s, dart, aug);
    break;
  case 5:
    cmodel = new PSPI_SplineBART(X_, Y_, binary, Z_, pi_, X_test_, n_knots, order, ntrees_s, dart, aug);
    break;
  case 6:
    cmodel = new PSPI_MSplineBART(X_, Y_, binary, Z_, pi_, X_test_, n_knots, order, ntrees_s, dart, aug);
    break;
  }
  long n = X_test_.nrow();
  long p_ = X_test_.ncol();
  int J = pi.cols();
  NumericVector post_sigma(npost);

  NumericMatrix post_outcome_train(npost, pi_.rows());
  
  List post_outcome;
  List post_interaction;
  List post_prob;
  List BART = List::create();
  for(int j = 0; j < J; ++j){
    NumericMatrix predict(npost, pi_test_.rows());
    IntegerMatrix varcount(npost, p_);
    NumericMatrix varprob(npost, p_);
    BART.push_back(List::create(Named("predict") = predict, Named("varcount") = varcount, Named("varprob") = varprob));
  }
  
  List beta;
  List gamma2;
  List post_spline_train;
  List post_spline_test;
  // List spline = List::create();
  // for(int j = 0; j < J; ++j){
  //   NumericMatrix predict(npost, pi_test_.rows());
  //   IntegerMatrix varcount(npost, p_);
  //   NumericMatrix varprob(npost, p_);
  //   BART.push_back(List::create(Named("predict_BART") = predict_BART, Named("varcount") = varcount, Named("varprob") = varprob));
  // }
  // 
  // 
  // 
  // NumericMatrix beta_main(npost, n_knots_main);
  // NumericMatrix beta_inter(npost, n_knots_inter);
  // 
  // NumericVector gamma2_main(npost);
  // NumericVector gamma2_inter(npost);
  // 
  
  //NumericMatrix cbart_pop(npost, pi_test_.rows());
  //Rcout << 123 << std::endl;
  Progress pr(nburn + npost, !verbose);
  for(int i = 0 ; i < nburn + npost; ++i){
    if(Progress::check_abort())
      return -1.0;
    pr.increment();
    if(verbose)
      Rcout << i << " " << nburn + npost << std::endl;
    if(i==(nburn/2)&&dart) cmodel->startdart();
    cmodel->update(verbose);
    
    if(i >= nburn){
      //Rcout << "posterior sampling" << std::endl;
      List posterior = cmodel->get_posterior();
      //Rcout << "posterior sampling" << std::endl;
      List var_split = cmodel->get_var_split();
      //Rcout << "posterior sampling" << std::endl;
      List predict_outcome = cmodel->predict(pi_test_);
      //Rcout << "posterior sampling" << std::endl;
      post_outcome.push_back(predict_outcome["outcome"]);
      post_interaction.push_back(predict_outcome["inter_model"]);
      //Rcout << "posterior sampling" << std::endl;
      if(model == 5 || model == 6){
        beta.push_back(posterior["beta"]);
        gamma2.push_back(posterior["gamma"]);
        post_spline_train.push_back(posterior["spline_train"]);
        post_spline_test.push_back(predict_outcome["spline_test"]);
      }
      
      // 
      // predict_s_BART(i - nburn, _) = as<NumericVector>(predict_outcome["inter_BART"]);
      // 
      // varcount_s(i - nburn, _) = as<NumericVector>(var_split["varcount_inter"]);
      // varcount_h(i - nburn, _) = as<NumericVector>(var_split["varcount_main"]);
      // varprob_s(i - nburn, _) = as<NumericVector>(var_split["varprob_inter"]);
      // varprob_h(i - nburn, _) = as<NumericVector>(var_split["varprob_main"]);

      
      // if(model == 5 || model == 6){
      //   predict_s(i - nburn, _) = as<NumericVector>(predict_outcome["predict_s"]);
      //   beta_inter(i - nburn, _) = as<NumericVector>(posterior["inter_beta"]);
      //   gamma2_inter[i - nburn] = pow((double)posterior["inter_gamma"], 2);
      //   
      //   if(model == 6){
      //     predict_h(i - nburn, _) = (as<NumericVector>(predict_outcome["predict_h"]));
      //     beta_main(i - nburn, _) = as<NumericVector>(posterior["main_beta"]);
      //     gamma2_main[i - nburn] = pow((double)posterior["main_gamma"], 2);
      //     predict_h_BART(i - nburn, _) = as<NumericVector>(predict_outcome["main_BART"]);
      //   }
      // }
      
      if(binary){
        post_outcome_train(i - nburn, _) = Rcpp::pnorm(as<NumericVector>(posterior["outcome_train"]));
        post_prob.push_back(predict_outcome["outcome_prob"]);
        
      }else{
        post_outcome_train(i - nburn, _) = (as<NumericVector>(posterior["outcome_train"]));
      }
      
      //post_te(i - nburn, _) = (post_outcome1(i - nburn, _) - post_outcome0(i - nburn, _));
      post_sigma[i - nburn] = posterior["sigma"];

      // if(model == 4 || model == 5 || model == 6){
      //   
      // }
      // if(model == 5 || model == 6 || model == 7 || model == 8){
      //   predict_s(i - nburn, _) = as<NumericVector>(predict_outcome["predict_s"]);
      //   cbart_pop(i - nburn, _) = as<NumericVector>(predict_outcome["cbart_pop"]);
      //   cbart_pre(i - nburn, _) = as<NumericVector>(posterior["cbart_pre"]);
      //   
      //   ns_beta(i - nburn, _) = as<NumericVector>(posterior["ns_beta"]);
      //   
      //   //post_beta(i - nburn, 0) = as<NumericVector>(posterior["ns_beta"])[0];
      //   //post_beta(i - nburn, 1) = as<NumericVector>(posterior["ns_beta"])[1];
      //   //post_gamma(i - nburn) = as<double>(posterior["gamma"]);
      // }
      // if(model == 6){
      //   predict_h(i - nburn, _) =  as<NumericVector>(predict_outcome["predict_h"]);
      //   //post_beta_main(i - nburn, _) = as<NumericVector>(posterior["ns_beta_main"]);
      // }
    }
  }
  return List::create(Named("post_outcome_train") = post_outcome_train, Named("post_outcome") = post_outcome, Named("post_interaction") = post_interaction, Named("post_prob") = post_prob, Named("post_sigma") = post_sigma, Named("post_beta") = beta, Named("post_gamma2") = gamma2, Named("post_spline_train") = post_spline_train, Named("post_spline_test") = post_spline_test);
  // if(model == 5 || model == 6){
  //   spline_inter["post_spline_population"] = predict_s;
  //   spline_inter["post_spline_coefficients"] = beta_inter;
  //   spline_inter["post_spline_gamma2"] = gamma2_inter;
  //   
  //   if(model == 6){
  //     spline_main["post_spline_population"] = predict_h;
  //     spline_main["post_spline_coefficients"] = beta_main;
  //     spline_main["post_spline_gamma2"] = gamma2_main;
  //     
  //     BART_main["post_BART_population"] = predict_h_BART;
  //   }
  // }
  // 
  // BART_inter["post_BART_population"] = predict_s_BART;
  // BART_inter["post_varcount"] = varcount_s;
  // BART_inter["post_varprob"] = varprob_s;
  // 
  // BART_main["post_varcount"] = varcount_h;
  // BART_main["post_varprob"] = varprob_h;
  // 
  // 
  // List results = List::create(Named("BART_main") = BART_main, Named("BART_inter") = BART_inter, Named("post_outcome_train") = post_outcome_train, Named("post_outcome1") = post_outcome1, Named("post_outcome0") = post_outcome0, Named("post_te") = post_te, Named("post_sigma") = post_sigma);
  // if(binary){
  //   results["post_outcome1_prob"] = post_outcome1_prob;
  //   results["post_outcome0_prob"] = post_outcome0_prob;
  // }
  // if(model == 5 || model == 6){
  //   results["spline_inter"] = spline_inter;
  //   if(model == 6){
  //     results["spline_main"] = spline_main;
  //   }
  // }
  // return results;
}



// ============================================================================
// PSPI_fit: fit-only MCMC (no population data needed)
// ============================================================================
// [[Rcpp::export]]
List MCMC_PSPI_fit(NumericMatrix X, NumericVector Y, bool binary, IntegerVector Z, NumericMatrix pi, int model, long nburn, long npost, IntegerVector n_knots, IntegerVector order, int ntrees_s = 200, bool verbose = false, bool dart = false, bool aug = false){

  BARTforPSPI * cmodel;
  NumericMatrix X_ = clone(X);
  NumericVector Y_ = clone(Y);
  IntegerVector Z_ = clone(Z);
  NumericMatrix pi_ = clone(pi);
  NumericMatrix X_test_dummy(1, X_.ncol());

  switch (model){
  case 2:
    cmodel = new BCF(X_, Y_, binary, Z_, pi_, X_test_dummy, ntrees_s, dart, aug);
    break;
  case 3:
    cmodel = new PSPI_BCF_P(X_, Y_, binary, Z_, pi_, X_test_dummy, ntrees_s, dart, aug);
    break;
  case 4:
    cmodel = new PSPI_FullBART(X_, Y_, binary, Z_, pi_, X_test_dummy, ntrees_s, dart, aug);
    break;
  case 5:
    cmodel = new PSPI_SplineBART(X_, Y_, binary, Z_, pi_, X_test_dummy, n_knots, order, ntrees_s, dart, aug);
    break;
  case 6:
    cmodel = new PSPI_MSplineBART(X_, Y_, binary, Z_, pi_, X_test_dummy, n_knots, order, ntrees_s, dart, aug);
    break;
  default:
    stop("Invalid model type");
  }

  int J = pi.cols();
  NumericVector post_sigma(npost);
  NumericMatrix post_outcome_train(npost, pi_.rows());

  List all_bart_draws(npost);
  List all_bart_pi_draws;
  if(model == 4) all_bart_pi_draws = List(npost);
  NumericMatrix all_bart_pre_mean;
  if(model == 4){
    all_bart_pre_mean = NumericMatrix(npost, (J > 1) ? (J - 1) : 1);
  }

  List all_spline_theta;
  if(model == 5 || model == 6) all_spline_theta = List(npost);

  List spline_info;
  if(model == 5 || model == 6){
    spline_info = cmodel->get_spline_info();
  }

  Progress pr(nburn + npost, !verbose);
  for(int i = 0; i < nburn + npost; ++i){
    if(Progress::check_abort())
      return -1.0;
    pr.increment();
    if(verbose)
      Rcout << i << " " << nburn + npost << std::endl;
    if(i == (nburn/2) && dart) cmodel->startdart();
    cmodel->update(verbose);

    if(i >= nburn){
      int idx = i - nburn;
      List state = cmodel->get_serialized_state();

      all_bart_draws[idx] = state["bart_states"];
      if(model == 4) all_bart_pi_draws[idx] = state["bart_pi_states"];
      if(model == 4){
        NumericVector bpm = state["bart_pre_mean"];
        all_bart_pre_mean(idx, _) = bpm;
      }
      if(model == 5 || model == 6) all_spline_theta[idx] = state["spline_theta"];
      post_sigma[idx] = as<double>(state["sigma"]);

      List posterior = cmodel->get_posterior();
      if(binary){
        post_outcome_train(idx, _) = Rcpp::pnorm(as<NumericVector>(posterior["outcome_train"]));
      } else {
        post_outcome_train(idx, _) = as<NumericVector>(posterior["outcome_train"]);
      }
    }
  }
  // Note: not deleting cmodel to avoid dangling references in R Lists
  // delete cmodel;

  List result = List::create(
    Named("model_type") = model,
    Named("J") = J,
    Named("binary") = binary,
    Named("npost") = npost,
    Named("bart_draws") = all_bart_draws,
    Named("post_sigma") = post_sigma,
    Named("post_outcome_train") = post_outcome_train
  );
  if(model == 4){
    result["bart_pi_draws"] = all_bart_pi_draws;
    result["bart_pre_mean"] = all_bart_pre_mean;
  }
  if(model == 5 || model == 6){
    result["spline_theta"] = all_spline_theta;
    result["spline_info"] = spline_info;
  }
  return result;
}

// ============================================================================
// PSPI_predict: predict from serialized fit object (no trial data needed)
// ============================================================================

// Helper: extract row 0 from cpwbart result (1-draw case)
static NumericVector cpwbart_one(List treedraws, NumericMatrix X_t){
  NumericMatrix pred = cpwbart(wrap(treedraws), wrap(X_t), false);
  return pred(0, _);
}

// Helper: transpose(cbind(X, v)) without forming the un-transposed intermediate
static NumericMatrix transpose_cbind_vec(NumericMatrix X_t, NumericVector v){
  int p = X_t.nrow();
  int N = X_t.ncol();
  NumericMatrix out(p + 1, N);
  for(int i = 0; i < p; ++i) out(i, _) = X_t(i, _);
  out(p, _) = v;
  return out;
}

// Helper: vector -> 1 x N transposed matrix
static NumericMatrix vec_to_t(NumericVector v){
  NumericMatrix out(1, v.size());
  out(0, _) = v;
  return out;
}

// [[Rcpp::export]]
List MCMC_PSPI_predict(List fit, NumericMatrix X_pop, NumericMatrix pi_pop, bool verbose = false){

  int model_type = as<int>(fit["model_type"]);
  int J = as<int>(fit["J"]);
  bool binary = as<bool>(fit["binary"]);
  int npost = as<int>(fit["npost"]);
  List bart_draws = as<List>(fit["bart_draws"]);
  NumericVector post_sigma = as<NumericVector>(fit["post_sigma"]);

  long N = X_pop.nrow();

  // Pre-transpose once
  NumericMatrix X_pop_t = transpose(clone(X_pop));
  NumericVector pi0_pop = pi_pop(_, 0);
  NumericMatrix X_pop_pi0_t = transpose_cbind_vec(X_pop_t, pi0_pop);

  // BCF_P: pre-compute cbind(X,pi_j) transposed per arm
  std::vector<NumericMatrix> X_pop_pij_t;
  if(model_type == 3 && J > 1){
    for(int j = 1; j < J; ++j)
      X_pop_pij_t.push_back(transpose_cbind_vec(X_pop_t, pi_pop(_, j)));
  }

  // FullBART: pi_j as 1xN
  std::vector<NumericMatrix> pi_j_t;
  if(model_type == 4){
    if(J > 1){
      for(int j = 1; j < J; ++j) pi_j_t.push_back(vec_to_t(pi_pop(_, j)));
    } else {
      pi_j_t.push_back(vec_to_t(pi_pop(_, 0)));
    }
  }

  // Spline bases from saved knots
  std::vector<NS_basis*> spline_bases;
  if(model_type == 5 || model_type == 6){
    List spline_info = as<List>(fit["spline_info"]);
    for(int j = 0; j < spline_info.size(); ++j){
      List si = as<List>(spline_info[j]);
      NumericVector knots = as<NumericVector>(si["knots"]);
      spline_bases.push_back(new NS_basis(knots, 3, false));
    }
  }

  // Pre-compute spline basis matrices (don't change per draw)
  std::vector<arma::mat> spline_basis_mats;
  if(model_type == 5){
    if(J > 1){
      for(int j = 1; j < J; ++j)
        spline_basis_mats.push_back(as<arma::mat>(spline_bases[j-1]->predict(pi_pop(_, j))));
    } else {
      spline_basis_mats.push_back(as<arma::mat>(spline_bases[0]->predict(pi_pop(_, 0))));
    }
  } else if(model_type == 6){
    for(int j = 0; j < J; ++j)
      spline_basis_mats.push_back(as<arma::mat>(spline_bases[j]->predict(pi_pop(_, j))));
  }

  NumericMatrix all_bart_pre_mean;
  if(model_type == 4)
    all_bart_pre_mean = as<NumericMatrix>(fit["bart_pre_mean"]);
  List all_spline_theta;
  if(model_type == 5 || model_type == 6)
    all_spline_theta = as<List>(fit["spline_theta"]);
  List all_bart_pi_draws;
  if(model_type == 4)
    all_bart_pi_draws = as<List>(fit["bart_pi_draws"]);

  List post_outcome;
  List post_interaction;
  List post_prob;

  Progress pr(npost, !verbose);
  for(int m = 0; m < npost; ++m){
    if(Progress::check_abort()) return -1.0;
    pr.increment();

    List bsm = as<List>(bart_draws[m]);
    double sigma_m = post_sigma[m];

    NumericMatrix outcome(N, J);
    NumericMatrix inter_model(N, J > 1 ? J - 1 : 0);
    NumericMatrix outcome_hidden(N, J);

    if(model_type == 2){
      // BCF
      List s0 = as<List>(bsm[0]);
      outcome(_, 0) = cpwbart_one(s0["treedraws"], X_pop_pi0_t) + as<double>(s0["mu"]);
      for(int j = 1; j < J; ++j){
        List sj = as<List>(bsm[j]);
        inter_model(_, j-1) = cpwbart_one(sj["treedraws"], X_pop_t) + as<double>(sj["mu"]);
        outcome(_, j) = outcome(_, 0) + inter_model(_, j-1);
      }
    } else if(model_type == 3){
      // BCF_P
      List s0 = as<List>(bsm[0]);
      outcome(_, 0) = cpwbart_one(s0["treedraws"], X_pop_pi0_t) + as<double>(s0["mu"]);
      for(int j = 1; j < J; ++j){
        List sj = as<List>(bsm[j]);
        inter_model(_, j-1) = cpwbart_one(sj["treedraws"], X_pop_pij_t[j-1]) + as<double>(sj["mu"]);
        outcome(_, j) = outcome(_, 0) + inter_model(_, j-1);
      }
    } else if(model_type == 4){
      // FullBART
      List bpsm = as<List>(all_bart_pi_draws[m]);
      NumericVector bpm = all_bart_pre_mean(m, _);
      if(J > 1){
        List s0 = as<List>(bsm[0]);
        outcome(_, 0) = cpwbart_one(s0["treedraws"], X_pop_pi0_t) + as<double>(s0["mu"]);
        for(int j = 1; j < J; ++j){
          List sj = as<List>(bsm[j]);
          NumericVector ij = cpwbart_one(sj["treedraws"], X_pop_t) + as<double>(sj["mu"]) - bpm[j-1];
          List spj = as<List>(bpsm[j-1]);
          ij = ij + cpwbart_one(spj["treedraws"], pi_j_t[j-1]) + as<double>(spj["mu"]);
          inter_model(_, j-1) = ij;
          outcome(_, j) = outcome(_, 0) + ij;
        }
      } else {
        List s0 = as<List>(bsm[0]);
        List sp0 = as<List>(bpsm[0]);
        outcome(_, 0) = cpwbart_one(s0["treedraws"], X_pop_t) + as<double>(s0["mu"]) - bpm[0]
                       + cpwbart_one(sp0["treedraws"], pi_j_t[0]) + as<double>(sp0["mu"]);
      }
    } else if(model_type == 5){
      // SplineBART (no BART centering, no NS intercept)
      List stm = as<List>(all_spline_theta[m]);
      if(J > 1){
        List s0 = as<List>(bsm[0]);
        outcome(_, 0) = cpwbart_one(s0["treedraws"], X_pop_pi0_t) + as<double>(s0["mu"]);
        for(int j = 1; j < J; ++j){
          List sj = as<List>(bsm[j]);
          NumericVector ij = cpwbart_one(sj["treedraws"], X_pop_t) + as<double>(sj["mu"]);
          arma::vec sp = spline_basis_mats[j-1] * as<arma::vec>(as<NumericVector>(stm[j-1]));
          ij = ij + as<NumericVector>(wrap(sp));
          inter_model(_, j-1) = ij;
          outcome(_, j) = outcome(_, 0) + ij;
        }
      }
    } else if(model_type == 6){
      // MSplineBART / DSplineBART (no BART centering, no NS intercept)
      List stm = as<List>(all_spline_theta[m]);
      if(J > 1){
        List s0 = as<List>(bsm[0]);
        arma::vec sp0 = spline_basis_mats[0] * as<arma::vec>(as<NumericVector>(stm[0]));
        outcome(_, 0) = cpwbart_one(s0["treedraws"], X_pop_t) + as<double>(s0["mu"])
                       + as<NumericVector>(wrap(sp0));
        for(int j = 1; j < J; ++j){
          List sj = as<List>(bsm[j]);
          NumericVector ij = cpwbart_one(sj["treedraws"], X_pop_t) + as<double>(sj["mu"]);
          arma::vec spj = spline_basis_mats[j] * as<arma::vec>(as<NumericVector>(stm[j]));
          ij = ij + as<NumericVector>(wrap(spj));
          inter_model(_, j-1) = ij;
          outcome(_, j) = outcome(_, 0) + ij;
        }
      }
    }

    // Add observation noise
    if(binary){
      for(int i = 0; i < N; ++i){
        for(int j = 0; j < J; ++j){
          outcome_hidden(i, j) = R::pnorm(outcome(i, j), 0, 1, true, false);
          outcome(i, j) = R::rbinom(1, outcome_hidden(i, j));
        }
      }
    } else {
      for(int i = 0; i < N; ++i){
        for(int j = 0; j < J; ++j){
          outcome(i, j) = outcome(i, j) + R::rnorm(0, sigma_m);
        }
      }
    }

    post_outcome.push_back(clone(outcome));
    post_interaction.push_back(clone(inter_model));
    if(binary) post_prob.push_back(clone(outcome_hidden));
  }

  for(int j = 0; j < (int)spline_bases.size(); ++j) delete spline_bases[j];

  return List::create(
    Named("post_outcome") = post_outcome,
    Named("post_interaction") = post_interaction,
    Named("post_prob") = post_prob
  );
}
