
#ifndef __EA__MCMCOptions__
#define __EA__MCMCOptions__

#include <stdio.h>
#include <time.h>

#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/random_device.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

using namespace std;

#include <algorithm>
#include <fstream>
#include <iterator>
#include <vector>

struct MCMCOptions {
 public:
  MCMCOptions(vector<double> sOptions, vector<double> hOptions,
              vector<vector<double>> etaOptions, vector<double> t0Options,
              vector<double> th, double diff_thr, double brid_thr, double AAM,
              int AAP, int burn_In, int look_Back, int print_Counter, bool save,
              bool save_aux, bool save_likelihood, string selTP)
      : sigmaPriorMeans(sOptions[0]),
        sigmaPriorStdevs(sOptions[1]),
        sigmaBoundsLower(sOptions[2]),
        sigmaBoundsUpper(sOptions[3]),
        sigmaProposalSteps(sOptions[4]),
        sigmaMeanPrec(sOptions[5]),
        sigmaStdevPrec(sOptions[6]),
        t0PriorMeans(t0Options[0]),
        t0PriorStdevs(t0Options[1]),
        t0ProposalSteps(t0Options[2]),
        t0MeanPrec(t0Options[3]),
        t0StdevPrec(t0Options[4]),
        hPriorMeans(hOptions[0]),
        hPriorStdevs(hOptions[1]),
        hBoundsLower(hOptions[2]),
        hBoundsUpper(hOptions[3]),
        hProposalSteps(hOptions[4]),
        hMeanPrec(hOptions[5]),
        hStdevPrec(hOptions[6]),
        diffusion_threshold(diff_thr),
        bridge_threshold(brid_thr),
        AlleleAgeMargin(AAM),
        etaPriorMeans(etaOptions[0]),
        etaPriorStdevs(etaOptions[1]),
        etaBoundsLower(etaOptions[2]),
        etaBoundsUpper(etaOptions[3]),
        etaProposalSteps(etaOptions[4]),
        etaMeanPrec(etaOptions[5]),
        etaStdevPrec(etaOptions[6]),
        Theta(th),
        AlleleAgePrior(AAP),
        burnIn(burn_In),
        lookBack(look_Back),
        printCounter(print_Counter),
        Saving(save),
        Save_Aux(save_aux),
        Save_Likelihood(save_likelihood),
        selTypePriors(selTP) {
    if (Saving == 1) {
      OutputNamer();
    };
  };
  double sigmaPriorMeans, sigmaPriorStdevs, sigmaBoundsLower, sigmaBoundsUpper,
      sigmaProposalSteps, sigmaMeanPrec, sigmaStdevPrec, t0PriorMeans,
      t0PriorStdevs, t0ProposalSteps, t0MeanPrec, t0StdevPrec, hPriorMeans,
      hPriorStdevs, hBoundsLower, hBoundsUpper, hProposalSteps, hMeanPrec,
      hStdevPrec, diffusion_threshold, bridge_threshold, AlleleAgeMargin;
  vector<double> etaPriorMeans, etaPriorStdevs, etaBoundsLower, etaBoundsUpper,
      etaProposalSteps, etaMeanPrec, etaStdevPrec, Theta;
  int AlleleAgePrior, burnIn, lookBack, printCounter;
  bool Saving, Save_Aux, Save_Likelihood;
  string selTypePriors, FilenameSigma, FilenameT0, FilenameX, FilenameTheta,
      FilenameOmega, FilenamePsi, FilenameGamma, FilenameKsi,
      FilenameT0KsiCounter, FilenameDatapointsInput, FilenameOtherInput,
      FilenameSigMean, FilenameSigStd, FilenameT0Mean, FilenameT0Std,
      FilenameSigmaLikelihood, FilenameT0Likelihood, FilenameKappa,
      FilenamePseudo, FilenameH, FilenameEta, FilenameHMean, FilenameHStd,
      FilenameEtaMean, FilenameEtaStd;
  void OutputNamer();
};

#endif
