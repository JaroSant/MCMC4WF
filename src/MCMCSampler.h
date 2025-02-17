
#ifndef __EA__MCMCSampler__
#define __EA__MCMCSampler__

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace boost::multiprecision;  // exp and pow use argument lookup to
                                        // determine which version - this or
                                        // std::

#include "MCMCOptions.h"
#include "WrightFisher.h"

// typedef boost::multiprecision::cpp_dec_float_100 double100;
// typedef double100 double100; // For debugging

class MCMCSampler {
 public:
  // CONSTRUCTOR

  MCMCSampler(vector<int> Datainput, vector<int> samples,
              vector<double100> Ttimes, int selType)
      : Data(Datainput),
        Samplesizes(samples),
        times(Ttimes),
        selectionType(selType) {
    XStore.push_back(Converter(Data, Samplesizes));
  }

  // USEFUL FUNCTIONS

  vector<double100> Converter(vector<int> Data, vector<int> SampleSize);
  double100 NormCDF(double100 x, double m, double v);
  double100 NormPDF(double100 x, double m, double v);
  double100 DiscreteNormalCDF(double100 x, double100 t, vector<double> thetaP);
  int computeC(vector<double> th, int m, double100 t);
  int computeE(vector<double> ths, double100 t);
  int computeK(double100 x, double100 z, vector<double> theta,
               const Options &o);
  double100 Getd1(int i, double100 x, double100 z, double100 t);
  double100 TransitionRatioApprox(WrightFisher Current, double100 xNum,
                                  double100 xDen, double100 zNum,
                                  double100 zDen, double100 tNum,
                                  double100 tDen, const Options &o);
  double100 TransitionApprox(WrightFisher Current, double100 x, double100 z,
                             double100 t, const Options &o);
  double100 selPriorRatio(double100 sigmaProp, double100 hProp,
                          vector<double100> etaProp, double100 sigmaCurr,
                          double100 hCurr, vector<double100> etaCurr,
                          const MCMCOptions &MCMCo);
  double100 selProposalRatio(double100 sigmaProp, double100 hProp,
                             vector<double100> etaProp, double100 sigmaCurr,
                             double100 hCurr, vector<double100> etaCurr,
                             const MCMCOptions &MCMCo);
  double100 t0PriorRatio(double100 t0Prop, double100 t0Curr,
                         const MCMCOptions &MCMCo);
  double100 t0ProposalRatio(double100 t0Prop, double100 t0Curr,
                            const MCMCOptions &MCMCo);
  double100 PoissonEstimator(WrightFisher Diffusion, double100 t, double100 x,
                             double100 y, const Options &o,
                             boost::random::mt19937 &gen);

  // PRINTING FUNCTIONS

  void printDatapoints(ofstream &outputFile);
  void printOtherInput(ofstream &outputFile, const MCMCOptions &MCMCo);
  void printVDouble(ofstream &outputFile, vector<double100> vec, int N);
  void printMatrixDouble(ofstream &outputFile, vector<vector<double100>> vec,
                         int N);
  void print3DMatrixDouble(ofstream &outputFile,
                           vector<vector<vector<double100>>> vec, int N);
  void printVInt(ofstream &outputFile, vector<int> vec, int N);
  void printMatrixInt(ofstream &outputFile, vector<vector<int>> vec, int N);
  void print3DMatrixInt(ofstream &outputFile, vector<vector<vector<int>>> vec,
                        int N);

  // INITIALISING FUNCTIONS

  void SetMutation(const MCMCOptions &MCMCo);
  void InitialiseSel(const MCMCOptions &MCMCo, boost::random::mt19937 &gen);
  void InitialiseSkeleton(WrightFisher Current, const Options &o,
                          boost::random::mt19937 &gen);
  void InitialiseT0(const MCMCOptions &MCMCo, boost::random::mt19937 &gen);
  void InitialisePseudo(WrightFisher Current, const Options &o,
                        boost::random::mt19937 &gen);

  // PROPOSAL MECHANISM FUNCTIONS

  pair<double100, vector<vector<double100>>>
  NonNeutralBridgePointBinomialObservations(
      WrightFisher Current, double100 x, double100 t1, double100 t2,
      double100 z, const Options &o, boost::random::mt19937 &gen,
      double100 testT);  //, int Yt1, int nt1);
  pair<double100, vector<vector<double100>>>
  NonNeutralDrawEndpointBinomialObservations(WrightFisher Current, double100 x,
                                             double100 t1, double100 t2,
                                             const Options &o,
                                             boost::random::mt19937 &gen);
  double100 t0Proposal(const MCMCOptions &MCMCo, boost::random::mt19937 &gen);
  vector<double100> selProposal(const MCMCOptions &MCMCo,
                                boost::random::mt19937 &gen);
  pair<bool, vector<double100>> TransitionRatioDecision(
      WrightFisher Current, double100 alpha, double100 u, double100 XNumstart,
      double100 XNumend, double100 XDenstart, double100 XDenend,
      double100 tNumstart, double100 tDenstart, double100 tNumend,
      double100 tDenend, const Options &o);
  void GenerateT0Z0Proposals(WrightFisher Current, double100 &t0Prop,
                             vector<double100> &XProp, vector<int> &kappaProp,
                             vector<vector<double100>> &omegaProp,
                             vector<vector<double100>> &psiProp,
                             vector<vector<double100>> &gammaProp,
                             vector<vector<double100>> &ksiProp,
                             double100 lambdaMax, const Options &o,
                             boost::random::mt19937 &gen);
  void AcceptT0Z0Proposals(
      WrightFisher Current, double100 &t0Prop, vector<double100> &XProp,
      vector<int> &kappaProp, vector<vector<double100>> &omegaProp,
      vector<vector<double100>> &psiProp, vector<vector<double100>> &gammaProp,
      vector<vector<double100>> &ksiProp, vector<double100> &XOut,
      vector<int> &kappaOut, vector<vector<double100>> &omegaOut,
      vector<vector<double100>> &psiOut, vector<vector<double100>> &gammaOut,
      vector<vector<double100>> &ksiOut, const Options &o,
      const MCMCOptions &MCMCo, boost::random::mt19937 &gen);
  void GenerateT0Z1Proposals(
      WrightFisher Current, int &indt0Prop, double100 &t0Prop,
      vector<double100> &XProp, vector<int> &kappaProp,
      vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
      vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
      double100 lambdaMax, const Options &o, boost::random::mt19937 &gen);
  void AcceptT0Z1Proposals(
      WrightFisher Current, int &indt0Prop, double100 &t0Prop,
      vector<double100> &XProp, vector<int> &kappaProp,
      vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
      vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
      vector<double100> &XOut, vector<int> &kappaOut,
      vector<vector<double100>> &omegaOut, vector<vector<double100>> &psiOut,
      vector<vector<double100>> &gammaOut, vector<vector<double100>> &ksiOut,
      const Options &o, const MCMCOptions &MCMCo, boost::random::mt19937 &gen);
  void GenerateT0Z2Proposals(
      WrightFisher Current, int &indt0Prop, double100 &t0Prop,
      vector<double100> &XProp, vector<int> &kappaProp,
      vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
      vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
      double100 lambdaMax, const Options &o, boost::random::mt19937 &gen);
  void AcceptT0Z2Proposals(
      WrightFisher Current, int &indt0Prop, double100 &t0Prop,
      vector<double100> &XProp, vector<int> &kappaProp,
      vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
      vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
      vector<double100> &XOut, vector<int> &kappaOut,
      vector<vector<double100>> &omegaOut, vector<vector<double100>> &psiOut,
      vector<vector<double100>> &gammaOut, vector<vector<double100>> &ksiOut,
      const Options &o, const MCMCOptions &MCMCo, boost::random::mt19937 &gen);
  void GenerateXtiProposals(WrightFisher Current, vector<double100> &XProp,
                            vector<int> &kappaProp,
                            vector<vector<double100>> &omegaProp,
                            vector<vector<double100>> &psiProp,
                            vector<vector<double100>> &gammaProp,
                            vector<vector<double100>> &ksiProp,
                            double100 lambdaMax, int iter, const Options &o,
                            boost::random::mt19937 &gen);
  void AcceptXtiProposals(
      vector<double100> &XProp, vector<int> &kappaProp,
      vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
      vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
      vector<double100> &XOut, vector<int> &kappaOut,
      vector<vector<double100>> &omegaOut, vector<vector<double100>> &psiOut,
      vector<vector<double100>> &gammaOut, vector<vector<double100>> &ksiOut,
      int iter, boost::random::mt19937 &gen);
  void GenerateXtnProposals(
      WrightFisher Current, vector<double100> &XProp, vector<int> &kappaProp,
      vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
      vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
      double100 lambdaMax, const Options &o, boost::random::mt19937 &gen);
  void AcceptXtnProposals(
      vector<double100> &XProp, vector<int> &kappaProp,
      vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
      vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
      vector<double100> &XOut, vector<int> &kappaOut,
      vector<vector<double100>> &omegaOut, vector<vector<double100>> &psiOut,
      vector<vector<double100>> &gammaOut, vector<vector<double100>> &ksiOut,
      boost::random::mt19937 &gen);

  // UPDATE FUNCTIONS

  void UpdateT0(WrightFisher Current, vector<double100> &XOut,
                vector<int> &kappaOut, vector<vector<double100>> &omegaOut,
                vector<vector<double100>> &psiOut,
                vector<vector<double100>> &gammaOut,
                vector<vector<double100>> &ksiOut, double100 lambdaMax,
                const Options &o, const MCMCOptions &MCMCo,
                boost::random::mt19937 &gen);
  void UpdateT0Z0(WrightFisher Current, vector<double100> &XOut,
                  vector<int> &kappaOut, vector<vector<double100>> &omegaOut,
                  vector<vector<double100>> &psiOut,
                  vector<vector<double100>> &gammaOut,
                  vector<vector<double100>> &ksiOut, double100 t0Prop,
                  double100 lambdaMax, const Options &o,
                  const MCMCOptions &MCMCo, boost::random::mt19937 &gen);
  void UpdateT0Z1(WrightFisher Current, vector<double100> &XOut,
                  vector<int> &kappaOut, vector<vector<double100>> &omegaOut,
                  vector<vector<double100>> &psiOut,
                  vector<vector<double100>> &gammaOut,
                  vector<vector<double100>> &ksiOut, double100 t0Prop,
                  double100 lambdaMax, const Options &o,
                  const MCMCOptions &MCMCo, boost::random::mt19937 &gen);
  void UpdateT0Z2(WrightFisher Current, vector<double100> &XOut,
                  vector<int> &kappaOut, vector<vector<double100>> &omegaOut,
                  vector<vector<double100>> &psiOut,
                  vector<vector<double100>> &gammaOut,
                  vector<vector<double100>> &ksiOut, double100 t0Prop,
                  double100 lambdaMax, const Options &o,
                  const MCMCOptions &MCMCo, boost::random::mt19937 &gen);
  void UpdateXti(WrightFisher Current, vector<double100> &XOut,
                 vector<int> &kappaOut, vector<vector<double100>> &omegaOut,
                 vector<vector<double100>> &psiOut,
                 vector<vector<double100>> &gammaOut,
                 vector<vector<double100>> &ksiOut, double100 lambdaMax,
                 int iter, const Options &o, boost::random::mt19937 &gen);
  void UpdateXtn(WrightFisher Current, vector<double100> &XOut,
                 vector<int> &kappaOut, vector<vector<double100>> &omegaOut,
                 vector<vector<double100>> &psiOut,
                 vector<vector<double100>> &gammaOut,
                 vector<vector<double100>> &ksiOut, double100 lambdaMax,
                 const Options &o, boost::random::mt19937 &gen);
  void UpdateSigma(WrightFisher Current, WrightFisher Proposal,
                   vector<double100> &selOut, vector<double100> &selProp,
                   const MCMCOptions &MCMCo, boost::random::mt19937 &gen);

  // ACTUAL PROGRAM

  void RunSampler(const MCMCOptions &MCMCo);

 private:
  // STORER VARIABLES

  vector<int> Data, Samplesizes;
  int selectionType, zerosCounter;
  vector<int> t0ksiCounter;
  double100 tc;
  vector<double> theta;
  vector<double100> times, sigmaStore, hStore, selLikelihood, t0Store,
      t0Likelihood, pseudoStore;
  vector<vector<vector<double100>>> omegaStore, psiStore, gammaStore, ksiStore;
  vector<vector<double100>> XStore, etaStore;
  vector<vector<int>> kappaStore;
  boost::random::mt19937 MCMCSampler_gen;

  // COMPUTATIONAL AIDES

  int computeC(int m, pair<vector<int>, double100> &C);
  int computeE(pair<vector<int>, double100> &C);
  double100 Getd(vector<double100> &d, int i, double100 x, double100 z,
                 double100 t);
  template <typename T>
  T Getlogakm(vector<double> th, int k, int m);
};

template <typename T>
T MCMCSampler::Getlogakm(vector<double> th, int k, int m) {
  double100 ths =
      static_cast<double100>(th.front()) + static_cast<double100>(th.back());
  assert(k >= m);

  T akm;

  if (k + m == 0) {
    akm = 0.0;
  } else {
    akm = log(ths + 2 * k - 1.0);

    for (int j = 2; j <= k; j++) {
      akm += log(ths + m + j - 2.0);
      if (j <= m) {
        akm -= log(static_cast<T>(j));
      }
      if (j <= k - m) {
        akm -= log(static_cast<T>(j));
      }
    }
  }
  return akm;
}

#endif
