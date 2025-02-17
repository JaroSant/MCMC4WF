#include "MCMCSampler.h"

#include <time.h>

#include <algorithm>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/bind.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/poisson.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/ref.hpp>
#include <cmath>
#include <fstream>
#include <map>

// USEFUL FUNCTIONS

vector<double100> MCMCSampler::Converter(vector<int> Data,
                                         vector<int> SampleSize) {
  vector<double100> Frequencies;

  // Checking we have at most 2 zero observations at the start, deleting entries
  // at start otherwise

  while ((Data.front() == 0) && (Data.at(1) == 0) && (Data.at(2) == 0)) {
    std::cout << "Trimming to have at most two zero observations at the start!"
              << std::endl;
    Data.erase(Data.begin());
    SampleSize.erase(SampleSize.begin());
    times.erase(times.begin());
  }

  // Setting first entries for X to be just Y/n

  vector<double100>::iterator itert = times.begin();
  for (vector<int>::iterator iterD = Data.begin(), iterS = SampleSize.begin();
       iterD != Data.end(); iterD++, iterS++, itert++) {
    if ((iterD == Data.begin()) && (*iterD == 0)) {
      std::cout
          << "Detected zero at the start! Setting zerosCounter to 1, and tc to "
          << *(itert + 1) << std::endl;
      tc = *(itert + 1);
      zerosCounter = 1;
    } else if ((iterD == Data.begin()) && (*iterD != 0)) {
      std::cout
          << "No zeros at the start! Setting zerosCounter to 0, and tc to "
          << times.front() << std::endl;
      tc = times.front();
      zerosCounter = 0;
    } else if ((*iterD == 0) && (tc == *(itert))) {
      std::cout << "Detected another zero at the start! Setting zerosCounter "
                   "to 2, and tc to "
                << *(itert + 1) << std::endl;
      tc = *(itert + 1);
      zerosCounter = 2;
    }
    if (tc <= *itert && *iterD == 0) {
      Frequencies.push_back(double(*iterD) / double(*iterS) + 0.00001);
    } else if (tc <= *itert && *iterD == *iterS) {
      Frequencies.push_back(double(*iterD) / double(*iterS) - 0.00001);
    } else {
      Frequencies.push_back(double(*iterD) / double(*iterS));
    }
  }

  return Frequencies;
}

double100 MCMCSampler::NormCDF(
    double100 x, double m, double v)  // Compute CDF of N(m,v) rv (v is stdev!)
{
  return 0.5 * erfc(-(x - m) / (sqrt(2.0) * v));
}

double100 MCMCSampler::NormPDF(double100 x, double m, double v)  // v is stdev!!
{
  return ((1 / (sqrt(2 * pi) * v)) *
          exp(-(((x - m) * (x - m)) / (2.0 * v * v))));
}

double100 MCMCSampler::DiscreteNormalCDF(
    double100 x, double100 t,
    vector<double> thetaP)  // Compute discrete normal cdf
{
  double theta = thetaP.empty() ? 0.0 : thetaP.back() + thetaP.front();
  double beta = 0.5 * (theta - 1.0) * static_cast<double>(t);
  double eta = (abs(beta) <= 2.5e-5 ? 1.0 : beta / (exp(beta) - 1.0));
  double mu = 2 * (eta / static_cast<double>(t));
  double sigma =
      (abs(beta) <= 2.5e-5
           ? 2.0 / (3.0 * static_cast<double>(t))
           : 2.0 * (eta / static_cast<double>(t)) * pow(eta + beta, 2) *
                 (1.0 + eta / (eta + beta) - 2.0 * eta) / pow(beta, 2));
  return NormCDF(x, mu, sqrt(sigma));
}

double100 MCMCSampler::Getd(vector<double100> &d, int i, double100 x,
                            double100 z, double100 t) {
  if (i > static_cast<int>(d.size()) - 1) d.resize(i + 1, -1.0);
  if (d[i] < 0.0) {
    d[i] = 0.0;
    int m = i / 2, offset = i % 2;

    for (int j = 0; j <= m; ++j) {
      int c1 = m + j + offset, c2 = m - j;
      boost::math::binomial_distribution<double100> B(c2, x);
      double100 Expected_Dirichlet = 0.0;
      for (int l = 0; l <= c2; ++l) {
        boost::math::beta_distribution<double100> D(theta.front() + l,
                                                    theta.back() + c2 - l);
        Expected_Dirichlet += pdf(B, l) * pdf(D, z);
      }
      d[i] +=
          exp(Getlogakm<double100>(theta, c1, c2) +
              static_cast<double100>(
                  -c1 * (c1 + theta.back() + theta.front() - 1) * t / 2.0)) *
          Expected_Dirichlet;
      assert(exp(Getlogakm<double100>(theta, c1, c2)) > 0.0);
    }
  }
  assert(d[i] >= 0.0);
  return d[i];
}

double100 MCMCSampler::Getd1(int i, double100 x, double100 z, double100 t) {
  double100 thsum = theta.front() + theta.back();
  double100 d = 0.0;
  int m = i / 2, offset = i % 2;

  for (int j = 0; j <= m; ++j) {
    int c1 = m + j + offset, c2 = m - j;
    boost::math::binomial_distribution<double100> B(c2, x);
    double100 Expected_Dirichlet = 0.0;
    for (int l = 0; l <= c2; ++l) {
      boost::math::beta_distribution<double100> D(theta.front() + l,
                                                  theta.back() + c2 - l);
      Expected_Dirichlet += pdf(B, l) * pdf(D, z);
    }

    d += exp(Getlogakm<double100>(theta, c1, c2) +
             static_cast<double100>(-c1 * (c1 + thsum - 1) * t / 2.0)) *
         Expected_Dirichlet;
    assert(exp(Getlogakm<double100>(theta, c1, c2)) > 0.0);
  }

  assert(d >= 0.0);
  return d;
}

int MCMCSampler::computeC(
    int m, pair<vector<int>, double100> &C)  /// Compute the quantity C_m
{
  assert(m >= 0);
  if (m > static_cast<int>(C.first.size() - 1) || C.first[m] < 0) {
    C.first.resize(m + 1, -1);
    int i = 0;
    double100 bimnext = exp(Getlogakm<double100>(theta, i + m + 1, m) -
                            (i + m + 1) *
                                (i + m + 1 + theta.front() + theta.back() - 1) *
                                C.second / 2.0),
              bim = exp(Getlogakm<double100>(theta, i + m, m) -
                        (i + m) * (i + m + theta.front() + theta.back() - 1) *
                            C.second / 2.0);
    while (bimnext > bim) {
      ++i;
      bim = bimnext;
      bimnext =
          exp(Getlogakm<double100>(theta, i + m + 1, m) -
              (i + m + 1) * (i + m + 1 + theta.front() + theta.back() - 1) *
                  C.second / 2.0);
    }
    C.first[m] = i;
  }
  assert(C.first[m] >= 0);
  return C.first[m];
}

int MCMCSampler::computeE(pair<vector<int>, double100> &C) {
  int next_constraint = computeC(0, C), curr_row = 0;
  int diag_index = next_constraint + (next_constraint % 2);
  bool Efound = false;
  while (!Efound) {
    ++curr_row;
    next_constraint = computeC(curr_row, C) + curr_row;
    if (diag_index - curr_row < next_constraint) {
      diag_index = next_constraint + curr_row;
      diag_index += (diag_index % 2);
    }
    if (curr_row == diag_index / 2) Efound = true;
  }
  return curr_row;  // = m
}

int MCMCSampler::computeK(double100 x, double100 z, vector<double> theta,
                          const Options &o) {
  double thsum = theta.front() + theta.back();
  return ceil(
      2.0 *
      ((max((thsum / theta.back()) * (1.0 - static_cast<double>(z)),
            (1.0 + thsum) / (1.0 - static_cast<double>(z)))) *
           (1.0 - static_cast<double>(x)) +
       (1.0 + (1.0 / static_cast<double>(z))) *
           (max(((static_cast<double>(z) * thsum) + 1.0) / theta.front(),
                1.0)) *
           static_cast<double>(x)) /
      o.eps);
}

double100 MCMCSampler::TransitionRatioApprox(WrightFisher Current,
                                             double100 xNum, double100 xDen,
                                             double100 zNum, double100 zDen,
                                             double100 tNum, double100 tDen,
                                             const Options &o) {
  double100 numerator =
      Current.UnconditionedDiffusionDensity(xNum, zNum, tNum, o);
  double100 denominator =
      Current.UnconditionedDiffusionDensity(xDen, zDen, tDen, o);

  return (numerator / denominator);
}

double100 MCMCSampler::TransitionApprox(
    WrightFisher Current, double100 x, double100 z, double100 t,
    const Options &o)  // For small t, uses discrete normal!
{
  return Current.UnconditionedDiffusionDensity(x, z, t, o);
}

// Returns log of prior ratio contribution
// to acceptance probability of sigma update
// - assuming here prior is normal!
double100 MCMCSampler::selPriorRatio(double100 sigmaProp, double100 hProp,
                                     vector<double100> etaProp,
                                     double100 sigmaCurr, double100 hCurr,
                                     vector<double100> etaCurr,
                                     const MCMCOptions &MCMCo) {
  if (MCMCo.selTypePriors == "Gaussian") {
    if (selectionType == 0) {
      double sigmaPriorMean = MCMCo.sigmaPriorMeans,
             sigmaPriorStd = MCMCo.sigmaPriorStdevs;
      return (log(NormPDF(sigmaProp, sigmaPriorMean, sigmaPriorStd)) -
              log(NormPDF(sigmaCurr, sigmaPriorMean, sigmaPriorStd)));
    } else if (selectionType == 1) {
      double sigmaPriorMean = MCMCo.sigmaPriorMeans,
             sigmaPriorStd = MCMCo.sigmaPriorStdevs;
      double hPriorMean = MCMCo.hPriorMeans, hPriorStdev = MCMCo.hPriorStdevs;
      return (log(NormPDF(sigmaProp, sigmaPriorMean, sigmaPriorStd)) -
              log(NormPDF(sigmaCurr, sigmaPriorMean, sigmaPriorStd)) +
              log(NormPDF(hProp, hPriorMean, hPriorStdev)) -
              log(NormPDF(hCurr, hPriorMean, hPriorStdev)));
    } else {
      double sigmaPriorMean = MCMCo.sigmaPriorMeans,
             sigmaPriorStd = MCMCo.sigmaPriorStdevs;
      double return_val =
          log(NormPDF(sigmaProp, sigmaPriorMean, sigmaPriorStd)) -
          log(NormPDF(sigmaCurr, sigmaPriorMean, sigmaPriorStd));
      vector<double>::const_reverse_iterator etaPM =
                                                 MCMCo.etaPriorMeans.crbegin(),
                                             etaPS =
                                                 MCMCo.etaPriorStdevs.crbegin();
      for (vector<double100>::reverse_iterator etaP = etaProp.rbegin(),
                                               etaC = etaCurr.rbegin();
           etaP != etaProp.rend(); etaP++, etaC++) {
        return_val = return_val + log(NormPDF(*etaP, *etaPM, *etaPS) -
                                      log(NormPDF(*etaC, *etaPM, *etaPS)));
        etaPM++;
        etaPS++;
      }
      return return_val;
    }
  } else {
    if (selectionType == 0) {
      bool sigmaProp_within = ((sigmaProp >= MCMCo.sigmaBoundsLower) &&
                               (sigmaProp <= MCMCo.sigmaBoundsUpper));
      if (sigmaProp_within) {
        return 0.0;
      } else {
        return -std::numeric_limits<double>::infinity();
      }
    } else if (selectionType == 1) {
      bool sigmaProp_within = ((sigmaProp >= MCMCo.sigmaBoundsLower) &&
                               (sigmaProp <= MCMCo.sigmaBoundsUpper));
      bool hProp_within =
          ((hProp >= MCMCo.hBoundsLower) && (hProp <= MCMCo.hBoundsUpper));
      if (sigmaProp_within && hProp_within) {
        return 0.0;
      } else {
        return -std::numeric_limits<double>::infinity();
      }
    } else {
      bool sigmaProp_within = ((sigmaProp >= MCMCo.sigmaBoundsLower) &&
                               (sigmaProp <= MCMCo.sigmaBoundsUpper));
      bool etaProp_within = true;
      vector<double>::const_reverse_iterator etaBL =
                                                 MCMCo.etaBoundsLower.crbegin(),
                                             etaBU =
                                                 MCMCo.etaBoundsUpper.crbegin();
      for (vector<double100>::reverse_iterator etaP = etaProp.rbegin();
           etaP != etaProp.rend(); etaP++) {
        if ((*etaP < *etaBL) || (*etaP > *etaBU)) {
          etaProp_within = false;
          break;
        }
        etaBL++;
        etaBU++;
      }
      if (sigmaProp_within && etaProp_within) {
        return 0.0;
      } else {
        return -std::numeric_limits<double>::infinity();
      }
    }
  }
}

double100 MCMCSampler::selProposalRatio(double100 sigmaProp, double100 hProp,
                                        vector<double100> etaProp,
                                        double100 sigmaCurr, double100 hCurr,
                                        vector<double100> etaCurr,
                                        const MCMCOptions &MCMCo) {
  if (MCMCo.selTypePriors == "Gaussian") {
    if (selectionType == 0) {
      double sigmaPropStep = MCMCo.sigmaProposalSteps;
      return (log(NormPDF(sigmaCurr, static_cast<double>(sigmaProp),
                          sigmaPropStep)) -
              log(NormPDF(sigmaProp, static_cast<double>(sigmaCurr),
                          sigmaPropStep)));
    } else if (selectionType == 1) {
      double sigmaPropStep = MCMCo.sigmaProposalSteps,
             hPropStep = MCMCo.hProposalSteps;
      return (log(NormPDF(sigmaCurr, static_cast<double>(sigmaProp),
                          sigmaPropStep)) -
              log(NormPDF(sigmaProp, static_cast<double>(sigmaCurr),
                          sigmaPropStep)) +
              log(NormPDF(hCurr, static_cast<double>(hProp), hPropStep)) -
              log(NormPDF(hProp, static_cast<double>(hCurr), hPropStep)));
    } else {
      double sigmaPropStep = MCMCo.sigmaProposalSteps;
      double return_val = log(NormPDF(sigmaCurr, static_cast<double>(sigmaProp),
                                      sigmaPropStep)) -
                          log(NormPDF(sigmaProp, static_cast<double>(sigmaCurr),
                                      sigmaPropStep));
      vector<double>::const_reverse_iterator etaPS =
          MCMCo.etaProposalSteps.crbegin();
      for (vector<double100>::reverse_iterator etaP = etaProp.rbegin(),
                                               etaC = etaCurr.rbegin();
           etaP != etaProp.rend(); etaP++, etaC++) {
        return_val =
            return_val +
            log(NormPDF(*etaC, static_cast<double>(*etaP), *etaPS) -
                log(NormPDF(*etaP, static_cast<double>(*etaC), *etaPS)));
        etaPS++;
      }
      return return_val;
    }
  } else {
    return 0.0;
  }

  // Returns log of proposal ratio contribution
  // to acceptance probability of sigma update -
  // assuming here proposal is normal!
}

double100 MCMCSampler::t0PriorRatio(double100 t0Prop, double100 t0Curr,
                                    const MCMCOptions &MCMCo) {
  double100 priormean = MCMCo.t0PriorMeans, returnval;

  if (MCMCo.AlleleAgePrior == 0) {
    returnval = exp(-priormean * (t0Curr - t0Prop));  // Exponential priors
  } else {
    double100 priorstdev = MCMCo.t0PriorStdevs;
    returnval = pow((tc - t0Prop) / (tc - t0Curr), priormean - 1.0) *
                exp(-(1.0 / priorstdev) * (t0Curr - t0Prop));
  }

  return returnval;  // Returns log of prior ratio contribution to t_0 AP - in
                     // this case we take Exponential prior
}

double100 MCMCSampler::t0ProposalRatio(double100 t0Prop, double100 t0Curr,
                                       const MCMCOptions &MCMCo) {
  return ((NormCDF(tc, static_cast<double>(t0Curr), MCMCo.t0ProposalSteps) /
           NormCDF(tc, static_cast<double>(t0Prop), MCMCo.t0ProposalSteps)));
}

double100 MCMCSampler::PoissonEstimator(WrightFisher Diffusion, double100 t,
                                        double100 x, double100 y,
                                        const Options &o,
                                        boost::random::mt19937 &gen) {
  vector<double100> phiminmaxrange{Diffusion.phiMin, Diffusion.phiMax,
                                   Diffusion.phiMax - Diffusion.phiMin};
  double100 phimin = phiminmaxrange[0], phimax = phiminmaxrange[1];
  double100 kapmean = phiminmaxrange[2] * t;
  boost::random::poisson_distribution<> kap(static_cast<double>(kapmean));

  int kappa = kap(gen);
  vector<double100> psi(kappa), gamma(kappa), omega(kappa), ksi(kappa);
  boost::random::uniform_01<> unifPsi, unifGam;
  double100 tstart = 0.0, tend = t, pseudo = 1.0;
  double100 tol = 5.0e-5;

  auto genPsi =
      [&tstart, &tend, &unifPsi,
       &gen]()  /// Routine to create uniform variates over [t0Prop,t1]
  { return (tstart + ((tend - tstart) * unifPsi(gen))); };
  bool suitable_times = false;
  while (!suitable_times) {
    suitable_times = true;
    std::generate(begin(psi), end(psi), genPsi);
    vector<double100> copy_times(psi);
    sortVectorsAscending(copy_times, copy_times);
    copy_times.insert(copy_times.begin(), tstart);
    copy_times.push_back(tend);
    for (vector<double100>::iterator cti = copy_times.begin();
         cti != copy_times.end() - 1; cti++) {
      if (fabs(*(cti + 1) - *cti) <= tol) {
        suitable_times = false;
      }
    }
  }
  auto genGam =
      [&unifGam, &gen]()  /// Routine to create uniform variates over [0,1]
  { return (unifGam(gen)); };

  // Generate the uniform variates
  std::generate(begin(gamma), end(gamma), genGam);

  sortVectorsAscending(
      psi, psi,
      gamma);  // Sort according to time ordering of psi_j marks

  for (vector<double100>::iterator Psiiter = psi.begin(),
                                   Gamiter = gamma.begin(),
                                   Omegiter = omega.begin();
       Psiiter != psi.end(); Psiiter++, Gamiter++, Omegiter++) {
    if (Psiiter == psi.begin()) {
      *Omegiter =
          Diffusion.DrawBridgepoint(x, y, tstart, tend, *Psiiter, o, gen)
              .first;  // Draw skeleton point from neutral bridge going from
                       // x_{t_{i-1}} to XtiProp at time psi_1
    } else {
      *Omegiter = Diffusion
                      .DrawBridgepoint(omega.back(), y, *(Psiiter - 1), tend,
                                       *Psiiter, o, gen)
                      .first;  // Draw skeleton point from neutral bridge going
                               // from omega_j-1 to XtiProp at time psi_j
    }

    pseudo *= ((phimax - Diffusion.Phitilde(*Omegiter)) / (phimax - phimin));
  }

  return pseudo;
}

// PRINTING FUNCTIONS

void MCMCSampler::printVDouble(ofstream &outputFile, vector<double100> vec,
                               int N) {
  int vector_size = static_cast<int>(vec.size());
  if (outputFile.is_open()) {
    if (vector_size < N) {
      ostream_iterator<double100> output_iterator(outputFile, "\n");
      copy(vec.begin(), vec.end(), output_iterator);
    } else {
      ostream_iterator<double100> output_iterator(outputFile, "\n");
      copy(vec.begin(), vec.begin() + N, output_iterator);
    }
    outputFile.close();
  } else {
    std::cout << "Can't open file!" << endl;
  }
}

void MCMCSampler::printVInt(ofstream &outputFile, vector<int> vec, int N) {
  int vector_size = static_cast<int>(vec.size());
  if (outputFile.is_open()) {
    if (vector_size < N) {
      ostream_iterator<int> output_iterator(outputFile, "\n");
      copy(vec.begin(), vec.end(), output_iterator);
    } else {
      ostream_iterator<int> output_iterator(outputFile, "\n");
      copy(vec.begin(), vec.begin() + N, output_iterator);
    }
    outputFile.close();
  } else {
    std::cout << "Can't open file!" << endl;
  }
}

void MCMCSampler::printMatrixInt(ofstream &outputFile, vector<vector<int>> vec,
                                 int N) {
  int vector_size = static_cast<int>(vec.size());
  if (outputFile.is_open()) {
    if (vector_size < N) {
      for (vector<vector<int>>::iterator it = vec.begin(); it != vec.end();
           it++) {
        for (vector<int>::iterator printer = (*it).begin();
             printer != (*it).end(); printer++) {
          if (printer != (*it).end() - 1) {
            outputFile << " " << *printer << " ";
          } else {
            outputFile << " " << *printer << "\n";
          }
        }
      }
    } else {
      for (vector<vector<int>>::iterator it = vec.begin();
           it != vec.begin() + N; it++) {
        for (vector<int>::iterator printer = (*it).begin();
             printer != (*it).end(); printer++) {
          if (printer != (*it).end() - 1) {
            outputFile << " " << *printer << " ";
          } else {
            outputFile << " " << *printer << "\n";
          }
        }
      }
    }
    outputFile.close();
  } else {
    std::cout << "Can't open file!" << endl;
  }
}

void MCMCSampler::printMatrixDouble(ofstream &outputFile,
                                    vector<vector<double100>> vec, int N) {
  int vector_size = static_cast<int>(vec.size());
  if (outputFile.is_open()) {
    if (vector_size < N) {
      for (vector<vector<double100>>::iterator it = vec.begin();
           it != vec.end(); it++) {
        for (vector<double100>::iterator printer = (*it).begin();
             printer != (*it).end(); printer++) {
          if (printer != (*it).end() - 1) {
            outputFile << " " << *printer << " ";
          } else {
            outputFile << " " << *printer << "\n";
          }
        }
      }
    } else {
      for (vector<vector<double100>>::iterator it = vec.begin();
           it != vec.begin() + N; it++) {
        for (vector<double100>::iterator printer = (*it).begin();
             printer != (*it).end(); printer++) {
          if (printer != (*it).end() - 1) {
            outputFile << " " << *printer << " ";
          } else {
            outputFile << " " << *printer << "\n";
          }
        }
      }
    }
    outputFile.close();
  } else {
    std::cout << "Can't open file!" << endl;
  }
}

void MCMCSampler::print3DMatrixDouble(ofstream &outputFile,
                                      vector<vector<vector<double100>>> vec,
                                      int N) {
  if (outputFile.is_open()) {
    int vector_size = static_cast<int>(vec.size());
    if (vector_size < N) {
      int itercount = -1;

      for (vector<vector<vector<double100>>>::iterator it = vec.begin();
           it != vec.end(); it++) {
        itercount++;

        for (vector<vector<double100>>::iterator itt = (*it).begin();
             itt != (*it).end(); itt++) {
          if (itt == (*it).begin()) {
            outputFile << itercount << " ";
          }

          for (vector<double100>::iterator printer = (*itt).begin();
               printer != (*itt).end(); printer++) {
            outputFile << *printer << " ";
          }

          if (itt == (*it).end() - 1) {
            outputFile << "\n";
          }
        }
      }
    } else {
      int itercount = 0;  // vec.size() - N - 1;

      for (vector<vector<vector<double100>>>::iterator it = vec.begin();
           it != vec.begin() + N; it++) {
        itercount++;

        for (vector<vector<double100>>::iterator itt = (*it).begin();
             itt != (*it).end(); itt++) {
          if (itt == (*it).begin()) {
            outputFile << itercount << " ";
          }

          for (vector<double100>::iterator printer = (*itt).begin();
               printer != (*itt).end(); printer++) {
            outputFile << *printer << " ";
          }

          if (itt == (*it).end() - 1) {
            outputFile << "\n";
          }
        }
      }
    }
    outputFile.close();
  } else {
    std::cout << "Can't open file!" << endl;
  }
}

void MCMCSampler::print3DMatrixInt(ofstream &outputFile,
                                   vector<vector<vector<int>>> vec, int N) {
  if (outputFile.is_open()) {
    int vector_size = static_cast<int>(vec.size());
    if (vector_size < N) {
      int itercount = -1;

      for (vector<vector<vector<int>>>::iterator it = vec.begin();
           it != vec.end(); it++) {
        itercount++;

        for (vector<vector<int>>::iterator itt = (*it).begin();
             itt != (*it).end(); itt++) {
          if (itt == (*it).begin()) {
            outputFile << itercount << " ";
          }

          for (vector<int>::iterator printer = (*itt).begin();
               printer != (*itt).end(); printer++) {
            outputFile << *printer << " ";
          }

          if (itt == (*it).end() - 1) {
            outputFile << "\n";
          }
        }
      }
    } else {
      int itercount = 0;  // vec.size() - N - 1;

      for (vector<vector<vector<int>>>::iterator it = vec.begin();
           it != vec.begin() + N; it++) {
        itercount++;

        for (vector<vector<int>>::iterator itt = (*it).begin();
             itt != (*it).end(); itt++) {
          if (itt == (*it).begin()) {
            outputFile << itercount << " ";
          }

          for (vector<int>::iterator printer = (*itt).begin();
               printer != (*itt).end(); printer++) {
            outputFile << *printer << " ";
          }

          if (itt == (*it).end() - 1) {
            outputFile << "\n";
          }
        }
      }
    }
    outputFile.close();
  } else {
    std::cout << "Can't open file!" << endl;
  }
}

void MCMCSampler::printDatapoints(ofstream &outputFile) {
  if (outputFile.is_open()) {
    copy(times.begin(), times.end(),
         ostream_iterator<double100>(outputFile, " "));
    outputFile << "\n";
    copy(Samplesizes.begin(), Samplesizes.end(),
         ostream_iterator<double100>(outputFile, " "));
    outputFile << "\n";
    copy(Data.begin(), Data.end(),
         ostream_iterator<double100>(outputFile, " "));
    outputFile << "\n";
    copy((XStore.front()).begin(), (XStore.front()).end(),
         ostream_iterator<double100>(outputFile, " "));
    outputFile << "\n";
    outputFile.close();
  } else {
    std::cout << "Can't open file!" << endl;
  }
}

void MCMCSampler::printOtherInput(ofstream &outputFile,
                                  const MCMCOptions &MCMCo) {
  if (outputFile.is_open()) {
    outputFile << "Mutation rates " << MCMCo.Theta[0] << " " << MCMCo.Theta[1]
               << "\n";
    if (selectionType == 0) {
      outputFile << "Prior mean for sigma " << MCMCo.sigmaPriorMeans << "\n";
      outputFile << "Prior stdev for sigma " << MCMCo.sigmaPriorStdevs << "\n";
    } else if (selectionType == 1) {
      outputFile << "Prior mean for sigma " << MCMCo.sigmaPriorMeans << "\n";
      outputFile << "Prior stdev for sigma " << MCMCo.sigmaPriorStdevs << "\n";
      outputFile << "Prior mean for h " << MCMCo.hPriorMeans << "\n";
      outputFile << "Prior stdev for h " << MCMCo.hPriorStdevs << "\n";
    } else {
      outputFile << "Prior mean for sigma " << MCMCo.sigmaPriorMeans << "\n";
      outputFile << "Prior stdev for sigma " << MCMCo.sigmaPriorStdevs << "\n";
      int count = 0;
      for (vector<double100>::const_reverse_iterator
               etaPM = MCMCo.etaPriorMeans.crbegin(),
               etaPS = MCMCo.etaPriorStdevs.crbegin();
           etaPM != MCMCo.etaPriorMeans.crend(); etaPM++, etaPS++) {
        outputFile << "Prior mean for " << count << "-order coefficient "
                   << *etaPM << "\n";
        outputFile << "Prior stdev for " << count << "-order coefficient "
                   << *etaPS << "\n";
        count++;
      }
    }
    if (selectionType == 0) {
      outputFile << "Proposal stdev for sigma " << MCMCo.sigmaProposalSteps
                 << "\n";
    } else if (selectionType == 1) {
      outputFile << "Proposal stdev for sigma " << MCMCo.sigmaProposalSteps
                 << "\n";
      outputFile << "Proposal stdev for h " << MCMCo.hProposalSteps << "\n";
    } else {
      outputFile << "Proposal stdev for sigma " << MCMCo.sigmaProposalSteps
                 << "\n";
      int count = 0;
      for (vector<double100>::const_iterator etaPS =
               MCMCo.etaProposalSteps.cbegin();
           etaPS != MCMCo.etaProposalSteps.cend(); etaPS++) {
        outputFile << "Proposal stdev for " << count << "-order coefficient "
                   << *etaPS << "\n";
        count++;
      }
    }
    if (selectionType == 0) {
      outputFile << "Precision for sigma mean " << MCMCo.sigmaMeanPrec << "\n";
      outputFile << "Precision for sigma stdev " << MCMCo.sigmaStdevPrec
                 << "\n";
    } else if (selectionType == 1) {
      outputFile << "Precision for sigma mean " << MCMCo.sigmaMeanPrec << "\n";
      outputFile << "Precision for sigma stdev " << MCMCo.sigmaStdevPrec
                 << "\n";
      outputFile << "Precision for h mean " << MCMCo.hMeanPrec << "\n";
      outputFile << "Precision for h stdev " << MCMCo.hStdevPrec << "\n";
    } else {
      int count = 0;
      for (vector<double100>::const_iterator
               etaMP = MCMCo.etaMeanPrec.cbegin(),
               etaSP = MCMCo.etaStdevPrec.cbegin();
           etaMP != MCMCo.etaMeanPrec.cend(); etaMP++, etaSP++) {
        outputFile << "Precision for " << count << "-order coefficient mean "
                   << *etaMP << "\n";
        outputFile << "Precision for " << count << "-order coefficient stdev "
                   << *etaSP << "\n";
        count++;
      }
    }
    outputFile << "Prior mean for t0 " << MCMCo.t0PriorMeans << "\n";
    outputFile << "Prior stdev for t0 " << MCMCo.t0PriorStdevs << "\n";
    outputFile << "Proposal stdev mean for t0 " << MCMCo.t0ProposalSteps
               << "\n";
    outputFile << "Precision for t0 mean " << MCMCo.t0MeanPrec << "\n";
    outputFile << "Precision for t0 stdev " << MCMCo.t0StdevPrec << "\n";
    outputFile << "Burn-in set to " << MCMCo.burnIn << "\n";
    outputFile << "LookBack set to " << MCMCo.lookBack << "\n";
    outputFile << "Printing to file every " << MCMCo.printCounter << "\n";
    outputFile.close();
  } else {
    std::cout << "Can't open file!" << endl;
  }
}

// INITIALISING FUNCTIONS

void MCMCSampler::SetMutation(const MCMCOptions &MCMCo) {
  theta.push_back(MCMCo.Theta[0]);
  theta.push_back(MCMCo.Theta[1]);
}

void MCMCSampler::InitialiseSel(const MCMCOptions &MCMCo,
                                boost::random::mt19937 &gen) {
  if (selectionType == 0) {
    boost::random::normal_distribution<> sigmaNORMAL(MCMCo.sigmaPriorMeans,
                                                     MCMCo.sigmaPriorStdevs);
    sigmaStore.push_back(sigmaNORMAL(gen));
    selLikelihood.push_back(0.0);
    std::cout << "Initialising sigma at " << sigmaStore.back() << std::endl;
  } else if (selectionType == 1) {
    boost::random::normal_distribution<> sigmaNORMAL(MCMCo.sigmaPriorMeans,
                                                     MCMCo.sigmaPriorStdevs),
        hNORMAL(MCMCo.hPriorMeans, MCMCo.hPriorStdevs);
    sigmaStore.push_back(sigmaNORMAL(gen));
    hStore.push_back(hNORMAL(gen));
    selLikelihood.push_back(0.0);
    std::cout << "Initialising sigma at " << sigmaStore.back() << std::endl;
    std::cout << "Initialising h at " << hStore.back() << std::endl;
  } else {
    boost::random::normal_distribution<> sigmaNORMAL(MCMCo.sigmaPriorMeans,
                                                     MCMCo.sigmaPriorStdevs);
    sigmaStore.push_back(sigmaNORMAL(gen));
    std::cout << "Initialising sigma at " << sigmaStore.back() << std::endl;
    int count = 0;
    vector<double100> initEta;
    for (vector<double100>::const_reverse_iterator
             etaPM = MCMCo.etaPriorMeans.crbegin(),
             etaPS = MCMCo.etaPriorStdevs.crbegin();
         etaPM != MCMCo.etaPriorMeans.crend(); etaPM++, etaPS++) {
      boost::random::normal_distribution<> NORMAL(*etaPM, *etaPS);
      initEta.insert(initEta.begin(), NORMAL(gen));
      selLikelihood.push_back(0.0);
      std::cout << "Initialising " << count << "-order coefficient of eta at "
                << initEta.back() << std::endl;
    }
    etaStore.push_back(initEta);
  }
}

void MCMCSampler::InitialiseSkeleton(WrightFisher Current, const Options &o,
                                     boost::random::mt19937 &gen) {
  vector<vector<double100>> omegaInit, psiInit, gammaInit, ksiInit;
  vector<int> kappaInit;

  if (t0Store.back() < times.front()) {
    vector<vector<double100>> initOut =
        Current.NonNeutralDrawBridge(0.0, t0Store.back(), times.front(),
                                     XStore.back().front(), false, o, gen);
    omegaInit.push_back(initOut.at(0));
    psiInit.push_back(initOut.at(1));
    gammaInit.push_back(initOut.at(2));
    ksiInit.push_back(vector<double100>(gammaInit.back().size(), 0.0));
    kappaInit.push_back(static_cast<int>(initOut.at(3).front()));

    for (vector<double100>::iterator iter = XStore.back().begin(),
                                     titer = times.begin();
         iter != XStore.back().end() - 1; iter++, titer++) {
      vector<vector<double100>> initOut = Current.NonNeutralDrawBridge(
          *iter, *titer, *(titer + 1), *(iter + 1), false, o, gen);
      omegaInit.push_back(initOut.at(0));
      psiInit.push_back(initOut.at(1));
      gammaInit.push_back(initOut.at(2));
      ksiInit.push_back(vector<double100>(gammaInit.back().size(), 0.0));
      kappaInit.push_back(static_cast<int>(initOut.at(3).front()));
    }

    int counter = 0;
    double100 phirange = Current.phiMax - Current.phiMin;

    for (vector<double100>::iterator it = ksiInit.front().begin();
         it != ksiInit.front().end(); it++) {
      if (*it < phirange) {
        counter++;
      }
    }

    omegaStore.push_back(omegaInit);
    psiStore.push_back(psiInit);
    gammaStore.push_back(gammaInit);
    ksiStore.push_back(ksiInit);
    kappaStore.push_back(kappaInit);
    t0ksiCounter.push_back(counter);

    omegaStore.push_back(omegaInit);
    psiStore.push_back(psiInit);
    gammaStore.push_back(gammaInit);
    ksiStore.push_back(ksiInit);
    kappaStore.push_back(kappaInit);
    t0ksiCounter.push_back(counter);
  } else {
    vector<double100>::iterator next_time = times.begin(),
                                next_x = XStore.back().begin();
    while (*next_time < t0Store.back()) {
      next_time++;
      next_x++;
      omegaInit.push_back(vector<double100>());
      psiInit.push_back(vector<double100>());
      gammaInit.push_back(vector<double100>());
      ksiInit.push_back(vector<double100>());
      kappaInit.push_back(0);
    }
    vector<vector<double100>> initOut = Current.NonNeutralDrawBridge(
        0.0, t0Store.back(), *next_time, *next_x, false, o, gen);
    omegaInit.push_back(initOut.at(0));
    psiInit.push_back(initOut.at(1));
    gammaInit.push_back(initOut.at(2));
    ksiInit.push_back(vector<double100>(gammaInit.back().size(), 0.0));
    kappaInit.push_back(static_cast<int>(initOut.at(3).front()));

    for (vector<double100>::iterator iter = next_x, titer = next_time;
         iter != XStore.back().end() - 1; iter++, titer++) {
      vector<vector<double100>> initOut = Current.NonNeutralDrawBridge(
          *iter, *titer, *(titer + 1), *(iter + 1), false, o, gen);
      omegaInit.push_back(initOut.at(0));
      psiInit.push_back(initOut.at(1));
      gammaInit.push_back(initOut.at(2));
      ksiInit.push_back(vector<double100>(gammaInit.back().size(), 0.0));
      kappaInit.push_back(static_cast<int>(initOut.at(3).front()));
    }

    int counter = 0;
    double100 phirange = Current.phiMax - Current.phiMin;

    for (vector<double100>::iterator it = ksiInit.front().begin();
         it != ksiInit.front().end(); it++) {
      if (*it < phirange) {
        counter++;
      }
    }

    omegaStore.push_back(omegaInit);
    psiStore.push_back(psiInit);
    gammaStore.push_back(gammaInit);
    ksiStore.push_back(ksiInit);
    kappaStore.push_back(kappaInit);
    t0ksiCounter.push_back(counter);

    omegaStore.push_back(omegaInit);
    psiStore.push_back(psiInit);
    gammaStore.push_back(gammaInit);
    ksiStore.push_back(ksiInit);
    kappaStore.push_back(kappaInit);
    t0ksiCounter.push_back(counter);
  }
}

void MCMCSampler::InitialiseT0(const MCMCOptions &MCMCo,
                               boost::random::mt19937 &gen) {
  if (MCMCo.AlleleAgePrior == 0) {
    boost::random::exponential_distribution<> EXPONENTIAL(MCMCo.t0PriorMeans);
    double100 z = tc - EXPONENTIAL(gen);
    t0Store.push_back(z);
    t0Likelihood.push_back(0.0);
  } else {
    boost::random::gamma_distribution<> GAMMA(MCMCo.t0PriorMeans,
                                              MCMCo.t0PriorStdevs);
    double100 z = tc - GAMMA(gen);
    t0Store.push_back(z);
    t0Likelihood.push_back(0.0);
  }
  std::cout << "Initialising t0 at " << t0Store.back() << std::endl;
}

void MCMCSampler::InitialisePseudo(WrightFisher Current, const Options &o,
                                   boost::random::mt19937 &gen) {
  vector<double>::iterator right_time = times.begin(),
                           right_x = XStore.back().begin();
  while (*right_time < t0Store.back()) {
    right_time++;
    right_x++;
  }
  right_time++;
  right_x++;

  pseudoStore.push_back(PoissonEstimator(Current, *(right_time)-t0Store.back(),
                                         0.0, *right_x, o, gen));
}

// PROPOSAL MECHANISM FUNCTIONS

pair<double100, vector<vector<double100>>>
MCMCSampler::NonNeutralDrawEndpointBinomialObservations(
    WrightFisher Current, double100 x, double100 t1, double100 t2,
    const Options &o,
    boost::random::mt19937
        &gen)  // Returns Endpoint at time t2 for non-neutral WF starting from x
// at time t1
{
  bool accept = false;
  vector<double100> paras{Current.phiMin, Current.phiMax,
                          Current.phiMax - Current.phiMin};
  vector<vector<double100>> ptr;
  double100 kapmean = paras[2] * (t2 - t1),
            XtestT;  // Generate Poisson number of points
  boost::random::poisson_distribution<> kap(static_cast<double>(kapmean));

  boost::random::uniform_01<> unift, unifm,
      unifU;  // Set up uniform points over [t1,t2], [0,paras[2]], [0,1]
  int rcount = 0;
  while (!accept)  // Until you get good point, keep going
  {
    int kappa = kap(gen);
    double100 u = unifU(gen);
    double100 tol = 5.0e-5;

    vector<double100> path, times(kappa), marks(kappa), rejcount;
    auto gent = [&t1, &t2, &unift, &gen]() {
      return (t1 + ((t2 - t1) * unift(gen)));
    };
    // Need to check that returned uniform times are strictly increasing!
    bool suitable_times = false;
    while (!suitable_times) {
      suitable_times = true;
      std::generate(begin(times), end(times), gent);
      vector<double100> copy_times(times);
      sortVectorsAscending(copy_times, copy_times);
      copy_times.insert(copy_times.begin(), t1);
      copy_times.push_back(t2);
      for (vector<double100>::iterator cti = copy_times.begin();
           cti != copy_times.end() - 1; cti++) {
        if (fabs(*(cti + 1) - *cti) <= tol) {
          suitable_times = false;
        }
      }
    }
    auto genm = [&paras, &unifm, &gen]() { return (paras[2] * unifm(gen)); };
    std::generate(begin(marks), end(marks), genm);
    sortVectorsAscending(
        times, times,
        marks);  // Generate kappan, uniform [t1,t2], uniform [0,paras[2]],
    // uniform [0,1] and sort according to timestamps

    times.push_back(t2);
    marks.push_back(u);  // add on end point as needs to be checked differently
    // to skeleton points

    for (vector<double100>::iterator itt = times.begin(), itm = marks.begin();
         itt != times.end(); itt++, itm++)  // iterate through skeleton points
    {
      if (kappa == 0)  // no skeletonpoints so end point directly
      {
        path.push_back(Current.DrawEndpoint(x, t1, t2, o, gen)
                           .first);  // generate endpoint

        if (exp(0.5 * sigmaStore.back() * (path.back() - 1.0)) <
            *itm)  // test if good, if not go back to while loop
        {
          rcount++;
          break;
        }

        XtestT = path.back();
        accept = true;
        rejcount.push_back(rcount);
        ptr.push_back(path);
        ptr.push_back(times);
        ptr.push_back(rejcount);
      } else {  // kappa >0 so we need to generate skeleton points and endpoint
        if (itt == times.begin()) {
          path.push_back(Current.DrawEndpoint(x, t1, *itt, o, gen)
                             .first);  // generate skeleton points in right way

          if (Current.Phitilde(path.back()) - paras[0] >
              *itm)  // test points are fine, otherwise just go back to while
                     // loop
          {
            rcount++;
            break;
          }
        } else if (*itt != t2) {  // need to separate first time stamp and rest
                                  // due to referencing inside function
          path.push_back(
              Current.DrawEndpoint(path.back(), *(itt - 1), *itt, o, gen)
                  .first);

          if (Current.Phitilde(path.back()) - paras[0] > *itm) {
            rcount++;
            break;
          }
        } else {  // Endpoint draw here
          path.push_back(
              Current.DrawEndpoint(path.back(), *(itt - 1), *itt, o, gen)
                  .first);

          if (exp(0.5 * sigmaStore.back() * (path.back() - 1.0)) <
              *itm)  // Check corresponding endpoint condition
          {
            rcount++;
            break;
          }

          rejcount.push_back(rcount);
          ptr.push_back(path);
          ptr.push_back(times);
          ptr.push_back(rejcount);
        }
      }

      if (*itt == t2) {
        XtestT = path.back();
        accept = true;
      }
    }
  }

  return make_pair(XtestT, ptr);
}

pair<double100, vector<vector<double100>>>
MCMCSampler::NonNeutralBridgePointBinomialObservations(
    WrightFisher Current, double100 x, double100 t1, double100 t2, double100 z,
    const Options &o, boost::random::mt19937 &gen,
    double100 testT)  // Returns skeleton points for interval [t1,t2] for
                      // non-neutral WF
//  bridge started from x ending at z
{
  bool accept = false;
  double100 XtestT;
  vector<double100> paras{Current.phiMin, Current.phiMax,
                          Current.phiMax - Current.phiMin};
  double100 kapmean = paras[2] * (t2 - t1);
  boost::random::poisson_distribution<> kap(static_cast<double>(kapmean));

  boost::random::uniform_01<> unift, unifm;
  vector<vector<double100>> ptr;
  int rcount = 0;
  while (!accept) {
    int kappa = kap(gen);
    double100 tol = 5.0e-5;

    vector<double100> path, times(kappa), marks(kappa), rejcount;
    auto gent = [&t1, &t2, &unift, &gen]() {
      return (t1 + ((t2 - t1) * unift(gen)));
    };
    // Need to check that returned uniform times are strictly increasing!
    bool suitable_times = false;
    while (!suitable_times) {
      suitable_times = true;
      std::generate(begin(times), end(times), gent);
      vector<double100> copy_times(times);
      copy_times.push_back(testT);
      sortVectorsAscending(copy_times, copy_times);
      copy_times.insert(copy_times.begin(), t1);
      copy_times.push_back(t2);
      for (vector<double100>::iterator cti = copy_times.begin();
           cti != copy_times.end() - 1; cti++) {
        if (fabs(*(cti + 1) - *cti) <= tol) {
          suitable_times = false;
        }
      }
    }
    auto genm = [&paras, &unifm, &gen]() { return (paras[2] * unifm(gen)); };
    std::generate(begin(marks), end(marks), genm);
    times.push_back(testT);
    marks.push_back(paras[2] + 1.0);
    sortVectorsAscending(times, times, marks);  // setting up all required stuff

    for (vector<double100>::iterator itt = times.begin(), itm = marks.begin();
         itt != times.end(); itt++, itm++) {
      if (kappa == 0) {
        path.push_back(
            Current.DrawBridgepoint(x, z, t1, t2, *itt, o, gen).first);
        rejcount.push_back(rcount);
        ptr.push_back(path);
        ptr.push_back(times);
        ptr.push_back(rejcount);
        XtestT = path.back();
        accept = true;
      } else {
        if (itt == times.begin()) {
          path.push_back(
              Current.DrawBridgepoint(x, z, t1, t2, *itt, o, gen).first);

          if (Current.Phitilde(path.back()) - paras[0] > *itm) {
            rcount++;
            break;
          }
        } else if (*itt != times.back()) {
          path.push_back(
              Current
                  .DrawBridgepoint(path.back(), z, *(itt - 1), t2, *itt, o, gen)
                  .first);

          if (Current.Phitilde(path.back()) - paras[0] > *itm) {
            rcount++;
            break;
          }
        } else {
          path.push_back(
              Current
                  .DrawBridgepoint(path.back(), z, *(itt - 1), t2, *itt, o, gen)
                  .first);
          if (Current.Phitilde(path.back()) - paras[0] > *itm) {
            rcount++;
            break;
          }
        }
        rejcount.push_back(rcount);
        ptr.push_back(path);
        ptr.push_back(times);
        ptr.push_back(rejcount);

        if (*itt == testT) {
          XtestT = path.back();
          accept = true;
        }
      }
    }
  }

  return make_pair(XtestT, ptr);
}

double100 MCMCSampler::t0Proposal(const MCMCOptions &MCMCo,
                                  boost::random::mt19937 &gen) {
  bool accept = false;
  double100 t0Prop;
  boost::random::normal_distribution<> NORMAL(
      static_cast<double>(t0Store.back()), MCMCo.t0ProposalSteps);
  while (!accept) {
    double100 t0Trial = NORMAL(gen);
    if (zerosCounter == 0) {
      if (t0Trial < tc - MCMCo.AlleleAgeMargin) {
        t0Prop = t0Trial;
        accept = true;
      }
    } else if (zerosCounter == 1) {
      if ((t0Trial < tc - MCMCo.AlleleAgeMargin) &&
          (abs(t0Trial - times.front()) >= MCMCo.AlleleAgeMargin)) {
        t0Prop = t0Trial;
        accept = true;
      }
    } else {
      if ((t0Trial < tc - MCMCo.AlleleAgeMargin) &&
          (abs(t0Trial - times.front()) >= MCMCo.AlleleAgeMargin) &&
          (abs(t0Trial - times.at(1)) >= MCMCo.AlleleAgeMargin)) {
        t0Prop = t0Trial;
        accept = true;
      }
    }
  }

  return t0Prop;
}

vector<double100> MCMCSampler::selProposal(const MCMCOptions &MCMCo,
                                           boost::random::mt19937 &gen) {
  if (MCMCo.selTypePriors == "Gaussian") {
    if (selectionType == 0) {
      boost::random::normal_distribution<double100> NORMAL(
          static_cast<double>(sigmaStore.back()), MCMCo.sigmaProposalSteps);
      return vector<double100>{NORMAL(gen)};
    } else if (selectionType == 1) {
      boost::random::normal_distribution<double100> sigmaNORMAL(
          static_cast<double>(sigmaStore.back()), MCMCo.sigmaProposalSteps);
      boost::random::normal_distribution<double100> hNORMAL(
          static_cast<double>(hStore.back()), MCMCo.hProposalSteps);
      return vector<double100>{hNORMAL(gen), sigmaNORMAL(gen)};
    } else {
      boost::random::normal_distribution<double100> sigmaNORMAL(
          static_cast<double>(sigmaStore.back()), MCMCo.sigmaProposalSteps);
      vector<double100> etaProp;
      vector<double100> etaCurr = etaStore.back();
      vector<double100>::const_reverse_iterator etaPS =
          MCMCo.etaProposalSteps.rbegin();
      for (vector<double100>::reverse_iterator etaC = etaCurr.rbegin();
           etaC != etaCurr.rend(); etaC++) {
        boost::random::normal_distribution<double100> NORMAL(
            static_cast<double>(*etaC), *etaPS);
        etaProp.push_back(NORMAL(gen));
        etaPS++;
      }
      etaProp.push_back(sigmaNORMAL(gen));
      return etaProp;
    }
  } else {
    if (selectionType == 0) {
      boost::random::normal_distribution<double100> NORMAL(
          static_cast<double>(sigmaStore.back()), MCMCo.sigmaProposalSteps);
      double100 sP = NORMAL(gen);
      while ((sP < MCMCo.sigmaBoundsLower) || (sP > MCMCo.sigmaBoundsUpper)) {
        sP = NORMAL(gen);
      }
      return vector<double100>{sP};
    } else if (selectionType == 1) {
      boost::random::normal_distribution<double100> sigmaNORMAL(
          static_cast<double>(sigmaStore.back()), MCMCo.sigmaProposalSteps);
      boost::random::normal_distribution<double100> hNORMAL(
          static_cast<double>(hStore.back()), MCMCo.hProposalSteps);
      double100 sP = sigmaNORMAL(gen), hP = hNORMAL(gen);
      while ((sP < MCMCo.sigmaBoundsLower) || (sP > MCMCo.sigmaBoundsUpper) ||
             (hP < MCMCo.hBoundsLower) || (hP > MCMCo.hBoundsUpper)) {
        sP = sigmaNORMAL(gen);
        hP = hNORMAL(gen);
      }
      return vector<double100>{hP, sP};
    } else {
      double100 sP;
      vector<double100> eP;
      bool redraw = true;
      while (redraw) {
        boost::random::normal_distribution<double100> sigmaNORMAL(
            static_cast<double>(sigmaStore.back()), MCMCo.sigmaProposalSteps);
        sP = sigmaNORMAL(gen);
        vector<double100> etaCurr = etaStore.back();
        bool eP_within = true;
        vector<double100>::const_reverse_iterator
            etaPS = MCMCo.etaProposalSteps.rbegin(),
            etaBL = MCMCo.etaBoundsLower.rbegin(),
            etaBU = MCMCo.etaBoundsUpper.rbegin();
        for (vector<double100>::reverse_iterator etaC = etaCurr.rbegin();
             etaC != etaCurr.rend(); etaC++) {
          boost::random::normal_distribution<double100> NORMAL(
              static_cast<double>(*etaC), *etaPS);
          eP.push_back(NORMAL(gen));
          if ((eP.back() < *etaBL) || (eP.back() > *etaBU)) {
            eP_within = false;
          }
          etaPS++;
        }
        eP.push_back(sP);
        bool sigma_within =
            ((sP < MCMCo.sigmaBoundsLower) || (sP > MCMCo.sigmaBoundsUpper));
        redraw = (sigma_within && eP_within);
      }
      return eP;
    }
  }
}

pair<bool, vector<double100>> MCMCSampler::TransitionRatioDecision(
    WrightFisher Current, double100 alpha, double100 u, double100 XNumstart,
    double100 XNumend, double100 XDenstart, double100 XDenend,
    double100 tNumstart, double100 tDenstart, double100 tNumend,
    double100 tDenend, const Options &o) {
  bool AcceptProposals;
  double100 NumL, NumU, DenL,
      DenU;  // Set up uniform variate for MH accept/reject
  vector<double> curr_theta = Current.get_Theta();
  if (!(XNumend > 0.0) || !(XNumend < 1.0) || !(XDenend > 0.0) ||
      !(XDenend < 1.0) || !(curr_theta[0] > 0.0) || !(curr_theta[1] > 0.0)) {
    double100 alphaL = Current.UnconditionedDiffusionDensity(
                           XNumstart, XNumend, tNumend - tNumstart, o) /
                       Current.UnconditionedDiffusionDensity(
                           XDenstart, XDenend, tDenend - tDenstart, o);
    if (alpha * alphaL >= u) {
      AcceptProposals = true;
    } else {
      AcceptProposals = false;
    }
  } else if (!(XNumstart > 0.0) || !(XNumstart < 1.0) || !(XDenstart > 0.0) ||
             !(XDenstart < 1.0) || !(curr_theta[0] > 0.0) ||
             !(curr_theta[1] > 0.0)) {
    double100 alphaL = Current.DiffusionDensityApproximation(
                           XNumstart, XNumend, tNumend - tNumstart, o) /
                       Current.DiffusionDensityApproximation(
                           XDenstart, XDenend, tDenend - tDenstart, o);
    if (alpha * alphaL >= u) {
      AcceptProposals = true;
    } else {
      AcceptProposals = false;
    }
  } else {
    if ((tNumend - tNumstart <= o.g1984threshold) &&
        (tDenend - tDenstart <=
         o.g1984threshold))  // If both time increments are too small, just use
                             // approximation
    {
      double100 alphaL =
          TransitionRatioApprox(Current, XNumstart, XDenstart, XNumend, XDenend,
                                tNumend - tNumstart, tDenend - tDenstart, o);

      if (alpha * alphaL >= u)  // Accept proposal
      {
        AcceptProposals = true;
      } else  // Reject
      {
        AcceptProposals = false;
      }
    } else if ((tNumend - tNumstart <= o.g1984threshold) &&
               (tDenend - tDenstart > o.g1984threshold)) {
      NumL =
          TransitionApprox(Current, XNumstart, XNumend, tNumend - tNumstart, o);
      NumU = NumL;

      pair<vector<int>, double100> C;
      C.second = tDenend - tDenstart;
      int cutoff =
          max(computeE(C),
              max(static_cast<int>(ceil(max(
                      0.0, 1.0 / (static_cast<double>(C.second)) -
                               (theta.front() + theta.back() + 1.0) / 2.0))),
                  computeK(XDenstart, XDenend, theta, o)));

      int v = -1;
      double100 tol = 0.1;
      DenL = 0.0;
      DenU = Getd1(0, XDenstart, XDenend, C.second);
      bool convergence = false;

      while (!convergence) {
        ++v;

        DenL = DenU - Getd1(2 * v + 1, XDenstart, XDenend,
                            C.second);  // d_{2v} - d_{2v+1}
        DenU = DenL + Getd1(2 * v + 2, XDenstart, XDenend, C.second);

        convergence = (v > cutoff) && (DenU - DenL < tol) && (DenL <= DenU) &&
                      (DenL >= 0.0);
      }

      double100 alphaL, alphaU;

      if (DenU == 0.0) {
        alphaL =
            (u / alpha) * 2.0;  // Set alphaL such that the proposal is accepted
      } else if (NumU == 0.0) {
        alphaU =
            (u / alpha) * 0.5;  // Set alphaU such that the proposal is rejected
      } else {
        alphaL = exp(
            log(NumL) -
            log(DenU));  // Both quantities are not 0.0 (because NumL = NumU)

        if (DenL == 0.0) {
          alphaU = (u / alpha) *
                   2.0;  // Set value so that we pass onto further refinement
        } else {
          alphaU = exp(log(NumU) - log(DenL));
        }
      }

      if (alpha * alphaL >= u)  // Accept proposal
      {
        AcceptProposals = true;
      } else if (alpha * alphaU <= u)  // Reject
      {
        AcceptProposals = false;
      } else  // Refine transition ratio
      {
        bool decision = false;
        while (!decision) {
          ++v;

          DenL = DenU - Getd1(2 * v + 1, XDenstart, XDenend, C.second);
          DenU = DenL + Getd1(2 * v + 2, XDenstart, XDenend, C.second);

          if (DenU == 0.0) {
            alphaL = (u / alpha) *
                     2.0;  // Set alphaL such that the proposal is accepted
          } else if (NumU == 0.0) {
            alphaU = (u / alpha) *
                     0.5;  // Set alphaU such that the proposal is rejected
          } else {
            alphaL = exp(log(NumL) - log(DenU));  // Both quantities are not 0.0
                                                  // (because NumL = NumU)

            if (DenL == 0.0) {
              alphaU =
                  (u / alpha) *
                  2.0;  // Set value so that we pass onto further refinement
            } else {
              alphaU = exp(log(NumU) - log(DenL));
            }
          }

          if (alpha * alphaL >= u)  // Accept proposal
          {
            decision = true;
            AcceptProposals = true;
          } else if (alpha * alphaU <= u)  // Reject
          {
            decision = true;
            AcceptProposals = false;
          }
        }
      }

    } else if ((tNumend - tNumstart > o.g1984threshold) &&
               (tDenend - tDenstart <=
                o.g1984threshold))  // If only t2-t0Curr is too small, apply
                                    // approx
    // for denominator and refinement for numerator
    {
      DenL =
          TransitionApprox(Current, XDenstart, XDenend, tDenend - tDenstart, o);
      DenU = DenL;

      pair<vector<int>, double100> C;
      C.second = tNumend - tNumstart;
      int cutoff =
          max(computeE(C),
              max(static_cast<int>(ceil(max(
                      0.0, 1.0 / static_cast<double>(C.second) -
                               (theta.front() + theta.back() + 1.0) / 2.0))),
                  computeK(XNumstart, XNumend, theta, o)));

      int v = -1;
      double100 tol = 0.1;
      NumL = 0.0;
      NumU = Getd1(0, XNumstart, XNumend, C.second);
      bool convergence = false;

      while (!convergence) {
        ++v;

        NumL = NumU - Getd1(2 * v + 1, XNumstart, XNumend,
                            C.second);  // d_{2v} - d_{2v+1}
        NumU = NumL + Getd1(2 * v + 2, XNumstart, XNumend, C.second);

        convergence = (v > cutoff) && (NumU - NumL < tol) && (NumL <= NumU) &&
                      (NumL >= 0.0);
      }

      double100 alphaL, alphaU;

      if (DenU == 0.0) {
        alphaL =
            (u / alpha) * 2.0;  // Set alphaL such that the proposal is accepted
      } else if (NumU == 0.0) {
        alphaU =
            (u / alpha) * 0.5;  // Set alphaU such that the proposal is rejected
      } else {
        if (NumL == 0.0) {
          alphaL =
              0.0;  // Set alphaL such that we pass onto to further refinement
        } else {
          alphaL = exp(log(NumL) - log(DenU));
        }

        alphaU = exp(log(NumU) - log(DenL));
      }

      if (alpha * alphaL >= u)  // Accept proposal
      {
        AcceptProposals = true;
      } else if (alpha * alphaU <= u)  // Reject
      {
        AcceptProposals = false;
      } else  // Refine transition ratio
      {
        bool decision = false;
        while (!decision) {
          ++v;

          NumL = NumU - Getd1(2 * v + 1, XNumstart, XNumend,
                              C.second);  // d_{2v} - d_{2v+1}
          NumU = NumL + Getd1(2 * v + 2, XNumstart, XNumend, C.second);

          if (DenU == 0.0) {
            alphaL = (u / alpha) *
                     2.0;  // Set alphaL such that the proposal is accepted
          } else if (NumU == 0.0) {
            alphaU = (u / alpha) *
                     0.5;  // Set alphaU such that the proposal is rejected
          } else {
            if (NumL == 0.0) {
              alphaL = 0.0;  // Set alphaL such that we pass onto to further
                             // refinement
            } else {
              alphaL = exp(log(NumL) - log(DenU));
            }

            alphaU = exp(log(NumU) - log(DenL));
          }

          if (alpha * alphaL >= u)  // Accept proposal
          {
            decision = true;
            AcceptProposals = true;
          } else if (alpha * alphaU <= u)  // Reject
          {
            decision = true;
            AcceptProposals = false;
          }
        }
      }

    } else  // Otherwise, use refinement scheme for numerator and denominator
    {
      pair<vector<int>, double100> C1, C2;
      C1.second = tNumend - tNumstart, C2.second = tDenend - tDenstart;
      int cutoff1 =
          max(computeE(C1),
              max(static_cast<int>(ceil(max(
                      0.0, 1.0 / static_cast<double>(C1.second) -
                               (theta.front() + theta.back() + 1.0) / 2.0))),
                  computeK(XNumstart, XNumend, theta, o)));
      int cutoff2 =
          max(computeE(C2),
              max(static_cast<int>(ceil(max(
                      0.0, 1.0 / static_cast<double>(C2.second) -
                               (theta.front() + theta.back() + 1.0) / 2.0))),
                  computeK(XDenstart, XDenend, theta, o)));
      int cutoff = max(cutoff1, cutoff2);

      int v = -1;
      double100 tol = 0.1;
      NumL = 0.0;
      NumU = Getd1(0, XNumstart, XNumend, C1.second);
      DenL = 0.0;
      DenU = Getd1(0, XDenstart, XDenend, C2.second);
      bool convergence = false;

      while (!convergence) {
        ++v;

        NumL = NumU - Getd1(2 * v + 1, XNumstart, XNumend,
                            C1.second);  // d_{2v} - d_{2v+1}
        NumU = NumL + Getd1(2 * v + 2, XNumstart, XNumend, C1.second);
        DenL = DenU - Getd1(2 * v + 1, XDenstart, XDenend,
                            C2.second);  // d_{2v} - d_{2v+1}
        DenU = DenL + Getd1(2 * v + 2, XDenstart, XDenend, C2.second);

        convergence = (v > cutoff) && (NumU - NumL < tol) &&
                      (DenU - DenL < tol) && (NumL <= NumU) && (DenL <= DenU) &&
                      (NumL >= 0.0) && (DenL >= 0.0);
      }

      double100 alphaL, alphaU;

      if (DenU == 0.0) {
        alphaL =
            (u / alpha) * 2.0;  // Set alphaL such that the proposal is accepted
      } else if (NumU == 0.0) {
        alphaU =
            (u / alpha) * 0.5;  // Set alphaU such that the proposal is rejected
      } else {
        if (NumL == 0.0) {
          alphaL =
              0.0;  // Set alphaL such that we pass onto to further refinement
        } else {
          alphaL = exp(log(NumL) - log(DenU));
        }

        if (DenL == 0.0) {
          alphaU = (u / alpha) *
                   2.0;  // Set value so that we pass onto further refinement
        } else {
          alphaU = exp(log(NumU) - log(DenL));
        }
      }

      if (alpha * alphaL >= u)  // Accept proposal
      {
        AcceptProposals = true;
      } else if (alpha * alphaU <= u)  // Reject
      {
        AcceptProposals = false;
      } else  // Refine transition ratio
      {
        bool decision = false;
        while (!decision) {
          ++v;

          NumL = NumU - Getd1(2 * v + 1, XNumstart, XNumend,
                              C1.second);  // d_{2v} - d_{2v+1}
          NumU = NumL + Getd1(2 * v + 2, XNumstart, XNumend, C1.second);
          DenL = DenU - Getd1(2 * v + 1, XDenstart, XDenend,
                              C2.second);  // d_{2v} - d_{2v+1}
          DenU = DenL + Getd1(2 * v + 2, XDenstart, XDenend, C2.second);

          if (DenU == 0.0) {
            alphaL = (u / alpha) *
                     2.0;  // Set alphaL such that the proposal is accepted
          } else if (NumU == 0.0) {
            alphaU = (u / alpha) *
                     0.5;  // Set alphaU such that the proposal is rejected
          } else {
            if (NumL == 0.0) {
              alphaL = 0.0;  // Set alphaL such that we pass onto to further
                             // refinement
            } else {
              alphaL = exp(log(NumL) - log(DenU));
            }

            if (DenL == 0.0) {
              alphaU =
                  (u / alpha) *
                  2.0;  // Set value so that we pass onto further refinement
            } else {
              alphaU = exp(log(NumU) - log(DenL));
            }
          }

          if (alpha * alphaL >= u)  // Accept proposal
          {
            decision = true;
            AcceptProposals = true;
          } else if (alpha * alphaU <= u)  // Reject
          {
            decision = true;
            AcceptProposals = false;
          }
        }
      }
    }
  }

  vector<double100> TransitionEstimates;
  TransitionEstimates.push_back(NumL);
  TransitionEstimates.push_back(NumU);
  TransitionEstimates.push_back(DenL);
  TransitionEstimates.push_back(DenU);

  return make_pair(AcceptProposals, TransitionEstimates);
}

void MCMCSampler::GenerateT0Z0Proposals(
    WrightFisher Current, double100 &t0Prop, vector<double100> &XProp,
    vector<int> &kappaProp, vector<vector<double100>> &omegaProp,
    vector<vector<double100>> &psiProp, vector<vector<double100>> &gammaProp,
    vector<vector<double100>> &ksiProp, double100 lambdaMax, const Options &o,
    boost::random::mt19937 &gen) {
  bool SuitableProposal = false;
  double100 tol = 5.0e-5;
  // Set up variables needed in subsequent computations

  double100 xt2 = XStore.back().at(1), t1 = times.front(),
            t2 = times.at(1);  // Setting x_{t_2}, t1, t2
  double100 phimin = Current.phiMin, phimax = Current.phiMax,
            phirange = phimax - phimin;  // Setting phimin/max

  double100 Xt1Prop = NonNeutralBridgePointBinomialObservations(
                          Current, 0.0, t0Prop, t2, xt2, o, gen, t1)
                          .first;

  while (!SuitableProposal) {
    double100 rateL = lambdaMax * (t1 - t0Prop), rateR = lambdaMax * (t2 - t1);
    boost::random::poisson_distribution<> kapL(static_cast<double>(rateL)),
        kapR(static_cast<double>(rateR));

    int kappaL = kapL(gen),
        kappaR =
            kapR(gen);  // Generate number of Poisson points to the left and
    // right of Xt1Prop

    vector<double100> PsiL(kappaL), GamL(kappaL), OmegaL(kappaL), KsiL(kappaL),
        PsiR(kappaR), GamR(kappaR), OmegaR(kappaR), KsiR(kappaR);

    boost::random::uniform_01<> unifPsiL, unifPsiR, unifGamL, unifGamR,
        unifKsiL, unifKsiR;
    auto genPsiL =
        [&t0Prop, &t1, &unifPsiL,
         &gen]()  // Routine to create uniform variates over [t0Prop,t1]
    { return (t0Prop + ((t1 - t0Prop) * unifPsiL(gen))); };
    auto genPsiR = [&t1, &t2, &unifPsiR,
                    &gen]()  // Routine to create uniform variates over [t1,t2]
    { return (t1 + ((t2 - t1) * unifPsiR(gen))); };
    bool suitable_times = false;
    // Routine to check we have distinct uniform draws, and also catch to ensure
    // we do not invoke too large a matrix in the WrightFisher::Drawbridgepoint
    while (!suitable_times) {
      suitable_times = true;
      std::generate(begin(PsiL), end(PsiL), genPsiL);
      vector<double100> copy_times(PsiL);
      sortVectorsAscending(copy_times, copy_times);
      copy_times.insert(copy_times.begin(), t0Prop);
      copy_times.push_back(t1);
      for (vector<double100>::iterator cti = copy_times.begin();
           cti != copy_times.end() - 1; cti++) {
        if (fabs(*(cti + 1) - *cti) <= tol) {
          suitable_times = false;
        }
      }
    }
    suitable_times = false;
    while (!suitable_times) {
      suitable_times = true;
      std::generate(begin(PsiR), end(PsiR), genPsiR);
      vector<double100> copy_times(PsiR);
      sortVectorsAscending(copy_times, copy_times);
      copy_times.insert(copy_times.begin(), t1);
      copy_times.push_back(t2);
      for (vector<double100>::iterator cti = copy_times.begin();
           cti != copy_times.end() - 1; cti++) {
        if (fabs(*(cti + 1) - *cti) <= tol) {
          suitable_times = false;
        }
      }
    }
    auto genGamL =
        [&unifGamL, &gen]()  // Routine to create uniform variates over [0,1]
    { return (unifGamL(gen)); };
    auto genGamR =
        [&unifGamR, &gen]()  // Routine to create uniform variates over [0,1]
    { return (unifGamR(gen)); };
    auto genKsiL =
        [&lambdaMax, &unifKsiL,
         &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
    { return (unifKsiL(gen) * lambdaMax); };
    auto genKsiR =
        [&lambdaMax, &unifKsiR,
         &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
    { return (unifKsiR(gen) * lambdaMax); };
    // Generate the uniform variates
    std::generate(begin(GamL), end(GamL), genGamL);
    std::generate(begin(GamR), end(GamR), genGamR);
    std::generate(begin(KsiL), end(KsiL), genKsiL);
    std::generate(begin(KsiR), end(KsiR), genKsiR);

    sortVectorsAscending(
        PsiL, PsiL, GamL,
        KsiL);  // Sort according to time ordering of psi_j marks
    sortVectorsAscending(PsiR, PsiR, GamR, KsiR);

    bool AcceptSkeleton = true;

    if (!PsiL.empty()) {
      for (vector<double100>::iterator PsiLiter = PsiL.begin(),
                                       GamLiter = GamL.begin(),
                                       OmegaLiter = OmegaL.begin(),
                                       KsiLiter = KsiL.begin();
           PsiLiter != PsiL.end();
           PsiLiter++, GamLiter++, OmegaLiter++, KsiLiter++) {
        if (PsiLiter == PsiL.begin()) {
          *OmegaLiter =
              Current
                  .DrawBridgepoint(0.0, Xt1Prop, t0Prop, t1, *PsiLiter, o, gen)
                  .first;  // Draw skeleton point from neutral bridge going from
          // 0 to Xt1Prop at time psi_1

          if ((((Current.Phitilde(*OmegaLiter) - phimin) / phirange) >=
               *GamLiter) &&
              (*KsiLiter < phirange)) {
            AcceptSkeleton = false;
            break;
          }
        } else {
          *OmegaLiter =
              Current
                  .DrawBridgepoint(OmegaL.back(), Xt1Prop, *(PsiLiter - 1), t1,
                                   *PsiLiter, o, gen)
                  .first;  // Draw skeleton point from neutral bridge going from
          // omega_j-1 to Xt1Prop at time psi_j

          if ((((Current.Phitilde(*OmegaLiter) - phimin) / phirange) >=
               *GamLiter) &&
              (*KsiLiter < phirange)) {
            AcceptSkeleton = false;
            break;
          }
        }
      }
    }

    if ((!PsiR.empty()) && (AcceptSkeleton)) {
      for (vector<double100>::iterator PsiRiter = PsiR.begin(),
                                       GamRiter = GamR.begin(),
                                       OmegRiter = OmegaR.begin(),
                                       KsiRiter = KsiR.begin();
           PsiRiter != PsiR.end();
           PsiRiter++, GamRiter++, OmegRiter++, KsiRiter++) {
        if (PsiRiter == PsiR.begin()) {
          *OmegRiter =
              Current.DrawBridgepoint(Xt1Prop, xt2, t1, t2, *PsiRiter, o, gen)
                  .first;  // Draw skeleton point from neutral bridge going from
          // Xt1Prop to x_{t_2} at time psi_1

          if ((((Current.Phitilde(*OmegRiter) - phimin) / phirange) >=
               *GamRiter) &&
              (*KsiRiter < phirange)) {
            AcceptSkeleton = false;
            break;
          }
        } else {
          *OmegRiter =
              Current
                  .DrawBridgepoint(OmegaR.back(), xt2, *(PsiRiter - 1), t2,
                                   *PsiRiter, o, gen)
                  .first;  // Draw skeleton point from neutral bridge going from
          // omega_j-1 to x_{t_2} at time psi_j

          if ((((Current.Phitilde(*OmegRiter) - phimin) / phirange) >=
               *GamRiter) &&
              (*KsiRiter < phirange)) {
            AcceptSkeleton = false;
            break;
          }
        }
      }
    }

    if (AcceptSkeleton) {
      // Load generated variables into prop variables

      XProp.push_back(Xt1Prop);
      kappaProp.push_back(kappaL);
      kappaProp.push_back(kappaR);
      omegaProp.push_back(OmegaL);
      omegaProp.push_back(OmegaR);
      psiProp.push_back(PsiL);
      psiProp.push_back(PsiR);
      gammaProp.push_back(GamL);
      gammaProp.push_back(GamR);
      ksiProp.push_back(KsiL);
      ksiProp.push_back(KsiR);

      SuitableProposal = true;
    }
  }
}

void MCMCSampler::AcceptT0Z0Proposals(
    WrightFisher Current, double100 &t0Prop, vector<double100> &XProp,
    vector<int> &kappaProp, vector<vector<double100>> &omegaProp,
    vector<vector<double100>> &psiProp, vector<vector<double100>> &gammaProp,
    vector<vector<double100>> &ksiProp, vector<double100> &XOut,
    vector<int> &kappaOut, vector<vector<double100>> &omegaOut,
    vector<vector<double100>> &psiOut, vector<vector<double100>> &gammaOut,
    vector<vector<double100>> &ksiOut, const Options &o,
    const MCMCOptions &MCMCo, boost::random::mt19937 &gen) {
  // Setup current values
  double100 t0Curr = t0Store.back(), t2 = times.at(1),
            xt2 = XStore.back().at(1);
  vector<double100> XCurr(XStore.back().begin(), XStore.back().begin() + 1),
      phiminmaxrange{Current.phiMin, Current.phiMax,
                     Current.phiMax - Current.phiMin};  // Setting phimin/max};
  for (vector<int>::iterator it = kappaStore.back().begin();
       it != kappaStore.back().end(); it++) {
  }
  vector<int> kappaCurr(kappaStore.back().begin(),
                        kappaStore.back().begin() + 2);
  vector<vector<double100>> omegaCurr(omegaStore.back().begin(),
                                      omegaStore.back().begin() + 2),
      psiCurr(psiStore.back().begin(), psiStore.back().begin() + 2),
      gammaCurr(gammaStore.back().begin(), gammaStore.back().begin() + 2),
      ksiCurr(ksiStore.back().begin(), ksiStore.back().begin() + 2);

  // Calculate contributions to acceptance probability

  double100 binomial_ratio =
      exp(static_cast<double100>(Data.front()) *
              (log(XProp.front()) - log(XCurr.front())) +
          static_cast<double100>((Samplesizes.front() - Data.front())) *
              (log(1.0 - XProp.front()) - log(1.0 - XCurr.front())));

  double100 alpha = binomial_ratio * t0ProposalRatio(t0Prop, t0Curr, MCMCo) *
                    t0PriorRatio(t0Prop, t0Curr, MCMCo) *
                    exp(-phiminmaxrange[0] * (t0Curr - t0Prop));
  double100 pseudo = PoissonEstimator(Current, t2 - t0Prop, 0.0, xt2, o, gen);

  double100 likelihood = exp(-phiminmaxrange[0] * (t2 - t0Prop));

  vector<double100> t_incs{times.front() - t0Prop, times.at(1) - times.front()};
  vector<double100>::iterator titer = t_incs.begin();
  for (vector<vector<double100>>::iterator PsiIter = psiProp.begin(),
                                           OmegaIter = omegaProp.begin(),
                                           KsiIter = ksiProp.begin();
       PsiIter != psiProp.begin() + 1; PsiIter++, OmegaIter++, KsiIter++) {
    int kappaCounter = 0;
    for (vector<double100>::iterator PsiiIter = (*PsiIter).begin(),
                                     OmegaiIter = (*OmegaIter).begin(),
                                     KsiiIter = (*KsiIter).begin();
         PsiiIter != (*PsiIter).end(); PsiiIter++, OmegaiIter++, KsiiIter++) {
      if ((*KsiiIter < phiminmaxrange[2]) && (*OmegaiIter >= 0.0)) {
        kappaCounter++;
        likelihood *=
            (1.0 - ((Current.Phitilde(*OmegaiIter) - phiminmaxrange[0]) /
                    phiminmaxrange[2]));
      }
    }
    double100 multiplicator =
        pow((phiminmaxrange[2] * (*titer)),
            static_cast<double100>(kappaCounter)) *
        (1.0 / boost::math::factorial<double100>(kappaCounter));
    likelihood *= exp(-phiminmaxrange[2] * (*titer)) * multiplicator;
    titer++;
  }

  alpha *= (pseudo / pseudoStore.back());

  boost::random::uniform_01<> U01;
  double100 u = U01(gen);  // Set up uniform variate for MH accept/reject

  pair<bool, vector<double100>> AccTrEst = TransitionRatioDecision(
      Current, alpha, u, 0.0, xt2, 0.0, xt2, t0Prop, t0Curr, t2, t2, o);
  bool Accept = AccTrEst.first;

  likelihood *= pow(XProp.front(), Data.front()) *
                pow(1.0 - XProp.front(), Samplesizes.front() - Data.front());

  if (Accept) {
    t0Store.push_back(t0Prop);
    XOut.insert(XOut.end(), XProp.begin(), XProp.end());
    kappaOut.insert(kappaOut.end(), kappaProp.begin(), kappaProp.end());
    omegaOut.insert(omegaOut.end(), omegaProp.begin(), omegaProp.end());
    psiOut.insert(psiOut.end(), psiProp.begin(), psiProp.end());
    gammaOut.insert(gammaOut.end(), gammaProp.begin(), gammaProp.end());
    ksiOut.insert(ksiOut.end(), ksiProp.begin(), ksiProp.end());
    pseudoStore.push_back(pseudo);
    t0Likelihood.push_back(likelihood);
  } else {
    t0Store.push_back(t0Store.back());
    XOut.insert(XOut.end(), XCurr.begin(), XCurr.end());
    kappaOut.insert(kappaOut.end(), kappaCurr.begin(), kappaCurr.end());
    omegaOut.insert(omegaOut.end(), omegaCurr.begin(), omegaCurr.end());
    psiOut.insert(psiOut.end(), psiCurr.begin(), psiCurr.end());
    gammaOut.insert(gammaOut.end(), gammaCurr.begin(), gammaCurr.end());
    ksiOut.insert(ksiOut.end(), ksiCurr.begin(), ksiCurr.end());
    pseudoStore.push_back(pseudoStore.back());
    t0Likelihood.push_back(t0Likelihood.back());
  }
}

void MCMCSampler::GenerateT0Z1Proposals(
    WrightFisher Current, int &indt0Prop, double100 &t0Prop,
    vector<double100> &XProp, vector<int> &kappaProp,
    vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
    vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
    double100 lambdaMax, const Options &o, boost::random::mt19937 &gen) {
  bool SuitableProposal = false;
  double100 tol = 5.0e-5;

  if (indt0Prop == 0) {
    // Set up variables needed in subsequent computations

    double100 xt3 = XStore.back().at(2), t1 = times.front(), t2 = times.at(1),
              t3 = times.at(2);  // Setting x_{t_2}, t1, t2
    double100 phimin = Current.phiMin, phimax = Current.phiMax,
              phirange = phimax - phimin;  // Setting phimin/max

    // Propose new X_{t_1} from WF diffusion

    double100 Xt1Prop = NonNeutralBridgePointBinomialObservations(
                            Current, 0.0, t0Prop, t3, xt3, o, gen, t1)
                            .first;

    double100 Xt2Prop = NonNeutralBridgePointBinomialObservations(
                            Current, Xt1Prop, t1, t3, xt3, o, gen, t2)
                            .first;

    //  Propose new Poisson & skeleton points

    while (!SuitableProposal) {
      double100 rate1 = lambdaMax * (t1 - t0Prop),
                rate2 = lambdaMax * (t2 - t1), rate3 = lambdaMax * (t3 - t2);
      boost::random::poisson_distribution<> kap1(static_cast<double>(rate1)),
          kap2(static_cast<double>(rate2)), kap3(static_cast<double>(rate3));

      int kappa1 = kap1(gen), kappa2 = kap2(gen),
          kappa3 = kap3(gen);  // Generate number of Poisson points to the left
      // and right of Xt1Prop

      vector<double100> Psi1(kappa1), Gam1(kappa1), Omega1(kappa1),
          Ksi1(kappa1), Psi2(kappa2), Gam2(kappa2), Omega2(kappa2),
          Ksi2(kappa2), Psi3(kappa3), Gam3(kappa3), Omega3(kappa3),
          Ksi3(kappa3);

      boost::random::uniform_01<> unifPsi1, unifPsi2, unifPsi3, unifGam1,
          unifGam2, unifGam3, unifKsi1, unifKsi2, unifKsi3;
      auto genPsi1 =
          [&t0Prop, &t1, &unifPsi1,
           &gen]()  // Routine to create uniform variates over [t0Prop,t1]
      { return (t0Prop + ((t1 - t0Prop) * unifPsi1(gen))); };
      auto genPsi2 =
          [&t1, &t2, &unifPsi2,
           &gen]()  // Routine to create uniform variates over [t1,t2]
      { return (t1 + ((t2 - t1) * unifPsi2(gen))); };
      auto genPsi3 =
          [&t2, &t3, &unifPsi3,
           &gen]()  // Routine to create uniform variates over [t1,t2]
      { return (t2 + ((t3 - t2) * unifPsi3(gen))); };
      bool suitable_times = false;
      while (!suitable_times) {
        suitable_times = true;
        std::generate(begin(Psi1), end(Psi1), genPsi1);
        vector<double100> copy_times(Psi1);
        sortVectorsAscending(copy_times, copy_times);
        copy_times.insert(copy_times.begin(), t0Prop);
        copy_times.push_back(t1);
        for (vector<double100>::iterator cti = copy_times.begin();
             cti != copy_times.end() - 1; cti++) {
          if (fabs(*(cti + 1) - *cti) <= tol) {
            suitable_times = false;
          }
        }
      }
      suitable_times = false;
      while (!suitable_times) {
        suitable_times = true;
        std::generate(begin(Psi2), end(Psi2), genPsi2);
        vector<double100> copy_times(Psi2);
        sortVectorsAscending(copy_times, copy_times);
        copy_times.insert(copy_times.begin(), t1);
        copy_times.push_back(t2);
        for (vector<double100>::iterator cti = copy_times.begin();
             cti != copy_times.end() - 1; cti++) {
          if (fabs(*(cti + 1) - *cti) <= tol) {
            suitable_times = false;
          }
        }
      }
      suitable_times = false;
      while (!suitable_times) {
        suitable_times = true;
        std::generate(begin(Psi3), end(Psi3), genPsi3);
        vector<double100> copy_times(Psi3);
        sortVectorsAscending(copy_times, copy_times);
        copy_times.insert(copy_times.begin(), t2);
        copy_times.push_back(t3);
        for (vector<double100>::iterator cti = copy_times.begin();
             cti != copy_times.end() - 1; cti++) {
          if (fabs(*(cti + 1) - *cti) <= tol) {
            suitable_times = false;
          }
        }
      }
      auto genGam1 =
          [&unifGam1, &gen]()  // Routine to create uniform variates over [0,1]
      { return (unifGam1(gen)); };
      auto genGam2 =
          [&unifGam2, &gen]()  // Routine to create uniform variates over [0,1]
      { return (unifGam2(gen)); };
      auto genGam3 =
          [&unifGam3, &gen]()  // Routine to create uniform variates over [0,1]
      { return (unifGam3(gen)); };
      auto genKsi1 =
          [&lambdaMax, &unifKsi1,
           &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
      { return (unifKsi1(gen) * lambdaMax); };
      auto genKsi2 =
          [&lambdaMax, &unifKsi2,
           &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
      { return (unifKsi2(gen) * lambdaMax); };
      auto genKsi3 =
          [&lambdaMax, &unifKsi3,
           &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
      { return (unifKsi3(gen) * lambdaMax); };
      // Generate the uniform variates
      std::generate(begin(Gam1), end(Gam1), genGam1);
      std::generate(begin(Gam2), end(Gam2), genGam2);
      std::generate(begin(Gam3), end(Gam3), genGam3);
      std::generate(begin(Ksi1), end(Ksi1), genKsi1);
      std::generate(begin(Ksi2), end(Ksi2), genKsi2);
      std::generate(begin(Ksi3), end(Ksi3), genKsi3);

      sortVectorsAscending(
          Psi1, Psi1, Gam1,
          Ksi1);  // Sort according to time ordering of psi_j marks
      sortVectorsAscending(Psi2, Psi2, Gam2, Ksi2);
      sortVectorsAscending(Psi3, Psi3, Gam3, Ksi3);

      bool AcceptSkeleton = true;

      if (!Psi1.empty()) {
        for (vector<double100>::iterator Psi1iter = Psi1.begin(),
                                         Gam1iter = Gam1.begin(),
                                         Omega1iter = Omega1.begin(),
                                         Ksi1iter = Ksi1.begin();
             Psi1iter != Psi1.end();
             Psi1iter++, Gam1iter++, Omega1iter++, Ksi1iter++) {
          if (Psi1iter == Psi1.begin()) {
            *Omega1iter =
                Current
                    .DrawBridgepoint(0.0, Xt1Prop, t0Prop, t1, *Psi1iter, o,
                                     gen)
                    .first;  // Draw skeleton point from neutral bridge
            // going from 0 to Xt1Prop at time psi_1

            if ((((Current.Phitilde(*Omega1iter) - phimin) / phirange) >=
                 *Gam1iter) &&
                (*Ksi1iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          } else {
            *Omega1iter =
                Current
                    .DrawBridgepoint(Omega1.back(), Xt1Prop, *(Psi1iter - 1),
                                     t1, *Psi1iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from omega_j-1 to Xt1Prop at time psi_j

            if ((((Current.Phitilde(*Omega1iter) - phimin) / phirange) >=
                 *Gam1iter) &&
                (*Ksi1iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          }
        }
      }

      if ((!Psi2.empty()) && (AcceptSkeleton)) {
        for (vector<double100>::iterator Psi2iter = Psi2.begin(),
                                         Gam2iter = Gam2.begin(),
                                         Omeg2iter = Omega2.begin(),
                                         Ksi2iter = Ksi2.begin();
             Psi2iter != Psi2.end();
             Psi2iter++, Gam2iter++, Omeg2iter++, Ksi2iter++) {
          if (Psi2iter == Psi2.begin()) {
            *Omeg2iter =
                Current
                    .DrawBridgepoint(Xt1Prop, Xt2Prop, t1, t2, *Psi2iter, o,
                                     gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from Xt1Prop to x_{t_2} at time psi_1

            if ((((Current.Phitilde(*Omeg2iter) - phimin) / phirange) >=
                 *Gam2iter) &&
                (*Ksi2iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          } else {
            *Omeg2iter =
                Current
                    .DrawBridgepoint(Omega2.back(), Xt2Prop, *(Psi2iter - 1),
                                     t2, *Psi2iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from omega_j-1 to x_{t_2} at time psi_j

            if ((((Current.Phitilde(*Omeg2iter) - phimin) / phirange) >=
                 *Gam2iter) &&
                (*Ksi2iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          }
        }
      }

      if ((!Psi3.empty()) && (AcceptSkeleton)) {
        for (vector<double100>::iterator Psi3iter = Psi3.begin(),
                                         Gam3iter = Gam3.begin(),
                                         Omeg3iter = Omega3.begin(),
                                         Ksi3iter = Ksi3.begin();
             Psi3iter != Psi3.end();
             Psi3iter++, Gam3iter++, Omeg3iter++, Ksi3iter++) {
          if (Psi3iter == Psi3.begin()) {
            *Omeg3iter =
                Current.DrawBridgepoint(Xt2Prop, xt3, t2, t3, *Psi3iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from Xt1Prop to x_{t_2} at time psi_1

            if ((((Current.Phitilde(*Omeg3iter) - phimin) / phirange) >=
                 *Gam3iter) &&
                (*Ksi3iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          } else {
            *Omeg3iter =
                Current
                    .DrawBridgepoint(Omega3.back(), xt3, *(Psi3iter - 1), t3,
                                     *Psi3iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from omega_j-1 to x_{t_2} at time psi_j

            if ((((Current.Phitilde(*Omeg3iter) - phimin) / phirange) >=
                 *Gam3iter) &&
                (*Ksi3iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          }
        }
      }

      if (AcceptSkeleton) {
        // Load generated variables into prop variables

        XProp.push_back(Xt1Prop);
        XProp.push_back(Xt2Prop);
        kappaProp.push_back(kappa1);
        kappaProp.push_back(kappa2);
        kappaProp.push_back(kappa3);
        omegaProp.push_back(Omega1);
        omegaProp.push_back(Omega2);
        omegaProp.push_back(Omega3);
        psiProp.push_back(Psi1);
        psiProp.push_back(Psi2);
        psiProp.push_back(Psi3);
        gammaProp.push_back(Gam1);
        gammaProp.push_back(Gam2);
        gammaProp.push_back(Gam3);
        ksiProp.push_back(Ksi1);
        ksiProp.push_back(Ksi2);
        ksiProp.push_back(Ksi3);

        SuitableProposal = true;
      }
    }
  } else {
    // Set up variables needed in subsequent computations

    double100 xt3 = XStore.back().at(2), t2 = times.at(1),
              t3 = times.at(2);  // Setting x_{t_2}, t1, t2
    double100 phimin = Current.phiMin, phimax = Current.phiMax,
              phirange = phimax - phimin;  // Setting phimin/max

    // Propose new X_{t_2} from WF bridge

    double100 Xt2Prop = NonNeutralBridgePointBinomialObservations(
                            Current, 0.0, t0Prop, t3, xt3, o, gen, t2)
                            .first;

    while (!SuitableProposal) {
      double100 rate1 = lambdaMax * (t2 - t0Prop),
                rate2 = lambdaMax * (t3 - t2);
      boost::random::poisson_distribution<> kap1(static_cast<double>(rate1)),
          kap2(static_cast<double>(rate2));
      double100 tol = 5.0e-5;

      int kappa1 = kap1(gen),
          kappa2 = kap2(gen);  // Generate number of Poisson points

      vector<double100> Psi1(kappa1), Gam1(kappa1), Omega1(kappa1),
          Ksi1(kappa1), Psi2(kappa2), Gam2(kappa2), Omega2(kappa2),
          Ksi2(kappa2);

      boost::random::uniform_01<> unifPsi1, unifGam1, unifKsi1, unifPsi2,
          unifGam2, unifKsi2;
      auto genPsi1 =
          [&t0Prop, &t2, &unifPsi1,
           &gen]()  // Routine to create uniform variates over [t0Prop,t2]
      { return (t0Prop + ((t2 - t0Prop) * unifPsi1(gen))); };
      auto genPsi2 =
          [&t2, &t3, &unifPsi2,
           &gen]()  // Routine to create uniform variates over [t0Prop,t2]
      { return (t2 + ((t3 - t2) * unifPsi2(gen))); };
      bool suitable_times = false;
      while (!suitable_times) {
        suitable_times = true;
        std::generate(begin(Psi1), end(Psi1), genPsi1);
        vector<double100> copy_times(Psi1);
        sortVectorsAscending(copy_times, copy_times);
        copy_times.insert(copy_times.begin(), t0Prop);
        copy_times.push_back(t2);
        for (vector<double100>::iterator cti = copy_times.begin();
             cti != copy_times.end() - 1; cti++) {
          if (fabs(*(cti + 1) - *cti) <= tol) {
            suitable_times = false;
          }
        }
      }
      suitable_times = false;
      while (!suitable_times) {
        suitable_times = true;
        std::generate(begin(Psi2), end(Psi2), genPsi2);
        vector<double100> copy_times(Psi2);
        sortVectorsAscending(copy_times, copy_times);
        copy_times.insert(copy_times.begin(), t2);
        copy_times.push_back(t3);
        for (vector<double100>::iterator cti = copy_times.begin();
             cti != copy_times.end() - 1; cti++) {
          if (fabs(*(cti + 1) - *cti) <= tol) {
            suitable_times = false;
          }
        }
      }
      auto genGam1 =
          [&unifGam1, &gen]()  // Routine to create uniform variates over [0,1]
      { return (unifGam1(gen)); };
      auto genGam2 =
          [&unifGam2, &gen]()  // Routine to create uniform variates over [0,1]
      { return (unifGam2(gen)); };
      auto genKsi1 =
          [&lambdaMax, &unifKsi1,
           &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
      { return (unifKsi1(gen) * lambdaMax); };
      auto genKsi2 =
          [&lambdaMax, &unifKsi2,
           &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
      { return (unifKsi2(gen) * lambdaMax); };

      // Generate the uniform variates
      std::generate(begin(Gam1), end(Gam1), genGam1);
      std::generate(begin(Gam2), end(Gam2), genGam2);
      std::generate(begin(Ksi1), end(Ksi1), genKsi1);
      std::generate(begin(Ksi2), end(Ksi2), genKsi2);

      sortVectorsAscending(
          Psi1, Psi1, Gam1,
          Ksi1);  // Sort according to time ordering of psi_j marks
      sortVectorsAscending(Psi2, Psi2, Gam2, Ksi2);

      bool AcceptSkeleton = true;

      if (!Psi1.empty()) {
        for (vector<double100>::iterator Psi1iter = Psi1.begin(),
                                         Gam1iter = Gam1.begin(),
                                         Omega1iter = Omega1.begin(),
                                         Ksi1iter = Ksi1.begin();
             Psi1iter != Psi1.end();
             Psi1iter++, Gam1iter++, Omega1iter++, Ksi1iter++) {
          if (Psi1iter == Psi1.begin()) {
            *Omega1iter =
                Current
                    .DrawBridgepoint(0.0, Xt2Prop, t0Prop, t2, *Psi1iter, o,
                                     gen)
                    .first;  // Draw skeleton point from neutral bridge
            // going from 0 to Xt1Prop at time psi_1

            if ((((Current.Phitilde(*Omega1iter) - phimin) / phirange) >=
                 *Gam1iter) &&
                (*Ksi1iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          } else {
            *Omega1iter =
                Current
                    .DrawBridgepoint(Omega1.back(), Xt2Prop, *(Psi1iter - 1),
                                     t2, *Psi1iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from omega_j-1 to Xt1Prop at time psi_j

            if ((((Current.Phitilde(*Omega1iter) - phimin) / phirange) >=
                 *Gam1iter) &&
                (*Ksi1iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          }
        }
      }

      if ((!Psi2.empty()) && (AcceptSkeleton)) {
        for (vector<double100>::iterator Psi2iter = Psi2.begin(),
                                         Gam2iter = Gam2.begin(),
                                         Omega2iter = Omega2.begin(),
                                         Ksi2iter = Ksi2.begin();
             Psi2iter != Psi2.end();
             Psi2iter++, Gam2iter++, Omega2iter++, Ksi2iter++) {
          if (Psi2iter == Psi2.begin()) {
            *Omega2iter =
                Current.DrawBridgepoint(Xt2Prop, xt3, t2, t3, *Psi2iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from 0 to Xt1Prop at time psi_1

            if ((((Current.Phitilde(*Omega2iter) - phimin) / phirange) >=
                 *Gam2iter) &&
                (*Ksi2iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          } else {
            *Omega2iter =
                Current
                    .DrawBridgepoint(Omega2.back(), xt3, *(Psi2iter - 1), t3,
                                     *Psi2iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from omega_j-1 to Xt1Prop at time psi_j

            if ((((Current.Phitilde(*Omega2iter) - phimin) / phirange) >=
                 *Gam2iter) &&
                (*Ksi2iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          }
        }
      }

      if (AcceptSkeleton) {
        // Load generated variables into prop variables

        XProp.push_back(0.0);
        XProp.push_back(Xt2Prop);
        kappaProp.push_back(0);
        kappaProp.push_back(kappa1);
        kappaProp.push_back(kappa2);
        vector<double100> emptyvec;
        omegaProp.push_back(emptyvec);
        omegaProp.push_back(Omega1);
        omegaProp.push_back(Omega2);
        psiProp.push_back(emptyvec);
        psiProp.push_back(Psi1);
        psiProp.push_back(Psi2);
        gammaProp.push_back(emptyvec);
        gammaProp.push_back(Gam1);
        gammaProp.push_back(Gam2);
        ksiProp.push_back(emptyvec);
        ksiProp.push_back(Ksi1);
        ksiProp.push_back(Ksi2);

        SuitableProposal = true;
      }
    }
  }
}

void MCMCSampler::AcceptT0Z1Proposals(
    WrightFisher Current, int &indt0Prop, double100 &t0Prop,
    vector<double100> &XProp, vector<int> &kappaProp,
    vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
    vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
    vector<double100> &XOut, vector<int> &kappaOut,
    vector<vector<double100>> &omegaOut, vector<vector<double100>> &psiOut,
    vector<vector<double100>> &gammaOut, vector<vector<double100>> &ksiOut,
    const Options &o, const MCMCOptions &MCMCo, boost::random::mt19937 &gen) {
  // Load in current values

  double100 t0Curr = t0Store.back(), t3 = times.at(2),
            xt3 = XStore.back().at(2), pseudo;
  vector<double100> XCurr(XStore.back().begin(), XStore.back().begin() + 2),
      phiminmaxrange{Current.phiMin, Current.phiMax,
                     Current.phiMax - Current.phiMin};
  vector<int> kappaCurr(kappaStore.back().begin(),
                        kappaStore.back().begin() + 3);
  vector<vector<double100>> omegaCurr(omegaStore.back().begin(),
                                      omegaStore.back().begin() + 3),
      psiCurr(psiStore.back().begin(), psiStore.back().begin() + 3),
      gammaCurr(gammaStore.back().begin(), gammaStore.back().begin() + 3),
      ksiCurr(ksiStore.back().begin(), ksiStore.back().begin() + 3);

  double100 binomial_ratio =
      pow((XProp.front() / XCurr.front()),
          static_cast<double100>(Data.front())) *
      pow((1.0 - XProp.front()) / (1.0 - XCurr.front()),
          static_cast<double100>(Samplesizes.front())) *
      pow(XProp.back() / XCurr.back(), static_cast<double100>(Data.at(1))) *
      pow((1.0 - XProp.back()) / (1.0 - XCurr.back()),
          static_cast<double100>(Samplesizes.at(1) - Data.at(1)));
  double100 alpha = t0ProposalRatio(t0Prop, t0Curr, MCMCo) *
                    t0PriorRatio(t0Prop, t0Curr, MCMCo) *
                    exp(-phiminmaxrange[0] * (t0Curr - t0Prop)) *
                    binomial_ratio;

  boost::random::uniform_01<> U01;
  double100 u = U01(gen),
            likelihood =
                exp(-phiminmaxrange[0] * (t3 - t0Prop)) *
                pow(XProp.front(), static_cast<double100>(Data.front())) *
                pow(1.0 - XProp.front(),
                    static_cast<double100>(Samplesizes.front() -
                                           Data.front())) *
                pow(XProp.at(1), static_cast<double100>(Data.at(1))) *
                pow(1.0 - XProp.at(1),
                    static_cast<double100>(Samplesizes.at(1) - Data.at(1)));

  pseudo = PoissonEstimator(Current, t3 - t0Prop, 0.0, xt3, o, gen);

  if (indt0Prop == 0) {
    vector<double100> t_incs{times.front() - t0Prop,
                             times.at(1) - times.front(),
                             times.at(2) - times.at(1)};
    vector<double100>::iterator titer = t_incs.begin();
    for (vector<vector<double100>>::iterator PsiIter = psiProp.begin(),
                                             OmegaIter = omegaProp.begin(),
                                             KsiIter = ksiProp.begin();
         titer != t_incs.end(); PsiIter++, OmegaIter++, KsiIter++) {
      int kappaCounter = 0;
      for (vector<double100>::iterator PsiiIter = (*PsiIter).begin(),
                                       OmegaiIter = (*OmegaIter).begin(),
                                       KsiiIter = (*KsiIter).begin();
           PsiiIter != (*PsiIter).end(); PsiiIter++, OmegaiIter++, KsiiIter++) {
        if ((*KsiiIter < phiminmaxrange[2]) && (*OmegaiIter >= 0.0)) {
          kappaCounter++;
          likelihood *=
              (1.0 - ((Current.Phitilde(*OmegaiIter) - phiminmaxrange[0]) /
                      phiminmaxrange[2]));
        }
      }
      double100 multiplicator =
          pow((phiminmaxrange[2] * (*titer)),
              static_cast<double100>(kappaCounter)) *
          (1.0 / boost::math::factorial<double100>(kappaCounter));
      likelihood *= exp(-phiminmaxrange[2] * (*titer)) * multiplicator;
      titer++;
    }
  } else {
    vector<double100> t_incs{0.0, times.at(1) - t0Prop,
                             times.at(2) - times.at(1)};
    vector<double100>::iterator titer = t_incs.begin();
    for (vector<vector<double100>>::iterator PsiIter = psiProp.begin(),
                                             OmegaIter = omegaProp.begin(),
                                             KsiIter = ksiProp.begin();
         titer != t_incs.end(); PsiIter++, OmegaIter++, KsiIter++) {
      int kappaCounter = 0;
      for (vector<double100>::iterator PsiiIter = (*PsiIter).begin(),
                                       OmegaiIter = (*OmegaIter).begin(),
                                       KsiiIter = (*KsiIter).begin();
           PsiiIter != (*PsiIter).end(); PsiiIter++, OmegaiIter++, KsiiIter++) {
        if ((*KsiiIter < phiminmaxrange[2]) && (*OmegaiIter >= 0.0)) {
          kappaCounter++;
          likelihood *=
              (1.0 - ((Current.Phitilde(*OmegaiIter) - phiminmaxrange[0]) /
                      phiminmaxrange[2]));
        }
      }
      double100 multiplicator =
          pow((phiminmaxrange[2] * (*titer)),
              static_cast<double100>(kappaCounter)) *
          (1.0 / boost::math::factorial<double100>(kappaCounter));
      likelihood *= exp(-phiminmaxrange[2] * (*titer)) * multiplicator;
      titer++;
    }
  }

  alpha *= (pseudo / pseudoStore.back());

  pair<bool, vector<double100>> AccTrEst = TransitionRatioDecision(
      Current, alpha, u, 0.0, xt3, 0.0, xt3, t0Prop, t0Curr, t3, t3, o);

  bool Accept = AccTrEst.first;

  if (Accept) {
    t0Store.push_back(t0Prop);
    XOut.insert(XOut.end(), XProp.begin(), XProp.end());
    kappaOut.insert(kappaOut.end(), kappaProp.begin(), kappaProp.end());
    omegaOut.insert(omegaOut.end(), omegaProp.begin(), omegaProp.end());
    psiOut.insert(psiOut.end(), psiProp.begin(), psiProp.end());
    gammaOut.insert(gammaOut.end(), gammaProp.begin(), gammaProp.end());
    ksiOut.insert(ksiOut.end(), ksiProp.begin(), ksiProp.end());
    pseudoStore.push_back(pseudo);
    t0Likelihood.push_back(likelihood);
  } else {
    t0Store.push_back(t0Store.back());
    XOut.insert(XOut.end(), XCurr.begin(), XCurr.end());
    kappaOut.insert(kappaOut.end(), kappaCurr.begin(), kappaCurr.end());
    omegaOut.insert(omegaOut.end(), omegaCurr.begin(), omegaCurr.end());
    psiOut.insert(psiOut.end(), psiCurr.begin(), psiCurr.end());
    gammaOut.insert(gammaOut.end(), gammaCurr.begin(), gammaCurr.end());
    ksiOut.insert(ksiOut.end(), ksiCurr.begin(), ksiCurr.end());
    pseudoStore.push_back(pseudoStore.back());
    t0Likelihood.push_back(t0Likelihood.back());
  }
}

void MCMCSampler::GenerateT0Z2Proposals(
    WrightFisher Current, int &indt0Prop, double100 &t0Prop,
    vector<double100> &XProp, vector<int> &kappaProp,
    vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
    vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
    double100 lambdaMax, const Options &o, boost::random::mt19937 &gen) {
  bool SuitableProposal = false;
  double100 tol = 5.0e-5;

  if (indt0Prop == 0) {
    // Set up variables needed in subsequent computations

    double100 xt4 = XStore.back().at(3), t1 = times.front(), t2 = times.at(1),
              t3 = times.at(2), t4 = times.at(3);  // Setting x_{t_2}, t1, t2
    double100 phimin = Current.phiMin, phimax = Current.phiMax,
              phirange = phimax - phimin;  // Setting phimin/max

    // Propose new X_{t_1}, X_{t_2} from diffusion

    double100 Xt1Prop = NonNeutralBridgePointBinomialObservations(
                            Current, 0.0, t0Prop, t4, xt4, o, gen, t1)
                            .first;

    double100 Xt2Prop = NonNeutralBridgePointBinomialObservations(
                            Current, Xt1Prop, t1, t4, xt4, o, gen, t2)
                            .first;

    // Propose new X_{t_3} from bridge

    double100 Xt3Prop = NonNeutralBridgePointBinomialObservations(
                            Current, Xt2Prop, t2, t4, xt4, o, gen, t3)
                            .first;  //, Yt3, nt3).first;

    while (!SuitableProposal) {
      // Propose new Poisson & skeleton points

      double100 rate1 = lambdaMax * (t1 - t0Prop),
                rate2 = lambdaMax * (t2 - t1), rate3 = lambdaMax * (t3 - t2),
                rate4 = lambdaMax * (t4 - t3);
      boost::random::poisson_distribution<> kap1(static_cast<double>(rate1)),
          kap2(static_cast<double>(rate2)), kap3(static_cast<double>(rate3)),
          kap4(static_cast<double>(rate4));

      int kappa1 = kap1(gen), kappa2 = kap2(gen), kappa3 = kap3(gen),
          kappa4 = kap4(gen);  // Generate number of Poisson points to the left
      // and right of Xt1Prop

      vector<double100> Psi1(kappa1), Gam1(kappa1), Omega1(kappa1),
          Ksi1(kappa1), Psi2(kappa2), Gam2(kappa2), Omega2(kappa2),
          Ksi2(kappa2), Psi3(kappa3), Gam3(kappa3), Omega3(kappa3),
          Ksi3(kappa3), Psi4(kappa4), Gam4(kappa4), Omega4(kappa4),
          Ksi4(kappa4);

      boost::random::uniform_01<> unifPsi1, unifPsi2, unifPsi3, unifPsi4,
          unifGam1, unifGam2, unifGam3, unifGam4, unifKsi1, unifKsi2, unifKsi3,
          unifKsi4;
      auto genPsi1 =
          [&t0Prop, &t1, &unifPsi1,
           &gen]()  // Routine to create uniform variates over [t0Prop,t1]
      { return (t0Prop + ((t1 - t0Prop) * unifPsi1(gen))); };
      auto genPsi2 =
          [&t1, &t2, &unifPsi2,
           &gen]()  // Routine to create uniform variates over [t1,t2]
      { return (t1 + ((t2 - t1) * unifPsi2(gen))); };
      auto genPsi3 =
          [&t2, &t3, &unifPsi3,
           &gen]()  // Routine to create uniform variates over [t2,t3]
      { return (t2 + ((t3 - t2) * unifPsi3(gen))); };
      auto genPsi4 =
          [&t3, &t4, &unifPsi4,
           &gen]()  // Routine to create uniform variates over [t2,t3]
      { return (t3 + ((t4 - t3) * unifPsi4(gen))); };
      bool suitable_times = false;
      while (!suitable_times) {
        suitable_times = true;
        std::generate(begin(Psi1), end(Psi1), genPsi1);
        vector<double100> copy_times(Psi1);
        sortVectorsAscending(copy_times, copy_times);
        copy_times.insert(copy_times.begin(), t0Prop);
        copy_times.push_back(t1);
        for (vector<double100>::iterator cti = copy_times.begin();
             cti != copy_times.end() - 1; cti++) {
          if (fabs(*(cti + 1) - *cti) <= tol) {
            suitable_times = false;
          }
        }
      }
      suitable_times = false;
      while (!suitable_times) {
        suitable_times = true;
        std::generate(begin(Psi2), end(Psi2), genPsi2);
        vector<double100> copy_times(Psi2);
        sortVectorsAscending(copy_times, copy_times);
        copy_times.insert(copy_times.begin(), t1);
        copy_times.push_back(t2);
        for (vector<double100>::iterator cti = copy_times.begin();
             cti != copy_times.end() - 1; cti++) {
          if (fabs(*(cti + 1) - *cti) <= tol) {
            suitable_times = false;
          }
        }
      }
      suitable_times = false;
      while (!suitable_times) {
        suitable_times = true;
        std::generate(begin(Psi3), end(Psi3), genPsi3);
        vector<double100> copy_times(Psi3);
        sortVectorsAscending(copy_times, copy_times);
        copy_times.insert(copy_times.begin(), t2);
        copy_times.push_back(t3);
        for (vector<double100>::iterator cti = copy_times.begin();
             cti != copy_times.end() - 1; cti++) {
          if (fabs(*(cti + 1) - *cti) <= tol) {
            suitable_times = false;
          }
        }
      }
      suitable_times = false;
      while (!suitable_times) {
        suitable_times = true;
        std::generate(begin(Psi4), end(Psi4), genPsi4);
        vector<double100> copy_times(Psi4);
        sortVectorsAscending(copy_times, copy_times);
        copy_times.insert(copy_times.begin(), t3);
        copy_times.push_back(t4);
        for (vector<double100>::iterator cti = copy_times.begin();
             cti != copy_times.end() - 1; cti++) {
          if (fabs(*(cti + 1) - *cti) <= tol) {
            suitable_times = false;
          }
        }
      }
      auto genGam1 =
          [&unifGam1, &gen]()  // Routine to create uniform variates over [0,1]
      { return (unifGam1(gen)); };
      auto genGam2 =
          [&unifGam2, &gen]()  // Routine to create uniform variates over [0,1]
      { return (unifGam2(gen)); };
      auto genGam3 =
          [&unifGam3, &gen]()  // Routine to create uniform variates over [0,1]
      { return (unifGam3(gen)); };
      auto genGam4 =
          [&unifGam4, &gen]()  // Routine to create uniform variates over [0,1]
      { return (unifGam4(gen)); };
      auto genKsi1 =
          [&lambdaMax, &unifKsi1,
           &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
      { return (unifKsi1(gen) * lambdaMax); };
      auto genKsi2 =
          [&lambdaMax, &unifKsi2,
           &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
      { return (unifKsi2(gen) * lambdaMax); };
      auto genKsi3 =
          [&lambdaMax, &unifKsi3,
           &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
      { return (unifKsi3(gen) * lambdaMax); };
      auto genKsi4 =
          [&lambdaMax, &unifKsi4,
           &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
      { return (unifKsi4(gen) * lambdaMax); };

      // Generate the uniform variates
      std::generate(begin(Gam1), end(Gam1), genGam1);
      std::generate(begin(Gam2), end(Gam2), genGam2);
      std::generate(begin(Gam3), end(Gam3), genGam3);
      std::generate(begin(Gam4), end(Gam4), genGam4);
      std::generate(begin(Ksi1), end(Ksi1), genKsi1);
      std::generate(begin(Ksi2), end(Ksi2), genKsi2);
      std::generate(begin(Ksi3), end(Ksi3), genKsi3);
      std::generate(begin(Ksi4), end(Ksi4), genKsi4);

      sortVectorsAscending(
          Psi1, Psi1, Gam1,
          Ksi1);  // Sort according to time ordering of psi_j marks
      sortVectorsAscending(Psi2, Psi2, Gam2, Ksi2);
      sortVectorsAscending(Psi3, Psi3, Gam3, Ksi3);
      sortVectorsAscending(Psi4, Psi4, Gam4, Ksi4);

      bool AcceptSkeleton = true;

      if (!Psi1.empty()) {
        for (vector<double100>::iterator Psi1iter = Psi1.begin(),
                                         Gam1iter = Gam1.begin(),
                                         Omega1iter = Omega1.begin(),
                                         Ksi1iter = Ksi1.begin();
             Psi1iter != Psi1.end();
             Psi1iter++, Gam1iter++, Omega1iter++, Ksi1iter++) {
          if (Psi1iter == Psi1.begin()) {
            *Omega1iter =
                Current
                    .DrawBridgepoint(0.0, Xt1Prop, t0Prop, t1, *Psi1iter, o,
                                     gen)
                    .first;  // Draw skeleton point from neutral bridge
            // going from 0 to Xt1Prop at time psi_1

            if ((((Current.Phitilde(*Omega1iter) - phimin) / phirange) >=
                 *Gam1iter) &&
                (*Ksi1iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          } else {
            *Omega1iter =
                Current
                    .DrawBridgepoint(Omega1.back(), Xt1Prop, *(Psi1iter - 1),
                                     t1, *Psi1iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from omega_j-1 to Xt1Prop at time psi_j

            if ((((Current.Phitilde(*Omega1iter) - phimin) / phirange) >=
                 *Gam1iter) &&
                (*Ksi1iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          }
        }
      }

      if ((!Psi2.empty()) && (AcceptSkeleton)) {
        for (vector<double100>::iterator Psi2iter = Psi2.begin(),
                                         Gam2iter = Gam2.begin(),
                                         Omeg2iter = Omega2.begin(),
                                         Ksi2iter = Ksi2.begin();
             Psi2iter != Psi2.end();
             Psi2iter++, Gam2iter++, Omeg2iter++, Ksi2iter++) {
          if (Psi2iter == Psi2.begin()) {
            *Omeg2iter =
                Current
                    .DrawBridgepoint(Xt1Prop, Xt2Prop, t1, t2, *Psi2iter, o,
                                     gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from Xt1Prop to x_{t_2} at time psi_1

            if ((((Current.Phitilde(*Omeg2iter) - phimin) / phirange) >=
                 *Gam2iter) &&
                (*Ksi2iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          } else {
            *Omeg2iter =
                Current
                    .DrawBridgepoint(Omega2.back(), Xt2Prop, *(Psi2iter - 1),
                                     t2, *Psi2iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from omega_j-1 to x_{t_2} at time psi_j

            if ((((Current.Phitilde(*Omeg2iter) - phimin) / phirange) >=
                 *Gam2iter) &&
                (*Ksi2iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          }
        }
      }

      if ((!Psi3.empty()) && (AcceptSkeleton)) {
        for (vector<double100>::iterator Psi3iter = Psi3.begin(),
                                         Gam3iter = Gam3.begin(),
                                         Omeg3iter = Omega3.begin(),
                                         Ksi3iter = Ksi3.begin();
             Psi3iter != Psi3.end();
             Psi3iter++, Gam3iter++, Omeg3iter++, Ksi3iter++) {
          if (Psi3iter == Psi3.begin()) {
            *Omeg3iter =
                Current
                    .DrawBridgepoint(Xt2Prop, Xt3Prop, t2, t3, *Psi3iter, o,
                                     gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from Xt1Prop to x_{t_2} at time psi_1

            if ((((Current.Phitilde(*Omeg3iter) - phimin) / phirange) >=
                 *Gam3iter) &&
                (*Ksi3iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          } else {
            *Omeg3iter =
                Current
                    .DrawBridgepoint(Omega3.back(), Xt3Prop, *(Psi3iter - 1),
                                     t3, *Psi3iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from omega_j-1 to x_{t_2} at time psi_j

            if ((((Current.Phitilde(*Omeg3iter) - phimin) / phirange) >=
                 *Gam3iter) &&
                (*Ksi3iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          }
        }
      }

      if ((!Psi4.empty()) && (AcceptSkeleton)) {
        for (vector<double100>::iterator Psi4iter = Psi4.begin(),
                                         Gam4iter = Gam4.begin(),
                                         Omeg4iter = Omega4.begin(),
                                         Ksi4iter = Ksi4.begin();
             Psi4iter != Psi4.end();
             Psi4iter++, Gam4iter++, Omeg4iter++, Ksi4iter++) {
          if (Psi4iter == Psi4.begin()) {
            *Omeg4iter =
                Current.DrawBridgepoint(Xt3Prop, xt4, t3, t4, *Psi4iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from Xt1Prop to x_{t_2} at time psi_1

            if ((((Current.Phitilde(*Omeg4iter) - phimin) / phirange) >=
                 *Gam4iter) &&
                (*Ksi4iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          } else {
            *Omeg4iter =
                Current
                    .DrawBridgepoint(Omega4.back(), xt4, *(Psi4iter - 1), t4,
                                     *Psi4iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from omega_j-1 to x_{t_2} at time psi_j

            if ((((Current.Phitilde(*Omeg4iter) - phimin) / phirange) >=
                 *Gam4iter) &&
                (*Ksi4iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          }
        }
      }

      if (AcceptSkeleton) {
        // Load generated variables into prop variables

        XProp.push_back(Xt1Prop);
        XProp.push_back(Xt2Prop);
        XProp.push_back(Xt3Prop);
        kappaProp.push_back(kappa1);
        kappaProp.push_back(kappa2);
        kappaProp.push_back(kappa3);
        kappaProp.push_back(kappa4);
        omegaProp.push_back(Omega1);
        omegaProp.push_back(Omega2);
        omegaProp.push_back(Omega3);
        omegaProp.push_back(Omega4);
        psiProp.push_back(Psi1);
        psiProp.push_back(Psi2);
        psiProp.push_back(Psi3);
        psiProp.push_back(Psi4);
        gammaProp.push_back(Gam1);
        gammaProp.push_back(Gam2);
        gammaProp.push_back(Gam3);
        gammaProp.push_back(Gam4);
        ksiProp.push_back(Ksi1);
        ksiProp.push_back(Ksi2);
        ksiProp.push_back(Ksi3);
        ksiProp.push_back(Ksi4);

        SuitableProposal = true;
      }
    }
  } else if (indt0Prop == 1) {
    // Set up variables needed in subsequent computations

    double100 xt4 = XStore.back().at(3), t2 = times.at(1), t3 = times.at(2),
              t4 = times.at(3);  // Setting x_{t_2}, t1, t2
    double100 phimin = Current.phiMin, phimax = Current.phiMax,
              phirange = phimax - phimin;
    // Setting phimin/max

    // Propose new X_{t_2} from diffusion, X_{t_3} from bridge

    double100 Xt2Prop = NonNeutralBridgePointBinomialObservations(
                            Current, 0.0, t0Prop, t4, xt4, o, gen, t2)
                            .first;

    // Propose new X_{t_3} from bridge

    double100 Xt3Prop = NonNeutralBridgePointBinomialObservations(
                            Current, Xt2Prop, t2, t4, xt4, o, gen, t3)
                            .first;

    while (!SuitableProposal) {
      // Propose new Poisson & skeleton points

      double100 rate2 = lambdaMax * (t2 - t0Prop),
                rate3 = lambdaMax * (t3 - t2), rate4 = lambdaMax * (t4 - t3);
      boost::random::poisson_distribution<> kap2(static_cast<double>(rate2)),
          kap3(static_cast<double>(rate3)), kap4(static_cast<double>(rate4));

      int kappa2 = kap2(gen), kappa3 = kap3(gen),
          kappa4 = kap4(gen);  // Generate number of Poisson points to the left
      // and right of Xt1Prop
      double100 tol = 5.0e-5;

      vector<double100> Psi2(kappa2), Gam2(kappa2), Omega2(kappa2),
          Ksi2(kappa2), Psi3(kappa3), Gam3(kappa3), Omega3(kappa3),
          Ksi3(kappa3), Psi4(kappa4), Gam4(kappa4), Omega4(kappa4),
          Ksi4(kappa4);

      boost::random::uniform_01<> unifPsi2, unifPsi3, unifPsi4, unifGam2,
          unifGam3, unifGam4, unifKsi2, unifKsi3, unifKsi4;
      auto genPsi2 =
          [&t0Prop, &t2, &unifPsi2,
           &gen]()  // Routine to create uniform variates over [t0Prop,t2]
      { return (t0Prop + ((t2 - t0Prop) * unifPsi2(gen))); };
      auto genPsi3 =
          [&t2, &t3, &unifPsi3,
           &gen]()  // Routine to create uniform variates over [t2,t3]
      { return (t2 + ((t3 - t2) * unifPsi3(gen))); };
      auto genPsi4 =
          [&t3, &t4, &unifPsi4,
           &gen]()  // Routine to create uniform variates over [t3,t4]
      { return (t3 + ((t4 - t3) * unifPsi4(gen))); };
      bool suitable_times = false;
      while (!suitable_times) {
        suitable_times = true;
        std::generate(begin(Psi2), end(Psi2), genPsi2);
        vector<double100> copy_times(Psi2);
        sortVectorsAscending(copy_times, copy_times);
        copy_times.insert(copy_times.begin(), t0Prop);
        copy_times.push_back(t2);
        for (vector<double100>::iterator cti = copy_times.begin();
             cti != copy_times.end() - 1; cti++) {
          if (fabs(*(cti + 1) - *cti) <= tol) {
            suitable_times = false;
          }
        }
      }
      suitable_times = false;
      while (!suitable_times) {
        suitable_times = true;
        std::generate(begin(Psi3), end(Psi3), genPsi3);
        vector<double100> copy_times(Psi3);
        sortVectorsAscending(copy_times, copy_times);
        copy_times.insert(copy_times.begin(), t2);
        copy_times.push_back(t3);
        for (vector<double100>::iterator cti = copy_times.begin();
             cti != copy_times.end() - 1; cti++) {
          if (fabs(*(cti + 1) - *cti) <= tol) {
            suitable_times = false;
          }
        }
      }
      suitable_times = false;
      while (!suitable_times) {
        suitable_times = true;
        std::generate(begin(Psi4), end(Psi4), genPsi4);
        vector<double100> copy_times(Psi4);
        sortVectorsAscending(copy_times, copy_times);
        copy_times.insert(copy_times.begin(), t3);
        copy_times.push_back(t4);
        for (vector<double100>::iterator cti = copy_times.begin();
             cti != copy_times.end() - 1; cti++) {
          if (fabs(*(cti + 1) - *cti) <= tol) {
            suitable_times = false;
          }
        }
      }
      auto genGam2 =
          [&unifGam2, &gen]()  // Routine to create uniform variates over [0,1]
      { return (unifGam2(gen)); };
      auto genGam3 =
          [&unifGam3, &gen]()  // Routine to create uniform variates over [0,1]
      { return (unifGam3(gen)); };
      auto genGam4 =
          [&unifGam4, &gen]()  // Routine to create uniform variates over [0,1]
      { return (unifGam4(gen)); };
      auto genKsi2 =
          [&lambdaMax, &unifKsi2,
           &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
      { return (unifKsi2(gen) * lambdaMax); };
      auto genKsi3 =
          [&lambdaMax, &unifKsi3,
           &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
      { return (unifKsi3(gen) * lambdaMax); };
      auto genKsi4 =
          [&lambdaMax, &unifKsi4,
           &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
      { return (unifKsi4(gen) * lambdaMax); };
      // Generate the uniform variates
      std::generate(begin(Gam2), end(Gam2), genGam2);
      std::generate(begin(Gam3), end(Gam3), genGam3);
      std::generate(begin(Gam4), end(Gam4), genGam4);
      std::generate(begin(Ksi2), end(Ksi2), genKsi2);
      std::generate(begin(Ksi3), end(Ksi3), genKsi3);
      std::generate(begin(Ksi4), end(Ksi4), genKsi4);

      sortVectorsAscending(
          Psi2, Psi2, Gam2,
          Ksi2);  // Sort according to time ordering of psi_j marks
      sortVectorsAscending(Psi3, Psi3, Gam3, Ksi3);
      sortVectorsAscending(Psi4, Psi4, Gam4, Ksi4);

      bool AcceptSkeleton = true;

      if (!Psi2.empty()) {
        for (vector<double100>::iterator Psi2iter = Psi2.begin(),
                                         Gam2iter = Gam2.begin(),
                                         Omega2iter = Omega2.begin(),
                                         Ksi2iter = Ksi2.begin();
             Psi2iter != Psi2.end();
             Psi2iter++, Gam2iter++, Omega2iter++, Ksi2iter++) {
          if (Psi2iter == Psi2.begin()) {
            *Omega2iter =
                Current
                    .DrawBridgepoint(0.0, Xt2Prop, t0Prop, t2, *Psi2iter, o,
                                     gen)
                    .first;  // Draw skeleton point from neutral bridge
            // going from 0 to Xt1Prop at time psi_1

            if ((((Current.Phitilde(*Omega2iter) - phimin) / phirange) >=
                 *Gam2iter) &&
                (*Ksi2iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          } else {
            *Omega2iter =
                Current
                    .DrawBridgepoint(Omega2.back(), Xt2Prop, *(Psi2iter - 1),
                                     t2, *Psi2iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from omega_j-1 to Xt1Prop at time psi_j

            if ((((Current.Phitilde(*Omega2iter) - phimin) / phirange) >=
                 *Gam2iter) &&
                (*Ksi2iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          }
        }
      }

      if ((!Psi3.empty()) && (AcceptSkeleton)) {
        for (vector<double100>::iterator Psi3iter = Psi3.begin(),
                                         Gam3iter = Gam3.begin(),
                                         Omeg3iter = Omega3.begin(),
                                         Ksi3iter = Ksi3.begin();
             Psi3iter != Psi3.end();
             Psi3iter++, Gam3iter++, Omeg3iter++, Ksi3iter++) {
          if (Psi3iter == Psi3.begin()) {
            *Omeg3iter =
                Current
                    .DrawBridgepoint(Xt2Prop, Xt3Prop, t2, t3, *Psi3iter, o,
                                     gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from Xt1Prop to x_{t_2} at time psi_1

            if ((((Current.Phitilde(*Omeg3iter) - phimin) / phirange) >=
                 *Gam3iter) &&
                (*Ksi3iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          } else {
            *Omeg3iter =
                Current
                    .DrawBridgepoint(Omega3.back(), Xt3Prop, *(Psi3iter - 1),
                                     t3, *Psi3iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from omega_j-1 to x_{t_2} at time psi_j

            if ((((Current.Phitilde(*Omeg3iter) - phimin) / phirange) >=
                 *Gam3iter) &&
                (*Ksi3iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          }
        }
      }

      if ((!Psi4.empty()) && (AcceptSkeleton)) {
        for (vector<double100>::iterator Psi4iter = Psi4.begin(),
                                         Gam4iter = Gam4.begin(),
                                         Omeg4iter = Omega4.begin(),
                                         Ksi4iter = Ksi4.begin();
             Psi4iter != Psi4.end();
             Psi4iter++, Gam4iter++, Omeg4iter++, Ksi4iter++) {
          if (Psi4iter == Psi4.begin()) {
            *Omeg4iter =
                Current.DrawBridgepoint(Xt3Prop, xt4, t3, t4, *Psi4iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from Xt1Prop to x_{t_2} at time psi_1

            if ((((Current.Phitilde(*Omeg4iter) - phimin) / phirange) >=
                 *Gam4iter) &&
                (*Ksi4iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          } else {
            *Omeg4iter =
                Current
                    .DrawBridgepoint(Omega4.back(), xt4, *(Psi4iter - 1), t4,
                                     *Psi4iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from omega_j-1 to x_{t_2} at time psi_j

            if ((((Current.Phitilde(*Omeg4iter) - phimin) / phirange) >=
                 *Gam4iter) &&
                (*Ksi4iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          }
        }
      }

      if (AcceptSkeleton) {
        // Load generated variables into prop variables

        XProp.push_back(0.0);
        XProp.push_back(Xt2Prop);
        XProp.push_back(Xt3Prop);
        kappaProp.push_back(0);
        kappaProp.push_back(kappa2);
        kappaProp.push_back(kappa3);
        kappaProp.push_back(kappa4);
        vector<double100> emptyvec;
        omegaProp.push_back(emptyvec);
        omegaProp.push_back(Omega2);
        omegaProp.push_back(Omega3);
        omegaProp.push_back(Omega4);
        psiProp.push_back(emptyvec);
        psiProp.push_back(Psi2);
        psiProp.push_back(Psi3);
        psiProp.push_back(Psi4);
        gammaProp.push_back(emptyvec);
        gammaProp.push_back(Gam2);
        gammaProp.push_back(Gam3);
        gammaProp.push_back(Gam4);
        ksiProp.push_back(emptyvec);
        ksiProp.push_back(Ksi2);
        ksiProp.push_back(Ksi3);
        ksiProp.push_back(Ksi4);

        SuitableProposal = true;
      }
    }

  } else {
    // Set up variables needed in subsequent computations

    double100 xt4 = XStore.back().at(3), t3 = times.at(2),
              t4 = times.at(3);  // Setting x_{t_2}, t1, t2
    double100 phimin = Current.phiMin, phimax = Current.phiMax,
              phirange = phimax - phimin;
    // Setting phimin/max

    // Propose new X_{t_3} from bridge

    double100 Xt3Prop = NonNeutralBridgePointBinomialObservations(
                            Current, 0.0, t0Prop, t4, xt4, o, gen, t3)
                            .first;

    while (!SuitableProposal) {
      double100 rate3 = lambdaMax * (t3 - t0Prop),
                rate4 = lambdaMax * (t4 - t3);
      boost::random::poisson_distribution<> kap3(static_cast<double>(rate3)),
          kap4(static_cast<double>(rate4));

      int kappa3 = kap3(gen),
          kappa4 = kap4(gen);  // Generate number of Poisson points
      double100 tol = 5.0e-5;

      vector<double100> Psi3(kappa3), Gam3(kappa3), Omega3(kappa3),
          Ksi3(kappa3), Psi4(kappa4), Gam4(kappa4), Omega4(kappa4),
          Ksi4(kappa4);

      boost::random::uniform_01<> unifPsi3, unifGam3, unifKsi3, unifPsi4,
          unifGam4, unifKsi4;
      auto genPsi3 =
          [&t0Prop, &t3, &unifPsi3,
           &gen]()  // Routine to create uniform variates over [t0Prop,t3]
      { return (t0Prop + ((t3 - t0Prop) * unifPsi3(gen))); };
      auto genPsi4 =
          [&t3, &t4, &unifPsi4,
           &gen]()  // Routine to create uniform variates over [t3,t4]
      { return (t3 + ((t4 - t3) * unifPsi4(gen))); };
      bool suitable_times = false;
      while (!suitable_times) {
        suitable_times = true;
        std::generate(begin(Psi3), end(Psi3), genPsi3);
        vector<double100> copy_times(Psi3);
        sortVectorsAscending(copy_times, copy_times);
        copy_times.insert(copy_times.begin(), t0Prop);
        copy_times.push_back(t3);
        for (vector<double100>::iterator cti = copy_times.begin();
             cti != copy_times.end() - 1; cti++) {
          if (fabs(*(cti + 1) - *cti) <= tol) {
            suitable_times = false;
          }
        }
      }
      suitable_times = false;
      while (!suitable_times) {
        suitable_times = true;
        std::generate(begin(Psi4), end(Psi4), genPsi4);
        vector<double100> copy_times(Psi4);
        sortVectorsAscending(copy_times, copy_times);
        copy_times.insert(copy_times.begin(), t3);
        copy_times.push_back(t4);
        for (vector<double100>::iterator cti = copy_times.begin();
             cti != copy_times.end() - 1; cti++) {
          if (fabs(*(cti + 1) - *cti) <= tol) {
            suitable_times = false;
          }
        }
      }
      auto genGam3 =
          [&unifGam3, &gen]()  // Routine to create uniform variates over [0,1]
      { return (unifGam3(gen)); };
      auto genGam4 =
          [&unifGam4, &gen]()  // Routine to create uniform variates over [0,1]
      { return (unifGam4(gen)); };
      auto genKsi3 =
          [&lambdaMax, &unifKsi3,
           &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
      { return (unifKsi3(gen) * lambdaMax); };
      auto genKsi4 =
          [&lambdaMax, &unifKsi4,
           &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
      { return (unifKsi4(gen) * lambdaMax); };

      // Generate the uniform variates
      std::generate(begin(Gam3), end(Gam3), genGam3);
      std::generate(begin(Ksi3), end(Ksi3), genKsi3);
      std::generate(begin(Gam4), end(Gam4), genGam4);
      std::generate(begin(Ksi4), end(Ksi4), genKsi4);

      sortVectorsAscending(
          Psi3, Psi3, Gam3,
          Ksi3);  // Sort according to time ordering of psi_j marks
      sortVectorsAscending(Psi4, Psi4, Gam4, Ksi4);

      bool AcceptSkeleton = true;

      if (!Psi3.empty()) {
        for (vector<double100>::iterator Psi3iter = Psi3.begin(),
                                         Gam3iter = Gam3.begin(),
                                         Omega3iter = Omega3.begin(),
                                         Ksi3iter = Ksi3.begin();
             Psi3iter != Psi3.end();
             Psi3iter++, Gam3iter++, Omega3iter++, Ksi3iter++) {
          if (Psi3iter == Psi3.begin()) {
            *Omega3iter =
                Current
                    .DrawBridgepoint(0.0, Xt3Prop, t0Prop, t3, *Psi3iter, o,
                                     gen)
                    .first;  // Draw skeleton point from neutral bridge
            // going from 0 to Xt1Prop at time psi_1

            if ((((Current.Phitilde(*Omega3iter) - phimin) / phirange) >=
                 *Gam3iter) &&
                (*Ksi3iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          } else {
            *Omega3iter =
                Current
                    .DrawBridgepoint(Omega3.back(), Xt3Prop, *(Psi3iter - 1),
                                     t3, *Psi3iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from omega_j-1 to Xt1Prop at time psi_j

            if ((((Current.Phitilde(*Omega3iter) - phimin) / phirange) >=
                 *Gam3iter) &&
                (*Ksi3iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          }
        }
      }

      if ((!Psi4.empty()) && (AcceptSkeleton)) {
        for (vector<double100>::iterator Psi4iter = Psi4.begin(),
                                         Gam4iter = Gam4.begin(),
                                         Omega4iter = Omega4.begin(),
                                         Ksi4iter = Ksi4.begin();
             Psi4iter != Psi4.end();
             Psi4iter++, Gam4iter++, Omega4iter++, Ksi4iter++) {
          if (Psi4iter == Psi4.begin()) {
            *Omega4iter =
                Current.DrawBridgepoint(Xt3Prop, xt4, t3, t4, *Psi4iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from 0 to Xt1Prop at time psi_1

            if ((((Current.Phitilde(*Omega4iter) - phimin) / phirange) >=
                 *Gam4iter) &&
                (*Ksi4iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          } else {
            *Omega4iter =
                Current
                    .DrawBridgepoint(Omega4.back(), xt4, *(Psi4iter - 1), t4,
                                     *Psi4iter, o, gen)
                    .first;  // Draw skeleton point from neutral bridge going
            // from omega_j-1 to Xt1Prop at time psi_j

            if ((((Current.Phitilde(*Omega4iter) - phimin) / phirange) >=
                 *Gam4iter) &&
                (*Ksi4iter < phirange)) {
              AcceptSkeleton = false;
              break;
            }
          }
        }
      }

      if (AcceptSkeleton) {
        // Load generated variables into prop variables

        XProp.push_back(0.0);
        XProp.push_back(0.0);
        XProp.push_back(Xt3Prop);
        kappaProp.push_back(0);
        kappaProp.push_back(0);
        kappaProp.push_back(kappa3);
        kappaProp.push_back(kappa4);
        vector<double100> emptyvec;
        omegaProp.push_back(emptyvec);
        omegaProp.push_back(emptyvec);
        omegaProp.push_back(Omega3);
        omegaProp.push_back(Omega4);
        psiProp.push_back(emptyvec);
        psiProp.push_back(emptyvec);
        psiProp.push_back(Psi3);
        psiProp.push_back(Psi4);
        gammaProp.push_back(emptyvec);
        gammaProp.push_back(emptyvec);
        gammaProp.push_back(Gam3);
        gammaProp.push_back(Gam4);
        ksiProp.push_back(emptyvec);
        ksiProp.push_back(emptyvec);
        ksiProp.push_back(Ksi3);
        ksiProp.push_back(Ksi4);

        SuitableProposal = true;
      }
    }
  }
}

void MCMCSampler::AcceptT0Z2Proposals(
    WrightFisher Current, int &indt0Prop, double100 &t0Prop,
    vector<double100> &XProp, vector<int> &kappaProp,
    vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
    vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
    vector<double100> &XOut, vector<int> &kappaOut,
    vector<vector<double100>> &omegaOut, vector<vector<double100>> &psiOut,
    vector<vector<double100>> &gammaOut, vector<vector<double100>> &ksiOut,
    const Options &o, const MCMCOptions &MCMCo, boost::random::mt19937 &gen) {
  // Load in current values

  double100 t0Curr = t0Store.back(), t4 = times.at(3);
  vector<double100> XCurr(XStore.back().begin(), XStore.back().begin() + 3),
      phiminmaxrange{Current.phiMin, Current.phiMax,
                     Current.phiMax - Current.phiMin};
  vector<int> kappaCurr(kappaStore.back().begin(),
                        kappaStore.back().begin() + 4);
  vector<vector<double100>> omegaCurr(omegaStore.back().begin(),
                                      omegaStore.back().begin() + 4),
      psiCurr(psiStore.back().begin(), psiStore.back().begin() + 4),
      gammaCurr(gammaStore.back().begin(), gammaStore.back().begin() + 4),
      ksiCurr(ksiStore.back().begin(), ksiStore.back().begin() + 4);

  boost::random::uniform_01<> U01;
  double100 u = U01(gen),
            likelihood =
                exp(-phiminmaxrange[0] * (t4 - t0Prop)) *
                pow(XProp.front(), static_cast<double100>(Data.front())) *
                pow(1.0 - XProp.front(),
                    static_cast<double100>(Samplesizes.front() -
                                           Data.front())) *
                pow(XProp.at(1), static_cast<double100>(Data.at(1))) *
                pow(1.0 - XProp.at(1),
                    static_cast<double100>(Samplesizes.at(1) - Data.at(1))) *
                pow(XProp.back(), static_cast<double100>(Data.at(2))) *
                pow(1.0 - XProp.back(),
                    static_cast<double100>(Samplesizes.at(2) - Data.at(2)));
  double100 binomial_ratio =
      pow((1.0 - XProp.front()) / (1.0 - XCurr.front()), Samplesizes.front()) *
      pow(XProp.at(1) / XCurr.at(1), static_cast<double100>(Data.at(1))) *
      pow((1.0 - XProp.at(1)) / (1.0 - XCurr.at(1)),
          static_cast<double100>(Samplesizes.at(1) - Data.at(1))) *
      pow(XProp.back() / XCurr.back(), static_cast<double100>(Data.at(2))) *
      pow((1.0 - XProp.back()) / (1.0 - XCurr.back()),
          static_cast<double100>(Samplesizes.at(2) - Data.at(2)));
  double100 alpha = t0ProposalRatio(t0Prop, t0Curr, MCMCo) *
                    t0PriorRatio(t0Prop, t0Curr, MCMCo) *
                    exp(-phiminmaxrange[0] * (t0Curr - t0Prop)) *
                    binomial_ratio;
  double100 pseudo =
      PoissonEstimator(Current, t4 - t0Prop, 0.0, XCurr.back(), o, gen);

  vector<double100> t_incs;
  if (indt0Prop == 0) {
    t_incs.push_back(times.at(1) - t0Prop);
    t_incs.push_back(times.at(2) - times.at(1));
    t_incs.push_back(times.at(3) - times.at(2));
    t_incs.push_back(times.at(4) - times.at(3));
  } else if (indt0Prop == 1) {
    t_incs.push_back(0.0);
    t_incs.push_back(times.at(2) - t0Prop);
    t_incs.push_back(times.at(3) - times.at(2));
    t_incs.push_back(times.at(4) - times.at(3));
  } else {
    t_incs.push_back(0.0);
    t_incs.push_back(0.0);
    t_incs.push_back(times.at(3) - t0Prop);
    t_incs.push_back(times.at(4) - times.at(3));
  }

  vector<double100>::iterator titer = t_incs.begin();
  for (vector<vector<double100>>::iterator PsiIter = psiProp.begin(),
                                           OmegaIter = omegaProp.begin(),
                                           KsiIter = ksiProp.begin();
       PsiIter != psiProp.end() - 3; PsiIter++, OmegaIter++, KsiIter++) {
    int kappaCounter = 0;
    for (vector<double100>::iterator PsiiIter = (*PsiIter).begin(),
                                     OmegaiIter = (*OmegaIter).begin(),
                                     KsiiIter = (*KsiIter).begin();
         PsiiIter != (*PsiIter).end(); PsiiIter++, OmegaiIter++, KsiiIter++) {
      kappaCounter++;
      if ((*KsiiIter < phiminmaxrange[2]) && (*OmegaiIter >= 0.0)) {
        likelihood *=
            (1.0 - ((Current.Phitilde(*OmegaiIter) - phiminmaxrange[0]) /
                    phiminmaxrange[2]));
      }
    }
    double100 multiplicator =
        pow((phiminmaxrange[2] * (*titer)),
            static_cast<double100>(kappaCounter)) *
        (1.0 / boost::math::factorial<double100>(kappaCounter));
    likelihood *= exp(-phiminmaxrange[2] * (*titer)) * multiplicator;
    titer++;
  }

  alpha *= (pseudo / pseudoStore.back());

  pair<bool, vector<double100>> AccTrEst =
      TransitionRatioDecision(Current, alpha, u, 0.0, XCurr.back(), 0.0,
                              XCurr.back(), t0Prop, t0Curr, t4, t4, o);

  bool Accept = AccTrEst.first;

  if (Accept) {
    t0Store.push_back(t0Prop);
    XOut.insert(XOut.end(), XProp.begin(), XProp.end());
    kappaOut.insert(kappaOut.end(), kappaProp.begin(), kappaProp.end());
    omegaOut.insert(omegaOut.end(), omegaProp.begin(), omegaProp.end());
    psiOut.insert(psiOut.end(), psiProp.begin(), psiProp.end());
    gammaOut.insert(gammaOut.end(), gammaProp.begin(), gammaProp.end());
    ksiOut.insert(ksiOut.end(), ksiProp.begin(), ksiProp.end());
    pseudoStore.push_back(pseudo);
    t0Likelihood.push_back(likelihood);

  } else {
    t0Store.push_back(t0Store.back());
    XOut.insert(XOut.end(), XCurr.begin(), XCurr.end());
    kappaOut.insert(kappaOut.end(), kappaCurr.begin(), kappaCurr.end());
    omegaOut.insert(omegaOut.end(), omegaCurr.begin(), omegaCurr.end());
    psiOut.insert(psiOut.end(), psiCurr.begin(), psiCurr.end());
    gammaOut.insert(gammaOut.end(), gammaCurr.begin(), gammaCurr.end());
    ksiOut.insert(ksiOut.end(), ksiCurr.begin(), ksiCurr.end());
    pseudoStore.push_back(pseudoStore.back());
    t0Likelihood.push_back(t0Likelihood.back());
  }
}

void MCMCSampler::GenerateXtiProposals(
    WrightFisher Current, vector<double100> &XProp, vector<int> &kappaProp,
    vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
    vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
    double100 lambdaMax, int iter, const Options &o,
    boost::random::mt19937 &gen) {
  bool SuitableProposals = false;

  // Set up variable for subsequent calculations

  double100 tim1 = times.at(iter - 1), ti = times.at(iter),
            tip1 = times.at(iter + 1);
  double100 xtim1 = XStore.back().at(iter - 1),
            xtip1 = (XStore.size() < 2)
                        ? XStore.back().at(iter + 1)
                        : XStore.at(XStore.size() - 2).at(iter + 1);
  double100 phimin = Current.phiMin, phimax = Current.phiMax,
            phirange = phimax - phimin;  // Setting phimin/max

  // Propose new X_{t_i} from bridge

  while (!SuitableProposals) {
    std::pair<double100, vector<vector<double100>>> dummy =
        NonNeutralBridgePointBinomialObservations(Current, xtim1, tim1, tip1,
                                                  xtip1, o, gen, ti);
    double100 XtiProp = dummy.first;

    // Propose new Poisson & skeleton points

    double100 rateL = lambdaMax * (ti - tim1), rateR = lambdaMax * (tip1 - ti);
    double100 tol = 5.0e-5;
    boost::random::poisson_distribution<> kapL(static_cast<double>(rateL)),
        kapR(static_cast<double>(rateR));
    int kappaL = kapL(gen),
        kappaR =
            kapR(gen);  // Generate number of Poisson points to the left and
    // right of Xt1Prop
    vector<double100> PsiL(kappaL), GamL(kappaL), OmegaL(kappaL), KsiL(kappaL),
        PsiR(kappaR), GamR(kappaR), OmegaR(kappaR), KsiR(kappaR);

    boost::random::uniform_01<> unifPsiL, unifPsiR, unifGamL, unifGamR,
        unifKsiL, unifKsiR;
    auto genPsiL =
        [&tim1, &ti, &unifPsiL,
         &gen]()  // Routine to create uniform variates over [t0Prop,t1]
    { return (tim1 + ((ti - tim1) * unifPsiL(gen))); };
    auto genPsiR = [&ti, &tip1, &unifPsiR,
                    &gen]()  // Routine to create uniform variates over [t1,t2]
    { return (ti + ((tip1 - ti) * unifPsiR(gen))); };
    bool suitable_times = false;
    while (!suitable_times) {
      suitable_times = true;
      std::generate(begin(PsiL), end(PsiL), genPsiL);
      vector<double100> copy_times(PsiL);
      sortVectorsAscending(copy_times, copy_times);
      copy_times.insert(copy_times.begin(), tim1);
      copy_times.push_back(ti);
      for (vector<double100>::iterator cti = copy_times.begin();
           cti != copy_times.end() - 1; cti++) {
        if (fabs(*(cti + 1) - *cti) <= tol) {
          suitable_times = false;
        }
      }
    }
    suitable_times = false;
    while (!suitable_times) {
      suitable_times = true;
      std::generate(begin(PsiR), end(PsiR), genPsiR);
      vector<double100> copy_times(PsiR);
      sortVectorsAscending(copy_times, copy_times);
      copy_times.insert(copy_times.begin(), ti);
      copy_times.push_back(tip1);
      for (vector<double100>::iterator cti = copy_times.begin();
           cti != copy_times.end() - 1; cti++) {
        if (fabs(*(cti + 1) - *cti) <= tol) {
          suitable_times = false;
        }
      }
    }
    auto genGamL =
        [&unifGamL, &gen]()  // Routine to create uniform variates over [0,1]
    { return (unifGamL(gen)); };
    auto genGamR =
        [&unifGamR, &gen]()  // Routine to create uniform variates over [0,1]
    { return (unifGamR(gen)); };
    auto genKsiL =
        [&lambdaMax, &unifKsiL,
         &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
    { return (unifKsiL(gen) * lambdaMax); };
    auto genKsiR =
        [&lambdaMax, &unifKsiR,
         &gen]()  // Routine to create uniform variates over [0,lambda_{max}]
    { return (unifKsiR(gen) * lambdaMax); };

    // Generate the uniform variates
    std::generate(begin(GamL), end(GamL), genGamL);
    std::generate(begin(GamR), end(GamR), genGamR);
    std::generate(begin(KsiL), end(KsiL), genKsiL);
    std::generate(begin(KsiR), end(KsiR), genKsiR);

    sortVectorsAscending(
        PsiL, PsiL, GamL,
        KsiL);  // Sort according to time ordering of psi_j marks
    sortVectorsAscending(PsiR, PsiR, GamR, KsiR);

    bool AcceptSkeleton = true;

    if (!PsiL.empty()) {
      for (vector<double100>::iterator PsiLiter = PsiL.begin(),
                                       GamLiter = GamL.begin(),
                                       OmegLiter = OmegaL.begin(),
                                       KsiLiter = KsiL.begin();
           PsiLiter != PsiL.end();
           PsiLiter++, GamLiter++, OmegLiter++, KsiLiter++) {
        if (PsiLiter == PsiL.begin()) {
          *OmegLiter =
              Current
                  .DrawBridgepoint(xtim1, XtiProp, tim1, ti, *PsiLiter, o, gen)
                  .first;  // Draw skeleton point from neutral bridge going from
          // x_{t_{i-1}} to XtiProp at time psi_1

          if ((((Current.Phitilde(*OmegLiter) - phimin) / phirange) >=
               *GamLiter) &&
              (*KsiLiter < phirange)) {
            AcceptSkeleton = false;
            break;
          }
        } else {
          *OmegLiter =
              Current
                  .DrawBridgepoint(OmegaL.back(), XtiProp, *(PsiLiter - 1), ti,
                                   *PsiLiter, o, gen)
                  .first;  // Draw skeleton point from neutral bridge going from
          // omega_j-1 to XtiProp at time psi_j

          if ((((Current.Phitilde(*OmegLiter) - phimin) / phirange) >=
               *GamLiter) &&
              (*KsiLiter < phirange)) {
            AcceptSkeleton = false;
            break;
          }
        }
      }
    }

    if ((!PsiR.empty()) && (AcceptSkeleton)) {
      for (vector<double100>::iterator PsiRiter = PsiR.begin(),
                                       GamRiter = GamR.begin(),
                                       OmegRiter = OmegaR.begin(),
                                       KsiRiter = KsiR.begin();
           PsiRiter != PsiR.end();
           PsiRiter++, GamRiter++, OmegRiter++, KsiRiter++) {
        if (PsiRiter == PsiR.begin()) {
          *OmegRiter =
              Current
                  .DrawBridgepoint(XtiProp, xtip1, ti, tip1, *PsiRiter, o, gen)
                  .first;  // Draw skeleton point from neutral bridge going from
          // XtiProp to x_{t_{i+1}} at time psi_1

          if ((((Current.Phitilde(*OmegRiter) - phimin) / phirange) >=
               *GamRiter) &&
              (*KsiRiter < phirange)) {
            AcceptSkeleton = false;
            break;
          }
        } else {
          *OmegRiter =
              Current
                  .DrawBridgepoint(OmegaR.back(), xtip1, *(PsiRiter - 1), tip1,
                                   *PsiRiter, o, gen)
                  .first;  // Draw skeleton point from neutral bridge going from
          // omega_j-1 to x_{t_{i+1}} at time psi_j

          if ((((Current.Phitilde(*OmegRiter) - phimin) / phirange) >=
               *GamRiter) &&
              (*KsiRiter < phirange)) {
            AcceptSkeleton = false;
            break;
          }
        }
      }
    }

    if (AcceptSkeleton) {
      XProp.push_back(XtiProp);
      kappaProp.push_back(kappaL);
      kappaProp.push_back(kappaR);
      omegaProp.push_back(OmegaL);
      omegaProp.push_back(OmegaR);
      psiProp.push_back(PsiL);
      psiProp.push_back(PsiR);
      gammaProp.push_back(GamL);
      gammaProp.push_back(GamR);
      ksiProp.push_back(KsiL);
      ksiProp.push_back(KsiR);

      SuitableProposals = true;
    }
  }
}

void MCMCSampler::AcceptXtiProposals(
    vector<double100> &XProp, vector<int> &kappaProp,
    vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
    vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
    vector<double100> &XOut, vector<int> &kappaOut,
    vector<vector<double100>> &omegaOut, vector<vector<double100>> &psiOut,
    vector<vector<double100>> &gammaOut, vector<vector<double100>> &ksiOut,
    int iter, boost::random::mt19937 &gen) {
  double100 XCurr_i = XStore.at(XStore.size() - 2).at(iter),
            XProp_i = XProp.back();
  double100 alpha = exp(
      static_cast<double100>(Data.at(iter)) * (log(XProp_i) - log(XCurr_i)) +
      static_cast<double100>((Samplesizes.at(iter) - Data.at(iter))) *
          (log(1.0 - XProp_i) - log(1.0 - XCurr_i)));
  boost::random::uniform_01<> U01;
  double100 u = U01(gen);  // Set up uniform variate for MH accept/reject
  bool AcceptProposals = alpha >= u;

  if (AcceptProposals)  // If both skeletons and MH accepted
  {
    // Update loading vectors with proposed values deleting entries already
    // there at i
    XOut = XProp;
    kappaOut = kappaProp;
    omegaOut = omegaProp;
    psiOut = psiProp;
    gammaOut = gammaProp;
    ksiOut = ksiProp;
  } else  // else just reject
  {
    // Update loading vectors with previous values
    XOut.push_back(XStore.at(XStore.size() - 2).at(iter));
    kappaOut.push_back(kappaStore.at(kappaStore.size() - 2).at(iter));
    kappaOut.push_back(kappaStore.at(kappaStore.size() - 3).at(iter + 1));
    omegaOut.push_back(omegaStore.at(omegaStore.size() - 2).at(iter));
    omegaOut.push_back(omegaStore.at(omegaStore.size() - 3).at(iter + 1));
    psiOut.push_back(psiStore.at(psiStore.size() - 2).at(iter));
    psiOut.push_back(psiStore.at(psiStore.size() - 3).at(iter + 1));
    gammaOut.push_back(gammaStore.at(gammaStore.size() - 2).at(iter));
    gammaOut.push_back(gammaStore.at(gammaStore.size() - 3).at(iter + 1));
    ksiOut.push_back(ksiStore.at(ksiStore.size() - 2).at(iter));
    ksiOut.push_back(ksiStore.at(ksiStore.size() - 3).at(iter + 1));
  }
}

void MCMCSampler::GenerateXtnProposals(
    WrightFisher Current, vector<double100> &XProp, vector<int> &kappaProp,
    vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
    vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
    double100 lambdaMax, const Options &o, boost::random::mt19937 &gen) {
  bool SuitableProposals = false;

  double100 tn = times.back(), tnm1 = times.rbegin()[1],
            xtnm1 = XStore.at(XStore.size() - 2).rbegin()[1];
  // int Ytn = Data.back(), ntn = Samplesizes.back();
  double100 phimin = Current.phiMin, phimax = Current.phiMax,
            phirange = phimax - phimin;  // Setting phimin/max

  while (!SuitableProposals) {
    double100 XtnProp = NonNeutralDrawEndpointBinomialObservations(
                            Current, xtnm1, tnm1, tn, o, gen)
                            .first;
    // Propose new Poisson & skeleton points

    double100 rate = lambdaMax * (tn - tnm1);
    double100 tol = 5.0e-5;
    boost::random::poisson_distribution<> kap(static_cast<double>(rate));
    int kappa = kap(gen);  // Generate number of Poisson points to the left and
    // right of Xt1Prop
    vector<double100> Psi(kappa), Gam(kappa), Omega(kappa), Ksi(kappa);

    boost::random::uniform_01<> unifPsi, unifGam, unifKsi;
    auto genPsi =
        [&tnm1, &tn, &unifPsi,
         &gen]()  // Routine to create uniform variates over [t0Prop,t1]
    { return (tnm1 + ((tn - tnm1) * unifPsi(gen))); };
    bool suitable_times = false;
    while (!suitable_times) {
      suitable_times = true;
      std::generate(begin(Psi), end(Psi), genPsi);
      vector<double100> copy_times(Psi);
      sortVectorsAscending(copy_times, copy_times);
      copy_times.insert(copy_times.begin(), tnm1);
      copy_times.push_back(tn);
      for (vector<double100>::iterator cti = copy_times.begin();
           cti != copy_times.end() - 1; cti++) {
        if (fabs(*(cti + 1) - *cti) <= tol) {
          suitable_times = false;
        }
      }
    }
    auto genGam =
        [&unifGam, &gen]()  // Routine to create uniform variates over [0,1]
    { return (unifGam(gen)); };
    auto genKsi =
        [&lambdaMax, &unifKsi,
         &gen]()  // Routine to create uniform variates over [0,\lambda_{max}]
    { return (unifKsi(gen) * lambdaMax); };

    // Generate the uniform variates
    std::generate(begin(Gam), end(Gam), genGam);
    std::generate(begin(Ksi), end(Ksi), genKsi);

    sortVectorsAscending(
        Psi, Psi, Gam,
        Ksi);  // Sort according to time ordering of psi_j marks

    bool AcceptSkeleton = true;

    if (!Psi.empty()) {
      for (vector<double100>::iterator Psiiter = Psi.begin(),
                                       Gamiter = Gam.begin(),
                                       Omegiter = Omega.begin(),
                                       Ksiiter = Ksi.begin();
           Psiiter != Psi.end(); Psiiter++, Gamiter++, Omegiter++, Ksiiter++) {
        if (Psiiter == Psi.begin()) {
          *Omegiter =
              Current
                  .DrawBridgepoint(xtnm1, XtnProp, tnm1, tn, *Psiiter, o, gen)
                  .first;  // Draw skeleton point from neutral bridge going from
          // x_{t_{i-1}} to XtiProp at time psi_1

          if ((((Current.Phitilde(*Omegiter) - phimin) / phirange) >=
               *Gamiter) &&
              (*Ksiiter < phirange)) {
            AcceptSkeleton = false;
            break;
          }
        } else {
          *Omegiter =
              Current
                  .DrawBridgepoint(Omega.back(), XtnProp, *(Psiiter - 1), tn,
                                   *Psiiter, o, gen)
                  .first;  // Draw skeleton point from neutral bridge going from
          // omega_j-1 to XtiProp at time psi_j

          if ((((Current.Phitilde(*Omegiter) - phimin) / phirange) >=
               *Gamiter) &&
              (*Ksiiter < phirange)) {
            AcceptSkeleton = false;
            break;
          }
        }
      }
    }

    if (AcceptSkeleton) {
      XProp.push_back(XtnProp);
      kappaProp.push_back(kappa);
      omegaProp.push_back(Omega);
      psiProp.push_back(Psi);
      gammaProp.push_back(Gam);
      ksiProp.push_back(Ksi);

      SuitableProposals = true;
    }
  }
}

void MCMCSampler::AcceptXtnProposals(
    vector<double100> &XProp, vector<int> &kappaProp,
    vector<vector<double100>> &omegaProp, vector<vector<double100>> &psiProp,
    vector<vector<double100>> &gammaProp, vector<vector<double100>> &ksiProp,
    vector<double100> &XOut, vector<int> &kappaOut,
    vector<vector<double100>> &omegaOut, vector<vector<double100>> &psiOut,
    vector<vector<double100>> &gammaOut, vector<vector<double100>> &ksiOut,
    boost::random::mt19937 &gen) {
  double100 XCurr_n = XStore.at(XStore.size() - 2).back(),
            XProp_n = XProp.back();
  double100 alpha =
      exp(static_cast<double100>(Data.back()) * (log(XProp_n) - log(XCurr_n)) +
          static_cast<double100>((Samplesizes.back() - Data.back())) *
              (log(1.0 - XProp_n) - log(1.0 - XCurr_n)));
  boost::random::uniform_01<> U01;
  double100 u = U01(gen);  // Set up uniform variate for MH accept/reject
  bool AcceptProposals = alpha >= u;

  if (AcceptProposals) {
    XOut.insert(XOut.end(), XProp.begin(), XProp.end());
    kappaOut.insert(kappaOut.end(), kappaProp.begin(), kappaProp.end());
    omegaOut.insert(omegaOut.end(), omegaProp.begin(), omegaProp.end());
    psiOut.insert(psiOut.end(), psiProp.begin(), psiProp.end());
    gammaOut.insert(gammaOut.end(), gammaProp.begin(), gammaProp.end());
    ksiOut.insert(ksiOut.end(), ksiProp.begin(), ksiProp.end());
  } else {
    XOut.push_back(XStore.at(XStore.size() - 2).back());
    kappaOut.push_back(kappaStore.at(kappaStore.size() - 2).back());
    omegaOut.push_back(omegaStore.at(omegaStore.size() - 2).back());
    psiOut.push_back(psiStore.at(psiStore.size() - 2).back());
    gammaOut.push_back(gammaStore.at(gammaStore.size() - 2).back());
    ksiOut.push_back(ksiStore.at(ksiStore.size() - 2).back());
  }
}

// UPDATE FUNCTIONS

void MCMCSampler::UpdateT0(
    WrightFisher Current, vector<double100> &XOut, vector<int> &kappaOut,
    vector<vector<double100>> &omegaOut, vector<vector<double100>> &psiOut,
    vector<vector<double100>> &gammaOut, vector<vector<double100>> &ksiOut,
    double100 lambdaMax, const Options &o, const MCMCOptions &MCMCo,
    boost::random::mt19937 &gen) {
  // Propose new allele age

  double100 t0Prop = t0Proposal(MCMCo, gen);

  if (zerosCounter == 0) {
    UpdateT0Z0(Current, XOut, kappaOut, omegaOut, psiOut, gammaOut, ksiOut,
               t0Prop, lambdaMax, o, MCMCo, gen);
  } else if (zerosCounter == 1) {
    UpdateT0Z1(Current, XOut, kappaOut, omegaOut, psiOut, gammaOut, ksiOut,
               t0Prop, lambdaMax, o, MCMCo, gen);
  } else {
    UpdateT0Z2(Current, XOut, kappaOut, omegaOut, psiOut, gammaOut, ksiOut,
               t0Prop, lambdaMax, o, MCMCo, gen);
  }
}

void MCMCSampler::UpdateT0Z0(
    WrightFisher Current, vector<double100> &XOut, vector<int> &kappaOut,
    vector<vector<double100>> &omegaOut, vector<vector<double100>> &psiOut,
    vector<vector<double100>> &gammaOut, vector<vector<double100>> &ksiOut,
    double100 t0Prop, double100 lambdaMax, const Options &o,
    const MCMCOptions &MCMCo, boost::random::mt19937 &gen) {
  // Set up variables for storing procedure at end of update

  vector<double100> XProp;
  vector<int> kappaProp;
  vector<vector<double100>> omegaProp, psiProp, gammaProp, ksiProp;

  // Propose (m_1,l_1,m_2,l_2), X_t_1, Phi according to set indicator variables

  GenerateT0Z0Proposals(Current, t0Prop, XProp, kappaProp, omegaProp, psiProp,
                        gammaProp, ksiProp, lambdaMax, o, gen);

  // Run MH A/R step

  AcceptT0Z0Proposals(Current, t0Prop, XProp, kappaProp, omegaProp, psiProp,
                      gammaProp, ksiProp, XOut, kappaOut, omegaOut, psiOut,
                      gammaOut, ksiOut, o, MCMCo, gen);
}

void MCMCSampler::UpdateT0Z1(
    WrightFisher Current, vector<double100> &XOut, vector<int> &kappaOut,
    vector<vector<double100>> &omegaOut, vector<vector<double100>> &psiOut,
    vector<vector<double100>> &gammaOut, vector<vector<double100>> &ksiOut,
    double100 t0Prop, double100 lambdaMax, const Options &o,
    const MCMCOptions &MCMCo, boost::random::mt19937 &gen) {
  // Set up variables for storing procedure at end of update

  vector<double100> XProp;
  vector<int> kappaProp;
  vector<vector<double100>> omegaProp, psiProp, gammaProp, ksiProp;

  // Generate candidate allele age t_0', set current allele age t_0 and figure
  // out which proposal and AP procedure to use

  int indt0Curr = (t0Store.back() < times.front()) ? 0 : 1,
      indt0Prop = (t0Prop < times.front()) ? 0 : 1;

  // Propose (m,l), X_t, Phi according to set indicator variables

  GenerateT0Z1Proposals(Current, indt0Prop, t0Prop, XProp, kappaProp, omegaProp,
                        psiProp, gammaProp, ksiProp, lambdaMax, o, gen);

  // Run MH A/R step

  AcceptT0Z1Proposals(Current, indt0Prop, t0Prop, XProp, kappaProp, omegaProp,
                      psiProp, gammaProp, ksiProp, XOut, kappaOut, omegaOut,
                      psiOut, gammaOut, ksiOut, o, MCMCo, gen);
}

void MCMCSampler::UpdateT0Z2(
    WrightFisher Current, vector<double100> &XOut, vector<int> &kappaOut,
    vector<vector<double100>> &omegaOut, vector<vector<double100>> &psiOut,
    vector<vector<double100>> &gammaOut, vector<vector<double100>> &ksiOut,
    double100 t0Prop, double100 lambdaMax, const Options &o,
    const MCMCOptions &MCMCo, boost::random::mt19937 &gen) {
  // Set up variables for storing procedure at end of update

  vector<double100> XProp;
  vector<int> kappaProp;
  vector<vector<double100>> omegaProp, psiProp, gammaProp, ksiProp;

  // Generate candidate allele age t_0', set current allele age t_0 and figure
  // out which proposal and AP procedure to use

  double100 t0Curr = t0Store.back();

  int indt0Curr, indt0Prop;
  if ((t0Curr < times.front())) {
    indt0Curr = 0;
  } else if ((t0Curr > times.front()) && (t0Curr < times.at(1))) {
    indt0Curr = 1;
  } else {
    indt0Curr = 2;
  }

  if ((t0Prop < times.front())) {
    indt0Prop = 0;
  } else if ((t0Prop > times.front()) && (t0Prop < times.at(1))) {
    indt0Prop = 1;
  } else {
    indt0Prop = 2;
  }

  // Propose (m,l), X_t, Phi according to set indicator variables

  GenerateT0Z2Proposals(Current, indt0Prop, t0Prop, XProp, kappaProp, omegaProp,
                        psiProp, gammaProp, ksiProp, lambdaMax, o, gen);

  // Run MH A/R step

  AcceptT0Z2Proposals(Current, indt0Prop, t0Prop, XProp, kappaProp, omegaProp,
                      psiProp, gammaProp, ksiProp, XOut, kappaOut, omegaOut,
                      psiOut, gammaOut, ksiOut, o, MCMCo, gen);
}

void MCMCSampler::UpdateXti(WrightFisher Current, vector<double100> &XOut,
                            vector<int> &kappaOut,
                            vector<vector<double100>> &omegaOut,
                            vector<vector<double100>> &psiOut,
                            vector<vector<double100>> &gammaOut,
                            vector<vector<double100>> &ksiOut,
                            double100 lambdaMax, int iter, const Options &o,
                            boost::random::mt19937 &gen) {
  vector<double100> XProp;
  vector<int> kappaProp;
  vector<vector<double100>> omegaProp, psiProp, gammaProp, ksiProp;

  // Propose X_t_i, Phi according to set indicator variables

  GenerateXtiProposals(Current, XProp, kappaProp, omegaProp, psiProp, gammaProp,
                       ksiProp, lambdaMax, iter, o, gen);

  // Run MH A/R step

  AcceptXtiProposals(XProp, kappaProp, omegaProp, psiProp, gammaProp, ksiProp,
                     XOut, kappaOut, omegaOut, psiOut, gammaOut, ksiOut, iter,
                     gen);
}

void MCMCSampler::UpdateXtn(
    WrightFisher Current, vector<double100> &XOut, vector<int> &kappaOut,
    vector<vector<double100>> &omegaOut, vector<vector<double100>> &psiOut,
    vector<vector<double100>> &gammaOut, vector<vector<double100>> &ksiOut,
    double100 lambdaMax, const Options &o, boost::random::mt19937 &gen) {
  vector<double100> XProp;
  vector<int> kappaProp;
  vector<vector<double100>> omegaProp, psiProp, gammaProp, ksiProp;

  // Propose (m_n,l_n), X_t_n, Phi according to set indicator variables

  GenerateXtnProposals(Current, XProp, kappaProp, omegaProp, psiProp, gammaProp,
                       ksiProp, lambdaMax, o, gen);

  // Run MH A/R step

  AcceptXtnProposals(XProp, kappaProp, omegaProp, psiProp, gammaProp, ksiProp,
                     XOut, kappaOut, omegaOut, psiOut, gammaOut, ksiOut, gen);
}

void MCMCSampler::UpdateSigma(WrightFisher Current, WrightFisher Proposal,
                              vector<double100> &selOut,
                              vector<double100> &selProp,
                              const MCMCOptions &MCMCo,
                              boost::random::mt19937 &gen) {
  // Set up variables needed in subsequent computations

  double100 t0 = t0Store.back(), tn = times.back(), Xtn = XStore.back().back(),
            sigmaCurr = sigmaStore.back(), hCurr,
            hProp;  // Setting x_{t_2}, t1, t2
  double100 sigmaProp = selProp.back();
  vector<double100> etaCurr, etaProp, selCurr{sigmaCurr};
  if (selectionType == 1) {
    hCurr = hStore.back();
    hProp = selProp.front();
    selCurr.insert(selCurr.begin(), hCurr);
  } else if (selectionType == 2) {
    for (vector<double100>::iterator etaVals = etaStore.back().begin();
         etaVals != etaStore.back().end(); etaVals++) {
      etaCurr.push_back(*etaVals);
    }
    etaProp.insert(etaProp.end(), selProp.begin(), selProp.end() - 1);
    selCurr.insert(selCurr.begin(), etaCurr.begin(), etaCurr.end());
  }

  vector<int> Kappa = kappaStore.back();
  vector<vector<double100>> Psi = psiStore.back(), Omega = omegaStore.back(),
                            Ksi = ksiStore.back();

  // Set phimin/max/range for current and proposed sigma
  vector<double100> phiminmaxrangeProp{Proposal.phiMin, Proposal.phiMax,
                                       Proposal.phiMax - Proposal.phiMin},
      phiminmaxrangeCurr{Current.phiMin, Current.phiMax,
                         Current.phiMax - Current.phiMin};
  // Calculate contributions to acceptance probability
  bool AcceptProposals;
  double100 priorRatio = selPriorRatio(sigmaProp, hProp, etaProp, sigmaCurr,
                                       hCurr, etaCurr, MCMCo);
  double100 proposalRatio = selProposalRatio(sigmaProp, hProp, etaProp,
                                             sigmaCurr, hCurr, etaCurr, MCMCo);
  double100 alpha =
      exp(priorRatio + proposalRatio + Proposal.Atilde(Xtn) -
          Current.Atilde(Xtn) -  //(sigmaProp - sigmaCurr) * 0.5 * Xtn -
          (phiminmaxrangeProp[0] - phiminmaxrangeCurr[0]) *
              (tn - t0));  // - (phirangeProp-phirangeCurr)*(tn-t0);
  double100 skelProp = 1.0, skelCurr = 1.0;
  for (vector<vector<double100>>::iterator PsiIter = Psi.begin(),
                                           OmegaIter = Omega.begin(),
                                           KsiIter = Ksi.begin();
       PsiIter != Psi.end(); PsiIter++, OmegaIter++, KsiIter++) {
    for (vector<double100>::iterator PsiiIter = (*PsiIter).begin(),
                                     OmegaiIter = (*OmegaIter).begin(),
                                     KsiiIter = (*KsiIter).begin();
         PsiiIter != (*PsiIter).end(); PsiiIter++, OmegaiIter++, KsiiIter++) {
      if ((*KsiiIter < phiminmaxrangeProp[2]) && (*OmegaiIter >= 0.0)) {
        skelProp *=
            (1.0 - ((Proposal.Phitilde(*OmegaiIter) - phiminmaxrangeProp[0]) /
                    phiminmaxrangeProp[2]));
      }
      if ((*KsiiIter < phiminmaxrangeCurr[2]) && (*OmegaiIter >= 0.0)) {
        skelCurr *=
            (1.0 - ((Current.Phitilde(*OmegaiIter) - phiminmaxrangeCurr[0]) /
                    phiminmaxrangeCurr[2]));
      }
    }
  }
  alpha *= (skelProp / skelCurr);
  boost::random::uniform_01<> U01;
  double100 u = U01(gen);
  if (alpha >= u) {
    AcceptProposals = true;
  } else {
    AcceptProposals = false;
  }
  if (AcceptProposals) {
    selOut = selProp;
  } else {
    selOut = selCurr;
  }
  vector<double100> phiminmax;
  double100 Afn;
  if (selOut == selCurr) {
    phiminmax = {Current.phiMin, Current.phiMax,
                 Current.phiMax - Current.phiMin};
    Afn = Current.Atilde(Xtn);
  } else {
    phiminmax = {Proposal.phiMin, Proposal.phiMax,
                 Proposal.phiMax - Proposal.phiMin};
    Afn = Proposal.Atilde(Xtn);
  }
  double100 likelihood =
      exp(Afn - (phiminmax[0] * (tn - t0))) *
      pow(Xtn, static_cast<double100>(Data.back())) *
      pow(1.0 - Xtn, static_cast<double100>(
                         Samplesizes.back() -
                         Data.back()));  // * pow(times.front() - t0,counter);
  vector<double100> t_incs;
  if (t0 < times.front()) {
    t_incs.push_back(times.front() - t0);
  } else {
    t_incs.push_back(0.0);
  }
  for (vector<double100>::iterator titer = times.begin() + 1;
       titer != times.end(); titer++) {
    if (t_incs.back() == 0.0) {
      t_incs.push_back(*titer - t0);
    } else {
      t_incs.push_back(*titer - *(titer - 1));
    }
  }
  vector<double100>::iterator titer = t_incs.begin();
  for (vector<vector<double100>>::iterator OmegaI = omegaStore.back().begin(),
                                           KsiI = ksiStore.back().begin();
       titer != t_incs.end(); OmegaI++, KsiI++) {
    int kappaCounter = 0;
    for (vector<double100>::iterator OmegaIJ = (*OmegaI).begin(),
                                     KsiIJ = (*KsiI).begin();
         OmegaIJ != (*OmegaI).end(); OmegaIJ++, KsiIJ++) {
      if ((*KsiIJ < phiminmax[2]) && (*OmegaIJ >= 0.0)) {
        kappaCounter++;
        if (selOut == selCurr) {
          likelihood *= (1.0 - ((Current.Phitilde(*OmegaIJ) - phiminmax[0]) /
                                (phiminmax[2])));
        } else {
          likelihood *= (1.0 - ((Proposal.Phitilde(*OmegaIJ) - phiminmax[0]) /
                                (phiminmax[2])));
        }
      }
    }
    double100 multiplicator =
        pow((phiminmax[2] * (*titer)), static_cast<double100>(kappaCounter)) *
        (1.0 / boost::math::factorial<double100>(kappaCounter));
    likelihood *= exp(-phiminmax[2] * (*titer)) * multiplicator;
    titer++;
  }
  selLikelihood.push_back(likelihood);
}

// ACTUAL PROGRAM

void MCMCSampler::RunSampler(const MCMCOptions &MCMCo) {
  const Options o(MCMCo.diffusion_threshold, MCMCo.bridge_threshold);
  clock_t starttimer = clock();
  // Create containers for convergence criteria
  vector<double100> sigmean, sigstdev, sigsqsums, t0mean, t0stdev, t0sqsums,
      hmean, hstdev, hsqsums;
  vector<vector<double100>> etamean, etastdev, etasqsums;
  double100 sigmeanmax, sigmeanmin, sigstdevmax, sigstdevmin, t0meanmax,
      t0meanmin, t0stdevmax, t0stdevmin, hmeanmax, hmeanmin, hstdevmax,
      hstdevmin;
  vector<double100> etameanmax(MCMCo.etaPriorMeans.size()),
      etameanmin(MCMCo.etaPriorMeans.size()),
      etastdevmax(MCMCo.etaPriorMeans.size()),
      etastdevmin(MCMCo.etaPriorMeans.size());

  // Convergence Details Printer
  std::cout << "Burn-in set to: " << MCMCo.burnIn << " samples" << endl;
  if (selectionType == 0) {
    std::cout << "Mean precision for sigma: " << MCMCo.sigmaMeanPrec << endl;
    std::cout << "Stdev precision for sigma: " << MCMCo.sigmaStdevPrec << endl;
  } else if (selectionType == 1) {
    std::cout << "Mean precision for sigma: " << MCMCo.sigmaMeanPrec << endl;
    std::cout << "Stdev precision for sigma: " << MCMCo.sigmaStdevPrec << endl;
    std::cout << "Mean precision for h: " << MCMCo.hMeanPrec << endl;
    std::cout << "Stdev precision for h: " << MCMCo.hStdevPrec << endl;
  } else {
    std::cout << "Mean precision for sigma: " << MCMCo.sigmaMeanPrec << endl;
    std::cout << "Stdev precision for sigma: " << MCMCo.sigmaStdevPrec << endl;
    int count = 0;
    for (vector<double100>::const_reverse_iterator
             etaMP = MCMCo.etaMeanPrec.crbegin(),
             etaSP = MCMCo.etaStdevPrec.crbegin();
         etaMP != MCMCo.etaMeanPrec.crend(); etaMP++, etaSP++) {
      std::cout << "Mean precision for " << count
                << "-order coefficient : " << *etaMP << endl;
      std::cout << "Stdev precision for " << count
                << "-order coefficient : " << *etaSP << endl;
    }
  }
  std::cout << "Mean precision for t0: " << MCMCo.t0MeanPrec << endl;
  std::cout << "Stdev precision for t0: " << MCMCo.t0StdevPrec << endl;
  std::cout << "Zeros counter is: " << zerosCounter << endl;
  std::cout << "tc is: " << tc << endl;

  std::cout << "Xstore initialised with: " << std::endl;
  int xc = 0;
  for (vector<double100>::iterator xit = XStore.back().begin();
       xit != XStore.back().end(); xit++) {
    std::cout << "X[" << xc << "] = " << *xit << ", ";
    xc++;
  }
  std::cout << "." << std::endl;

  // Program Runner
  {
    bool convergence = false;
    int itercount = 1;
    int aux_scale = 0;

    SetMutation(MCMCo);                     // Setting mutation rates
    InitialiseSel(MCMCo, MCMCSampler_gen);  // Initialise s
    InitialiseT0(MCMCo, MCMCSampler_gen);   // Initialise t_0
    double100 configSigma, configH;
    vector<double100> configSel;
    if (selectionType == 0) {
      configSigma = sigmaStore.back();
      configH = 0.0;
      configSel = vector<double100>{};
    } else if (selectionType == 1) {
      configSigma = sigmaStore.back();
      configH = hStore.back();
      configSel = vector<double100>{};
    } else {
      configSigma = sigmaStore.back();
      configH = hStore.back();
      configSel = etaStore.back();
    }
    WrightFisher InitWF(
        theta, true, configSigma, selectionType, configH, configSel.size(),
        configSel);  // Initialise WF class with initial setup values
    InitialiseSkeleton(InitWF, o,
                       MCMCSampler_gen);  // Initialise skeleton points
    InitialisePseudo(
        InitWF, o, MCMCSampler_gen);  // Initiailise pseudo-marginal quantities

    // Print info on current run and datapoints used
    ofstream OtherInput(MCMCo.FilenameOtherInput),
        DatapointsInput(MCMCo.FilenameDatapointsInput);
    printDatapoints(DatapointsInput);
    printOtherInput(OtherInput, MCMCo);

    clock_t clock1 = clock(), clock2;
    double100 elapsed_secs = double(clock1 - starttimer) / CLOCKS_PER_SEC;
    std::cout << "Time elapsed so far: " << elapsed_secs << endl;

    // Enter main MCMC loop
    while (!convergence) {
      vector<double100> selOut;
      double100 sigmaProp, hProp, sigmaCurr, hCurr;
      vector<double100> etaProp, etaCurr;
      vector<double100> selProp = selProposal(MCMCo, MCMCSampler_gen);
      if (selectionType == 0) {
        sigmaProp = selProp.back();
        sigmaCurr = sigmaStore.back();
      } else if (selectionType == 1) {
        sigmaProp = selProp.back();
        hProp = selProp.front();
        sigmaCurr = sigmaStore.back();
        hCurr = hStore.back();
      } else {
        sigmaProp = selProp.back();
        etaProp.insert(etaProp.end(), selProp.begin(), selProp.end() - 1);
        sigmaCurr = sigmaStore.back();
        etaCurr = etaStore.back();
      }
      bool isSigmaPropOK = true;
      WrightFisher Proposal(theta, true, sigmaProp, selectionType, hProp,
                            configSel.size(),
                            etaProp);  // Have another WF class for the
                                       // proposed diffusion
      WrightFisher Current(
          theta, true, sigmaCurr, selectionType, hCurr, configSel.size(),
          etaCurr);  // Have a WF class for the current diffusion

      double100 lambdaMax = isSigmaPropOK
                                ? max(Proposal.phiMax - Proposal.phiMin,
                                      Current.phiMax - Current.phiMin)
                                : (Current.phiMax - Current.phiMin);

      int i = zerosCounter == 0 ? 1 : zerosCounter == 1 ? 2 : 3;

      // Iterate over latent diffusion path
      for (vector<double100>::iterator iter = XStore.front().begin();
           iter != XStore.front().end(); iter++) {
        // We are at the start of the path
        if (iter == XStore.front().begin()) {
          vector<double100> XOut;
          vector<int> kappaOut;
          vector<vector<double100>> omegaOut, psiOut, gammaOut, ksiOut;

          // Generate new values and run AR step
          UpdateT0(Current, XOut, kappaOut, omegaOut, psiOut, gammaOut, ksiOut,
                   lambdaMax, o, MCMCo, MCMCSampler_gen);

          // Update store variables accordingly
          XStore.push_back(XOut);
          kappaStore.push_back(kappaOut);
          omegaStore.push_back(omegaOut);
          psiStore.push_back(psiOut);
          gammaStore.push_back(gammaOut);
          ksiStore.push_back(ksiOut);

          vector<int> kappaVec(kappaOut.begin(), kappaOut.begin() + i);
          vector<vector<double100>> omegaVec(omegaOut.begin(),
                                             omegaOut.begin() + i),
              psiVec(psiOut.begin(), psiOut.begin() + i),
              gammaVec(gammaOut.begin(), gammaOut.begin() + i),
              ksiVec(ksiOut.begin(), ksiOut.begin() + i);

          kappaStore.push_back(kappaVec);
          omegaStore.push_back(omegaVec);
          psiStore.push_back(psiVec);
          gammaStore.push_back(gammaVec);
          ksiStore.push_back(ksiVec);

          // In cases when we have consecutive zeros at the start, we need to
          // advance to the next section not containing consecutive zero
          // observations!
          std::advance(iter, zerosCounter);

        }

        // We are at the end of the diffusion path
        else if (iter == XStore.front().end() - 1) {
          vector<double100> XOut;
          vector<int> kappaOut;
          vector<vector<double100>> omegaOut, psiOut, gammaOut, ksiOut;

          // Generate new values and run AR step
          UpdateXtn(Current, XOut, kappaOut, omegaOut, psiOut, gammaOut, ksiOut,
                    lambdaMax, o, MCMCSampler_gen);

          // Update store variables accordingly
          XStore.back().push_back(XOut.front());
          kappaStore.back().push_back(kappaOut.front());
          omegaStore.back().push_back(omegaOut.front());
          psiStore.back().push_back(psiOut.front());
          gammaStore.back().push_back(gammaOut.front());
          ksiStore.back().push_back(ksiOut.front());
        }
        // We are in an interior path segment
        else {
          vector<double100> XOut;
          vector<int> kappaOut;
          vector<vector<double100>> omegaOut, psiOut, gammaOut, ksiOut;

          // Generate new values and run AR step
          UpdateXti(Current, XOut, kappaOut, omegaOut, psiOut, gammaOut, ksiOut,
                    lambdaMax, i, o, MCMCSampler_gen);

          // Update store variables accordingly
          XStore.back().push_back(XOut.front());
          kappaStore.back().push_back(kappaOut.front());
          omegaStore.back().push_back(omegaOut.front());
          psiStore.back().push_back(psiOut.front());
          gammaStore.back().push_back(gammaOut.front());
          ksiStore.back().push_back(ksiOut.front());

          // Because we store these variables in a matrix of double the size, we
          // need to update it accordingly
          if (itercount != 0) {
            int readjusted_counter =
                itercount - (aux_scale * MCMCo.printCounter);

            kappaStore[(2 * readjusted_counter)].insert(
                kappaStore[(2 * readjusted_counter)].end(), kappaOut.back());
            omegaStore[(2 * readjusted_counter)].insert(
                omegaStore[(2 * readjusted_counter)].end(), omegaOut.back());
            psiStore[(2 * readjusted_counter)].insert(
                psiStore[(2 * readjusted_counter)].end(), psiOut.back());
            gammaStore[(2 * readjusted_counter)].insert(
                gammaStore[(2 * readjusted_counter)].end(), gammaOut.back());
            ksiStore[(2 * readjusted_counter)].insert(
                ksiStore[(2 * readjusted_counter)].end(), ksiOut.back());
          }
          ++i;
        }
      }

      if (isSigmaPropOK) {
        // Generate new sigma value and run AR step
        UpdateSigma(Current, Proposal, selOut, selProp, MCMCo, MCMCSampler_gen);
        if (selectionType == 0) {
          sigmaStore.push_back(selOut.back());
        } else if (selectionType == 1) {
          sigmaStore.push_back(selOut.back());
          hStore.push_back(selOut.front());
        } else {
          sigmaStore.push_back(selOut.back());
          vector<double100> etaOut(selOut.begin(), selOut.end() - 1);
          etaStore.push_back(etaOut);
        }
      } else {  // Otherwise keep the previous values
        sigmaStore.push_back(sigmaStore.back());
        if (selectionType == 1) {
          hStore.push_back(hStore.back());
        } else {
          etaStore.push_back(etaStore.back());
        }
        selLikelihood.push_back(
            selLikelihood.back());  // technically incorrect, as you would need
                                    // to re-calculate likelihood...
      }

      // Compute convergence statistics once we exceed burn-in
      if (itercount >= MCMCo.burnIn) {
        if (sigmean.empty()) {  // Initialise all the quantities
          sigmean.push_back(sigmaStore.back());
          sigsqsums.push_back(sigmaStore.back() * sigmaStore.back());
          sigstdev.push_back(0.0);
          t0mean.push_back(t0Store.back());
          t0sqsums.push_back(t0Store.back() * t0Store.back());
          t0stdev.push_back(0.0);
          if (selectionType == 1) {
            hmean.push_back(hStore.back());
            hsqsums.push_back(hStore.back() * hStore.back());
            hstdev.push_back(0.0);
          } else if (selectionType == 2) {
            vector<double100> eta_m, eta_sqsums, eta_stdev;
            for (vector<double100>::reverse_iterator etaVals =
                     etaStore.back().rbegin();
                 etaVals != etaStore.back().rend(); etaVals++) {
              eta_m.insert(eta_m.begin(), *etaVals);
              eta_sqsums.insert(eta_sqsums.begin(), (*etaVals) * (*etaVals));
              eta_stdev.insert(eta_stdev.begin(), 0.0);
            }
            etamean.push_back(eta_m);
            etasqsums.push_back(eta_sqsums);
            etastdev.push_back(eta_stdev);
          }
        } else {  // Switch to using size of storeing vector minus burnIn
          sigmean.push_back(
              (1.0 / (static_cast<double100>(itercount - MCMCo.burnIn) + 1.0)) *
              ((sigmean.back() *
                static_cast<double100>(itercount - MCMCo.burnIn)) +
               sigmaStore.back()));
          sigsqsums.push_back(sigsqsums.back() +
                              (sigmaStore.back() * sigmaStore.back()));
          if (sigstdev.size() == 1 && itercount < MCMCo.burnIn + 2) {
            sigstdev.push_back(sqrt(((sigmean.front() - sigmean.back()) *
                                     (sigmean.front() - sigmean.back())) +
                                    ((sigmaStore.back() - sigmean.back()) *
                                     (sigmaStore.back() - sigmean.back()))));
          } else {
            sigstdev.push_back(sqrt(
                (1.0 / (static_cast<double100>(itercount - MCMCo.burnIn))) *
                (sigsqsums.back() -
                 (static_cast<double100>(itercount - MCMCo.burnIn + 1.0) *
                  sigmean.back() * sigmean.back()))));
          }
          t0mean.push_back(
              (1.0 / (static_cast<double100>(itercount - MCMCo.burnIn) + 1.0)) *
              ((t0mean.back() *
                static_cast<double100>(itercount - MCMCo.burnIn)) +
               t0Store.back()));
          t0sqsums.push_back(t0sqsums.back() +
                             (t0Store.back() * t0Store.back()));
          if (t0stdev.size() == 1 && itercount < MCMCo.burnIn + 2) {
            t0stdev.push_back(sqrt(((t0mean.front() - t0mean.back()) *
                                    (t0mean.front() - t0mean.back())) +
                                   ((t0Store.back() - t0mean.back()) *
                                    (t0Store.back() - t0mean.back()))));
          } else {
            t0stdev.push_back(sqrt(
                (1.0 / (static_cast<double100>(itercount - MCMCo.burnIn))) *
                (t0sqsums.back() -
                 (static_cast<double100>(itercount - MCMCo.burnIn - 1) *
                  t0mean.back() * t0mean.back()))));
          }
          if (selectionType == 1) {
            hmean.push_back(
                (1.0 /
                 (static_cast<double100>(itercount - MCMCo.burnIn) + 1.0)) *
                ((hmean.back() *
                  static_cast<double100>(itercount - MCMCo.burnIn)) +
                 hStore.back()));
            hsqsums.push_back(hsqsums.back() + (hStore.back() * hStore.back()));
            if (hstdev.size() == 1 && itercount < MCMCo.burnIn + 2) {
              hstdev.push_back(sqrt(((hmean.front() - hmean.back()) *
                                     (hmean.front() - hmean.back())) +
                                    ((hStore.back() - hmean.back()) *
                                     (hStore.back() - hmean.back()))));
            } else {
              hstdev.push_back(sqrt(
                  (1.0 / (static_cast<double100>(itercount - MCMCo.burnIn))) *
                  (hsqsums.back() -
                   (static_cast<double100>(itercount - MCMCo.burnIn + 1.0) *
                    hmean.back() * hmean.back()))));
            }
          } else if (selectionType == 2) {
            vector<double100> eta_m, eta_sqsums, eta_stdev;
            bool condition = !etastdev.empty() && etastdev[0].size() == 1;
            for (vector<double100>::reverse_iterator
                     etaVals = etaStore.back().rbegin(),
                     etam = etamean.back().rbegin(),
                     etasq = etasqsums.back().rbegin(),
                     etastd = etastdev.back().rbegin();
                 etaVals != etaStore.back().rend();
                 etaVals++, etam++, etasq++, etastd++) {
              eta_m.insert(
                  eta_m.begin(),
                  (1.0 /
                   (static_cast<double100>(itercount - MCMCo.burnIn) + 1.0)) *
                          ((*etam) *
                           static_cast<double100>(itercount - MCMCo.burnIn)) +
                      (*etaVals));
              eta_sqsums.insert(eta_sqsums.begin(),
                                (*etasq) + ((*etam) * (*etam)));
              if (condition && itercount < MCMCo.burnIn + 2) {
                eta_stdev.insert(
                    eta_stdev.begin(),
                    sqrt(((*etam) - eta_m.front()) * ((*etam) - eta_m.front()) +
                         ((*etaVals) - eta_m.front()) *
                             ((*etaVals) - eta_m.front())));
              } else {
                eta_stdev.insert(
                    eta_stdev.begin(),
                    sqrt((1.0 /
                          (static_cast<double100>(itercount - MCMCo.burnIn))) *
                         (*etastd - (static_cast<double100>(
                                         itercount - MCMCo.burnIn + 1.0) *
                                     eta_m.front() * eta_m.front()))));
              }
            }
          }
        }
      }

      if (itercount % 100 == 0) {
        clock2 = clock();
        double time_el = double(clock2 - starttimer) / CLOCKS_PER_SEC;
        std::cerr << "Iteration " << itercount << endl;
        std::cerr << "Time elapsed " << time_el << endl;
      }

      // Print output to file
      if (itercount % MCMCo.printCounter == 0) {
        clock2 = clock();
        elapsed_secs = double(clock2 - starttimer) / CLOCKS_PER_SEC;
        std::cout << "Iteration " << itercount << endl;
        std::cout << "Time elapsed: " << elapsed_secs << endl;
        if (itercount > MCMCo.burnIn) {
          std::cout << "Mean of sigma is " << sigmean.back() << endl;
          std::cout << "Stdev of sigma is " << sigstdev.back() << endl;
          std::cout << "Mean of t0 is " << t0mean.back() << endl;
          std::cout << "Stdev of t0 is " << t0stdev.back() << endl;
        }

        ofstream OutputSigma(MCMCo.FilenameSigma, ofstream::app),
            OutputT0(MCMCo.FilenameT0, ofstream::app),
            OutputSigMean(MCMCo.FilenameSigMean, ofstream::app),
            OutputSigStd(MCMCo.FilenameSigStd, ofstream::app),
            OutputT0Mean(MCMCo.FilenameT0Mean, ofstream::app),
            OutputT0Std(MCMCo.FilenameT0Std, ofstream::app);
        printVDouble(OutputSigma, sigmaStore, MCMCo.printCounter);
        printVDouble(OutputT0, t0Store, MCMCo.printCounter);
        printVDouble(OutputSigMean, sigmean, MCMCo.printCounter);
        printVDouble(OutputSigStd, sigstdev, MCMCo.printCounter);
        printVDouble(OutputT0Mean, t0mean, MCMCo.printCounter);
        printVDouble(OutputT0Std, t0stdev, MCMCo.printCounter);
        if (selectionType == 1) {
          ofstream OutputH(MCMCo.FilenameH, ofstream::app),
              OutputHMean(MCMCo.FilenameHMean, ofstream::app),
              OutputHStd(MCMCo.FilenameHStd, ofstream::app);
          printVDouble(OutputH, hStore, MCMCo.printCounter);
          printVDouble(OutputHMean, hmean, MCMCo.printCounter);
          printVDouble(OutputHStd, hstdev, MCMCo.printCounter);
        } else if (selectionType == 2) {
          ofstream OutputEta(MCMCo.FilenameEta, ofstream::app),
              OutputEtaMean(MCMCo.FilenameEtaMean, ofstream::app),
              OutputEtaStd(MCMCo.FilenameEtaStd, ofstream::app);
          printMatrixDouble(OutputEta, etaStore, MCMCo.printCounter);
          printMatrixDouble(OutputEtaMean, etamean, MCMCo.printCounter);
          printMatrixDouble(OutputEtaStd, etastdev, MCMCo.printCounter);
        }

        if (MCMCo.Save_Likelihood) {
          ofstream OutputSigmaLikelihood(MCMCo.FilenameSigmaLikelihood,
                                         ofstream::app),
              OutputT0Likelihood(MCMCo.FilenameT0Likelihood, ofstream::app),
              OutputX(MCMCo.FilenameX, ofstream::app);
          printVDouble(OutputSigmaLikelihood, selLikelihood,
                       MCMCo.printCounter);
          printVDouble(OutputT0Likelihood, t0Likelihood, MCMCo.printCounter);
          printMatrixDouble(OutputX, XStore, MCMCo.printCounter);
        }
        if (MCMCo.Save_Aux) {
          ofstream OutputPseudo(MCMCo.FilenamePseudo, ofstream::app),
              OutputOmega(MCMCo.FilenameOmega, ofstream::app),
              OutputPsi(MCMCo.FilenamePsi, ofstream::app),
              OutputGamma(MCMCo.FilenameGamma, ofstream::app),
              OutputKsi(MCMCo.FilenameKsi, ofstream::app),
              OutputT0KsiCounter(MCMCo.FilenameT0KsiCounter, ofstream::app),
              OutputKappa(MCMCo.FilenameKappa, ofstream::app);
          printVDouble(OutputPseudo, pseudoStore, MCMCo.printCounter);
          printMatrixInt(OutputKappa, kappaStore, 2 * MCMCo.printCounter);
          print3DMatrixDouble(OutputOmega, omegaStore, 2 * MCMCo.printCounter);
          print3DMatrixDouble(OutputPsi, psiStore, 2 * MCMCo.printCounter);
          print3DMatrixDouble(OutputGamma, gammaStore, 2 * MCMCo.printCounter);
          print3DMatrixDouble(OutputKsi, ksiStore, 2 * MCMCo.printCounter);
          printVInt(OutputT0KsiCounter, t0ksiCounter, MCMCo.printCounter);
        }

        // Keep track of the maximum and minimum for the means and standard
        // deviations within last lookBack samples
        if (itercount > MCMCo.burnIn + MCMCo.lookBack) {
          sigmeanmax = *(max_element(sigmean.begin(), sigmean.end()));
          sigmeanmin = *(min_element(sigmean.begin(), sigmean.end()));
          sigstdevmax = *(max_element(sigstdev.begin(), sigstdev.end()));
          sigstdevmin = *(min_element(sigstdev.begin(), sigstdev.end()));
          t0meanmax = *(max_element(t0mean.begin(), t0mean.end()));
          t0meanmin = *(min_element(t0mean.begin(), t0mean.end()));
          t0stdevmax = *(max_element(t0stdev.begin(), t0stdev.end()));
          t0stdevmin = *(min_element(t0stdev.begin(), t0stdev.end()));
          if (selectionType == 1) {
            hmeanmax = *(max_element(hmean.begin(), hmean.end()));
            hmeanmin = *(min_element(hmean.begin(), hmean.end()));
            hstdevmax = *(max_element(hstdev.begin(), hstdev.end()));
            hstdevmin = *(min_element(hstdev.begin(), hstdev.end()));
          } else if (selectionType == 2) {
            int numCols = etameanmax.size(), numRows = etamean[0].size();
            for (int col = 0; col < numCols; col++) {
              vector<double> dummy_mean(numRows), dummy_stdev(numRows);
              for (int row = 0; row < numRows; row++) {
                dummy_mean[row] = etamean[col][row];
                dummy_stdev[row] = etastdev[col][row];
              }
              etameanmax[col] =
                  *std::max_element(dummy_mean.begin(), dummy_mean.end());
              etameanmin[col] =
                  *std::min_element(dummy_mean.begin(), dummy_mean.end());
              etastdevmax[col] =
                  *std::max_element(dummy_stdev.begin(), dummy_stdev.end());
              etastdevmin[col] =
                  *std::min_element(dummy_stdev.begin(), dummy_stdev.end());
            }
          }
          convergence =
              ((abs(sigmeanmax - sigmeanmin) < MCMCo.sigmaMeanPrec) &&
               (abs(sigstdevmax - sigstdevmin) < MCMCo.sigmaStdevPrec) &&
               (abs(t0meanmax - t0meanmin) < MCMCo.t0MeanPrec) &&
               (abs(t0stdevmax - t0stdevmin) < MCMCo.t0StdevPrec));
          // TODO: Remove this condition!!!
          convergence = false;
          if (itercount % MCMCo.printCounter == 0) {
            std::cout << "Convergence diagnostics: " << endl;
            std::cout << "Range of sigma mean for last " << MCMCo.lookBack
                      << " samples is " << abs(sigmeanmax - sigmeanmin) << endl;
            std::cout << "Max of sigma mean is " << sigmeanmax << std::endl;
            std::cout << "Min of sigma mean is " << sigmeanmin << std::endl;
            std::cout << "Range of sigma stdev for last " << MCMCo.lookBack
                      << " samples is " << abs(sigstdevmax - sigstdevmin)
                      << endl;
            std::cout << "Max of sigma stdev is " << sigstdevmax << std::endl;
            std::cout << "Min of sigma stdev is " << sigstdevmin << std::endl;
            std::cout << "Range of t0 mean for last " << MCMCo.lookBack
                      << " samples is " << abs(t0meanmax - t0meanmin) << endl;
            std::cout << "Max of t0 mean is " << t0meanmax << std::endl;
            std::cout << "Min of t0 mean is " << t0meanmin << std::endl;
            std::cout << "Range of t0 stdev for last " << MCMCo.lookBack
                      << " samples is " << abs(t0stdevmax - t0stdevmin) << endl;
            std::cout << "Max of t0 stdev is " << t0stdevmax << std::endl;
            std::cout << "Min of t0 stdev is " << t0stdevmin << std::endl;
            if (selectionType == 1) {
              std::cout << "Range of h mean for last " << MCMCo.lookBack
                        << " samples is " << abs(hmeanmax - hmeanmin) << endl;
              std::cout << "Max of h mean is " << hmeanmax << std::endl;
              std::cout << "Min of h mean is " << hmeanmin << std::endl;
              std::cout << "Range of h stdev for last " << MCMCo.lookBack
                        << " samples is " << abs(hstdevmax - hstdevmin) << endl;
              std::cout << "Max of h stdev is " << hstdevmax << std::endl;
              std::cout << "Min of h stdev is " << hstdevmin << std::endl;
            } else if (selectionType == 2) {
              int order_count = 0;
              for (vector<double>::reverse_iterator
                       etaMmax = etameanmax.rbegin(),
                       etaMmin = etameanmin.rbegin(),
                       etaSmax = etastdevmax.rbegin(),
                       etaSmin = etastdevmin.rbegin();
                   etaMmax != etameanmax.rend();
                   etaMmax++, etaMmin++, etaSmax++, etaSmin++) {
                std::cout << "Range of " << order_count
                          << "-coefficient mean for last " << MCMCo.lookBack
                          << " samples is " << abs(*etaMmax - *etaMmin) << endl;
                std::cout << "Max of " << order_count
                          << "-coefficient  mean is " << *etaMmax << std::endl;
                std::cout << "Min of " << order_count
                          << "-coefficient  mean is " << *etaMmin << std::endl;
                std::cout << "Range of h stdev for last " << MCMCo.lookBack
                          << " samples is " << abs(*etaSmax - *etaSmin) << endl;
                std::cout << "Max of " << order_count
                          << "-coefficient  stdev is " << *etaSmax << std::endl;
                std::cout << "Min of " << order_count
                          << "-coefficient  stdev is " << *etaSmin << std::endl;
                order_count++;
              }
            }
          }
        }

        // Memory clean up!
        sigmaStore.erase(sigmaStore.begin(), sigmaStore.end() - 1);
        t0Store.erase(t0Store.begin(), t0Store.end() - 1);
        if (selectionType == 1) {
          hStore.erase(hStore.begin(), hStore.end() - 1);
        } else if (selectionType == 2) {
          etaStore.erase(etaStore.begin(), etaStore.end() - 1);
        }
        selLikelihood.erase(selLikelihood.begin(), selLikelihood.end() - 1);
        t0Likelihood.erase(t0Likelihood.begin(), t0Likelihood.end() - 1);
        XStore.erase(XStore.begin(), XStore.end() - 2);
        pseudoStore.erase(pseudoStore.begin(), pseudoStore.end() - 1);
        kappaStore.erase(kappaStore.begin(), kappaStore.end() - 2);
        omegaStore.erase(omegaStore.begin(), omegaStore.end() - 2);
        psiStore.erase(psiStore.begin(), psiStore.end() - 2);
        gammaStore.erase(gammaStore.begin(), gammaStore.end() - 2);
        ksiStore.erase(ksiStore.begin(), ksiStore.end() - 2);
        t0ksiCounter.erase(t0ksiCounter.begin(), t0ksiCounter.end() - 1);
        if (itercount > MCMCo.burnIn) {
          sigmean.erase(sigmean.begin(), sigmean.end() - 1);
          sigstdev.erase(sigstdev.begin(), sigstdev.end() - 1);
          t0mean.erase(t0mean.begin(), t0mean.end() - 1);
          t0stdev.erase(t0stdev.begin(), t0stdev.end() - 1);
          if (selectionType == 1) {
            hmean.erase(hmean.begin(), hmean.end() - 1);
            hstdev.erase(hstdev.begin(), hstdev.end() - 1);
          } else if (selectionType == 2) {
            etamean.erase(etamean.begin(), etamean.end() - 1);
            etastdev.erase(etastdev.begin(), etastdev.end() - 1);
          }
        }
        aux_scale++;
      }

      itercount++;
    }
    clock_t clock3 = clock();
    elapsed_secs = double(clock3 - starttimer) / CLOCKS_PER_SEC;
    std::cout << "Computation performed over " << itercount << " iterations."
              << endl;
    std::cout << "Total time elapsed: " << elapsed_secs << " s" << endl;

    // Once we have achieved convergence, print out the last few rows of output
    // which were not printed to file
    int remainder = itercount % MCMCo.printCounter;
    ofstream OutputSigma(MCMCo.FilenameSigma, ofstream::app),
        OutputT0(MCMCo.FilenameT0, ofstream::app),
        OutputSigMean(MCMCo.FilenameSigMean, ofstream::app),
        OutputSigStd(MCMCo.FilenameSigStd, ofstream::app),
        OutputT0Mean(MCMCo.FilenameT0Mean, ofstream::app),
        OutputT0Std(MCMCo.FilenameT0Std, ofstream::app);
    printVDouble(OutputSigma, sigmaStore, remainder);
    printVDouble(OutputT0, t0Store, remainder);
    printVDouble(OutputSigMean, sigmean, remainder);
    printVDouble(OutputSigStd, sigstdev, remainder);
    printVDouble(OutputT0Mean, t0mean, remainder);
    printVDouble(OutputT0Std, t0stdev, remainder);
    if (selectionType == 1) {
      ofstream OutputH(MCMCo.FilenameH, ofstream::app),
          OutputHMean(MCMCo.FilenameHMean, ofstream::app),
          OutputHStd(MCMCo.FilenameHStd, ofstream::app);
      printVDouble(OutputH, hStore, remainder);
      printVDouble(OutputHMean, hmean, remainder);
      printVDouble(OutputHStd, hstdev, remainder);
    } else if (selectionType == 2) {
      ofstream OutputEta(MCMCo.FilenameEta, ofstream::app),
          OutputEtaMean(MCMCo.FilenameEtaMean, ofstream::app),
          OutputEtaStd(MCMCo.FilenameEtaStd, ofstream::app);
      printMatrixDouble(OutputEta, etaStore, remainder);
      printMatrixDouble(OutputEtaMean, etamean, remainder);
      printMatrixDouble(OutputEtaStd, etastdev, remainder);
    }
    if (MCMCo.Save_Likelihood) {
      ofstream OutputSigmaLikelihood(MCMCo.FilenameSigmaLikelihood,
                                     ofstream::app),
          OutputT0Likelihood(MCMCo.FilenameT0Likelihood, ofstream::app),
          OutputX(MCMCo.FilenameX, ofstream::app);
      printVDouble(OutputSigmaLikelihood, selLikelihood, remainder);
      printVDouble(OutputT0Likelihood, t0Likelihood, remainder);
      printMatrixDouble(OutputX, XStore, remainder);
    }
    if (MCMCo.Save_Aux) {
      ofstream OutputPseudo(MCMCo.FilenamePseudo, ofstream::app),
          OutputOmega(MCMCo.FilenameOmega, ofstream::app),
          OutputPsi(MCMCo.FilenamePsi, ofstream::app),
          OutputGamma(MCMCo.FilenameGamma, ofstream::app),
          OutputKsi(MCMCo.FilenameKsi, ofstream::app),
          OutputT0KsiCounter(MCMCo.FilenameT0KsiCounter, ofstream::app),
          OutputKappa(MCMCo.FilenameKappa, ofstream::app);
      printVDouble(OutputPseudo, pseudoStore, remainder);
      printMatrixInt(OutputKappa, kappaStore, 2 * remainder);
      print3DMatrixDouble(OutputOmega, omegaStore, 2 * (remainder + 1));
      print3DMatrixDouble(OutputPsi, psiStore, 2 * (remainder + 1));
      print3DMatrixDouble(OutputGamma, gammaStore, 2 * (remainder + 1));
      print3DMatrixDouble(OutputKsi, ksiStore, 2 * (remainder + 1));
      printVInt(OutputT0KsiCounter, t0ksiCounter, remainder);
    }
  }
}