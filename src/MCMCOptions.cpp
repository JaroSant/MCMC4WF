#include "MCMCOptions.h"

#include <boost/filesystem.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>

void MCMCOptions::OutputNamer() {
  //    char RootFolder;
  //    std::cout << "Please enter directory where you want to save the program
  //    output: " << endl; std::cin >> RootFolder;
  //
  //    if (!boost::filesystem::exists(RootFolder))
  //    {
  //        std::cout << "Directory entered was not found, so the following
  //        directory is being created now!" << endl; std::cout << "Directory
  //        path: " << RootFolder << endl;
  //        boost::filesystem::create_directory(RootFolder); std::cout <<
  //        "Directory created!" << endl;
  //    }
  //
  //    std::cout << "Output will be saved to directory path: " << endl;
  //    std::cout << RootFolder << endl;

  time_t t = time(0);  // get time now
  struct tm *now = localtime(&t);

  char bufferSigma[80];
  char bufferT0[80];
  char bufferX[80];
  char bufferOmega[80];
  char bufferPsi[80];
  char bufferGamma[80];
  char bufferKsi[80];
  char bufferDatapointsInput[80];
  char bufferOtherInput[80];
  char bufferSigMean[80];
  char bufferSigStd[80];
  char bufferT0Mean[80];
  char bufferT0Std[80];
  char bufferSigmaLikelihood[80];
  char bufferT0Likelihood[80];
  char bufferKappa[80];
  char bufferPseudo[80];
  char bufferT0KsiCounter[80];
  char bufferH[80];
  char bufferEta[80];
  char bufferHMean[80];
  char bufferHStd[80];
  char bufferEtaMean[80];
  char bufferEtaStd[80];

  strftime(bufferSigma, 80, "%Y-%m-%d-%H-%MSigma.txt", now);
  strftime(bufferT0, 80, "%Y-%m-%d-%H-%MT0.txt", now);
  strftime(bufferX, 80, "%Y-%m-%d-%H-%MX.txt", now);
  strftime(bufferOmega, 80, "%Y-%m-%d-%H-%MOmega.txt", now);
  strftime(bufferPsi, 80, "%Y-%m-%d-%H-%MPsi.txt", now);
  strftime(bufferGamma, 80, "%Y-%m-%d-%H-%MGamma.txt", now);
  strftime(bufferKsi, 80, "%Y-%m-%d-%H-%MKsi.txt", now);
  strftime(bufferDatapointsInput, 80, "%Y-%m-%d-%H-%MDatapoints.txt", now);
  strftime(bufferOtherInput, 80, "%Y-%m-%d-%H-%MOtherInfo.txt", now);
  strftime(bufferSigMean, 80, "%Y-%m-%d-%H-%MSigMean.txt", now);
  strftime(bufferSigStd, 80, "%Y-%m-%d-%H-%MSigStd.txt", now);
  strftime(bufferT0Mean, 80, "%Y-%m-%d-%H-%MT0Mean.txt", now);
  strftime(bufferT0Std, 80, "%Y-%m-%d-%H-%MT0Std.txt", now);
  strftime(bufferSigmaLikelihood, 80, "%Y-%m-%d-%H-%MSigmaLikelihood.txt", now);
  strftime(bufferT0Likelihood, 80, "%Y-%m-%d-%H-%MT0Likelihood.txt", now);
  strftime(bufferKappa, 80, "%Y-%m-%d-%H-%MKappa.txt", now);
  strftime(bufferPseudo, 80, "%Y-%m-%d-%H-%MPseudo.txt", now);
  strftime(bufferT0KsiCounter, 80, "%Y-%m-%d-%H-%MT0KsiCounter.txt", now);
  strftime(bufferH, 80, "%Y-%m-%d-%H-%MH.txt", now);
  strftime(bufferEta, 80, "%Y-%m-%d-%H-%MEta.txt", now);
  strftime(bufferHMean, 80, "%Y-%m-%d-%H-%MHMean.txt", now);
  strftime(bufferHStd, 80, "%Y-%m-%d-%H-%MHStd.txt", now);
  strftime(bufferEtaMean, 80, "%Y-%m-%d-%H-%MEtaMean.txt", now);
  strftime(bufferEtaStd, 80, "%Y-%m-%d-%H-%MEtaStd.txt", now);
  FilenameSigma = bufferSigma;
  FilenameT0 = bufferT0;
  FilenameX = bufferX;
  FilenameOmega = bufferOmega;
  FilenamePsi = bufferPsi;
  FilenameGamma = bufferGamma;
  FilenameKsi = bufferKsi;
  FilenameDatapointsInput = bufferDatapointsInput;
  FilenameOtherInput = bufferOtherInput;
  FilenameSigMean = bufferSigMean;
  FilenameSigStd = bufferSigStd;
  FilenameT0Mean = bufferT0Mean;
  FilenameT0Std = bufferT0Std;
  FilenameSigmaLikelihood = bufferSigmaLikelihood;
  FilenameT0Likelihood = bufferT0Likelihood;
  FilenameKappa = bufferKappa;
  FilenamePseudo = bufferPseudo;
  FilenameT0KsiCounter = bufferT0KsiCounter;
  FilenameH = bufferH;
  FilenameEta = bufferEta;
  FilenameHMean = bufferHMean;
  FilenameHStd = bufferHStd;
  FilenameEtaMean = bufferEtaMean;
  FilenameEtaStd = bufferEtaStd;
  std::cout << "Sigma output being saved to " << FilenameSigma << endl;
  std::cout << "t_0 output being saved to " << FilenameT0 << endl;
  std::cout << "X output being saved to " << FilenameX << endl;
  std::cout << "Omega output being saved to " << FilenameOmega << endl;
  std::cout << "Psi output being saved to " << FilenamePsi << endl;
  std::cout << "Gamma output being saved to " << FilenameGamma << endl;
  std::cout << "Ksi output being saved to " << FilenameKsi << endl;
  std::cout << "Kappa values being saved to " << FilenameKappa << endl;
  std::cout << "Pseudo values being saved to " << FilenamePseudo << endl;
  std::cout << "h values being saved to " << FilenameH << endl;
  std::cout << "Eta values being saved to " << FilenameEta << endl;
  std::cout << "Sigma likelihood output being saved to "
            << FilenameSigmaLikelihood << endl;
  std::cout << "t_0 likelihood output being saved to " << FilenameT0Likelihood
            << endl;
  std::cout << "Number of skeleton points for t_0 being saved to "
            << FilenameT0KsiCounter << endl;
  std::cout << "Datapoints output being saved to " << FilenameDatapointsInput
            << endl;
  std::cout << "Other program details being saved to " << FilenameOtherInput
            << endl;
  std::cout << "Sigma convergence diagnostics being saved to "
            << FilenameSigMean << " and " << FilenameSigStd << endl;
  std::cout << "T0 convergence diagnostics being saved to " << FilenameT0Mean
            << " and " << FilenameT0Std << endl;
  std::cout << "h convergence diagnostics being saved to " << FilenameHMean
            << " and " << FilenameHStd << endl;
  std::cout << "Eta convergence diagnostics being saved to " << FilenameEtaMean
            << " and " << FilenameEtaStd << endl;
}
