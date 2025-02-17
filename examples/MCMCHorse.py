import MCMC4WF_pybind as MCMC4WF
import EWF_pybind as EWF
import numpy as np

# Prepare MCMCOptions
sOptions = np.array([0.0, 10.0, -30.0, 30.0, 10.0, 0.001, 0.001]) # Prior mean, Prior Std, Sig Bound Low, Sig Bound High, Proposal Std, Mean Prec, Std Prec
hOptions = np.array([0.5, 0.1, -1.5, 1.5, 0.25, 1.e-4, 1.e-4])
etaOptions = np.array([[1], [1], [1], [1], [1], [1], [1]])
t0Options = np.array([1.5 * np.log(100.0), 0.2, 0.1, 1.e-6, 1.e-6])
theta = np.array([0.1, 0.1])
AlleleAgeMargin = 1.e-4
AlleleAgePrior = 0
burnIn, lookBack, printCounter = 10000, 10000, 1000
save, saveAux, saveLikelihood = True, True, True
selTP = 'Uniform'

MCMCo = MCMC4WF.MCMCOptions(sOptions, hOptions, etaOptions, t0Options, theta, AlleleAgeMargin, AlleleAgePrior, burnIn, lookBack, printCounter, save, saveAux, saveLikelihood, selTP)

# Prepare dataset for analysis - values as reported in Ludwig et al 2009
ASIP_observations = np.array([0, 1, 15, 12, 15, 18])
MC1R_observations = np.array([0, 0, 1, 6, 13, 24])
sample_sizes = np.array([10, 22, 20, 20, 36, 38])
times = np.array([20000., 13100., 3700., 2800., 1100., 500.]) # These times are in years BP - we need to convert to diffusion time scale!
Ne = 3000
g = 8
selectionSetup = 1 # 0 for genic inference, 1 for diploid, 2 for more general polynomial selection function

# Convert times to time in diffusion time units
times = np.cumsum(np.concatenate(([0.], np.abs(np.diff(times)) / (2 * Ne * g))))

ASIP_MCMCSampler = MCMC4WF.MCMCSampler(ASIP_observations, sample_sizes, times, selectionSetup)
#MC1R_MCMCSampler = MCMC4WF.MCMCSampler(MC1R_observations, sample_sizes, times, selectionSetup)

ASIP_MCMCSampler.RunSampler(MCMCo)
#MC1R_MCMCSampler.RunSampler(MCMCo)