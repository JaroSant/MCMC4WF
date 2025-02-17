import MCMC4WF_pybind as MCMC4WF
import EWF_pybind as EWF
import numpy as np

# Define Wright--Fisher diffusion class parameters
theta = np.array([0.5, 0.5])
non_neutral = True
sigma = 10.0
selectionSetup = 0
dominance_parameter = 0.0
selectionPolynomialDegree = 1
selectionCoefficients = np.array([])

# Create and initialise WrightFisher class
WF = EWF.WrightFisher(theta, non_neutral, sigma, selectionSetup, dominance_parameter, selectionPolynomialDegree, selectionCoefficients)

# Define simulation parameters
nSim = 1
x = 0.1
times = np.linspace(0., 0.5, 5)
Absorption = False
Filename_sim = "EWF_diffusion_sim.txt"

# Run simulator
WF.DiffusionTrajectoryVector(1, x, times, Absorption, Filename_sim)

# Load path
path = np.concatenate(([x], np.loadtxt(Filename_sim)))
sample_sizes = 20*np.ones(np.shape(path), dtype=int)
obs = np.random.binomial(sample_sizes, path)

# Prepare MCMCOptions
sOptions = np.array([0.0, 10.0, -30.0, 30.0, 10.0, 0.001, 0.001]) # Prior mean, Prior Std, Sig Bound Low, Sig Bound High, Proposal Std, Mean Prec, Std Prec
hOptions = np.array([0.5, 0.1, -1.5, 1.5, 0.25, 1.e-4, 1.e-4])
etaOptions = np.array([[1], [1], [1], [1], [1], [1], [1]])
t0Options = np.array([4 * np.log(100.0), 0.2, 0.1, 1.e-6, 1.e-6])
AlleleAgeMargin = 1.e-4
AlleleAgePrior = 0
burnIn, lookBack, printCounter = 10000, 10000, 1000
save, saveAux, saveLikelihood = True, True, True
selTP = 'Uniform'

MCMCo = MCMC4WF.MCMCOptions(sOptions, hOptions, etaOptions, t0Options, theta, AlleleAgeMargin, AlleleAgePrior, burnIn, lookBack, printCounter, save, saveAux, saveLikelihood, selTP)

MCMCSampler = MCMC4WF.MCMCSampler(obs, sample_sizes, times, selectionSetup)

MCMCSampler.RunSampler(MCMCo)