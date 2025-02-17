# MCMC4WF
An efficient MCMC inference scheme for genetic data assuming a Wright-Fisher diffusion model. Our setup allows for the inference of a wide class of selective regimes (genic, diploid and arbitrary polynomial selection), whilst properly accounting for the presence/absence of mutation. The algorithm does away with any discretisation or approximations and thus offers draws from the true posterior of interest. Please consult the UserManual.pdf for all details with regards to installing dependencies, installing the program, and calling it from within python.

*Dependencies*

MCMC4WF requires the following:

- g++ compiler
- boost library (https://boost.org)
- python and pip
- CMake
- pybind11

*Installation*

To install, run the following instructions in terminal at the root directory of MCMC4WF:

```bash
mkdir build
cd build
cmake ..
cmake --build .
cd ..
pip install .
```

*Calling in python*

Once installed, you can call MCMC4WF from python by including
`import MCMC4WF_pybind`
at the start of your python script. Please consult the scripts in the `examples` directory for more details and use cases.

If you come across any bugs, or have any queries/comments/suggestions please do get in touch using the email address Jaromir.Sant@gmail.com!