#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <deque>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "MCMCOptions.h"
#include "MCMCSampler.h"

namespace py = pybind11;

PYBIND11_MODULE(MCMC4WF_pybind, m) {
  py::class_<MCMCOptions>(m, "MCMCOptions")
      .def(py::init<vector<double>, vector<double>, vector<vector<double>>,
                    vector<double>, vector<double>, double, double, double, int,
                    int, int, int, bool, bool, bool, string>(),
           py::arg("sOptions"), py::arg("hOptions"), py::arg("etaOptions"),
           py::arg("t0Options"), py::arg("theta"),
           py::arg("diffusion_threshold"), py::arg("bridge_threshold"),
           py::arg("AlleleAgeMargin"), py::arg("AlleleAgePrior"),
           py::arg("burnIn"), py::arg("lookBack"), py::arg("printCounter"),
           py::arg("save"), py::arg("saveAux"), py::arg("saveLikelihood"),
           py::arg("selTP"));

  py::class_<MCMCSampler>(m, "MCMCSampler")
      .def(py::init<vector<int>, vector<int>, vector<double>, int>(),
           py::arg("Data"), py::arg("Sample"), py::arg("Times"),
           py::arg("selType"))
      .def("RunSampler", &MCMCSampler::RunSampler, py::arg("MCMCOptions"));
}
