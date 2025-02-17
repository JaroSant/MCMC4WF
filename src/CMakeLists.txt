# Boost setup (needed for Windows to work)
set(Boost_USE_STATIC_LIBS OFF CACHE BOOL "use static libraries from Boost")
find_package(Boost REQUIRED COMPONENTS)
include_directories(${Boost_INCLUDE_DIRS})
link_libraries(${Boost_LIBRARIES})

if (WIN32)
  # disable autolinking in boost
  add_definitions( -DBOOST_ALL_NO_LIB )

  # force all boost libraries to dynamic link (we already disabled
  # autolinking, so I don't know why we need this, but we do!)
  add_definitions( -DBOOST_ALL_DYN_LINK )
endif()

set(MCMC4WF_src Polynomial.cpp PolynomialRootFinder.cpp WrightFisher.cpp MCMCOptions.cpp MCMCSampler.cpp)

set(MCMC4WF_hdr Polynomial.h PolynomialRootFinder.h WrightFisher.h MCMCOptions.h MCMCSampler.h)

add_library(MCMC4WF STATIC ${MCMC4WF_src} ${MCMC4WF_hdr})

target_include_directories(
    MCMC4WF INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:>
)

set_target_properties(MCMC4WF PROPERTIES PUBLIC_HEADER "${MCMC4WF_hdr}")

target_link_libraries(MCMC4WF PRIVATE Boost::boost)

target_include_directories(MCMC4WF PUBLIC ${Boost_INCLUDE_DIRS})

if (PYTHON_BINDINGS)
    set_target_properties(MCMC4WF PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
    pybind11_add_module(MCMC4WF_pybind MCMC4WF_pybind.cpp)
    target_link_libraries(MCMC4WF_pybind PRIVATE MCMC4WF)
endif()