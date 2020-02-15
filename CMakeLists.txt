CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
SET(CMAKE_CXX_STANDARD 14)
PROJECT(GaussianMixtureModelEM)

FIND_PACKAGE( EIGEN3 REQUIRED NO_MODULE )
INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES( gmm )

ADD_EXECUTABLE(GMMTests gmm/GaussianMixture.h gmm/GaussianMixture.cpp test/NormalRandomVariable.h test/NormalRandomVariable.cpp test/test_gmm.cpp)
