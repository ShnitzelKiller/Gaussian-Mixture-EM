CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
SET(CMAKE_CXX_STANDARD 14)
PROJECT(GaussianMixtureModelEM)

FIND_PACKAGE( EIGEN3 REQUIRED NO_MODULE )
INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES( gmm )

SET( SOURCES
     gmm/GaussianMixture.h
     gmm/GaussianMixture.cpp
     test/NormalRandomVariable.h
     test/NormalRandomVariable.cpp
     test/test_gmm.cpp
)
     

SET( USE_OPENCV false CACHE BOOL "build test with opencv for rendering image results" )
IF ( USE_OPENCV )
   FIND_PACKAGE( OpenCV REQUIRED )
   ADD_DEFINITIONS( -DUSE_OPENCV )
ENDIF ( USE_OPENCV )

ADD_EXECUTABLE( GMMTests ${SOURCES} )

IF ( USE_OPENCV )
   TARGET_LINK_LIBRARIES( GMMTests ${OpenCV_LIBS} )
ENDIF ( USE_OPENCV )
