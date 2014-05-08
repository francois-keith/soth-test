#
#  Copyright 2010 CNRS
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/cpack.cmake)
INCLUDE(cmake/eigen.cmake)

SET(PROJECT_NAME soth)
SET(PROJECT_DESCRIPTION "...")
SET(PROJECT_URL "http://github.com/laas/soth")

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE true)

project(testsoth)

ADD_REQUIRED_DEPENDENCY("soth")
ADD_EXECUTABLE(testsoth HQPtest.cpp HQPtest.h main.cpp)
PKG_CONFIG_USE_DEPENDENCY(testsoth soth)

